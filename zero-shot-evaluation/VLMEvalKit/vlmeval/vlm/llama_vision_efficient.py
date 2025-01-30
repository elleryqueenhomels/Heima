import torch
from PIL import Image
import os.path as osp
import sys
from .base import BaseModel
from ..smp import *
from ..dataset import DATASET_TYPE

# added by Yizhou
from typing import Any, Dict, List
from torchtune import config, training, utils
from torchtune.data import load_image, Message, padded_collate_tiled_images_and_mask
from omegaconf import DictConfig, OmegaConf
from torchtune.generation import sample

from torchtune.modules.transforms import Transform

import re


class SingleTurnYAMLToMessages(Transform):
    """
    Converts a single turn conversation in YAML format to a list of messages.

    Expects the YAML to look like:
        system: You are a helpful AI assistant.
        user: What is the capital of France?

    or if it includes an image:
        system: You are a helpful AI assistant.
        user:
            image: url or path_to_image
            text: Describe the image in detail.
    """

    def __call__(self, prompt: Dict[str, Any]) -> List[Message]:
        messages = []

        # Iterate through roles and add content
        for role, content in prompt.items():
            if isinstance(content, str):
                new_content = [{"type": "text", "content": content}]
            else:
                assert (
                    "image" in content.keys()
                ), "Multiple entries per role expect an image key"
                image_loc = content["image"]
                image = load_image(image_loc)
                new_content = [
                    {"type": "image", "content": image},
                    {"type": "text", "content": content["text"]},
                ]
            messages.append(Message(role=role, content=new_content))

        # Finally, add an empty assistant message to kick-start generation
        messages.append(Message(role="assistant", content=""))
        return messages


class llama_vision_efficient(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = False

    # This function is used to split Llama-3.2-90B
    def split_model(self):
        import math
        device_map = {}
        num_gpus = torch.cuda.device_count()
        rank, world_size = get_rank_and_world_size()
        num_gpus = num_gpus // world_size

        num_layers = 100
        # GPU0: -5, GPU-1: -7
        total_cost = num_layers + 5 + 7

        # Since the first GPU will be used for ViT, treat it as 0.8 GPU.
        num_layers_per_gpu = total_cost // num_gpus
        num_layers_per_gpu = [num_layers_per_gpu] * num_gpus
        # The total number of GPUs might be odd
        num_layers_per_gpu[-1] = total_cost - sum(num_layers_per_gpu[:-1])
        num_layers_per_gpu[0] -= 5
        num_layers_per_gpu[-1] -= 7

        layer_cnt = 0
        for i, num_layer in enumerate(num_layers_per_gpu):
            for j in range(num_layer):
                device_map[f'language_model.model.layers.{layer_cnt}'] = rank + world_size * i
                layer_cnt += 1

        device_map['vision_model'] = rank
        device_map['language_model.model.embed_tokens'] = rank
        device_map['language_model.model.rotary_emb'] = rank
        device_map['language_model.model.norm'] = rank + world_size * (num_gpus - 1)
        device_map['language_model.lm_head'] = rank + world_size * (num_gpus - 1)
        device_map['multi_modal_projector'] = rank + world_size * (num_gpus - 1)
        return device_map

    def __init__(self, model_path='meta-llama/Llama-3.2-11B-Vision-Instruct', cfg = '/home/xuans/sensei-fs-link/code/efficient-reasoning/efficient-reasoning/zero-shot-evaluation/VLMEvalKit/main_python/3-llama3_2_vision-11b-generation.yaml', **kwargs):
        try:
            from transformers import MllamaForConditionalGeneration, AutoProcessor
        except Exception as e:
            logging.critical('Please install transformers>=4.45.0 before using llama_vision.')
            raise e

        self.cfg = OmegaConf.load(cfg)

        self._device = utils.get_device(device=self.cfg.device)
        self._dtype = training.get_dtype(dtype=self.cfg.dtype, device=self._device)

        # Load checkpointer and state_dict
        _checkpointer = config.instantiate(self.cfg.checkpointer)
        _ckpt_dict = _checkpointer.load_checkpoint()

        # # Xuan: load lora weights
        # if hasattr(self.cfg, "lora_adapter_path") and self.cfg.lora_adapter_path is not None:
        #     _lora_dict = torch.load(self.cfg.lora_adapter_path, map_location=self._device)
        #     _ckpt_dict[training.MODEL_KEY].update(_lora_dict)

        # Instantiate model
        with training.set_default_dtype(self._dtype), self._device:
            model = config.instantiate(self.cfg.model)
        model.load_state_dict(_ckpt_dict[training.MODEL_KEY])
        self.model = model.cuda().eval()

        # Instantiate transforms
        self.model_transform = config.instantiate(self.cfg.tokenizer)
        self.to_messages = SingleTurnYAMLToMessages()
        

    def generate_inner(self, message, dataset=None):
        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)

        try:
            current_prompt = OmegaConf.create({
                    # "system": "You are a helpful AI assistant.",
                    "user": {
                        "image": image_path,
                        "text": prompt
                    }
                })
        except Exception as e:
            print('Error in generating prompt:', e)
            return ""

        # step 1 -> step 8
        # 1. Convert input to messages
        messages = self.to_messages(OmegaConf.to_container(current_prompt))
        is_multimodal_input = any([m.contains_media for m in messages])

        # 2. Apply model transform
        model_inputs = self.model_transform({"messages": messages}, inference=True)
        seq_len = len(model_inputs["tokens"])
        total_response_length = seq_len + self.cfg.max_new_tokens

        # 3. Setup KV cache
        with self._device:
            self.model.setup_caches(
                batch_size=1,
                dtype=self._dtype,
                encoder_max_seq_len=(
                    self.model_transform.image_seq_len if is_multimodal_input else None
                ),
                decoder_max_seq_len=total_response_length,
            )

        # 4. Pre-allocate causal mask and input_pos
        causal_mask = torch.tril(
            torch.ones(
                size=(total_response_length, total_response_length),
                dtype=torch.bool,
                device=self._device,
            )
        )
        input_pos = torch.arange(total_response_length)

        # 5. Collate to batch size of 1 and tensor-ify
        batch = {}
        if is_multimodal_input:
            batch = padded_collate_tiled_images_and_mask(
                [model_inputs],
                pad_direction="left",
                pad_max_images=1,
                pad_max_tiles=self.model_transform.max_num_tiles,
            )
            batch["encoder_mask"] = batch["encoder_mask"][:, :seq_len]
            prompt = batch.pop("tokens").to(self._device)
        else:
            prompt = torch.tensor(
                model_inputs["tokens"], device=self._device
            ).unsqueeze(0)
        batch["mask"] = causal_mask[None, :seq_len]
        batch["input_pos"] = input_pos[None, :seq_len]
        utils.batch_to_device(batch, self._device)

        # 6. Prefill step
        generated_tokens = []
        # t0 = time.perf_counter()
        logits = self.model(prompt, **batch)[:, -1]
        token = sample(logits, temperature=self.cfg.temperature, top_k=self.cfg.top_k)
        generated_tokens.append(token.item())

        if is_multimodal_input:
            # Don't need image info b/c we only support 1 image and it's been
            # processed by the model now
            batch.pop("encoder_input")
            batch["encoder_mask"] = batch["encoder_mask"][:, -1:]

        # 7. Continue generating
        for i in range(self.cfg.max_new_tokens):

            # Update position and mask for incremental decoding
            batch["input_pos"] = input_pos[None, seq_len]
            batch["mask"] = causal_mask[None, seq_len, None, :]

            if token.item() in self.model_transform.stop_tokens:
                break

            logits = self.model(token, **batch)[:, -1]
            token = sample(logits, temperature=self.cfg.temperature, top_k=self.cfg.top_k)
            generated_tokens.append(token.item())
            seq_len += 1

        # t = time.perf_counter() - t0

        # 8. Translate tokens back to text
        decoded = self.model_transform.decode(generated_tokens, skip_special_tokens=True)

        # Extract the conclusion part while preserving the tags
        if match := re.search(r'(<CONCLUSION>.*?</CONCLUSION>)', decoded, re.DOTALL):
            decoded = match.group(1)

        # print(decoded)
        return decoded, len(generated_tokens)   # Xuan: also return the number of the generated tokens

