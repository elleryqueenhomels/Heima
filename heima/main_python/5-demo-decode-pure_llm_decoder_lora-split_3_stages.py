# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import itertools
import sys
import time
from typing import Any, Dict, List
import json
import re
import torch
from omegaconf import DictConfig, OmegaConf
import numpy as np

from torchtune import config, training, utils
from torchtune.data import load_image, Message, padded_collate_tiled_images_and_mask
from torchtune.generation import sample
from torchtune.modules.transforms import Transform


def load_from_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


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


class InferenceRecipe:
    """
    Recipe for generating tokens from a dense Transformer-based LLM.
    This works for text-only generation and image-text generation.

    This *does not* currently support the following features:
        - torch.compile
        - quantization through torchao
        - multi-GPU generation
        - batch generation
    """

    def __init__(self, cfg: DictConfig) -> None:
        self._device = utils.get_device(device=cfg.device)
        self._dtype = training.get_dtype(dtype=cfg.dtype, device=self._device)
        self._logger = utils.get_logger(cfg.log_level)
        training.set_seed(seed=cfg.seed)

    def setup(self, cfg: DictConfig) -> None:
        """Setup the model and transforms."""
        # Load checkpointer and state_dict
        _checkpointer = config.instantiate(cfg.checkpointer)
        _ckpt_dict = _checkpointer.load_checkpoint()

        # Xuan: load lora weights
        if hasattr(cfg, "lora_adapter_path") and cfg.lora_adapter_path is not None:
            _lora_dict = torch.load(cfg.lora_adapter_path, map_location=self._device)
            _ckpt_dict[training.MODEL_KEY].update(_lora_dict)

        # Instantiate model
        with training.set_default_dtype(self._dtype), self._device:
            model = config.instantiate(cfg.model)
        model.load_state_dict(_ckpt_dict[training.MODEL_KEY])
        self.model = model
        self._logger.info(f"Model was initialized with precision {self._dtype}.")

        # Instantiate transforms
        self.model_transform = config.instantiate(cfg.tokenizer)
        self.to_messages = SingleTurnYAMLToMessages()

        # Xuan: get the token index of the special tokens
        self._token_idx_thinking_of_summary = self.model_transform.tokenizer.special_tokens['<THINKING_OF_SUMMARY>']     # 128013
        self._token_idx_thinking_of_caption = self.model_transform.tokenizer.special_tokens['<THINKING_OF_CAPTION>']     # 128014
        self._token_idx_thinking_of_reasoning = self.model_transform.tokenizer.special_tokens['<THINKING_OF_REASONING>'] # 128015
    
    def build_decoder(self, _ckpt_dict_decoder, lora_adapter_path_decoder, projector_weight_path):
        # Xuan: load lora weights
        _lora_dict = torch.load(lora_adapter_path_decoder, map_location=self._device)
        _ckpt_dict_decoder[training.MODEL_KEY].update(_lora_dict)
        
        # Xuan: load projector weights
        _projector_dict = torch.load(projector_weight_path, map_location=self._device)
        _ckpt_dict_decoder[training.MODEL_KEY].update(_projector_dict)

        return _ckpt_dict_decoder
    
    def setup_decoder(self, cfg: DictConfig) -> None:
        """Setup the model and transforms."""
        # Load checkpointer and state_dict
        _checkpointer_decoder = config.instantiate(cfg.checkpointer_decoder)
        _ckpt_dict_decoder_summary = _checkpointer_decoder.load_checkpoint()
        _ckpt_dict_decoder_caption = _checkpointer_decoder.load_checkpoint()
        _ckpt_dict_decoder_reasoning = _checkpointer_decoder.load_checkpoint()

        # Xuan: add lora and projector weights to state dict
        _ckpt_dict_decoder_summary = self.build_decoder(
            _ckpt_dict_decoder_summary, cfg.lora_adapter_path_decoder_summary, cfg.projector_weight_path_summary
        )
        _ckpt_dict_decoder_caption = self.build_decoder(
            _ckpt_dict_decoder_caption, cfg.lora_adapter_path_decoder_caption, cfg.projector_weight_path_caption
        )
        _ckpt_dict_decoder_reasoning = self.build_decoder(
            _ckpt_dict_decoder_reasoning, cfg.lora_adapter_path_decoder_reasoning, cfg.projector_weight_path_reasoning
        )

        # Instantiate model
        with training.set_default_dtype(self._dtype), self._device:
            model_decoder_summary = config.instantiate(cfg.model_decoder)
        model_decoder_summary.load_state_dict(_ckpt_dict_decoder_summary[training.MODEL_KEY])
        self.model_decoder_summary = model_decoder_summary
        with training.set_default_dtype(self._dtype), self._device:
            model_decoder_caption = config.instantiate(cfg.model_decoder)
        model_decoder_caption.load_state_dict(_ckpt_dict_decoder_caption[training.MODEL_KEY])
        self.model_decoder_caption = model_decoder_caption
        with training.set_default_dtype(self._dtype), self._device:
            model_decoder_reasoning = config.instantiate(cfg.model_decoder)
        model_decoder_reasoning.load_state_dict(_ckpt_dict_decoder_reasoning[training.MODEL_KEY])
        self.model_decoder_reasoning = model_decoder_reasoning
        self._logger.info(f"Model decoders were initialized with precision {self._dtype}.")

    def log_metrics(self, total_time: int, tokens_per_second: float) -> None:
        """Logs the following metrics: total time for inference, tokens/sec,
        bandwidth achieved, and max memory allocated.

        Feel free to modify this function to log additional metrics.
        """
        model_size = sum(
            [
                p.numel() * p.dtype.itemsize
                for p in itertools.chain(self.model.parameters(), self.model.buffers())
            ]
        )
        self._logger.info(
            f"Time for inference: {total_time:.02f} sec total, {tokens_per_second:.02f} tokens/sec"
        )
        self._logger.info(
            f"Bandwidth achieved: {model_size * tokens_per_second / 1e9:.02f} GB/s"
        )
        self._logger.info(
            f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB"
        )
    
    @torch.inference_mode()
    def generate_detail(self, current_prompt, cfg: DictConfig):
        # 1. Convert input to messages
        messages = self.to_messages(OmegaConf.to_container(current_prompt))
        is_multimodal_input = any([m.contains_media for m in messages])

        # 2. Apply model transform
        model_inputs = self.model_transform({"messages": messages}, inference=True)
        seq_len = len(model_inputs["tokens"])
        total_response_length = seq_len + cfg.max_new_tokens
        
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
        last_hidden_state_records = []  # Xuan: record last hidden state
        # t0 = time.perf_counter()
        logits = self.model(prompt, **batch)[:, -1]
        token = sample(logits, temperature=cfg.temperature, top_k=cfg.top_k)
        generated_tokens.append(token.item())
        last_hidden_state_records.append(self.model.decoder.last_hidden_state)

        if is_multimodal_input:
            # Don't need image info b/c we only support 1 image and it's been
            # processed by the model now
            batch.pop("encoder_input")
            batch["encoder_mask"] = batch["encoder_mask"][:, -1:]

        # 7. Continue generating
        for i in range(cfg.max_new_tokens):

            # Update position and mask for incremental decoding
            batch["input_pos"] = input_pos[None, seq_len]
            batch["mask"] = causal_mask[None, seq_len, None, :]

            if token.item() in self.model_transform.stop_tokens:
                break

            logits = self.model(token, **batch)[:, -1]
            token = sample(logits, temperature=cfg.temperature, top_k=cfg.top_k)
            generated_tokens.append(token.item())
            seq_len += 1

            last_hidden_state_records.append(self.model.decoder.last_hidden_state)

        # t = time.perf_counter() - t0

        # 8. Translate tokens back to text
        generated_tokens_for_translate = [e for e in generated_tokens if e < 128255]
        decoded = self.model_transform.decode(generated_tokens_for_translate, skip_special_tokens=True)

        return decoded, generated_tokens, last_hidden_state_records

    def get_middle_sublist(self, input_list, start_sublist, end_sublist):
        # Find the indices of the start and end sublists
        try:
            start_index = next(i for i in range(len(input_list)) if input_list[i:i+len(start_sublist)] == start_sublist)
            end_index = next(i for i in range(start_index + len(start_sublist), len(input_list)) if input_list[i:i+len(end_sublist)] == end_sublist)
            
            # Indices for the middle sublist
            middle_start_index = start_index + len(start_sublist)
            middle_end_index = end_index
            
            return middle_start_index, middle_end_index
        except StopIteration:
            return None, None
        
    @torch.inference_mode()
    def generate_detail_decoder(self, current_prompt, cfg: DictConfig, generated_token_idx, last_hidden_state_records, stage):
        # 0. Xuan: remove image info in the prompt
        current_prompt = current_prompt.copy()  # Xuan: avoid changing the original prompt
        if "image" in current_prompt.user:  # Xuan: pure LLM decoder does not need image info
            current_prompt.user.pop("image")
        if "<image>" in current_prompt.user.text:
            current_prompt.user = current_prompt.user.text.replace("<image>", "")
        else:
            current_prompt.user = current_prompt.user.text  # Xuan: avoid assert at line 51
        if stage == 'summary':
            current_model_decoder = self.model_decoder_summary
        elif stage == 'caption':
            current_model_decoder = self.model_decoder_caption
        elif stage == 'reasoning':
            current_model_decoder = self.model_decoder_reasoning

        # 1. Convert input to messages
        messages = self.to_messages(OmegaConf.to_container(current_prompt))
        is_multimodal_input = False

        # 2. Apply model transform
        model_inputs = self.model_transform({"messages": messages}, inference=True)
        seq_len = len(model_inputs["tokens"])
        total_response_length = seq_len + cfg.max_new_tokens
        
        # Xuan: get mask_special_token according to the stage
        if stage == 'summary':
            mask_special_token = torch.from_numpy(np.array(model_inputs['tokens']) == self._token_idx_thinking_of_summary).unsqueeze(0)
        elif stage == 'caption':
            mask_special_token = torch.from_numpy(np.array(model_inputs['tokens']) == self._token_idx_thinking_of_caption).unsqueeze(0)
        elif stage == 'reasoning':
            mask_special_token = torch.from_numpy(np.array(model_inputs['tokens']) == self._token_idx_thinking_of_reasoning).unsqueeze(0)
        
        # Xuan: find special token in the generated_token_idx
            # [19389, 2864, 49970, 29, 220] -> '<SUMMARY> '
            # [694, 28477, 49970, 1363] -> '</SUMMARY>\n\n'
            # [20996, 2599, 60459, 29, 220] -> '<CAPTION> '
            # [694, 32500, 60459, 1363] -> '</CAPTION>\n\n'
            # [27, 793, 36404, 1753, 29, 220] -> '<REASONING> '
            # [694, 793, 36404, 1753, 1363] -> '</REASONING>\n\n'
        # Therefore, in the generated_token_idx, we can find the corresponding token:
            # [19389, 2864, 49970, 29, 220, 
            # 128xxx, 
            # 694, 28477, 49970, 1363, 
            # 20996, 2599, 60459, 29, 220, 
            # 128xxx, 
            # 694, 32500, 60459, 1363, 
            # 27, 793, 36404, 1753, 29, 220, 
            # 128xxx, 
            # 694, 793, 36404, 1753, 1363, 
            # ...]
        if stage == 'summary':
            thinking_token_idx_start, thinking_token_idx_end = self.get_middle_sublist(generated_token_idx, [19389, 2864, 49970, 29, 220], [694, 28477, 49970, 1363])
            if thinking_token_idx_start is None:
                return None
            thinking_token_main_pipeline = torch.cat(last_hidden_state_records[thinking_token_idx_start : thinking_token_idx_end], dim=1)   # shape: [B, N, D]
            assert thinking_token_main_pipeline.shape[1] == thinking_token_idx_end - thinking_token_idx_start, "The shape of thinking_token_main_pipeline is not correct"
        elif stage == 'caption':
            thinking_token_idx_start, thinking_token_idx_end = self.get_middle_sublist(generated_token_idx, [20996, 2599, 60459, 29, 220], [694, 32500, 60459, 1363])
            if thinking_token_idx_start is None:
                return None
            thinking_token_main_pipeline = torch.cat(last_hidden_state_records[thinking_token_idx_start : thinking_token_idx_end], dim=1)   # shape: [B, N, D]
            assert thinking_token_main_pipeline.shape[1] == thinking_token_idx_end - thinking_token_idx_start, "The shape of thinking_token_main_pipeline is not correct"
        elif stage == 'reasoning':
            thinking_token_idx_start, thinking_token_idx_end = self.get_middle_sublist(generated_token_idx, [27, 793, 36404, 1753, 29, 220], [694, 793, 36404, 1753, 1363])
            if thinking_token_idx_start is None:
                return None
            thinking_token_main_pipeline = torch.cat(last_hidden_state_records[thinking_token_idx_start : thinking_token_idx_end], dim=1)   # shape: [B, N, D]
            assert thinking_token_main_pipeline.shape[1] == thinking_token_idx_end - thinking_token_idx_start, "The shape of thinking_token_main_pipeline is not correct"

        # 3. Setup KV cache
        with self._device:
            current_model_decoder.setup_caches(
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
        logits = current_model_decoder(prompt, **batch, 
                                    thinking_token=thinking_token_main_pipeline,
                                    thinking_token_mask=mask_special_token.to(self._device),
                                    )[:, -1]
        token = sample(logits, temperature=cfg.temperature, top_k=cfg.top_k)
        generated_tokens.append(token.item())

        if is_multimodal_input:
            # Don't need image info b/c we only support 1 image and it's been
            # processed by the model now
            batch.pop("encoder_input")
            batch["encoder_mask"] = batch["encoder_mask"][:, -1:]

        # 7. Continue generating
        for i in range(cfg.max_new_tokens):

            # Update position and mask for incremental decoding
            batch["input_pos"] = input_pos[None, seq_len]
            batch["mask"] = causal_mask[None, seq_len, None, :]

            if token.item() in self.model_transform.stop_tokens:
                break

            logits = current_model_decoder(token, **batch)[:, -1]
            token = sample(logits, temperature=cfg.temperature, top_k=cfg.top_k)
            generated_tokens.append(token.item())
            seq_len += 1

        # t = time.perf_counter() - t0

        # 8. Translate tokens back to text
        decoded = self.model_transform.decode(generated_tokens, skip_special_tokens=True)

        return decoded


    @torch.inference_mode()
    def generate(self, cfg: DictConfig):
        """The main entry point for generating tokens from a prompt."""

        decoded_text, generated_token_idx, last_hidden_state_records = self.generate_detail(cfg.prompt, cfg)
        if hasattr(cfg, "prompt_summary") and cfg.prompt_summary is not None:
            cfg.prompt_summary.user.text = cfg.prompt_summary.user.text.replace("<THINKING_OF_SUMMARY>", "<THINKING_OF_SUMMARY>"*cfg.num_thinking_token)
            decoded_text_summary = self.generate_detail_decoder(cfg.prompt_summary, cfg, generated_token_idx, last_hidden_state_records, 'summary')
        if hasattr(cfg, "prompt_caption") and cfg.prompt_caption is not None:
            cfg.prompt_caption.user.text = cfg.prompt_caption.user.text.replace("<THINKING_OF_CAPTION>", "<THINKING_OF_CAPTION>"*cfg.num_thinking_token)
            decoded_text_caption = self.generate_detail_decoder(cfg.prompt_caption, cfg, generated_token_idx, last_hidden_state_records, 'caption')
        if hasattr(cfg, "prompt_reasoning") and cfg.prompt_reasoning is not None:
            cfg.prompt_reasoning.user.text = cfg.prompt_reasoning.user.text.replace("<THINKING_OF_REASONING>", "<THINKING_OF_REASONING>"*cfg.num_thinking_token)
            decoded_text_reasoning = self.generate_detail_decoder(cfg.prompt_reasoning, cfg, generated_token_idx, last_hidden_state_records, 'reasoning')

        print("############################################")
        print(f"Question: \n{cfg.prompt.user.text}")
        if hasattr(cfg.prompt.user, "image") and cfg.prompt.user.image is not None:
            print(f"Image: \n{cfg.prompt.user.image}")
        print("############################################")
        print(f"Generated: \n{decoded_text}")
        if hasattr(cfg, "prompt_summary") and cfg.prompt_summary is not None:
            print("############################################")
            print(f"Generated Summary: \n{decoded_text_summary}")
        if hasattr(cfg, "prompt_caption") and cfg.prompt_caption is not None:
            print("############################################")
            print(f"Generated Caption: \n{decoded_text_caption}")
        if hasattr(cfg, "prompt_reasoning") and cfg.prompt_reasoning is not None:
            print("############################################")
            print(f"Generated Reasoning: \n{decoded_text_reasoning}")
        print("############################################")
        # exit()


@config.parse
def main(cfg: DictConfig) -> None:
    config.log_config(recipe_name="InferenceRecipe", cfg=cfg)
    recipe = InferenceRecipe(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.setup_decoder(cfg=cfg)
    recipe.generate(cfg=cfg)


if __name__ == "__main__":
    sys.exit(main())