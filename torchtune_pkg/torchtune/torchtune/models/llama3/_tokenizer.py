# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
from typing import Any, Dict, List, Mapping, Optional, Tuple

from torchtune.data import Message, PromptTemplate, truncate
from torchtune.modules.tokenizers import ModelTokenizer, TikTokenBaseTokenizer
from torchtune.modules.transforms import Transform


CL100K_PATTERN = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""  # noqa

SPECIAL_TOKENS = {
    "<|begin_of_text|>": 128000,
    "<|end_of_text|>": 128001,
    "<|reserved_special_token_0|>": 128002,
    "<|reserved_special_token_1|>": 128003,
    "<|finetune_right_pad_id|>": 128004,
    "<|step_id|>": 128005,
    "<|start_header_id|>": 128006,
    "<|end_header_id|>": 128007,
    "<|eom_id|>": 128008,
    "<|eot_id|>": 128009,
    "<|python_tag|>": 128010,
    "<|image|>": 128256,
    "<|video|>": 128012,
    "<|reserved_special_token_2|>": 128011,

    "<THINKING_OF_SUMMARY>": 128013,
    "<THINKING_OF_CAPTION>": 128014,
    "<THINKING_OF_REASONING>": 128015,

    # Xuan: if you want to try larger number of thinking tokens, use the following:
    #   (do not exceed the 128256 as it will raise cuda error during training)

    # "<THINKING_OF_SUMMARY_1>": 128013,
    # "<THINKING_OF_SUMMARY_2>": 128014,
    # "<THINKING_OF_SUMMARY_3>": 128015,
    # "<THINKING_OF_SUMMARY_4>": 128016,
    # "<THINKING_OF_SUMMARY_5>": 128017,
    # "<THINKING_OF_SUMMARY_6>": 128018,
    # "<THINKING_OF_SUMMARY_7>": 128019,
    # "<THINKING_OF_SUMMARY_8>": 128020,
    # "<THINKING_OF_SUMMARY_9>": 128021,
    # "<THINKING_OF_SUMMARY_10>": 128022,
    # "<THINKING_OF_SUMMARY_11>": 128023,
    # "<THINKING_OF_SUMMARY_12>": 128024,
    # "<THINKING_OF_SUMMARY_13>": 128025,
    # "<THINKING_OF_SUMMARY_14>": 128026,
    # "<THINKING_OF_SUMMARY_15>": 128027,
    # "<THINKING_OF_SUMMARY_16>": 128028,
    # "<THINKING_OF_SUMMARY_17>": 128029,
    # "<THINKING_OF_SUMMARY_18>": 128030,
    # "<THINKING_OF_SUMMARY_19>": 128031,
    # "<THINKING_OF_SUMMARY_20>": 128032,
    # "<THINKING_OF_SUMMARY_21>": 128033,
    # "<THINKING_OF_SUMMARY_22>": 128034,
    # "<THINKING_OF_SUMMARY_23>": 128035,
    # "<THINKING_OF_SUMMARY_24>": 128036,
    # "<THINKING_OF_SUMMARY_25>": 128037,
    # "<THINKING_OF_SUMMARY_26>": 128038,
    # "<THINKING_OF_SUMMARY_27>": 128039,
    # "<THINKING_OF_SUMMARY_28>": 128040,
    # "<THINKING_OF_SUMMARY_29>": 128041,
    # "<THINKING_OF_SUMMARY_30>": 128042,
    # "<THINKING_OF_SUMMARY_31>": 128043,
    # "<THINKING_OF_SUMMARY_32>": 128044,
    #
    # "<THINKING_OF_CAPTION_1>": 128045,
    # "<THINKING_OF_CAPTION_2>": 128046,
    # "<THINKING_OF_CAPTION_3>": 128047,
    # "<THINKING_OF_CAPTION_4>": 128048,
    # "<THINKING_OF_CAPTION_5>": 128049,
    # "<THINKING_OF_CAPTION_6>": 128050,
    # "<THINKING_OF_CAPTION_7>": 128051,
    # "<THINKING_OF_CAPTION_8>": 128052,
    # "<THINKING_OF_CAPTION_9>": 128053,
    # "<THINKING_OF_CAPTION_10>": 128054,
    # "<THINKING_OF_CAPTION_11>": 128055,
    # "<THINKING_OF_CAPTION_12>": 128056,
    # "<THINKING_OF_CAPTION_13>": 128057,
    # "<THINKING_OF_CAPTION_14>": 128058,
    # "<THINKING_OF_CAPTION_15>": 128059,
    # "<THINKING_OF_CAPTION_16>": 128060,
    # "<THINKING_OF_CAPTION_17>": 128061,
    # "<THINKING_OF_CAPTION_18>": 128062,
    # "<THINKING_OF_CAPTION_19>": 128063,
    # "<THINKING_OF_CAPTION_20>": 128064,
    # "<THINKING_OF_CAPTION_21>": 128065,
    # "<THINKING_OF_CAPTION_22>": 128066,
    # "<THINKING_OF_CAPTION_23>": 128067,
    # "<THINKING_OF_CAPTION_24>": 128068,
    # "<THINKING_OF_CAPTION_25>": 128069,
    # "<THINKING_OF_CAPTION_26>": 128070,
    # "<THINKING_OF_CAPTION_27>": 128071,
    # "<THINKING_OF_CAPTION_28>": 128072,
    # "<THINKING_OF_CAPTION_29>": 128073,
    # "<THINKING_OF_CAPTION_30>": 128074,
    # "<THINKING_OF_CAPTION_31>": 128075,
    # "<THINKING_OF_CAPTION_32>": 128076,
    #
    # "<THINKING_OF_REASONING_1>": 128077,
    # "<THINKING_OF_REASONING_2>": 128078,
    # "<THINKING_OF_REASONING_3>": 128079,
    # "<THINKING_OF_REASONING_4>": 128080,
    # "<THINKING_OF_REASONING_5>": 128081,
    # "<THINKING_OF_REASONING_6>": 128082,
    # "<THINKING_OF_REASONING_7>": 128083,
    # "<THINKING_OF_REASONING_8>": 128084,
    # "<THINKING_OF_REASONING_9>": 128085,
    # "<THINKING_OF_REASONING_10>": 128086,
    # "<THINKING_OF_REASONING_11>": 128087,
    # "<THINKING_OF_REASONING_12>": 128088,
    # "<THINKING_OF_REASONING_13>": 128089,
    # "<THINKING_OF_REASONING_14>": 128090,
    # "<THINKING_OF_REASONING_15>": 128091,
    # "<THINKING_OF_REASONING_16>": 128092,
    # "<THINKING_OF_REASONING_17>": 128093,
    # "<THINKING_OF_REASONING_18>": 128094,
    # "<THINKING_OF_REASONING_19>": 128095,
    # "<THINKING_OF_REASONING_20>": 128096,
    # "<THINKING_OF_REASONING_21>": 128097,
    # "<THINKING_OF_REASONING_22>": 128098,
    # "<THINKING_OF_REASONING_23>": 128099,
    # "<THINKING_OF_REASONING_24>": 128100,
    # "<THINKING_OF_REASONING_25>": 128101,
    # "<THINKING_OF_REASONING_26>": 128102,
    # "<THINKING_OF_REASONING_27>": 128103,
    # "<THINKING_OF_REASONING_28>": 128104,
    # "<THINKING_OF_REASONING_29>": 128105,
    # "<THINKING_OF_REASONING_30>": 128106,
    # "<THINKING_OF_REASONING_31>": 128107,
    # "<THINKING_OF_REASONING_32>": 128108,

}


# NUM_RESERVED_SPECIAL_TOKENS = 256
# RESERVED_TOKENS = {
#     f"<|reserved_special_token_{3 + i}|>": 128016 + i
#     for i in range(NUM_RESERVED_SPECIAL_TOKENS - len(SPECIAL_TOKENS))
# }
RESERVED_TOKENS = {}

LLAMA3_SPECIAL_TOKENS = {**SPECIAL_TOKENS, **RESERVED_TOKENS}


class Llama3Tokenizer(ModelTokenizer, Transform):
    """
    tiktoken tokenizer configured with Llama3 Instruct's special tokens, as described in
    https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3

    Args:
        path (str): Path to pretrained tiktoken tokenizer file.
        special_tokens (Optional[Dict[str, int]]): mapping containing special text tokens and
            their registered token IDs. If left as None, this will be set to the canonical
            Llama3 special tokens.
        max_seq_len (Optional[int]): maximum sequence length for tokenizing a single list of messages,
            after which the input will be truncated. Default is None.
        prompt_template (Optional[PromptTemplate]): template used to format the messages based on their role. This is used
            to add structured text around the actual messages. The structured text is used in three scenarios:

            - Task-specific templates to gear models for a particular task that it will expect after training
            - Model-specific templates that are required whenever the model is prompted, such as the [INST]
              tags in Llama2 and in Mistral
            - Community standardized templates, such as :class:`~torchtune.data.ChatMLTemplate`

            The extra text will still get tokenized as normal text, not as special tokens. Default is None.

    Examples:
        >>> tokenizer = Llama3Tokenizer("/path/to/tt_model")
        >>> tokenized_text = tokenizer.encode("Hello world!", add_bos=True, add_eos=True)
        >>> print(tokenized_text)
        [1, 31587, 29644, 102, 2]
    """

    def __init__(
        self,
        path: str,
        special_tokens: Optional[Dict[str, int]] = None,
        max_seq_len: Optional[int] = None,
        prompt_template: Optional[PromptTemplate] = None,
    ):
        self.special_tokens = (
            special_tokens if special_tokens is not None else LLAMA3_SPECIAL_TOKENS
        )

        self._validate_special_tokens()

        # Encode BOS and EOS, define pad ID
        self.bos_id = self.special_tokens["<|begin_of_text|>"]
        self.eos_id = self.special_tokens["<|end_of_text|>"]
        self.pad_id = self.special_tokens["<|finetune_right_pad_id|>"]
        self.step_id = self.special_tokens["<|step_id|>"]

        # Encode extra special tokens
        self.start_header_id = self.special_tokens["<|start_header_id|>"]
        self.end_header_id = self.special_tokens["<|end_header_id|>"]
        self.eot_id = self.special_tokens["<|eot_id|>"]

        self.eom_id = self.special_tokens["<|eom_id|>"]
        self.python_tag = self.special_tokens["<|python_tag|>"]

        # Media tokens
        self.image_id = self.special_tokens["<|image|>"]

        # During generation, stop when either eos_id, eot_id, or eom_id is encountered
        self.stop_tokens = [self.eos_id, self.eot_id, self.eom_id]

        self.tt_model = TikTokenBaseTokenizer(
            path=path,
            name="llama3_tiktoken",
            pattern=CL100K_PATTERN,
            bos_id=self.bos_id,
            eos_id=self.eos_id,
            special_tokens=self.special_tokens,
        )
        self.max_seq_len = max_seq_len

        self.prompt_template = prompt_template

        # Regex for removing special tokens from the decoded string
        self._special_token_regex = re.compile(r"<\|.*?\|>")
        self._special_token_header_regex = re.compile(
            r"<\|start_header_id\|>.*?<\|end_header_id\|>\n\n"
        )

    def _validate_special_tokens(
        self,
    ):
        """
        Validate that required special tokens are passed into the tokenizer.
        """
        for token in [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|eom_id|>",
            "<|eot_id|>",
            "<|python_tag|>",
        ]:
            if token not in self.special_tokens:
                raise ValueError(f"{token} missing from special_tokens")

    def _remove_special_tokens(self, text: str) -> str:
        """
        Remove special tokens from the decoded string.
        """
        # First remove the headers, then the remaining special tokens
        return self._special_token_regex.sub(
            "", self._special_token_header_regex.sub("", text)
        )

    @property
    def base_vocab_size(self) -> int:
        return self.tt_model.base_vocab_size

    @property
    def vocab_size(self) -> int:
        return self.tt_model.vocab_size

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> List[int]:
        return self.tt_model.encode(text=text, add_bos=add_bos, add_eos=add_eos)

    def decode(
        self,
        token_ids: List[int],
        truncate_at_eos: bool = True,
        skip_special_tokens: bool = False,  # Xuan: we want to show the special tokens
    ) -> str:
        """
        Decode a list of token ids into a string.

        Args:
            token_ids (List[int]): The list of token ids.
            truncate_at_eos (bool): Whether to truncate the string at the end of
                sequence token. Default is True.
            skip_special_tokens (bool): Whether to show or skip special tokens in the decoded string.
                Default is True.

        Returns:
            str: The decoded string.
        """
        # We will remove special tokens manually via regex on the decoded string.
        # This is because removing all special tokens does not remove the role and
        # whitespace added from the special tokens, i.e., the "user" and "\n\n" in
        # "<|start_header_id|>user<|end_header_id|>\n\n"
        decoded_string = self.tt_model.decode(
            token_ids=token_ids,
            truncate_at_eos=truncate_at_eos,
        )
        return (
            self._remove_special_tokens(decoded_string)
            if skip_special_tokens
            else decoded_string
        )

    def _tokenize_header(self, message: Message) -> List[int]:
        """
        Tokenize header start, message role, and header end as list of ids
        """
        return (
            [self.start_header_id]
            + self.encode(message.role.strip(), add_bos=False, add_eos=False)
            + [self.end_header_id]
            + self.encode("\n\n", add_bos=False, add_eos=False)
        )

    def _tokenize_end(self, message: Message) -> List[int]:
        """
        Add eot or eom id at the end of the message.
        """
        return [self.eot_id] if message.eot else [self.eom_id]

    def _tokenize_body(self, message: Message) -> List[int]:
        """
        Tokenize message content as list of ids
        """
        tokenized_body = []
        for item in message.content:
            if item["type"] == "text":
                tokenized_body += self.encode(
                    item["content"].strip(), add_bos=False, add_eos=False
                )
            elif item["type"] == "image":
                tokenized_body += [self.image_id]
            else:
                raise RuntimeError(f"Unsupported message content type: {item['type']}")

        if message.ipython:
            tokenized_body = [self.python_tag] + tokenized_body

        return tokenized_body

    def tokenize_message(
        self,
        message: Message,
        *,
        add_start_tokens: bool = True,
        add_end_tokens: bool = True,
    ) -> List[int]:
        """
        Tokenize a message into a list of token ids.

        Args:
            message (Message): The message to tokenize.
            add_start_tokens (bool): Whether to prepend a tokenized header to the message. Default is True.
            add_end_tokens (bool): Whether to append eot or eom id at the end of the message. Default is True.

        Returns:
            List[int]: The list of token ids.
        """
        tokenized_header = self._tokenize_header(message) if add_start_tokens else []
        tokenized_body = self._tokenize_body(message)
        tokenized_end = self._tokenize_end(message) if add_end_tokens else []

        tokenized_message = tokenized_header + tokenized_body + tokenized_end
        return tokenized_message

    def tokenize_messages(
        self,
        messages: List[Message],
        *,
        add_end_tokens: bool = True,
    ) -> Tuple[List[int], List[bool]]:
        """
        Tokenize a list of messages into a list of token ids and masks.

        Args:
            messages (List[Message]): The list of messages to tokenize.
            add_end_tokens (bool): Whether to append end tokens ids (end-of-seq, end-of-turn, end-of-message) at the end of the
                last assistant message. This value should be set to False for generation. Default is True.

        Examples:
            >>> # Tokenize a list of messages with default settings
            >>> messages = [
            ...     Message(role="user", content="Hello world!", masked=True),
            ...     Message(role="assistant", content="How are you?", masked=False),
            ... ]
            >>> tokenizer = Llama3Tokenizer("/path/to/tt_model")
            >>> tokenizer.tokenize_messages(messages)
            ([1, 31587, 29644, 102, 1, 31587, 29644, 102, 2], [True, True, True, True, True, False, False, False, True])

            >>> # Tokenize a list of messages with add_end_tokens set to False
            >>> tokenizer.tokenize_messages(messages, add_end_tokens=False)
            ([1, 31587, 29644, 102, 1, 31587, 29644], [True, True, True, True, True, False, False])

        Returns:
            Tuple[List[int], List[bool]]: The list of token ids and the list of masks.
        """
        templated_messages = (
            self.prompt_template(messages)
            if self.prompt_template is not None
            else messages
        )
        tokens = [self.bos_id]
        # bos and eos are always masked
        mask = [True]

        num_messages = len(templated_messages)
        for i, message in enumerate(templated_messages):
            # Add end tokens to the last assistant message if add_end_tokens is True
            # Otherwise, end tokens should always be added
            add_end_tokens_to_message = (
                add_end_tokens if i == num_messages - 1 else True
            )
            tokenized_message = self.tokenize_message(
                message, add_end_tokens=add_end_tokens_to_message
            )

            tokens = tokens + tokenized_message
            mask = mask + ([message.masked] * len(tokenized_message))
            if self.max_seq_len and len(tokens) >= self.max_seq_len:
                break

        if add_end_tokens:
            tokens = tokens + [self.eos_id]
            mask = mask + [True]

        if self.max_seq_len:
            tokens = truncate(
                tokens, self.max_seq_len, self.eos_id if add_end_tokens else None
            )
            mask = truncate(mask, self.max_seq_len, True if add_end_tokens else None)

        return tokens, mask

    def __call__(
        self, sample: Mapping[str, Any], inference: bool = False
    ) -> Mapping[str, Any]:
        """
        Apply ``tokenize_messages`` to the "messages" field in the sample.

        Args:
            sample (Mapping[str, Any]): A sample with a "messages" field containing
                a List[Message] to tokenize
            inference (bool): Whether the template is being used for inference or not.

        Returns:
            Mapping[str, Any]: The sample with added "tokens" and "mask" fields
                and the "messages" field removed.
        """
        messages = sample.pop("messages")
        tokens, mask = self.tokenize_messages(messages, add_end_tokens=not inference)
        sample["tokens"] = tokens
        sample["mask"] = mask
        return sample
