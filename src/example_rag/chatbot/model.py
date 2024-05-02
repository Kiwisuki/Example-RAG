from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.example_rag.settings import MODEL_DEVICE as TORCH_DEVICE

DEFAULT_MODEL = 'meta-llama/Meta-Llama-3-8B-Instruct'

from abc import ABC, abstractmethod


class BaseChatModel(ABC):
    @abstractmethod
    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        pass


class MiniLlama(BaseChatModel):
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        max_new_tokens: int = 512,
        temperature: float = 0.25,
        top_p: float = 0.90,
    ) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=TORCH_DEVICE,
        )
        self._model.eval()

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        input_ids = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors='pt'
        ).to(self._model.device)

        terminators = [
            self._tokenizer.eos_token_id,
            self._tokenizer.convert_tokens_to_ids('<|eot_id|>'),
        ]

        outputs = self._model.generate(
            input_ids,
            max_new_tokens=self.max_new_tokens,
            eos_token_id=terminators,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        response = outputs[0][input_ids.shape[-1] :]
        return self._tokenizer.decode(response, skip_special_tokens=True)
