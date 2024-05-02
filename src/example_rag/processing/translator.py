from typing import List, Literal

from transformers import MarianMTModel, MarianTokenizer

from src.example_rag.settings import TRANSLATOR_DEVICE as TORCH_DEVICE

DIRECTIONS = {
    'lt-en': 'Helsinki-NLP/opus-mt-tc-big-lt-en',  # TODO: try out "scoris/scoris/scoris-mt-lt-en"
    'en-lt': 'Helsinki-NLP/opus-mt-tc-big-en-lt',
}


class Translator:
    def __init__(self, translation_direction: Literal['lt-en', 'en-lt']):
        self.tokenizer = MarianTokenizer.from_pretrained(
            DIRECTIONS[translation_direction]
        )
        self.model = MarianMTModel.from_pretrained(
            DIRECTIONS[translation_direction]
        ).to(TORCH_DEVICE)

    def translate_texts(self, texts: List[str]) -> List[str]:
        input_ids = self.tokenizer(texts, return_tensors='pt', padding=True).to(
            TORCH_DEVICE
        )
        translated = self.model.generate(**input_ids).to(TORCH_DEVICE)
        return [self.tokenizer.decode(t, skip_special_tokens=True) for t in translated]

    def translate_text(self, text: str) -> str:
        return self.translate_texts([text])[0]
