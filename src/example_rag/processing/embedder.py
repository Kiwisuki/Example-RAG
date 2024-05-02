from typing import Any, List, Literal

import torch
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.embeddings import BaseEmbedding
from transformers import AutoModel, AutoTokenizer

from src.example_rag.settings import EMBEDDER_DEVICE as TORCH_DEVICE

EMBEDDING_MODELS = {
    'long': 'Snowflake/snowflake-arctic-embed-m-long',
    'large': 'Snowflake/snowflake-arctic-embed-l',
}

QUERY_PREFIX = 'Represent this sentence for searching relevant passages: '
TOKEN_LIMIT = 512  # Could be set to 2048 for 'long' model


class ArcticEmbedder(BaseEmbedding):
    _model: AutoModel = PrivateAttr()
    _tokenizer: AutoTokenizer = PrivateAttr()
    _instruction: str = PrivateAttr()

    def __init__(
        self,
        model_type: Literal['long', 'large'] = 'large',
        instruction: str = QUERY_PREFIX,
        **kwargs: Any,
    ) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODELS[model_type])
        self._model = AutoModel.from_pretrained(
            EMBEDDING_MODELS[model_type],
            trust_remote_code=True,
            add_pooling_layer=False,
        ).to(TORCH_DEVICE)
        self._instruction = instruction
        super().__init__(**kwargs)

    def _embed_batch(self, texts: List[str]) -> List[float]:
        query_tokens = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=TOKEN_LIMIT,
        ).to(TORCH_DEVICE)

        with torch.no_grad():
            query_embeddings = self._model(**query_tokens)[0][:, 0]

        query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
        return query_embeddings

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._embed_batch([text])[0].tolist()

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._embed_batch(texts).tolist()

    def _get_query_embedding(self, query: str) -> List[float]:
        query = '{}{}'.format(self._instruction, query)
        return self._embed_batch([query])[0].tolist()

    def _get_query_embeddings(self, queries: List[str]) -> List[List[float]]:
        queries = ['{}{}'.format(self._instruction, i) for i in queries]
        return self._embed_batch(queries).tolist()

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self.get_text_embedding(text)
