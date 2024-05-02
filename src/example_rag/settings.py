import logging
from dataclasses import dataclass

import torch

LOGGER = logging.getLogger(__name__)

IS_CUDA_AVAILABLE = torch.cuda.is_available()

MODEL_DEVICE = 'cuda' if IS_CUDA_AVAILABLE else 'cpu'
TRANSLATOR_DEVICE = 'cuda' if IS_CUDA_AVAILABLE else 'cpu'
EMBEDDER_DEVICE = 'cpu'

COLLECTION_NAME = 'LiudasVasaris'

SYSTEM_PROMPT = (
    "You are a chatbot, able to have normal interactions, as well as talk"
    " about a book 'Shadow of Altars' by Vincas Mykolaitis-Putinas,\n"
    "Where the main character is Liudas Vasaris."
    "\nInstruction: Use the previous chat history, or the context above, to interact and help the user."
)

CONTEXT_PROMPT = (
    'We found some context that might be helpful for you:\n'
    '---------------------\n'
    '{context_str}\n'
    '---------------------\n'
    '\n\n'
    'Query: {query_str}\n'
)


@dataclass
class GenerationSettings:
    max_new_tokens: int = 512
    temperature: float = 0.50
    top_p: float = 0.50
    similiarity_top_k: int = 10


@dataclass
class ChromaConfig:
    host: str = 'chroma'
    port: int = 8000


@dataclass
class ProcessingConfig:
    chunk_size: int = 700
    chunk_overlap: int = 32
