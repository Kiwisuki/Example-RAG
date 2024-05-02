import chromadb
from chromadb.config import Settings
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.example_rag.processing.embedder import ArcticEmbedder
from src.example_rag.settings import ChromaConfig


def get_retriever(
    collection_name: str,
    similarity_top_k: int,
    host: str = ChromaConfig.host,
    port: int = ChromaConfig.port,
) -> VectorIndexRetriever:
    embed_model = ArcticEmbedder()

    remote_db = chromadb.HttpClient(
        host=host,
        port=port,
        settings=Settings(allow_reset=True, anonymized_telemetry=False),
    )
    chroma_collection = remote_db.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store, embed_model=embed_model
    )
    return index.as_retriever(similarity_top_k=similarity_top_k)
