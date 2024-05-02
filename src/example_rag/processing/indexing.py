import logging

import chromadb
from chromadb.config import Settings
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from tqdm import tqdm

from src.example_rag.processing.embedder import ArcticEmbedder
from src.example_rag.processing.translator import Translator
from src.example_rag.settings import COLLECTION_NAME, ChromaConfig, ProcessingConfig

NAME_REPLACEMENTS = {'Liud': 'Jon', 'Vasar': 'Karol'}

RESTORED_NAMES = {
    'John': 'Liudas',
    'Carol': 'Vasaris',
    'Charles': 'Vasaris',
    'Jon': 'Liud',
    'Karol': 'Vasar',
}

LOGGER = logging.getLogger(__name__)


def index(parsed_data_dir: str, db_host: str = ChromaConfig.host, db_port: int = ChromaConfig.port) -> None:
    """Index chromadb with the data from the data directory."""
    LOGGER.info('Populating the index...')

    LOGGER.info('Loading data...')
    reader = SimpleDirectoryReader(input_dir=parsed_data_dir)
    documents = reader.load_data()

    LOGGER.info('Splitting sentences...')
    splitter = SentenceSplitter(
        chunk_size=ProcessingConfig.chunk_size,
        chunk_overlap=ProcessingConfig.chunk_overlap,
    )

    nodes = splitter.get_nodes_from_documents(documents)[:10]  # [:20]

    lt_en_translator = Translator('lt-en')

    LOGGER.info('Translating...')
    for node in tqdm(nodes, desc='Node Translation'):
        for name, replacement in NAME_REPLACEMENTS.items():
            node.text = node.text.replace(name, replacement)
        node.text = lt_en_translator.translate_text(node.text)
        for name, replacement in RESTORED_NAMES.items():
            node.text = node.text.replace(name, replacement)

    LOGGER.info('Indexing...')
    embed_model = ArcticEmbedder()
    remote_db = chromadb.HttpClient(
        host=db_host,
        port=db_port,
        settings=Settings(allow_reset=True, anonymized_telemetry=False),
    )
    chroma_collection = remote_db.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    _ = VectorStoreIndex(
        storage_context=storage_context, embed_model=embed_model, nodes=nodes
    )
