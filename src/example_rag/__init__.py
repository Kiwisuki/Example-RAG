import logging
from pathlib import Path

from src.example_rag.processing.indexing import index
from src.example_rag.processing.parsing import parse
from src.example_rag.settings import EMBEDDER_DEVICE, MODEL_DEVICE, TRANSLATOR_DEVICE
from src.example_rag.utils.logging import set_logger_config


def initialize():
    LOGGER = logging.getLogger(__name__)
    set_logger_config()

    LOGGER.info('Initializing...')

    LOGGER.info(f'Model device: {MODEL_DEVICE}')
    LOGGER.info(f'Translator device: {TRANSLATOR_DEVICE}')
    LOGGER.info(f'Embedder device: {EMBEDDER_DEVICE}')

    CURRENT_DIR = Path(__file__).resolve().parent
    RAW_DATA_DIR = CURRENT_DIR / 'data' / 'raw_data'
    PARSED_DATA_DIR = CURRENT_DIR / 'data' / 'parsed_data'
    DB_DATA_DIR = CURRENT_DIR / 'data' / 'db_data'

    IS_PARSED = len(list(PARSED_DATA_DIR.iterdir())) > 1
    IS_MOUNTED = len(list(DB_DATA_DIR.iterdir())) > 1
    IS_INDEXED = len(list(DB_DATA_DIR.iterdir())) > 2

    if not IS_PARSED:
        logging.info('Data is not parsed: Parsing...')
        parse(RAW_DATA_DIR, PARSED_DATA_DIR)

    if not IS_MOUNTED:
        logging.critical('Chroma container is not mounted: Data will not persist!')

    if not IS_INDEXED:
        logging.info('Data is not indexed: Indexing...')
        index(PARSED_DATA_DIR)

    LOGGER.info('Initialization complete!')


initialize()
