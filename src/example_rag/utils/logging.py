import logging


def set_logger_config():
    logging.basicConfig(
        format='%(asctime)s [%(levelname)s] - %(name)s - %(message)s',
        level=logging.INFO,
        handlers=[logging.StreamHandler()],
    )
