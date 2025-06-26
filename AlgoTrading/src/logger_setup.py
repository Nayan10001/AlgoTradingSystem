import logging
import sys

def setup_logger(name='AlgoTrader', level=logging.INFO):
    logger = logging.getLogger(name)
    if not logger.handlers: # <--to Avoid adding multiple handlers
        logger.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Console Handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

if __name__ == '__main__':
    logger = setup_logger()
    logger.info("Logger setup complete.")
    logger.warning("This is a warning.")
    logger.error("This is an error.")