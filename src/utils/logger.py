import logging, sys


def get_logger(name):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(
        logging.Formatter(
            "[%(asctime)s] %(levelname)-8s %(name)s: %(message)s", "%H:%M:%S"
        )
    )
    logger.addHandler(h)
    return logger
