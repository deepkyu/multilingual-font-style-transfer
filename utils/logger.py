import fire
import logging
import time

def _custom_logger(name):
    fmt = '[{}|%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s >>> %(message)s'.format(name)
    fmt_date = '%Y-%m-%d_%T %Z'

    handler = logging.StreamHandler()

    formatter = logging.Formatter(fmt, fmt_date)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)

def set_logger(logger_name, level):
    try:
        time.tzset()
    except AttributeError as e:
        print(e)
        print("Skipping timezone setting.")
    _custom_logger(name=logger_name)
    logger = logging.getLogger(logger_name)
    if level == 'DEBUG':
        logger.setLevel(logging.DEBUG)
    elif level == 'INFO':
        logger.setLevel(logging.INFO)
    elif level == 'WARNING':
        logger.setLevel(logging.WARNING)
    elif level == 'ERROR':
        logger.setLevel(logging.ERROR)
    elif level == 'CRITICAL':
        logger.setLevel(logging.CRITICAL)
    return logger

if __name__ == '__main__':
    set_logger("test", "DEBUG")