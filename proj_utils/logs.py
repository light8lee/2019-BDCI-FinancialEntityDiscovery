import logging
import logging.handlers

def log_info(filename):
    logger = logging.getLogger('e4g_logger')
    logger.setLevel(logging.DEBUG)

    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    def _log(*values):
        message = ' '.join([str(v) for v in values])
        logger.info(message)
    return _log