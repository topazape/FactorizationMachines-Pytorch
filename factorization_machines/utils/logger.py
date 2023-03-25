import logging


def get_logger(filename: str | None = None) -> logging.Logger:
    """Get a logger.

    Args:
        filename: The filename to log to.
    """
    fmt = logging.Formatter(
        fmt="[%(asctime)s] :%(name)s: [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    handler: logging.StreamHandler | logging.FileHandler = (
        logging.FileHandler(filename=filename)
        if filename is not None
        else logging.StreamHandler()
    )

    handler.setFormatter(fmt)
    logger.addHandler(handler)

    return logger
