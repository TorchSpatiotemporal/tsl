import logging.config

from tsl.imports import _HYDRA_AVAILABLE

if not _HYDRA_AVAILABLE:
    DEFAULT_LOGGING = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {"format": "%(asctime)s [%(levelname)s]: %(message)s"},
        },
        "handlers": {
            "default": {
                "level": "INFO",
                "formatter": "standard",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "log": {"handlers": ["default"], "level": "INFO", "propagate": True}
        },
    }
    logging.config.dictConfig(DEFAULT_LOGGING)
logger = logging.getLogger("tsl")
