import logging
from typing import Final

_LOG_LEVELS: Final[dict[str, int]] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}
_LOG_FORMAT: Final[str] = "%(asctime)s %(levelname)s %(name)s: %(message)s"

LOG_LEVEL_CHOICES: Final[tuple[str, ...]] = tuple(_LOG_LEVELS)
DEFAULT_LOG_LEVEL: Final[str] = "INFO"


def configure_logging(level: str = DEFAULT_LOG_LEVEL) -> None:
    logging.basicConfig(level=_LOG_LEVELS[level], format=_LOG_FORMAT)
