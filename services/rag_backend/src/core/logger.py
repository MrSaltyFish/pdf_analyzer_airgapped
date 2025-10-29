# src/core/logger.py
import logging
from datetime import datetime
from rich.logging import RichHandler
from src.core import config

config.LOG_DIR.mkdir(exist_ok=True)
log_file = config.LOG_DIR / f"run_{datetime.now().strftime('%Y%m%d')}.log"

formatter = logging.Formatter(
    "%(asctime)s.%(msecs)03d [%(levelname)-8s] [%(name)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

file_handler = logging.FileHandler(log_file, encoding="utf-8")
file_handler.setFormatter(formatter)

console_handler = RichHandler(rich_tracebacks=True)
console_handler.setFormatter(formatter)

logging.basicConfig(
    level=config.LOG_LEVEL,
    handlers=[file_handler, console_handler],
)

def get_logger(name: str = "app"):
    """Get a logger instance for a module."""
    return logging.getLogger(name)
