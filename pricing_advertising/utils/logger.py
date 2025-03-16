import sys

from loguru import logger

# Configure the logger
logger.remove()
logger.add(
    sys.stdout,
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <lvl>{level: <8}</lvl> | {extra[app]} | {extra[component]} | {message}",
)
