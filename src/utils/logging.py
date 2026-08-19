"""Logging configuration helpers."""
import logging


def configure_logging() -> None:
    """Configure the standard structured console log format."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
