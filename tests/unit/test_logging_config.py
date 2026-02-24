import logging

from src.utils.logging_config import configure_logging


def test_configure_logging_returns_logger(monkeypatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    logger = configure_logging("unit-test")
    assert isinstance(logger, logging.Logger)
