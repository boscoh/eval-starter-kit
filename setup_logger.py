"""
Logging configuration module for xConf Assistant FastAPI server.

This module provides a centralized logging setup using Rich for beautiful console output.
The configuration is designed to capture all relevant log messages including:
- Application logs from FastAPI server and MCP clients
- AWS/boto3 credential and connection errors
- Uvicorn startup and error messages
- Third-party library logs at appropriate levels

Key Features:
- Rich console formatting with custom theme
- Early initialization to capture startup errors
- Proper logger hierarchy configuration
- Integration with uvicorn logging
- Fallback handling for AWS credential issues

Usage:
    from setup_logger import setup_logging_with_rich_logger
    setup_logging_with_rich_logger()
"""

import logging
from typing import Union

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme


def setup_logging_with_rich_logger(
    level: Union[int, str] = logging.INFO,
) -> None:
    """Set up logging with Rich formatter and switching off noisy modules.

    Note:
        This should be called as early as possible in the application startup,
        preferably before any imports that might trigger AWS or other external
        service calls.
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper())

    custom_theme = Theme(
        {
            "logging.level.debug": "cyan",
            "logging.level.info": "green",
            "logging.level.warning": "yellow",
            "logging.level.error": "red",
            "logging.level.critical": "bold red",
        }
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    for handler in root_logger.handlers[:]:
        handler.close()
        root_logger.removeHandler(handler)

    console = Console(theme=custom_theme, stderr=True, width=140)

    formatter = logging.Formatter(
        "%(message)s",
        datefmt="%H:%M:%S",
    )

    rich_handler = RichHandler(
        console=console,
        rich_tracebacks=False,
        tracebacks_show_locals=False,
        show_time=True,
        show_path=True,
        show_level=True,
        markup=True,
        log_time_format="[%X]",
        keywords=[],
        omit_repeated_times=True,
    )

    rich_handler.setFormatter(formatter)
    rich_handler.setLevel(level)

    root_logger.addHandler(rich_handler)

    logger_configs = [
        ("__main__", logging.INFO, True),
        ("boto3", logging.INFO, True),
        ("botocore", logging.WARNING, True),
        ("aiobotocore", logging.WARNING, True),
        ("urllib3", logging.WARNING, True),
        ("httpx", logging.WARNING, True),
        ("httpcore", logging.WARNING, True),
        ("openai", logging.WARNING, True),
        ("h11", logging.WARNING, True),
        ("uvicorn", logging.INFO, True),
        ("uvicorn.access", logging.WARNING, True),
        ("uvicorn.error", logging.INFO, True),
        ("uvicorn.server", logging.INFO, True),
        ("mcp.server", logging.WARNING, True),
    ]

    for name, lvl, propagate in logger_configs:
        logger = logging.getLogger(name)
        if name != "":
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
        logger.setLevel(lvl)
        logger.propagate = propagate
