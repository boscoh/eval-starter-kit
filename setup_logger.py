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
    """Set up logging with Rich formatter and comprehensive logger configuration.

    This function configures the root logger and all relevant child loggers to ensure
    that all important log messages are captured and formatted consistently. It's
    particularly important for capturing AWS credential errors and startup messages
    that might otherwise be missed.

    Args:
        level: Logging level (e.g., 'DEBUG', 'INFO', 'WARNING', etc.)

    Note:
        This should be called as early as possible in the application startup,
        preferably before any imports that might trigger AWS or other external
        service calls.
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper())

    # Create a custom theme for Rich
    custom_theme = Theme(
        {
            "logging.level.debug": "cyan",
            "logging.level.info": "green",
            "logging.level.warning": "yellow",
            "logging.level.error": "red",
            "logging.level.critical": "bold red",
        }
    )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove all existing handlers
    for handler in root_logger.handlers[:]:
        handler.close()
        root_logger.removeHandler(handler)

    # Create Rich handler with detailed configuration
    console = Console(theme=custom_theme, stderr=True, width=140)

    # Create a clean formatter without duplicate log levels
    formatter = logging.Formatter(
        "%(message)s",  # RichHandler will add the level/name/line info
        datefmt="%H:%M:%S",
    )

    # Configure the Rich handler
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
        omit_repeated_times=False,
    )

    # Apply the formatter
    rich_handler.setFormatter(formatter)
    rich_handler.setLevel(level)

    root_logger.addHandler(rich_handler)

    # Configure specific loggers with proper level and propagation
    # Format: (logger_name, level, propagate_to_parent)
    logger_configs = [
        # Application loggers - capture all application-level messages
        ("__main__", logging.INFO, True),
        ("fastapi_server", logging.INFO, True),
        ("quest_mcp_client", logging.INFO, True),
        ("mcp_client", logging.INFO, True),
        ("database", logging.INFO, True),
        ("aws_auth", logging.INFO, True),
        ("resources", logging.INFO, True),
        # AWS/Boto3 loggers - CRITICAL: capture these at INFO level to see credential errors
        # These loggers often contain important error messages about AWS authentication
        # that would otherwise be missed, especially during application startup
        ("boto3", logging.INFO, True),
        ("botocore", logging.INFO, True),
        ("botocore.credentials", logging.INFO, True),  # SSO token expiration errors
        ("botocore.auth", logging.INFO, True),  # Authentication errors
        ("urllib3", logging.WARNING, True),  # Keep HTTP noise down
        # HTTP and other third-party loggers - reduce noise but keep errors
        ("httpx", logging.WARNING, True),
        ("httpcore", logging.WARNING, True),
        ("openai", logging.WARNING, True),
        ("h11", logging.WARNING, True),
        # Uvicorn loggers - keep these at INFO to capture startup messages
        # These are important for debugging server startup issues
        ("uvicorn", logging.INFO, True),
        ("uvicorn.access", logging.WARNING, True),  # Reduce HTTP access log noise
        ("uvicorn.error", logging.INFO, True),  # Capture server errors
        ("uvicorn.server", logging.INFO, True),  # Capture server lifecycle events
    ]

    for name, lvl, propagate in logger_configs:
        logger = logging.getLogger(name)
        # Don't remove handlers for root logger to avoid conflicts
        if name != "":
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
        logger.setLevel(lvl)
        logger.propagate = propagate
