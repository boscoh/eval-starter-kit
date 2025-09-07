import logging

from rich.console import Console
from rich.logging import RichHandler


def setup_logging_with_rich_logger(
    level: int = logging.INFO,
) -> None:
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    console = Console()
    console_handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        show_time=True,
        show_path=True,
        markup=True,
        log_time_format="[%X]",
    )
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger.addHandler(console_handler)

    # Configure specific loggers
    logging.getLogger("uvicorn").handlers = []
    logging.getLogger("uvicorn").propagate = True
    logging.getLogger("uvicorn.access").handlers = []
    logging.getLogger("uvicorn.access").propagate = True
    logging.getLogger("uvicorn.error").handlers = []
    logging.getLogger("uvicorn.error").propagate = True

    # Suppress HTTP client logs
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpcore.http11").setLevel(logging.WARNING)
    logging.getLogger("httpcore.connection").setLevel(logging.WARNING)

    # Suppress h11 and server logs
    logging.getLogger("h11").setLevel(logging.WARNING)
    logging.getLogger("h11._impl").setLevel(logging.WARNING)
    logging.getLogger("h11_impl").setLevel(logging.WARNING)

    # Suppress server logs
    logging.getLogger("uvicorn.server").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
