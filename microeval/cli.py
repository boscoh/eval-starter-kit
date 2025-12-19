"""CLI for microeval LLM evaluation framework."""

import asyncio
import logging
import os
import shutil
import threading
from pathlib import Path

import cyclopts
import uvicorn

from microeval.chat import main as chat_main
from microeval.chat_client import LLMService
from microeval.runner import run_all
from microeval.schemas import evals_dir
from microeval.server import is_in_container, poll_and_open_browser
from microeval.setup_logger import setup_logging

logger = logging.getLogger(__name__)

setup_logging()

app = cyclopts.App(help_format="markdown")


@app.default
def help_command():
    """Show help information."""
    print(__doc__)
    print()
    app.help_print([])


@app.command(sort_key=0)
def ui(
    base_dir: str,
    port: int = 8000,
    reload: bool = False,
):
    """Run the web UI for evaluations.
    
    Args:
        base_dir: Base directory for evals
        port: Port to run the server on
        reload: Enable auto-reload
    """
    evals_dir.set_base(base_dir)
    os.environ["EVALS_DIR"] = base_dir

    if not is_in_container():
        poller_thread = threading.Thread(
            target=poll_and_open_browser, args=(port,), daemon=True
        )
        poller_thread.start()
    else:
        logger.info("Running in container, skipping browser auto-open")

    uvicorn.run(
        "microeval.server:app",
        host="0.0.0.0",
        port=port,
        reload=reload,
        reload_dirs=[base_dir],
        log_config=None,
    )


@app.command(sort_key=1)
def run(
    base_dir: str,
):
    """Run all LLM evaluations in a directory.
    
    Args:
        base_dir: Base directory for evals (e.g., evals-consultant, evals-engineer)
    """
    evals_dir.set_base(base_dir)

    logger.info(f"Running all configs in './{evals_dir.runs}/*.yaml'")
    file_paths = list(evals_dir.runs.glob("*.yaml"))

    if not file_paths:
        logger.warning(f"No config files found in {evals_dir.runs}")
        return

    asyncio.run(run_all(file_paths))


def _run_demo(template_name: str, base_dir: str, port: int):
    """Helper to run demo with a specific template."""
    demo_dir = Path(base_dir)
    template_path = Path(__file__).parent / template_name
    
    if demo_dir.exists():
        logger.info(f"Using existing {base_dir}")
    else:
        if not template_path.exists():
            logger.error(f"{template_name} template not found at {template_path}")
            raise SystemExit(1)
        logger.info(f"Creating {base_dir} from template")
        shutil.copytree(template_path, demo_dir)
    
    evals_dir.set_base(base_dir)
    os.environ["EVALS_DIR"] = base_dir
    
    if not is_in_container():
        poller_thread = threading.Thread(
            target=poll_and_open_browser, args=(port,), daemon=True
        )
        poller_thread.start()
    else:
        logger.info("Running in container, skipping browser auto-open")
    
    uvicorn.run(
        "microeval.server:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        reload_dirs=[base_dir],
        log_config=None,
    )


@app.command(sort_key=2)
def demo1(
    base_dir: str = "summary-evals",
    port: int = 8000,
):
    """Demo with summary evaluations."""
    _run_demo("summary-evals", base_dir, port)


@app.command(sort_key=3)
def demo2(
    base_dir: str = "json-evals",
    port: int = 8000,
):
    """Demo with JSON evaluations."""
    _run_demo("json-evals", base_dir, port)


@app.command(sort_key=4)
def chat(
    service: LLMService | None = None,
):
    """Interactive chat loop with LLM providers."""
    if service is None:
        app.help_print(["chat"])
        return
    chat_main(service)


def main():
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
