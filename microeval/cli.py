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
from microeval.chat_client import load_config
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


@app.command
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


@app.command
def run(
    base_dir: str,
):
    """Run all LLM evaluations in a directory.
    
    Args:
        base_dir: Base directory for evals (e.g., evals-consultant, evals-engineer)
    """
    evals_dir.set_base(base_dir)

    logger.info(f"Running all configs in `./{evals_dir.runs}/*.yaml`")
    file_paths = list(evals_dir.runs.glob("*.yaml"))

    if not file_paths:
        logger.warning(f"No config files found in {evals_dir.runs}")
        return

    asyncio.run(run_all(file_paths))


@app.command
def demo(
    base_dir: str = "sample-evals",
    port: int = 8000,
):
    """Create sample evaluations and launch UI.
    
    Args:
        base_dir: Directory for demo evals
        port: Port to run the server on
    """
    demo_dir = Path(base_dir)
    sample_evals_path = Path(__file__).parent / "sample-evals"
    
    if demo_dir.exists():
        logger.info(f"Using existing {base_dir}")
    else:
        if not sample_evals_path.exists():
            logger.error(f"sample-evals template not found at {sample_evals_path}")
            raise SystemExit(1)
        logger.info(f"Creating {base_dir} from template")
        shutil.copytree(sample_evals_path, demo_dir)
    
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


@app.command
def chat(
    service: str,
):
    """Interactive chat loop with LLM providers.
    
    Args:
        service: LLM service to use
    """
    config = load_config()
    available_services = config.get("chat_models", {}).keys()
    
    if service not in available_services:
        services = ", ".join(available_services)
        raise ValueError(f"Unknown service '{service}'. Available: {services}")

    chat_main(service)


def main():
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
