"""Typer CLI for starteval."""

import asyncio
import logging
import os
import shutil
import threading
from pathlib import Path

import typer
import uvicorn

from starteval.chat import main as chat_main
from starteval.runner import run_all
from starteval.schemas import evals_dir
from starteval.server import is_in_container, poll_and_open_browser
from starteval.setup_logger import setup_logging

logger = logging.getLogger(__name__)

app = typer.Typer(no_args_is_help=True)


@app.command()
def ui(
    ctx: typer.Context,
    base_dir: str = typer.Argument(
        None, help="Base directory for evals"
    ),
    port: int = typer.Option(8000, help="Port to run the server on"),
    reload: bool = typer.Option(False, help="Enable auto-reload"),
) -> None:
    """Run the web UI for evaluations."""
    if not base_dir:
        typer.echo(ctx.get_help())
        raise typer.Exit()
    
    setup_logging()
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
        "starteval.server:app",
        host="0.0.0.0",
        port=port,
        reload=reload,
        reload_dirs=[base_dir],
        log_config=None,
    )


@app.command()
def run(
    ctx: typer.Context,
    base_dir: str = typer.Argument(
        None, help="Base directory for evals (e.g., evals-consultant, evals-engineer)"
    ),
) -> None:
    """Run all LLM evaluations in a directory."""
    if not base_dir:
        typer.echo(ctx.get_help())
        raise typer.Exit()

    setup_logging()
    evals_dir.set_base(base_dir)

    logger.info(f"Running all configs in `./{evals_dir.runs}/*.yaml`")
    file_paths = list(evals_dir.runs.glob("*.yaml"))

    if not file_paths:
        logger.warning(f"No config files found in {evals_dir.runs}")
        return

    asyncio.run(run_all(file_paths))


@app.command()
def demo(
    base_dir: str = typer.Argument(
        "sample-evals", help="Directory for demo evals"
    ),
    port: int = typer.Option(8000, help="Port to run the server on"),
) -> None:
    """Create sample evaluations and launch UI."""
    setup_logging()
    
    demo_dir = Path(base_dir)
    sample_evals_path = Path(__file__).parent / "sample-evals"
    
    if demo_dir.exists():
        logger.info(f"Using existing {base_dir}")
    else:
        if not sample_evals_path.exists():
            logger.error(f"sample-evals template not found at {sample_evals_path}")
            raise typer.Exit(1)
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
        "starteval.server:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        reload_dirs=[base_dir],
        log_config=None,
    )


@app.command()
def chat(
    ctx: typer.Context,
    service: str = typer.Argument(
        None, help="LLM service to use (openai, bedrock, ollama, groq)"
    ),
) -> None:
    """Interactive chat loop with LLM providers."""
    if not service:
        typer.echo(ctx.get_help())
        raise typer.Exit()

    setup_logging()
    chat_main(service)


def main() -> None:
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
