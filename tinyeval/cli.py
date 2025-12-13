"""Typer CLI for tinyeval."""

import asyncio
import logging
import os
import threading

import typer
import uvicorn

from tinyeval.chat import main as chat_main
from tinyeval.runner import run_all
from tinyeval.schemas import set_evals_dir
from tinyeval.server import is_in_container, poll_and_open_browser
import tinyeval.schemas as schemas
from tinyeval.setup_logger import setup_logging

logger = logging.getLogger(__name__)

app = typer.Typer(no_args_is_help=True)


@app.command()
def ui(
    evals_dir: str = typer.Argument(
        "evals-consultant", help="Base directory for evals"
    ),
    port: int = typer.Option(8000, help="Port to run the server on"),
    reload: bool = typer.Option(False, help="Enable auto-reload"),
) -> None:
    """Run the web UI for evaluations."""
    setup_logging()
    set_evals_dir(evals_dir)
    os.environ["EVALS_DIR"] = evals_dir

    if not is_in_container():
        poller_thread = threading.Thread(
            target=poll_and_open_browser, args=(port,), daemon=True
        )
        poller_thread.start()
    else:
        logger.info("Running in container, skipping browser auto-open")

    uvicorn.run(
        "tinyeval.server:app",
        host="0.0.0.0",
        port=port,
        reload=reload,
        reload_dirs=[evals_dir],
        log_config=None,
    )


@app.command()
def run(
    ctx: typer.Context,
    evals_dir: str = typer.Argument(
        None, help="Base directory for evals (e.g., evals-consultant, evals-engineer)"
    ),
) -> None:
    """Run all LLM evaluations in a directory."""
    if not evals_dir:
        typer.echo(ctx.get_help())
        raise typer.Exit()

    setup_logging()
    set_evals_dir(evals_dir)

    logger.info(f"Running all configs in `./{schemas.RUNS_DIR}/*.yaml`")
    file_paths = list(schemas.RUNS_DIR.glob("*.yaml"))

    if not file_paths:
        logger.warning(f"No config files found in {schemas.RUNS_DIR}")
        return

    asyncio.run(run_all(file_paths))


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
