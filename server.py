import json
import logging
import os
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from path import Path
from rich.logging import RichHandler

from evaluator import EvaluationRunner
from runner import Runner
from schemas import PROMPTS_DIR, QUERIES_DIR, RESULTS_DIR, RUNS_DIR, RunConfig
from util import load_yaml, save_yaml

logger = logging.getLogger(__name__)

uvicorn_access = logging.getLogger("uvicorn.access")
uvicorn_access.handlers = []
uvicorn_access.propagate = False
logging.getLogger("h11").setLevel(logging.WARNING)


logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        RichHandler(
            rich_tracebacks=True,
            show_time=True,
            show_path=True,
            markup=True,
            log_time_format="[%X]",
        )
    ],
    force=True,
)


dir_from_table = {
    "result": RESULTS_DIR,
    "run": RUNS_DIR,
    "prompt": PROMPTS_DIR,
    "query": QUERIES_DIR,
}

ext_from_table = {
    "result": ".yaml",
    "run": ".yaml",
    "prompt": ".txt",
    "query": ".yaml",
}


async def get_json_from_request(request) -> Dict[str, Any]:
    """Returns parsed json from request"""
    return (
        await request.json()
        if hasattr(request, "json")
        else json.loads(await request.body())
    )


def read_content(file_path: str):
    """ Returns a JSON object for ext='.yaml', or a string if ext='.txt" """
    file_path = Path(file_path)
    ext = file_path.suffix
    if ext == ".yaml":
        content = load_yaml(file_path)
    else:  # assume text file
        content = file_path.read_text(encoding="utf-8")
    return content


def save_content(content, file_path):
    ext = file_path.suffix
    if ext == ".yaml":
        save_yaml(content, file_path)
    else:
        file_path.write_text(content)


app = FastAPI()


# Allow CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    # Only log non-asset requests
    if not any(
        ext in str(request.url) for ext in [".js", ".css", ".ico", ".png", ".jpg"]
    ):
        logger.info(f"{request.method} {request.url.path}")
    return await call_next(request)


@app.get("/", response_class=HTMLResponse)
def serve_index():
    """Serves the main index page (index.html)"""
    index_path = Path("./index.html")
    logger.info(f"Serving index page from: {index_path}")
    try:
        html_content = index_path.read_text(encoding="utf-8")
        return HTMLResponse(content=html_content)
    except Exception as ex:
        logger.error(f"Error serving index.html: {ex}")
        raise HTTPException(status_code=500, detail=f"Error serving index.html: {ex}")


@app.get("/defaults")
def list_evaluators():
    """
    Response: { "evaluators": ["string"] }
    """
    try:
        allowed_evaluators = EvaluationRunner.allowed_evaluators()
        logger.info(f"Available evaluators: {', '.join(allowed_evaluators)}")
        return {
            "content": {
                "evaluators": allowed_evaluators,
                "run_config": {
                    "promptRef": "",
                    "queryRef": "",
                    "prompt": "",
                    "input": "",
                    "output": "",
                    "service": "ollama",
                    "model": "llama3.2",
                    "repeat": 1,
                    "evaluators": ["CoherenceEvaluator"],
                },
            }
        }
    except Exception as ex:
        logger.error(f"Error listing evaluators: {ex}")
        raise HTTPException(status_code=500, detail=f"Error listing evaluators: {ex}")


@app.get("/list/{table}")
async def list_objects(table):
    """Response: { "content": ["string"] }"""
    table_dir = dir_from_table[table]
    ext = ext_from_table[table]
    logger.info(f"Listing basenames in directory: {table_dir}")
    try:
        basenames = [f.stem for f in table_dir.iterdir() if f.suffix == ext]
        logger.info(f"Found basenames: {', '.join(basenames) or 'none'}")
        return {"content": basenames}
    except Exception as ex:
        logger.error(f"Error listing basenames: {ex}")
        raise HTTPException(status_code=500, detail=f"Error listing basenames: {ex}")


@app.post("/fetch")
async def fetch_object(request: Request):
    """
    Request Body: { "table": string, "basename": "string" }
    Response: { "content": ["string"] }
    Note: we post because basename is not guaranteed to be a valid URL parameter
    """
    try:
        data = await get_json_from_request(request)
        for field in ["table", "basename"]:
            if not data.get(field):
                logger.error(f"Missing '{field}' save request")
                raise HTTPException(status_code=400, detail=f"{field} is required")
        basename = data.get("basename")
        table = data.get("table")
        logger.info(f"Request to fetch {table} for basename: {basename}")
        table_dir = dir_from_table[table]
        ext = ext_from_table[table]
        file_path = (table_dir / basename).with_suffix(ext)
        logger.info(f"Reading {file_path} to fetch {table} for basename: {basename}")
        content = read_content(file_path)
        logger.info(f"Successfully loaded '{file_path}'")
        return {"content": content}
    except Exception as ex:
        logger.error(f"Error reading object: {ex}")
        raise HTTPException(status_code=500, detail=f"Error reading: {ex}")


@app.post("/save")
async def save_object(request: Request):
    """
    Request Body: { "table": string, "basename": "string", "content": object }
    Response: { "message": "string" }
    """
    try:
        data = await get_json_from_request(request)
        for field in ["table", "basename", "content"]:
            if not data.get(field):
                logger.error(f"Missing '{field}' save request")
                raise HTTPException(status_code=400, detail=f"{field} is required")
        basename = data.get("basename")
        table = data.get("table")
        content = data.get("content")

        logger.info(f"Request to save {table} for basename: {basename}")
        parent_dir = dir_from_table[table]
        ext = ext_from_table[table]
        file_path = (parent_dir / basename).with_suffix(ext)

        save_content(content, file_path)

        msg = f"{dir} '{basename}' saved successfully at {file_path}"
        logger.info(msg)
        return {"message": msg}
    except Exception as ex:
        logger.error(f"Error saving {dir}: {ex}")
        raise HTTPException(status_code=500, detail=f"Error saving {dir}: {ex}")


@app.post("/evaluate")
async def evaluate(request: Request):
    """
    Request Body: { "basename": "string", "content": object }
    Response: { "success": true }
    """
    try:
        data = await get_json_from_request(request)
        basename = data.get("basename")
        if not basename:
            raise HTTPException(status_code=400, detail="basename is required")

        config = data.get("content")
        if not config:
            raise HTTPException(status_code=400, detail="config is required")

        config_path = (RUNS_DIR / basename).with_suffix(".yaml")
        run_config = RunConfig(**config)
        run_config.save(config_path)
        job_runner = Runner(config_path)
        logger.info(f"Running evaluation with config '{config_path}'")
        await job_runner.save_results()
        return {
            "success": True,
        }
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in request body")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in evaluation: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error during evaluation: {str(e)}"
        )


if __name__ == "__main__":
    os.system("uvicorn server:app --host 0.0.0.0 --port 8000 --reload")
