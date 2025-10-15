import json
import logging
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from path import Path
from pydantic import BaseModel

from evaluator import EvaluationRunner
from runner import Runner
from schemas import (
    RunConfig,
    TableType,
    dir_from_table,
    ext_from_table,
)
from setup_logger import setup_logging_with_rich_logger
from yaml_utils import load_yaml, save_yaml

logger = logging.getLogger(__name__)
setup_logging_with_rich_logger()
logger.info("Logging configured with Rich handler")


async def get_json_from_request(request) -> Dict[str, Any]:
    """Returns parsed json from request"""
    return (
        await request.json()
        if hasattr(request, "json")
        else json.loads(await request.body())
    )


def read_content(file_path: str):
    """Returns a JSON object for ext='.yaml', or a string if ext='.txt" """
    file_path = Path(file_path)
    ext = file_path.suffix
    if ext == ".yaml":
        content = load_yaml(file_path)
    else:
        content = file_path.read_text(encoding="utf-8")
    return content


def save_content(content, file_path):
    if file_path.parent:
        file_path.parent.makedirs_p()
    ext = file_path.suffix
    if ext == ".yaml":
        save_yaml(content, file_path)
    else:
        file_path.write_text(content)


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
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
def get_defaults():
    """
    Response: { "content": object }
    """
    return {
        "content": {
            "evaluators": EvaluationRunner.evaluators(),
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


class ContentResponse(BaseModel):
    content: Any


@app.get("/list/{table}")
async def list_objects(table):
    """Response: { "content": ["string"] }"""
    try:
        table_dir = dir_from_table[table]
        ext = ext_from_table[table]
        logger.info(f"Request for names in: {table_dir}")
        basenames = [f.stem for f in table_dir.iterdir() if f.suffix == ext]
        logger.info(f"Found: {len(basenames)} names in {table_dir} with {ext}")
        return ContentResponse(content=basenames)
    except Exception as ex:
        logger.error(f"Error listing basenames: {ex}")
        raise HTTPException(status_code=500, detail=f"Error listing basenames: {ex}")


class FetchObjectRequest(BaseModel):
    table: TableType
    basename: str


@app.post("/fetch", response_model=ContentResponse)
async def fetch_object(request: FetchObjectRequest):
    try:
        logger.info(f"Request to fetch {request.table}/{request.basename}")
        table_dir = dir_from_table[request.table]
        ext = ext_from_table[request.table]
        file_path = (table_dir / request.basename).with_suffix(ext)
        content = read_content(file_path)
        logger.info(f"Successfully loaded '{file_path}'")
        return ContentResponse(content=content)

    except KeyError as ke:
        error_msg = f"Invalid table or basename: {ke}"
        logger.error(error_msg)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=error_msg)
    except FileNotFoundError as fnf:
        error_msg = f"File not found: {fnf}"
        logger.error(error_msg)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=error_msg)
    except Exception as ex:
        error_msg = f"Error reading object: {ex}"
        logger.error(error_msg)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_msg
        )


class SaveObjectRequest(BaseModel):
    table: TableType
    basename: str
    content: Any


class MessageResponse(BaseModel):
    message: str


@app.post("/save", response_model=MessageResponse)
async def save_object(request: SaveObjectRequest):
    try:
        logger.info(f"Request to save {request.table}/{request.basename}")
        table = request.table
        table_dir = dir_from_table[table]
        ext = ext_from_table[table]
        file_path = (table_dir / request.basename).with_suffix(ext)
        save_content(request.content, file_path)
        logger.info(f"Successfully saved to '{file_path}'")
        return MessageResponse(
            message=f"Successfully saved {request.table}/{request.basename}"
        )

    except Exception as ex:
        error_msg = f"Error saving object: {ex}"
        logger.error(error_msg)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_msg
        )


class EvaluateRequest(BaseModel):
    basename: str
    content: Any


@app.post("/evaluate", response_model=MessageResponse)
async def evaluate(request: EvaluateRequest):
    try:
        basename = request.basename
        config = request.content
        config_path = (dir_from_table["run"] / basename).with_suffix(".yaml")
        logger.info(f"Running evaluation of runs/{basename}")
        run_config = RunConfig(**config)
        run_config.save(config_path)
        await Runner(config_path).run()
        return MessageResponse(message=f"Successfully evaluated {request.basename}")

    except Exception as e:
        logger.error(f"Error in evaluation: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error during evaluation: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True, log_config=None)
