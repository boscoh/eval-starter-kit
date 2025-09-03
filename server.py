import json
import logging
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from path import Path
from rich.logging import RichHandler
from rich.pretty import pretty_repr

from evaluator import EvaluationRunner
from runner import Runner
from schemas import PROMPTS_DIR, QUERIES_DIR, RESULTS_DIR, RUNS_DIR, RunConfig
from util import load_yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        RichHandler(
            rich_tracebacks=True,
            show_time=False,
            show_path=True,
            markup=True,
            log_time_format="[%X]",
        )
    ],
    force=True,
)

logger = logging.getLogger(__name__)

# uvicorn_logger = logging.getLogger("uvicorn")
# uvicorn_logger.handlers = []
# uvicorn_logger.propagate = False

uvicorn_access = logging.getLogger("uvicorn.access")
uvicorn_access.handlers = []
uvicorn_access.propagate = False

# Configure h11 logging
logging.getLogger("h11").setLevel(logging.WARNING)

app = FastAPI()

# Allow CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def get_json_from_request(request) -> Dict[str, Any]:
    """Helper function to extract JSON data from a request.

    Args:
        request: The FastAPI request object

    Returns:
        Dict containing the parsed JSON data
    """
    return (
        await request.json()
        if hasattr(request, "json")
        else json.loads(await request.body())
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
    """Serves the main index page (index.html)
    
    Response:
        HTML content of the index page
    """
    try:
        index_path = Path("./index.html")
        logger.info(f"Serving index page from: {index_path}")
        if not index_path.exists():
            logger.error(f"index.html not found at {index_path}")
            raise HTTPException(status_code=404, detail="index.html not found")
        html_content = index_path.read_text(encoding="utf-8")
        return HTMLResponse(content=html_content)
    except Exception as ex:
        logger.error(f"Error serving index.html: {ex}")
        raise HTTPException(status_code=500, detail=f"Error serving index.html: {ex}")


def read_text_or_yaml(path: Path, ext: str):
    """
    Read a file with a given extension, or a YAML file if no extension is provided.
    If yaml file is read, will return a dictionary.
    Will raise HTTPException if the file is not found.
    """
    if ext:
        file_path = path.with_suffix(ext)
    else:
        file_path = path
    if not file_path.exists():
        logger.error(f"{path} '{file_path}' not found")
        raise HTTPException(
            status_code=404,
            detail=f"{path} '{file_path}' not found",
        )
    if file_path.suffix == ".yaml":
        try:
            result = load_yaml(file_path)
        except Exception as ex:
            logger.error(f"Error loading YAML: {ex}")
            raise HTTPException(status_code=500, detail=f"Error loading YAML: {ex}")
    else:  # assume is a text file
        try:
            result = file_path.read_text(encoding="utf-8")
        except Exception as ex:
            logger.error(f"Error reading path: {ex}")
            raise HTTPException(status_code=500, detail=f"Error reading path: {ex}")

    logger.info(f"Successfully loaded '{file_path}'")
    return result


async def get_json_field_from_request(request: Request, key: str):
    """
    Read a field from a JSON request body.
    Will raise HTTPException if the field is not found.
    """
    try:
        data = await get_json_from_request(request)
        value = data.get(key)
        if not value:
            logger.error(f"Missing '{key}' parameter in request")
            raise HTTPException(status_code=400, detail=f"{key} parameter is required")
        return value
    except Exception as ex:
        logger.error(f"Error getting {key} from request: {ex}")
        raise HTTPException(
            status_code=500, detail=f"Error getting {key} from request: {ex}"
        )


async def get_basenames_of_directory(this_dir: Path, ext: str = ".txt"):
    """
    Lists all basenames in a directory.
    Raises HTTPException if the directory is not found.
    """
    logger.info(f"Listing basenames in directory: {this_dir}")
    this_dir = Path(this_dir)
    try:
        basenames = [f.stem for f in this_dir.iterdir() if f.suffix == ext]
        logger.debug(f"Found basenames: {', '.join(basenames) or 'none'}")
        return basenames
    except Exception as ex:
        logger.error(f"Error listing basenames: {ex}")
        raise HTTPException(status_code=500, detail=f"Error listing basenames: {ex}")


@app.get("/evaluators")
def list_evaluators():
    """    
    Response:
        {
            "evaluators": ["string"] 
        }
    """
    try:
        allowed_evaluators = EvaluationRunner.allowed_evaluators()
        logger.debug(f"Available evaluators: {', '.join(allowed_evaluators)}")
        return {"evaluators": allowed_evaluators}
    except Exception as ex:
        logger.error(f"Error listing evaluators: {ex}")
        raise HTTPException(status_code=500, detail=f"Error listing evaluators: {ex}")


@app.get("/queries")
async def list_queries():
    """    
    Response:
        {
            "queries": ["string"] 
        }
    """
    return {"queries": await get_basenames_of_directory(QUERIES_DIR, ".yaml")}


@app.post("/query")
async def get_query(request: Request):
    """    
    Request Body:
        {
            "basename": "string"  
        }

    Response:
        {
            "content": object 
        }
    """
    basename = await get_json_field_from_request(request, "basename")
    logger.info(f"Request to get query for basename: '{basename}'")
    return {
        "content": read_text_or_yaml(QUERIES_DIR / basename, ".yaml"),
    }


@app.get("/system-prompts")
async def list_system_prompts():
    """    
    Response:
        {
            "system_prompts": ["string"] 
        }
    """
    logger.info(f"Listing system prompts in directory: {PROMPTS_DIR}")
    return {"system_prompts": await get_basenames_of_directory(PROMPTS_DIR, ".txt")}


@app.post("/system-prompt")
async def get_system_prompt(request: Request):
    """    
    Request Body:
        {
            "basename": "string"  
        }

    Response:
        {
            "content": "string" 
        }
    """
    basename = await get_json_field_from_request(request, "basename")
    logger.info(f"Request to get system prompt for basename: '{basename}'")
    return {
        "content": read_text_or_yaml(PROMPTS_DIR / basename, ".txt"),
    }


@app.get("/evals")
async def list_evals():
    """    
    Response:
        {
            "files": ["string"] 
        }
    """
    logger.info(f"Listing evals in directory: {RUNS_DIR}")
    return {"files": await get_basenames_of_directory(RUNS_DIR, ".yaml")}


@app.post("/create-eval")
async def create_eval(request: Request):
    """    
    Request Body:
        {
            "basename": "string",
            "content": object    
        }

    Response:
        {
            "basename": "string",
            "filename": "string",
            "content": object    
        }
    """
    try:
        data = await get_json_from_request(request)
        basename = data.get("basename")
        logger.info(f"Request to create eval for basename: {basename}")
        if not basename:
            logger.error("Missing 'basename' parameter in create_eval request")
            raise HTTPException(
                status_code=400, detail="basename parameter is required"
            )
        file_path = (RUNS_DIR / basename).with_suffix(".yaml")
        if file_path.exists():
            logger.error(f"Eval '{file_path}' already exists")
            raise HTTPException(
                status_code=400, detail=f"Eval '{file_path}' already exists"
            )
        config = RunConfig()
        config.query_ref = basename
        config.save(file_path)
        logger.info(pretty_repr(config))
        content = config.model_dump()
        logger.info(f"Eval '{file_path}' created successfully at {file_path}")
        return {
            "basename": basename,
            "content": content,
        }
    except Exception as ex:
        logger.error(f"Error creating eval: {ex}")
        raise HTTPException(status_code=500, detail=f"Error creating eval: {ex}")


@app.post("/eval")
async def get_eval(request: Request):
    """    
    Request Body:
        {
            "basename": "string"  
        }

    Response:
        {
            "content": object  
        }
    """
    basename = await get_json_field_from_request(request, "basename")
    logger.info(f"Received request for eval with basename: {basename}")
    return {
        "content": read_text_or_yaml(RUNS_DIR / basename, ".yaml")
    }


@app.post("/evaluate")
async def evaluate(request: Request):
    """    
    Request Body:
        {
            "basename": "string",  
            "testConfig": object
        }

    Response:
        {
            "success": true,
            "message": "string",
        }
    """
    try:
        data = await get_json_from_request(request)
        basename = data.get("basename")
        if not basename:
            raise HTTPException(status_code=400, detail="basename is required")

        test_config = data.get("testConfig")
        if not test_config:
            raise HTTPException(status_code=400, detail="testConfig is required")

        config_path = (RUNS_DIR / basename).with_suffix(".yaml")
        config = RunConfig(**test_config)
        config.save(config_path)
        job_runner = Runner(config_path)
        await job_runner.save_results()
        return {
            "success": True,
            "message": "Evaluation completed successfully",
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


@app.post("/result")
async def result(request: Request):
    """    
    Request Body:
        {
            "basename": "string"  
        }

    Response:
        {
            "basename": "string", 
            "content": object 
        }
    """
    basename = await get_json_field_from_request(request, "basename")
    logger.info(f"Received request at /result endpoint for basename: {basename}")
    return {
        "basename": basename,
        "content": read_text_or_yaml(RESULTS_DIR / basename, ".yaml"),
    }


@app.post("/create-system-prompt")
async def create_system_prompt(request: Request):
    """    
    Request Body:
        {
            "basename": "string",  
            "content": "string"    
        }

    Response:
        {
            "basename": "string", 
            "filename": "string", 
            "content": "string"   
        }
    """
    try:
        data = await get_json_from_request(request)
        basename = data.get("basename")
        logger.info(f"Request to create system prompt for basename: {basename}")
        if not basename:
            logger.error("Missing 'basename' parameter in create-system-prompt request")
            raise HTTPException(
                status_code=400, detail="basename parameter is required"
            )
        content = data.get("content")
        if content is None:
            logger.error("Missing 'content' parameter in create-system-prompt request")
            raise HTTPException(status_code=400, detail="content parameter is required")
        file_path = (PROMPTS_DIR / basename).with_suffix(".txt")
        if file_path.exists():
            logger.error(f"System prompt '{file_path}' already exists")
            raise HTTPException(
                status_code=400, detail=f"System prompt '{file_path}' already exists"
            )
        file_path.write_text(content, encoding="utf-8")
        logger.info(f"System prompt '{file_path}' created successfully at {file_path}")
        return {
            "basename": basename,
            "content": content,
        }
    except Exception as ex:
        logger.error(f"Error creating system prompt: {ex}")
        raise HTTPException(
            status_code=500, detail=f"Error creating system prompt: {ex}"
        )


@app.post("/save-system-prompt")
async def save_system_prompt(request: Request):
    """    
    Request Body:
        {
            "basename": "string",  
            "content": "string"    
        }

    Response:
        {
            "basename": "string", 
            "filename": "string", 
            "content": "string"    
        }
    """
    try:
        data = await get_json_from_request(request)
        basename = data.get("basename")
        content = data.get("content")
        if not basename:
            logger.error("Missing 'basename' parameter in save-system-prompt request")
            raise HTTPException(
                status_code=400, detail="basename parameter is required"
            )
        if content is None:
            logger.error("Missing 'content' parameter in save-system-prompt request")
            raise HTTPException(status_code=400, detail="content parameter is required")
        logger.info(f"Request to save system prompt for basename: {basename}")
        file_path = (PROMPTS_DIR / basename).with_suffix(".txt")
        file_path.write_text(content, encoding="utf-8")
        logger.info(f"System prompt '{basename}' saved successfully at {file_path}")
        return {"message": f"System prompt '{file_path}' saved successfully"}
    except Exception as ex:
        logger.error(f"Error saving system prompt: {ex}")
        raise HTTPException(status_code=500, detail=f"Error saving system prompt: {ex}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            RichHandler(
                rich_tracebacks=True,
                show_time=False,
                show_path=True,
                markup=True,
                log_time_format="[%X]",
            )
        ],
        force=True,
    )
    import os

    os.system("uvicorn server:app --host 0.0.0.0 --port 8000 --reload")
    # uvicorn.run(
    #     f"{Path(__file__).stem}:app",
    #     host="0.0.0.0",
    #     port=8000,
    #     reload=False,
    #     log_config=None,  # We handle our own logging
    #     access_log=False,  # Disable uvicorn access logs
    # )
