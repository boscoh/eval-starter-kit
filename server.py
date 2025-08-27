import json
import logging
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from path import Path
from rich.logging import RichHandler

from evaluator import EvaluationRunner
from runner import JobRunner
from util import load_yaml, save_yaml

JOB_DIR = Path("./jobs")
PROMPTS_DIR = Path("system-prompts")
PROMPTS_DIR.makedirs_p()
RESULTS_DIR = Path("results")
RESULTS_DIR.makedirs_p()

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


# GET /summary
# Retrieves a summary YAML file by basename
#
# Request Body:
# {
#     "basename": "string"  # Required: The basename of the summary file to retrieve
# }
#
# Response:
# {
#     "basename": "string",
#     "content": "object"  # Parsed YAML content
# }
@app.post("/summary")
async def summary(request: Request):
    try:
        data = await get_json_from_request(request)
        basename = data.get("basename")
        logger.info(f"Received request at /summary endpoint for basename: {basename}")
        if not basename:
            logger.error("Missing 'basename' parameter in /summary request")
            raise HTTPException(
                status_code=400, detail="basename parameter is required"
            )
        file_path = (RESULTS_DIR / basename).with_suffix(".yaml")
        if not file_path.exists():
            logger.error(f"Summary file '{file_path}' not found in {RESULTS_DIR}")
            raise HTTPException(
                status_code=404,
                detail=f"Summary file '{file_path}' not found in {RESULTS_DIR}",
            )
        logger.debug(f"Loaded summary for {basename}")
        return {
            "basename": basename,
            "content": load_yaml(file_path),
        }
    except Exception as ex:
        logger.error(f"Error in /summary endpoint: {ex}")
        raise HTTPException(status_code=500, detail=f"Error in /summary endpoint: {ex}")


# GET /evaluators
# Lists all available evaluators
#
# Response:
# {
#     "evaluators": ["string"]  # Array of available evaluator names
# }
@app.get("/evaluators")
def list_evaluators():
    try:
        allowed_evaluators = EvaluationRunner.allowed_evaluators()
        logger.debug(f"Available evaluators: {', '.join(allowed_evaluators)}")
        return {"evaluators": allowed_evaluators}
    except Exception as ex:
        logger.error(f"Error listing evaluators: {ex}")
        raise HTTPException(status_code=500, detail=f"Error listing evaluators: {ex}")


# GET /system-prompts
# Lists available system prompt files
#
# Response:
# {
#     "system_prompts": ["string"]  # Array of system prompt basenames
# }
@app.get("/system-prompts")
def list_system_prompts():
    logger.info(f"Listing system prompts in directory: {PROMPTS_DIR}")
    try:
        basenames = [f.stem for f in PROMPTS_DIR.iterdir() if f.suffix == ".txt"]
        logger.debug(f"Found system prompts: {', '.join(basenames) or 'none'}")
        return {"system_prompts": basenames}
    except Exception as ex:
        logger.error(f"Error listing system prompts: {ex}")
        raise HTTPException(
            status_code=500, detail=f"Error listing system prompts: {ex}"
        )


# POST /system-prompt
# Retrieves a system prompt file by basename
#
# Request Body:
# {
#     "basename": "string"  # Required: The basename of the system prompt file
# }
#
# Response:
# {
#     "basename": "string",
#     "content": "string"  # Raw text content of the system prompt
# }
@app.post("/system-prompt")
async def get_system_prompt(request: Request):
    try:
        data = await get_json_from_request(request)
        basename = data.get("basename")
        if not basename:
            logger.error("Missing 'basename' parameter in system-prompt request")
            raise HTTPException(
                status_code=400, detail="basename parameter is required"
            )
        logger.info(f"Request to get system prompt for basename: {basename}")
        file_path = (PROMPTS_DIR / basename).with_suffix(".txt")
        if not file_path.exists():
            logger.error(
                f"System prompt '{file_path}' not found in system prompts directory"
            )
            raise HTTPException(
                status_code=404,
                detail=f"System prompt '{file_path}' not found in system prompts directory",
            )
        content = file_path.read_text(encoding="utf-8")
        logger.info(f"Successfully loaded system prompt '{file_path}'")
        return {
            "basename": basename,
            "content": content,
        }
    except Exception as ex:
        logger.error(f"Error reading system prompt: {ex}")
        raise HTTPException(
            status_code=500, detail=f"Error reading system prompt: {ex}"
        )


# GET /
# Serves the main index page (index.html)
#
# Response:
# HTML content of the index page
@app.get("/", response_class=HTMLResponse)
def serve_index():
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


# GET /evals
# Lists available evaluation files
#
# Response:
# {
#     "files": ["string"]  # Array of evaluation file basenames
# }
@app.get("/evals")
def list_evals():
    logger.info(f"Listing evals in directory: {JOB_DIR}")
    try:
        if not JOB_DIR.exists():
            logger.error(f"Test directory not found: {JOB_DIR}")
            raise HTTPException(
                status_code=404,
                detail=f"Test directory not found: {JOB_DIR}",
            )
        basenames = [f.stem for f in JOB_DIR.iterdir() if f.suffix == ".yaml"]
        logger.debug(f"Found evaluations: {', '.join(basenames) or 'none'}")
        return {"files": basenames}
    except Exception as ex:
        logger.error(f"Error reading test config directory: {ex}")
        raise HTTPException(
            status_code=500, detail=f"Error reading test config directory: {ex}"
        )


# POST /eval
# Retrieves an evaluation file by basename
#
# Request Body:
# {
#     "basename": "string"  # Required: The basename of the eval file
# }
#
# Response:
# {
#     "basename": "string",
#     "filename": "string",
#     "content": "object"  # Parsed YAML content of the eval file
# }
@app.post("/eval")
async def get_eval(request: Request):
    try:
        data = await get_json_from_request(request)
        basename = data.get("basename")
        logger.info(f"Received request for eval with basename: {basename}")
        if not basename:
            logger.error("Missing 'basename' parameter in request")
            raise HTTPException(
                status_code=400, detail="basename parameter is required"
            )
        if not JOB_DIR.exists():
            logger.error(f"Test directory not found: {JOB_DIR}")
            raise HTTPException(
                status_code=404,
                detail=f"Test directory not found: {JOB_DIR}",
            )
        file_path = (JOB_DIR / basename).with_suffix(".yaml")
        if not file_path.exists():
            logger.error(f"File '{basename}.yaml' not found in test directory")
            raise HTTPException(
                status_code=404,
                detail=f"File '{basename}.yaml' not found in test directory",
            )
        logger.debug(f"Loaded YAML for {basename}")
        return {
            "basename": basename,
            "filename": f"{basename}.yaml",
            "content": load_yaml(file_path),
        }
    except Exception as ex:
        logger.error(f"Error reading or converting file '{basename}.yaml': {ex}")
        raise HTTPException(
            status_code=500,
            detail=f"Error reading or converting file '{basename}.yaml': {ex}",
        )


# POST /create-system-prompt
# Creates a new system prompt file
#
# Request Body:
# {
#     "basename": "string"  # Required: The basename for the new system prompt
#     "content": "string"   # Required: The content of the system prompt
# }
#
# Response:
# {
#     "basename": "string",
#     "filename": "string",
#     "content": "string"  # The created system prompt content
# }
@app.post("/create-system-prompt")
async def create_system_prompt(request: Request):
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


# POST /save-system-prompt
# Saves an existing system prompt file
#
# Request Body:
# {
#     "basename": "string"  # Required: The basename of the system prompt to save
#     "content": "string"   # Required: The new content of the system prompt
# }
#
# Response:
# {
#     "basename": "string",
#     "filename": "string",
#     "content": "string"  # The saved system prompt content
# }
@app.post("/save-system-prompt")
async def save_system_prompt(request: Request):
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


@app.post("/evaluate")
async def evaluate(request: Request):
    """
    Run evaluation for a given test configuration using JobRunner.

    Request Body:
    {
        "basename": "string"  # Required: The basename of the evaluation config
        "testConfig": Dict of TestConfig
    }

    Response:
    {
        "success": bool,
        "message": "string",
        "results": {
            "texts": ["string"],  # List of generated responses
            "evaluations": [      # List of evaluation results
                {
                    "name": "string",
                    "values": [float],
                    "average": float,
                    "standard_deviation": float
                }
            ]
        }
    }
    """
    try:
        data = await get_json_from_request(request)
        basename = data.get("basename")
        if not basename:
            raise HTTPException(status_code=400, detail="basename is required")
        config_path = (JOB_DIR / basename).with_suffix(".yaml")
        test_config = data.get("testConfig")
        if not test_config:
            raise HTTPException(status_code=400, detail="testConfig is required")
        save_yaml(test_config, config_path)
        job_runner = JobRunner(config_path)
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


if __name__ == "__main__":
    uvicorn.run(
        f"{Path(__file__).stem}:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_config=None,  # We handle our own logging
        access_log=False,  # Disable uvicorn access logs
    )
