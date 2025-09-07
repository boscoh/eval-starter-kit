import copy
import logging
from typing import List, Literal, Optional

from path import Path
from pydantic import BaseModel, Field

from util import load_yaml, save_yaml

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path("prompts")
QUERIES_DIR = Path("queries")
RESULTS_DIR = Path("results")
RUNS_DIR = Path("runs")

RUNS_DIR.makedirs_p()
PROMPTS_DIR.makedirs_p()
QUERIES_DIR.makedirs_p()
RESULTS_DIR.makedirs_p()


TableType = Literal["result", "run", "prompt", "query"]


class RunConfig(BaseModel):
    file_path: Optional[str] = None
    query_ref: Optional[str] = None
    prompt_ref: Optional[str] = None
    prompt: str = ""
    input: str = ""
    output: str = ""
    service: str = "ollama"
    model: str = "llama3.2"
    repeat: int = 1
    evaluators: List[str] = Field(default_factory=lambda: ["CoherenceEvaluator"])

    @staticmethod
    def read_from_yaml(file_path: str) -> "RunConfig":
        data = load_yaml(file_path)
        result = RunConfig(**data)
        result.file_path = file_path
        logger.info(f"Loaded run config from '{file_path}'")

        # Get system prompt from PROMPTS_DIR
        system_prompt_path = PROMPTS_DIR / f"{result.prompt_ref}.txt"
        if system_prompt_path.exists():
            result.prompt = system_prompt_path.read_text()
            logger.info(f"Loaded system prompt from '{system_prompt_path}'")
        else:
            logger.warning(f"System prompt file not found: {system_prompt_path}")

        # Get input/output from QUERIES_DIR
        query_path = QUERIES_DIR / f"{result.query_ref}.yaml"
        try:
            query = load_yaml(query_path)
        except FileNotFoundError:
            raise ValueError(f"Query file not found: {query_path}")
        if "input" not in query:
            raise ValueError(f"Query file must contain a 'input' key: {query_path}")
        if "output" not in query:
            raise ValueError(f"Query file must contain a 'output' key: {query_path}")
        logger.info(f"Loaded query from '{query_path}'")
        result.input = query["input"]
        result.output = query["output"]
        logger.info(f"Loaded run config from '{query_path}'")

        return result

    def save(self, file_path: str):
        save_yaml(self.model_dump(), file_path)

        save_config = copy.deepcopy(self.model_dump())
        del save_config["input"]
        del save_config["output"]
        del save_config["prompt"]
        save_yaml(save_config, file_path)
        logger.info(f"Saved test config to '{file_path}'")


class RunResult(BaseModel):
    name: str
    values: List[Optional[float]] = Field(default_factory=list)
    average: Optional[float] = None
    standard_deviation: Optional[float] = None
