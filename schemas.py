import copy
import logging
from typing import List, Optional

from path import Path
from pydantic import BaseModel, Field

from util import load_yaml, save_yaml, write_text

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path("system-prompts")
QUERIES_DIR = Path("queries")
RESULTS_DIR = Path("results")
RUNS_DIR = Path("runs")

RUNS_DIR.makedirs_p()
PROMPTS_DIR.makedirs_p()
QUERIES_DIR.makedirs_p()
RESULTS_DIR.makedirs_p()


class RunConfig(BaseModel):
    file_path: Optional[str] = None
    query_ref: Optional[str] = None
    system_prompt_ref: Optional[str] = None
    system_prompt: str = ""
    prompt: str = ""
    expected_answer: str = ""
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
        system_prompt_path = PROMPTS_DIR / f"{result.system_prompt_ref}.txt"
        if system_prompt_path.exists():
            result.system_prompt = system_prompt_path.read_text()
            logger.info(f"Loaded system prompt from '{system_prompt_path}'")
        else:
            logger.warning(f"System prompt file not found: {system_prompt_path}")

        # Get prompt/expected_answer from QUERIES_DIR
        query_path = QUERIES_DIR / f"{result.query_ref}.yaml"
        try:
            query = load_yaml(query_path)
        except FileNotFoundError:
            raise ValueError(f"Query file not found: {query_path}")
        if "prompt" not in query:
            raise ValueError(f"Query file must contain a 'prompt' key: {query_path}")
        if "expected_answer" not in query:
            raise ValueError(
                f"Query file must contain a 'expected_answer' key: {query_path}"
            )
        logger.info(f"Loaded query from '{query_path}'")
        result.prompt = query["prompt"]
        result.expected_answer = query["expected_answer"]

        return result

    def save(self, file_path: str):
        save_yaml(self.model_dump(), file_path)

        save_config = copy.deepcopy(self.model_dump())
        del save_config["prompt"]
        del save_config["expected_answer"]
        del save_config["system_prompt"]
        save_yaml(save_config, file_path)
        logger.info(f"Saved test config to '{file_path}'")

        # query_ref = self.query_ref
        # query_path = (QUERIES_DIR / f"{query_ref}").with_suffix(".yaml")
        # if query_ref:
        #     data = {"prompt": self.prompt, "expected_answer": self.expected_answer}
        # else:
        #     data = {"prompt": "", "expected_answer": ""}
        # save_yaml(data, query_path)
        # logger.info(f"Saved query to '{query_path}'")

        # prompt_ref = self.system_prompt_ref
        # if prompt_ref:
        #     prompt_path = (PROMPTS_DIR / prompt_ref).with_suffix(".txt")
        #     write_text(self.system_prompt, prompt_path)
        #     logger.info(f"Saved prompt to '{prompt_path}'")


class RunResult(BaseModel):
    name: str
    values: List[Optional[float]] = Field(default_factory=list)
    average: Optional[float] = None
    standard_deviation: Optional[float] = None
