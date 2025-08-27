from typing import List, Optional

from path import Path
from pydantic import BaseModel, Field

from util import load_yaml


class JobConfig(BaseModel):
    name: Optional[str] = None
    file_path: Optional[str] = None
    system_prompt: Optional[str] = None
    system_prompt_ref: Optional[str] = None
    service: str = "openai"
    model: str = "gpt-4o-mini"
    prompt: Optional[str] = None
    expected_answer: Optional[str] = None
    repeat: int = 1
    evaluators: List[str] = Field(default_factory=lambda: ["CoherenceEvaluator"])

    @staticmethod
    def read_from_yaml(file_path: str) -> "JobConfig":
        data = load_yaml(file_path)
        data["file_path"] = file_path
        data["name"] = Path(file_path).stem
        return JobConfig(**data)


class JobResult(BaseModel):
    name: str
    values: List[Optional[float]] = Field(default_factory=list)
    average: Optional[float] = None
    standard_deviation: Optional[float] = None
