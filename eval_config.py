from typing import List, Optional

import yaml
from path import Path
from pydantic import BaseModel, Field


class EvalConfig(BaseModel):
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
    def read_from_yaml(file_path: str) -> "EvalConfig":
        data = yaml.safe_load(Path(file_path).read_text())
        data["file_path"] = file_path
        data["name"] = Path(file_path).stem
        return EvalConfig(**data)
