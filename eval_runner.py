import asyncio
from statistics import mean, stdev
from typing import List, Optional

import yaml
from path import Path
from pydantic import BaseModel, Field

from chat_client import get_chat_client
from eval_config import EvalConfig
from evaluator import EvaluationRunner


class EvalResult(BaseModel):
    name: str
    values: List[Optional[float]] = Field(default_factory=list)
    average: Optional[float] = None
    standard_deviation: Optional[float] = None


class TestRunner:
    def __init__(self, file_path: str):
        self._eval_config = EvalConfig.read_from_yaml(file_path)
        self._eval_config.file_path = file_path
        self._eval_config_dir = Path(file_path).parent
        self._eval_config.name = Path(file_path).stem
        self._chat_client = get_chat_client("ollama", model="llama3.2")
        self._cost_per_token = self._chat_client.get_token_cost()
        self._evaluation_runner = EvaluationRunner(self._chat_client, self._eval_config)

    def read_system_prompt(self) -> str:
        if not self._eval_config.system_prompt_ref:
            return ""
        parent = Path(self._eval_config.file_path or "").parent
        filename = (parent / "system-prompts" / self._eval_config.system_prompt_ref).with_suffix(".txt")
        if filename.exists():
            system_prompt = filename.read_text()
            return f"System: {system_prompt}\n"
        return ""

    async def run_evaluations(self) -> List[EvalResult]:
        fields = self._eval_config.evaluators + ["elapsed_ms", "token_count", "cost"]
        eval_results_dict = {name: EvalResult(name=name) for name in fields}

        prompt = self.read_system_prompt() + (self._eval_config.prompt or "")

        response_texts = []
        for i in range(self._eval_config.repeat):
            print(f"Iteration #{i}")

            start = asyncio.get_event_loop().time()


            response = await self._chat_client.get_completion(
                [{"role": "user", "content": prompt}]
            )

            response_texts.append(response["text"])

            elapsed = (asyncio.get_event_loop().time() - start) * 1000
            token_count = response["metadata"]["Usage"]["TotalTokenCount"]
            cost_value = (
                token_count * self._cost_per_token if token_count is not None else None
            )

            print(f"ElapsedMs: {elapsed}")
            print(f"TokenCount: {token_count}")

            eval_results_dict["elapsed_ms"].values.append(elapsed)
            eval_results_dict["token_count"].values.append(token_count)
            eval_results_dict["cost"].values.append(cost_value)

            results = await self._evaluation_runner.evaluate_response(response)
            for evaluator_name, value in results.items():
                eval_results_dict[evaluator_name].values.append(value["score"])

        for eval_result in eval_results_dict.values():
            valid_values = [v for v in eval_result.values if v is not None]
            if valid_values:
                eval_result.average = mean(valid_values)
                eval_result.standard_deviation = (
                    stdev(valid_values) if len(valid_values) > 1 else 0.0
                )

        # Convert EvalResult objects to dictionaries using Pydantic's model_dump()
        evaluations = [result.model_dump() for result in eval_results_dict.values()]
        return {"texts": response_texts, "evaluations": evaluations}

    async def save_evaluation_results(self):
        eval_results = await self.run_evaluations()

        source_file_path = Path(self._eval_config.file_path)
        stem = source_file_path.stem
        top_dir = source_file_path.parent or Path(".")
        summary_directory = top_dir / "summary"
        summary_directory.makedirs_p()

        summary_path = summary_directory / f"{stem}-summary.yaml"
        summary_path.write_text(
            yaml.dump(
                eval_results, allow_unicode=True
            )
        )
        print(f"Summary results saved to: {summary_path}")


if __name__ == "__main__":
    import sys

    file_path = sys.argv[1] if len(sys.argv) > 1 else "./sample-evals/engineer.yaml"
    if file_path:
        asyncio.run(TestRunner(file_path).save_evaluation_results())
    else:
        print("Usage: python test_runner.py <config_file_path>")
