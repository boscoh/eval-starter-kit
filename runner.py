import asyncio
import logging
from statistics import mean, stdev
from typing import List

from path import Path

from chat_client import get_chat_client
from evaluator import EvaluationRunner
from schemas import JobConfig, JobResult
from util import save_yaml

logger = logging.getLogger(__name__)


class JobRunner:
    def __init__(self, file_path: str):
        self._config = JobConfig.read_from_yaml(file_path)
        self._config.file_path = file_path
        self._config.name = Path(file_path).stem

        self._chat_client = get_chat_client("ollama", model="llama3.2")

        self._cost_per_token = self._chat_client.get_token_cost()

        self._evaluation_runner = EvaluationRunner(self._chat_client, self._config)

        self.results_dir = Path("./results")
        self.results_dir.makedirs_p()

    async def run_evaluations(self) -> List[JobResult]:
        fields = self._config.evaluators + ["elapsed_seconds", "token_count", "cost"]
        eval_results_dict = {name: JobResult(name=name) for name in fields}

        response_texts = []
        for i in range(self._config.repeat):
            logger.info(f"Iteration #{i}")

            response = await self._chat_client.invoke(
                system_prompt_key=self._config.system_prompt_ref,
                prompt=self._config.prompt,
            )

            response_texts.append(response["text"])

            elapsed_seconds = response["metadata"]["usage"]["elapsed_seconds"]
            token_count = response["metadata"]["usage"].get("total_tokens", 0)
            cost_value = (
                token_count * self._cost_per_token if token_count is not None else None
            )

            logger.debug(f"ElapsedSeconds: {elapsed_seconds}")
            logger.debug(f"TokenCount: {token_count}")

            eval_results_dict["elapsed_seconds"].values.append(elapsed_seconds)
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

        # Convert JobResult objects to dictionaries using Pydantic's model_dump()
        evaluations = [result.model_dump() for result in eval_results_dict.values()]
        return {"texts": response_texts, "evaluations": evaluations}

    async def save_results(self):
        eval_results = await self.run_evaluations()
        stem = Path(self._config.file_path).stem
        results_path = self.results_dir / f"{stem}.yaml"
        save_yaml(eval_results, results_path)
        logger.info(f"Results saved to: {results_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        logger.info("Usage: python runner.py <config_file_path>")
        logger.info("No file path provided, run all in `./queries/*.yaml`")
        file_paths = Path("queries").glob("*.yaml")
    else:
        file_paths = [Path(sys.argv[1])]

    for file_path in file_paths:
        logger.info(f"Running job: {file_path}")
        asyncio.run(JobRunner(file_path).save_results())
