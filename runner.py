import asyncio
import logging
import sys
from statistics import mean, stdev
from typing import List

from path import Path
from rich.logging import RichHandler

from chat_client import get_chat_client
from evaluator import EvaluationRunner
from schemas import RESULTS_DIR, RunConfig, RunResult
from util import save_yaml

logger = logging.getLogger(__name__)


class Runner:
    def __init__(self, file_path: str):
        self._config = RunConfig.read_from_yaml(file_path)
        self._chat_client = get_chat_client(
            self._config.service, model=self._config.model
        )
        self._cost_per_token = self._chat_client.get_token_cost()
        self._evaluation_runner = EvaluationRunner(self._chat_client, self._config)
        RESULTS_DIR.makedirs_p()

    async def run_evaluations(self) -> List[RunResult]:
        fields = self._config.evaluators + ["elapsed_seconds", "token_count", "cost"]
        eval_results_dict = {f: RunResult(name=f) for f in fields}

        response_texts = []
        for i in range(self._config.repeat):
            logger.info(f">>> Evaluate iteration {i + 1}/{self._config.repeat}")

            response = await self._chat_client.get_completion(
                messages=[
                    {"role": "system", "content": self._config.prompt},
                    {"role": "user", "content": self._config.input},
                ]
            )

            response_texts.append(response["text"])

            elapsed_seconds = response["metadata"]["usage"]["elapsed_seconds"]
            logger.debug(f"ElapsedSeconds: {elapsed_seconds}")

            token_count = response["metadata"]["usage"].get("total_tokens", 0)
            cost_value = (
                token_count * self._cost_per_token if token_count is not None else None
            )
            logger.debug(f"TokenCount: {token_count}")

            eval_results_dict["elapsed_seconds"].values.append(elapsed_seconds)
            eval_results_dict["token_count"].values.append(token_count)
            eval_results_dict["cost"].values.append(cost_value)

            results = await self._evaluation_runner.evaluate_response(response)
            for evaluator_name, value in results.items():
                eval_results_dict[evaluator_name].values.append(value["score"])

        # Take averages and standard deviation for every eval_result
        for eval_result in eval_results_dict.values():
            valid_values = [v for v in eval_result.values if v is not None]
            if valid_values:
                eval_result.average = mean(valid_values)
                eval_result.standard_deviation = (
                    stdev(valid_values) if len(valid_values) > 1 else 0.0
                )

        evaluations = [result.model_dump() for result in eval_results_dict.values()]
        return {"texts": response_texts, "evaluations": evaluations}

    async def save_results(self):
        eval_results = await self.run_evaluations()
        stem = Path(self._config.file_path).stem
        results_path = RESULTS_DIR / f"{stem}.yaml"
        save_yaml(eval_results, results_path)
        logger.info(f"Results saved to: {results_path}")


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
    if len(sys.argv) == 1:
        logger.info("Usage: python runner.py <config_file_path>")
        logger.info("No file path provided, run all in `./runs/*.yaml`")
        file_paths = Path("runs").glob("*.yaml")
    else:
        file_paths = [Path(sys.argv[1])]

    for file_path in file_paths:
        logger.info(f"Running job: {file_path}")
        asyncio.run(Runner(file_path).save_results())
