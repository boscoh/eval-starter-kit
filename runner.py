import asyncio
import logging
from statistics import mean, stdev

from path import Path

import schemas
from chat_client import get_chat_client
from evaluator import EvaluationRunner
from schemas import RunConfig, RunResult, set_evals_dir
from yaml_utils import save_yaml

logger = logging.getLogger(__name__)


class Runner:
    def __init__(self, file_path: str):
        self._config = RunConfig.read_from_yaml(file_path)
        self._chat_client = get_chat_client(
            self._config.service, model=self._config.model
        )
        self._cost_per_token = self._chat_client.get_token_cost()
        self._evaluation_runner = EvaluationRunner(self._chat_client, self._config)
        schemas.RESULTS_DIR.makedirs_p()

    async def run(self):
        try:
            await self._chat_client.connect()
            logger.info(f"Connected to chat client: {self._chat_client}")
        except Exception as e:
            logger.error(f"Error connecting to chat client: {e}")
            raise RuntimeError(f"Failed to connect to chat client: {e}") from e

        try:
            fields = self._config.evaluators + [
                "elapsed_seconds",
                "token_count",
                "cost",
            ]
            eval_results_dict = {f: RunResult(name=f) for f in fields}

            response_texts = []
            for i in range(self._config.repeat):
                logger.info(f">>> Evaluate iteration {i + 1}/{self._config.repeat}")

                response = await self._chat_client.get_completion(
                    messages=[
                        {"role": "system", "content": self._config.prompt},
                        {"role": "user", "content": self._config.input},
                    ],
                    temperature=self._config.temperature,
                )

                # Check if the response contains an error
                if "error" in response.get("metadata", {}):
                    error_msg = response["metadata"]["error"]
                    logger.error(f"Chat client error: {error_msg}")
                    raise RuntimeError(f"Chat client error: {error_msg}")

                response_texts.append(response["text"])

                elapsed_seconds = response["metadata"]["usage"]["elapsed_seconds"]
                logger.debug(f"ElapsedSeconds: {elapsed_seconds}")

                token_count = response["metadata"]["usage"].get("total_tokens", 0)
                cost_value = (
                    token_count * self._cost_per_token / 1000
                    if token_count is not None
                    else None
                )
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

            evaluations = [result.model_dump() for result in eval_results_dict.values()]

            eval_results = {"texts": response_texts, "evaluations": evaluations}
            results_path = (
                schemas.RESULTS_DIR / Path(self._config.file_path).name
            ).with_suffix(".yaml")
            save_yaml(eval_results, results_path)

            logger.info(f"Results saved to: {results_path}")
        finally:
            await self._chat_client.close()


async def run_all(file_paths):
    for run_config in file_paths:
        logger.info(f"Running job: {run_config}")
        await Runner(run_config).run()


if __name__ == "__main__":
    import argparse

    from setup_logger import setup_logging

    setup_logging()

    parser = argparse.ArgumentParser(description="Run LLM evaluations")
    parser.add_argument(
        "config_file",
        nargs="?",
        help="Path to config file (if not provided, runs all configs in evals/runs/)",
    )
    parser.add_argument(
        "--evals-dir",
        default="evals-consultant",
        help="Base directory for evals (default: evals-consultant)",
    )
    args = parser.parse_args()

    set_evals_dir(args.evals_dir)

    if args.config_file:
        file_paths = [Path(args.config_file)]
    else:
        logger.info(f"No file path provided, run all in `./{schemas.RUNS_DIR}/*.yaml`")
        file_paths = schemas.RUNS_DIR.glob("*.yaml")

    asyncio.run(run_all(file_paths))
