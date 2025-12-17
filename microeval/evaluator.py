import logging
import re
import textwrap
from typing import Any, Dict

from microeval.schemas import RunConfig

logger = logging.getLogger(__name__)


def parse_score_text(score_text: str) -> float:
    """Parse a score text string into a float value clamped between 0.0 and 1.0."""
    try:
        return max(0.0, min(1.0, float(score_text.strip())))
    except (ValueError, TypeError):
        numbers = re.findall(r"\b0?\.\d+\b|\b1(?:\.0+)?\b", str(score_text.strip()))
        if numbers:
            return max(0.0, min(1.0, float(numbers[0])))
    return 0.5


class EvaluationRunner:
    def __init__(self, chat_client, run_config: RunConfig):
        self.chat_client = chat_client
        self.run_config = run_config
        self.evaluators = {
            "coherence": CoherenceEvaluator(chat_client, run_config),
            "equivalence": EquivalenceEvaluator(chat_client, run_config),
            "word_count": WordCountEvaluator(run_config),
        }

    @staticmethod
    def evaluators() -> list:
        return ["coherence", "equivalence", "word_count"]

    async def evaluate_response(self, response: Any) -> Dict[str, dict]:
        """
        Evaluate the response using all configured evaluators.

        Args:
            response: The response to evaluate

        Returns:
            Dict[str, dict]: A dictionary mapping evaluator names to their result dictionaries.
        """
        results = {}
        response_text = response.get("text", "")

        for evaluator_name in self.run_config.evaluators:
            evaluator_name = evaluator_name.lower()
            try:
                if evaluator_name in self.evaluators:
                    evaluator = self.evaluators[evaluator_name]
                    result = await evaluator.evaluate(response_text)
                    results[evaluator_name] = result
                else:
                    results[evaluator_name] = {
                        "score": 1.0,
                        "text": f"Unknown evaluator: {evaluator_name}",
                        "elapsed_ms": 0,
                        "token_count": 0,
                    }
            except Exception as e:
                logging.error(
                    f"Error in {evaluator_name} evaluation: {e}", exc_info=True
                )
                results[evaluator_name] = {
                    "score": 0.5,
                    "text": str(e),
                    "elapsed_ms": 0,
                    "token_count": 0,
                }

        return results


class CoherenceEvaluator:
    def __init__(self, chat_client, run_config: RunConfig):
        self.chat_client = chat_client
        self.run_config = run_config

    async def evaluate(self, response_text: str) -> Dict[str, Any]:
        """
        Evaluate the coherence of the response text against the input question.

        Args:
            response_text: The text to evaluate for coherence

        Returns:
            dict: Evaluation result with:
                - score (float): Coherence score between 0.0 and 1.0
                - text (str): Detailed evaluation text
                - elapsed_ms (int): Evaluation time in milliseconds
                - token_count (int): Number of tokens used in evaluation
        """
        result = {
            "score": 0.5,
            "text": "",
            "elapsed_ms": 0,
            "token_count": 0,
        }

        if not response_text.strip():
            result["score"] = 0.0
            result["text"] = "Empty response text provided"
            return result

        question = self.run_config.input or ""

        try:
            coherence_prompt = textwrap.dedent(f"""
                Evaluate the coherence of the following answer to the given question on a scale of 0.0 to 1.0.
                
                Coherence means:
                - The answer is logically structured
                - Ideas flow naturally from one to another
                - The response is internally consistent
                - The language is clear and well-organized
                
                Question: {question}
                Answer: {response_text}
                
                Please respond with only a number between 0.0 and 1.0 representing the coherence score.
                """).strip()

            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful evaluation assistant.",
                },
                {"role": "user", "content": textwrap.dedent(coherence_prompt.strip())},
            ]

            response = await self.chat_client.get_completion(messages)

            # Check if the response contains an error
            if "error" in response.get("metadata", {}):
                error_msg = response["metadata"]["error"]
                logger.error(f"Chat client error in coherence evaluation: {error_msg}")
                raise RuntimeError(f"Chat client error: {error_msg}")

            result.update(
                {
                    "text": response.get("text", ""),
                    "elapsed_ms": response.get("elapsed_ms", 0),
                    "token_count": response.get("token_count", 0),
                }
            )

            result["score"] = parse_score_text(response.get("text", ""))

        except Exception:
            logging.error("Error in coherence evaluation", exc_info=True)
            raise

        return result


class EquivalenceEvaluator:
    def __init__(self, chat_client, run_config: RunConfig):
        self.chat_client = chat_client
        self.run_config = run_config

    async def evaluate(self, response_text: str) -> Dict[str, Any]:
        """
        Evaluate how well the response matches the expected answer.

        Args:
            response_text: The text to evaluate against the expected answer

        Returns:
            dict: Evaluation result with:
                - score (float): Similarity score between 0.0 and 1.0
                - text (str): Detailed evaluation text
                - elapsed_ms (int): Evaluation time in milliseconds
                - token_count (int): Number of tokens used in evaluation
        """
        result = {
            "score": 0.5,
            "text": "",
            "elapsed_ms": 0,
            "token_count": 0,
        }

        if not response_text.strip():
            result["score"] = 0.0
            result["text"] = "Empty response text provided"
            return result

        if not self.run_config.output:
            result["score"] = 0.0
            result["text"] = "No expected answer provided for comparison"
            return result

        answer = self.run_config.output

        if not answer.strip() or not response_text.strip():
            result["score"] = 0.0
            result["text"] = "Empty answer or response text"
            return result

        if answer.strip().lower() == response_text.strip().lower():
            result["score"] = 1.0
            result["text"] = "Response exactly matches expected answer"
            return result

        try:
            prompt = textwrap.dedent(f"""
                Compare the following two answers and determine how semantically equivalent they are.
                Consider the meaning and key information, not just exact wording.
                
                Expected Answer: {answer}
                
                Actual Answer: {response_text}
                
                Rate the semantic equivalence on a scale from 0.0 to 1.0, where:
                - 1.0 means the answers are completely equivalent in meaning
                - 0.5 means the answers are somewhat related but differ in important ways
                - 0.0 means the answers are completely different or contradictory
                
                Respond with only a number between 0.0 and 1.0, nothing else.
                """).strip()

            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful evaluation assistant.",
                },
                {"role": "user", "content": prompt},
            ]

            response = await self.chat_client.get_completion(messages)

            # Check if the response contains an error
            if "error" in response.get("metadata", {}):
                error_msg = response["metadata"]["error"]
                logger.error(
                    f"Chat client error in equivalence evaluation: {error_msg}"
                )
                raise RuntimeError(f"Chat client error: {error_msg}")

            result.update(
                {
                    "text": response.get("text", ""),
                    "elapsed_ms": response.get("elapsed_ms", 0),
                    "token_count": response.get("token_count", 0),
                }
            )

            result["score"] = parse_score_text(response.get("text", ""))

        except Exception as e:
            logger.error(f"Error in semantic equivalence evaluation: {e}")
            result["text"] = str(e)
            raise

        return result


class WordCountEvaluator:
    """Evaluates if the response meets word count requirements.
    Can check against minimum words, maximum words, or target word count.
    """

    def __init__(self, run_config: RunConfig):
        self.run_config = run_config

    async def evaluate(self, response_text: str) -> Dict[str, Any]:
        """
        Evaluate if the response meets word count requirements.

        Args:
            response_text: The text to evaluate for word count

        Returns:
            dict: Evaluation result with:
                - score (float): Score between 0.0 and 1.0
                - text (str): Evaluation details
                - elapsed_ms (int): Always 0 (synchronous operation)
                - token_count (int): Always 0 (no tokens used)
        """
        result = {
            "score": 1.0,
            "text": "",
            "elapsed_ms": 0,
            "token_count": 0,
        }

        if not response_text.strip():
            result["score"] = 0.0
            result["text"] = "Empty response text provided"
            return result

        min_words = getattr(self.run_config, "min_words", None)
        max_words = getattr(self.run_config, "max_words", None)
        target_words = getattr(self.run_config, "target_words", None)

        try:
            word_count = len(response_text.split())

            if target_words is not None:
                if word_count == 0:
                    result["score"] = 0.0
                    return result
                distance = abs(word_count - target_words)
                if distance >= target_words:
                    result["score"] = 0.5 * (
                        1 - (distance - target_words) / (target_words + 1)
                    )
                else:
                    result["score"] = 1.0 - (0.5 * (distance / target_words))
                return result

            if min_words is not None and word_count < min_words:
                result["score"] = 0.5 + (0.5 * min(1.0, word_count / max(1, min_words)))
                return result

            if max_words is not None and word_count > max_words:
                excess = word_count - max_words
                result["score"] = max(
                    0.5, 1.0 - (0.5 * min(1.0, excess / max(1, max_words)))
                )
                return result

            return result

        except Exception as e:
            logger.error(f"Error in word count evaluation: {e}")
            result["score"] = 0.5
            return result
