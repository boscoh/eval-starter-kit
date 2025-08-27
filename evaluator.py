import logging
import re
import textwrap
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

from schemas import JobConfig


def parse_score_text(score_text: str) -> float:
    """
    Parse a score text string into a float value clamped between 0.0 and 1.0.

    Args:
        score_text: The text containing the score to parse

    Returns:
        float: The parsed score clamped between 0.0 and 1.0
    """
    try:
        return max(0.0, min(1.0, float(score_text.strip())))
    except (ValueError, TypeError):
        numbers = re.findall(r"\b0?\.\d+\b|\b1(?:\.0+)?\b", str(score_text.strip()))
        if numbers:
            return max(0.0, min(1.0, float(numbers[0])))
    return 0.5

class EvaluationRunner:
    @staticmethod
    def allowed_evaluators() -> list:
        return ["coherence", "equivalence", "word_count"]

    def __init__(self, chat_client, job_config: JobConfig):
        self.chat_client = chat_client
        self.job_config = job_config
        self.coherence_evaluator = CoherenceEvaluator(chat_client)
        self.equivalence_evaluator = EquivalenceEvaluator(chat_client)
        self.word_count_evaluator = WordCountEvaluator()

    async def evaluate_response(self, response: Any) -> Dict[str, dict]:
        """
        Evaluate the response using all configured evaluators.

        Args:
            response: The response to evaluate

        Returns:
            Dict[str, dict]: A dictionary mapping evaluator names to their result dictionaries.
                           Each result dictionary contains:
                           - score (float): The evaluation score
                           - text (str): The full response text from the chat client (if applicable)
                           - elapsed_ms (int): Time taken for evaluation in milliseconds (if applicable)
                           - token_count (int): Number of tokens used in evaluation (if applicable)
        """
        results = {}

        for evaluator_name in self.job_config.evaluators:
            try:
                if evaluator_name.lower() == "coherence":
                    result = await self.coherence_evaluator.evaluate_coherence(
                        question=self.job_config.prompt or "", answer=response["text"]
                    )
                    results[evaluator_name] = result
                elif evaluator_name.lower() == "equivalence":
                    if not self.job_config.expected_answer:
                        logging.warning("No answer provided for equivalence evaluation")
                        results[evaluator_name] = {
                            "score": None,
                            "text": "",
                            "elapsed_ms": 0,
                            "token_count": 0,
                        }
                    else:
                        result = await self.equivalence_evaluator.evaluate_equivalence(
                            answer=self.job_config.expected_answer,
                            actual_answer=response["text"],
                        )
                        results[evaluator_name] = result
                elif evaluator_name.lower() == "word_count":
                    result = await self.word_count_evaluator.evaluate_word_count(
                        response["text"],
                        min_words=getattr(self.job_config, "min_words", None),
                        max_words=getattr(self.job_config, "max_words", None),
                        target_words=getattr(self.job_config, "target_words", None),
                    )
                    results[evaluator_name] = result
                else:
                    results[evaluator_name] = {
                    "score": 1.0,
                    "text": "",
                    "elapsed_ms": 0,
                    "token_count": 0,
                }
            except Exception as e:
                logging.error(f"Error in {evaluator_name} evaluation: {e}", exc_info=True)
                results[evaluator_name] = {
                    "score": 0.5,
                    "text": str(e),
                    "elapsed_ms": 0,
                    "token_count": 0,
                }

        return results


class CoherenceEvaluator:
    def __init__(self, chat_client):
        self.chat_client = chat_client

    async def evaluate_coherence(self, question: str, answer: str) -> dict:
        """
        Evaluate coherence using a simple prompt-based approach with any ChatClient
        Since ragas requires specific setup, we'll use a direct coherence evaluation

        Returns:
            dict: A dictionary containing:
                - score (float): The coherence score between 0.0 and 1.0
                - text (str): The full response text from the chat client
                - elapsed_ms (int): Time taken for the evaluation in milliseconds
                - token_count (int): Number of tokens used in the evaluation
        """
        result = {
            "score": 0.5,
            "text": "",
            "elapsed_ms": 0,
            "token_count": 0,
        }

        try:
            coherence_prompt = f"""
            Evaluate the coherence of the following answer to the given question on a scale of 0.0 to 1.0.
            
            Coherence means:
            - The answer is logically structured
            - Ideas flow naturally from one to another
            - The response is internally consistent
            - The language is clear and well-organized
            
            Question: {question}
            Answer: {answer}
            
            Please respond with only a number between 0.0 and 1.0 representing the coherence score.
            """

            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful evaluation assistant.",
                },
                {"role": "user", "content": textwrap.dedent(coherence_prompt.strip())},
            ]

            response = await self.chat_client.get_completion(messages)

            # Update result with response details
            result.update(
                {
                    "text": response.get("text", ""),
                    "elapsed_ms": response.get("elapsed_ms", 0),
                    "token_count": response.get("token_count", 0),
                }
            )

            result["score"] = parse_score_text(response.get("text", ""))

        except Exception as e:
            logging.error("Error in coherence evaluation", exc_info=True)

        return result


class EquivalenceEvaluator:
    def __init__(self, chat_client):
        self.chat_client = chat_client

    async def evaluate_equivalence(self, answer: str, actual_answer: str) -> dict:
        """
        Evaluate how well the actual answer matches the expected answer.

        Returns:
            dict: A dictionary containing:
                - score (float): A score between 0.0 (completely different) and 1.0 (perfect match)
                - text (str): The full response text from the chat client (empty if not using chat client)
                - elapsed_ms (int): Time taken for the evaluation in milliseconds
                - token_count (int): Number of tokens used in the evaluation
        """
        result = {
            "score": 0.5,
            "text": "",
            "elapsed_ms": 0,
            "token_count": 0,
        }

        if not answer.strip() or not actual_answer.strip():
            result["score"] = 0.0
            return result

        if answer.strip().lower() == actual_answer.strip().lower():
            result["score"] = 1.0
            return result

        try:
            prompt = f"""
            Compare the following two answers and determine how semantically equivalent they are.
            Consider the meaning and key information, not just exact wording.
            
            Expected Answer: {answer}
            
            Actual Answer: {actual_answer}
            
            Rate the semantic equivalence on a scale from 0.0 to 1.0, where:
            - 1.0 means the answers are completely equivalent in meaning
            - 0.5 means the answers are somewhat related but differ in important ways
            - 0.0 means the answers are completely different or contradictory
            
            Respond with only a number between 0.0 and 1.0, nothing else.
            """

            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful evaluation assistant.",
                },
                {"role": "user", "content": textwrap.dedent(prompt.strip())},
            ]

            response = await self.chat_client.get_completion(messages)

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

        return result


class WordCountEvaluator:
    """
    Evaluates if the response meets word count requirements.
    Can check against minimum words, maximum words, or target word count.
    """

    async def evaluate_word_count(
        self,
        text: str,
        min_words: Optional[int] = None,
        max_words: Optional[int] = None,
        target_words: Optional[int] = None,
    ) -> dict:
        """
        Evaluate if the text meets word count requirements.

        Args:
            text: The text to evaluate
            min_words: Minimum required words (inclusive)
            max_words: Maximum allowed words (inclusive)
            target_words: Target number of words (scores higher the closer to target)

        Returns:
            dict: A dictionary containing:
                - score (float): Score between 0.0 and 1.0 indicating how well the word count matches requirements
                - text (str): Empty string (no chat client used)
                - elapsed_ms (int): 0 (no chat client used)
                - token_count (int): 0 (no chat client used)
        """
        result = {
            "score": 1.0,
            "text": "",
            "elapsed_ms": 0,
            "token_count": 0,
        }

        try:
            word_count = len(text.split())

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
