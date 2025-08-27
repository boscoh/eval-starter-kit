import asyncio
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import ollama
from openai import OpenAI

# Configure logging
logger = logging.getLogger(__name__)


def get_chat_client(client_type: str, **kwargs) -> "IChatClient":
    """
    Factory function to create a chat client instance based on the specified type.

    This is the main entry point for creating chat clients in the application.
    It supports creating instances of different chat model providers while
    maintaining a consistent interface through the ChatClient abstract base class.

    Args:
        client_type: Type of chat client to create. Supported types:
            - "openai": Creates an OpenAIChatClient instance
            - "ollama": Creates an OllamaChatClient instance
        **kwargs: Additional keyword arguments specific to the chat client type:
            - For OpenAI: model (str), max_tokens (int)
            - For Ollama: model (str)

    Returns:
        ChatClient: An instance of the requested chat client type

    Raises:
        ValueError: If an unsupported client_type is provided

    Example:
        ```python
        # Create an OpenAI chat client
        openai_client = get_chat_client("openai", model="gpt-4", max_tokens=1000)

        # Create an Ollama chat client
        ollama_client = get_chat_client("ollama", model="llama3.2")
        ```
    """
    client_type = client_type.lower()
    if client_type == "openai":
        return OpenAIChatClient(**kwargs)
    if client_type == "ollama":
        return OllamaChatClient(**kwargs)
    raise ValueError(f"Unknown chat client type: {client_type}")


def parse_response_as_json_list(response):
    """Parse JSON from text response, extracting from markdown blocks if needed.
    
    Returns transactions list if found in dict, otherwise the parsed data.
    """
    import re
    
    # Extract text from response
    if isinstance(response, dict):
        response_text = response.get("text", "")
    elif isinstance(response, str):
        response_text = response
    else:
        return None

    if not response_text:
        return None

    def try_parse(text):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    def extract_transactions(data):
        """Extract transactions list from dict if present"""
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "transactions" in data:
            transactions = data["transactions"]
            if isinstance(transactions, list):
                return transactions
        return data

    # Try direct parsing
    parsed = try_parse(response_text)
    if parsed is not None:
        return extract_transactions(parsed)

    # Try markdown code blocks
    patterns = [
        r"```(?:json|python)?\s*([\s\S]*?)\s*```",  # Any code block
        r"```(?:json)?\s*({[\s\S]*})\s*```",        # JSON object in block
        r"\{[\s\S]*\}",                             # Any JSON object
        r"({[\s\S]*})"                              # Last resort: any braces
    ]

    for pattern in patterns:
        matches = re.findall(pattern, response_text, re.IGNORECASE)
        for match in matches:
            parsed = try_parse(match)
            if parsed is not None:
                return extract_transactions(parsed)

    return None


class IChatClient(ABC):
    """
    Abstract base class for chat client implementations.

    This interface defines a standardized way to interact with various chat
    model providers (like OpenAI, Ollama, etc.) using a consistent API.
    All chat clients should implement this interface to ensure compatibility
    across different model providers.
    """

    @abstractmethod
    async def get_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Get a chat completion from the model.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            tools: Optional list of tool/function definitions
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)

        Returns:
            Dict with 'text' and 'metadata' keys
        """
        pass

    @abstractmethod
    def get_token_cost(self) -> float:
        """
        Get the cost per token for the specified model.

        Returns:
            Cost per token in USD
        """
        pass

    async def invoke(
        self,
        prompt: str,
        system_prompt_key: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Invoke the chat model with a prompt and a system prompt loaded from a file.

        Args:
            prompt: The user's input prompt
            system_prompt_key: The key to load the system prompt from system-prompts/<key>.txt
            max_tokens: Maximum number of tokens to generate (default: None for model's default)
            temperature: Sampling temperature (0.0 for deterministic output, higher for more randomness)

        Returns:
            Dict with 'text' and 'metadata' keys, same as get_completion
        """
        # Load system prompt from file
        system_prompt_path = Path("system-prompts") / f"{system_prompt_key}.txt"
        try:
            system_prompt = system_prompt_path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            raise ValueError(f"System prompt file not found: {system_prompt_path}")

        # Prepare messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        # Get completion with parameters
        return await self.get_completion(
            messages=messages, max_tokens=max_tokens, temperature=temperature
        )


class OllamaChatClient(IChatClient):
    def __init__(self, model: str = "llama3.2"):
        self.model = model
        self.client = ollama.AsyncClient()

    async def get_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Get a chat completion from the Ollama model.
        """
        try:
            options = {
                "temperature": temperature,
            }
            if max_tokens is not None:
                options["num_predict"] = max_tokens

            response = await self.client.chat(
                model=self.model, messages=messages, options=options
            )

            response_text = response["message"]["content"]
            token_count = sum(
                len(m.get("content", "").split()) for m in messages
            ) + len(response_text.split())

            return {
                "text": response_text,
                "metadata": {
                    "Usage": {
                        "TotalTokenCount": token_count,
                        "elapsed_ms": response.get("elapsed_ms", 0),
                    }
                },
            }
        except Exception as e:
            logger.error(f"Error calling Ollama: {e}")
            return {
                "text": f"Error: {str(e)}",
                "metadata": {"Usage": {"TotalTokenCount": 0, "elapsed_ms": 0}},
            }

    def get_token_cost(self) -> float:
        """
        Get the cost per token for Ollama models.
        Since Ollama is typically run locally, the cost is 0.
        """
        return 0.0


class OpenAIChatClient(IChatClient):
    def __init__(self, model: str = "gpt-4o", max_tokens: int = 1000):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set.")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    async def get_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Get a chat completion from the OpenAI model.
        """

        def sync_call():
            return self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                max_tokens=max_tokens or self.max_tokens,
                temperature=temperature,
            )

        try:
            loop = asyncio.get_event_loop()
            completion = await loop.run_in_executor(None, sync_call)

            text = completion.choices[0].message.content if completion.choices else ""
            token_count = (
                completion.usage.total_tokens
                if hasattr(completion, "usage") and completion.usage
                else 0
            )

            return {
                "text": text,
                "metadata": {"Usage": {"TotalTokenCount": token_count}},
            }
        except Exception as e:
            logger.error(f"Error calling OpenAI: {e}")
            return {
                "text": f"Error: {str(e)}",
                "metadata": {"Usage": {"TotalTokenCount": 0}},
            }

    def get_token_cost(self) -> float:
        """
        Get the cost per token for OpenAI models.
        """
        pricing = {
            "gpt-4": 0.03,  # $0.03 per 1K tokens
            "gpt-4o": 0.03,  # Assuming same pricing as gpt-4
            "gpt-3.5-turbo": 0.002,  # $0.002 per 1K tokens
        }
        return pricing.get(self.model.lower(), 0.0)
