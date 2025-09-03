import asyncio
import json
import logging
import os
import subprocess
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import ollama
import openai
from dotenv import load_dotenv

load_dotenv()

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
            - For OpenAI: model (str)
            - For Ollama: model (str)

    Returns:
        ChatClient: An instance of the requested chat client type

    Raises:
        ValueError: If an unsupported client_type is provided

    Example:
        ```python
        # Create an OpenAI chat client
        openai_client = get_chat_client("openai", model="gpt-4")

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
        r"```(?:json)?\s*({[\s\S]*})\s*```",  # JSON object in block
        r"\{[\s\S]*\}",  # Any JSON object
        r"({[\s\S]*})",  # Last resort: any braces
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
        """Get a chat completion from the model.

        Args:
            messages: List of message dictionaries, where each dict has structure:
                {
                    'role': 'user' | 'assistant' | 'system',  # Message role
                    'content': str,  # The message content
                    'name': str  # Optional: Name of the message sender
                }
            tools: Optional list of tool definitions, where each dict has structure:
                {
                    'type': str,  # Type of tool (e.g., 'function')
                    'function': {
                        'name': str,  # Function name
                        'description': str,  # Optional: Function description
                        'parameters': dict  # JSON Schema for parameters
                    }
                }
            max_tokens: Maximum number of tokens to generate (None for model default)
            temperature: Sampling temperature (0.0 to 1.0)

        Returns:
            {
                'text': str,  # Generated response content
                'metadata': {
                    'usage': {
                        'prompt_tokens': int,  # Tokens in prompt
                        'completion_tokens': int,  # Tokens in completion
                        'total_tokens': int,  # Total tokens used
                        'elapsed_seconds': float  # Time taken for completion in seconds
                    },
                    'model': str,  # Model used
                    'finish_reason': str  # Reason generation stopped
                }
            }
        """
        pass

    @abstractmethod
    def get_token_cost(self) -> float:
        """Get the cost per 1K tokens for the model.

        Returns:
            float: Cost in USD per 1K tokens
        """
        pass


class OllamaChatClient(IChatClient):
    def __init__(self, model: str = "llama3.2"):
        """Initialize Ollama chat client.

        Args:
            model: Name of the Ollama model to use (default: "llama3.2")

        Raises:
            RuntimeError: If Ollama is not running or the model is not available
        """
        self.model = model
        self.client = None

    def load_client(self):
        if self.client:
            return

        # Check if Ollama is running
        try:
            subprocess.run(["ollama", "--version"], capture_output=True, check=True)
        except (subprocess.SubprocessError, FileNotFoundError):
            raise RuntimeError(
                "Ollama is not running or not installed. "
                "Please start the Ollama service and try again."
            )

        # Initialize the client and check if model is available
        self.client = ollama.AsyncClient()
        try:
            # This will raise an exception if the model is not found
            ollama.show(self.model)
        except Exception as e:
            raise RuntimeError(
                f"Model '{self.model}' is not available. "
                f"Please ensure the model is pulled and available. Error: {str(e)}"
            )

    async def get_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """Get a chat completion from the Ollama model.

        Args:
            messages: List of message dicts with format:
                [{"role": "user"|"assistant"|"system", "content": str}, ...]
            tools: Not currently supported in Ollama implementation
            max_tokens: Optional maximum number of tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)

        Returns:
            Dict with structure:
            {
                'text': str,  # Generated response content
                'metadata': {
                    'usage': {
                        'prompt_tokens': int,  # Tokens in prompt
                        'completion_tokens': int,  # Tokens in completion
                        'total_tokens': int,  # Total tokens used
                        'elapsed_seconds': float  # Time taken for completion in seconds
                    },
                    'model': str,  # Model used
                    'finish_reason': str  # Reason generation stopped
                }
            }

            On error, returns:
            {
                'text': f"Error: {error_message}",
                'metadata': {
                    'usage': {
                        'prompt_tokens': 0,
                        'completion_tokens': 0,
                        'total_tokens': 0,
                        'elapsed_seconds': 0.0
                    }
                }
            }
        """
        self.load_client()

        start_time = time.time()

        try:
            options = {
                "temperature": temperature,
            }
            if max_tokens is not None:
                options["num_predict"] = max_tokens

            response = await self.client.chat(
                model=self.model, messages=messages, options=options
            )
            elapsed_seconds = time.time() - start_time

            response_text = response["message"]["content"]
            completion_tokens = len(response_text.split())
            prompt_tokens = sum(len(m.get("content", "").split()) for m in messages)
            total_tokens = prompt_tokens + completion_tokens

            return {
                "text": response_text,
                "metadata": {
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                        "elapsed_seconds": elapsed_seconds,
                    },
                    "model": self.model,
                    "finish_reason": response.get("done_reason", "stop"),
                },
            }
        except Exception as e:
            logger.error(f"Error calling Ollama: {e}")
            return {
                "text": f"Error: {str(e)}",
                "metadata": {
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                        "elapsed_seconds": time.time() - start_time,
                    }
                },
            }

    def get_token_cost(self) -> float:
        """Get the cost per 1K tokens for the Ollama model.

        Since Ollama runs locally, the cost is always 0.

        Returns:
            float: Always returns 0.0 (no cost for local models)
        """
        return 0.0


class OpenAIChatClient(IChatClient):
    def __init__(
        self,
        model: str = "gpt-4o",
    ):
        """Initialize OpenAI chat client.

        Args:
            model: Name of the OpenAI model to use (default: "gpt-4o")

        Raises:
            ValueError: If OPENAI_API_KEY environment variable is not set
            RuntimeError: If the API key is invalid or the model is not available
        """
        self.model = model
        self.client = None

    def load_client(self):
        if self.client:
            return

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set. "
                "Please set OPENAI_API_KEY in your .env file or environment variables."
            )

        self.client = openai.OpenAI(api_key=api_key)

        # Verify API key and model availability
        try:
            # Make a lightweight request to check the API key and model
            self.client.models.retrieve(self.model)
        except openai.AuthenticationError as e:
            raise RuntimeError(
                "Invalid OpenAI API key. Please check your API key and try again."
            ) from e
        except openai.NotFoundError as e:
            raise RuntimeError(
                f"Model '{self.model}' not found or you don't have access to it. "
                f"Please check the model name and your API permissions."
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to connect to OpenAI API: {str(e)}") from e

    async def get_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """Get a chat completion from the OpenAI model.

        Args:
            messages: List of message dicts with format:
                [{"role": "user"|"assistant"|"system", "content": str}, ...]
            tools: Optional list of tool definitions with format:
                [{
                    'type': 'function',
                    'function': {
                        'name': str,
                        'description': str,
                        'parameters': dict
                    }
                }, ...]
            max_tokens: Optional maximum number of tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)

        Returns:
            Dict with structure:
            {
                'text': str,  # Generated response content
                'metadata': {
                    'usage': {
                        'prompt_tokens': int,  # Tokens in prompt
                        'completion_tokens': int,  # Tokens in completion
                        'total_tokens': int,  # Total tokens used
                        'elapsed_seconds': float  # Time taken for completion in seconds
                    },
                    'model': str,  # Model used
                    'finish_reason': str  # Reason generation stopped
                }
            }

            On error, returns:
            {
                'text': f"Error: {error_message}",
                'metadata': {
                    'usage': {
                        'prompt_tokens': 0,
                        'completion_tokens': 0,
                        'total_tokens': 0,
                        'elapsed_seconds': 0.0
                    }
                }
            }
        """
        self.load_client()

        start_time = time.time()

        def sync_call():
            """Synchronous wrapper for get_completion."""
            return self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                max_tokens=max_tokens,
                temperature=temperature,
            )

        try:
            loop = asyncio.get_event_loop()
            completion = await loop.run_in_executor(None, sync_call)
            elapsed_seconds = time.time() - start_time

            text = completion.choices[0].message.content if completion.choices else ""

            if hasattr(completion, "usage") and completion.usage:
                usage = {
                    "prompt_tokens": completion.usage.prompt_tokens,
                    "completion_tokens": completion.usage.completion_tokens,
                    "total_tokens": completion.usage.total_tokens,
                    "elapsed_seconds": elapsed_seconds,
                }
            else:
                usage = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "elapsed_seconds": elapsed_seconds,
                }

            return {
                "text": text,
                "metadata": {
                    "usage": usage,
                    "model": self.model,
                    "finish_reason": completion.choices[0].finish_reason
                    if completion.choices and completion.choices[0].finish_reason
                    else "stop",
                },
            }
        except Exception as e:
            logger.error(f"Error calling OpenAI: {e}")
            return {
                "text": f"Error: {str(e)}",
                "metadata": {"Usage": {"TotalTokenCount": 0}},
            }

    def get_token_cost(self) -> float:
        """Get the cost per 1K tokens for the OpenAI model.

        Returns:
            float: Cost in USD per 1K tokens based on the model

        Note:
            - gpt-4o: $0.005/1K tokens
            - gpt-4-turbo: $0.01/1K tokens
            - gpt-4: $0.03/1K tokens
            - gpt-3.5-turbo: $0.0005/1K tokens
        """
        pricing = {
            "gpt-4": 0.03,  # $0.03 per 1K tokens
            "gpt-4o": 0.005,  # $0.005 per 1K tokens
            "gpt-4-turbo": 0.01,  # $0.01 per 1K tokens
            "gpt-3.5-turbo": 0.0005,  # $0.0005 per 1K tokens
        }
        return pricing.get(self.model.lower(), 0.0)
