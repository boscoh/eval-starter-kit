from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


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


class IChatClient(ABC):
    """
    Abstract base class for chat client implementations.

    This interface defines a standardized way to interact with various chat
    model providers (like OpenAI, Ollama, etc.) using a consistent API.
    All chat clients should implement this interface to ensure compatibility
    across different model providers.

    The interface uses basic Python data structures (lists and dicts) for
    inputs and outputs, making it easy to adapt to different chat libraries
    while maintaining a consistent API contract.

    Key Features:
    - Standardized message format using list of dictionaries
    - Support for tool/function calling through optional tools parameter
    - Asynchronous interface for better performance
    - Standardized response format with text and metadata

    Usage:
    ```python
    # Create a chat client instance
    client = get_chat_client("ollama", model="llama3.2")

    # Prepare messages
    messages = [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"}
    ]

    # Get completion
    response = await client.get_completion(messages)
    ```
    """

    @abstractmethod
    async def get_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Get a chat completion from the model.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            tools: Optional list of tool/function definitions

        Returns:
            Dict with keys:
                - 'text': The generated response text
                - 'metadata': Dict containing usage information
                    - 'Usage': Dict with 'TotalTokenCount' key
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


class OllamaChatClient(IChatClient):
    def __init__(self, model: str = "llama3.2"):
        import ollama

        self.model = model
        self.client = ollama.AsyncClient()

    async def get_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> dict:
        try:
            response = await self.client.chat(model=self.model, messages=messages)
            response_text = response["message"]["content"]
            token_count = sum(
                len(m.get("content", "").split()) for m in messages
            ) + len(response_text.split())
            return {
                "text": response_text,
                "metadata": {"Usage": {"TotalTokenCount": token_count}},
            }
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            return {
                "text": f"Error: {str(e)}",
                "metadata": {"Usage": {"TotalTokenCount": 0}},
            }

    def get_token_cost(self) -> float:
        """
        Get the cost per token for Ollama models.

        Args:
            model_id: The model identifier (e.g., 'llama3.2')

        Returns:
            0.0 since Ollama is typically free/local
        """
        return 0


class OpenAIChatClient(IChatClient):
    def __init__(self, model: str = "gpt-4o", max_tokens: int = 1000):
        import os

        from openai import OpenAI

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
    ) -> dict:
        # Note: OpenAI Python SDK is synchronous, so run in thread executor if needed
        import asyncio

        loop = asyncio.get_event_loop()

        def sync_call():
            return self.client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=messages,
                tools=tools,
            )

        completion = await loop.run_in_executor(None, sync_call)
        text = completion.choices[0].message.content if completion.choices else ""
        token_count = completion.usage.total_tokens if completion.usage else None
        return {
            "text": text,
            "metadata": {"Usage": {"TotalTokenCount": token_count}},
        }

    def get_token_cost(self) -> float:
        """
        Get the cost per token for OpenAI models.

        Args:
            model_id: The OpenAI model identifier (e.g., 'gpt-4', 'gpt-3.5-turbo')

        Returns:
            Cost per token in USD (as of 2025-07-19):
            - gpt-4: $0.03 per 1K tokens
            - gpt-4o: $0.03 per 1K tokens
            - gpt-3.5-turbo: $0.002 per 1K tokens
        """
        pricing = {
            "gpt-4": 0.03,  # $0.03 per 1K tokens
            "gpt-4o": 0.03,  # Assuming same pricing as gpt-4
            "gpt-3.5-turbo": 0.002,  # $0.002 per 1K tokens
        }
        return pricing.get(self.model.lower(), 0.0)
