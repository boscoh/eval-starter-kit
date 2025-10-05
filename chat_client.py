import asyncio
import json
import logging
import os
import subprocess
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Dict, List, Optional


import aioboto3
import boto3
from botocore.exceptions import ClientError, ProfileNotFound
from dotenv import load_dotenv
import ollama
import openai
from rich.pretty import pretty_repr

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
    if client_type == "bedrock":
        return BedrockChatClient(**kwargs)
    raise ValueError(f"Unknown chat client type: {client_type}")


def parse_response_as_json_list(response):
    """Parse JSON from text response, extracting from markdown or .transactions if needed."""
    import re

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

    parsed = try_parse(response_text)
    if parsed is not None:
        return extract_transactions(parsed)

    patterns = [
        r"```(?:json|python)?\s*([\s\S]*?)\s*```",
        r"```(?:json)?\s*({[\s\S]*})\s*```",
        r"\{[\s\S]*\}",
        r"({[\s\S]*})",
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

    async def connect(self):
        """Connect to the chat client."""
        pass

    async def close(self):
        """Close the chat client."""
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

        try:
            subprocess.run(["ollama", "--version"], capture_output=True, check=True)
        except (subprocess.SubprocessError, FileNotFoundError):
            raise RuntimeError(
                "Ollama is not running or not installed. "
                "Please start the Ollama service and try again."
            )

        self.client = ollama.AsyncClient()
        try:
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

        try:
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


@lru_cache(maxsize=None)
def get_aws_config():
    """
    Prepare AWS configuration for boto3 client initialization.
    
    This function searches for AWS profiles and saved credentials to build a 
    configuration dictionary that can be used to initialize boto3 clients and 
    sessions. It validates the discovered credentials to ensure they are properly 
    configured and not expired.
    
    Credential Discovery Process:
    1. Looks for AWS_PROFILE environment variable to determine profile name
    2. Searches for saved credentials in ~/.aws/credentials file
    3. Creates a boto3 session using the discovered profile (or default)
    4. Validates credentials contain required access_key and secret_key
    5. Tests credential validity with an STS GetCallerIdentity call
    6. Checks for token expiration on temporary/session credentials
    
    Environment Variables:
        AWS_PROFILE (str, optional): Name of the AWS profile to use from 
                                   ~/.aws/credentials. If not set, uses the 
                                   default profile.
    
    Returns:
        dict: AWS configuration dictionary for boto3 client initialization:
            - profile_name (str, optional): The AWS profile name to pass to 
                                          boto3.client() or boto3.Session()
    
    Note:
        This function is cached to avoid repeated credential discovery and 
        validation. The returned configuration can be unpacked directly into 
        boto3 client constructors. All validation errors are logged but do not 
        raise exceptions - returns gracefully with empty config on failure.
        
    Examples:
        >>> aws_config = get_aws_config()
        >>> s3_client = boto3.client('s3', **aws_config)
        >>> 
        >>> # Or with session
        >>> session = boto3.Session(**aws_config)
        >>> dynamodb = session.client('dynamodb')
    """
    aws_config = {}

    profile_name = os.getenv("AWS_PROFILE")
    if profile_name:
        logger.info(f"Authenticate with AWS_PROFILE={profile_name}")
        aws_config["profile_name"] = profile_name

    try:
        aws_credentials_path = os.path.expanduser("~/.aws/credentials")
        if not os.path.exists(aws_credentials_path):
            logger.info("No AWS credentials file at ~/.aws/credentials")
            return aws_config

        session = boto3.Session(profile_name=profile_name) if profile_name else boto3.Session()
        credentials = session.get_credentials()
        
        if not credentials or not credentials.access_key or not credentials.secret_key:
            logger.warning("AWS credentials not properly configured")
            return aws_config

        sts = session.client("sts")
        identity = sts.get_caller_identity()

        if hasattr(credentials, "token"):
            creds = credentials.get_frozen_credentials()
            if hasattr(creds, "expiry_time") and creds.expiry_time < datetime.now(timezone.utc):
                logger.warning(f"AWS credentials expired on {creds.expiry_time}")
                return aws_config

        logger.info(f"Valid AWS credentials found for '{identity['Arn']}'")

    except ProfileNotFound:
        logger.warning(f"AWS profile '{profile_name}' not found")
    except ClientError as e:
        if e.response["Error"]["Code"] == "ExpiredToken":
            logger.warning("AWS credentials have expired")
        else:
            logger.warning(f"Error validating AWS credentials: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error checking AWS credentials: {str(e)}")

    return aws_config


class BedrockChatClient(IChatClient):
    def __init__(
        self,
        model: str = "anthropic.claude-3-sonnet-20240229-v1:0",
        embedding_model: str = "amazon.titan-embed-text-v2:0",
        region_name: str = "us-east-1",  # Update if you're in a different region
    ):
        """
        Initialize Bedrock chat client.

        Args:
            model: Text generation model ID for Bedrock.
            embedding_model: Text embedding model ID for Bedrock.
            region_name: AWS region name.
        """
        self.model = model
        self.embedding_model = embedding_model
        self.region_name = region_name
        self.client = None
        self._session = None
        self._closed = True

    async def connect(self):
        """Initialize the async client session and client."""
        if self.client is not None and not self._closed:
            return

        logger.info(f"Initializing Bedrock client for model {self.model}")
        self._session = aioboto3.Session(**get_aws_config())
        self.client = await self._session.client("bedrock-runtime").__aenter__()
        logger.info(f"Initialized Bedrock client")
        self._closed = False

    async def close(self):
        """Close the client and release resources."""
        if self.client is not None and not self._closed:
            await self.client.__aexit__(None, None, None)
            self.client = None
            self._closed = True

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def get_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Get a chat completion from the Bedrock model.

        Handles both Claude models (using Converse API) and other models (using invoke_model).

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            tools: Optional list of tool definitions for function calling
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)

        Returns:
            Dictionary with 'text' response and 'metadata' including usage info
        """
        await self.connect()
        start_time = time.time()

        try:
            system_parts = []
            formatted_messages = []

            for msg in messages:
                role = msg["role"]
                content = msg.get("content", "")

                if role == "system":
                    system_parts.append(content)
                else:
                    role = "user" if role == "user" else "assistant"
                    formatted_messages.append(
                        {"role": role, "content": [{"text": content}]}
                    )

            system_blocks = (
                [{"text": "\n\n".join(system_parts)}] if system_parts else []
            )

            formatted_tools = None
            if tools:
                formatted_tools = [
                    {
                        "toolSpec": {
                            "name": tool["function"]["name"],
                            "description": tool["function"].get("description", ""),
                            "inputSchema": {
                                "json": tool["function"].get("parameters", {})
                            },
                        }
                    }
                    for tool in tools
                ]

            try:
                request_kwargs = {
                    "modelId": self.model,
                    "messages": formatted_messages,
                    "system": system_blocks,
                    "inferenceConfig": {
                        "temperature": temperature,
                        "maxTokens": max_tokens or 1024,
                    },
                }

                if tools:
                    request_kwargs["toolConfig"] = {"tools": formatted_tools}

                logger.info(f"Request kwargs: {pretty_repr(request_kwargs)}")

                response = await self.client.converse(**request_kwargs)

                logger.info(f"Response: {pretty_repr(response)}")

                text_parts = []
                tool_calls = []

                if isinstance(response, str):
                    text_parts.append(response)
                    usage = {}
                    stop_reason = "stop"
                else:
                    output = response.get("output", {})
                    if isinstance(output, dict) and "message" in output:
                        message = output["message"]
                        for content in message.get("content", []):
                            if "text" in content:
                                text_parts.append(content["text"])
                            elif "toolUse" in content:
                                tool_use = content["toolUse"]
                                tool_calls.append(
                                    {
                                        "function": {
                                            "name": tool_use["name"],
                                            "arguments": json.dumps(
                                                tool_use.get("input", {})
                                            ),
                                            "tool_call_id": tool_use.get(
                                                "toolUseId", ""
                                            ),
                                        }
                                    }
                                )

                    usage = response.get("usage", {})
                    stop_reason = response.get("stopReason", "unknown")

                return {
                    "text": "\n".join(text_parts).strip(),
                    "metadata": {
                        "usage": {
                            "prompt_tokens": usage.get("inputTokens", 0),
                            "completion_tokens": usage.get("outputTokens", 0),
                            "total_tokens": usage.get("inputTokens", 0)
                            + usage.get("outputTokens", 0),
                            "elapsed_seconds": time.time() - start_time,
                        },
                        "model": self.model,
                        "finish_reason": stop_reason,
                    },
                    "tool_calls": tool_calls if tool_calls else None,
                }
            except Exception as e:
                logger.error(f"Error in Converse API call: {str(e)}")
                raise

        except Exception as e:
            logger.error(f"Error in get_completion: {e}")
            return {
                "text": f"Error: {str(e)}",
                "metadata": {
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                        "elapsed_seconds": time.time() - start_time,
                    },
                    "model": self.model,
                    "error": str(e),
                },
            }

    async def embed(self, input: str) -> List[float]:
        """
        Generate an embedding using Bedrock's embedding model.

        Args:
            input: The input text to generate embeddings for

        Returns:
            List of floats representing the embedding vector

        Raises:
            RuntimeError: If there's an error generating the embedding
        """
        try:
            await self.connect()

            if hasattr(self.client, "embed"):
                response = await self.client.embed(
                    modelId=self.embedding_model, inputText=input
                )
                return response["embedding"]
            else:
                response = await self.client.invoke_model(
                    modelId=self.embedding_model,
                    contentType="application/json",
                    accept="application/json",
                    body=json.dumps({"inputText": input}),
                )
                raw_body = await response["body"].read()
                body = json.loads(raw_body.decode("utf-8"))
                return body["embedding"]

        except Exception as e:
            logger.error(f"Error calling Bedrock embed: {e}")
            raise RuntimeError(f"Error generating embedding: {str(e)}")

    def get_token_cost(self) -> float:
        """
        Get the cost per 1K tokens for the model.

        Returns:
            float: Cost in USD per 1K tokens

        Note:
            - Claude 3 Opus: $15.00/1M input, $75.00/1M output
            - Claude 3 Sonnet: $3.00/1M input, $15.00/1M output
            - Claude 3 Haiku: $0.25/1M input, $1.25/1M output
            - Other models default to 0.0
        """
        model_lower = self.model.lower()

        if "opus" in model_lower:
            return 0.015  # $15 per 1M tokens = $0.015 per 1K tokens
        elif "sonnet" in model_lower:
            return 0.003  # $3 per 1M tokens = $0.003 per 1K tokens
        elif "haiku" in model_lower:
            return 0.00025  # $0.25 per 1M tokens = $0.00025 per 1K tokens

        return 0.0  # Default for other models
