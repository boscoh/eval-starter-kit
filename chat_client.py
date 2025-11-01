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
import ollama
import openai
from botocore.exceptions import ClientError, ProfileNotFound
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def get_chat_client(client_type: str, **kwargs) -> "IChatClient":
    """
    Gets a chat client that satisfies IChatClient interface.

    Args:
        client_type: "openai", "ollama", or "bedrock"
        **kwargs: Additional keyword arguments specific to the chat client type:
            - model: str
    """
    client_type = client_type.lower()
    if client_type == "openai":
        return OpenAIChatClient(**kwargs)
    if client_type == "ollama":
        return OllamaChatClient(**kwargs)
    if client_type == "bedrock":
        return BedrockChatClient(**kwargs)
    raise ValueError(f"Unknown chat client type: {client_type}")


class IChatClient(ABC):
    """Abstract base class for IChatClient, an API for LLM with async interface"""

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

        Note:
            - All implementations should handle errors gracefully by returning
              the standardized error format rather than raising exceptions
            - Tool/function calling support varies by provider
            - Token counting methods may vary between providers

        Args:
            messages: List of message dictionaries representing the conversation history.
                Each message dict must contain:
                - 'role': str - One of 'user', 'assistant', or 'system'
                - 'content': str - The message content/text
                - 'name': str (optional) - Name of the message sender

                Example:
                [
                    {'role': 'system', 'content': 'You are a helpful assistant.'},
                    {'role': 'user', 'content': 'Hello, how are you?'}
                ]

            tools: Optional list of tool/function definitions for function calling.
                Each tool dict should contain:
                - 'type': str - Tool type (typically 'function')
                - 'function': dict - Function specification with:
                  - 'name': str - Function name
                  - 'description': str - Function description
                  - 'parameters': dict - JSON Schema defining parameters

                Example:
                [{
                    'type': 'function',
                    'function': {
                        'name': 'get_weather',
                        'description': 'Get current weather',
                        'parameters': {
                            'type': 'object',
                            'properties': {
                                'location': {'type': 'string'}
                            }
                        }
                    }
                }]

            max_tokens: Maximum number of tokens to generate in the completion.
                If None, uses the model's default limit. Different models have
                different default and maximum token limits.

            temperature: Controls randomness in generation (0.0 to 1.0).
                - 0.0: Deterministic, always picks most likely token
                - 1.0: Maximum randomness
                - Values between 0.1-0.7 are typically good for most use cases

        Returns:
            Dict[str, Any]: Standardized response dictionary containing:
            {
                'text': str,
                'metadata': {
                    'usage': {
                        'prompt_tokens': int,
                        'completion_tokens': int,
                        'total_tokens': int,
                        'elapsed_seconds': float
                    },
                    'model': str,
                    'finish_reason': str
                },
                'tool_calls': [
                    {
                        'function': {
                            'name': str,
                            'arguments': str,
                            'tool_call_id': str
                        }
                    }
                ] | None
            }

            On error, returns:
            {
                'text': 'Error: <error_message>',
                'metadata': {
                    'usage': {
                        'prompt_tokens': 0,
                        'completion_tokens': 0,
                        'total_tokens': 0,
                        'elapsed_seconds': float
                    },
                    'model': str,
                    'error': str
                }
            }

        """
        pass

    @abstractmethod
    async def embed(self, input: str) -> List[float]:
        """Generate a text embedding vector for the given input string."""
        pass

    @abstractmethod
    def get_token_cost(self) -> float:
        """Get the cost per 1K tokens for the model in AUD."""
        pass

    async def connect(self):
        """Initialize async resources. Override in subclasses as needed."""
        pass

    async def close(self):
        """Clean up async resources. Override in subclasses as needed."""
        pass

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


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

    parsed = try_parse(response_text)
    if parsed:
        return parsed

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
            if parsed:
                return parsed

    return None


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

    async def connect(self):
        if self.client:
            return

        try:
            subprocess.run(["ollama", "--version"], capture_output=True, check=True)
        except (subprocess.SubprocessError, FileNotFoundError):
            raise RuntimeError(
                "Ollama is not running or not installed. "
                "Please start the Ollama service and try again."
            )

        logger.info(f"Initializing 'ollama:{self.model}'")
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
        """Ollama implementation of get_completion. Tools not supported."""
        await self.connect()

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

    async def embed(self, input: str) -> List[float]:
        """Generate text embeddings using Ollama's embedding capabilities."""
        await self.connect()

        try:
            response = await self.client.embeddings(model=self.model, prompt=input)
            return response["embedding"]
        except Exception as e:
            logger.error(f"Error calling Ollama embed: {e}")
            raise RuntimeError(f"Error generating embedding: {str(e)}")

    def get_token_cost(self) -> float:
        """Returns 0.0 AUD since Ollama runs locally with no API costs."""
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
        self._closed = True

    async def connect(self):
        if self.client and not self._closed:
            return

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set. "
                "Please set OPENAI_API_KEY in your .env file or environment variables."
            )

        logger.info(f"Initializing 'openai:{self.model}'")
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self._closed = False

        try:
            await self.client.models.retrieve(self.model)
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

    async def close(self):
        """Close the OpenAI client and release resources."""
        if self.client is not None and not self._closed:
            await self.client.close()
            self.client = None
            self._closed = True

    async def get_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """OpenAI implementation of get_completion with full tool support."""
        await self.connect()

        start_time = time.time()

        try:
            completion = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                max_tokens=max_tokens,
                temperature=temperature,
            )
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

            # Extract tool calls if present
            tool_calls = None
            if completion.choices and completion.choices[0].message.tool_calls:
                tool_calls = []
                for tool_call in completion.choices[0].message.tool_calls:
                    tool_calls.append(
                        {
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments,
                                "tool_call_id": tool_call.id,
                            }
                        }
                    )

            return {
                "text": text,
                "metadata": {
                    "usage": usage,
                    "model": self.model,
                    "finish_reason": completion.choices[0].finish_reason
                    if completion.choices and completion.choices[0].finish_reason
                    else "stop",
                },
                "tool_calls": tool_calls,
            }
        except Exception as e:
            logger.error(f"Error calling OpenAI: {e}")
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
        """Generate text embeddings using OpenAI's embedding model."""
        await self.connect()

        try:
            response = await self.client.embeddings.create(
                model="text-embedding-3-small", input=input
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error calling OpenAI embed: {e}")
            raise RuntimeError(f"Error generating embedding: {str(e)}")

    def get_token_cost(self) -> float:
        """Returns OpenAI model pricing per 1K tokens in AUD."""
        pricing = {
            "gpt-4": 0.045,
            "gpt-4o": 0.00375,
            "gpt-4o-mini": 0.000225,
            "gpt-4-turbo": 0.015,
            "gpt-3.5-turbo": 0.00075,
        }
        model_key = self.model.lower()
        if model_key not in pricing:
            logger.warning(
                f"Unknown OpenAI model '{self.model}', using default cost of 0.0 AUD"
            )
        return pricing.get(model_key, 0.0)


@lru_cache(maxsize=None)
def get_aws_config(is_raise_exception: bool = True):
    """
    Returns AWS configuration for boto3 client initialization.

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
        aws_config["profile_name"] = profile_name

    try:
        aws_credentials_path = os.path.expanduser("~/.aws/credentials")
        if not os.path.exists(aws_credentials_path):
            logger.info("No AWS credentials file at ~/.aws/credentials")
            return aws_config

        session = (
            boto3.Session(profile_name=profile_name)
            if profile_name
            else boto3.Session()
        )
        credentials = session.get_credentials()

        if not credentials or not credentials.access_key or not credentials.secret_key:
            return aws_config

        sts = session.client("sts")
        identity = sts.get_caller_identity()

        if hasattr(credentials, "token"):
            creds = credentials.get_frozen_credentials()
            if hasattr(creds, "expiry_time") and creds.expiry_time < datetime.now(
                timezone.utc
            ):
                logger.warning(f"AWS credentials expired on {creds.expiry_time}")
                return aws_config

    except ProfileNotFound:
        if is_raise_exception:
            raise
        logger.warning(f"AWS profile '{profile_name}' not found")
    except ClientError as e:
        if is_raise_exception:
            raise
        if e.response["Error"]["Code"] == "ExpiredToken":
            logger.warning("AWS credentials have expired")
        else:
            logger.warning(f"Error validating AWS credentials: {str(e)}")
    except Exception as e:
        if is_raise_exception:
            raise
        logger.error(f"Unexpected error checking AWS credentials: {str(e)}")

    return aws_config


class BedrockChatClient(IChatClient):
    def __init__(
        self,
        model: str = "anthropic.claude-3-sonnet-20240229-v1:0",
    ):
        """
        Initialize Bedrock chat client.

        This implementation exclusively uses the Bedrock Converse API to enable
        tool/function calling support. As a result, only Claude models are supported
        since they are the primary models that work well with the Converse API for
        tool usage. Other Bedrock models may not support tools through this API.

        Args:
            model: Claude model ID for Bedrock (e.g., "anthropic.claude-3-sonnet-20240229-v1:0").
        """
        self.model = model
        self.client = None
        self._session = None
        self._closed = True

    async def connect(self):
        """Initialize the async client session and client."""
        if self.client is not None and not self._closed:
            return

        logger.info(f"Initializing 'bedrock:{self.model}'")
        self._session = aioboto3.Session(**get_aws_config())
        self.client = await self._session.client("bedrock-runtime").__aenter__()
        self._closed = False

    async def close(self):
        """Close the client and release resources."""
        if self.client is not None and not self._closed:
            await self.client.__aexit__(None, None, None)
            self.client = None
            self._closed = True

    async def get_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """Bedrock implementation using Converse API with tool support."""
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

                response = await self.client.converse(**request_kwargs)

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
        """Generate text embeddings using Bedrock's embedding model."""
        try:
            await self.connect()

            response = await self.client.invoke_model(
                modelId=self.model,
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
        """Returns Bedrock model pricing per 1K tokens in AUD based on model type."""
        model_lower = self.model.lower()

        if "opus" in model_lower:
            return 0.0225
        elif "sonnet" in model_lower:
            return 0.0045
        elif "haiku" in model_lower:
            return 0.000375
        else:
            logger.warning(
                f"Unknown Bedrock model '{self.model}', using default cost of 0.0 AUD"
            )
            return 0.0
