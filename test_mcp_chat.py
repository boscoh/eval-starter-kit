#!/usr/bin/env python3
"""
Test client demonstrating MCP tool integration with multi-step reasoning for speaker queries.
"""

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from path import Path

from chat_client import IChatClient, get_chat_client
from setup_logger import setup_logging_with_rich_logger

load_dotenv()
setup_logging_with_rich_logger()

logger = logging.getLogger(__name__)


class SpeakerMcpClient:
    def __init__(self, llm_service: str = "bedrock"):
        self.mcp_client: Optional[ClientSession] = None
        self._session_context: Optional[ClientSession] = None
        self._stdio_context: Optional[StdioServerParameters] = None

        self.tools: Optional[List[Dict[str, Any]]] = None

        self.llm_service = llm_service
        self.chat_client: IChatClient = None
        models = {
            "bedrock": "anthropic.claude-3-sonnet-20240229-v1:0",
            "openai": "gpt-4o"
        }
        if self.llm_service not in models:
            raise ValueError(f"Unsupported LLM service for tools: {self.llm_service}")
        model = models[self.llm_service]
        self.chat_client = get_chat_client(self.llm_service, model=model)

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()
        return False

    async def connect(self):
        if self.mcp_client:
            return

        env = os.environ.copy()
        server_script_path = Path(__file__).parent / "mcp_server.py"
        env["PYTHONPATH"] = server_script_path.parent
        env["LLM_SERVICE"] = self.llm_service
        server_params = StdioServerParameters(
            command="uv",
            args=["run", "python", server_script_path],
            env=env,
        )
        self._stdio_context = stdio_client(server_params)
        _stdio_read, _stdio_write = await self._stdio_context.__aenter__()
        self._session_context = ClientSession(_stdio_read, _stdio_write)
        self.mcp_client = await self._session_context.__aenter__()
        await self.mcp_client.initialize()

        self.tools = await self.get_tools()
        names = [tool["function"]["name"] for tool in self.tools]
        logger.info(f"Connected Server to MCP tools: {', '.join(names)}")

        await self.chat_client.connect()

    async def disconnect(self):
        try:
            if self.chat_client:
                await self.chat_client.close()
        except Exception:
            pass
        try:
            if self._session_context:
                await self._session_context.__aexit__(None, None, None)
        except Exception:
            pass
        try:
            if self._stdio_context:
                await self._stdio_context.__aexit__(None, None, None)
        except Exception:
            pass
        self.mcp_client = None

    async def get_tools(self):
        """Returns tool in format compatible with IChatClient.get_completion()"""
        response = await self.mcp_client.list_tools()
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                },
            }
            for tool in response.tools
        ]

    def parse_tool_args(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        tool_args_json = tool_call["function"].get("arguments", "")
        if not tool_args_json:
            return {}
        try:
            return json.loads(tool_args_json)
        except json.JSONDecodeError:
            return {"__raw": tool_args_json}

    def is_duplicate_call(self, tool_name: str, tool_args: Dict[str, Any], seen_calls: set) -> bool:
        """Return True if this tool call was seen before; otherwise record it and return False."""
        try:
            normalized_args = json.dumps(tool_args, sort_keys=True)
        except Exception:
            normalized_args = str(tool_args)
        call_key = (tool_name, normalized_args)
        if call_key in seen_calls:
            logger.info(f"Skipped duplicate tool call: {call_key}")
            return True
        seen_calls.add(call_key)
        return False

    def log_messages(self, messages: List[Dict[str, Any]], max_length: int = 100):
        """Log each message with truncated content if too long."""
        logger.info(f"Calling LLM with {len(messages)} messages:")
        for i, msg in enumerate(messages):
            msg_content = msg.get("content", "")
            content = msg_content[:max_length]
            truncated_content = " ".join(content.split()) + ("..." if len(msg_content) > max_length else "")
            role = msg.get("role", 'unknown')
            logger.info(f"- {role}: {truncated_content}")

    async def process_query(self, query: str) -> str:
        """Returns a response to a user query by getting a completion with tool calls.

        Supports multi-step tool chaining (e.g., find correct name -> get_speaker_by_name)
        by iteratively executing returned tool calls and re-querying the model
        with tool outputs until no more tool calls are requested or a safety
        limit is reached.
        """
        await self.connect()

        system_prompt = """You are a helpful assistant that can use tools to answer 
        questions about speakers.

        IMPORTANT: Proactively use tools in multiple rounds to gather, refine, and
        verify information. Prefer taking several small, iterative tool steps over
        guessing. You should:

        MULTI-ROUND TOOL CHAINING STRATEGY:
        1. ANALYZE the query and list what you need to know to answer it well
        2. PLAN a sequence of tool calls (potentially across multiple rounds)
        3. EXECUTE one or more tool calls, then reassess what you learned
        4. ITERATE with additional calls to fill gaps, cross-check, or drill down
        5. AVOID exact duplicate calls with identical parameters (vary params to explore)
        6. SYNTHESIZE the gathered evidence into a comprehensive final answer

        TOOL USAGE GUIDELINES:
        - Feel free to make multiple tool calls across multiple reasoning rounds
        - Use follow-up calls to verify, compare alternatives, and resolve ambiguity
        - Avoid repeating the exact same call with the same parameters
        - ALWAYS provide a complete, detailed final answer that includes specific
          information you obtained via the tools

        Available tools can help you search, filter, and retrieve speaker
        information. Use them iteratively and transparently. Your final answer
        should be detailed and include specific speaker information."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": str(query)},
        ]

        self.log_messages(messages)

        response = await self.chat_client.get_completion(messages, self.tools)

        tool_calls = response.get("tool_calls")
        max_iterations = 5
        iterations = 0
        seen_calls = set()

        while tool_calls and iterations < max_iterations:
            iterations += 1
            logger.info(
                f"Reasoning step {iterations} with {len(tool_calls)} tool calls"
            )

            if content := response.get("text", ""):
                messages.append( { "role": "assistant", "content": content } )

            tool_results = []
            for tool_call in tool_calls:
                tool_name = tool_call["function"]["name"]
                tool_args = self.parse_tool_args(tool_call)
                if self.is_duplicate_call(tool_name, tool_args, seen_calls):
                    tool_result = f"Duplicate tool call: {tool_name}({tool_args})"
                else:
                    try:
                        logger.info(f"Calling tool {tool_name}({tool_args})...")
                        result = await self.mcp_client.call_tool(tool_name, tool_args)
                        tool_result = f"Tool used: {tool_name}({tool_args}) \nTool result: {getattr(result, "content", result)}\n\n"
                    except Exception as e:
                        tool_result = f"Tool used {tool_name}({tool_args}) but failed: {str(e)}"
                        logger.error(tool_result)
                tool_results.append(tool_result)

            content = f"""
            {'\n'.join(tool_results)}

            REVIEW TOOL RESULTS: What did you learn, and what's still needed?

            KEY QUESTIONS TO ASK YOURSELF:
            - Did the results reveal any ambiguities that need clarification?
            - Are there alternative approaches or parameters worth exploring?
            - Could you cross-reference or verify the information you found?
            - Is there related information that would strengthen your answer?
            - Would filtering, sorting, or searching differently provide better results?

            ✓ USE MORE TOOLS if you can:
            - Verify or cross-check information from different angles
            - Explore alternative search terms or parameters
            - Fill gaps in the information you've gathered
            - Get more specific details about promising results
            - Compare multiple options before concluding

            ✗ PROVIDE FINAL ANSWER only when:
            - You've explored the main approaches to gathering information
            - You have comprehensive data that fully addresses the question
            - Additional tool calls would genuinely be duplicates (same parameters)

            FINAL ANSWER REQUIREMENTS:
            - Incorporate specific data and details from the tool results
            - Be comprehensive and directly address the user's question
            - Do not mention that you used tools or the reasoning process
            """

            messages.append( { "role": "assistant", "content": content } )

            self.log_messages(messages)

            response = await self.chat_client.get_completion(messages, self.tools)

            tool_calls = response.get("tool_calls")

        if iterations >= max_iterations:
            logger.warning(f"Reached maximum tool iterations ({max_iterations})")

        return response.get("text", "")


async def setup_async_exception_handler():
    loop = asyncio.get_event_loop()

    def silence_event_loop_closed(loop, context):
        if "exception" not in context or not isinstance(
            context["exception"], (RuntimeError, GeneratorExit)
        ):
            loop.default_exception_handler(context)

    loop.set_exception_handler(silence_event_loop_closed)


async def amain(service):
    await setup_async_exception_handler()
    async with SpeakerMcpClient(service) as client:
        print("Type your questions about speakers.")
        print("Type 'quit', 'exit', or 'q' to end the conversation.")
        while True:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ["quit", "exit", "q", ""]:
                print("Goodbye!")
                return
            response = await client.process_query(query=user_input)
            print(f"\nResponse: {response}")


if __name__ == "__main__":
    service = os.getenv("LLM_SERVICE", "openai")  # "bedrock", "openai"
    try:
        asyncio.run(amain(service))
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
