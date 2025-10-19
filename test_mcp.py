#!/usr/bin/env python3
"""
MCP Client for Minimal Speaker Recommendation System

This client connects to the MCP server using STDIO and provides
a simple interface to interact with the speaker recommendation tools.
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from chat_client import IChatClient, get_chat_client
from setup_logger import setup_logging_with_rich_logger

logger = logging.getLogger(__name__)


class MinimalMcpClient:
    """Client for interacting with the xConf Speaker Assistant MCP server."""

    def __init__(self, llm_service: str = "bedrock"):
        self.mcp_client: Optional[ClientSession] = None
        self.server_script_path = Path(__file__).parent / "mcp_server.py"
        self._session_context: Optional[ClientSession] = None
        self._stdio_context: Optional[StdioServerParameters] = None

        self.tools: Optional[List[Dict[str, Any]]] = None

        self.chat_client: IChatClient = None
        if llm_service == "bedrock":
            self.chat_client = get_chat_client(
                llm_service, model="anthropic.claude-3-sonnet-20240229-v1:0"
            )
        else:
            self.chat_client = get_chat_client(llm_service, model="gpt-4o-mini")

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

    async def connect(self):
        if self.mcp_client:
            return

        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.server_script_path.parent)
        server_params = StdioServerParameters(
            command="uv",
            args=["run", "python", str(self.server_script_path)],
            env=env,
        )
        self._stdio_context = stdio_client(server_params)
        _stdio_read, _stdio_write = await self._stdio_context.__aenter__()
        self._session_context = ClientSession(_stdio_read, _stdio_write)
        self.mcp_client = await self._session_context.__aenter__()
        await self.mcp_client.initialize()

        self.tools = await self.get_tools()
        names = [tool["function"]["name"] for tool in self.tools]
        logger.info(f"Connected to MCP server with tools: {', '.join(names)}")

        await self.chat_client.connect()


    async def disconnect(self):
        if self._session_context:
            await self._session_context.__aexit__(None, None, None)
        if self._stdio_context:
            await self._stdio_context.__aexit__(None, None, None)
        self.mcp_client = None

    async def process_query(self, query: str) -> str:
        """Returns a response to a user query by getting a completion with tool calls.

        Supports multi-step tool chaining (e.g., find correct name -> get_speaker_by_name)
        by iteratively executing returned tool calls and re-querying the model
        with tool outputs until no more tool calls are requested or a safety
        limit is reached.
        """
        await self.connect()

        messages = [{"role": "user", "content": str(query)}]

        response = await self.chat_client.get_completion(messages, self.tools)

        max_tool_iterations = 5
        iterations = 0
        while response.get("tool_calls") and iterations < max_tool_iterations:
            iterations += 1

            for tool_call in response["tool_calls"]:
                tool_name = tool_call["function"]["name"]
                raw_args = tool_call["function"].get("arguments", {})
                if isinstance(raw_args, str):
                    try:
                        tool_args = json.loads(raw_args) if raw_args else {}
                    except Exception:
                        tool_args = {"__raw": raw_args}
                else:
                    tool_args = raw_args or {}

                logger.info(f"Calling tool {tool_name}({tool_args})")
                result = await self.mcp_client.call_tool(tool_name, tool_args)

                # Append a minimal trace plus the tool result back into the convo
                messages.extend([
                    {"role": "assistant", "content": f"Tool {tool_name} called."},
                    {"role": "user", "content": str(getattr(result, "content", str(result)))}
                ])

            response = await self.chat_client.get_completion(messages=messages, tools=self.tools)

        return response.get("text", "")

    async def run_chat_loop(self):
        print("Type your questions about speakers.")
        print("Type 'quit', 'exit', or 'q' to end the conversation.")
        while True:
            try:
                user_input = input("\nYou: ").strip()
                if user_input.lower() in ["quit", "exit", "q", ""]:
                    print("Goodbye!")
                    break
                response = await self.process_query(query=user_input)
                print("Response: ", response)
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break


async def amain():
    setup_logging_with_rich_logger()
    try:    
        client = MinimalMcpClient()
        await client.connect()
        await client.run_chat_loop()
    except Exception as e:
        logger.error(f"Error in chat: {e}")
    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(amain())
