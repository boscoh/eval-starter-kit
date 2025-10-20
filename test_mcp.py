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
from typing import Any, Dict, List, Optional

from setup_logger import setup_logging_with_rich_logger
setup_logging_with_rich_logger()

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from path import Path

from chat_client import IChatClient, get_chat_client


logger = logging.getLogger(__name__)


class MinimalMcpChatLoop:

    def __init__(self, llm_service: str = "bedrock"):
        self.mcp_client: Optional[ClientSession] = None
        self.server_script_path = Path(__file__).parent / "mcp_server.py"
        self._session_context: Optional[ClientSession] = None
        self._stdio_context: Optional[StdioServerParameters] = None

        self.tools: Optional[List[Dict[str, Any]]] = None

        self.llm_service = llm_service
        self.chat_client: IChatClient = None
        if self.llm_service == "bedrock":
            self.chat_client = get_chat_client(
                self.llm_service, model="anthropic.claude-3-sonnet-20240229-v1:0"
            )
        elif self.llm_service == "openai":
            self.chat_client = get_chat_client(self.llm_service, model="gpt-4o")
        else:
            raise ValueError(f"Unsupported LLM service: {self.llm_service}")

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
        env["LLM_SERVICE"] = self.llm_service
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
        logger.info(f"Connected Server to MCP tools: {', '.join(names)}")

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

        system_prompt = """You are a helpful assistant that can use tools to answer 
        questions about speakers. 

        IMPORTANT: You must use tools to gather ALL necessary information before 
        responding. Do NOT ask the user for clarification or additional 
        information. Instead:

        EFFICIENT TOOL USAGE STRATEGY:
        1. ANALYZE the query to identify what information you need
        2. PLAN the minimum necessary tool calls to get complete information
        3. EXECUTE tool calls efficiently - avoid duplicate calls with same parameters
        4. REASON through results to determine if you have enough information
        5. ONLY make additional tool calls if you're missing critical information
        6. SYNTHESIZE all gathered information into a comprehensive final answer

        TOOL USAGE GUIDELINES:
        - Use tools efficiently - don't repeat the same tool call with identical parameters
        - Get complete information in the fewest tool calls possible
        - Only make additional tool calls if you're missing essential information
        - ALWAYS provide a complete, detailed final answer that includes specific 
          information about the speakers you found through the tools

        Available tools can help you search, filter, and retrieve speaker 
        information. Use them efficiently to get complete information without 
        unnecessary repetition. Your final answer should be detailed and include 
        specific speaker information."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": str(query)}
        ]

        logger.info(f"Calling model with query='{query[:50]}' in {len(messages)} messages")
        response = await self.chat_client.get_completion(messages, self.tools)
        tool_calls = response.get("tool_calls")

        max_tool_iterations = 5
        iterations = 0
        executed_tool_calls = set()  # Track executed tool calls to prevent duplicates

        while tool_calls and iterations < max_tool_iterations:
            iterations += 1
            logger.info(f"Reasoning step {iterations} with {len(tool_calls)} tool calls")

            assistant_content = response.get("text", "")
            if assistant_content:
                messages.append({"role": "assistant", "content": assistant_content})

            tool_results = []
            for tool_call in tool_calls:
                tool_name = tool_call["function"]["name"]
                raw_args = tool_call["function"].get("arguments", {})
                if isinstance(raw_args, str):
                    try:
                        tool_args = json.loads(raw_args) if raw_args else {}
                    except Exception:
                        tool_args = {"__raw": raw_args}
                else:
                    tool_args = raw_args or {}

                # Create a unique identifier for this tool call
                tool_call_id = f"{tool_name}({json.dumps(tool_args, sort_keys=True)})"
                
                # Skip if we've already executed this exact tool call
                if tool_call_id in executed_tool_calls:
                    logger.info(f"Skipping duplicate tool call: {tool_call_id}")
                    tool_results.append(f"Tool {tool_name} already executed with same parameters - skipping duplicate")
                    continue
                
                executed_tool_calls.add(tool_call_id)

                try:
                    logger.info(f"Calling tool {tool_name}({tool_args})...")
                    result = await self.mcp_client.call_tool(tool_name, tool_args)
                    tool_result = str(getattr(result, "content", str(result)))
                    tool_results.append(f"Tool {tool_name} returned: {tool_result}")
                except Exception as e:
                    error_msg = f"Tool {tool_name} failed: {str(e)}"
                    tool_results.append(error_msg)
                    logger.error(error_msg)

            if tool_results:
                follow_up_prompt = """Based on the tool results above, analyze what you have:

                EFFICIENT ANALYSIS PROCESS:
                1. REVIEW all tool results to understand what information you now have
                2. DETERMINE if you have sufficient information to answer the user's question
                3. ONLY make additional tool calls if you're missing critical information
                4. AVOID repeating tool calls with the same parameters you've already used
                5. SYNTHESIZE all available information into a comprehensive answer

                REASONING GUIDELINES:
                - Assess whether you have enough information to provide a complete answer
                - Only use additional tools if you're missing essential information
                - Don't repeat tool calls with identical parameters
                - Focus on providing a complete answer with the information you have

                IMPORTANT: When you provide your final answer, make sure to 
                incorporate and reference the specific information you gathered 
                from the tools. Don't just mention that you used tools - include 
                the actual data and results in your response. Provide a complete, 
                detailed answer that directly addresses the user's question with 
                specific speaker information."""

                messages.append({
                    "role": "user",
                    "content": "\n".join(tool_results) + f"\n\n{follow_up_prompt}"
                })

            logger.info(f"Calling model with tool results in {len(messages)} messages")
            response = await self.chat_client.get_completion(
                messages=messages, tools=self.tools
            )
            tool_calls = response.get("tool_calls")

            if not tool_calls:
                logger.info("No more tool calls requested")

        if iterations >= max_tool_iterations:
            logger.warning(f"Reached maximum tool iterations ({max_tool_iterations})")

        return response.get("text", "")

    async def run_chat_loop(self):
        await self.connect()
        print("Type your questions about speakers.")
        print("Type 'quit', 'exit', or 'q' to end the conversation.")
        print("The assistant will use parallel processing and multi-step reasoning with tools to answer your questions.")
        while True:
            try:
                user_input = input("\nYou: ").strip()
                if user_input.lower() in ["quit", "exit", "q", ""]:
                    print("Goodbye!")
                    break
                response = await self.process_query(query=user_input)
                print(f"\nResponse: {response}")
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break


async def amain():
    try:
        client = MinimalMcpChatLoop("bedrock")
        await client.run_chat_loop()
    except Exception as e:
        logger.error(f"Error in chat: {e}")
    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(amain())
