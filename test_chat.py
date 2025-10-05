#!/usr/bin/env python3
"""
MCP Client for xConf Assistant - Speaker Recommendation System

This client connects to the MCP server using STDIO and provides
a simple interface to interact with the speaker recommendation tools.
"""

import asyncio

from chat_client import get_chat_client
from setup_logger import setup_logging_with_rich_logger

setup_logging_with_rich_logger()

async def main(service):
    client = get_chat_client(service)
    print(f"Chat loop with {service}-{client.model}")
    try:
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ["quit", "exit"]:
                print("Goodbye!")
                break
            messages = [{"role": "user", "content": user_input}]
            result = await client.get_completion(messages)
            print(f"\nResponse: {result.get("text", "")}")
    except Exception as ex:
        print(f"Error: {ex}")
    finally:
        await client.close()

if __name__ == "__main__":
    service = "openai" # "bedrock" "ollama" or "openai"
    asyncio.run(main(service)) 
