#!/usr/bin/env python3
"""
Interactive chat loop with LLM providers.
"""

import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

from starteval.chat_client import get_chat_client

load_dotenv()

config_path = Path(__file__).parent / "config.json"
with open(config_path) as f:
    config = json.load(f)
    chat_models = config["chat_models"]


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
    async with get_chat_client(service, model=chat_models[service]) as client:
        print(f"Chat loop with {service}-{client.model}")
        conversation_history = []
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ["quit", "exit"]:
                print("Goodbye!")
                break
            conversation_history.append({"role": "user", "content": user_input})
            messages = [{"role": "user", "content": user_input}]
            result = await client.get_completion(messages)
            response_text = result.get("text", "")
            print(f"\nResponse: {response_text}")
            conversation_history.append({"role": "assistant", "content": response_text})


def main(service: str = "openai"):
    try:
        asyncio.run(amain(service))
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")


if __name__ == "__main__":
    main()
