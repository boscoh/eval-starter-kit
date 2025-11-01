#!/usr/bin/env python3
"""Chat client test script"""

import asyncio

from chat_client import get_chat_client


async def main(service):
    client = get_chat_client(service)
    await client.connect()
    print(f"Chat loop with {service}-{client.model}")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break
        messages = [{"role": "user", "content": user_input}]
        result = await client.get_completion(messages)
        print(f"\nResponse: {result.get('text', '')}")


if __name__ == "__main__":
    service = "openai"
    asyncio.run(main(service))
