#!/usr/bin/env python3
"""
Simple test script to test embeddings for Ollama, Bedrock, and OpenAI chat clients.

Usage:
    uv run test_embeddings.py

Requirements:
    - Set OPENAI_API_KEY in .env file for OpenAI testing
    - Configure AWS credentials for Bedrock testing
    - Have Ollama running locally for Ollama testing
"""

import asyncio
import sys

from chat_client import get_chat_client


async def test_embedding(client_type: str, test_text: str) -> None:
    """Test embedding generation for a specific client type."""
    print(f"\n{'=' * 50}")
    print(f"Testing {client_type.upper()} Embeddings")

    try:
        if client_type == "ollama":
            client = get_chat_client("ollama", model="llama3.2")
        elif client_type == "openai":
            client = get_chat_client("openai", model="gpt-4o")
        elif client_type == "bedrock":
            client = get_chat_client(
                "bedrock", embedding_model="amazon.titan-embed-text-v2:0"
            )
        else:
            raise ValueError(f"Unknown client type: {client_type}")

        async with client:
            embedding = await client.embed(test_text)
            if isinstance(embedding, list) and all(
                isinstance(x, (int, float)) for x in embedding
            ):
                print("Embedding format is correct (List[float])")
                print(f"Embedding dimensions: {len(embedding)}")
                print(f"First 5 values: {embedding[:5]}")
            else:
                print("ERROR: Embedding format is incorrect")

    except Exception as e:
        print(f"ERROR: {str(e)}")
        print(f"Error type: {type(e).__name__}")


async def main():
    """Main test function."""
    print("=" * 60)
    print("Embedding Test Suite")
    test_text = "The quick brown fox jumps over the lazy dog."
    print(f"Test text: '{test_text}'")
    providers = ["ollama", "openai", "bedrock"]
    for provider in providers:
        await test_embedding(provider, test_text)
    print(f"\n{'=' * 60}")
    print("Test Suite Complete")
    print("=" * 60)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        sys.exit(1)
