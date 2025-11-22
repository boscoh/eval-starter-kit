"""Default models for each service."""

chat_models = {
    "bedrock": "amazon.nova-pro-v1:0",
    "openai": "gpt-4o",
    "ollama": "llama3.2",
}

embed_models = {
    "openai": "text-embedding-3-small",
    "ollama": "nomic-embed-text",
    "bedrock": "amazon.titan-embed-text-v2:0",
}
