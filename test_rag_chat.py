#!/usr/bin/env python3
"""
Test client demonstrating RAG service integration for speaker queries.
"""

import asyncio
import logging
import os
from textwrap import dedent

from dotenv import load_dotenv

from chat_client import get_chat_client
from config import chat_models
from rag import RAGService
from setup_logger import setup_logging

load_dotenv()
setup_logging()

logger = logging.getLogger(__name__)


class SpeakerRagClient:
    def __init__(self, llm_service: str = "openai"):
        self.llm_service = llm_service
        self.rag_service: RAGService = None
        self.chat_client = None

    async def __aenter__(self):
        if self.rag_service:
            return

        self.rag_service = RAGService(llm_service=self.llm_service)
        await self.rag_service.__aenter__()

        model = chat_models.get(self.llm_service)
        self.chat_client = get_chat_client(self.llm_service, model=model)
        await self.chat_client.connect()

        logger.info(f"Connected to RAG service with {self.llm_service}")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        try:
            if self.chat_client:
                await self.chat_client.close()
        except Exception:
            pass
        try:
            if self.rag_service:
                await self.rag_service.__aexit__(None, None, None)
        except Exception:
            pass
        self.chat_client = None
        self.rag_service = None

        return False

    @staticmethod
    def format_embedding(embedding: list[float], n: int = 9) -> str:
        """Format embedding as string with length and first n numbers."""
        length = len(embedding)
        first_n = embedding[:n]
        numbers_str = " ".join([f"{num:.3f}" for num in first_n])
        return f"embedding({length})[{numbers_str}...]"

    async def process_query(self, query: str) -> str:
        """Process query using RAG service to find best speaker.

        Explicitly reproduces the get_best_speaker logic:
        1. Get embedding for the query
        2. Calculate distances to all speakers
        3. Return the speaker with the minimum distance
        """
        await self.__aenter__()

        embedding = await self.rag_service.embed_client.embed(query)
        logger.info(f"Query {self.format_embedding(embedding)}")

        distances = []
        for speaker in self.rag_service.speakers_with_embeddings:
            distance = self.rag_service.get_speaker_distance(embedding, speaker)
            distances.append(distance)
            logger.debug(f"Distance to {speaker['name']}: {distance:.4f}")

        i_speaker_best = distances.index(min(distances))
        best_distance = distances[i_speaker_best]

        speaker_with_embeddings = self.rag_service.speakers_with_embeddings[
            i_speaker_best
        ]
        abstract_embedding_str = self.format_embedding(
            speaker_with_embeddings["abstract_embedding"]
        )
        distance_to_abstract = self.rag_service.cosine_distance(
            embedding, speaker_with_embeddings["abstract_embedding"]
        )
        bio_embedding_str = self.format_embedding(
            speaker_with_embeddings["bio_embedding"]
        )
        distance_to_bio = self.rag_service.cosine_distance(
            embedding, speaker_with_embeddings["bio_embedding"]
        )

        logger.info(f"Speaker distances: {' '.join(f'{d:.3f}' for d in distances)}")
        logger.info(
            f"Best match of d={best_distance:.3f} to Speaker[{i_speaker_best}] "
        )
        logger.info(
            f"Bio[{i_speaker_best}] (d={distance_to_bio:.3f}) {bio_embedding_str}"
        )
        logger.info(
            f"Abstract[{i_speaker_best}] (d={distance_to_abstract:.3f}) {abstract_embedding_str}"
        )

        best_speaker = self.rag_service.speakers[i_speaker_best]

        system_prompt = """You are an expert at analyzing speaker-query matches. 
        Explain why a speaker is a good match for a given query by analyzing their 
        bio and presentation abstract. Be specific and concise."""

        user_prompt = dedent(f"""                         
Query: {query}

Best matching speaker: {best_speaker["name"]}

Bio: {best_speaker["bio_max_120_words"]}

Abstract: {best_speaker["final_abstract_max_150_words"]}

Explain in 2-3 sentences why this speaker is a good match for the query. 
Focus on specific aspects of their expertise or presentation that align with the query.""")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        logger.info("Calling LLM to explain speaker choice...")
        llm_result = await self.chat_client.get_completion(messages)
        explanation = llm_result.get("text", "No explanation available.")

        response = dedent(f"""             
## Speaker
{best_speaker["name"]}

## Bio
{best_speaker["bio_max_120_words"]}

## Abstract
{best_speaker["final_abstract_max_150_words"]}  

# Why this speaker matches your query
{explanation}""")
        return response


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
     async with SpeakerRagClient(service) as client:
         print("Type your questions about speakers.")
         print("Type 'quit', 'exit', or 'q' to end the conversation.")
         conversation_history = []
         while True:
             user_input = input("Query: ").strip()
             if user_input.lower() in ["quit", "exit", "q", ""]:
                 print("Goodbye!")
                 return
             conversation_history.append({"role": "user", "content": user_input})
             response = await client.process_query(query=user_input)
             print(f"\n# Best Match\n\n{response}")
             conversation_history.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    service = os.getenv("LLM_SERVICE", "openai")  # "bedrock", "ollama", "openai"
    try:
        asyncio.run(amain(service))
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
