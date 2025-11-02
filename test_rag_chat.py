#!/usr/bin/env python3
"""
Test client demonstrating RAG service integration for speaker queries.
"""

import asyncio
import logging
from textwrap import dedent

from setup_logger import setup_logging_with_rich_logger


from rag import RAGService

setup_logging_with_rich_logger()

logger = logging.getLogger(__name__)


class SpeakerRagClient:
    def __init__(self, llm_service: str = "openai"):
        self.llm_service = llm_service
        self.rag_service: RAGService = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()
        return False

    async def connect(self):
        if self.rag_service:
            return

        self.rag_service = RAGService(llm_service=self.llm_service)
        await self.rag_service.__aenter__()
        logger.info(f"Connected to RAG service with {self.llm_service}")

    async def disconnect(self):
        try:
            if self.rag_service:
                await self.rag_service.__aexit__(None, None, None)
        except Exception:
            pass
        self.rag_service = None

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
        await self.connect()

        logger.info(f"Query: {query}")

        embedding = await self.rag_service.embed_client.embed(query)
        logger.info(f"Query: {self.format_embedding(embedding)}")

        distances = []
        for speaker in self.rag_service.speakers_with_embeddings:
            distance = self.rag_service.get_speaker_distance(embedding, speaker)
            distances.append(distance)
            logger.debug(f"Distance to {speaker['name']}: {distance:.4f}")

        i_speaker_best = distances.index(min(distances))
        best_distance = distances[i_speaker_best]

        speaker_with_embeddings = self.rag_service.speakers_with_embeddings[i_speaker_best]
        abstract_embedding_str = self.format_embedding(speaker_with_embeddings['abstract_embedding'])
        distance_to_abstract = self.rag_service.cosine_distance(embedding, speaker_with_embeddings['abstract_embedding'])
        bio_embedding_str = self.format_embedding(speaker_with_embeddings['bio_embedding'])
        distance_to_bio = self.rag_service.cosine_distance(embedding, speaker_with_embeddings['bio_embedding'])

        logger.info(f"Speaker distances: {' '.join(f'{d:.3f}' for d in distances)}")
        logger.info( f"Best match to Speaker\\[{i_speaker_best}] d={best_distance:.3f}")
        logger.info( f"Bio\\[{i_speaker_best}]: d={distance_to_bio:.3f} {bio_embedding_str}" )
        logger.info( f"Abstract\\[{i_speaker_best}]: d={distance_to_abstract:.3f} {abstract_embedding_str}" )

        best_speaker = self.rag_service.speakers[i_speaker_best]
        response = dedent(f"""\
Best speaker: {best_speaker['name']}
    
Bio: {best_speaker['bio_max_120_words']}

Abstract: {best_speaker['final_abstract_max_150_words']}
   
""")
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
        while True:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ["quit", "exit", "q", ""]:
                print("Goodbye!")
                return
            response = await client.process_query(query=user_input)
            print(f"\nResponse:\n{response}")


if __name__ == "__main__":
    try:
        asyncio.run(amain("openai"))
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")

