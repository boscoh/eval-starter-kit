import asyncio
import copy
import csv
import json
import logging
import os
from io import StringIO
from typing import List, Optional, Union

import numpy as np
from dotenv import load_dotenv
from path import Path
from pydash import py_

from chat_client import get_chat_client

logger = logging.getLogger(__name__)

data_dir = Path(__file__).parent / "data"


class RAGService:
    """Service class for Retrieval-Augmented Generation functionality."""

    def __init__(self, llm_service: Optional[str] = None):
        self.llm_service = llm_service or os.getenv("LLM_SERVICE", "openai").lower()

        models = {
            "openai": "text-embedding-3-small",
            "ollama": "nomic-embed-text",
            "bedrock": "amazon.titan-embed-text-v2:0",
        }

        model = models.get(self.llm_service)
        if model is None:
            raise ValueError(f"Unsupported service: {self.llm_service}")

        self.embed_client = get_chat_client(self.llm_service, model=model)
        self.embed_json = f"embeddings-{py_.kebab_case(model)}.json"
        logger.info(f"RAG LLM Service: '{llm_service}:{model}'")

        self.speakers_with_embeddings: Optional[List[dict]] = None
        self.speakers: Optional[List[dict]] = None

    async def __aenter__(self):
        await self.embed_client.connect()
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.embed_client.close()
        return False

    async def connect(self):
        logger.info("Initializing RAG service")
        if self.speakers_with_embeddings and self.speakers:
            return
        elif self.is_exists(self.embed_json):
            self.speakers_with_embeddings = json.loads(
                self.read_text_file(self.embed_json)
            )
            self.speakers = [
                self._strip_embeddings(speaker)
                for speaker in self.speakers_with_embeddings
            ]
        else:
            self.speakers_with_embeddings = await self._generate_speaker_embeddings()
            self.save_text_file(
                json.dumps(self.speakers_with_embeddings, indent=2), self.embed_json
            )
            self.speakers = [
                self._strip_embeddings(speaker)
                for speaker in self.speakers_with_embeddings
            ]
            logger.info(
                f"Embeddings saved to '{self._resolve_data_path(self.embed_json)}'"
            )

    def _resolve_data_path(self, file_path: Union[Path, str]) -> Path:
        path = Path(file_path)
        return path if path.isabs() else data_dir / path

    def is_exists(self, file_path: Union[Path, str]) -> bool:
        return self._resolve_data_path(file_path).exists()

    def read_text_file(self, file_path: Union[Path, str]) -> str:
        return self._resolve_data_path(file_path).read_text()

    def save_text_file(self, text: str, file_path: Union[Path, str]):
        self._resolve_data_path(file_path).write_text(text)

    async def _generate_speaker_embeddings(self) -> List[dict]:
        csv_text = self.read_text_file("2025-09-02-speaker-bio.csv")
        csv_reader = csv.DictReader(StringIO(csv_text))
        speakers = [dict(row) for row in csv_reader]

        logger.info(
            f"Generating embeddings for {len(speakers)} speakers with '{self.llm_service}:{self.embed_client.model}'"
        )
        result = []
        for speaker in speakers:
            speaker = py_.map_keys(speaker, lambda v, k: py_.snake_case(k))
            logger.info(f"Getting text embeddings for '{speaker['name']}'")

            abstract_text = speaker["final_abstract_max_150_words"]
            response = await self.embed_client.embed(abstract_text)
            speaker["abstract_embedding"] = response

            bio_text = speaker["bio_max_120_words"]
            response = await self.embed_client.embed(bio_text)
            speaker["bio_embedding"] = response

            result.append(speaker)
            logger.info(f"Finished text embeddings for '{speaker['name']}'")

        return result

    @staticmethod
    def cosine_distance(
        vec1: Union[List[float], np.ndarray], vec2: Union[List[float], np.ndarray]
    ) -> float:
        """Return cosine distance between 2 vectors, 0 being the best match."""
        a = np.asarray(vec1, dtype=np.float64)
        b = np.asarray(vec2, dtype=np.float64)

        if a.size != b.size:
            raise ValueError(
                f"Vectors must be of the same length (got {a.size} and {b.size})"
            )
        if a.size == 0:
            raise ValueError("Vectors cannot be empty")

        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            raise ValueError("One or both vectors have zero magnitude")

        cosine_similarity = dot_product / (norm_a * norm_b)
        return 1.0 - cosine_similarity

    @staticmethod
    def _strip_embeddings(speaker: dict) -> dict:
        clean_speaker = copy.deepcopy(speaker)
        for key in ["abstract_embedding", "bio_embedding"]:
            clean_speaker.pop(key, None)
        return clean_speaker

    async def get_best_speaker(self, query: str) -> Optional[dict]:
        await self.connect()
        embedding = await self.embed_client.embed(query)
        distances = []
        for speaker in self.speakers_with_embeddings:
            if "abstract_embedding" in speaker and "bio_embedding" in speaker:
                d1 = self.cosine_distance(embedding, speaker["abstract_embedding"])
                d2 = self.cosine_distance(embedding, speaker["bio_embedding"])
                distances.append((d1 + d2) / 2)
            else:
                distances.append(float("inf"))
        i_speaker_best = distances.index(min(distances))
        return self.speakers[i_speaker_best]

    async def get_speakers(self) -> List[dict]:
        await self.connect()
        return self.speakers


async def main():
    """Run embeddings generation."""
    from setup_logger import setup_logging_with_rich_logger

    setup_logging_with_rich_logger(level=logging.INFO)
    load_dotenv()
    async with RAGService() as service:
        await service.get_speakers()


if __name__ == "__main__":
    asyncio.run(main())
