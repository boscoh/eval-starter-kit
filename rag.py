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
from setup_logger import setup_logging_with_rich_logger

logger = logging.getLogger(__name__)

data_dir = Path(__file__).parent / "data"

class RAGService:
    """Service class for Retrieval-Augmented Generation functionality."""
    
    def __init__(self, llm_service: Optional[str] = None):
        self.llm_service = llm_service or os.getenv("LLM_SERVICE", "openai").lower()
        if self.llm_service == "openai":
            model = "text-embedding-3-small"
        elif self.llm_service == "ollama":
            model = "nomic-embed-text"
        elif self.llm_service == "bedrock":
            model = "amazon.titan-embed-text-v2:0"
        else:
            raise ValueError(f"Unsupported service: {self.llm_service}")
       
        self.embed_client = get_chat_client(self.llm_service, model=model)
        self.embed_json = data_dir / f"embeddings-{py_.kebab_case(model)}.json"
    
        # to be created in ainit
        self.speakers: Optional[List[dict]] = None
        self.clean_speakers: Optional[List[dict]] = None

    async def ainit(self):
        if self.speakers and self.clean_speakers:
            return
        elif self.is_exists(self.embed_json):
            self.speakers = json.loads(self.read_text_file(self.embed_json))
            self.clean_speakers = [self._strip_embeddings(speaker) for speaker in self.speakers]
        else:
            self.speakers = await self._generate_speaker_embeddings()
            self.save_text_file(json.dumps(self.speakers, indent=2), self.embed_json)
            self.clean_speakers = [self._strip_embeddings(speaker) for speaker in self.speakers]
            logger.info(f"Embeddings saved to '{self.embed_json}'")

    @staticmethod
    def is_exists(file_path: Path) -> str:
        return file_path.exists()

    @staticmethod
    def read_text_file(file_path: Path) -> str:
        return file_path.read_text()
    
    @staticmethod
    def save_text_file(text: str, file_path: Path):
        file_path.write_text(text)
    
    async def _generate_speaker_embeddings(self) -> List[dict]:
        csv_text = self.read_text_file(data_dir / "2025-09-02-speaker-bio.csv")
        csv_reader = csv.DictReader(StringIO(csv_text))
        speakers = [dict(row) for row in csv_reader]

        logger.info(f"Generating embeddings for '{self.llm_service}:{self.embed_client.model}'")
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
    
    async def get_best_speaker(self, query: str, speakers: Optional[List[dict]] = None) -> Optional[dict]:
        await self.ainit()
       
        query_embedding = await self.embed_client.embed(query)

        speaker_pairs = []
        for i, speaker in enumerate(self.speakers):
            if "abstract_embedding" not in speaker or "bio_embedding" not in speaker:
                logger.warning(
                    f"Speaker {speaker.get('name', 'Unknown')} missing embedding data"
                )
                distance = float("inf")
            else:
                abstract_distance = self.cosine_distance(
                    query_embedding, speaker["abstract_embedding"]
                )
                bio_distance = self.cosine_distance(query_embedding, speaker["bio_embedding"])
                distance = (abstract_distance + bio_distance) / 2
            speaker_pairs.append((i, distance))
        
        best_pair = min(speaker_pairs, key=lambda x: x[1])
        best_index = best_pair[0]
    
        return self.clean_speakers[best_index]
    
    async def get_speakers(self) -> List[dict]:
        """Get speaker storage, generating if not found."""
        await self.ainit()
        return self.clean_speakers


async def main():
    """Run embeddings generation."""
    setup_logging_with_rich_logger(level=logging.INFO)
    load_dotenv()
    service = RAGService()
    await service.ainit()


if __name__ == "__main__":
    asyncio.run(main())
