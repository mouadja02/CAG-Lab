from dataclasses import dataclass

from openai import OpenAI
from pinecone import Pinecone

from cag_lab.config import get_settings


@dataclass
class Chunk:
    text: str
    score: float
    source: dict


class Retriever:
    def __init__(
        self,
        index_name: str,
        embedding_model: str = "text-embedding-3-small",
        embedding_dimensions: int = 512,
        top_k: int = 5,
    ):
        settings = get_settings()
        self._pinecone = Pinecone(
            api_key=settings.pinecone_api_key.get_secret_value()
        )
        self._openai = OpenAI(
            api_key=settings.openai_api_key.get_secret_value()
        )
        self._index = self._pinecone.Index(index_name)
        self._embedding_model = embedding_model
        self._embedding_dimensions = embedding_dimensions
        self._top_k = top_k

    def retrieve(self, query: str) -> list[Chunk]:
        response = self._openai.embeddings.create(
            model=self._embedding_model,
            input=query,
            dimensions=self._embedding_dimensions,
        )
        query_embedding = response.data[0].embedding

        results = self._index.query(
            vector=query_embedding,
            top_k=self._top_k,
            include_metadata=True,
        )

        chunks: list[Chunk] = []
        for match in results.matches:
            metadata = match.metadata or {}
            text = metadata.pop("content", "")
            chunks.append(
                Chunk(
                    text=text,
                    score=match.score,
                    source=metadata,
                )
            )

        return chunks
