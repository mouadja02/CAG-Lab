from dataclasses import dataclass

from pinecone import Pinecone

from cag_lab.config import get_settings
from cag_lab.embeddings import embed


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
        embed_api_base: str | None = None,
    ):
        settings = get_settings()
        pinecone_key = settings.pinecone_api_key
        if not pinecone_key:
            raise ValueError("PINECONE_API_KEY is required for Pinecone retrieval")
        self._pinecone = Pinecone(api_key=pinecone_key.get_secret_value())
        self._index = self._pinecone.Index(index_name)
        self._embedding_model = embedding_model
        self._embedding_dimensions = embedding_dimensions
        self._top_k = top_k
        self._embed_api_base = embed_api_base

    def retrieve(self, query: str) -> list[Chunk]:
        query_embedding = embed(
            query,
            model=self._embedding_model,
            dimensions=self._embedding_dimensions,
            api_base=self._embed_api_base,
        )

        results = self._index.query(
            vector=query_embedding,
            top_k=self._top_k,
            include_metadata=True,
        )

        chunks: list[Chunk] = []
        for match in results.matches:
            metadata = dict(match.metadata or {})
            text = metadata.pop("content", "")
            chunks.append(
                Chunk(
                    text=text,
                    score=match.score,
                    source=metadata,
                )
            )

        return chunks
