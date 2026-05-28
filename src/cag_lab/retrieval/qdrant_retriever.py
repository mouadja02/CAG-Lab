from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, HnswConfigDiff, SearchParams, VectorParams

from cag_lab.embeddings import embed

_HNSW_M = 32
_HNSW_EF_CONSTRUCT = 200
_HNSW_EF_SEARCH = 128


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
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        embed_api_base: str | None = None,
    ):
        self._qdrant = QdrantClient(host=qdrant_host, port=qdrant_port)
        self._collection_name = index_name
        self._embedding_model = embedding_model
        self._embedding_dimensions = embedding_dimensions
        self._top_k = top_k
        self._embed_api_base = embed_api_base

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        collections = [c.name for c in self._qdrant.get_collections().collections]
        if self._collection_name not in collections:
            self._qdrant.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(
                    size=self._embedding_dimensions,
                    distance=Distance.COSINE,
                ),
                hnsw_config=HnswConfigDiff(
                    m=_HNSW_M,
                    ef_construct=_HNSW_EF_CONSTRUCT,
                ),
            )
        else:
            self._qdrant.update_collection(
                collection_name=self._collection_name,
                hnsw_config=HnswConfigDiff(
                    m=_HNSW_M,
                    ef_construct=_HNSW_EF_CONSTRUCT,
                ),
            )

    def retrieve(self, query: str) -> list[Chunk]:
        query_embedding = embed(
            query,
            model=self._embedding_model,
            dimensions=self._embedding_dimensions,
            api_base=self._embed_api_base,
        )

        results = self._qdrant.query_points(
            collection_name=self._collection_name,
            query=query_embedding,
            limit=self._top_k,
            with_payload=True,
            search_params=SearchParams(hnsw_ef=_HNSW_EF_SEARCH),
        )

        chunks: list[Chunk] = []
        for point in results.points:
            payload = dict(point.payload or {})
            text = payload.pop("content", "")
            chunks.append(
                Chunk(
                    text=text,
                    score=point.score,
                    source=payload,
                )
            )

        return chunks
