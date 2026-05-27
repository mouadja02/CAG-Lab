"""Semantic cache backed by Redis vector search.

Store (query embedding, answer, sources, model, timestamp). Lookup returns the
nearest cached entry if similarity >= threshold and the entry has not expired.
"""

import json
import struct
import time
import uuid
from dataclasses import dataclass

from redis import Redis
from redis.commands.search.field import NumericField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType

from cag_lab.embeddings import embed

_INDEX_NAME = "idx:semantic_cache"
_KEY_PREFIX = "cache:"
_DISTANCE_METRIC = "COSINE"
_HNSW_M = 16
_HNSW_EF = 200


@dataclass
class CacheResult:
    hit: bool
    answer: str | None = None
    sources: list[int] | None = None
    score: float | None = None
    cached_source_id: str | None = None


class SemanticCache:
    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        embedding_model: str = "text-embedding-3-small",
        embedding_dimensions: int = 512,
        similarity_threshold: float = 0.92,
        ttl_seconds: int = 3600,
        embed_api_base: str | None = None,
    ):
        self._redis = Redis(host=redis_host, port=redis_port, decode_responses=False)
        self._redis_decoded = Redis(
            host=redis_host, port=redis_port, decode_responses=True
        )
        self._embedding_model = embedding_model
        self._embedding_dimensions = embedding_dimensions
        self._similarity_threshold = similarity_threshold
        self._ttl_seconds = ttl_seconds
        self._embed_api_base = embed_api_base
        self._ensure_index()

    def _ensure_index(self) -> None:
        try:
            self._redis_decoded.ft(_INDEX_NAME).info()
        except Exception:
            schema = (
                NumericField("created_at"),
                VectorField(
                    "embedding",
                    "HNSW",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": self._embedding_dimensions,
                        "DISTANCE_METRIC": _DISTANCE_METRIC,
                        "M": _HNSW_M,
                        "EF_CONSTRUCTION": _HNSW_EF,
                    },
                ),
            )
            definition = IndexDefinition(
                prefix=[_KEY_PREFIX], index_type=IndexType.HASH
            )
            self._redis_decoded.ft(_INDEX_NAME).create_index(
                schema, definition=definition
            )

    def store(
        self,
        query: str,
        answer: str,
        sources: list[int],
        model: str,
        source_id: str,
    ) -> None:
        embedding = embed(
            query,
            model=self._embedding_model,
            dimensions=self._embedding_dimensions,
            api_base=self._embed_api_base,
        )
        key = f"{_KEY_PREFIX}{uuid.uuid4().hex}"
        now = int(time.time())
        mapping = {
            "embedding": self._vector_to_bytes(embedding),
            "answer": answer,
            "sources": json.dumps(sources),
            "model": model,
            "created_at": now,
            "question_text": query,
            "source_id": source_id,
        }
        self._redis.hset(key, mapping=mapping)
        self._redis.expire(key, self._ttl_seconds)

    def lookup(self, query: str) -> CacheResult:
        embedding = embed(
            query,
            model=self._embedding_model,
            dimensions=self._embedding_dimensions,
            api_base=self._embed_api_base,
        )
        vec_bytes = self._vector_to_bytes(embedding)

        try:
            results = self._redis_decoded.ft(_INDEX_NAME).search(
                f"*=>[KNN 1 @embedding $vec AS score]",
                query_params={"vec": vec_bytes},
            )
        except Exception:
            return CacheResult(hit=False)

        if not results.docs:
            return CacheResult(hit=False)

        doc = results.docs[0]
        distance = float(doc.score) if doc.score else 1.0
        similarity = 1.0 - distance

        if similarity < self._similarity_threshold:
            return CacheResult(hit=False)

        return CacheResult(
            hit=True,
            answer=doc.answer,
            sources=json.loads(doc.sources) if doc.sources else [],
            score=round(similarity, 4),
            cached_source_id=doc.source_id,
        )

    @staticmethod
    def _vector_to_bytes(vec: list[float]) -> bytes:
        return struct.pack(f"{len(vec)}f", *vec)
