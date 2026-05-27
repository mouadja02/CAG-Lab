"""Migrate vectors and metadata from a Pinecone index to a local Qdrant collection.

Usage:
    python scripts/migrate_pinecone_to_qdrant.py --index aws-docs --force

Requires:
    - Qdrant running locally (docker compose up qdrant)
    - .env with PINECONE_API_KEY set
"""

import argparse
import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dotenv import load_dotenv

load_dotenv()

from pinecone import Pinecone
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


def migrate(
    pinecone_index_name: str = "aws-docs",
    qdrant_collection_name: str | None = None,
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    force: bool = False,
) -> None:
    if qdrant_collection_name is None:
        qdrant_collection_name = pinecone_index_name

    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("Error: PINECONE_API_KEY not set in .env")
        sys.exit(1)

    print(f"Connecting to Pinecone index '{pinecone_index_name}'...")
    pc = Pinecone(api_key=api_key)
    index = pc.Index(pinecone_index_name)

    stats = index.describe_index_stats()
    total_vectors = stats.total_vector_count
    dimension = stats.dimension
    print(f"  Index stats: {total_vectors} vectors, {dimension} dimensions")

    print(f"Connecting to Qdrant at {qdrant_host}:{qdrant_port}...")
    qdrant = QdrantClient(host=qdrant_host, port=qdrant_port)

    existing = [c.name for c in qdrant.get_collections().collections]
    if qdrant_collection_name in existing:
        if force:
            qdrant.delete_collection(qdrant_collection_name)
        else:
            print(
                f"  Warning: Qdrant collection '{qdrant_collection_name}' already exists."
            )
            choice = input("  Recreate it? [y/N] ").strip().lower()
            if choice == "y":
                qdrant.delete_collection(qdrant_collection_name)
            else:
                print("  Aborted.")
                return

    qdrant.create_collection(
        collection_name=qdrant_collection_name,
        vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
    )
    print(f"  Created collection '{qdrant_collection_name}'")

    print("Fetching vectors...")
    fetched = 0
    for batch in index.list(prefix=""):
        ids = [v.id for v in batch.vectors] if hasattr(batch, "vectors") else batch
        if not ids:
            continue

        fetch_response = index.fetch(ids=ids)
        points: list[PointStruct] = []

        for vec_id, vec in fetch_response.vectors.items():
            payload = dict(vec.metadata or {})
            payload["_pinecone_id"] = vec_id
            points.append(
                PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, vec_id)),
                    vector=vec.values,
                    payload=payload,
                )
            )

        if points:
            qdrant.upsert(
                collection_name=qdrant_collection_name,
                points=points,
            )

        fetched += len(points)
        print(f"\r  Migrated {fetched}/{total_vectors} vectors...", end="")

    print(f"\nDone. Migrated {fetched} vectors to '{qdrant_collection_name}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate Pinecone index to Qdrant")
    parser.add_argument("--index", default="aws-docs", help="Pinecone index name")
    parser.add_argument(
        "--collection",
        default=None,
        help="Qdrant collection name (default: same as index)",
    )
    parser.add_argument("--host", default="localhost", help="Qdrant host")
    parser.add_argument("--port", type=int, default=6333, help="Qdrant REST port")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing collection without prompt",
    )
    args = parser.parse_args()

    migrate(
        pinecone_index_name=args.index,
        qdrant_collection_name=args.collection,
        qdrant_host=args.host,
        qdrant_port=args.port,
        force=args.force,
    )
