#!/usr/bin/env python3
"""Download the AWS docs vector database from HuggingFace and load into Qdrant.

Usage:
    # Default: download from HuggingFace and load into local Qdrant
    python scripts/setup_vectordb.py

    # Use a local parquet file instead
    python scripts/setup_vectordb.py --from-file data/aws_docs_vectors.parquet

    # Custom HuggingFace repo
    python scripts/setup_vectordb.py --repo-id mouadja/aws-docs

Requires:
    pip install pyarrow huggingface_hub
    Qdrant running locally: docker compose up -d qdrant
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

DEFAULT_REPO_ID = "mouadja/aws-docs"
DEFAULT_COLLECTION = "aws-docs"
EMBEDDING_DIM = 512


def download_from_huggingface(repo_id: str, cache_dir: str = "data") -> str:
    from huggingface_hub import hf_hub_download

    print(f"Downloading dataset from hf.co/datasets/{repo_id}...")
    path = hf_hub_download(
        repo_id=repo_id,
        filename="data/train-00000-of-00001.parquet",
        repo_type="dataset",
        cache_dir=cache_dir,
    )
    print(f"  Downloaded to {path}")
    return path


def load_into_qdrant(
    parquet_path: str,
    collection_name: str = DEFAULT_COLLECTION,
    host: str = "localhost",
    port: int = 6333,
    force: bool = False,
) -> None:
    import pyarrow.parquet as pq
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, HnswConfigDiff, PointStruct, VectorParams

    print(f"Reading {parquet_path}...")
    table = pq.read_table(parquet_path)
    n = len(table)
    print(f"  {n} vectors, {len(table.column_names)} columns")

    print(f"Connecting to Qdrant at {host}:{port}...")
    client = QdrantClient(host=host, port=port)

    existing = [c.name for c in client.get_collections().collections]
    if collection_name in existing:
        if force:
            client.delete_collection(collection_name)
        else:
            choice = (
                input(f"  Collection '{collection_name}' exists. Recreate? [y/N] ")
                .strip()
                .lower()
            )
            if choice != "y":
                print("  Aborted.")
                return
            client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        hnsw_config=HnswConfigDiff(m=32, ef_construct=200),
    )
    print(f"  Created collection '{collection_name}'")

    ids = table["id"].to_pylist()
    embeddings = table["embedding"].to_pylist()
    contents = table["content"].to_pylist()
    file_paths = table["filePath"].to_pylist()
    chunk_indices = table["chunkIndex"].to_pylist()
    pinecone_ids = table["_pinecone_id"].to_pylist()

    batch_size = 256
    upserted = 0

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        points = []
        for i in range(start, end):
            payload = {
                "content": contents[i],
                "filePath": file_paths[i],
                "chunkIndex": chunk_indices[i],
            }
            if pinecone_ids[i]:
                payload["_pinecone_id"] = pinecone_ids[i]
            points.append(
                PointStruct(
                    id=ids[i],
                    vector=embeddings[i],
                    payload=payload,
                )
            )

        client.upsert(collection_name=collection_name, points=points)
        upserted += len(points)
        print(f"\r  Loaded {upserted}/{n} vectors...", end="", flush=True)

    print(f"\nDone. {upserted} vectors loaded into '{collection_name}'.")


def main():
    parser = argparse.ArgumentParser(
        description="Download AWS docs vectors and load into Qdrant"
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"HuggingFace dataset repo ID (default: {DEFAULT_REPO_ID})",
    )
    parser.add_argument(
        "--from-file",
        default=None,
        help="Load from a local parquet file instead of downloading",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help=f"Qdrant collection name (default: {DEFAULT_COLLECTION})",
    )
    parser.add_argument("--host", default="localhost", help="Qdrant host")
    parser.add_argument("--port", type=int, default=6333, help="Qdrant port")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing collection without prompt",
    )
    args = parser.parse_args()

    if args.from_file:
        parquet_path = args.from_file
    else:
        parquet_path = download_from_huggingface(args.repo_id)

    load_into_qdrant(
        parquet_path=parquet_path,
        collection_name=args.collection,
        host=args.host,
        port=args.port,
        force=args.force,
    )


if __name__ == "__main__":
    main()
