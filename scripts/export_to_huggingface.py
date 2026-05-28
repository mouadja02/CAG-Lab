#!/usr/bin/env python3
"""Export Qdrant collection to a Parquet file and upload to HuggingFace Hub.

Usage:
    # Export to local parquet only
    python scripts/export_to_huggingface.py --export-only

    # Export and upload to HuggingFace Hub
    python scripts/export_to_huggingface.py --repo-id mouadja/aws-docs

    # Export from Pinecone directly (if Qdrant not available)
    python scripts/export_to_huggingface.py --source pinecone --repo-id mouadja/aws-docs

Requires:
    pip install pyarrow huggingface_hub
    Qdrant running locally (docker compose up qdrant) OR Pinecone API key in .env
"""

import argparse
import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dotenv import load_dotenv

load_dotenv()


def export_from_qdrant(
    collection_name: str = "aws-docs",
    host: str = "localhost",
    port: int = 6333,
    output_path: str = "data/aws_docs_vectors.parquet",
) -> str:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from qdrant_client import QdrantClient

    print(f"Connecting to Qdrant at {host}:{port}...")
    client = QdrantClient(host=host, port=port)

    info = client.get_collection(collection_name)
    total = info.points_count
    print(f"  Collection '{collection_name}': {total} vectors")

    ids, vectors, contents, file_paths, chunk_indices, pinecone_ids = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    seen_ids: set[str] = set()

    offset = None
    fetched = 0
    while True:
        result = client.scroll(
            collection_name=collection_name,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        points, offset = result
        if not points:
            break

        if offset is None:
            for p in points:
                pid = str(p.id)
                if pid in seen_ids:
                    continue
                seen_ids.add(pid)
                payload = p.payload or {}
                ids.append(pid)
                vectors.append(p.vector)
                contents.append(payload.get("content", ""))
                file_paths.append(payload.get("filePath", ""))
                chunk_indices.append(str(payload.get("chunkIndex", "")))
                pinecone_ids.append(payload.get("_pinecone_id", ""))
            fetched += len(points)
            print(
                f"\r  Scrolled {fetched} ({len(seen_ids)} unique)...",
                end="",
                flush=True,
            )
            break

        for p in points:
            pid = str(p.id)
            if pid in seen_ids:
                continue
            seen_ids.add(pid)
            payload = p.payload or {}
            ids.append(pid)
            vectors.append(p.vector)
            contents.append(payload.get("content", ""))
            file_paths.append(payload.get("filePath", ""))
            chunk_indices.append(str(payload.get("chunkIndex", "")))
            pinecone_ids.append(payload.get("_pinecone_id", ""))

        fetched += len(points)
        print(f"\r  Scrolled {fetched} ({len(seen_ids)} unique)...", end="", flush=True)

    print(f"\n  Total: {len(seen_ids)} unique vectors")

    table = pa.table(
        {
            "id": ids,
            "embedding": vectors,
            "content": contents,
            "filePath": file_paths,
            "chunkIndex": chunk_indices,
            "_pinecone_id": pinecone_ids,
        }
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(out), compression="zstd")

    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"  Exported to {out} ({size_mb:.1f} MB)")
    return str(out)


def export_from_pinecone(
    index_name: str = "aws-docs",
    output_path: str = "data/aws_docs_vectors.parquet",
) -> str:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from pinecone import Pinecone

    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("Error: PINECONE_API_KEY not set in .env")
        sys.exit(1)

    print(f"Connecting to Pinecone index '{index_name}'...")
    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)

    stats = index.describe_index_stats()
    total = stats.total_vector_count
    print(f"  Index: {total} vectors, {stats.dimension} dimensions")

    ids, vectors, contents, file_paths, chunk_indices, pinecone_ids = (
        [],
        [],
        [],
        [],
        [],
        [],
    )

    fetched = 0
    for batch in index.list(prefix=""):
        batch_ids = (
            [v.id for v in batch.vectors] if hasattr(batch, "vectors") else batch
        )
        if not batch_ids:
            continue

        fetch_response = index.fetch(ids=batch_ids)
        for vec_id, vec in fetch_response.vectors.items():
            meta = dict(vec.metadata or {})
            ids.append(str(uuid.uuid5(uuid.NAMESPACE_URL, vec_id)))
            vectors.append(vec.values)
            contents.append(meta.get("content", ""))
            file_paths.append(meta.get("filePath", ""))
            chunk_indices.append(str(meta.get("chunkIndex", "")))
            pinecone_ids.append(vec_id)

        fetched += len(batch_ids)
        print(f"\r  Fetched {fetched}/{total} vectors...", end="", flush=True)

    print(f"\n  Total: {fetched} vectors")

    table = pa.table(
        {
            "id": ids,
            "embedding": vectors,
            "content": contents,
            "filePath": file_paths,
            "chunkIndex": chunk_indices,
            "_pinecone_id": pinecone_ids,
        }
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(out), compression="zstd")

    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"  Exported to {out} ({size_mb:.1f} MB)")
    return str(out)


def upload_to_huggingface(
    parquet_path: str,
    repo_id: str,
    private: bool = False,
) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(
        repo_id=repo_id, repo_type="dataset", exist_ok=True, private=private
    )

    print(f"  Uploading {parquet_path} to hf.co/datasets/{repo_id}...")
    api.upload_file(
        path_or_fileobj=parquet_path,
        path_in_repo="data/train-00000-of-00001.parquet",
        repo_id=repo_id,
        repo_type="dataset",
    )

    card = f"""---
license: mit
task_categories:
  - text-retrieval
language:
  - en
size_categories:
  - 10K<n<100K
---

# CAG-Lab AWS Docs Vectors

89,221 document chunk embeddings from AWS public documentation.

- **Embedding model**: OpenAI `text-embedding-3-small` (512 dimensions)
- **Distance metric**: Cosine
- **Source**: AWS public documentation chunks

## Schema

| Column | Type | Description |
|--------|------|-------------|
| id | string | Deterministic UUID |
| embedding | list[float32] | 512-dim vector |
| content | string | Document chunk text |
| filePath | string | Original file path |
| chunkIndex | string | Chunk position |
| _pinecone_id | string | Original Pinecone vector ID |

## Usage

```python
from datasets import load_dataset

ds = load_dataset("{repo_id}")
```

Or use with CAG-Lab:

```bash
python scripts/setup_vectordb.py
```
"""
    api.upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )

    print(f"  Done! Dataset available at https://huggingface.co/datasets/{repo_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Export vector DB to Parquet and upload to HuggingFace Hub"
    )
    parser.add_argument(
        "--source",
        choices=["qdrant", "pinecone"],
        default="qdrant",
        help="Source vector database",
    )
    parser.add_argument(
        "--collection", default="aws-docs", help="Collection/index name"
    )
    parser.add_argument("--host", default="localhost", help="Qdrant host")
    parser.add_argument("--port", type=int, default=6333, help="Qdrant port")
    parser.add_argument(
        "--output",
        default="data/aws_docs_vectors.parquet",
        help="Output parquet path",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="HuggingFace repo ID (e.g. mouadja/aws-docs)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the HuggingFace dataset private",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Export to parquet without uploading",
    )
    args = parser.parse_args()

    if args.source == "qdrant":
        parquet_path = export_from_qdrant(
            collection_name=args.collection,
            host=args.host,
            port=args.port,
            output_path=args.output,
        )
    else:
        parquet_path = export_from_pinecone(
            index_name=args.collection,
            output_path=args.output,
        )

    if args.export_only:
        print("Export complete. Skipping upload (--export-only).")
        return

    if not args.repo_id:
        print("Error: --repo-id required for upload (or use --export-only)")
        sys.exit(1)

    upload_to_huggingface(parquet_path, args.repo_id, private=args.private)


if __name__ == "__main__":
    main()
