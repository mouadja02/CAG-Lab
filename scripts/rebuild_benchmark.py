"""Rebuild benchmark dataset from Qdrant vector content using LLM-generated questions.

Samples random vectors from Qdrant, extracts the doc content, and uses an LLM
to generate benchmark questions that match the existing dataset schema.

Usage:
    python scripts/rebuild_benchmark.py --count 100 --output data/benchmark_sets/aws_docs_v2.jsonl
"""

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dotenv import load_dotenv

load_dotenv()

from qdrant_client import QdrantClient
from cag_lab.generation.llm_client import complete

_GENERATE_SYSTEM = """You are an AWS documentation expert creating benchmark questions for a RAG evaluation dataset.
Given a document chunk, generate ONE question that tests retrieval quality — the question must be answerable from the chunk content.

Rules:
- The question must require information FROM the chunk to answer correctly
- Include an "expected_answer" that cites specific details from the chunk
- Vary question types: factual, procedural, comparison, troubleshooting
- Vary difficulty: easy (direct lookup), medium (requires synthesis), hard (multi-step reasoning)
- "expected_sources" should be an empty list []
- Use "aws-docs" as domain
- Output ONLY valid JSON, one object per line, no markdown wrapping

Output format (one JSON object):
{"id": "aws-XXX", "question": "...", "expected_answer": "...", "expected_sources": [], "query_type": "factual|procedural|comparison|troubleshooting|multi_hop", "domain": "aws-docs", "difficulty": "easy|medium|hard"}"""


def sample_chunks_from_qdrant(
    collection: str = "aws-docs",
    count: int = 120,
    host: str = "localhost",
    port: int = 6333,
) -> list[dict]:
    """Sample random chunks from Qdrant by searching with random vectors."""
    print(f"Sampling {count} random chunks from Qdrant...")
    client = QdrantClient(host=host, port=port)
    chunks: list[dict] = []
    seen_ids: set[str] = set()

    while len(chunks) < count:
        random_vec = [random.uniform(-1, 1) for _ in range(512)]
        results = client.query_points(
            collection_name=collection,
            query=random_vec,
            limit=5,
            with_payload=True,
        )
        for point in results.points:
            pid = str(point.id)
            if pid in seen_ids:
                continue
            seen_ids.add(pid)
            payload = dict(point.payload or {})
            content = payload.get("content", "")
            if content and len(content) > 100:
                chunks.append({"content": content, "source": payload})
            if len(chunks) >= count:
                break

    print(f"  Sampled {len(chunks)} unique chunks")
    return chunks


def generate_question(chunk: dict, idx: int, total: int) -> dict | None:
    """Generate one benchmark question from a chunk using the LLM, with retries."""
    content = chunk["content"][:3000]
    user_msg = f"Document chunk:\n\n{content}\n\nGenerate one benchmark question from this content."

    messages = [
        {"role": "system", "content": _GENERATE_SYSTEM},
        {"role": "user", "content": user_msg},
    ]

    for attempt in range(5):
        try:
            result = complete(
                "openrouter/nvidia/nemotron-3-nano-30b-a3b:free",
                messages,
            )

            text = result.answer.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

            parsed = json.loads(text)
            parsed["id"] = f"aws-{idx:04d}"
            parsed.setdefault("expected_sources", [])
            parsed.setdefault("domain", "aws-docs")
            parsed.setdefault("query_type", "factual")
            parsed.setdefault("difficulty", "medium")
            return parsed
        except Exception as e:
            err = str(e)
            if "429" in err:
                wait = (attempt + 1) * 3
                time.sleep(wait)
                continue
            if attempt < 1:
                time.sleep(1)
                continue
            print(f"  Chunk {idx} failed: {err[:80]}")

    return None


def rebuild_dataset(
    output_path: str,
    count: int = 100,
    workers: int = 5,
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
) -> None:
    chunks = sample_chunks_from_qdrant(
        count=count + 20, host=qdrant_host, port=qdrant_port
    )

    questions: list[dict] = []
    print(f"\nGenerating questions with {workers} workers...")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(generate_question, chunk, i, len(chunks)): i
            for i, chunk in enumerate(chunks, 1)
        }
        for future in as_completed(futures):
            result = future.result()
            if result:
                questions.append(result)
                print(
                    f"  [{len(questions)}/{count}] Generated: {result['question'][:80]}..."
                )

    questions.sort(key=lambda q: q["id"])
    for i, q in enumerate(questions[:count]):
        q["id"] = f"aws-{i + 1:03d}"

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for q in questions[:count]:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"\nDone. Wrote {min(len(questions), count)} questions to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rebuild benchmark dataset from Qdrant"
    )
    parser.add_argument("--count", type=int, default=100, help="Number of questions")
    parser.add_argument(
        "--output", default="data/benchmark_sets/aws_docs_v3.jsonl", help="Output path"
    )
    parser.add_argument("--workers", type=int, default=5, help="Parallel workers")
    parser.add_argument("--host", default="localhost", help="Qdrant host")
    parser.add_argument("--port", type=int, default=6333, help="Qdrant REST port")
    args = parser.parse_args()

    rebuild_dataset(
        output_path=args.output,
        count=args.count,
        workers=args.workers,
        qdrant_host=args.host,
        qdrant_port=args.port,
    )
