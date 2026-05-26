import json
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError


class QueryType(str, Enum):
    factual = "factual"
    procedural = "procedural"
    troubleshooting = "troubleshooting"
    comparison = "comparison"
    multi_hop = "multi_hop"
    paraphrase_repeated = "paraphrase_repeated"
    near_duplicate = "near_duplicate"


class Difficulty(str, Enum):
    easy = "easy"
    medium = "medium"
    hard = "hard"


class BenchmarkQuestion(BaseModel):
    id: str
    question: str
    expected_answer: str
    expected_sources: list[str] = Field(default_factory=list)
    query_type: QueryType
    domain: str
    difficulty: Difficulty


def load_dataset(path: str | Path) -> list[BenchmarkQuestion]:
    path = Path(path)
    if not path.suffix == ".jsonl":
        raise ValueError(f"Expected .jsonl file, got {path.suffix}")

    questions: list[BenchmarkQuestion] = []
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                data = json.loads(stripped)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON on line {line_num} of {path}: {e}"
                ) from e
            try:
                questions.append(BenchmarkQuestion.model_validate(data))
            except ValidationError as e:
                raise ValueError(
                    f"Validation error on line {line_num} of {path}: {e}"
                ) from e

    return questions
