import re
from dataclasses import dataclass

from cag_lab.generation.llm_client import complete
from cag_lab.retrieval import Chunk


@dataclass
class AnswerResult:
    answer: str
    sources: list[int]
    prompt_tokens: int
    completion_tokens: int


SYSTEM_PROMPT = (
    "Answer the question using ONLY the provided document chunks below. "
    "If the chunks do not contain the information needed to answer, "
    'say "I cannot answer this question based on the provided documents." '
    "and do NOT include a Sources line. "
    "If you do answer, end with a line in exactly this format: "
    "Sources: [1], [3] (listing only the chunk numbers you actually used)."
)


def _build_messages(question: str, chunks: list[Chunk]) -> list[dict[str, str]]:
    numbered = "\n\n".join(f"[{i}] {chunk.text}" for i, chunk in enumerate(chunks, 1))
    user = f"Document chunks:\n\n{numbered}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def _parse_sources(answer: str) -> list[int]:
    match = re.search(r"Sources:\s*(.+)", answer, re.IGNORECASE)
    if not match:
        return []
    numbers = re.findall(r"\d+", match.group(1))
    return [int(n) for n in numbers]


def generate_answer(
    question: str,
    chunks: list[Chunk],
    model: str = "gpt-4o-mini",
    api_base: str | None = None,
) -> AnswerResult:
    messages = _build_messages(question, chunks)
    result = complete(model, messages, api_base=api_base)
    sources = _parse_sources(result.answer)

    return AnswerResult(
        answer=result.answer,
        sources=sources,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
    )
