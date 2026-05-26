import tempfile
from pathlib import Path

import pytest

from cag_lab.benchmark.dataset import load_dataset


def test_rejects_malformed_line():
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as f:
        f.write('{"id": "ok", "question": "q", "expected_answer": "a", "query_type": "factual", "domain": "d", "difficulty": "easy"}\n')
        f.write('not json at all\n')
        f.write('{"id": "also-ok", "question": "q2", "expected_answer": "a2", "query_type": "procedural", "domain": "d2", "difficulty": "hard"}\n')
        tmp = Path(f.name)

    try:
        with pytest.raises(ValueError, match="Invalid JSON on line 2"):
            load_dataset(tmp)
    finally:
        tmp.unlink()
