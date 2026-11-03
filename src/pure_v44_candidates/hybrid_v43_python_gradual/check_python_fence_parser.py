"""Read-only full-corpus match against the pinned pure-v44 Python fence grammar."""
from __future__ import annotations

import json
from pathlib import Path
import re

from source_pools import CONVERTED_ROWS, PINNED_CONVERTED_SHA, file_sha


PURE_V44_PYTHON_BLOCK = re.compile(r'(?ms)^```python[ \t]*\r?\n(.*?)^```[ \t]*(?:\r?\n|$)')


def check(path: Path) -> dict:
    if file_sha(path) != PINNED_CONVERTED_SHA:
        raise ValueError("Pinned converted corpus changed")
    rows = blocks = bad = 0
    examples = []
    with path.open(encoding="utf-8") as source:
        for raw in source:
            row = json.loads(raw)
            count = row["target"].count("```python")
            parsed = len(PURE_V44_PYTHON_BLOCK.findall(row["target"]))
            rows += 1
            blocks += parsed
            if not count or parsed != count:
                bad += 1
                if len(examples) < 5:
                    examples.append({"source_line": row["source_ref"]["line"],
                                     "fences": count, "parsed": parsed})
    if rows != CONVERTED_ROWS:
        raise ValueError("Converted corpus row count differs")
    return {"rows": rows, "parsed_python_blocks": blocks,
            "unparseable_rows": bad, "examples": examples,
            "training_updates": 0}


if __name__ == "__main__":
    path = Path(__file__).parents[1] / "v43_alu_to_python" / "provenance_full_final" / "converted_with_source_refs.jsonl"
    print(json.dumps(check(path), sort_keys=True))
