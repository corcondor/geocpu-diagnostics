"""Audit pinned hybrid teaching text and fixed diagnostic prompt overlap.

Reads only the three inputs.  The JSON report is the sole output.  Surface
counts describe text structure; they do not certify semantic grounding.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re
import unicodedata
import zipfile


PINS = {
    "v43_zip": "0102dc9b01b4b795c3123b18cefc7a8734e47a06dd85292af464d3232704d198",
    "converted": "53387b2987df02d91b51b4afc8ab2369cc0e53b9e4bdd0cfd480c3586de7f3db",
    "cases": "4032a2c17de2ce56a7758cd35c16db2c2b357ce76b394522411e86c0aa682b45",
}
TRAIN_MEMBER = "data/train_v43_knowledge_unified.jsonl"
JAPANESE = re.compile(r"[ぁ-んァ-ン一-龯]")
CODE_FENCE = re.compile(r"(?m)^```python[ \t]*$")
NUMERIC_ONLY = re.compile(r"\s*\d+\s*\Z")


def file_sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_question(value: str) -> str:
    """The question key used by the hybrid source-pool quarantine."""
    text = unicodedata.normalize("NFKC", value).strip()
    text = re.sub(r"^(?:ユーザー|user)\s*[:：]\s*", "", text, flags=re.I)
    text = re.sub(r"\n\s*(?:アシスタント|assistant)\s*[:：]\s*$", "", text, flags=re.I)
    text = re.sub(r"^\s*【分野:[^\n]*】\s*", "", text)
    text = re.sub(r"(?m)^\s*分野\s*[:：][^\n]*\n?", "", text)
    text = re.sub(r"\s*(?:正解|回答|答え)\s*[:：]\s*$", "", text)
    return re.sub(r"\s+", "", text)


def audit(v43_zip: Path, converted: Path, cases: Path) -> dict:
    paths = {"v43_zip": v43_zip, "converted": converted, "cases": cases}
    observed = {name: file_sha(path) for name, path in paths.items()}
    if observed != PINS:
        raise ValueError(f"Pinned input SHA-256 differs: {observed}")

    family = defaultdict(Counter)
    total = Counter()
    with converted.open(encoding="utf-8") as stream:
        for raw in stream:
            row = json.loads(raw)
            target = row["target"]
            spans = row["target_segments"]
            if not spans:
                raise ValueError("Converted row has no observation segment")
            suffix = target[spans[-1]["end"]:]
            code_count = len(CODE_FENCE.findall(target))
            values = {
                "rows": 1,
                "python_code_blocks": code_count,
                "tool_observation_segments": len(spans),
                "all_segments_are_tool": int(all(s.get("role") == "tool" for s in spans)),
                "code_count_matches_observations": int(code_count == len(spans)),
                "two_or_more_tool_steps": int(len(spans) >= 2),
                "target_contains_japanese": int(bool(JAPANESE.search(target))),
                "final_suffix_contains_japanese": int(bool(JAPANESE.search(suffix))),
                "final_suffix_has_answer_marker": int("答え:" in suffix or "答え：" in suffix),
                "final_suffix_numeric_only": int(bool(NUMERIC_ONLY.fullmatch(suffix))),
            }
            family[row["original_source"]].update(values)
            total.update(values)
    if total["rows"] != 24_836:
        raise ValueError("Pinned conversion row count differs")

    fixed = [json.loads(raw) for raw in cases.read_text(encoding="utf-8").splitlines()]
    if len(fixed) != 44 or len({row["id"] for row in fixed}) != 44:
        raise ValueError("Fixed diagnostic cases are incomplete")
    fixed_keys = {canonical_question(row["prompt"]) for row in fixed}
    original_matches = 0
    original_rows = 0
    with zipfile.ZipFile(v43_zip) as archive, archive.open(TRAIN_MEMBER) as stream:
        for raw in stream:
            row = json.loads(raw)
            original_rows += 1
            original_matches += canonical_question(row.get("prompt", "")) in fixed_keys
    if original_rows != 268_032:
        raise ValueError("Pinned original corpus row count differs")

    return {
        "schema": "geocpu_hybrid_semantic_readonly_audit_v1",
        "inputs": {name: {"path": str(path), "sha256": observed[name]} for name, path in paths.items()},
        "converted_total": dict(total),
        "converted_by_original_family": {name: dict(counts) for name, counts in sorted(family.items())},
        "fixed_diagnostic": {
            "cases": len(fixed),
            "original_rows_scanned": original_rows,
            "canonical_prompt_matches_in_original_rows": original_matches,
        },
        "interpretation_limit": (
            "Japanese characters and answer markers are surface checks. "
            "They do not establish a grounded explanation; the eight SWE rows "
            "have numeric-only final suffixes."
        ),
        "training_updates": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("v43_zip", "converted", "cases", "output"):
        parser.add_argument("--" + name.replace("_", "-"), required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.v43_zip, args.converted, args.cases)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps({"rows": result["converted_total"]["rows"],
                      "fixed_prompt_matches": result["fixed_diagnostic"]["canonical_prompt_matches_in_original_rows"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
