"""Freeze small, source-pinned *interim* diagnostics, never training rows."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import zipfile

from source_pools import PINNED_V43_ZIP_SHA, REQUIRED_EVAL_SHA, file_sha

ACADEMIC_CATEGORIES = ("japanese_linguistics", "history", "law", "medicine_health",
                       "philosophy_ethics", "physics", "economics", "computer_science")
EXPLANATION_INSTRUCTION = "\n選択肢を一つ選び、本文や知識を根拠に日本語で理由を説明してください。\n回答:"


def _load(archive, member):
    import hashlib
    raw = archive.read(member)
    if hashlib.sha256(raw).hexdigest() != REQUIRED_EVAL_SHA[member]:
        raise ValueError("Pinned diagnostic source changed: " + member)
    return [json.loads(line) for line in raw.splitlines()]


def build(v43_zip: Path):
    if file_sha(v43_zip) != PINNED_V43_ZIP_SHA:
        raise ValueError("Pinned original v43 ZIP changed")
    cases = []
    with zipfile.ZipFile(v43_zip) as archive:
        swe = _load(archive, "data/eval_swe_step_control_heldout.jsonl")
        by_subject = defaultdict(list)
        for row in swe:
            by_subject[row["subject"]].append(row)
        if len(by_subject) != 5 or any(len(rows) != 40 for rows in by_subject.values()):
            raise ValueError("Pinned SWE200 family distribution changed")
        for subject in sorted(by_subject):
            for row in by_subject[subject][:4]:
                cases.append({"id": row["id"], "group": "swe20", "source_family": subject,
                              "prompt": row["prompt"], "target": row["target"],
                              "expected_loops": row["expected_loops"],
                              "profiles": ("native", "python_cued", "python_uncued")})
        search = _load(archive, "data/eval_v42_search_retrieval_heldout.jsonl")
        search_seen = defaultdict(int)
        for row in search:
            group = row.get("hop_type", "unclassified")
            if search_seen[group] >= 4:
                continue
            cases.append({"id": row["id"], "group": "search8", "source_family": group,
                          "prompt": row["prompt"], "target": row["target"],
                          "profiles": ("native",)})
            search_seen[group] += 1
            if len(cases) == 28:
                break
        if len(cases) != 28:
            raise ValueError("Pinned search diagnostic families are incomplete")
        logic = _load(archive, "data/eval_v41_logical_instruction_operators_heldout.jsonl")
        operators_seen = defaultdict(int)
        for row in logic:
            op = row["operator_type"]
            if operators_seen[op] >= 2:
                continue
            cases.append({"id": row["id"], "group": "logic8", "source_family": op,
                          "prompt": row["prompt"], "target": row["target"],
                          "profiles": ("native",)})
            operators_seen[op] += 1
            if sum(operators_seen.values()) == 8:
                break
        if len(operators_seen) != 4 or any(count != 2 for count in operators_seen.values()):
            raise ValueError("Pinned logic operators are incomplete")
        academic = _load(archive, "data/benchmark_8100_manifest.jsonl")
        selected = set()
        for row in academic:
            category = row.get("category")
            if row.get("pillar") != "JMMLU_Academic" or category not in ACADEMIC_CATEGORIES or category in selected:
                continue
            cases.append({"id": row["id"], "group": "japanese_explanation8",
                          "source_family": category,
                          "prompt": row["prompt"].rstrip() + EXPLANATION_INSTRUCTION,
                          "original_prompt": row["prompt"], "target": row["target"],
                          "profiles": ("native",)})
            selected.add(category)
            if len(selected) == 8:
                break
    if len(cases) != 44 or len({row["id"] for row in cases}) != 44:
        raise ValueError("Pinned interim diagnostic set is incomplete or duplicate")
    return cases


def main():
    root = Path(r"C:\Users\81808\.openclaw\reports\jmmlu_v1\ag_fused_fullqa_training_v1")
    v43 = root / "v43_v44fix_paired_eval_assets" / "v43_source.zip"
    output = Path(__file__).with_name("fixed_interim_diagnostic_44_v2.jsonl")
    cases = build(v43)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        for row in cases:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"cases": len(cases), "sha256": file_sha(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
