"""Add exact v43 ZIP-row references to an already CPU-verified output snapshot.

The full converter remains the only source of CPU/Python-verification claims.
This script reconstructs its deterministic text from each raw ZIP row and
requires byte-identical prompt/target/source before attaching provenance.
It never executes code or overwrites the original running conversion output.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import convert_sample as candidate


def expected_target(row: dict) -> str | None:
    if row.get("source") not in candidate.ACCEPTED_SOURCES:
        return None
    prompt, target, full_text = row.get("prompt"), row.get("target"), row.get("full_text")
    if not all(isinstance(v, str) for v in (prompt, target, full_text)):
        return None
    if not full_text.startswith(prompt) or full_text not in (prompt + target, prompt + " " + target):
        return None
    suffix = full_text[len(prompt):]
    if "```" in suffix:
        return None
    try:
        pairs = candidate.parse_pairs(suffix)
        candidate.source_semantics(row, pairs)
    except candidate.Reject:
        return None
    pieces = []
    cursor = 0
    for index, pair in enumerate(pairs):
        pieces.append(suffix[cursor:pair.start()])
        code = candidate.instruction_code(pair, index == 0)
        pieces.append("\n```python\n" + code + "```\n")
        pieces.append("\n```output\n[" + pair.group("observed") + "=" + pair.group("value") + "]\n```\n")
        cursor = pair.end()
    pieces.append(suffix[cursor:])
    return "".join(pieces)


def key(item: dict) -> tuple[str, str, str]:
    return item["prompt"], item["target"], item["original_source"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip-path", type=Path, required=True)
    parser.add_argument("--verified-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--through-progress", action="store_true")
    args = parser.parse_args()
    if args.output_dir.resolve() == args.verified_dir.resolve():
        raise ValueError("Provenance output must be a separate directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.through_progress:
        progress = json.loads((args.verified_dir / "progress.json").read_text(encoding="utf-8"))
        max_line = int(progress["rows_completed"])
        expected_count = int(progress["accepted"])
    else:
        final = json.loads((args.verified_dir / "admission_report.json").read_text(encoding="utf-8"))
        max_line = int(final["bounded_rows_examined"])
        expected_count = int(final["counts"]["accepted"])
    original_rows = collections.defaultdict(list)
    with zipfile.ZipFile(args.zip_path) as archive:
        eval_exact, eval_normalized, _ = candidate.evaluation_prompts(archive)
        with archive.open(candidate.TRAIN_MEMBER) as source:
            for line_number, raw in enumerate(source, 1):
                if line_number > max_line:
                    break
                row = json.loads(raw)
                prompt = row.get("prompt")
                if not isinstance(prompt, str) or prompt in eval_exact or candidate.normalized_prompt(prompt) in eval_normalized:
                    continue
                target = expected_target(row)
                if target is None:
                    continue
                original_rows[(prompt, target, row["source"])].append({
                    "kind": "v43_zip_member",
                    "file": candidate.TRAIN_MEMBER,
                    "line": line_number,
                    "raw_sha256": hashlib.sha256(raw).hexdigest(),
                })
    verified = []
    with (args.verified_dir / "converted_sample.jsonl").open(encoding="utf-8") as source:
        for line in source:
            if len(verified) >= expected_count:
                break
            verified.append(json.loads(line))
    if len(verified) != expected_count:
        raise RuntimeError(f"Verified snapshot had {len(verified)} rows, progress promised {expected_count}")
    output_counts = collections.Counter(key(item) for item in verified)
    ambiguous = {k for k, count in output_counts.items() if len(original_rows[k]) != count}
    output = args.output_dir / "converted_with_source_refs.jsonl"
    assigned = collections.Counter()
    unresolved = collections.Counter()
    with output.open("w", encoding="utf-8", newline="\n") as target:
        for item in verified:
            k = key(item)
            if k in ambiguous:
                unresolved["non_unique_or_missing_raw_row"] += 1
                continue
            ref = original_rows[k][assigned[k]]
            assigned[k] += 1
            if item.get("original_prompt_sha256") != hashlib.sha256(item["prompt"].encode("utf-8")).hexdigest():
                unresolved["prompt_sha_mismatch"] += 1
                continue
            if item.get("source_ref") is not None and item["source_ref"] != ref:
                unresolved["existing_source_ref_mismatch"] += 1
                continue
            item["source_ref"] = ref
            target.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
    report = {"source_ref_schema": "{kind:'v43_zip_member',file,line,raw_sha256}",
              "zip_member": candidate.TRAIN_MEMBER, "line_number_base": 1,
              "raw_sha256_definition": "SHA256 of exact JSONL line bytes including its newline",
              "source_rows_through_line_1based": max_line, "verified_snapshot_rows": expected_count,
              "source_ref_attached": sum(assigned.values()) - unresolved["prompt_sha_mismatch"] - unresolved["existing_source_ref_mismatch"],
              "unresolved": dict(unresolved), "ambiguous_key_groups": len(ambiguous),
              "enriched_output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
              "status": "partial snapshot" if args.through_progress else "completed verified output",
              "warning": "Only source_ref_attached rows are unambiguously linked to original ZIP lines."}
    (args.output_dir / "source_ref_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
