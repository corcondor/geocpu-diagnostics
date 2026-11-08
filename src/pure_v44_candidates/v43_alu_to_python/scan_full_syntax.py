"""Count syntactically admissible v43 rows without executing any model or CPU.

This is a coverage census, not proof that rows are numerically correct.
"""

import argparse
import collections
import hashlib
import json
import time
import zipfile
from pathlib import Path

import convert_sample as candidate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip-path", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    counters = collections.Counter()
    source_candidates = collections.Counter()
    with zipfile.ZipFile(args.zip_path) as archive:
        exact, normalized, eval_counts = candidate.evaluation_prompts(archive)
        with archive.open(candidate.TRAIN_MEMBER) as lines:
            for line_number, raw in enumerate(lines, 1):
                row = json.loads(raw)
                prompt = row.get("prompt")
                if not isinstance(prompt, str):
                    counters["invalid_prompt"] += 1
                    continue
                if prompt in exact:
                    counters["eval_prompt_exact_excluded"] += 1
                    continue
                if candidate.normalized_prompt(prompt) in normalized:
                    counters["eval_prompt_normalized_excluded"] += 1
                    continue
                if row.get("source") not in candidate.ACCEPTED_SOURCES:
                    counters["source_not_selected"] += 1
                    continue
                target, full_text = row.get("target"), row.get("full_text")
                if not isinstance(target, str) or not isinstance(full_text, str):
                    counters["invalid_row_schema"] += 1
                    continue
                if not full_text.startswith(prompt) or full_text not in (prompt + target, prompt + " " + target):
                    counters["unverified_prompt_boundary"] += 1
                    continue
                suffix = full_text[len(prompt):]
                if "```" in suffix:
                    counters["preexisting_code_or_observation_fence"] += 1
                    continue
                try:
                    pairs = candidate.parse_pairs(suffix)
                    candidate.source_semantics(row, pairs)
                except candidate.Reject as exc:
                    counters[str(exc)] += 1
                    continue
                counters["syntactic_candidate_unverified"] += 1
                source_candidates[row["source"]] += 1
                if line_number % 50000 == 0:
                    print(json.dumps({"rows_reached": line_number, "elapsed_seconds": round(time.perf_counter()-started, 2)}), flush=True)
    with args.zip_path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    report = {"source_zip_sha256": digest, "train_member": candidate.TRAIN_MEMBER,
              "rows_scanned": sum(counters.values()), "counts": dict(counters),
              "syntactic_candidates_by_source": dict(source_candidates),
              "bundled_eval_unique_prompts": len(exact), "bundled_eval_file_counts": eval_counts,
              "elapsed_wall_seconds": round(time.perf_counter()-started, 3),
              "warning": "Syntactic candidates have NOT passed original CPU, Python executor, vocabulary, EOS, or model tests."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows_scanned": report["rows_scanned"], "counts": report["counts"],
                      "elapsed_wall_seconds": report["elapsed_wall_seconds"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
