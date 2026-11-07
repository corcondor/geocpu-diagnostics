#!/usr/bin/env python3
"""Strict, read-only gate for additive learning exposure; never reduce old groups.

The reference and candidate must be comparable training_windows. This gate does
not prove retention or authorize adoption: T1-T8 and Python must still be tested.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

NEW_DOMAINS = ("python", "logic", "evidence", "japanese")

def last_step(path: Path) -> dict:
    if path.suffix != ".jsonl":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("preview"):
            return data["preview"]
        return data
    item = None
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("event") == "train_step":
                item = event
    if item is None:
        raise ValueError("No train_step record in telemetry")
    return item

def gate(old: dict, new: dict, tolerance: float = 0.15) -> dict:
    if not 0 <= tolerance < 1:
        raise ValueError("Tolerance must be in [0,1)")
    previous = old.get("source_supervised_tokens")
    current = new.get("source_supervised_tokens")
    if not isinstance(previous, dict) or not isinstance(current, dict):
        raise ValueError("Both runs must have source_supervised_tokens")
    if not previous or any(not isinstance(v, int) or v <= 0 for v in previous.values()):
        raise ValueError("Malformed reference exposure")
    old_total = int(old.get("supervised_tokens", sum(previous.values())))
    new_total = int(new.get("supervised_tokens", sum(current.values())))
    if old_total <= 0 or new_total <= 0:
        raise ValueError("Non-positive training exposure")
    decreases = {group: {"before": before, "after": int(current.get(group, 0))}
                 for group, before in previous.items()
                 if int(current.get(group, 0)) < before}
    missing = [key for key in NEW_DOMAINS if int(current.get("v48_"+key, 0)) <= 0]
    extras = {key: int(current.get("v48_"+key, 0)) for key in NEW_DOMAINS}
    vals = list(extras.values())
    balanced = bool(vals and min(vals) and min(vals) >= (1-tolerance)*max(vals))
    total_grew = new_total > old_total
    # A new dataset cannot be described as non-reducing unless every old
    # group receives at least its previous absolute supervised-token count.
    passed = not decreases and not missing and balanced and total_grew
    return {"passed": passed,
            "reference_supervised_tokens": old_total,
            "candidate_supervised_tokens": new_total,
            "absolute_old_group_decreases": decreases,
            "missing_new_domains": missing,
            "new_domain_supervised_tokens": extras,
            "new_domain_tolerance": tolerance,
            "new_domain_balanced_by_supervised_tokens": balanced,
            "total_teacher_tokens_increased": total_grew,
            "old_groups_compared": len(previous),
            "note": "Needs same-scope phase telemetry. Does not replace frozen T1-T8 or model capability scores."}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=0.15)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = gate(last_step(args.reference),last_step(args.candidate),args.tolerance)
    payload = json.dumps(report,ensure_ascii=False,indent=2)+"\n"
    print(payload)
    if args.output:
        if args.output.exists():
            raise FileExistsError("Refuse to overwrite an existing exposure gate report")
        args.output.write_text(payload,encoding="utf-8")
    if not report["passed"]:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
