#!/usr/bin/env python3
"""CPU-only source-exposure repair for the v48 *candidate* dataset.

No GPU work, checkpoint edits, teacher-token floor reductions or edits to
the parent training JSONL. All replicas keep their original supervision and
source labels; they are sampling weights, NOT new independent questions.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import shutil

MAX_WEIGHT = 32
MAX_REPLAY_REPLICAS = 75000


def parse_weights(text: str) -> dict[str, int]:
    """'source_a=16,source_b=16' -> {'source_a': 16, 'source_b': 16}.

    Explicit replay weights, so the repetition used to keep an old source at its
    previous exposure is a stated, reviewable number (each in [2, MAX_WEIGHT])."""
    weights: dict[str, int] = {}
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        name, sep, value = part.partition("=")
        name, value = name.strip(), value.strip()
        if not sep or not name or not value.isdigit():
            raise ValueError("replay weight must look like source=integer: " + repr(part))
        weight = int(value)
        if not 2 <= weight <= MAX_WEIGHT:
            raise ValueError(f"replay weight for {name!r} must be in [2, {MAX_WEIGHT}], got {weight}")
        if name in weights:
            raise ValueError("replay weight given twice: " + name)
        weights[name] = weight
    if not weights:
        raise ValueError("no replay weights given")
    return weights


def next_weights(declines: dict, old_weights: dict[str, int]) -> dict[str, int]:
    """Conservatively boost only sources below the absolute v47 token floor."""
    if not declines:
        raise ValueError("No old-source shortfall; do not invent replay weights")
    weights = dict(old_weights)
    for source, counts in declines.items():
        before, after = int(counts["baseline"]), int(counts["candidate"])
        if before <= 0 or after < 0 or after >= before:
            raise ValueError("Invalid source floor comparison: " + source)
        ratio = max(2, math.ceil(1.20 * before / max(after, 1)))
        proposed = int(weights.get(source, 1)) * min(ratio, 8)
        if proposed > MAX_WEIGHT:
            raise RuntimeError(
                f"Source {source!r} still lacks exposure at maximum replay weight; "
                "stop before GPU and inspect the sampler instead."
            )
        weights[source] = proposed
    return weights


def write_weighted_candidate(original: Path, destination: Path,
                             weights: dict[str, int]) -> dict:
    """Copy source exactly and append bounded, clearly marked *training* replicas."""
    original, destination = Path(original), Path(destination)
    if destination.exists() or original.resolve() == destination.resolve():
        raise FileExistsError("Refuse to overwrite the original or another candidate")
    if not original.is_file():
        raise FileNotFoundError(original)
    if not weights or any(not isinstance(w, int) or not 2 <= w <= MAX_WEIGHT
                          for w in weights.values()):
        raise ValueError("Weights must be explicit integers in [2,32]")
    destination.parent.mkdir(parents=True, exist_ok=True)
    matches = dict.fromkeys(weights, 0)
    added = 0
    try:
        shutil.copyfile(original, destination)
        with original.open(encoding="utf-8") as source, destination.open("a", encoding="utf-8") as sink:
            for number, raw in enumerate(source):
                if not raw.strip():
                    continue
                row = json.loads(raw)
                group = str(row.get("_practical_source", row.get("_v46_group", "legacy")))
                if group not in weights:
                    continue
                matches[group] += 1
                for replica in range(1, weights[group]):
                    added += 1
                    if added > MAX_REPLAY_REPLICAS:
                        raise RuntimeError(
                            "Excess replay replicas would distort curriculum; stop before GPU"
                        )
                    extra = dict(row)
                    # Never pretend repeated rows are independent exercises.
                    extra["id"] = str(row.get("id", f"old_{number}")) + f"_floor_replay_{number}_{replica}"
                    extra["_v48_replay_replica"] = True
                    sink.write(json.dumps(extra, ensure_ascii=False, separators=(",", ":")) + "\n")
        missing = [name for name, found in matches.items() if found == 0]
        if missing:
            raise RuntimeError(
                "Reference source absent from candidate; cannot repair with weights: "
                + ", ".join(missing)
            )
        return {"replay_replicas_added": added, "original_rows_with_matching_source": matches,
                "source_training_weights": dict(weights)}
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
