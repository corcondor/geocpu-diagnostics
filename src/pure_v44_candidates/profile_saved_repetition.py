"""Profile saved pure-v44 generations without loading or changing a model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def profile(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["records"]
    if len(rows) != 4:
        raise ValueError("Expected the four saved public examples")
    result = []
    for index, row in enumerate(rows):
        generation = row["generation"]
        ids = generation["generated_tokens"]
        if not isinstance(ids, list) or any(not isinstance(value, int) for value in ids):
            raise ValueError(f"Case {index}: missing generated token IDs")
        if generation["ended_with_eos"]:
            ids = ids[:-1]
        tail = ids[-512:]
        grams = [tuple(tail[start:start + 8]) for start in range(len(tail) - 7)]
        distinct = len(set(grams))
        result.append({
            "case_index": index,
            "sampled_tokens_including_eos": len(generation["generated_tokens"]),
            "natural_eos": bool(generation["ended_with_eos"]),
            "stop_reason": generation["stop_reason"],
            "tail_window_tokens": len(tail),
            "tail_eight_gram_count": len(grams),
            "tail_distinct_eight_grams": distinct,
            "tail_distinct_fraction": distinct / len(grams) if grams else None,
            "tool_invocations": generation["tool_invocations"],
        })
    return {
        "schema": "geocpu.pure_v44.saved_repetition_profile.v1",
        "input_file_sha256": sha256(path),
        "metric_definition": "Distinct contiguous eight-token windows divided by all such windows in the last at most 512 model-generated tokens, excluding EOS; low values indicate token recurrence but do not establish a weight cause or answer quality.",
        "cases": result,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    value = profile(args.input)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(value, ensure_ascii=False))


if __name__ == "__main__":
    main()
