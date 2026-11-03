"""Verify that a pure-v44 head intervention has a complete, hashed 64-case record."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ARMS = ("parent", "child", "child_parent_head")
CASE_COUNT = 64


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def expected_paths(case_count: int = CASE_COUNT) -> tuple[str, ...]:
    paths = ["claim.json", "summary.json"]
    for index in range(case_count):
        paths.extend(f"arms/case_{index:02d}_{arm}.json" for arm in ARMS)
        paths.append(f"pairs/case_{index:02d}.json")
    return tuple(paths)


def verify_records(output: Path, *, case_count: int = CASE_COUNT) -> dict[str, str]:
    """Reject missing, mismatched, or unvalidated cases before completion."""
    if case_count <= 0:
        raise ValueError("case_count must be positive")
    claim = read_json(output / "claim.json")
    ids = claim.get("case_ids")
    case_hashes = claim.get("case_sha256")
    if (claim.get("suite") != "mixed64" or claim.get("case_count") != case_count
            or claim.get("arms") != list(ARMS) or claim.get("training_updates") != 0
            or not isinstance(ids, list) or len(ids) != case_count
            or len(set(ids)) != case_count
            or not isinstance(case_hashes, list) or len(case_hashes) != case_count
            or any(not isinstance(value, str) or len(value) != 64 for value in case_hashes)
            or claim.get("decode") != {
                "max_new_tokens": 2048, "arm": "learned", "search": False,
                "repetition_penalty": 1.0, "python_tool_calls": 0, "autocast": False}):
        raise ValueError("Claim does not bind the fixed mixed64 three-arm cohort")
    summary = read_json(output / "summary.json")
    if (summary.get("case_count") != case_count or summary.get("arms") != list(ARMS)
            or summary.get("training_updates") != 0 or summary.get("errors") != 0):
        raise ValueError("Summary does not certify all cases")
    expected = expected_paths(case_count)
    expected_arms = {name for name in expected if name.startswith("arms/")}
    expected_pairs = {name for name in expected if name.startswith("pairs/")}
    actual_arms = {path.relative_to(output).as_posix() for path in (output / "arms").glob("*.json")}
    actual_pairs = {path.relative_to(output).as_posix() for path in (output / "pairs").glob("*.json")}
    if actual_arms != expected_arms or actual_pairs != expected_pairs:
        raise ValueError("Arm or pair file set differs from the fixed cohort")
    hashes = {}
    for name in expected:
        path = output / name
        if not path.is_file():
            raise ValueError(f"Missing record: {name}")
        hashes[name] = sha256(path)
    stop_counts = {arm: {} for arm in ARMS}
    eos_counts = {arm: 0 for arm in ARMS}
    for index, case_id in enumerate(ids):
        pair = read_json(output / f"pairs/case_{index:02d}.json")
        if (pair.get("case_index") != index or pair.get("case_id") != case_id
                or pair.get("case_sha256") != case_hashes[index]
                or set(pair.get("arms", {})) != set(ARMS)):
            raise ValueError(f"Pair identity or arms differ: {index}")
        for arm in ARMS:
            answer = read_json(output / f"arms/case_{index:02d}_{arm}.json")
            if pair["arms"][arm] != answer:
                raise ValueError(f"Pair/arm content differs: {index} {arm}")
            if (answer.get("case_index") != index or answer.get("case_id") != case_id
                    or answer.get("case_sha256") != case_hashes[index]
                    or answer.get("error") is not None
                    or answer.get("tool_invocations") != 0
                    or answer.get("first_token_probe_error") is not None
                    or answer.get("first_logit_probe_matches_generated_first_token") is not True
                    or type(answer.get("sampled_token_count")) is not int
                    or not 0 < answer["sampled_token_count"] <= 2048
                    or not isinstance(answer.get("generated_tokens"), list)
                    or len(answer["generated_tokens"]) != answer["sampled_token_count"]
                    or type(answer.get("ended_with_eos")) is not bool
                    or answer["ended_with_eos"] != (answer.get("stop_reason") == "eos")
                    or (answer["ended_with_eos"] and answer["generated_tokens"][-1] != 3)
                    or not isinstance(answer.get("raw_answer"), str)):
                raise ValueError(f"Invalid generated arm: {index} {arm}")
            reason = answer["stop_reason"]
            stop_counts[arm][reason] = stop_counts[arm].get(reason, 0) + 1
            eos_counts[arm] += int(answer["ended_with_eos"])
    if summary.get("stop_reasons") != stop_counts or summary.get("natural_eos") != eos_counts:
        raise ValueError("Summary does not match the saved arm records")
    return hashes


def verify_complete(output: Path) -> None:
    """Independently verify the final marker and every file it binds."""
    complete = read_json(output / "COMPLETE.json")
    if (complete.get("status") != "complete" or complete.get("case_count") != CASE_COUNT
            or complete.get("arms") != list(ARMS) or complete.get("training_updates") != 0):
        raise ValueError("Invalid COMPLETE marker")
    manifest_path = output / "result_manifest.json"
    if complete.get("result_manifest_sha256") != sha256(manifest_path):
        raise ValueError("Result manifest hash differs")
    manifest = read_json(manifest_path)
    if manifest.get("files") != verify_records(output):
        raise ValueError("Result file closure differs")
