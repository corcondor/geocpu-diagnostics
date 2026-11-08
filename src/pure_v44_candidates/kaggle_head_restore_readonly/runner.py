"""Kaggle-only read-only three-arm test of the pure-v44 output head.

The unchanged pinned evaluators are supplied in the source bundle. This runner
reuses their preflight, strict checkpoint loading, generation and temporary
head hook, while keeping its own bounded Linux/Kaggle output transaction.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import tempfile
import time
from uuid import uuid4

from verify_outputs import ARMS, CASE_COUNT, read_json, verify_complete, verify_records

MAX_OUTER_WALL_SECONDS = 28_800
PER_ARM_SECONDS = 900
DECODE = {"max_new_tokens": 2048, "arm": "learned", "search": False,
          "repetition_penalty": 1.0, "python_tool_calls": 0, "autocast": False}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(path.name + "." + uuid4().hex + ".partial")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            raise FileExistsError(path)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_or_match(path: Path, value: object) -> None:
    """A resumed attempt may finish an already written finalization step once."""
    if path.exists():
        if read_json(path) != value:
            raise RuntimeError(f"Immutable output differs: {path}")
    else:
        atomic_json_new(path, value)


def verify_source_bundle(source_root: Path) -> dict:
    manifest = json.loads((source_root / "source_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "geocpu_pure_v44_kaggle_head_restore_sources_v1":
        raise RuntimeError("Source bundle schema differs")
    for name, expected in manifest["files_sha256"].items():
        path = source_root / name
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"Source bundle member differs: {name}")
    return manifest


def canonical_sha256(value: object) -> str:
    content = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def make_contract(*, source_manifest_sha256: str, source_pins: dict[str, str],
                  asset_paths: dict[str, str], cases: list[dict]) -> dict:
    """Bind source paths, hashes, decode settings and every case metadata field."""
    if len(cases) != CASE_COUNT:
        raise ValueError("The intervention requires exactly 64 fixed cases")
    ids = [case["id"] for case in cases]
    if len(set(ids)) != CASE_COUNT:
        raise ValueError("Case IDs must be unique")
    return {
        "schema": "geocpu_pure_v44_head_restore_three_arm_v1",
        "suite": "mixed64", "case_count": CASE_COUNT, "case_ids": ids,
        "case_sha256": [canonical_sha256(case) for case in cases],
        "arms": list(ARMS), "source_manifest_sha256": source_manifest_sha256,
        "source_pins": source_pins, "asset_paths": asset_paths,
        "weights_modified": False, "training_updates": 0,
        "matched_cache": None, "same_gpu_runtime_across_arms": True,
        "decode": dict(DECODE),
    }


def check_asset_identity(matched, assets_root: Path, source_manifest: dict) -> dict[str, str]:
    pins = {name: [pin[1], pin[2]] for name, pin in matched.PINS.items()}
    if pins != source_manifest["asset_pins"]:
        raise RuntimeError("Source manifest and pinned asset identities differ")
    paths = {name: pin[0].relative_to(assets_root).as_posix()
             for name, pin in matched.PINS.items()}
    if paths != source_manifest["asset_paths"]:
        raise RuntimeError("Pinned original asset layout differs")
    return paths


@contextmanager
def alarm_after(seconds: int):
    if seconds <= 0:
        raise TimeoutError("No wall time remains")
    def expired(_signum, _frame):
        raise TimeoutError("Per-arm wall deadline expired")
    old = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def check_case(result: dict, *, index: int, case: dict, case_sha256: str) -> None:
    case_id = case["id"]
    if (result.get("case_index") != index or result.get("case_id") != case_id
            or result.get("case_sha256") != case_sha256
            or result.get("prompt") != case["prompt"]
            or result.get("reference") != case["target"]
            or result.get("bucket") != case["bucket"]
            or result.get("error") is not None or result.get("tool_invocations") != 0
            or result.get("first_token_probe_error") is not None
            or result.get("first_logit_probe_matches_generated_first_token") is not True
            or type(result.get("sampled_token_count")) is not int
            or not 0 < result["sampled_token_count"] <= 2048
            or not isinstance(result.get("generated_tokens"), list)
            or len(result["generated_tokens"]) != result["sampled_token_count"]
            or type(result.get("ended_with_eos")) is not bool
            or result["ended_with_eos"] != (result.get("stop_reason") == "eos")
            or (result["ended_with_eos"] and result["generated_tokens"][-1] != 3)
            or result.get("origin") != {"kind": "generated_this_job"}):
        raise RuntimeError(f"Generated answer failed read-only checks: {index} {case_id}")


def prepare_output(output: Path, contract: dict, evidence: dict) -> bool:
    """Create a PARTIAL run or verify the exact contract before resuming it.

    Returns False only when a verified COMPLETE run already exists.
    """
    if output.exists():
        if (output / "COMPLETE.json").exists():
            verify_complete(output)
            (output / "PARTIAL.json").unlink(missing_ok=True)
            return False
        if not (output / "PARTIAL.json").is_file():
            raise RuntimeError("Existing output lacks PARTIAL and COMPLETE markers")
        saved = read_json(output / "PARTIAL.json")
        if saved.get("contract") != contract:
            raise RuntimeError("Resume contract differs from the original paths, SHA, cases, or profile")
        if not (output / "claim.json").exists():
            atomic_json_new(output / "claim.json", saved["claim"])
        if read_json(output / "claim.json") != saved["claim"]:
            raise RuntimeError("Resume claim differs from the PARTIAL record")
        return True
    output.mkdir(parents=True, exist_ok=False)
    claim = dict(contract)
    claim.update({
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "selection_evidence": evidence,
        "candidate": "child upstream with parent base.head output projection",
    })
    atomic_json_new(output / "PARTIAL.json", {"status": "partial", "contract": contract,
                                               "claim": claim})
    atomic_json_new(output / "claim.json", claim)
    return True


def saved_or_generate(arm_path: Path, *, index: int, case: dict,
                      case_sha256: str, generate) -> dict:
    """Revalidate immutable prior output; generate only a missing arm."""
    if arm_path.exists():
        result = read_json(arm_path)
        check_case(result, index=index, case=case, case_sha256=case_sha256)
        return result
    result = generate()
    result["case_sha256"] = case_sha256
    result["origin"] = {"kind": "generated_this_job"}
    check_case(result, index=index, case=case, case_sha256=case_sha256)
    atomic_json_new(arm_path, result)
    return result


def run(assets_root: Path, output: Path, wall_seconds: int) -> None:
    if not 600 <= wall_seconds <= MAX_OUTER_WALL_SECONDS - 300:
        raise ValueError("Internal wall budget must be between 600 and 28500 seconds")
    source_root = Path(__file__).resolve().parent
    source_manifest = verify_source_bundle(source_root)
    os.environ["GEOCPU_DRIVE_ROOT"] = str(assets_root)
    import matched_pure_v44_readonly as matched
    import partial_head_restore_readonly as head

    asset_paths = check_asset_identity(matched, assets_root, source_manifest)
    seen, cases, broad, meta, evidence = head.static_preflight("mixed64")
    if len(cases) != CASE_COUNT or cases != broad or len(seen) != 5:
        raise RuntimeError("Fixed practice/held-out selection differs")

    import torch
    if not Path("/kaggle/input").is_dir() or not Path("/kaggle/working").is_dir():
        raise RuntimeError("This execution is limited to a Kaggle kernel")
    if not assets_root.is_relative_to(Path("/kaggle/input")):
        raise RuntimeError("Assets must be mounted as read-only Kaggle input")
    if not output.is_relative_to(Path("/kaggle/working")):
        raise RuntimeError("Output must be written under Kaggle working storage")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    if shutil.disk_usage("/kaggle/working").free < 8_000_000_000:
        raise RuntimeError("At least 8 GB free local space is required")

    contract = make_contract(
        source_manifest_sha256=sha256(source_root / "source_manifest.json"),
        source_pins={name: pin[2] for name, pin in matched.PINS.items()},
        asset_paths=asset_paths, cases=cases)
    if not prepare_output(output, contract, evidence):
        print("READ_ONLY_PURE_HEAD_RESTORE_ALREADY_COMPLETE", output, flush=True)
        return
    started = time.monotonic()
    deadline = started + wall_seconds
    atomic_json_new(output / "attempts" / f"{uuid4().hex}_start.json", {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": wall_seconds, "per_arm_seconds": PER_ARM_SECONDS,
        "status": "started; PARTIAL remains until complete closure"})
    from weight_direction.weight_direction import repeated_suffix
    with tempfile.TemporaryDirectory(prefix="geocpu_pure_head_", dir="/kaggle/working") as temporary:
        models = head.load_models(Path(temporary), meta)
        vocab = meta["vocab"]
        for index, case in enumerate(cases):
            pair = {"case_index": index, "case_id": case["id"], "bucket": case["bucket"],
                    "case_sha256": contract["case_sha256"][index],
                    "arms": {}}
            for arm in ARMS:
                arm_path = output / "arms" / f"case_{index:02d}_{arm}.json"
                try:
                    def generate():
                        remaining = int(deadline - time.monotonic())
                        if remaining <= 0:
                            raise TimeoutError("Internal job wall deadline expired")
                        with alarm_after(min(PER_ARM_SECONDS, remaining)):
                            if arm == "child_parent_head":
                                with head.temporary_parent_head(models["parent"], models["child"]):
                                    fresh = head.evaluate_case(models["child"], arm, index, case, vocab, torch)
                            else:
                                fresh = head.evaluate_case(models[arm], arm, index, case, vocab, torch)
                        fresh["repetition"] = repeated_suffix(fresh["raw_answer"])
                        return fresh
                    result = saved_or_generate(
                        arm_path, index=index, case=case,
                        case_sha256=contract["case_sha256"][index], generate=generate)
                    pair["arms"][arm] = result
                except BaseException as error:
                    atomic_json_new(output / "attempts" / f"{uuid4().hex}.json", {
                        "case_index": index, "case_id": case["id"], "arm": arm,
                        "error": repr(error)})
                    raise
            pair_path = output / "pairs" / f"case_{index:02d}.json"
            if pair_path.exists():
                if read_json(pair_path) != pair:
                    raise RuntimeError(f"Saved pair differs from immutable arms: {index}")
            else:
                atomic_json_new(pair_path, pair)
        del models
        torch.cuda.empty_cache()
    stop_counts = {arm: {} for arm in ARMS}
    eos_counts = {arm: 0 for arm in ARMS}
    for index in range(CASE_COUNT):
        pair = read_json(output / "pairs" / f"case_{index:02d}.json")
        for arm in ARMS:
            result = pair["arms"][arm]
            reason = result["stop_reason"]
            stop_counts[arm][reason] = stop_counts[arm].get(reason, 0) + 1
            eos_counts[arm] += int(result["ended_with_eos"])
    write_or_match(output / "summary.json", {
        "case_count": CASE_COUNT, "arms": list(ARMS), "training_updates": 0,
        "errors": 0, "natural_eos": eos_counts, "stop_reasons": stop_counts,
        "semantic_correctness_scored": False, "python_tool_use_scored": False,
        "same_gpu_runtime_across_arms": True,
    })
    hashes = verify_records(output)
    write_or_match(output / "result_manifest.json", {
        "schema": "geocpu_pure_v44_head_restore_result_closure_v1",
        "files": hashes,
    })
    atomic_json_new(output / "COMPLETE.json", {
        "status": "complete", "case_count": CASE_COUNT, "arms": list(ARMS),
        "training_updates": 0,
        "result_manifest_sha256": sha256(output / "result_manifest.json"),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    })
    verify_complete(output)
    (output / "PARTIAL.json").unlink(missing_ok=True)
    print("READ_ONLY_PURE_HEAD_RESTORE_COMPLETE", output, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--wall-seconds", type=int)
    args = parser.parse_args()
    assets_root = args.assets_root.resolve(strict=True)
    if args.preflight:
        source_manifest = verify_source_bundle(Path(__file__).resolve().parent)
        os.environ["GEOCPU_DRIVE_ROOT"] = str(assets_root)
        import matched_pure_v44_readonly as matched
        import partial_head_restore_readonly as head
        check_asset_identity(matched, assets_root, source_manifest)
        seen, heldout, _, _, _ = head.static_preflight("mixed64")
        print("STATIC_PREFLIGHT_OK", len(seen), len(heldout), flush=True)
        return
    if args.wall_seconds is None:
        parser.error("--wall-seconds is required for generation")
    try:
        run(assets_root, args.output_dir.resolve(), args.wall_seconds)
    except TimeoutError as error:
        # The inner deadline fires before the OS hard stop. Exit successfully
        # with PARTIAL so Kaggle can publish the immutable answered cases.
        print("READ_ONLY_PURE_HEAD_RESTORE_PARTIAL", args.output_dir.resolve(),
              repr(error), flush=True)


if __name__ == "__main__":
    main()
