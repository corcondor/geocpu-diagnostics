"""Build one private, source-pinned T4x2 Kaggle stage-1 package; never submit it."""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
import zlib

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
OVERLAY = HERE.parents[2] / "v44fix"
sys.path.insert(0, str(STUDY))
from prelaunch_contract import verify_prelaunch_contract  # noqa: E402
from train_hybrid_stage_v2 import (  # noqa: E402
    _verify_original_swe_baseline, _verify_pretrain_raw_diagnostic,
    PARENT_CHECKPOINT_SHA, PINNED_INTERIM_CASES_SHA, SOURCE_OVERLAY_SHA,
)
from source_pools import PINNED_V43_ZIP_SHA, PINNED_CONVERTED_SHA, file_sha  # noqa: E402
from source_pools_v2 import SWE_SHA, SWE_MANIFEST_SHA, SWE_AUDIT_SHA  # noqa: E402

KERNEL_ID = "geocpuresearch/geocpu-v43-python-stage1-paired-t4x2-20261104"
ORIGINAL_DATASET = "geocpuresearch/geocpu-v43-v44fix-paired-eval-20261103"
BRIDGE_DATASET = "geocpuresearch/geocpu-v43-python-bridge-verified-20261104"
RUN_SECONDS = 1800
SAVE_RESERVE_SECONDS = 180
SOURCE_SHA = {
    "train_hybrid_stage_v2.py": "1a79b35ae134a2a17874874823f9bd11d7f1c19e4592e467e2396f6387f9b085",
    "hybrid_data.py": "05e1a2a822af7e8b45f3f50ec673fb331936b0734f65661148a7f28c6eb3e90d",
    "source_pools.py": "32a4d56c67dc59bd9c41599a64c648d3e06d0cc33a59d987a8aba569f65b3e85",
    "source_pools_v2.py": "74ae3d0c246916fbcfa2ccdcb97e13a980136688bcd33fd3b0150ad1442be979",
    "stratified_exposure_v2.py": "c4e51065aa88ba5bf871cd940cac59a4e53cd0a6a5596b05827cc119fd3c1dec",
    "prelaunch_contract.py": "eb0463165936a2d47e6a1346361fe940c5869bc0de36c19b09c9190f4dc56165",
    "eval_fixed_diagnostic.py": "bfce8f14f28e8e3714d30234da445e3d2c1e47a97e8666ca0a959e0dad630d8b",
    "fixed_interim_diagnostic_44_v2.jsonl": PINNED_INTERIM_CASES_SHA,
    "replay_python_full.py": "e1d1939625f6067339fe71354bfa4b41496dcf30dbcd2f08a12d5a267727802d",
}
SWE_PROOF_FILES = (
    "COMPLETE.json", "swe200_full_answers.jsonl", "swe200_original_grades.jsonl",
    "original_swe_audit.jsonl", "v43_native_full_answers.jsonl",
    "v43_native_external_grades.jsonl", "v43_original_answers.jsonl",
    "old_skill_eval.jsonl",
)
FIXED44_PROOF_FILES = ("COMPLETE.json", "raw_answers_44_profiles.jsonl")
BRIDGE_EXPECTED = {
    "converted_v1.jsonl": PINNED_CONVERTED_SHA,
    "converted_swe_v2.jsonl": SWE_SHA,
    "swe_manifest_v2.json": SWE_MANIFEST_SHA,
    "swe_audit_v2.json": SWE_AUDIT_SHA,
}
PRIOR_TERMINAL_JOBS = (
    "geocpuresearch/geocpu-v43-native-swe200-read-only-proof-20261104",
    "geocpuresearch/geocpu-v43-parent-fixed44-fp16-20261104",
)


def _safe_relative(value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError("Claim evidence path is not package-relative")
    return relative


def _claim_files(path: Path, arm: str, now: datetime) -> tuple[dict, dict[str, Path]]:
    claim = verify_prelaunch_contract(
        path, now=now, expected_arm=arm, required_run_seconds=RUN_SECONDS,
        required_save_reserve_seconds=SAVE_RESERVE_SECONDS,
    )
    if (claim["gpu_count"] != 1 or claim["precision_profile"] != "train_fp16_gradscaler_v1"
            or len(claim["gpu_names"]) != 1 or "T4" not in claim["gpu_names"][0].upper()):
        raise ValueError("Each arm requires one explicitly claimed FP16 Tesla T4")
    paths = {"prelaunch_contract_once.json": path}
    for item in claim["evidence"].values():
        relative = _safe_relative(item["file"])
        key = relative.as_posix()
        candidate = path.parent / relative
        if key in paths and paths[key] != candidate:
            raise ValueError("Claim evidence names collide")
        paths[key] = candidate
    return claim, paths


def _pair_claims(control: dict, python: dict, control_path: Path, python_path: Path) -> None:
    if (control["claim_id"] == python["claim_id"]
            or control["run_id"] == python["run_id"]):
        raise ValueError("The two arms need distinct one-use claims and run IDs")
    if control["gpu_names"] != python["gpu_names"]:
        raise ValueError("The arms must claim the same T4 model")
    if control["hard_stop_at_utc"] != python["hard_stop_at_utc"]:
        raise ValueError("The paired arms require one common hard stop")
    for field in ("official_quota_observed_at_utc", "active_jobs_observed_at_utc",
                  "official_minimum_free_device_seconds"):
        if control[field] != python[field]:
            raise ValueError("The arms must share one fresh official quota and terminal-job snapshot")
    for key in ("official_quota_evidence", "active_jobs_snapshot"):
        if control["evidence"][key]["sha256"] != python["evidence"][key]["sha256"]:
            raise ValueError("The two arms have different quota or terminal-job evidence")
    required = control["root_reserved_device_seconds"] + python["root_reserved_device_seconds"]
    if control["official_minimum_free_device_seconds"] < required:
        raise ValueError("Fresh official free quota does not cover both T4 reservations")
    jobs_item = control["evidence"]["active_jobs_snapshot"]
    jobs_path = control_path.parent / _safe_relative(jobs_item["file"])
    jobs = json.loads(jobs_path.read_text(encoding="utf-8"))
    terminal = {
        item.get("kernel_id", item.get("ref", item.get("id"))): item.get("status")
        for item in jobs.get("jobs", [])
    }
    if any(terminal.get(job_id) != "COMPLETE" for job_id in PRIOR_TERMINAL_JOBS):
        raise ValueError("A required prior Kaggle proof job is not terminal COMPLETE")
    # The independent ledgers must describe the same paired launch and account
    # for both claims. Root consumes each claim externally before submission.
    for claim, path in ((control, control_path), (python, python_path)):
        item = claim["evidence"]["root_hold_ledger"]
        ledger = json.loads((path.parent / _safe_relative(item["file"])).read_text(encoding="utf-8"))
        if (ledger.get("paired_kernel_id") != KERNEL_ID
                or ledger.get("paired_claim_ids") != sorted([control["claim_id"], python["claim_id"]])
                or ledger.get("paired_reserved_device_seconds") != required):
            raise ValueError("Root ledger does not reserve this exact two-arm launch")


def _bridge_manifest(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    if (data.get("schema") != "geocpu_hybrid_python_bridge_dataset_v2"
            or data.get("original_v43_source_sha256") != PINNED_V43_ZIP_SHA):
        raise ValueError("Verified bridge dataset manifest differs")
    for name, expected in BRIDGE_EXPECTED.items():
        if data.get("files", {}).get(name, {}).get("sha256") != expected:
            raise ValueError("Bridge asset differs: " + name)
    for name in ("replay_COMPLETE.json", "all_row_replay_receipts.jsonl"):
        if not isinstance(data.get("files", {}).get(name, {}).get("sha256"), str):
            raise ValueError("Independent Python replay is absent from bridge manifest")


def _embedded_item(path: Path, expected: str | None = None) -> dict:
    raw = path.read_bytes()
    digest = sha256(raw).hexdigest()
    if expected is not None and digest != expected:
        raise ValueError("Pinned package asset changed: " + str(path))
    return {"bytes": len(raw), "sha256": digest,
            "zlib_base64": base64.b64encode(zlib.compress(raw, 9)).decode("ascii")}


def package(*, swe_proof_dir: Path, fixed44_proof_dir: Path,
            control_claim: Path, python_claim: Path, bridge_manifest: Path,
            staged: Path, now: datetime | None = None) -> dict:
    if staged.exists():
        raise FileExistsError("Use a new private, one-use Kaggle staging directory")
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    _verify_original_swe_baseline(swe_proof_dir, expected_precision_profile="eval_bf16_native")
    _verify_pretrain_raw_diagnostic(fixed44_proof_dir, expected_precision_profile="eval_fp16_native_v1")
    control, control_files = _claim_files(control_claim, "control", now)
    python, python_files = _claim_files(python_claim, "python", now)
    _pair_claims(control, python, control_claim, python_claim)
    _bridge_manifest(bridge_manifest)
    files = {name: (STUDY / name, expected) for name, expected in SOURCE_SHA.items()}
    files.update({"overlay/" + name: (OVERLAY / name, expected)
                  for name, expected in SOURCE_OVERLAY_SHA.items()})
    files.update({"proofs/swe200/" + name: (swe_proof_dir / name, None)
                  for name in SWE_PROOF_FILES})
    files.update({"proofs/fixed44/" + name: (fixed44_proof_dir / name, None)
                  for name in FIXED44_PROOF_FILES})
    files.update({"claims/control/" + name: (path, None)
                  for name, path in control_files.items()})
    files.update({"claims/python/" + name: (path, None)
                  for name, path in python_files.items()})
    embedded = {name: _embedded_item(path, expected) for name, (path, expected) in files.items()}
    template = (HERE / "bootstrap_template.py").read_text(encoding="utf-8")
    if template.count("__EMBEDDED__") != 1:
        raise ValueError("Bootstrap insertion point changed")
    bootstrap = template.replace("__EMBEDDED__", repr(embedded)).encode("utf-8")
    metadata = {
        "id": KERNEL_ID, "title": KERNEL_ID.split("/", 1)[1],
        "code_file": "bootstrap.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "enable_internet": False, "keywords": [],
        "dataset_sources": [ORIGINAL_DATASET, BRIDGE_DATASET],
        "kernel_sources": [], "competition_sources": [], "model_sources": [],
        "machine_shape": "NvidiaTeslaT4", "session_timeout_seconds": 2400,
    }
    metadata_raw = (json.dumps(metadata, sort_keys=True, indent=2) + "\n").encode()
    manifest = {
        "schema": "geocpu_stage1_paired_t4x2_package_v1", "kernel_id": KERNEL_ID,
        "parent_sha256": PARENT_CHECKPOINT_SHA, "source_zip_sha256": PINNED_V43_ZIP_SHA,
        "arm_claim_ids": {"control": control["claim_id"], "python": python["claim_id"]},
        "claim_sha256": {"control": file_sha(control_claim), "python": file_sha(python_claim)},
        "bridge_manifest_sha256": file_sha(bridge_manifest),
        "bootstrap_sha256": sha256(bootstrap).hexdigest(),
        "metadata_sha256": sha256(metadata_raw).hexdigest(),
        "embedded_sha256": {name: item["sha256"] for name, item in embedded.items()},
        "stage": 1, "update_barrier": 250, "stage_promotion": False,
    }
    staged.mkdir(parents=True)
    (staged / "bootstrap.py").write_bytes(bootstrap)
    (staged / "kernel-metadata.json").write_bytes(metadata_raw)
    (staged / "package-manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return {"staged": str(staged), "kernel_id": KERNEL_ID,
            "embedded_files": len(embedded), "bootstrap_sha256": manifest["bootstrap_sha256"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("swe_proof_dir", "fixed44_proof_dir", "control_claim",
                 "python_claim", "bridge_manifest", "staged"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    print(json.dumps(package(**vars(parser.parse_args())), sort_keys=True))


if __name__ == "__main__":
    main()
