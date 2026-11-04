"""Stage a private, read-only T4x2 evaluation of the sealed stage-1 pair."""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
import zlib

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
OVERLAY = HERE.parents[2] / "v44fix"
sys.path.insert(0, str(STUDY))
from prelaunch_contract import verify_prelaunch_contract  # noqa: E402
from source_pools import PINNED_CONVERTED_SHA, PINNED_V43_ZIP_SHA, file_sha  # noqa: E402
from source_pools_v2 import SWE_AUDIT_SHA, SWE_MANIFEST_SHA, SWE_SHA  # noqa: E402
from train_hybrid_stage_v2 import (  # noqa: E402
    PARENT_CHECKPOINT_BYTES, PARENT_CHECKPOINT_SHA, SOURCE_OVERLAY_SHA,
    _verify_original_swe_baseline, _verify_pretrain_raw_diagnostic,
)

KERNEL_ID = "geocpuresearch/geocpu-v43py-fixed44-eval-20261104"
ORIGINAL_DATASET = "geocpuresearch/geocpu-v43-v44fix-paired-eval-20261103"
PAIRED_KERNEL_ID = "geocpuresearch/geocpu-v43py-stage1-m4096-20261104"
PAIRED_ROOT_REL = "geocpu_hybrid_stage1_fp16_paired_m4096_20261104_v3"
RUN_SECONDS = 1800
SAVE_RESERVE_SECONDS = 180
PROFILE = "eval_fp16_native_v1"
CASES_SHA = "4032a2c17de2ce56a7758cd35c16db2c2b357ce76b394522411e86c0aa682b45"
EVALUATOR_SHA = "bfce8f14f28e8e3714d30234da445e3d2c1e47a97e8666ca0a959e0dad630d8b"
TRAIN_V2_SHA = "1a79b35ae134a2a17874874823f9bd11d7f1c19e4592e467e2396f6387f9b085"
SOURCE_SHA = {
    "train_hybrid_stage_v2.py": TRAIN_V2_SHA,
    "hybrid_data.py": "05e1a2a822af7e8b45f3f50ec673fb331936b0734f65661148a7f28c6eb3e90d",
    "source_pools.py": "32a4d56c67dc59bd9c41599a64c648d3e06d0cc33a59d987a8aba569f65b3e85",
    "source_pools_v2.py": "74ae3d0c246916fbcfa2ccdcb97e13a980136688bcd33fd3b0150ad1442be979",
    "stratified_exposure_v2.py": "c4e51065aa88ba5bf871cd940cac59a4e53cd0a6a5596b05827cc119fd3c1dec",
    "prelaunch_contract.py": "eb0463165936a2d47e6a1346361fe940c5869bc0de36c19b09c9190f4dc56165",
    "eval_fixed_diagnostic.py": EVALUATOR_SHA,
    "launch_fixed_diagnostic_bounded.py": "4239d2d17d6690e2b3bdc00a40a8dec778305c5309fea5d108d69cce3cb87156",
    "fixed_interim_diagnostic_44_v2.jsonl": CASES_SHA,
}
SWE_PROOF_FILES = (
    "COMPLETE.json", "swe200_full_answers.jsonl", "swe200_original_grades.jsonl",
    "original_swe_audit.jsonl", "v43_native_full_answers.jsonl",
    "v43_native_external_grades.jsonl", "v43_original_answers.jsonl", "old_skill_eval.jsonl",
)
FIXED44_PROOF_FILES = ("COMPLETE.json", "raw_answers_44_profiles.jsonl")
PRIOR_TERMINAL_JOBS = (
    "geocpuresearch/geocpu-v43-native-swe200-read-only-proof-20261104",
    "geocpuresearch/geocpu-v43-parent-fixed44-fp16-20261104",
    PAIRED_KERNEL_ID,
)


def _safe_relative(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError("Evidence path must remain package-relative")
    return path


def _claim(path: Path, arm: str, now: datetime) -> tuple[dict, dict[str, Path]]:
    claim = verify_prelaunch_contract(
        path, now=now, expected_arm=arm, required_run_seconds=RUN_SECONDS,
        required_save_reserve_seconds=SAVE_RESERVE_SECONDS,
    )
    if (claim["gpu_count"] != 1 or claim["precision_profile"] != PROFILE
            or len(claim["gpu_names"]) != 1 or "T4" not in claim["gpu_names"][0].upper()):
        raise ValueError("Each evaluation arm requires one claimed FP16 Tesla T4")
    files = {"prelaunch_contract_once.json": path}
    for item in claim["evidence"].values():
        relative = _safe_relative(item["file"])
        key = relative.as_posix()
        candidate = path.parent / relative
        if file_sha(candidate) != item["sha256"]:
            raise ValueError("Claim evidence SHA differs: " + key)
        if key in files and files[key] != candidate:
            raise ValueError("Claim evidence filenames collide")
        files[key] = candidate
    return claim, files


def _pair_claims(control: dict, python: dict, control_path: Path, python_path: Path,
                 *, swe_sha: str, fixed44_sha: str) -> None:
    if (control["claim_id"] == python["claim_id"] or control["run_id"] == python["run_id"]
            or control["gpu_names"] != python["gpu_names"]
            or control["hard_stop_at_utc"] != python["hard_stop_at_utc"]):
        raise ValueError("Evaluation claims are not distinct synchronized T4 reservations")
    for claim in (control, python):
        if (claim.get("parent_swe200_baseline_complete_sha256") != swe_sha
                or claim.get("parent_fixed44_raw_diagnostic_complete_sha256") != fixed44_sha):
            raise ValueError("Evaluation claim is not bound to the stage-1 parent proofs")
    for field in ("official_quota_observed_at_utc", "active_jobs_observed_at_utc",
                  "official_minimum_free_device_seconds"):
        if control[field] != python[field]:
            raise ValueError("Evaluation claims need one common fresh quota and job snapshot")
    for key in ("official_quota_evidence", "active_jobs_snapshot"):
        if control["evidence"][key]["sha256"] != python["evidence"][key]["sha256"]:
            raise ValueError("Evaluation claims have different provider evidence")
    required = sum(claim["root_reserved_device_seconds"] for claim in (control, python))
    if control["official_minimum_free_device_seconds"] < required:
        raise ValueError("Official free device time cannot cover two simultaneous T4 evaluations")
    quota_item = control["evidence"]["official_quota_evidence"]
    quota = json.loads((control_path.parent / _safe_relative(quota_item["file"])).read_text(encoding="utf-8"))
    existing_holds = quota.get("existing_unreturned_hold_device_seconds")
    if (type(existing_holds) not in (int, float) or existing_holds < 0
            or quota.get("available_after_existing_holds_device_seconds") !=
            control["official_minimum_free_device_seconds"] - existing_holds
            or quota.get("new_paired_reservation_device_seconds") != required
            or quota.get("total_allowance_counted") is not False
            or quota.get("pay_to_scale_allowance_counted") is not False
            or control["official_minimum_free_device_seconds"] - existing_holds < required):
        raise ValueError("Official free time after existing local holds cannot cover this pair")
    jobs_item = control["evidence"]["active_jobs_snapshot"]
    jobs = json.loads((control_path.parent / _safe_relative(jobs_item["file"])).read_text(encoding="utf-8"))
    terminal = {item.get("kernel_id", item.get("ref", item.get("id"))): item.get("status")
                for item in jobs["jobs"]}
    if any(terminal.get(job) != "COMPLETE" for job in PRIOR_TERMINAL_JOBS):
        raise ValueError("Parent or paired stage-1 Kaggle proof job is not COMPLETE")
    for claim, path in ((control, control_path), (python, python_path)):
        item = claim["evidence"]["root_hold_ledger"]
        ledger = json.loads((path.parent / _safe_relative(item["file"])).read_text(encoding="utf-8"))
        if (ledger.get("paired_kernel_id") != KERNEL_ID
                or ledger.get("paired_claim_ids") != sorted([control["claim_id"], python["claim_id"]])
                or ledger.get("paired_reserved_device_seconds") != required):
            raise ValueError("Root ledger does not reserve this exact evaluation pair")


def _pair_checkpoints(paired_dir: Path, *, swe_sha: str, fixed44_sha: str) -> dict:
    pair_status_path = paired_dir / "pair_status.json"
    pair = json.loads(pair_status_path.read_text(encoding="utf-8"))
    if (pair.get("schema") != "geocpu_stage1_paired_t4x2_result_v1"
            or pair.get("status") != "BLOCK1_NEEDS_PAIRED_EVALUATION"
            or pair.get("stage_promotion") is not False
            or pair.get("absolute_stop_reached") is not False
            or pair.get("process_exit_codes") != {"control": 0, "python": 0}):
        raise ValueError("Paired stage 1 did not seal both 250-update arms")
    files = {"pair_status.json": {"sha256": file_sha(pair_status_path)}}
    bindings = {}
    for arm in ("control", "python"):
        checkpoint_rel = f"{arm}/hybrid_stage1_{arm}_update_0250.pt"
        checkpoint = paired_dir / checkpoint_rel
        receipt_rel = checkpoint_rel + ".receipt.json"
        plan_rel = f"{arm}/admitted_actual_target_plan.json"
        receipt_path = paired_dir / receipt_rel
        plan_path = paired_dir / plan_rel
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        binding = plan.get("binding", {})
        declared_sha = receipt.get("sha256")
        declared_bytes = receipt.get("bytes")
        if (not isinstance(declared_sha, str) or re.fullmatch(r"[0-9a-f]{64}", declared_sha) is None
                or type(declared_bytes) is not int or declared_bytes <= 0):
            raise ValueError("Sealed checkpoint receipt lacks a SHA-256 digest and byte count")
        # A small Kaggle output proof bundle is enough to stage the direct
        # kernel-source mount. If local weights are present, check them too.
        if checkpoint.exists() and (checkpoint.stat().st_size != declared_bytes
                                    or file_sha(checkpoint) != declared_sha):
            raise ValueError("Local stage-1 checkpoint differs from its sealed receipt: " + arm)
        digest = declared_sha
        if (receipt.get("schema") != "geocpu_v43_python_gradual_checkpoint_receipt_v1"
                or receipt.get("status") != "BLOCK1_NEEDS_PAIRED_EVALUATION"
                or receipt.get("accepted_updates") != 250
                or receipt.get("sha256") != digest
                or receipt.get("binding_sha256") != sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
                or pair.get("arm_results", {}).get(arm, {}).get("accepted_updates") != 250
                or pair["arm_results"][arm].get("checkpoint_sha256") != digest
                or pair["arm_results"][arm].get("first_evaluation_barrier_reached") is not True
                or pair["arm_results"][arm].get("stage_complete") is not False):
            raise ValueError("Stage-1 checkpoint, receipt, plan, or pair status differs: " + arm)
        if (binding.get("schema") != "geocpu_v43_python_gradual_stage_v1"
                or binding.get("curriculum_version") != 2 or binding.get("stage") != 1
                or binding.get("arm") != arm or binding.get("parent_sha256") != PARENT_CHECKPOINT_SHA
                or binding.get("v43_zip_sha256") != PINNED_V43_ZIP_SHA
                or binding.get("converted_sha256") != PINNED_CONVERTED_SHA
                or binding.get("swe_converted_sha256") != SWE_SHA
                or binding.get("swe_manifest_sha256") != SWE_MANIFEST_SHA
                or binding.get("swe_independent_audit_sha256") != SWE_AUDIT_SHA
                or binding.get("stage_source_sha256") != TRAIN_V2_SHA
                or binding.get("source_pools_sha256") != SOURCE_SHA["source_pools.py"]
                or binding.get("source_pools_v2_sha256") != SOURCE_SHA["source_pools_v2.py"]
                or binding.get("stratified_schedule_sha256") != SOURCE_SHA["stratified_exposure_v2.py"]
                or binding.get("hybrid_data_sha256") != SOURCE_SHA["hybrid_data.py"]
                or binding.get("prelaunch_verifier_sha256") != SOURCE_SHA["prelaunch_contract.py"]
                or binding.get("overlay_sha256") != SOURCE_OVERLAY_SHA
                or binding.get("precision_profile") != "train_fp16_gradscaler_v1"
                or binding.get("max_micro_tokens") != 4096
                or binding.get("python_actual_target_cap") != (0.0 if arm == "control" else 0.01)
                or binding.get("original_swe200_baseline_complete_sha256") != swe_sha
                or binding.get("original_swe200_baseline_precision_profile") != "eval_bf16_native_historical_only"
                or binding.get("parent_fixed44_raw_diagnostic_complete_sha256") != fixed44_sha
                or not isinstance(binding.get("replay_complete_sha256"), str)
                or re.fullmatch(r"[0-9a-f]{64}", binding["replay_complete_sha256"]) is None
                or not isinstance(binding.get("gpu_names"), list)
                or len(binding["gpu_names"]) != 1
                or not isinstance(binding["gpu_names"][0], str)
                or "T4" not in binding["gpu_names"][0].upper()):
            raise ValueError("Stage-1 plan is not bound to the pinned v2 sources and parent: " + arm)
        bindings[arm] = binding
        files[checkpoint_rel] = {"sha256": digest, "bytes": declared_bytes}
        files[receipt_rel] = {"sha256": file_sha(receipt_path)}
        files[plan_rel] = {"sha256": file_sha(plan_path)}
    common = ("common_boundary_plan_sha256", "seed", "lr", "batch_size",
              "max_micro_tokens", "ce_normalization", "compression_loss",
              "microbatch_boundary", "converted_sha256", "swe_converted_sha256",
              "swe_manifest_sha256", "swe_independent_audit_sha256",
              "replay_complete_sha256", "gpu_names")
    if any(bindings["control"].get(key) is None or
           bindings["control"][key] != bindings["python"].get(key) for key in common):
        raise ValueError("Paired stage-1 arms do not share one parent data and schedule profile")
    return files


def _embedded(path: Path, expected: str | None = None) -> dict:
    raw = path.read_bytes()
    digest = sha256(raw).hexdigest()
    if expected is not None and digest != expected:
        raise ValueError("Pinned source or proof changed: " + str(path))
    return {"bytes": len(raw), "sha256": digest,
            "zlib_base64": base64.b64encode(zlib.compress(raw, 9)).decode("ascii")}


def package(*, paired_dir: Path, v43_zip: Path,
            parent_checkpoint: Path, swe_proof_dir: Path, fixed44_proof_dir: Path,
            control_claim: Path, python_claim: Path, staged: Path,
            now: datetime | None = None) -> dict:
    if staged.exists():
        raise FileExistsError("Use a new private, one-use evaluation staging directory")
    if len(KERNEL_ID.split("/", 1)[1]) > 50:
        raise ValueError("Kaggle evaluation kernel slug/title exceeds 50 characters")
    if (file_sha(v43_zip) != PINNED_V43_ZIP_SHA
            or parent_checkpoint.stat().st_size != PARENT_CHECKPOINT_BYTES
            or file_sha(parent_checkpoint) != PARENT_CHECKPOINT_SHA):
        raise ValueError("Pinned original v43 ZIP or parent checkpoint changed")
    swe_sha = _verify_original_swe_baseline(swe_proof_dir,
                                            expected_precision_profile="eval_bf16_native")
    fixed44_sha = _verify_pretrain_raw_diagnostic(fixed44_proof_dir,
                                                   expected_precision_profile=PROFILE)
    paired_files = _pair_checkpoints(paired_dir, swe_sha=swe_sha, fixed44_sha=fixed44_sha)
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    control, control_files = _claim(control_claim, "control", now)
    python, python_files = _claim(python_claim, "python", now)
    _pair_claims(control, python, control_claim, python_claim,
                 swe_sha=swe_sha, fixed44_sha=fixed44_sha)
    files = {name: (STUDY / name, digest) for name, digest in SOURCE_SHA.items()}
    # The unchanged evaluator imports this name. Its bytes are the sealed v2
    # trainer, so its stage_source_sha256 check still matches both checkpoints.
    files["train_hybrid_stage.py"] = (STUDY / "train_hybrid_stage_v2.py", TRAIN_V2_SHA)
    files["compare_fixed44.py"] = (HERE / "compare_fixed44.py", None)
    files.update({"overlay/" + name: (OVERLAY / name, digest)
                  for name, digest in SOURCE_OVERLAY_SHA.items()})
    files.update({"proofs/swe200/" + name: (swe_proof_dir / name, None)
                  for name in SWE_PROOF_FILES})
    files.update({"proofs/fixed44/" + name: (fixed44_proof_dir / name, None)
                  for name in FIXED44_PROOF_FILES})
    files.update({"claims/control/" + name: (path, None)
                  for name, path in control_files.items()})
    files.update({"claims/python/" + name: (path, None)
                  for name, path in python_files.items()})
    embedded = {name: _embedded(path, expected) for name, (path, expected) in files.items()}
    expected = {"paired_files": paired_files, "paired_kernel_id": PAIRED_KERNEL_ID,
                "swe_complete_sha256": swe_sha, "fixed44_complete_sha256": fixed44_sha,
                "claim_sha256": {"control": file_sha(control_claim),
                                 "python": file_sha(python_claim)}}
    template = (HERE / "bootstrap_template.py").read_text(encoding="utf-8")
    if template.count("__EMBEDDED__") != 1 or template.count("__EXPECTED__") != 1:
        raise ValueError("Bootstrap insertion points changed")
    bootstrap = template.replace("__EMBEDDED__", repr(embedded)).replace(
        "__EXPECTED__", repr(expected)).encode("utf-8")
    metadata = {
        "id": KERNEL_ID, "title": KERNEL_ID.split("/", 1)[1],
        "code_file": "bootstrap.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "enable_internet": False, "keywords": [],
        "dataset_sources": [ORIGINAL_DATASET],
        "kernel_sources": [PAIRED_KERNEL_ID], "competition_sources": [], "model_sources": [],
        "machine_shape": "NvidiaTeslaT4", "session_timeout_seconds": 2400,
    }
    metadata_raw = (json.dumps(metadata, sort_keys=True, indent=2) + "\n").encode()
    manifest = {
        "schema": "geocpu_stage1_paired_fixed44_eval_package_v1",
        "kernel_id": KERNEL_ID, "paired_kernel_id": PAIRED_KERNEL_ID,
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA,
        "source_zip_sha256": PINNED_V43_ZIP_SHA,
        "stage1_files": paired_files,
        "parent_swe200_complete_sha256": swe_sha,
        "parent_fixed44_complete_sha256": fixed44_sha,
        "claim_sha256": expected["claim_sha256"],
        "embedded_sha256": {name: item["sha256"] for name, item in embedded.items()},
        "bootstrap_sha256": sha256(bootstrap).hexdigest(),
        "metadata_sha256": sha256(metadata_raw).hexdigest(),
        "evaluation_only": True, "stage_promotion": False,
        "precision_profile": PROFILE, "max_generated_tokens": 2048,
    }
    staged.mkdir(parents=True)
    (staged / "bootstrap.py").write_bytes(bootstrap)
    (staged / "kernel-metadata.json").write_bytes(metadata_raw)
    (staged / "package-manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return {"staged": str(staged), "kernel_id": KERNEL_ID,
            "checkpoint_sha256": {arm: paired_files[f"{arm}/hybrid_stage1_{arm}_update_0250.pt"]["sha256"]
                                  for arm in ("control", "python")},
            "bootstrap_sha256": manifest["bootstrap_sha256"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("paired_dir", "v43_zip", "parent_checkpoint", "swe_proof_dir",
                 "fixed44_proof_dir", "control_claim", "python_claim", "staged"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    print(json.dumps(package(**vars(parser.parse_args())), sort_keys=True))


if __name__ == "__main__":
    main()
