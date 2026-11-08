"""Issue two one-use read-only evaluation claims from fresh official Kaggle observations.

This reads quota and job status only. It does not submit a job, call a GPU, or
edit older claims. The caller supplies reconciled unreturned local GPU holds.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Callable
from uuid import uuid4

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from prelaunch_contract import MAX_AGE_SECONDS, SCHEMA, verify_prelaunch_contract  # noqa: E402
from prepare import (  # noqa: E402
    KERNEL_ID, PAIRED_KERNEL_ID, PRIOR_TERMINAL_JOBS, PROFILE, RUN_SECONDS, SAVE_RESERVE_SECONDS,
    _pair_claims, _verify_original_swe_baseline, _verify_pretrain_raw_diagnostic,
)

PER_ARM_DEVICE_SECONDS = RUN_SECONDS + SAVE_RESERVE_SECONDS
PAIRED_DEVICE_SECONDS = 2 * PER_ARM_DEVICE_SECONDS
CONSERVATIVE_PRIOR_HOLDS_DEVICE_SECONDS = 15_840
TERMINAL = {"COMPLETE", "CANCELLED", "ERROR", "FAILED"}
PAGE_SIZE = 100


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("A UTC timestamp requires an offset")
    return value.astimezone(timezone.utc)


def _seconds(value: object, name: str) -> float:
    if hasattr(value, "total_seconds"):
        value = value.total_seconds()
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("Invalid official " + name + " GPU device seconds")
    return float(value)


def _status(value: object) -> str:
    if hasattr(value, "name"):
        value = value.name
    result = str(value).rsplit(".", 1)[-1].upper()
    if result not in TERMINAL:
        raise RuntimeError("A Kaggle job is nonterminal or has unknown status: " + result)
    return result


def _last_run(kernel: object) -> datetime | None:
    value = getattr(kernel, "last_run_time", None)
    if value is None and hasattr(kernel, "to_dict"):
        value = kernel.to_dict().get("lastRunTime")
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(value, datetime):
        return None
    # The provider currently returns some naive timestamps. They are used
    # only to skip historical status calls, never to admit a live job.
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _mine_job_snapshot(api: object, clock: Callable[[], datetime]) -> dict:
    snapshot_time = _utc(clock())
    jobs = []
    refs = set()
    ignored_placeholders = 0
    old_not_queried = 0
    for page in range(1, 1001):
        response = api.kernels_list_with_response(
            page=page, page_size=PAGE_SIZE, mine=True, sort_by="dateRun")
        if response is None or not isinstance(getattr(response, "kernels", None), list):
            raise RuntimeError("Official Kaggle mine-job listing is incomplete")
        for kernel in response.kernels:
            ref = getattr(kernel, "ref", None)
            if not ref:
                detail = kernel.to_dict() if hasattr(kernel, "to_dict") else vars(kernel)
                if (detail.get("title") == "[Private Notebook]"
                        and detail.get("lastRunTime") == "2010-04-01T00:00:00.000Z"):
                    ignored_placeholders += 1
                    continue
                raise RuntimeError("Kaggle mine-job listing has an unaddressable job")
            if not isinstance(ref, str) or ref.count("/") != 1 or ref in refs:
                raise RuntimeError("Kaggle mine-job listing has a repeated or invalid reference")
            refs.add(ref)
            last_run = _last_run(kernel)
            if (ref not in PRIOR_TERMINAL_JOBS and last_run is not None
                    and snapshot_time - last_run > timedelta(hours=24)):
                old_not_queried += 1
                continue
            jobs.append({"kernel_id": ref,
                         "status": _status(getattr(api.kernels_status(ref), "status", None)),
                         "source": "mine_job_listing"})
        if len(response.kernels) < PAGE_SIZE:
            break
    else:
        raise RuntimeError("Kaggle mine-job listing exceeded the safety page limit")
    for ref in PRIOR_TERMINAL_JOBS:
        if ref not in refs:
            jobs.append({"kernel_id": ref,
                         "status": _status(getattr(api.kernels_status(ref), "status", None)),
                         "source": "required_prior_job"})
    status_by_ref = {item["kernel_id"]: item["status"] for item in jobs}
    if any(status_by_ref.get(ref) != "COMPLETE" for ref in PRIOR_TERMINAL_JOBS):
        raise RuntimeError("BF16 SWE200, FP16 parent, or paired stage-1 job is not COMPLETE")
    if KERNEL_ID in refs:
        raise RuntimeError("The one-use evaluation kernel ID already exists")
    return {"source": "official_kaggle_mine_job_statuses",
            "observed_at_utc": _utc(clock()).isoformat(),
            "sort_by": "dateRun", "all_pages_read": True,
            "ignored_provider_placeholders": ignored_placeholders,
            "historical_over_24h_not_status_queried": old_not_queried,
            "jobs": jobs}


def _json_bytes(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _write_json(path: Path, value: dict) -> str:
    raw = _json_bytes(value)
    with path.open("xb") as sink:
        sink.write(raw)
    return sha256(raw).hexdigest()


def _reconciled_holds(path: Path, clock: Callable[[], datetime]) -> tuple[float, bytes]:
    raw = path.read_bytes()
    record = json.loads(raw)
    if record.get("schema") != "geocpu_kaggle_hold_reconciliation_v1":
        raise ValueError("Existing-hold reconciliation schema differs")
    checked = _utc(datetime.fromisoformat(record["checked_utc"].replace("Z", "+00:00")))
    age = (_utc(clock()) - checked).total_seconds()
    if not 0 <= age <= MAX_AGE_SECONDS:
        raise ValueError("Existing-hold reconciliation is stale or future-dated")
    holds = _seconds(record.get("unreturned_device_seconds"), "reconciled local hold")
    if holds == 0 and record.get("zero_hold_evidence") is not True:
        raise ValueError("A zero prior hold requires explicit reconciliation evidence")
    return holds, raw


def issue(*, api: object, output_dir: Path, hard_stop_at_utc: datetime,
          existing_holds_device_seconds: float = CONSERVATIVE_PRIOR_HOLDS_DEVICE_SECONDS,
          reconciliation_evidence: Path | None = None,
          swe_proof_dir: Path, fixed44_proof_dir: Path,
          clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc)) -> dict:
    """Create two verified claim bundles under one previously absent directory."""
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() or not output_dir.parent.is_dir():
        raise FileExistsError("Claim output must be a new directory under an existing parent")
    existing_holds = _seconds(existing_holds_device_seconds, "existing local hold")
    reconciliation_raw = None
    if existing_holds < CONSERVATIVE_PRIOR_HOLDS_DEVICE_SECONDS:
        if reconciliation_evidence is None:
            raise ValueError("Reducing the 15,840 device-second prior hold needs fresh reconciliation")
        reconciled, reconciliation_raw = _reconciled_holds(reconciliation_evidence, clock)
        if existing_holds != reconciled:
            raise ValueError("Existing hold differs from reconciliation evidence")
    elif reconciliation_evidence is not None:
        reconciled, reconciliation_raw = _reconciled_holds(reconciliation_evidence, clock)
        if existing_holds != reconciled:
            raise ValueError("Existing hold differs from reconciliation evidence")
    hard_stop = _utc(hard_stop_at_utc)
    swe_sha = _verify_original_swe_baseline(
        swe_proof_dir, expected_precision_profile="eval_bf16_native")
    fixed44_sha = _verify_pretrain_raw_diagnostic(
        fixed44_proof_dir, expected_precision_profile=PROFILE)

    response = api.quota_view()
    quota = getattr(response, "gpu_quota", None)
    if quota is None:
        raise RuntimeError("Official Kaggle GPU quota is missing")
    minimum = _seconds(getattr(quota, "minimum_time_allowed", None), "minimum")
    used = _seconds(getattr(quota, "time_used", None), "used")
    reserved = _seconds(getattr(quota, "time_reserved", None), "reserved")
    observed = _utc(clock())
    minimum_free = minimum - used - reserved
    if not math.isfinite(minimum_free) or minimum_free < existing_holds + PAIRED_DEVICE_SECONDS:
        raise RuntimeError("Official guaranteed free GPU time cannot cover existing holds and both arms")
    quota_evidence = {
        "source": "official_kaggle_free_quota",
        "official_endpoint": "GetAcceleratorQuotaStatistics",
        "observed_at_utc": observed.isoformat(),
        "minimum_time_allowed_device_seconds": minimum,
        "time_used_device_seconds": used,
        "time_reserved_device_seconds": reserved,
        "minimum_free_device_seconds": minimum_free,
        "existing_unreturned_hold_device_seconds": existing_holds,
        "available_after_existing_holds_device_seconds": minimum_free - existing_holds,
        "new_paired_reservation_device_seconds": PAIRED_DEVICE_SECONDS,
        "prior_hold_basis": ("fresh_reconciliation" if reconciliation_raw is not None
                             else "conservative_v3_ledger_floor"),
        "total_allowance_counted": False,
        "pay_to_scale_allowance_counted": False,
    }
    refresh = getattr(response, "quota_refresh_time", None)
    quota_evidence["quota_refresh_at_utc"] = (
        _utc(refresh).isoformat() if refresh and refresh.tzinfo else None)
    quota_evidence["quota_refresh_provider_unzoned"] = (
        refresh.isoformat() if refresh and not refresh.tzinfo else None)

    jobs = _mine_job_snapshot(api, clock)
    created = _utc(clock())
    if ((created - observed).total_seconds() > MAX_AGE_SECONDS
            or (created - datetime.fromisoformat(jobs["observed_at_utc"])).total_seconds()
            > MAX_AGE_SECONDS):
        raise RuntimeError("Official quota or job status became stale during claim issuance")
    if hard_stop <= created + timedelta(seconds=PER_ARM_DEVICE_SECONDS):
        raise ValueError("Hard stop does not leave the full run and save reserve")

    ids = {arm: "claim_" + uuid4().hex for arm in ("control", "python")}
    runs = {arm: "run_" + uuid4().hex for arm in ("control", "python")}
    stage = Path(tempfile.mkdtemp(prefix=".eval-claims-", dir=output_dir.parent))
    try:
        paths = {}
        for arm in ("control", "python"):
            arm_dir = stage / arm
            arm_dir.mkdir()
            quota_sha = _write_json(arm_dir / "quota_evidence.json", quota_evidence)
            jobs_sha = _write_json(arm_dir / "active_jobs_snapshot.json", jobs)
            ledger_sha = _write_json(arm_dir / "root_hold_ledger.json", {
                "claim_id": ids[arm], "reserved_device_seconds": PER_ARM_DEVICE_SECONDS,
                "status": "reserved_once_unspent", "paired_kernel_id": KERNEL_ID,
                "paired_claim_ids": sorted(ids.values()),
                "paired_reserved_device_seconds": PAIRED_DEVICE_SECONDS,
                "existing_unreturned_hold_device_seconds": existing_holds,
                "existing_holds_reconciliation_sha256": (
                    sha256(reconciliation_raw).hexdigest() if reconciliation_raw else None),
                "all_unreturned_holds_after_issue_device_seconds":
                    existing_holds + PAIRED_DEVICE_SECONDS,
            })
            claim = {
                "schema": SCHEMA, "root_authorized_once": True,
                "arm": arm, "run_id": runs[arm], "claim_id": ids[arm],
                "official_quota_observed_at_utc": observed.isoformat(),
                "active_jobs_observed_at_utc": jobs["observed_at_utc"],
                "root_claim_created_at_utc": created.isoformat(),
                "hard_stop_at_utc": hard_stop.isoformat(),
                "official_minimum_free_device_seconds": minimum_free,
                "root_reserved_device_seconds": PER_ARM_DEVICE_SECONDS,
                "bounded_run_seconds": RUN_SECONDS,
                "save_reserve_seconds": SAVE_RESERVE_SECONDS,
                "gpu_count": 1, "gpu_names": ["Tesla T4"],
                "precision_profile": PROFILE,
                "parent_swe200_baseline_complete_sha256": swe_sha,
                "parent_fixed44_raw_diagnostic_complete_sha256": fixed44_sha,
                "evidence": {
                    "official_quota_evidence": {"file": "quota_evidence.json", "sha256": quota_sha},
                    "active_jobs_snapshot": {"file": "active_jobs_snapshot.json", "sha256": jobs_sha},
                    "root_hold_ledger": {"file": "root_hold_ledger.json", "sha256": ledger_sha},
                },
            }
            if reconciliation_raw is not None:
                with (arm_dir / "existing_holds_reconciliation.json").open("xb") as sink:
                    sink.write(reconciliation_raw)
                claim["evidence"]["existing_holds_reconciliation"] = {
                    "file": "existing_holds_reconciliation.json",
                    "sha256": sha256(reconciliation_raw).hexdigest(),
                }
            path = arm_dir / "prelaunch_contract_once.json"
            _write_json(path, claim)
            verify_prelaunch_contract(path, now=created + timedelta(microseconds=1),
                                      expected_arm=arm, required_run_seconds=RUN_SECONDS,
                                      required_save_reserve_seconds=SAVE_RESERVE_SECONDS)
            paths[arm] = path
        control = json.loads(paths["control"].read_text(encoding="utf-8"))
        python = json.loads(paths["python"].read_text(encoding="utf-8"))
        _pair_claims(control, python, paths["control"], paths["python"],
                     swe_sha=swe_sha, fixed44_sha=fixed44_sha)
        if output_dir.exists():
            raise FileExistsError(output_dir)
        os.rename(stage, output_dir)
    finally:
        if stage.exists():
            if stage.resolve().parent != output_dir.parent.resolve():
                raise RuntimeError("Temporary claim directory escaped its parent")
            shutil.rmtree(stage)
    return {"output_dir": str(output_dir), "paired_kernel_id": KERNEL_ID,
            "control_claim": str(output_dir / "control" / "prelaunch_contract_once.json"),
            "python_claim": str(output_dir / "python" / "prelaunch_contract_once.json"),
            "parent_swe200_complete_sha256": swe_sha,
            "parent_fixed44_complete_sha256": fixed44_sha,
            "official_minimum_free_device_seconds": minimum_free,
            "existing_unreturned_hold_device_seconds": existing_holds,
            "paired_reserved_device_seconds": PAIRED_DEVICE_SECONDS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hard-stop-at-utc", required=True)
    parser.add_argument("--existing-holds-device-seconds", type=float,
                        default=CONSERVATIVE_PRIOR_HOLDS_DEVICE_SECONDS)
    parser.add_argument("--reconciliation-evidence", type=Path)
    parser.add_argument("--swe-proof-dir", type=Path, required=True)
    parser.add_argument("--fixed44-proof-dir", type=Path, required=True)
    args = parser.parse_args()
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    result = issue(api=api, output_dir=args.output_dir,
                   hard_stop_at_utc=datetime.fromisoformat(
                       args.hard_stop_at_utc.replace("Z", "+00:00")),
                   existing_holds_device_seconds=args.existing_holds_device_seconds,
                   reconciliation_evidence=args.reconciliation_evidence,
                   swe_proof_dir=args.swe_proof_dir,
                   fixed44_proof_dir=args.fixed44_proof_dir)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
