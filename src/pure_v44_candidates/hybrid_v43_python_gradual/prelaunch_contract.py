"""Fail-closed handoff from Root's fresh free-Kaggle resource check.

Kaggle quota is checked *before* a private version is created.  An already
running notebook cannot retroactively prove that free capacity was available
at launch, so this module requires the Root's fresh provider/ledger/job-status
evidence copied into the job and rechecks their bytes and declared bounds.
The in-job copy cannot atomically consume the shared claim across multiple
jobs. Root must consume it in the external ledger before a serialized submit.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re

from source_pools import file_sha


SCHEMA = "geocpu_hybrid_private_kaggle_prelaunch_v1"
MAX_AGE_SECONDS = 600


def _utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Prelaunch timestamp lacks UTC offset")
    return parsed.astimezone(timezone.utc)


def verify_prelaunch_contract(path: str | Path, *, now: datetime | None = None,
                              expected_arm: str | None = None,
                              required_run_seconds: int | None = None,
                              required_save_reserve_seconds: int | None = None) -> dict:
    path = Path(path)
    record = json.loads(path.read_text(encoding="utf-8"))
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    if record.get("schema") != SCHEMA or record.get("root_authorized_once") is not True:
        raise ValueError("A one-use Root launch claim is missing")
    if expected_arm is not None and record.get("arm") != expected_arm:
        raise ValueError("The launch claim belongs to another arm")
    if (not isinstance(record.get("run_id"), str)
            or re.fullmatch(r"[A-Za-z0-9_-]{16,128}", record["run_id"]) is None
            or type(record.get("gpu_count")) is not int or not 1 <= record["gpu_count"] <= 4):
        raise ValueError("Root run ID or GPU count is invalid")
    if (record.get("precision_profile") not in {"train_bf16_native", "train_fp16_gradscaler_v1",
                                                "eval_bf16_native", "eval_fp16_native_v1"}
            or not isinstance(record.get("gpu_names"), list)
            or len(record["gpu_names"]) != record["gpu_count"]
            or any(not isinstance(name, str) or not name.strip() for name in record["gpu_names"])):
        raise ValueError("Root claim lacks a fixed precision and GPU model profile")
    observed = _utc(record["official_quota_observed_at_utc"])
    if not 0 <= (now - observed).total_seconds() <= MAX_AGE_SECONDS:
        raise ValueError("Official fresh free quota observation is stale")
    if not 0 < (now - _utc(record["root_claim_created_at_utc"])).total_seconds() <= MAX_AGE_SECONDS:
        raise ValueError("Root one-use claim is stale or future dated")
    if _utc(record["hard_stop_at_utc"]) <= now:
        raise ValueError("The predeclared hard stop has already passed")
    required_files = ("official_quota_evidence", "root_hold_ledger", "active_jobs_snapshot")
    evidence = record.get("evidence", {})
    for name in required_files:
        item = evidence.get(name)
        if (not isinstance(item, dict) or not isinstance(item.get("file"), str)
                or not isinstance(item.get("sha256"), str)):
            raise ValueError("A required prelaunch evidence file is missing")
        relative = Path(item["file"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Prelaunch evidence must stay inside its package")
        if file_sha(path.parent / relative) != item["sha256"]:
            raise ValueError("Prelaunch resource evidence SHA differs: " + name)
    quota = json.loads((path.parent / evidence["official_quota_evidence"]["file"]).read_text(encoding="utf-8"))
    ledger = json.loads((path.parent / evidence["root_hold_ledger"]["file"]).read_text(encoding="utf-8"))
    jobs = json.loads((path.parent / evidence["active_jobs_snapshot"]["file"]).read_text(encoding="utf-8"))
    minimum = record.get("official_minimum_free_device_seconds")
    reserve = record.get("root_reserved_device_seconds")
    run_seconds = record.get("bounded_run_seconds")
    save_seconds = record.get("save_reserve_seconds")
    if (type(run_seconds) is not int or type(save_seconds) is not int
            or not 900 <= run_seconds <= 3600 or save_seconds < 120
            or (required_run_seconds is not None and run_seconds != required_run_seconds)
            or (required_save_reserve_seconds is not None and save_seconds != required_save_reserve_seconds)):
        raise ValueError("Predeclared run/save duration differs from the actual entry")
    if (type(minimum) not in (int, float) or type(reserve) not in (int, float)
            or minimum < reserve or reserve < (run_seconds + save_seconds) * record["gpu_count"]):
        raise ValueError("Official free minimum or Root hold cannot cover this bounded stage")
    if (quota.get("source") != "official_kaggle_free_quota"
            or quota.get("observed_at_utc") != record["official_quota_observed_at_utc"]
            or quota.get("minimum_free_device_seconds") != minimum):
        raise ValueError("Official quota raw evidence and claim differ")
    claim_id = record.get("claim_id")
    if (not isinstance(claim_id, str) or re.fullmatch(r"[A-Za-z0-9_-]{16,128}", claim_id) is None
            or ledger.get("claim_id") != claim_id
            or ledger.get("reserved_device_seconds") != reserve
            or ledger.get("status") != "reserved_once_unspent"):
        raise ValueError("Root hold ledger lacks an unspent one-use reservation")
    if (jobs.get("observed_at_utc") != record.get("active_jobs_observed_at_utc")
            or not isinstance(jobs.get("jobs"), list)
            or any(not isinstance(item, dict) or item.get("status") not in
                   {"COMPLETE", "CANCELLED", "ERROR", "FAILED"} for item in jobs["jobs"])):
        raise ValueError("Another Kaggle job is active or its status is unknown")
    jobs_age = (now - _utc(jobs["observed_at_utc"])).total_seconds()
    if not 0 <= jobs_age <= MAX_AGE_SECONDS:
        raise ValueError("Active-job snapshot is stale")
    return record
