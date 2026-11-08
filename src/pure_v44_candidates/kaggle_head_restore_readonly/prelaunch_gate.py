"""Fail-closed official Kaggle GPU quota gate; prepares but never pushes a job."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from package_sources import ENTRYPOINT_NAME, HERE, OUTPUT

EXISTING_JOBS = (
    "geocpuresearch/geocpu-v43-missing-only-resume-b4-20261103",
    "geocpuresearch/geocpu-v44fix-missing-only-resume-b4-v4-20261103",
)
TERMINAL = {"complete", "error", "cancelled"}
DEVICE_COUNT_RESERVED = 2
MAX_OUTER_SECONDS = 28_800
MIN_OUTER_SECONDS = 900
RESERVE_DEVICE_SECONDS = 3_600


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_status(status: object) -> str:
    return str(status).rsplit(".", 1)[-1].lower()


def select_budget(*, minimum_device_seconds: float, used_device_seconds: float,
                  reserved_device_seconds: float, unreturned_hold_device_seconds: float,
                  existing_job_statuses: dict[str, str]) -> int:
    if set(existing_job_statuses) != set(EXISTING_JOBS):
        raise RuntimeError("Existing GPU job set differs")
    if any(status not in TERMINAL for status in existing_job_statuses.values()):
        raise RuntimeError("An existing GPU job is still running or its status is unknown")
    values = (minimum_device_seconds, used_device_seconds,
              reserved_device_seconds, unreturned_hold_device_seconds)
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise RuntimeError("Official GPU quota values are invalid")
    remaining = (minimum_device_seconds - used_device_seconds
                 - reserved_device_seconds - unreturned_hold_device_seconds)
    available_wall = int((remaining - RESERVE_DEVICE_SECONDS) // DEVICE_COUNT_RESERVED)
    outer = min(MAX_OUTER_SECONDS, available_wall)
    if outer < MIN_OUTER_SECONDS:
        raise RuntimeError("Official remaining GPU quota is too small for a bounded attempt")
    return outer


def load_hold_ledger(path: Path) -> tuple[float, str]:
    """Bind every unreturned local hold to a checked evidence file."""
    path = path.resolve(strict=True)
    ledger_sha = sha256(path)
    ledger = json.loads(path.read_text(encoding="utf-8"))
    if ledger.get("schema") != "geocpu_kaggle_unreturned_holds_v1":
        raise RuntimeError("Root hold ledger schema differs")
    reconciled = datetime.fromisoformat(ledger["reconciled_utc"])
    if reconciled.tzinfo is None or not 0 <= (datetime.now(timezone.utc) - reconciled).total_seconds() <= 3_600:
        raise RuntimeError("Root hold ledger was not freshly reconciled")
    holds = ledger.get("holds")
    if not isinstance(holds, list):
        raise RuntimeError("Root hold ledger has no hold list")
    reconciliation_path = Path(ledger["reconciliation_evidence_path"])
    if not reconciliation_path.is_absolute():
        reconciliation_path = path.parent / reconciliation_path
    reconciliation_path = reconciliation_path.resolve(strict=True)
    if sha256(reconciliation_path) != ledger.get("reconciliation_evidence_sha256"):
        raise RuntimeError("Root hold reconciliation evidence SHA differs")
    reconciliation = json.loads(reconciliation_path.read_text(encoding="utf-8"))
    if reconciliation.get("schema") != "geocpu_kaggle_hold_reconciliation_v1":
        raise RuntimeError("Root hold reconciliation schema differs")
    checked = datetime.fromisoformat(reconciliation["checked_utc"])
    if checked.tzinfo is None or not 0 <= (datetime.now(timezone.utc) - checked).total_seconds() <= 3_600:
        raise RuntimeError("Root hold reconciliation is stale")
    total = 0.0
    ids = set()
    for hold in holds:
        if not isinstance(hold, dict) or not isinstance(hold.get("id"), str) or hold["id"] in ids:
            raise RuntimeError("Root hold ledger has an invalid or repeated hold")
        ids.add(hold["id"])
        value = hold.get("unreturned_device_seconds")
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise RuntimeError("Root hold ledger has an invalid device-second amount")
        evidence_path = Path(hold["evidence_path"])
        if not evidence_path.is_absolute():
            evidence_path = path.parent / evidence_path
        evidence_path = evidence_path.resolve(strict=True)
        if sha256(evidence_path) != hold.get("evidence_sha256"):
            raise RuntimeError("Root hold evidence SHA differs")
        total += value
    if reconciliation.get("unreturned_device_seconds") != total:
        raise RuntimeError("Root reconciliation and hold ledger totals differ")
    if not holds and reconciliation.get("zero_hold_evidence") is not True:
        raise RuntimeError("Zero independent holds have no explicit reconciliation evidence")
    return total, ledger_sha


def fresh_official_receipt(source_sha256: str, hold_ledger: Path) -> dict:
    from kaggle.api.kaggle_api_extended import KaggleApi

    root_holds, ledger_sha = load_hold_ledger(hold_ledger)
    api = KaggleApi()
    api.authenticate()
    # The official endpoint must succeed in this invocation. No cached or
    # estimated quota is accepted if Kaggle returns an error.
    response = api.quota_view()
    quota = response.gpu_quota
    if (quota is None or quota.time_used is None or quota.time_reserved is None
            or quota.minimum_time_allowed is None):
        raise RuntimeError("Official Kaggle GPU quota response is incomplete")
    statuses = {slug: normalize_status(api.kernels_status(slug).status)
                for slug in EXISTING_JOBS}
    minimum = quota.minimum_time_allowed.total_seconds()
    used = quota.time_used.total_seconds()
    reserved = quota.time_reserved.total_seconds()
    outer = select_budget(minimum_device_seconds=minimum, used_device_seconds=used,
                          reserved_device_seconds=reserved,
                          unreturned_hold_device_seconds=root_holds,
                          existing_job_statuses=statuses)
    return {
        "schema": "geocpu_kaggle_fresh_gpu_budget_v1",
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "official_endpoint": "GetAcceleratorQuotaStatistics",
        "quota_refresh_utc": (response.quota_refresh_time.isoformat()
                              if response.quota_refresh_time else None),
        "gpu_minimum_guaranteed_device_seconds": minimum,
        "gpu_used_device_seconds": used,
        "gpu_official_reserved_device_seconds": reserved,
        "root_unreturned_hold_device_seconds": root_holds,
        "root_hold_ledger_sha256": ledger_sha,
        "free_guaranteed_device_seconds_after_holds": minimum - used - reserved - root_holds,
        "pay_to_scale_allowance_counted": False,
        "reserved_device_count": DEVICE_COUNT_RESERVED,
        "untouched_existing_jobs": list(EXISTING_JOBS),
        "existing_job_statuses": statuses,
        "safety_reserve_device_seconds": RESERVE_DEVICE_SECONDS,
        "outer_wall_seconds": outer,
        "source_bundle_sha256": source_sha256,
    }


def prepare(submission_dir: Path, *, kernel_id: str,
            source_dataset: str, asset_dataset: str,
            hold_ledger: Path,
            resume_kernel_source: str | None = None) -> dict:
    if kernel_id in EXISTING_JOBS or kernel_id.count("/") != 1:
        raise ValueError("Use a new isolated Kaggle kernel ID")
    if (source_dataset.count("/") != 1 or asset_dataset.count("/") != 1
            or source_dataset == asset_dataset):
        raise ValueError("Two distinct source and asset Dataset IDs are required")
    if submission_dir.exists():
        raise FileExistsError(submission_dir)
    if not OUTPUT.is_file():
        raise FileNotFoundError("Build the small source ZIP first")
    source_sha = sha256(OUTPUT)
    receipt = fresh_official_receipt(source_sha, hold_ledger)
    receipt["resume_kernel_source"] = resume_kernel_source
    receipt["kernel_id"] = kernel_id
    receipt["source_dataset"] = source_dataset
    receipt["asset_dataset"] = asset_dataset
    template = (HERE / "entrypoint_template.py").read_text(encoding="utf-8")
    if template.count("__BUNDLE_SHA256__") != 1 or template.count("BUDGET_RECEIPT = None") != 1:
        raise RuntimeError("Entrypoint template no longer has unique receipt fields")
    entrypoint = template.replace("__BUNDLE_SHA256__", source_sha).replace(
        "BUDGET_RECEIPT = None", "BUDGET_RECEIPT = " + repr(receipt))
    compile(entrypoint, ENTRYPOINT_NAME, "exec")
    metadata = {
        "id": kernel_id, "title": "Pure v44 head restore read only matched64",
        "code_file": ENTRYPOINT_NAME, "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_internet": False,
        "dataset_sources": [source_dataset, asset_dataset],
        "competition_sources": [],
        "kernel_sources": [resume_kernel_source] if resume_kernel_source else [],
        "model_sources": [],
    }
    submission_dir.mkdir(parents=True, exist_ok=False)
    (submission_dir / ENTRYPOINT_NAME).write_text(entrypoint, encoding="utf-8")
    (submission_dir / "budget_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    (submission_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission-dir", type=Path, required=True)
    parser.add_argument("--kernel-id", required=True)
    parser.add_argument("--source-dataset", required=True)
    parser.add_argument("--asset-dataset", required=True)
    parser.add_argument("--hold-ledger", type=Path, required=True)
    parser.add_argument("--resume-kernel-source")
    args = parser.parse_args()
    receipt = prepare(args.submission_dir.resolve(), kernel_id=args.kernel_id,
                      source_dataset=args.source_dataset,
                      asset_dataset=args.asset_dataset,
                      hold_ledger=args.hold_ledger,
                      resume_kernel_source=args.resume_kernel_source)
    print("FRESH_OFFICIAL_QUOTA_GATE_PASS", receipt["outer_wall_seconds"],
          args.submission_dir.resolve(), flush=True)
    print("Submission prepared only; no Kaggle kernel was pushed", flush=True)


if __name__ == "__main__":
    main()
