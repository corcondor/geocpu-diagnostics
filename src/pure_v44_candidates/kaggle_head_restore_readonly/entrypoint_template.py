"""Single Kaggle kernel entrypoint. The source ZIP SHA is filled at packaging."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import zipfile

SOURCE_ZIP_NAME = "kaggle_pure_head_restore_sources_v1.zip"
EXPECTED_SOURCE_SHA256 = "__BUNDLE_SHA256__"
BUDGET_RECEIPT = None  # prelaunch_gate.py fills this only after fresh official checks
INPUT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
EXISTING_JOBS = {
    "geocpuresearch/geocpu-v43-missing-only-resume-b4-20261103",
    "geocpuresearch/geocpu-v44fix-missing-only-resume-b4-v4-20261103",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    receipt = BUDGET_RECEIPT
    if not isinstance(receipt, dict) or receipt.get("schema") != "geocpu_kaggle_fresh_gpu_budget_v1":
        raise RuntimeError("A fresh official quota receipt is required before submission")
    if receipt.get("source_bundle_sha256") != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("Quota receipt targets a different source bundle")
    outer_seconds = receipt.get("outer_wall_seconds")
    if type(outer_seconds) is not int or not 900 <= outer_seconds <= 28_800:
        raise RuntimeError("Quota receipt has no bounded wall time")
    checked = datetime.fromisoformat(receipt["checked_utc"])
    age = (datetime.now(timezone.utc) - checked).total_seconds()
    if not 0 <= age <= 3_600:
        raise RuntimeError("Quota receipt is stale or dated in the future")
    statuses = receipt.get("existing_job_statuses", {})
    if set(statuses) != EXISTING_JOBS or any(
            value not in ("complete", "error", "cancelled") for value in statuses.values()):
        raise RuntimeError("An existing Kaggle GPU job still holds quota")
    if (receipt.get("pay_to_scale_allowance_counted") is not False
            or not isinstance(receipt.get("root_hold_ledger_sha256"), str)
            or len(receipt["root_hold_ledger_sha256"]) != 64
            or receipt.get("reserved_device_count") != 2):
        raise RuntimeError("Receipt does not certify free guaranteed quota")
    available = (receipt["gpu_minimum_guaranteed_device_seconds"]
                 - receipt["gpu_used_device_seconds"]
                 - receipt["gpu_official_reserved_device_seconds"]
                 - receipt["root_unreturned_hold_device_seconds"]
                 - receipt["safety_reserve_device_seconds"])
    if outer_seconds != min(28_800, int(available // 2)):
        raise RuntimeError("Receipt wall time differs from the guaranteed free budget")
    if not INPUT.is_dir() or not WORKING.is_dir():
        raise RuntimeError("Kaggle input and working directories are required")
    import torch
    if not 1 <= torch.cuda.device_count() <= receipt.get("reserved_device_count", 0):
        raise RuntimeError("Actual GPU device count exceeds the official quota reservation")
    packages = sorted(INPUT.rglob(SOURCE_ZIP_NAME))
    if len(packages) != 1 or sha256(packages[0]) != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("Exactly one pinned source package is required")
    with zipfile.ZipFile(packages[0]) as archive:
        manifest = json.loads(archive.read("source_manifest.json"))
        required = tuple(manifest["asset_paths"][name]
                         for name in ("v43_source", "pure_source", "parent", "child",
                                      "dataset_manifest"))
        roots = [root for root in INPUT.iterdir()
                 if root.is_dir() and all((root / path).is_file() for path in required)]
        if len(roots) != 1:
            raise RuntimeError("Exactly one asset dataset root with the original paths is required")
        source = WORKING / "geocpu_pure_head_source_v1"
        source.mkdir(exist_ok=True)
        for name in archive.namelist():
            member = PurePosixPath(name)
            if member.is_absolute() or ".." in member.parts:
                raise RuntimeError("Unsafe source member: " + name)
            data = archive.read(name)
            path = source.joinpath(*member.parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                if path.read_bytes() != data:
                    raise RuntimeError("Existing source member differs: " + name)
            else:
                path.write_bytes(data)
    output = WORKING / "pure_head_restore_v1"
    previous = [path.parent for path in INPUT.rglob("PARTIAL.json")
                if path.parent.name == "pure_head_restore_v1"
                and (path.parent / "claim.json").is_file()
                and not (path.parent / "COMPLETE.json").exists()]
    expected_previous = receipt.get("resume_kernel_source") is not None
    if len(previous) != int(expected_previous):
        raise RuntimeError("Prior PARTIAL output does not match explicit resume selection")
    if previous:
        if output.exists():
            raise RuntimeError("Working output exists before PARTIAL restore")
        shutil.copytree(previous[0], output)
    subprocess.run(["bash", "launch.sh", str(outer_seconds), "--assets-root", str(roots[0]),
                    "--output-dir", str(output)],
                   cwd=source, check=True)


if __name__ == "__main__":
    main()
