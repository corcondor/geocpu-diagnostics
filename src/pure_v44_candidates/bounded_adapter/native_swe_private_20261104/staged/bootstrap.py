"""One-use private Kaggle entry for the original v43 SWE200 capture.

This file is rendered with three pinned source modules embedded. It never
submits a Kaggle job, trains a model, or writes into a mounted input dataset.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import sys
import zipfile
import zlib
from datetime import datetime, timedelta, timezone
from pathlib import Path


EMBEDDED_SOURCE = {'adapter.py': {'bytes': 8824, 'sha256': '3f9c3e2bf676f2e53eb66d57f0a8152297ca01fafe608738fe3f9cd3c0a4fe68', 'zlib_base64': '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'}, 'admission.py': {'bytes': 28764, 'sha256': '0fbcf14fc08a6d3a0120c0ba063e598e7e1f26e4f62e8419333522f13fdfcd59', 'zlib_base64': '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'}, 'v43_native_swe_capture_readonly.py': {'bytes': 18824, 'sha256': '6bcf453c32cacff89d3e6b3062162c3ae43f6027fe8f6b3336c8c2c8740fc12d', 'zlib_base64': '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'}}
DATASET_SLUG = "geocpu-v43-v44fix-paired-eval-20261103"
EXPECTED_DATASET_ID = "geocpuresearch/" + DATASET_SLUG
V43_ZIP_SHA256 = "0102dc9b01b4b795c3123b18cefc7a8734e47a06dd85292af464d3232704d198"
V43_ZIP_BYTES = 67_976_709
PARENT_SHA256 = "a4b95d7d0f486146196271b3b1fbc6a1b6b3342d8fadd590320fc3842c1e80e5"
PARENT_BYTES = 990_545_719
META_SHA256 = "7efa8dea9a0100e2250f72805c89cca7a281efec30b4479795743814756cdcd5"
SWE_SHA256 = "0bf62099ea2a41c8eb35c1ebdc2584f2c164ae3c82f741ab3eca44113f70f8cd"
EVALUATOR_SHA256 = "c3589b97f531b6460a95907e56284df5f20be990eff1a4d2e56a921b4c705899"
HARD_STOP_UTC = datetime.fromisoformat("2026-11-04T03:00:00+00:00")
MAX_WALL_SECONDS = 1800
SAVE_RESERVE_SECONDS = 120
MIN_START_SECONDS = 900
ROOT = Path("/kaggle/working/geocpu_v43_native_swe200_once_20261104")


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_new(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as out:
        json.dump(value, out, ensure_ascii=False, indent=2, sort_keys=True)
        out.write("\n")
        out.flush()
        os.fsync(out.fileno())


def input_root() -> Path:
    candidates = (Path("/kaggle/input/datasets/geocpuresearch") / DATASET_SLUG,
                  Path("/kaggle/input") / DATASET_SLUG)
    found = [path for path in candidates if path.is_dir()]
    if len(found) != 1:
        raise FileNotFoundError("Expected one mounted private v43 asset dataset")
    return found[0]


def pinned_inputs(dataset: Path) -> tuple[Path, Path]:
    zips = [path for name in ("v43_source.zip", "v43_source.zip.bin")
            if (path := dataset / name).is_file()]
    if len(zips) != 1:
        raise ValueError("Exactly one original v43 source ZIP is required")
    source_zip = zips[0]
    checkpoint = dataset / "v43_step15000.pt"
    for path, size, expected in ((source_zip, V43_ZIP_BYTES, V43_ZIP_SHA256),
                                 (checkpoint, PARENT_BYTES, PARENT_SHA256)):
        if path.stat().st_size != size or sha_file(path) != expected:
            raise ValueError("Mounted original v43 asset differs: " + path.name)
    with zipfile.ZipFile(source_zip) as archive:
        for name, expected in (("base_metadata.json", META_SHA256),
                               ("data/eval_swe_step_control_heldout.jsonl", SWE_SHA256),
                               ("eval_step_control_heldout.py", EVALUATOR_SHA256)):
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError("Original v43 ZIP member differs: " + name)
    return source_zip, checkpoint


def materialize_pinned_source(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    for name, descriptor in EMBEDDED_SOURCE.items():
        if name not in {"adapter.py", "admission.py", "v43_native_swe_capture_readonly.py"}:
            raise ValueError("Unrecognized embedded source")
        raw = zlib.decompress(base64.b64decode(descriptor["zlib_base64"], validate=True))
        if len(raw) != descriptor["bytes"] or hashlib.sha256(raw).hexdigest() != descriptor["sha256"]:
            raise ValueError("Embedded source SHA differs: " + name)
        (destination / name).write_bytes(raw)


def main() -> None:
    if not Path("/kaggle").is_dir() or Path("/content/drive/MyDrive").exists():
        raise RuntimeError("This read-only capture requires private Kaggle without mounted Drive")
    now = datetime.now(timezone.utc)
    if now >= HARD_STOP_UTC - timedelta(seconds=MIN_START_SECONDS):
        raise RuntimeError("Predeclared hard stop leaves less than fifteen minutes; do not launch")
    dataset = input_root()
    source_zip, checkpoint = pinned_inputs(dataset)
    if ROOT.exists():
        raise FileExistsError("Single-use output directory already exists")
    ROOT.mkdir(parents=True)
    source_dir = ROOT / "pinned_code"
    materialize_pinned_source(source_dir)
    write_new(ROOT / "entry_receipt.json", {
        "status": "INPUTS_VERIFIED_NOT_YET_COMPLETE",
        "source_dataset": EXPECTED_DATASET_ID,
        "source_zip_sha256": V43_ZIP_SHA256,
        "parent_sha256": PARENT_SHA256,
        "metadata_sha256": META_SHA256,
        "swe200_sha256": SWE_SHA256,
        "original_evaluator_sha256": EVALUATOR_SHA256,
        "hard_stop_utc": HARD_STOP_UTC.isoformat(),
        "max_wall_seconds": MAX_WALL_SECONDS,
        "training_updates": 0,
        "source_modules": {name: value["sha256"] for name, value in EMBEDDED_SOURCE.items()},
    })
    result_dir = ROOT / "actual"
    command = [sys.executable, str(source_dir / "v43_native_swe_capture_readonly.py"),
               "--source-zip", str(source_zip), "--parent", str(checkpoint),
               "--output-dir", str(result_dir)]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(source_dir)
    seconds_left = (HARD_STOP_UTC - datetime.now(timezone.utc)).total_seconds()
    timeout = int(min(MAX_WALL_SECONDS, seconds_left - SAVE_RESERVE_SECONDS))
    if timeout < MIN_START_SECONDS:
        raise RuntimeError("Input verification consumed the declared execution window")
    try:
        with (ROOT / "native_capture.log").open("x", encoding="utf-8") as log:
            completed = subprocess.run(command, cwd=source_dir, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        write_new(ROOT / "INCOMPLETE_TIMEOUT.json", {
            "status": "INCOMPLETE_TIMEOUT", "training_updates": 0,
            "hard_stop_utc": HARD_STOP_UTC.isoformat(), "timeout_seconds": timeout})
        raise
    if completed.returncode != 0:
        write_new(ROOT / "INCOMPLETE_ERROR.json", {
            "status": "INCOMPLETE_ERROR", "training_updates": 0,
            "returncode": completed.returncode})
        raise RuntimeError("Original v43 native SWE200 capture failed; inspect native_capture.log")
    complete_path = result_dir / "COMPLETE.json"
    result = json.loads(complete_path.read_text(encoding="utf-8"))
    if (result.get("status") != "COMPLETE" or result.get("cases") != 200
            or result.get("source_zip_sha256") != V43_ZIP_SHA256
            or result.get("parent_sha256") != PARENT_SHA256
            or result.get("eval_member_sha256") != SWE_SHA256
            or result.get("original_evaluator_sha256") != EVALUATOR_SHA256
            or result.get("generation_limit") != 2048
            or result.get("training_updates") != 0):
        raise ValueError("Native capture completion record is incomplete")
    for name, expected in result["files_sha256"].items():
        if "/" in name or "\\" in name or sha_file(result_dir / name) != expected:
            raise ValueError("Native capture output SHA differs: " + name)
    write_new(ROOT / "ENTRY_COMPLETE.json", {
        "status": "COMPLETE", "swe200_cases": 200,
        "native_result_sha256": sha_file(complete_path),
        "originally_correct_and_natural_eos": result["originally_correct_and_natural_eos"],
        "training_updates": 0,
    })
    print(json.dumps({"status": "COMPLETE", "result_dir": str(result_dir),
                      "strict_old_skill_rows": result["eligible_old_skill_rows"]},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
