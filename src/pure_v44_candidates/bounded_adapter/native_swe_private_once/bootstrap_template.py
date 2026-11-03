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


EMBEDDED_SOURCE = __EMBEDDED_SOURCE__
DATASET_SLUG = "geocpu-v43-v44fix-paired-eval-20261103"
EXPECTED_DATASET_ID = "geocpuresearch/" + DATASET_SLUG
V43_ZIP_SHA256 = "0102dc9b01b4b795c3123b18cefc7a8734e47a06dd85292af464d3232704d198"
V43_ZIP_BYTES = 67_976_709
PARENT_SHA256 = "a4b95d7d0f486146196271b3b1fbc6a1b6b3342d8fadd590320fc3842c1e80e5"
PARENT_BYTES = 990_545_719
META_SHA256 = "7efa8dea9a0100e2250f72805c89cca7a281efec30b4479795743814756cdcd5"
SWE_SHA256 = "0bf62099ea2a41c8eb35c1ebdc2584f2c164ae3c82f741ab3eca44113f70f8cd"
EVALUATOR_SHA256 = "c3589b97f531b6460a95907e56284df5f20be990eff1a4d2e56a921b4c705899"
HARD_STOP_UTC = datetime.fromisoformat("2026-11-03T18:00:00+00:00")
MAX_WALL_SECONDS = 1800
SAVE_RESERVE_SECONDS = 120
MIN_START_SECONDS = 900
ROOT = Path("/kaggle/working/geocpu_v43_native_swe200_once_20261103")


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
