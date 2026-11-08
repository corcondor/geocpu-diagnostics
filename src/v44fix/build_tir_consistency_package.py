"""Build an immutable local-only release for the isolated v44fix gate.

This script does not open Colab, start learning, or write to Google Drive.
It refuses to overwrite an existing release directory.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

from resume_v44fix import PARENT_BYTES, PARENT_SHA256, sha256_file


ROOT = Path(__file__).resolve().parent
RELEASE = ROOT / "release_v1"
ZIP_NAME = "GeoCPU_v44fix_TIR_consistency_bundle.zip"
SOURCE_SHA256 = "0102dc9b01b4b795c3123b18cefc7a8734e47a06dd85292af464d3232704d198"


def members() -> list[Path]:
    entries = [p for p in ROOT.iterdir() if p.is_file() and p.suffix in (".py", ".json", ".md")]
    entries += [p for folder in (ROOT / "data", ROOT / "gpt") for p in folder.rglob("*") if p.is_file()]
    return sorted(entries, key=lambda path: path.relative_to(ROOT).as_posix())


def launch_source(zip_sha: str, manifest_sha: str) -> str:
    return f'''# Paste this entire cell into the existing Colab GPU session. Gate trial only.
import hashlib, json, os, sys, zipfile
from pathlib import Path
from google.colab import drive
if not Path("/content/drive/MyDrive").is_dir():
    drive.mount("/content/drive")
archive_path = Path("/content/drive/MyDrive/GeoCPU_v44fix_TIR_consistency_20261103/{ZIP_NAME}")
expected_zip_sha = "{zip_sha}"
expected_manifest_sha = "{manifest_sha}"
def digest_file(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()
if not archive_path.is_file() or digest_file(archive_path) != expected_zip_sha:
    raise RuntimeError("The isolated v44fix bundle is missing or changed")
workspace = Path("/content/v44fix_tir_consistency_workspace_20261103")
with zipfile.ZipFile(archive_path) as archive:
    manifest_bytes = archive.read("release_manifest.json")
    if hashlib.sha256(manifest_bytes).hexdigest() != expected_manifest_sha:
        raise RuntimeError("Release manifest changed")
    manifest = json.loads(manifest_bytes)
    if manifest.get("training_ready") is not True:
        raise RuntimeError("This audit release is blocked; no Colab training or evaluation is authorized")
    expected_names = set(manifest["files"]) | {{"release_manifest.json"}}
    if set(archive.namelist()) != expected_names:
        raise RuntimeError("Bundle member list changed")
    for name, spec in manifest["files"].items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or "\\\\" in name or ":" in name:
            raise RuntimeError("Unsafe bundle member")
        raw = archive.read(name)
        if len(raw) != spec["bytes"] or hashlib.sha256(raw).hexdigest() != spec["sha256"]:
            raise RuntimeError("Bundle file hash changed: " + name)
    if workspace.exists():
        for name, spec in manifest["files"].items():
            local = workspace / name
            if not local.is_file() or local.stat().st_size != spec["bytes"] or digest_file(local) != spec["sha256"]:
                raise RuntimeError("Existing Colab workspace differs; use a fresh runtime")
    else:
        workspace.mkdir(parents=True, exist_ok=False)
        archive.extractall(workspace)
        for name, spec in manifest["files"].items():
            local = workspace / name
            if local.stat().st_size != spec["bytes"] or digest_file(local) != spec["sha256"]:
                raise RuntimeError("Extracted file hash changed: " + name)
os.chdir(workspace)
sys.path.insert(0, str(workspace))
import colab_v44fix
colab_v44fix.main(full=False)
'''


def main() -> None:
    if RELEASE.exists():
        raise FileExistsError("Release already exists; create a new version rather than overwrite")
    files = members()
    names = [path.relative_to(ROOT).as_posix() for path in files]
    if len(names) != len(set(names)) or not all(name and ".." not in Path(name).parts for name in names):
        raise ValueError("Unsafe or duplicate release members")
    required = {"build_v44fix_data.py", "runtime_v44fix.py", "resume_v44fix.py",
                "tir_masking_v44fix.py", "gate_eval_v44fix.py", "colab_v44fix.py",
                "resource_limited_worker_v44r1.py",
                "test_resume_v44fix.py", "test_tir_consistency_v44fix.py",
                "test_colab_stage_v44fix.py", "test_tool_authority_v44fix.py",
                "gpt/prepare_public_v44.py", "gpt/resource_limited_worker_v44r1.py",
                "gpt/public_source_lock_v44.json"}
    if not required <= set(names):
        raise ValueError("Release lacks a required source or fixture")
    manifest = {"schema": "geocpu_v44fix_tir_consistency_release_v1",
                "mode": "independent_gate_then_separate_full_run",
                "training_executed_in_packaging": False,
                "training_ready": False,
                "training_admission": "Colab must first execute and validate every selected public TIR observation; no such run is claimed by this release",
                "v43_source_zip_sha256": SOURCE_SHA256,
                "v43_checkpoint_sha256": PARENT_SHA256,
                "v43_checkpoint_bytes": PARENT_BYTES,
                "output_folder": "GeoCPU_v44fix_TIR_consistency_20261103/run_1",
                "files": {name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
                          for name, path in zip(names, files)}}
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    RELEASE.mkdir(parents=False, exist_ok=False)
    zip_path = RELEASE / ZIP_NAME
    with zipfile.ZipFile(zip_path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, path in zip(names, files):
            archive.write(path, arcname=name)
        archive.writestr("release_manifest.json", manifest_bytes)
    (RELEASE / "release_manifest.json").write_bytes(manifest_bytes)
    zip_sha = sha256_file(zip_path)
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    (RELEASE / "Colab_one_cell_gate.py").write_text(
        launch_source(zip_sha, manifest_sha), encoding="utf-8")
    receipt = {"schema": "geocpu_v44fix_tir_consistency_package_receipt_v1",
               "zip": {"name": ZIP_NAME, "bytes": zip_path.stat().st_size, "sha256": zip_sha},
               "manifest": {"bytes": len(manifest_bytes), "sha256": manifest_sha},
               "members": len(names) + 1, "training_run": False, "colab_run": False,
               "drive_written": False}
    (RELEASE / "package_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
