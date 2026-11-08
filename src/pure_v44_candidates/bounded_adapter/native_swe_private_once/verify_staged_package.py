"""Read-only local closure check for the staged private Kaggle source package."""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import importlib.util
import json
import sys
import zipfile
import zlib
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify(asset_dir: Path) -> dict:
    manifest = json.loads((HERE / "package_manifest.json").read_text(encoding="utf-8"))
    staged = HERE / "staged"
    for name, descriptor in manifest["files"].items():
        path = staged / name
        if path.stat().st_size != descriptor["bytes"] or digest(path) != descriptor["sha256"]:
            raise ValueError("Staged source differs: " + name)
    metadata = json.loads((staged / "kernel-metadata.json").read_text(encoding="utf-8"))
    if (metadata.get("is_private") is not True or metadata.get("enable_gpu") is not True
            or metadata.get("enable_internet") is not False
            or metadata.get("dataset_sources") != [manifest["source_dataset"]]
            or metadata.get("code_file") != "bootstrap.py"):
        raise ValueError("Kaggle source metadata differs")
    bootstrap = staged / "bootstrap.py"
    ast.parse(bootstrap.read_text(encoding="utf-8"), filename=str(bootstrap))
    spec = importlib.util.spec_from_file_location("native_swe_staged_bootstrap", bootstrap)
    if spec is None or spec.loader is None:
        raise ValueError("Staged bootstrap cannot be inspected")
    module = importlib.util.module_from_spec(spec)
    sys.dont_write_bytecode = True
    spec.loader.exec_module(module)
    for name, descriptor in manifest["embedded_source"].items():
        raw = zlib.decompress(base64.b64decode(module.EMBEDDED_SOURCE[name]["zlib_base64"]))
        if (len(raw) != descriptor["bytes"]
                or hashlib.sha256(raw).hexdigest() != descriptor["sha256"]
                or raw != (SOURCE / name).read_bytes()):
            raise ValueError("Embedded original source differs: " + name)
        ast.parse(raw, filename=name)
    package = HERE / "private_kaggle_source_package.zip"
    with zipfile.ZipFile(package) as archive:
        if set(archive.namelist()) != {"bootstrap.py", "kernel-metadata.json", "package_manifest.json"}:
            raise ValueError("Private package archive members differ")
        for name in archive.namelist():
            expected = (staged / name).read_bytes() if name != "package_manifest.json" else (HERE / name).read_bytes()
            if archive.read(name) != expected:
                raise ValueError("Private package archive member differs: " + name)
    source_zip = asset_dir / "v43_source.zip"
    parent = asset_dir / "v43_step15000.pt"
    if (source_zip.stat().st_size != module.V43_ZIP_BYTES
            or digest(source_zip) != module.V43_ZIP_SHA256
            or parent.stat().st_size != module.PARENT_BYTES
            or digest(parent) != module.PARENT_SHA256):
        raise ValueError("Original mounted v43 input candidate differs")
    with zipfile.ZipFile(source_zip) as archive:
        for name, expected in (("base_metadata.json", module.META_SHA256),
                               ("data/eval_swe_step_control_heldout.jsonl", module.SWE_SHA256),
                               ("eval_step_control_heldout.py", module.EVALUATOR_SHA256)):
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError("Original v43 source member differs: " + name)
        metadata = json.loads(archive.read("base_metadata.json"))
        cases = [json.loads(line) for line in archive.read("data/eval_swe_step_control_heldout.jsonl").splitlines()
                 if line.strip()]
    if metadata["config"]["max_positions"] != 2048 or len(cases) != 200:
        raise ValueError("Original v43 context or SWE suite differs")
    result = {"status": "STATIC_PACKAGE_AND_LOCAL_INPUTS_VERIFIED_NO_GPU_RUN",
              "source_zip_sha256": digest(source_zip),
              "parent_sha256": digest(parent),
              "package_zip_sha256": digest(package),
              "bootstrap_sha256": digest(bootstrap),
              "swe_cases": len(cases), "generation_limit": 2048,
              "hard_stop_utc": module.HARD_STOP_UTC.isoformat(),
              "max_wall_seconds": module.MAX_WALL_SECONDS,
              "training_updates": 0}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-asset-dir", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.local_asset_dir)
    (HERE / "package_verification.json").write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
