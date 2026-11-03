"""Prepare a private Kaggle source package; never push or start a job."""

from __future__ import annotations

import base64
import hashlib
import json
import zlib
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent
STAGED = HERE / "staged"
EXPECTED_SOURCE = {
    "adapter.py": "3f9c3e2bf676f2e53eb66d57f0a8152297ca01fafe608738fe3f9cd3c0a4fe68",
    "admission.py": "0fbcf14fc08a6d3a0120c0ba063e598e7e1f26e4f62e8419333522f13fdfcd59",
    "v43_native_swe_capture_readonly.py": "6bcf453c32cacff89d3e6b3062162c3ae43f6027fe8f6b3336c8c2c8740fc12d",
}


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def make() -> dict:
    if STAGED.exists():
        raise FileExistsError("Refusing to overwrite an existing single-use Kaggle package")
    source = {}
    for name, expected in EXPECTED_SOURCE.items():
        raw = (SOURCE / name).read_bytes()
        if sha(raw) != expected:
            raise ValueError("Pinned source changed: " + name)
        source[name] = {"bytes": len(raw), "sha256": expected,
                        "zlib_base64": base64.b64encode(zlib.compress(raw, 9)).decode("ascii")}
    template = (HERE / "bootstrap_template.py").read_text(encoding="utf-8")
    if template.count("__EMBEDDED_SOURCE__") != 1:
        raise ValueError("Bootstrap source insertion point changed")
    bootstrap = template.replace("__EMBEDDED_SOURCE__", repr(source)).encode("utf-8")
    metadata = {
        "id": "geocpuresearch/geocpu-v43-native-swe200-proof-20261103-v1",
        "title": "GeoCPU v43 native SWE200 read-only proof 20261103",
        "code_file": "bootstrap.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "enable_internet": False, "keywords": [],
        "dataset_sources": ["geocpuresearch/geocpu-v43-v44fix-paired-eval-20261103"],
        "kernel_sources": [], "competition_sources": [], "model_sources": [],
        "machine_shape": "NvidiaTeslaT4", "session_timeout_seconds": 2400,
    }
    metadata_bytes = (json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    STAGED.mkdir(parents=True)
    (STAGED / "bootstrap.py").write_bytes(bootstrap)
    (STAGED / "kernel-metadata.json").write_bytes(metadata_bytes)
    manifest = {
        "schema": "geocpu_v43_native_swe200_private_package_v1",
        "state": "STAGED_ONLY_NO_PROVIDER_VERSION_OR_CLAIM",
        "source_dataset": metadata["dataset_sources"][0],
        "hard_stop_utc": "2026-11-03T18:00:00+00:00",
        "max_wall_seconds": 1800, "generation_limit": 2048,
        "swe_cases": 200, "training_updates": 0,
        "input_source_zip_sha256": "0102dc9b01b4b795c3123b18cefc7a8734e47a06dd85292af464d3232704d198",
        "input_checkpoint_sha256": "a4b95d7d0f486146196271b3b1fbc6a1b6b3342d8fadd590320fc3842c1e80e5",
        "input_metadata_sha256": "7efa8dea9a0100e2250f72805c89cca7a281efec30b4479795743814756cdcd5",
        "original_swe_sha256": "0bf62099ea2a41c8eb35c1ebdc2584f2c164ae3c82f741ab3eca44113f70f8cd",
        "original_evaluator_sha256": "c3589b97f531b6460a95907e56284df5f20be990eff1a4d2e56a921b4c705899",
        "files": {"bootstrap.py": {"bytes": len(bootstrap), "sha256": sha(bootstrap)},
                  "kernel-metadata.json": {"bytes": len(metadata_bytes), "sha256": sha(metadata_bytes)}},
        "embedded_source": {name: {"bytes": item["bytes"], "sha256": item["sha256"]}
                            for name, item in source.items()},
    }
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    (HERE / "package_manifest.json").write_bytes(manifest_bytes)
    archive_path = HERE / "private_kaggle_source_package.zip"
    if archive_path.exists():
        raise FileExistsError(archive_path)
    with ZipFile(archive_path, "x", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in (("bootstrap.py", bootstrap), ("kernel-metadata.json", metadata_bytes),
                           ("package_manifest.json", manifest_bytes)):
            info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            archive.writestr(info, data)
    manifest["package_zip_sha256"] = sha(archive_path.read_bytes())
    return manifest


if __name__ == "__main__":
    print(json.dumps(make(), ensure_ascii=False, sort_keys=True))
