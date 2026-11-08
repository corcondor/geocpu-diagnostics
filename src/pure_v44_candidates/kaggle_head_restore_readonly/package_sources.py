"""Build a small, deterministic source-only Kaggle bundle; never copy checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import zipfile

HERE = Path(__file__).resolve().parent
CANDIDATES = HERE.parent
REPOSITORY = CANDIDATES.parent.parent
OUTPUT = REPOSITORY / "reports" / "v44fix" / "kaggle_pure_head_restore_sources_v1.zip"
ENTRYPOINT_NAME = "kaggle_pure_head_restore_entrypoint_v1.py"
MEMBERS = {
    "runner.py": HERE / "runner.py",
    "verify_outputs.py": HERE / "verify_outputs.py",
    "launch.sh": HERE / "launch.sh",
    "matched_pure_v44_readonly.py": CANDIDATES / "matched_pure_v44_readonly.py",
    "partial_head_restore_readonly.py": CANDIDATES / "partial_head_restore_readonly.py",
    "weight_direction/weight_direction.py": CANDIDATES / "weight_direction" / "weight_direction.py",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build(output: Path = OUTPUT) -> tuple[Path, str]:
    sys.path.insert(0, str(CANDIDATES))
    import matched_pure_v44_readonly as matched

    file_bytes = {name: path.read_bytes() for name, path in MEMBERS.items()}
    manifest = {
        "schema": "geocpu_pure_v44_kaggle_head_restore_sources_v1",
        "files_sha256": {name: sha256(data) for name, data in sorted(file_bytes.items())},
        "asset_paths": {name: pin[0].relative_to(matched.DRIVE).as_posix()
                        for name, pin in matched.PINS.items()},
        "asset_pins": {name: [pin[1], pin[2]] for name, pin in matched.PINS.items()},
        "checkpoint_files_in_source_bundle": False,
        "generator": "same pinned pure-v44 Hotfix evaluator; child-only temporary parent base.head",
        "suite": "mixed64", "case_count": 64,
        "arms": ["parent", "child", "child_parent_head"],
        "generation_limit_per_arm": 2048,
        "cache_reuse": False,
    }
    file_bytes["source_manifest.json"] = (json.dumps(manifest, ensure_ascii=False,
                                                     sort_keys=True, indent=2) + "\n").encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    with tempfile.NamedTemporaryFile(prefix="geocpu_head_source_", suffix=".zip",
                                     dir=output.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=9) as archive:
            for name, data in sorted(file_bytes.items()):
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = (0o100644 << 16)
                archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED,
                                 compresslevel=9)
        with zipfile.ZipFile(temporary) as archive:
            if set(archive.namelist()) != set(file_bytes):
                raise RuntimeError("Source archive members differ")
            if any(archive.read(name) != data for name, data in file_bytes.items()):
                raise RuntimeError("Source archive readback differs")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    digest = sha256(output.read_bytes())
    entrypoint = output.with_name(ENTRYPOINT_NAME)
    if entrypoint.exists():
        raise FileExistsError(entrypoint)
    template = (HERE / "entrypoint_template.py").read_text(encoding="utf-8")
    entrypoint.write_text(template.replace("__BUNDLE_SHA256__", digest), encoding="utf-8")
    return output, digest


if __name__ == "__main__":
    path, digest = build()
    print(path, digest)
