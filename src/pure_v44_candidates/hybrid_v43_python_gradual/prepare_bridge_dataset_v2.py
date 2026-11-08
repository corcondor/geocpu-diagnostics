"""Stage the verified Python bridge as a new private Kaggle input dataset.

The original v43 checkpoint/source live in a separate pinned input dataset.
This package contains only independently checked derived teaching rows and
their verification receipts.  It never modifies a model or starts training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from source_pools import (PINNED_CONVERTED_SHA, PINNED_REPLAY_SOURCE_SHA,
                          PINNED_V43_ZIP_SHA, file_sha)
from source_pools_v2 import SWE_SHA, SWE_MANIFEST_SHA, SWE_AUDIT_SHA


def stage(*, output: Path, converted: Path, swe: Path, swe_manifest: Path,
          swe_audit: Path, replay_complete: Path) -> dict:
    if output.exists():
        raise FileExistsError("Use a new private dataset staging directory")
    replay = json.loads(replay_complete.read_text(encoding="utf-8"))
    replay_rows = replay_complete.parent / "all_row_replay_receipts.jsonl"
    if (replay.get("schema") != "geocpu_hybrid_independent_python_replay_v1"
            or replay.get("source_zip_sha256") != PINNED_V43_ZIP_SHA
            or replay.get("converted_sha256") != PINNED_CONVERTED_SHA
            or replay.get("verifier_source_sha256") != PINNED_REPLAY_SOURCE_SHA
            or replay.get("accepted_rows") != 24_836
            or replay.get("all_python_blocks_executed") is not True
            or file_sha(replay_rows) != replay.get("row_receipts_sha256")):
        raise ValueError("The full independent Python replay is incomplete")
    source = {
        "converted_v1.jsonl": (converted, PINNED_CONVERTED_SHA),
        "converted_swe_v2.jsonl": (swe, SWE_SHA),
        "swe_manifest_v2.json": (swe_manifest, SWE_MANIFEST_SHA),
        "swe_audit_v2.json": (swe_audit, SWE_AUDIT_SHA),
        "replay_COMPLETE.json": (replay_complete, file_sha(replay_complete)),
        "all_row_replay_receipts.jsonl": (replay_rows, replay["row_receipts_sha256"]),
    }
    for path, expected in source.values():
        if not path.is_file() or file_sha(path) != expected:
            raise ValueError("Pinned bridge input changed: " + str(path))
    output.mkdir(parents=True)
    files = {}
    for name, (path, expected) in source.items():
        target = output / name
        with path.open("rb") as reader, target.open("xb") as writer:
            shutil.copyfileobj(reader, writer, 8 << 20)
        if file_sha(target) != expected:
            raise ValueError("Staged bridge input differs: " + name)
        files[name] = {"sha256": expected, "bytes": target.stat().st_size}
    metadata = {"id": "geocpuresearch/geocpu-v43-python-bridge-verified-20261104",
                "title": "GeoCPU v43 Python bridge verified 20261104",
                "licenses": [{"name": "other"}]}
    (output / "dataset-metadata.json").write_text(
        json.dumps(metadata, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    manifest = {"schema": "geocpu_hybrid_python_bridge_dataset_v2",
                "original_v43_source_sha256": PINNED_V43_ZIP_SHA,
                "replay_executed_python_blocks": replay["python_blocks_executed"],
                "files": files}
    (output / "assets_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return {"dataset_id": metadata["id"], "files": len(files),
            "bytes": sum(item["bytes"] for item in files.values()),
            "manifest_sha256": file_sha(output / "assets_manifest.json")}


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("output", "converted", "swe", "swe_manifest", "swe_audit",
                 "replay_complete"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(stage(**vars(args)), sort_keys=True))


if __name__ == "__main__":
    main()
