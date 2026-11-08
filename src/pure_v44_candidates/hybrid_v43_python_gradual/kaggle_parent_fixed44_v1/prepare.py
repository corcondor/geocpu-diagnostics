"""Build an immutable private Kaggle parent diagnostic after Root's fresh claim."""
from __future__ import annotations

import argparse
import base64
from hashlib import sha256
import json
from pathlib import Path
import zlib

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
OVERLAY = Path(__file__).resolve().parents[3] / "v44fix"
SOURCES = (
    "hybrid_data.py", "source_pools.py", "prelaunch_contract.py",
    "train_hybrid_stage.py", "eval_fixed_diagnostic.py",
    "launch_fixed_diagnostic_bounded.py", "fixed_interim_diagnostic_44_v2.jsonl",
)
OVERLAY_FILES = (
    "integrated_model_v44fix.py", "tir_engine_v44r1.py",
    "executor_worker_v44r1.py", "resource_limited_worker_v44r1.py",
)
CLAIM_FILES = (
    "prelaunch_contract_once.json", "quota_evidence.json",
    "root_hold_ledger.json", "active_jobs_snapshot.json",
)


def package(claim_dir: Path, staged: Path) -> dict:
    if staged.exists():
        raise FileExistsError("Use a new one-use private Kaggle staging folder")
    files = {name: STUDY / name for name in SOURCES}
    files.update({"overlay/" + name: OVERLAY / name for name in OVERLAY_FILES})
    files.update({name: claim_dir / name for name in CLAIM_FILES})
    contract = json.loads(files["prelaunch_contract_once.json"].read_text(encoding="utf-8"))
    if (contract.get("arm") != "parent"
            or contract.get("precision_profile") != "eval_fp16_native_v1"
            or contract.get("root_authorized_once") is not True):
        raise ValueError("Expected a fresh FP16 parent-only Root claim")
    embedded = {}
    for name, path in files.items():
        raw = path.read_bytes()
        embedded[name] = {"bytes": len(raw), "sha256": sha256(raw).hexdigest(),
                          "zlib_base64": base64.b64encode(zlib.compress(raw, 9)).decode("ascii")}
    template = (HERE / "bootstrap_template.py").read_text(encoding="utf-8")
    if template.count("__EMBEDDED__") != 1:
        raise ValueError("Bootstrap insertion point changed")
    bootstrap = template.replace("__EMBEDDED__", repr(embedded)).encode("utf-8")
    kernel_id = "geocpuresearch/geocpu-v43-parent-fixed44-fp16-20261104"
    metadata = {
        "id": kernel_id, "title": kernel_id.split("/", 1)[1],
        "code_file": "bootstrap.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "enable_internet": False, "keywords": [],
        "dataset_sources": ["geocpuresearch/geocpu-v43-v44fix-paired-eval-20261103"],
        "kernel_sources": [], "competition_sources": [], "model_sources": [],
        "machine_shape": "NvidiaTeslaT4", "session_timeout_seconds": 2400,
    }
    metadata_raw = (json.dumps(metadata, sort_keys=True, indent=2) + "\n").encode()
    staged.mkdir(parents=True)
    (staged / "bootstrap.py").write_bytes(bootstrap)
    (staged / "kernel-metadata.json").write_bytes(metadata_raw)
    return {"kernel_id": kernel_id, "embedded": {name: item["sha256"] for name, item in embedded.items()},
            "bootstrap_sha256": sha256(bootstrap).hexdigest(),
            "metadata_sha256": sha256(metadata_raw).hexdigest(),
            "contract_sha256": embedded["prelaunch_contract_once.json"]["sha256"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--claim-dir", type=Path, required=True)
    parser.add_argument("--staged", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.claim_dir, args.staged), sort_keys=True))


if __name__ == "__main__":
    main()
