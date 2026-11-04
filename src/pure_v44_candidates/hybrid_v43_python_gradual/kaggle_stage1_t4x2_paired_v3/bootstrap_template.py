"""Private Kaggle T4x2 entry: matched FP16 parent proof, then 0%/1% stage 1."""
from __future__ import annotations

import base64
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zlib

EMBEDDED = __EMBEDDED__
ROOT = Path("/kaggle/working/geocpu_hybrid_stage1_fp16_paired_m4096_20261104_v3")
V43_ZIP_SHA = "0102dc9b01b4b795c3123b18cefc7a8734e47a06dd85292af464d3232704d198"
PARENT_SHA = "a4b95d7d0f486146196271b3b1fbc6a1b6b3342d8fadd590320fc3842c1e80e5"
BRIDGE_SHA = {
    "converted_v1.jsonl": "53387b2987df02d91b51b4afc8ab2369cc0e53b9e4bdd0cfd480c3586de7f3db",
    "converted_swe_v2.jsonl": "89b129cebec20eda5622c4c169601f8c6a67f844719f986b170f172421f5abc5",
    "swe_manifest_v2.json": "b4b054164bdb8c8b50883ccdc22ebb936fe54a3c60ba190bccbd707e7917174a",
    "swe_audit_v2.json": "78e3558c92bfd90cf5553734d09336c6e5a696331b7f450b68d2aef698705b64",
}


def digest(path: Path) -> str:
    h = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def mounted(slug: str) -> Path:
    candidates = (
        Path("/kaggle/input") / slug,
        Path("/kaggle/input/datasets/geocpuresearch") / slug,
    )
    found = [path for path in candidates if path.is_dir()]
    if len(found) != 1:
        raise FileNotFoundError("Expected exactly one Kaggle input mount: " + slug)
    return found[0]


def unpack(code: Path) -> None:
    for name, item in EMBEDDED.items():
        relative = Path(name)
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise ValueError("Bundled file path is unsafe")
        raw = zlib.decompress(base64.b64decode(item["zlib_base64"], validate=True))
        if len(raw) != item["bytes"] or sha256(raw).hexdigest() != item["sha256"]:
            raise ValueError("Bundled file changed: " + name)
        target = code / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)


def verify_inputs(code: Path) -> tuple[Path, Path, Path]:
    original = mounted("geocpu-v43-v44fix-paired-eval-20261103")
    bridge = mounted("geocpu-v43-python-bridge-verified-20261104")
    source_zip = original / "v43_source.zip.bin"
    parent = original / "v43_step15000.pt"
    if (digest(source_zip) != V43_ZIP_SHA or digest(parent) != PARENT_SHA
            or parent.stat().st_size != 990_545_719):
        raise ValueError("Pinned original step-15000 checkpoint or source changed")
    manifest = json.loads((bridge / "assets_manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("schema") != "geocpu_hybrid_python_bridge_dataset_v2"
            or manifest.get("original_v43_source_sha256") != V43_ZIP_SHA):
        raise ValueError("Verified Python bridge manifest differs")
    for name, expected in BRIDGE_SHA.items():
        if manifest.get("files", {}).get(name, {}).get("sha256") != expected:
            raise ValueError("Bridge manifest differs: " + name)
        if digest(bridge / name) != expected:
            raise ValueError("Bridge asset differs: " + name)
    for name in ("replay_COMPLETE.json", "all_row_replay_receipts.jsonl"):
        expected = manifest.get("files", {}).get(name, {}).get("sha256")
        if not isinstance(expected, str) or digest(bridge / name) != expected:
            raise ValueError("Independent full Python replay differs: " + name)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(code))
    from prelaunch_contract import verify_prelaunch_contract
    from train_hybrid_stage_v2 import (
        _verify_original_swe_baseline, _verify_pretrain_raw_diagnostic,
    )
    now = datetime.now(timezone.utc)
    claims = {}
    for arm in ("control", "python"):
        claims[arm] = verify_prelaunch_contract(
            code / "claims" / arm / "prelaunch_contract_once.json",
            now=now, expected_arm=arm, required_run_seconds=1800,
            required_save_reserve_seconds=180,
        )
        if (claims[arm]["gpu_count"] != 1
                or claims[arm]["precision_profile"] != "train_fp16_gradscaler_v1"):
            raise ValueError("Each arm requires one FP16 T4 claim")
    if (claims["control"]["claim_id"] == claims["python"]["claim_id"]
            or claims["control"]["run_id"] == claims["python"]["run_id"]
            or claims["control"]["hard_stop_at_utc"] != claims["python"]["hard_stop_at_utc"]):
        raise ValueError("The paired claims are not distinct and synchronized")
    if (claims["control"]["official_minimum_free_device_seconds"] <
            sum(claim["root_reserved_device_seconds"] for claim in claims.values())):
        raise ValueError("Official free quota cannot cover the two T4 reservations")
    import torch
    if torch.cuda.device_count() != 2:
        raise RuntimeError("Exactly two Kaggle T4 GPUs are required")
    for index, arm in enumerate(("control", "python")):
        name = torch.cuda.get_device_name(index)
        if "T4" not in name.upper() or claims[arm]["gpu_names"] != [name]:
            raise RuntimeError("Actual GPU differs from the claimed T4 for " + arm)
    _verify_original_swe_baseline(
        code / "proofs" / "swe200", expected_precision_profile="eval_bf16_native")
    _verify_pretrain_raw_diagnostic(
        code / "proofs" / "fixed44", expected_precision_profile="eval_fp16_native_v1")
    return source_zip, parent, bridge


def command(code: Path, source_zip: Path, parent: Path, bridge: Path, arm: str) -> list[str]:
    return [
        sys.executable, "-u", str(code / "train_hybrid_stage_v2.py"),
        "--stage", "1", "--arm", arm,
        "--v43-zip", str(source_zip), "--parent-checkpoint", str(parent),
        "--converted", str(bridge / "converted_v1.jsonl"),
        "--replay-receipt", str(bridge / "replay_COMPLETE.json"),
        "--swe-jsonl", str(bridge / "converted_swe_v2.jsonl"),
        "--swe-manifest", str(bridge / "swe_manifest_v2.json"),
        "--swe-audit", str(bridge / "swe_audit_v2.json"),
        "--overlay-source", str(code / "overlay"),
        "--baseline-proof-dir", str(code / "proofs" / "swe200"),
        "--baseline-diagnostic-dir", str(code / "proofs" / "fixed44"),
        "--prelaunch-contract", str(code / "claims" / arm / "prelaunch_contract_once.json"),
        "--output-dir", str(ROOT / arm),
        "--seed", "20261103", "--hard-stop-seconds", "1800",
        "--save-reserve-seconds", "180", "--max-micro-tokens", "4096",
    ]


def run_pair(code: Path, source_zip: Path, parent: Path, bridge: Path) -> dict:
    processes = {}
    timed_out = False
    claim = json.loads((code / "claims" / "control" / "prelaunch_contract_once.json").read_text())
    absolute_stop = datetime.fromisoformat(claim["hard_stop_at_utc"].replace("Z", "+00:00"))
    with (ROOT / "control.log").open("x", encoding="utf-8") as control_log, \
            (ROOT / "python.log").open("x", encoding="utf-8") as python_log:
        try:
            for index, (arm, log) in enumerate((("control", control_log), ("python", python_log))):
                env = dict(os.environ)
                env["CUDA_VISIBLE_DEVICES"] = str(index)
                env["PYTHONDONTWRITEBYTECODE"] = "1"
                processes[arm] = subprocess.Popen(
                    command(code, source_zip, parent, bridge, arm), cwd=code,
                    env=env, stdout=log, stderr=subprocess.STDOUT, text=True,
                )
            while any(process.poll() is None for process in processes.values()):
                if datetime.now(timezone.utc) >= absolute_stop:
                    timed_out = True
                    break
                time.sleep(5)
        finally:
            for process in processes.values():
                if process.poll() is None:
                    process.terminate()
            for process in processes.values():
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    results = {arm: process.returncode for arm, process in processes.items()}
    status = {"schema": "geocpu_stage1_paired_t4x2_result_v1", "process_exit_codes": results,
              "absolute_stop_reached": timed_out,
              "arm_results": {}, "stage_promotion": False}
    for arm in ("control", "python"):
        log = ROOT / (arm + ".log")
        lines = log.read_text(encoding="utf-8").splitlines()
        try:
            report = json.loads(lines[-1])
        except (IndexError, json.JSONDecodeError):
            report = {"error": "No final training report; inspect log"}
        status["arm_results"][arm] = report
    if not timed_out and all(results.get(arm) == 0 for arm in ("control", "python")):
        valid = True
        for arm in ("control", "python"):
            report = status["arm_results"][arm]
            checkpoint = ROOT / arm / f"hybrid_stage1_{arm}_update_0250.pt"
            receipt_path = checkpoint.with_suffix(checkpoint.suffix + ".receipt.json")
            try:
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                if (report.get("accepted_updates") != 250
                        or report.get("first_evaluation_barrier_reached") is not True
                        or report.get("stage_complete") is not False
                        or receipt.get("status") != "BLOCK1_NEEDS_PAIRED_EVALUATION"
                        or receipt.get("accepted_updates") != 250
                        or receipt.get("sha256") != digest(checkpoint)):
                    raise ValueError("Checkpoint, receipt, or update count differs")
                report["checkpoint_sha256"] = receipt["sha256"]
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                valid = False
                report["barrier_error"] = str(exc)
        status["status"] = ("BLOCK1_NEEDS_PAIRED_EVALUATION" if valid
                            else "PARTIAL_OR_FAILED")
    else:
        status["status"] = "PARTIAL_OR_FAILED"
    (ROOT / "pair_status.json").write_text(json.dumps(status, sort_keys=True, indent=2) + "\n")
    if status["status"] != "BLOCK1_NEEDS_PAIRED_EVALUATION":
        raise RuntimeError("Paired stage 1 is incomplete; inspect the two logs and partial checkpoints")
    return status


def main() -> None:
    if not Path("/kaggle").is_dir() or Path("/content/drive/MyDrive").exists():
        raise RuntimeError("Only a private disposable Kaggle runtime is admitted")
    if ROOT.exists():
        raise FileExistsError("The paired output folder is one-use")
    ROOT.mkdir(parents=True)
    code = ROOT / "source"
    code.mkdir()
    unpack(code)
    source_zip, parent, bridge = verify_inputs(code)
    print(json.dumps(run_pair(code, source_zip, parent, bridge), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
