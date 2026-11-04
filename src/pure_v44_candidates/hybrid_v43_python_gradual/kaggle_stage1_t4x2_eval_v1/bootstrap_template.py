"""Private Kaggle T4x2 entry for read-only, matched stage-1 fixed44 evaluation."""
from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zlib

EMBEDDED = __EMBEDDED__
EXPECTED = __EXPECTED__
ROOT = Path("/kaggle/working/geocpu_stage1_paired_fixed44_eval_20261104")
PAIRED_ROOT_REL = "geocpu_hybrid_stage1_fp16_paired_20261104"
V43_ZIP_SHA = "0102dc9b01b4b795c3123b18cefc7a8734e47a06dd85292af464d3232704d198"
PARENT_SHA = "a4b95d7d0f486146196271b3b1fbc6a1b6b3342d8fadd590320fc3842c1e80e5"


def digest(path: Path) -> str:
    h = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def mounted(identifier: str) -> Path:
    owner, slug = identifier.split("/", 1)
    candidates = (Path("/kaggle/input") / slug,
                  Path("/kaggle/input/datasets") / owner / slug)
    found = [path for path in candidates if path.is_dir()]
    if len(found) != 1:
        raise FileNotFoundError("Expected exactly one Kaggle dataset mount: " + identifier)
    return found[0]


def unpack(code: Path) -> None:
    for name, item in EMBEDDED.items():
        relative = Path(name)
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise ValueError("Bundled path is not relative")
        raw = zlib.decompress(base64.b64decode(item["zlib_base64"], validate=True))
        if len(raw) != item["bytes"] or sha256(raw).hexdigest() != item["sha256"]:
            raise ValueError("Bundled source, proof, or claim changed: " + name)
        target = code / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)


def verify_inputs(code: Path) -> tuple[Path, Path, Path, dict]:
    original = mounted("geocpuresearch/geocpu-v43-v44fix-paired-eval-20261103")
    paired = mounted(EXPECTED["paired_dataset"]) / PAIRED_ROOT_REL
    source_zip = original / "v43_source.zip.bin"
    parent = original / "v43_step15000.pt"
    if (digest(source_zip) != V43_ZIP_SHA or parent.stat().st_size != 990_545_719
            or digest(parent) != PARENT_SHA):
        raise ValueError("Original v43 ZIP or parent checkpoint differs")
    for relative, item in EXPECTED["paired_files"].items():
        path = paired / relative
        if ("bytes" in item and path.stat().st_size != item["bytes"]
                or digest(path) != item["sha256"]):
            raise ValueError("Sealed paired checkpoint or receipt differs: " + relative)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(code))
    from prelaunch_contract import verify_prelaunch_contract
    from source_pools import file_sha
    from train_hybrid_stage import (_verify_original_swe_baseline,
                                    _verify_pretrain_raw_diagnostic)
    swe_sha = _verify_original_swe_baseline(
        code / "proofs" / "swe200", expected_precision_profile="eval_bf16_native")
    fixed44_sha = _verify_pretrain_raw_diagnostic(
        code / "proofs" / "fixed44", expected_precision_profile="eval_fp16_native_v1")
    if (swe_sha != EXPECTED["swe_complete_sha256"]
            or fixed44_sha != EXPECTED["fixed44_complete_sha256"]):
        raise ValueError("Parent SWE200 or fixed44 baseline differs")
    claims = {}
    now = datetime.now(timezone.utc)
    for arm in ("control", "python"):
        path = code / "claims" / arm / "prelaunch_contract_once.json"
        if file_sha(path) != EXPECTED["claim_sha256"][arm]:
            raise ValueError("Evaluation claim changed: " + arm)
        claim = verify_prelaunch_contract(path, now=now, expected_arm=arm,
                                          required_run_seconds=1800,
                                          required_save_reserve_seconds=180)
        if (claim["gpu_count"] != 1 or claim["precision_profile"] != "eval_fp16_native_v1"
                or claim.get("parent_swe200_baseline_complete_sha256") != swe_sha
                or claim.get("parent_fixed44_raw_diagnostic_complete_sha256") != fixed44_sha):
            raise ValueError("Evaluation claim profile or parent binding differs: " + arm)
        claims[arm] = claim
    if (claims["control"]["claim_id"] == claims["python"]["claim_id"]
            or claims["control"]["run_id"] == claims["python"]["run_id"]
            or claims["control"]["hard_stop_at_utc"] != claims["python"]["hard_stop_at_utc"]):
        raise ValueError("Paired evaluation claims differ or are reused")
    import torch
    if torch.cuda.device_count() != 2:
        raise RuntimeError("Exactly two Kaggle T4 GPUs are required")
    for index, arm in enumerate(("control", "python")):
        gpu_name = torch.cuda.get_device_name(index)
        if "T4" not in gpu_name.upper() or claims[arm]["gpu_names"] != [gpu_name]:
            raise RuntimeError("Actual GPU differs from the claimed T4: " + arm)
    return source_zip, parent, paired, claims


def command(code: Path, source_zip: Path, paired: Path, arm: str) -> list[str]:
    checkpoint = paired / arm / f"hybrid_stage1_{arm}_update_0250.pt"
    return [sys.executable, "-u", str(code / "launch_fixed_diagnostic_bounded.py"),
            "--v43-zip", str(source_zip),
            "--overlay-source", str(code / "overlay"),
            "--checkpoint", str(checkpoint), "--checkpoint-kind", "hybrid_250",
            "--output-dir", str(ROOT / arm),
            "--prelaunch-contract", str(code / "claims" / arm / "prelaunch_contract_once.json"),
            "--hard-stop-seconds", "1800", "--save-reserve-seconds", "180"]


def run_pair(code: Path, source_zip: Path, paired: Path, claims: dict) -> dict:
    deadline = datetime.fromisoformat(
        claims["control"]["hard_stop_at_utc"].replace("Z", "+00:00")).astimezone(timezone.utc)
    processes = {}
    timed_out = False
    with (ROOT / "control.log").open("x", encoding="utf-8") as control_log, \
            (ROOT / "python.log").open("x", encoding="utf-8") as python_log:
        try:
            for index, (arm, log) in enumerate((("control", control_log), ("python", python_log))):
                env = dict(os.environ)
                env["CUDA_VISIBLE_DEVICES"] = str(index)
                env["PYTHONDONTWRITEBYTECODE"] = "1"
                processes[arm] = subprocess.Popen(
                    command(code, source_zip, paired, arm), cwd=code,
                    env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
            while any(process.poll() is None for process in processes.values()):
                if datetime.now(timezone.utc) >= deadline - timedelta(seconds=30):
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
    status = {"schema": "geocpu_stage1_paired_fixed44_eval_status_v1",
              "process_exit_codes": {arm: process.returncode for arm, process in processes.items()},
              "absolute_stop_reached": timed_out, "arm_results": {},
              "evaluation_only": True, "stage_promotion": False}
    for arm in ("control", "python"):
        output = ROOT / arm
        complete = output / "COMPLETE.json"
        partial = output / "PARTIAL.json"
        status["arm_results"][arm] = {
            "complete": complete.is_file(), "partial": partial.is_file(),
            "complete_sha256": digest(complete) if complete.is_file() else None,
            "partial_sha256": digest(partial) if partial.is_file() else None,
            "raw_answers_sha256": digest(output / "raw_answers_44_profiles.jsonl")
                if (output / "raw_answers_44_profiles.jsonl").is_file() else None,
        }
    status["status"] = ("READY_TO_COMPARE" if not timed_out and all(
        status["process_exit_codes"].get(arm) == 0 and status["arm_results"][arm]["complete"]
        for arm in ("control", "python")) else "PARTIAL_OR_FAILED")
    with (ROOT / "evaluation_status.json").open("x", encoding="utf-8") as sink:
        json.dump(status, sink, sort_keys=True, indent=2)
        sink.write("\n")
    return status


def main() -> None:
    if not Path("/kaggle").is_dir() or Path("/content/drive/MyDrive").exists():
        raise RuntimeError("Only a private disposable Kaggle runtime is admitted")
    if ROOT.exists():
        raise FileExistsError("Evaluation output directory is one-use")
    ROOT.mkdir(parents=True)
    code = ROOT / "source"
    code.mkdir()
    unpack(code)
    source_zip, parent, paired, claims = verify_inputs(code)
    status = run_pair(code, source_zip, paired, claims)
    if status["status"] != "READY_TO_COMPARE":
        raise RuntimeError("A diagnostic is incomplete; inspect both logs and raw-answer folders")
    from compare_fixed44 import compare
    checkpoints = {arm: EXPECTED["paired_files"][
        f"{arm}/hybrid_stage1_{arm}_update_0250.pt"]["sha256"]
        for arm in ("control", "python")}
    comparison = compare(parent_dir=code / "proofs" / "fixed44",
                         control_dir=ROOT / "control", python_dir=ROOT / "python",
                         checkpoint_sha=checkpoints,
                         cases_path=code / "fixed_interim_diagnostic_44_v2.jsonl",
                         output_dir=ROOT)
    if (digest(parent) != PARENT_SHA or digest(source_zip) != V43_ZIP_SHA
            or any(digest(paired / relative) != item["sha256"]
                   for relative, item in EXPECTED["paired_files"].items())):
        raise ValueError("Original source or sealed stage-1 inputs changed during evaluation")
    print(json.dumps({"status": "COMPLETE_RAW_COMPARISON", "matched_case_profiles": 84,
                      "comparison_sha256": digest(ROOT / "comparison.json"),
                      "matched_rows_sha256": comparison["matched_rows_sha256"],
                      "stage_promotion": False}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
