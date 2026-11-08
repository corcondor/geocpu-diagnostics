#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Colab orchestrator for v44fix (A100). Every stage writes its result to Drive and is skipped on re-run.

  stage A  external public curriculum receipt and input-hash admission (no public-code execution)
  stage B  v44fix mixture   build_v44fix_data.py      (v43 format + real v43 anchor incl. ALU/SEARCH)
  stage C  baseline         gate_eval_v44fix.py on v43 step15000
  stage D  gate training    runtime_v44fix.py 15000 -> 16000 (LR schedule of the full 25000 run)
  stage E  gate verdict     gate_eval_v44fix.py vs baseline  -> PASS / FAIL
  stage F  full training    runtime_v44fix.py 15000 -> 25000 + v43 7-tier suite; resumes from its own newest checkpoint
  stage G  final verdict    gate_eval_v44fix.py on step25000 vs baseline

Originals are never written: v43 checkpoint and v43 zip are read-only inputs; outputs go to OUT.
All child output is also appended to OUT/logs/ so results survive a dropped browser session.
"""
import hashlib, json, os, shutil, subprocess, sys, time, zipfile
from pathlib import Path
from resume_v44fix import (PARENT_BYTES, PARENT_SHA256, receipt_path,
                           sha256_file, verify_checkpoint)

DRIVE = Path("/content/drive/MyDrive")
V43_ZIP = DRIVE / "GeoCPU_Frontier_CoTraining_v43" / "Grand_cotraining_v43.zip"
V43_CKPT = DRIVE / "GeoCPU_Frontier_CoTraining_v43" / "checkpoints" / "checkpoint_step_15000.pt"
OUT = DRIVE / "GeoCPU_v44fix_TIR_consistency_20261103" / "run_1"
WORK = Path(__file__).parent.resolve()
PUBLIC = Path("/content/v44fix_tir_consistency_public_curriculum")
DATA = Path("/content/v44fix_tir_consistency_data")
GATE_STEP = 16000
FULL_STEPS = 25000
V43_SOURCE_SHA256 = "0102dc9b01b4b795c3123b18cefc7a8734e47a06dd85292af464d3232704d198"
MIN_DRIVE_FREE_BYTES = 18_000_000_000
PUBLIC_MAX_TOTAL_TOKENS = 8192
# Filled only after an independently isolated, credential-free verification run
# and review of its receipt. None keeps this distribution training-blocked.
OFFLINE_VERIFICATION_RECEIPT_SHA256 = None
PUBLIC_EVAL_PATTERNS = ("benchmark_*.jsonl", "*heldout*.jsonl", "eval_*.jsonl",
                        "*manifest_test*.jsonl", "*manifest_dev*.jsonl")
REQUIRED_PUBLIC_EVAL_FILES = {
    "benchmark_8100_manifest.jsonl", "eval_swe_step_control_heldout.jsonl",
    "eval_v33_state_tracking_heldout.jsonl", "eval_v38_decomposed_state_bench200.jsonl",
    "eval_v41_logical_instruction_operators_heldout.jsonl",
    "eval_v42_search_retrieval_aligned.jsonl", "eval_v42_search_retrieval_heldout.jsonl",
    "jamcqa_manifest_test.jsonl",
}


def run(cmd, log_name, cwd=WORK, ok_codes=(0,)):
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    print("\n$ " + " ".join(str(c) for c in cmd), flush=True)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8",
           "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")}
    with open(OUT / "logs" / f"{log_name}.log", "a", encoding="utf-8") as log:
        log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} $ {' '.join(str(c) for c in cmd)}\n")
        p = subprocess.Popen([str(c) for c in cmd], cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding="utf-8", bufsize=1, env=env)
        for line in p.stdout:
            print(line, end="", flush=True)
            log.write(line); log.flush()
        code = p.wait()
        log.write(f"===== exit {code}\n")
    if code not in ok_codes:
        raise subprocess.CalledProcessError(code, cmd)
    return code


def ckpt_done(p):
    p = Path(p)
    if not p.exists() and not receipt_path(p).exists():
        return False
    step = int(p.stem.rsplit("_", 1)[1])
    verify_checkpoint(p, expected_step=step)
    return True


def fingerprint():
    from gate_eval_v44fix import evaluation_fingerprint
    return evaluation_fingerprint(DATA / "tir_heldout.jsonl")


def ensure_inputs():
    if not (DRIVE / "GeoCPU_Frontier_CoTraining_v43").is_dir():
        raise RuntimeError("Google Drive is not mounted at the expected MyDrive path")
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or "A100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("This pinned training run requires one assigned A100 GPU")
    reported_free = shutil.disk_usage(DRIVE).free
    if reported_free < MIN_DRIVE_FREE_BYTES:
        raise RuntimeError(f"Drive reports {reported_free} free bytes; exact checkpoint run needs at least {MIN_DRIVE_FREE_BYTES}")
    print(f"[preflight] Drive reports {reported_free} free bytes (minimum {MIN_DRIVE_FREE_BYTES}).", flush=True)
    if (not V43_ZIP.is_file() or sha256_file(V43_ZIP) != V43_SOURCE_SHA256
            or not V43_CKPT.is_file() or V43_CKPT.stat().st_size != PARENT_BYTES
            or sha256_file(V43_CKPT) != PARENT_SHA256):
        raise ValueError("Pinned v43 source ZIP or step-15000 checkpoint is missing/changed")
    if not (WORK / "data" / "train_v43_knowledge_unified.jsonl").exists():
        with zipfile.ZipFile(V43_ZIP) as zf:
            for n in zf.namelist():
                if n.startswith("data/") and not (WORK / n).exists():
                    zf.extract(n, WORK)
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        import pyarrow  # noqa: F401  (GPT's public loader reads parquet)
    except ImportError:
        run([sys.executable, "-m", "pip", "install", "-q", "pyarrow"], "pip")


def require_offline_verifier_pin():
    if (not isinstance(OFFLINE_VERIFICATION_RECEIPT_SHA256, str)
            or len(OFFLINE_VERIFICATION_RECEIPT_SHA256) != 64):
        raise SystemExit("[A] training_ready=false: no reviewed offline verifier receipt is pinned")


def stage_public():
    require_offline_verifier_pin()
    m = PUBLIC / "dataset_manifest.json"
    receipt_path = PUBLIC / "offline_verification_receipt.json"
    if (not m.is_file() or not receipt_path.is_file()
            or sha256_file(receipt_path) != OFFLINE_VERIFICATION_RECEIPT_SHA256):
        raise SystemExit("[A] pinned offline verifier receipt or public manifest is missing/changed")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (receipt.get("schema") != "geocpu_public_tir_offline_verification_v1"
            or receipt.get("public_manifest_sha256") != sha256_file(m)
            or receipt.get("isolation") != {"kind": "credential_free_disposable_vm",
                                            "network": "disabled", "drive_mounted": False}):
        raise SystemExit("[A] offline verification receipt is not bound to this isolated preparation")
    code = {name: sha256_file(WORK / "gpt" / name) for name in
            ("prepare_public_v44.py", "tir_engine_v44r1.py", "executor_worker_v44r1.py")}
    if receipt.get("source_code_sha256") != code:
        raise SystemExit("[A] offline verifier source code differs from this package")

    man = json.loads(m.read_text(encoding="utf-8"))
    verification = man.get("tool_verification", {})
    stats = man.get("statistics", {})
    counts = man.get("counts", {})
    train_pool = counts.get("unique_training_pool_by_bucket", {}) if isinstance(counts, dict) else {}
    if (man.get("training_eligible") is not True or man.get("public_input_pins_match") is not True
            or man.get("max_total_tokens") != PUBLIC_MAX_TOTAL_TOKENS
            or not isinstance(verification, dict) or verification.get("mode") != "all"
            or verification.get("all_observations_locally_executed") is not True
            or not isinstance(stats, dict) or not isinstance(train_pool, dict)
            or not isinstance(stats.get("tir:accepted_unique_rows"), int)
            or not isinstance(stats.get("tir:sample_execution_passed"), int)
            or stats["tir:accepted_unique_rows"] > stats["tir:sample_execution_passed"]
            or not isinstance(train_pool.get("tir"), int) or train_pool["tir"] <= 0):
        raise SystemExit("[A] public TIR manifest does not establish complete execution")

    lock_path = WORK / "gpt" / "public_source_lock_v44.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    expected_sources = {(name, split): (entry["bytes"], entry["sha256"])
                        for name, spec in lock["repos"].items()
                        for split, entry in spec["files"].items()}
    pins = man.get("public_source_inputs")
    if (man.get("source_lock_sha256") != sha256_file(lock_path)
            or not isinstance(pins, list) or len(pins) != len(expected_sources)
            or {(pin.get("source"), pin.get("split")): (pin.get("bytes"), pin.get("sha256"))
                for pin in pins if isinstance(pin, dict)} != expected_sources):
        raise SystemExit("[A] public source lock or pinned source inputs differ")

    vocab_path = WORK / "base_metadata.json"
    anchor_path = WORK / "data" / "train_v43_knowledge_unified.jsonl"
    anchor = man.get("anchor_input", {})
    if (man.get("vocabulary_sha256") != sha256_file(vocab_path)
            or not isinstance(anchor, dict) or anchor.get("sha256") != sha256_file(anchor_path)
            or anchor.get("bytes") != anchor_path.stat().st_size):
        raise SystemExit("[A] vocabulary or v43 anchor differs from offline verification")

    eval_dir = WORK / "data"
    local_eval = {path.name: (path.stat().st_size, sha256_file(path))
                  for pattern in PUBLIC_EVAL_PATTERNS for path in eval_dir.glob(pattern) if path.is_file()}
    eval_pins = man.get("evaluation_inputs")
    if not isinstance(eval_pins, list):
        raise SystemExit("[A] evaluation file pins are missing")
    recorded_eval = {Path(pin.get("path", "")).name: (pin.get("bytes"), pin.get("sha256"))
                     for pin in eval_pins if isinstance(pin, dict) and not pin.get("source")}
    official = [pin for pin in eval_pins if isinstance(pin, dict)
                and pin.get("source") == "Numina official test: never training"]
    if (not REQUIRED_PUBLIC_EVAL_FILES.issubset(local_eval)
            or len(recorded_eval) != len(eval_pins) - len(official)
            or recorded_eval != local_eval or len(official) != 1
            or (official[0].get("bytes"), official[0].get("sha256")) != expected_sources[("numina", "test")]
            or not isinstance(official[0].get("rows"), int) or official[0]["rows"] <= 0):
        raise SystemExit("[A] evaluation inputs differ from offline verification")

    expected = man.get("outputs", {})
    if (not isinstance(expected, dict) or set(expected) != {"train.jsonl", "dev.jsonl", "test.jsonl"}
            or any(not isinstance(entry, dict) or not (PUBLIC / name).is_file()
                   or sha256_file(PUBLIC / name) != entry.get("sha256")
                   or (PUBLIC / name).stat().st_size != entry.get("bytes")
                   for name, entry in expected.items())):
        raise SystemExit("[A] public curriculum output hash or size differs")
    shutil.copy(m, OUT / "public_dataset_manifest.json")
    shutil.copy(receipt_path, OUT / "offline_verification_receipt.json")
    print("[A] externally verified public curriculum admitted", flush=True)


def stage_build():
    if (DATA / "manifest.json").exists():
        man = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
        inputs = {"converter": sha256_file(WORK / "build_v44fix_data.py"),
                  "public_train": sha256_file(PUBLIC / "train.jsonl"),
                  "v43_train": sha256_file(WORK / "data" / "train_v43_knowledge_unified.jsonl"),
                  "vocab": sha256_file(WORK / "base_metadata.json")}
        if (man.get("input_sha256") != inputs
                or man.get("train_sha256") != sha256_file(DATA / "train.jsonl")
                or man.get("heldout_sha256") != sha256_file(DATA / "tir_heldout.jsonl")):
            raise SystemExit("[B] v44fix mixture source/data hash differs; refusing reuse")
        print("[B] verified v44fix mixture already built", flush=True); return
    if DATA.exists():
        raise SystemExit("[B] partial v44fix data directory exists; refusing overwrite")
    run([sys.executable, "build_v44fix_data.py", "--public-dir", PUBLIC, "--v43-train", WORK / "data" / "train_v43_knowledge_unified.jsonl",
         "--eval-dir", WORK / "data", "--vocab", WORK / "base_metadata.json", "--out-dir", DATA], "B_build")
    shutil.copy(DATA / "manifest.json", OUT / "v44fix_data_manifest.json")
    shutil.copy(DATA / "tir_heldout.jsonl", OUT / "tir_heldout.jsonl")


def gate_eval(ckpt, out_json, log_name, baseline=None, n_tir=100):
    out_json = Path(out_json)
    ckpt = Path(ckpt)
    if ckpt != V43_CKPT:
        verify_checkpoint(ckpt)
    ckpt_sha = sha256_file(ckpt)
    if out_json.exists():
        r = json.loads(out_json.read_text(encoding="utf-8"))
        if (r.get("fingerprint") == fingerprint() and r.get("checkpoint_sha256") == ckpt_sha
                and r.get("n_tir_requested") == n_tir
                and r.get("baseline_sha256") == (sha256_file(baseline) if baseline else None)):
            print(f"[eval] reuse {out_json.name}", flush=True)
            return r
        raise RuntimeError("Existing gate evaluation has different checkpoint, source, set, or baseline")
    cmd = [sys.executable, "gate_eval_v44fix.py", "--ckpt", ckpt, "--out", out_json,
           "--tir", DATA / "tir_heldout.jsonl", "--n-tir", n_tir]
    if baseline:
        cmd += ["--baseline", baseline]
    run(cmd, log_name, ok_codes=(0, 2))
    return json.loads(out_json.read_text(encoding="utf-8"))


def train(out_dir, log_name, stop_at=None, skip_eval=False, resume=False):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    warm = V43_CKPT
    if resume:
        orphan_receipts = [p for p in out_dir.glob("checkpoint_step_*.pt.receipt.json")
                           if not Path(str(p).removesuffix(".receipt.json")).is_file()]
        if orphan_receipts:
            raise RuntimeError("Incomplete checkpoint receipts exist: " + repr(orphan_receipts))
        candidates = list(out_dir.glob("checkpoint_step_*.pt"))
        done = sorted((p for p in candidates if ckpt_done(p)),
                      key=lambda p: int(p.stem.rsplit("_", 1)[1]))
        if done:
            warm = done[-1]
            print(f"[train] exact resume from verified {warm.name}; target {FULL_STEPS}", flush=True)
    cmd = [sys.executable, "runtime_v44fix.py", "--train_path", DATA / "train.jsonl", "--warmstart_path", warm,
           "--output_dir", out_dir, "--max_steps", FULL_STEPS]
    if stop_at:
        cmd += ["--stop_at_step", stop_at]
    if skip_eval:
        cmd += ["--skip_final_eval"]
    if warm != V43_CKPT:
        cmd += ["--require_exact_resume"]
    run(cmd, log_name)


def summarize(tag, r):
    s = {k: r.get(k, {}).get("acc") for k in ("swe", "state", "search", "alu", "tir")}
    s["loop_rate"] = r.get("loops", {}).get("loop_rate")
    s["python_call_rate"] = r.get("tir", {}).get("python_call_rate")
    gate = f"  GATE={r['gate']['verdict']} {r['gate']['checks']}" if "gate" in r else ""
    print(f"[{tag}] " + json.dumps(s, ensure_ascii=False) + gate, flush=True)


def require_gate_counts(result: dict, *, n_tir: int) -> None:
    expected = {"swe": 200, "state": 200, "search": 200, "alu": 8, "tir": n_tir}
    for name, count in expected.items():
        actual = result.get(name, {}).get("n")
        if type(actual) is not int or actual != count:
            raise RuntimeError(f"Gate {name} evaluated {actual!r} cases; expected {count}")
    loops = result.get("loops", {}).get("samples")
    if not isinstance(loops, list) or len(loops) != 10:
        raise RuntimeError("Gate loop evaluator did not return 10 samples")


def seven_tier_profile() -> str:
    names = ("runtime_v44fix.py", "eval_step_control_heldout.py",
             "eval_v38_decomposed_state_bench.py", "eval_jamcqa_official.py",
             "eval_benchmark_full_8100.py", "eval_logical_instruction_operators.py",
             "eval_v42_search_retrieval.py", "eval_v43_ab_benchmark.py",
             "integrated_model_v42.py", "integrated_model_v30.py",
             "GeoCPU_CED_v1.py", "GeoCPU_CED_v2.py", "local_search_engine_v42.py",
             "qwenputer_controller_v30.py", "cpu_circuit_v30.py", "base_metadata.json")
    data = ("eval_swe_step_control_heldout.jsonl", "eval_v38_decomposed_state_bench200.jsonl",
            "jamcqa_manifest_test.jsonl", "benchmark_8100_manifest.jsonl",
            "eval_v41_logical_instruction_operators_heldout.jsonl",
            "eval_v42_search_retrieval_aligned.jsonl", "knowledge_corpus_v43.jsonl")
    digest = hashlib.sha256()
    for path in [*(WORK / name for name in names), *(WORK / "data" / name for name in data)]:
        if not path.is_file():
            raise FileNotFoundError("Seven-tier source/data missing: " + str(path))
        digest.update(str(path.relative_to(WORK)).encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def run_seven_tiers(checkpoint: Path) -> dict:
    verify_checkpoint(checkpoint, expected_step=FULL_STEPS)
    ckpt_sha, profile = sha256_file(checkpoint), seven_tier_profile()
    receipt_path_local = OUT / "seven_tier_evaluation_receipt.json"
    log = OUT / "logs" / "G_seven_tiers.log"
    if receipt_path_local.exists():
        receipt = json.loads(receipt_path_local.read_text(encoding="utf-8"))
        if (receipt.get("status") != "COMPLETE" or receipt.get("checkpoint_sha256") != ckpt_sha
                or receipt.get("profile_sha256") != profile or not log.is_file()
                or receipt.get("log_sha256") != sha256_file(log)):
            raise RuntimeError("Seven-tier completion receipt differs from checkpoint, source, or log")
        print("[G] verified seven-tier evaluation already complete", flush=True)
        return receipt
    run([sys.executable, "runtime_v44fix.py", "--eval_only", "--require_exact_resume",
         "--checkpoint", checkpoint, "--output_dir", OUT / "checkpoints"], "G_seven_tiers")
    receipt = {"schema": "geocpu_v44fix_seven_tier_evaluation_receipt_v1", "status": "COMPLETE",
               "checkpoint_sha256": ckpt_sha, "profile_sha256": profile,
               "log_sha256": sha256_file(log), "training_updates_in_this_stage": 0}
    with receipt_path_local.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return receipt


def main(full=False):
    require_offline_verifier_pin()
    ensure_inputs()
    lineage = OUT / "lineage.json"
    record = {"schema": "geocpu_v44fix_tir_consistency_lineage_v1",
              "v43_checkpoint_sha256": PARENT_SHA256, "v43_source_sha256": V43_SOURCE_SHA256,
              "gate_run": "independent_15000_to_16000_trial",
              "full_run": "separate_15000_to_25000_from_v43_parent",
              "gate_pass_does_not_guarantee_full_result": True,
              "exact_resume_within_each_run_only": True}
    if lineage.exists():
        if json.loads(lineage.read_text(encoding="utf-8")) != record:
            raise RuntimeError("Existing lineage receipt differs")
    else:
        with lineage.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    stage_public()
    stage_build()
    base = gate_eval(V43_CKPT, OUT / "gate_baseline_v43.json", "C_baseline", n_tir=20)
    require_gate_counts(base, n_tir=20)
    summarize("baseline v43", base)
    gate_ckpt = OUT / "gate_checkpoints" / f"checkpoint_step_{GATE_STEP}.pt"
    if not ckpt_done(gate_ckpt):
        train(OUT / "gate_checkpoints", "D_gate_train", stop_at=GATE_STEP, skip_eval=True)
    gate = gate_eval(gate_ckpt, OUT / f"gate_{GATE_STEP}.json", "E_gate_eval", baseline=OUT / "gate_baseline_v43.json")
    require_gate_counts(gate, n_tir=100)
    summarize(f"gate step{GATE_STEP}", gate)
    if gate.get("gate", {}).get("verdict") != "PASS":
        print("\nGATE FAILED: full training is NOT started. See", OUT / f"gate_{GATE_STEP}.json", flush=True)
        return gate
    if not full:
        print("\nGATE PASSED. Run main(full=True) to train 15000 -> 25000.", flush=True)
        return gate
    final_ckpt = OUT / "checkpoints" / f"checkpoint_step_{FULL_STEPS}.pt"
    if not ckpt_done(final_ckpt):
        train(OUT / "checkpoints", "F_full_train", resume=True, skip_eval=True)
    run_seven_tiers(final_ckpt)
    final = gate_eval(final_ckpt, OUT / f"final_{FULL_STEPS}.json", "G_final_eval", baseline=OUT / "gate_baseline_v43.json")
    require_gate_counts(final, n_tir=100)
    summarize(f"final step{FULL_STEPS}", final)
    return final


if __name__ == "__main__":
    main(full="--full" in sys.argv)
