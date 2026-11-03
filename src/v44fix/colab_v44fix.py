#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Colab orchestrator for v44fix (A100). Every stage writes its result to Drive and is skipped on re-run.

  stage A  public data      gpt/prepare_public_v44.py (pinned public sources, eval-overlap exclusion)
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

DRIVE = Path("/content/drive/MyDrive")
V43_ZIP = DRIVE / "GeoCPU_Frontier_CoTraining_v43" / "Grand_cotraining_v43.zip"
V43_CKPT = DRIVE / "GeoCPU_Frontier_CoTraining_v43" / "checkpoints" / "checkpoint_step_15000.pt"
OUT = DRIVE / "GeoCPU_v44fix_20261103" / "run_1"
WORK = Path(__file__).parent.resolve()
PUBLIC = Path("/content/v44fix_public_curriculum")
DATA = Path("/content/v44fix_data")
GATE_STEP = 16000
FULL_STEPS = 25000
MIN_CKPT_BYTES = 500_000_000   # v43's own rule for "this checkpoint is complete"


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
    return Path(p).exists() and Path(p).stat().st_size > MIN_CKPT_BYTES


def fingerprint():
    h = hashlib.sha256()
    for p in (WORK / "gate_eval_v44fix.py", WORK / "integrated_model_v44fix.py", DATA / "tir_heldout.jsonl"):
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()


def ensure_inputs():
    assert V43_CKPT.exists(), f"missing {V43_CKPT}"
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


def stage_public():
    m = PUBLIC / "dataset_manifest.json"
    if m.exists():
        man = json.loads(m.read_text(encoding="utf-8"))
        if man.get("training_eligible"):
            print("[A] public curriculum already prepared", flush=True); return
        raise SystemExit("[A] public curriculum is NOT training-eligible: " + json.dumps(man.get("admission_blockers"), ensure_ascii=False))
    if PUBLIC.exists():
        shutil.rmtree(PUBLIC)   # GPT's script refuses to write into an existing partial directory
    run([sys.executable, "prepare_public_v44.py", "--output-dir", PUBLIC, "--vocab-path", WORK / "base_metadata.json",
         "--evaluation-dir", WORK / "data", "--anchor-path", WORK / "data" / "train_v43_knowledge_unified.jsonl"],
        "A_public", cwd=WORK / "gpt")
    shutil.copy(m, OUT / "public_dataset_manifest.json")


def stage_build():
    if (DATA / "manifest.json").exists():
        print("[B] v44fix mixture already built", flush=True); return
    run([sys.executable, "build_v44fix_data.py", "--public-dir", PUBLIC, "--v43-train", WORK / "data" / "train_v43_knowledge_unified.jsonl",
         "--eval-dir", WORK / "data", "--vocab", WORK / "base_metadata.json", "--out-dir", DATA], "B_build")
    shutil.copy(DATA / "manifest.json", OUT / "v44fix_data_manifest.json")
    shutil.copy(DATA / "tir_heldout.jsonl", OUT / "tir_heldout.jsonl")


def gate_eval(ckpt, out_json, log_name, baseline=None, n_tir=100):
    out_json = Path(out_json)
    if out_json.exists():
        r = json.loads(out_json.read_text(encoding="utf-8"))
        if r.get("fingerprint") == fingerprint():
            print(f"[eval] reuse {out_json.name}", flush=True)
            return r
        print(f"[eval] {out_json.name} was produced by a different gate script/TIR set; recomputing", flush=True)
        out_json.rename(out_json.with_suffix(f".stale_{int(time.time())}.json"))
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
        done = sorted((p for p in out_dir.glob("checkpoint_step_*.pt") if ckpt_done(p)),
                      key=lambda p: int(p.stem.rsplit("_", 1)[1]))
        if done:
            warm = done[-1]
            print(f"[train] resuming from {warm.name} (LR schedule continues; target {FULL_STEPS})", flush=True)
    cmd = [sys.executable, "runtime_v44fix.py", "--train_path", DATA / "train.jsonl", "--warmstart_path", warm,
           "--output_dir", out_dir, "--max_steps", FULL_STEPS]
    if stop_at:
        cmd += ["--stop_at_step", stop_at]
    if skip_eval:
        cmd += ["--skip_final_eval"]
    run(cmd, log_name)


def summarize(tag, r):
    s = {k: r.get(k, {}).get("acc") for k in ("swe", "state", "search", "alu", "tir")}
    s["loop_rate"] = r.get("loops", {}).get("loop_rate")
    s["python_call_rate"] = r.get("tir", {}).get("python_call_rate")
    gate = f"  GATE={r['gate']['verdict']} {r['gate']['checks']}" if "gate" in r else ""
    print(f"[{tag}] " + json.dumps(s, ensure_ascii=False) + gate, flush=True)


def main(full=False):
    ensure_inputs()
    stage_public()
    stage_build()
    base = gate_eval(V43_CKPT, OUT / "gate_baseline_v43.json", "C_baseline", n_tir=20)
    summarize("baseline v43", base)
    gate_ckpt = OUT / "gate_checkpoints" / f"checkpoint_step_{GATE_STEP}.pt"
    if not ckpt_done(gate_ckpt):
        train(OUT / "gate_checkpoints", "D_gate_train", stop_at=GATE_STEP, skip_eval=True)
    gate = gate_eval(gate_ckpt, OUT / f"gate_{GATE_STEP}.json", "E_gate_eval", baseline=OUT / "gate_baseline_v43.json")
    summarize(f"gate step{GATE_STEP}", gate)
    if gate.get("gate", {}).get("verdict") != "PASS":
        print("\nGATE FAILED: full training is NOT started. See", OUT / f"gate_{GATE_STEP}.json", flush=True)
        return gate
    if not full:
        print("\nGATE PASSED. Run main(full=True) to train 15000 -> 25000.", flush=True)
        return gate
    final_ckpt = OUT / "checkpoints" / f"checkpoint_step_{FULL_STEPS}.pt"
    if not ckpt_done(final_ckpt):
        train(OUT / "checkpoints", "F_full_train", resume=True)
    final = gate_eval(final_ckpt, OUT / f"final_{FULL_STEPS}.json", "G_final_eval", baseline=OUT / "gate_baseline_v43.json")
    summarize(f"final step{FULL_STEPS}", final)
    return final


if __name__ == "__main__":
    main(full="--full" in sys.argv)
