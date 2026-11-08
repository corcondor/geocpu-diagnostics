"""Builds the Kaggle kernel (notebook + kernel-metadata.json) that measures two checkpoints, one per T4, in parallel.

  python make_kaggle_notebook.py OUT_DIR

The notebook carries its own code (robust_eval.py, kaggle_runner.py, kaggle_compare.py are embedded as strings), so the
kernel needs no internet and no second dataset. Input: the private dataset geocpuresearch/geocpu-v48-eval-assets-20261108.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
DATASET = "geocpuresearch/geocpu-v48-eval-assets-20261108"
SLUG = "geocpu-v48-robust-eval-20261108"

files = {n: (HERE / n).read_text(encoding="utf-8") for n in ("robust_eval.py", "kaggle_runner.py", "kaggle_compare.py")}

cell_setup = '''# 1. setup: write the code, locate the dataset, check the GPUs
import json, os, subprocess, sys, time
from pathlib import Path
CODE = Path("/kaggle/working/code"); CODE.mkdir(parents=True, exist_ok=True)
FILES = %s
for name, text in FILES.items():
    (CODE / name).write_text(text, encoding="utf-8")
hits = list(Path("/kaggle/input").rglob("v48_61500_checkpoint_step_61500.pt"))
assert hits, "dataset geocpu-v48-eval-assets-20261108 is not attached"
INPUT = hits[0].parent
print("input:", INPUT)
print(subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv"], capture_output=True, text=True).stdout)
''' % json.dumps(files, ensure_ascii=False)

cell_run = '''# 2. run both checkpoints in parallel, one per GPU (falls back to one after the other if only one GPU is visible)
OUT = Path("/kaggle/working/out"); OUT.mkdir(exist_ok=True)
JOBS = [("v47_30500", "v47_from28000_to30500_curriculum_v1", 30500), ("v48_61500", "v48_A100_from40500_20261108T045428Z", 61500)]
n_gpu = len([l for l in subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.splitlines() if l.startswith("GPU")])
LIMIT = None      # set e.g. 8 for a smoke test
procs = []
for k, (prefix, run, step) in enumerate(JOBS):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(k if n_gpu > 1 else 0), PYTHONUNBUFFERED="1")
    cmd = [sys.executable, "-u", str(CODE / "kaggle_runner.py"), "--input", str(INPUT), "--study", f"/kaggle/working/study_{prefix}",
           "--run", run, "--step", str(step), "--prefix", prefix, "--out", str(OUT / prefix), "--code", str(CODE)]
    if LIMIT:
        cmd += ["--limit", str(LIMIT)]
    log = open(OUT / f"{prefix}.log", "w")
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    procs.append((prefix, p, log))
    if n_gpu <= 1:
        p.wait()
t0 = time.time()
while any(p.poll() is None for _, p, _ in procs):
    time.sleep(120)
    for prefix, p, _ in procs:
        tail = (OUT / f"{prefix}.log").read_text(encoding="utf-8", errors="replace").splitlines()[-1:]
        print(f"[{int(time.time()-t0)//60} min] {prefix} rc={p.poll()} :: {tail[0][:160] if tail else ''}", flush=True)
for prefix, p, log in procs:
    log.close()
    print(prefix, "exit", p.returncode)
'''

cell_report = '''# 3. paired comparison and report
print(subprocess.run([sys.executable, str(CODE / "kaggle_compare.py"), "--out", str(OUT), "--base", "v47_30500", "--new", "v48_61500"],
                     capture_output=True, text=True).stdout)
'''


def code(src, cid):
    return {"cell_type": "code", "execution_count": None, "id": cid, "metadata": {}, "outputs": [], "source": src.splitlines(keepends=True)}


nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
      "cells": [{"cell_type": "markdown", "id": "intro", "metadata": {},
                 "source": ["# GeoCPU v48: 修正版の評価（T5・T6・JMMLU の位置入れ替え）\n", "30,500 と 61,500 を、T4 を 1 台ずつ使って同時に測り、対応のある検定で比べます。学習はしません。\n"]},
                code(cell_setup, "setup"), code(cell_run, "run"), code(cell_report, "report")]}
(OUT / f"{SLUG}.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
meta = {"id": f"geocpuresearch/{SLUG}", "title": "GeoCPU v48 robust eval 20261108", "code_file": f"{SLUG}.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
        "keywords": [], "dataset_sources": [DATASET], "kernel_sources": [], "competition_sources": [], "model_sources": [],
        "machine_shape": "NvidiaTeslaT4"}
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("written", OUT)
