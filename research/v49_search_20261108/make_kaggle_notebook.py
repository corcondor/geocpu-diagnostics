"""Builds the Kaggle kernel (notebook + kernel-metadata.json) that measures two checkpoints, one per T4, in parallel.

  python make_kaggle_notebook.py OUT_DIR

The notebook carries its own code (robust_eval.py, kaggle_runner.py, kaggle_compare.py are embedded as strings), so the
kernel needs no internet. Inputs: two private datasets
  geocpuresearch/geocpu-v48-eval-assets-20261108  (checkpoints, receipts, manifests)
  geocpuresearch/geocpu-v48-eval-code-20261108    (the two pinned ZIPs, stored as *.zip.bin because Kaggle unpacks *.zip)
Stages: setup (pins + GPU check) -> smoke (2 items per task, both GPUs, must pass) -> full run -> report + cleanup.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
DATASET = "geocpuresearch/geocpu-v48-eval-assets-20261108"
DATASET_CODE = "geocpuresearch/geocpu-v48-eval-code-20261108"
SLUG = "geocpu-v48-robust-eval-20261108"

files = {n: (HERE / n).read_text(encoding="utf-8") for n in ("robust_eval.py", "kaggle_runner.py", "kaggle_compare.py")}

cell_setup = '''# 1. setup: write the code, find the inputs, verify the pinned archives, look at the GPUs
import hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path
CODE = Path("/kaggle/working/code"); CODE.mkdir(parents=True, exist_ok=True)
FILES = __FILES__
for name, text in FILES.items():
    (CODE / name).write_text(text, encoding="utf-8")
sys.path.insert(0, str(CODE))
import kaggle_runner as K

hits = list(Path("/kaggle/input").rglob("v48_61500_checkpoint_step_61500.pt"))
assert hits, "dataset geocpu-v48-eval-assets-20261108 is not attached"
INPUT = hits[0].parent
ZIP_ROOTS = [p.parent for p in Path("/kaggle/input").rglob(K.BUNDLE + ".bin")] or [INPUT]
bundle, code_zip = K.locate([*ZIP_ROOTS, INPUT], K.BUNDLE), K.locate([*ZIP_ROOTS, INPUT], K.CODE_ZIP)
assert K.file_sha256(bundle) == K.BASE_BUNDLE_SHA256, f"base bundle bytes differ from the pin: {bundle}"
assert K.file_sha256(code_zip) == K.CODE_ZIP_SHA256, f"evaluation code bytes differ from the pin: {code_zip}"
for f in sorted(INPUT.glob("v4*_*")):
    print(f"{f.name:60s} {f.stat().st_size:>14,d}")
print("bundle", bundle, bundle.stat().st_size, "| code", code_zip, code_zip.stat().st_size)
print("input:", INPUT, "| zips:", ZIP_ROOTS)
gpus = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout
print(gpus)
N_GPU = len([l for l in gpus.splitlines() if l.strip()])
assert N_GPU >= 1, "no GPU"
JOBS = [("v47_30500", "v47_from28000_to30500_curriculum_v1", 30500), ("v48_61500", "v48_A100_from40500_20261108T045428Z", 61500)]


def launch(out_dir, extra=()):
    out_dir.mkdir(parents=True, exist_ok=True)
    procs = []
    for k, (prefix, run, step) in enumerate(JOBS):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(k if N_GPU > 1 else 0), PYTHONUNBUFFERED="1", PYTHONUTF8="1")
        cmd = [sys.executable, "-u", str(CODE / "kaggle_runner.py"), "--input", str(INPUT), "--zips", *map(str, ZIP_ROOTS),
               "--study", f"/kaggle/working/study_{prefix}", "--run", run, "--step", str(step), "--prefix", prefix,
               "--out", str(out_dir / prefix), "--code", str(CODE), *extra]
        log = open(out_dir / f"{prefix}.log", "w", encoding="utf-8")
        procs.append((prefix, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env), log))
        if N_GPU <= 1:                  # a single GPU: one after the other
            procs[-1][1].wait()
    return procs


def wait(procs, out_dir, deadline=None, every=120):
    t0 = time.time()
    while any(p.poll() is None for _, p, _ in procs):
        time.sleep(every)
        for prefix, p, _ in procs:
            lines = (out_dir / f"{prefix}.log").read_text(encoding="utf-8", errors="replace").splitlines()
            n5 = len((out_dir / prefix / "t5.jsonl").read_text().splitlines()) if (out_dir / prefix / "t5.jsonl").exists() else 0
            n6 = len((out_dir / prefix / "t6.jsonl").read_text().splitlines()) if (out_dir / prefix / "t6.jsonl").exists() else 0
            print(f"[{int(time.time() - t0) // 60} min] {prefix} rc={p.poll()} t5={n5} t6={n6} :: {lines[-1][:150] if lines else ''}", flush=True)
        if deadline and time.time() > deadline:
            print("DEADLINE reached: stopping children so that the report is still produced", flush=True)
            for _, p, _ in procs:
                if p.poll() is None:
                    p.terminate()
            time.sleep(30)
    bad = []
    for prefix, p, log in procs:
        log.close()
        print(prefix, "exit", p.returncode)
        if p.returncode:
            bad.append(prefix)
            print((out_dir / f"{prefix}.log").read_text(encoding="utf-8", errors="replace")[-4000:])
    return bad
'''.replace("__FILES__", json.dumps(files, ensure_ascii=False))

cell_smoke = '''# 2. smoke test on the real GPUs: 2 items per task (+ the longest JMMLU prompt), tiny generation caps. Must pass before the full run.
SMOKE = Path("/kaggle/working/out_smoke")
bad = wait(launch(SMOKE, ["--limit", "2", "--t5-cap", "16", "--t6-cap", "16"]), SMOKE, every=20)
for prefix, *_ in JOBS:
    log = (SMOKE / f"{prefix}.log").read_text(encoding="utf-8", errors="replace")
    summ = json.loads((SMOKE / prefix / "summary.json").read_text()) if (SMOKE / prefix / "summary.json").exists() else {}
    print(prefix, "ALL_DONE" in log, summ.get("seconds_per_task"), summ.get("environment"))
    if "ALL_DONE" not in log and prefix not in bad:
        bad.append(prefix)
assert not bad, f"smoke test failed for {bad}; see the log tails above (nothing was spent on the full run)"
print("SMOKE OK")
'''

cell_run = '''# 3. the full measurement: both checkpoints at once, one per GPU
OUT = Path("/kaggle/working/out")
DEADLINE = time.time() + 10.5 * 3600
bad = wait(launch(OUT), OUT, deadline=DEADLINE)
(OUT / "STATUS.txt").write_text(("FAILED: " + ",".join(bad)) if bad else "OK", encoding="utf-8")
'''

cell_report = '''# 4. paired comparison, report, cleanup
r = subprocess.run([sys.executable, str(CODE / "kaggle_compare.py"), "--out", str(OUT), "--base", "v47_30500", "--new", "v48_61500"],
                   capture_output=True, text=True, env=dict(os.environ, PYTHONUTF8="1"))
print(r.stdout)
print(r.stderr[-3000:])
print("compare rc", r.returncode)
for prefix, *_ in JOBS:                 # only symlinks / small copies; the results stay in out/
    shutil.rmtree(f"/kaggle/working/study_{prefix}", ignore_errors=True)
shutil.rmtree(SMOKE, ignore_errors=True)
assert not bad and r.returncode == 0, f"failed: children={bad} compare_rc={r.returncode}; the report above says what is missing"
'''


def code(src, cid):
    return {"cell_type": "code", "execution_count": None, "id": cid, "metadata": {}, "outputs": [], "source": src.splitlines(keepends=True)}


nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
      "cells": [{"cell_type": "markdown", "id": "intro", "metadata": {},
                 "source": ["# GeoCPU v48: 修正版の評価（T5・T6・JMMLU の位置入れ替え）\n", "30,500 と 61,500 を、T4 を 1 台ずつ使って同時に測り、対応のある検定で比べます。学習はしません。\n"]},
                code(cell_setup, "setup"), code(cell_smoke, "smoke"), code(cell_run, "run"), code(cell_report, "report")]}
(OUT / f"{SLUG}.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
meta = {"id": f"geocpuresearch/{SLUG}", "title": "GeoCPU v48 robust eval 20261108", "code_file": f"{SLUG}.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
        "keywords": [], "dataset_sources": [DATASET, DATASET_CODE], "kernel_sources": [], "competition_sources": [], "model_sources": [],
        "machine_shape": "NvidiaTeslaT4"}
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("written", OUT)
