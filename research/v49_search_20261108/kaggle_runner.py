"""Runs the robust re-measurements for ONE checkpoint on ONE GPU (set CUDA_VISIBLE_DEVICES before launching).

  python kaggle_runner.py --input /kaggle/input/geocpu-v48-eval-assets-20261108 --study /kaggle/working/study_61500 \
      --run v48_A100_from40500_20261108T045428Z --step 61500 --prefix v48_61500 --out /kaggle/working/out/v48_61500 \
      --code /kaggle/working/code --zips /kaggle/input/geocpu-v48-eval-code-20261108

Uses the original evaluation machinery (eval_candidate.validate_checkpoint / prepare_source / load_model, the same
base bundle and frozen sources). Nothing is trained, nothing leaves the machine.

The two ZIP files (base bundle, evaluation code) are looked up by name, with or without a '.bin' suffix: Kaggle unpacks
every *.zip it is given, and the evaluation pins those archives by the SHA-256 of their exact bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import time
import traceback
import zipfile
from pathlib import Path

CODE_ZIP_SHA256 = "939e75d61aba5cd9f271b46e3d210d42af5a0c51904b80df101dbf40072d55d5"
BASE_BUNDLE_SHA256 = "a25d312429d5b2ee43ec4b812e7a1fb948d0625fbfdfbe9cdf7fac54c075fb70"
BUNDLE = "GeoCPU_v47_fullmodel_bundle_v1_READY.zip"
CODE_ZIP = "GeoCPU_practical_training_logging_20261107_10k.zip"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def link(src: Path, dst: Path):
    src = Path(src)
    if not src.is_file():                       # a symlink to a missing file would only fail much later
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(src, dst)
    except OSError:
        import shutil
        shutil.copyfile(src, dst)


def locate(roots, name: str) -> Path:
    """First regular file called `name` or `name + '.bin'` under any root (never a directory)."""
    for root in roots:
        root = Path(root)
        for cand in (name, name + ".bin"):
            direct = root / cand
            if direct.is_file():
                return direct
            for hit in root.rglob(cand):
                if hit.is_file():
                    return hit
    raise FileNotFoundError(f"{name}[.bin] not found under {[str(r) for r in roots]}")


def build_study(inp: Path, study: Path, run: str, step: int, prefix: str, zips=None) -> Path:
    """The folder layout eval_candidate expects, built from the flat files of the Kaggle dataset(s)."""
    roots = [Path(r) for r in ([inp] + list(zips or []))]
    link(locate(roots, BUNDLE), study / BUNDLE)
    link(locate(roots, CODE_ZIP), study / "GeoCPU_practical_training_logging_20261107" / CODE_ZIP)
    ck = study / run / "checkpoints"
    link(inp / f"{prefix}_checkpoint_step_{step}.pt", ck / f"checkpoint_step_{step}.pt")
    link(inp / f"{prefix}_receipt_step_{step}.json", ck / f"receipt_step_{step}.json")
    link(inp / f"{prefix}_training_saved.json", study / run / "training_saved.json")
    link(inp / f"{prefix}_input_manifest.json", study / run / "input_manifest.json")
    return study


def extract_code(zip_path: Path, dest: Path) -> Path:
    if file_sha256(zip_path) != CODE_ZIP_SHA256:
        raise RuntimeError("evaluation code ZIP differs from the pinned SHA-256")
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)
    return dest


def dedupe_by_prompt(items: list) -> list:
    """T6 has 200 rows but only 21 distinct prompts: generate each once, remember how many rows it stands for."""
    seen, out = {}, []
    for it in items:
        key = it["prompt"]
        if key in seen:
            seen[key]["weight"] += 1
            continue
        it = dict(it, weight=1)
        seen[key] = it
        out.append(it)
    return out


def make_mcq_scorer(model, vocab, arm="learned"):
    """A..D logits at the last prompt position, exactly like the original JMMLU scorer (frozen/r6/eval_benchmark_full_8100.py):
    same normalisation, same vocabulary filter, right padding, bf16 autocast whenever the device is CUDA (the original has no
    hardware check either; recent torch emulates bf16 on a T4)."""
    import torch
    from qwenputer_controller_v30 import canonical_normalize_text           # on sys.path after load_model
    ids = [vocab[c] for c in "ABCD"]
    dev = next(model.parameters()).device
    use_amp = dev.type == "cuda"

    def score(prompts):
        seqs = [[vocab[c] for c in canonical_normalize_text(p) if c in vocab] or [0] for p in prompts]
        max_len = max(32, int(math.ceil(max(len(s) for s in seqs) / 32.0)) * 32)
        x = torch.zeros((len(seqs), max_len), dtype=torch.long)
        valid = torch.zeros((len(seqs), max_len), dtype=torch.bool)
        for r, s in enumerate(seqs):
            x[r, :len(s)] = torch.tensor(s, dtype=torch.long)
            valid[r, :len(s)] = True
        with torch.no_grad(), torch.amp.autocast(device_type=dev.type, enabled=use_amp, dtype=torch.bfloat16):
            out = model(x.to(dev), valid.to(dev), arm=arm)
        return [out.logits[r, len(s) - 1, ids].float().cpu().tolist() for r, s in enumerate(seqs)]
    score.use_amp = use_amp
    return score


def environment_report() -> dict:
    info = {"python": sys.version.split()[0]}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["device"] = torch.cuda.get_device_name(0)
            info["capability"] = list(torch.cuda.get_device_capability(0))
            info["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
    except Exception as e:                      # noqa: BLE001
        info["torch_error"] = repr(e)
    for mod in ("fast_encoder_scan", "integrated_model_v30"):
        m = sys.modules.get(mod)
        if m is not None and hasattr(m, "HAVE_TRITON"):
            info[f"{mod}.HAVE_TRITON"] = bool(m.HAVE_TRITON)
    return info


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--zips", type=Path, nargs="*", default=[], help="extra folders searched for the two ZIP files (name or name.bin)")
    ap.add_argument("--study", type=Path, required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--code", type=Path, required=True, help="folder holding robust_eval.py")
    ap.add_argument("--tasks", default="mcq,t5,t6")
    ap.add_argument("--t5-cap", type=int, default=192)
    ap.add_argument("--t6-cap", type=int, default=256)
    ap.add_argument("--mcq-batch", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke test: items per task (MCQ also gets the longest prompt, to test the worst-case batch)")
    args = ap.parse_args(argv)
    t0 = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.code))
    import robust_eval as R

    study = build_study(args.input, args.study, args.run, args.step, args.prefix, zips=args.zips)
    work = extract_code(study / "GeoCPU_practical_training_logging_20261107" / CODE_ZIP, Path(tempfile.mkdtemp(prefix="geocpu_code_")))
    sys.path.insert(0, str(work))
    import eval_candidate as E
    checkpoint, receipt = E.validate_checkpoint(study, args.run, args.step)
    print("CHECKPOINT_OK", args.prefix, receipt["checkpoint_sha256"][:16], flush=True)
    source = E.prepare_source(study, tempfile.mkdtemp(prefix="geocpu_src_"))
    model, vocab, inverse = E.load_model(source, checkpoint, receipt["checkpoint_sha256"])
    print("MODEL_LOADED", round(time.time() - t0), "s", flush=True)
    data = source / "data"
    tasks = [t for t in args.tasks.split(",") if t]
    summary = {"prefix": args.prefix, "run": args.run, "step": args.step, "checkpoint_sha256": receipt["checkpoint_sha256"],
               "caps": {"t5": args.t5_cap, "t6": args.t6_cap}, "limit": args.limit, "tasks": {}, "seconds_per_task": {},
               "complete": False}
    summary_path = args.out / "summary.json"

    def save_summary():
        summary["seconds"] = round(time.time() - t0)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    failed = []

    def guarded(name, fn):
        """One task failing must not take the others down; the failure is recorded and the exit code reflects it."""
        t_task = time.time()
        try:
            fn()
        except Exception as e:                  # noqa: BLE001
            traceback.print_exc()
            sys.stdout.flush()
            summary["tasks"][name] = {"error": repr(e)}
            failed.append(name)
        summary["seconds_per_task"][name] = round(time.time() - t_task)
        save_summary()

    def run_mcq():
        rows = [json.loads(x) for x in (data / "benchmark_8100_manifest.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        items = [r for r in rows if r.get("task_type") == "multiple_choice"]
        if args.limit:
            longest = max(items, key=lambda r: len(r["prompt"]))
            items = items[:args.limit] + [longest]
        scorer = make_mcq_scorer(model, vocab)
        summary["amp_bf16"] = scorer.use_amp
        t_mcq = time.time()

        def progress(done, total):
            print(f"MCQ_PROGRESS {done}/{total} prompts, {round(time.time() - t_mcq)} s, ETA {round((time.time() - t_mcq) * (total - done) / max(1, done))} s",
                  flush=True)
        res = R.mcq_rotation_eval(items, scorer, batch=args.mcq_batch, cache_path=args.out / "mcq_logits.jsonl", progress=progress)
        (args.out / "mcq.json").write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
        summary["tasks"]["mcq"] = {k: res[k] for k in ("n", "rotation_accuracy", "debiased_accuracy", "debiased_ci",
                                                        "picked_letter_share", "nonfinite", "ties")}
        print("MCQ_DONE", json.dumps(summary["tasks"]["mcq"], ensure_ascii=False), flush=True)

    def run_t5():
        items = R.read_jsonl(data / "eval_v41_logical_instruction_operators_heldout.jsonl")
        seen = {"n": 0}

        def gen(it):
            seen["n"] += 1
            if seen["n"] % 20 == 1:
                print(f"T5_PROGRESS item {seen['n']} {round(time.time() - t0)} s", flush=True)
            return model.generate_with_qwenputer(it["prompt"], vocab, inverse, max_tokens=args.t5_cap, arm="learned",
                                                 repetition_penalty=1.0)

        rows = R.run_items(items, gen, lambda it, g: R.t5_score(it, g, cap=args.t5_cap), args.out / "t5.jsonl", limit=args.limit)
        k = sum(r["correct"] for r in rows)
        summary["tasks"]["t5"] = {"n": len(rows), "correct": k, "ci": R.wilson(k, len(rows)),
                                  "marker_rate": sum(r["has_answer_marker"] for r in rows) / max(1, len(rows)),
                                  "hit_cap_rate": sum(r.get("hit_cap", False) for r in rows) / max(1, len(rows))}
        print("T5_DONE", json.dumps(summary["tasks"]["t5"], ensure_ascii=False), flush=True)

    def run_t6():
        from local_search_engine_v42 import LocalSearchEngineV42
        model.search_engine = LocalSearchEngineV42(corpus_path=data / "knowledge_corpus_v42.jsonl", geosearch_url="")
        items = dedupe_by_prompt(R.read_jsonl(data / "eval_v42_search_retrieval_aligned.jsonl"))

        def gen6(it):
            prompt = it["prompt"]
            if "\n思考:" in prompt:
                prompt = prompt.split("\n思考:")[0] + "\n思考:"
            elif not prompt.endswith("思考:"):
                prompt = prompt.strip() + "\n思考:"
            return model.generate_with_qwenputer_v42(prompt, vocab, inverse, max_tokens=args.t6_cap, arm="learned",
                                                     repetition_penalty=1.0, enable_search=True)

        def score6(it, g):
            s = R.t6_score(it, g)
            s["weight"] = it["weight"]
            return s
        rows = R.run_items(items, gen6, score6, args.out / "t6.jsonl", limit=args.limit)
        summary["tasks"]["t6"] = {"unique_prompts": len(rows), "rows_represented": sum(r["weight"] for r in rows),
                                  "answer_in_output": sum(r["answer_in_output"] for r in rows),
                                  "answer_retrieved": sum(r["answer_retrieved"] for r in rows),
                                  "answer_written": sum(r["answer_written"] for r in rows),
                                  "used_after_retrieval": sum(r["used_after_retrieval"] for r in rows),
                                  "searched": sum(r["search_calls"] > 0 for r in rows)}
        print("T6_DONE", json.dumps(summary["tasks"]["t6"], ensure_ascii=False), flush=True)

    summary["environment"] = environment_report()
    print("ENVIRONMENT", json.dumps(summary["environment"], ensure_ascii=False), flush=True)
    runners = {"mcq": run_mcq, "t5": run_t5, "t6": run_t6}
    for name in tasks:
        guarded(name, runners[name])
    summary["complete"] = not failed
    save_summary()
    print("ALL_DONE" if not failed else "FINISHED_WITH_ERRORS " + ",".join(failed), summary["seconds"], "s", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
