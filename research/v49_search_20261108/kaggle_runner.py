"""Runs the robust re-measurements for ONE checkpoint on ONE GPU (set CUDA_VISIBLE_DEVICES before launching).

  python kaggle_runner.py --input /kaggle/input/geocpu-v48-eval-assets-20261108 --study /kaggle/working/study_61500 \
      --run v48_A100_from40500_20261108T045428Z --step 61500 --prefix v48_61500 --out /kaggle/working/out/v48_61500 \
      --code /kaggle/working/code

Uses the original evaluation machinery (eval_candidate.validate_checkpoint / prepare_source / load_model, the same
base bundle and frozen sources). Nothing is trained, nothing leaves the machine.
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
import zipfile
from pathlib import Path

CODE_ZIP_SHA256 = "939e75d61aba5cd9f271b46e3d210d42af5a0c51904b80df101dbf40072d55d5"
BUNDLE = "GeoCPU_v47_fullmodel_bundle_v1_READY.zip"
CODE_ZIP = "GeoCPU_practical_training_logging_20261107_10k.zip"


def link(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(src, dst)
    except OSError:
        import shutil
        shutil.copyfile(src, dst)


def build_study(inp: Path, study: Path, run: str, step: int, prefix: str) -> Path:
    """The folder layout eval_candidate expects, built from the flat files of the Kaggle dataset."""
    for name in (BUNDLE,):
        link(inp / name, study / name)
    link(inp / CODE_ZIP, study / "GeoCPU_practical_training_logging_20261107" / CODE_ZIP)
    ck = study / run / "checkpoints"
    link(inp / f"{prefix}_checkpoint_step_{step}.pt", ck / f"checkpoint_step_{step}.pt")
    link(inp / f"{prefix}_receipt_step_{step}.json", ck / f"receipt_step_{step}.json")
    link(inp / f"{prefix}_training_saved.json", study / run / "training_saved.json")
    link(inp / f"{prefix}_input_manifest.json", study / run / "input_manifest.json")
    return study


def extract_code(zip_path: Path, dest: Path) -> Path:
    h = hashlib.sha256()
    with zip_path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    if h.hexdigest() != CODE_ZIP_SHA256:
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
    import torch
    from qwenputer_controller_v30 import canonical_normalize_text           # on sys.path after load_model
    ids = [vocab[c] for c in "ABCD"]
    dev = next(model.parameters()).device
    use_amp = dev.type == "cuda" and torch.cuda.is_bf16_supported()

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
    return score


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
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
    ap.add_argument("--limit", type=int, default=None, help="smoke test: items per task")
    args = ap.parse_args(argv)
    t0 = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.code))
    import robust_eval as R

    study = build_study(args.input, args.study, args.run, args.step, args.prefix)
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
               "caps": {"t5": args.t5_cap, "t6": args.t6_cap}, "tasks": {}}

    if "mcq" in tasks:
        rows = [json.loads(x) for x in (data / "benchmark_8100_manifest.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        items = [r for r in rows if r.get("task_type") == "multiple_choice"]
        if args.limit:
            items = items[:args.limit]
        res = R.mcq_rotation_eval(items, make_mcq_scorer(model, vocab), batch=args.mcq_batch)
        (args.out / "mcq.json").write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
        summary["tasks"]["mcq"] = {k: res[k] for k in ("n", "rotation_accuracy", "debiased_accuracy", "debiased_ci", "picked_letter_share")}
        print("MCQ_DONE", json.dumps(summary["tasks"]["mcq"], ensure_ascii=False), flush=True)

    if "t5" in tasks:
        items = R.read_jsonl(data / "eval_v41_logical_instruction_operators_heldout.jsonl")
        gen = lambda it: model.generate_with_qwenputer(it["prompt"], vocab, inverse, max_tokens=args.t5_cap, arm="learned",
                                                      repetition_penalty=1.0)
        rows = R.run_items(items, gen, R.t5_score, args.out / "t5.jsonl", limit=args.limit)
        k = sum(r["correct"] for r in rows)
        summary["tasks"]["t5"] = {"n": len(rows), "correct": k, "ci": R.wilson(k, len(rows)),
                                  "marker_rate": sum(r["has_answer_marker"] for r in rows) / max(1, len(rows))}
        print("T5_DONE", json.dumps(summary["tasks"]["t5"], ensure_ascii=False), flush=True)

    if "t6" in tasks:
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
                                  "used_after_retrieval": sum(r["used_after_retrieval"] for r in rows),
                                  "searched": sum(r["search_calls"] > 0 for r in rows)}
        print("T6_DONE", json.dumps(summary["tasks"]["t6"], ensure_ascii=False), flush=True)

    summary["seconds"] = round(time.time() - t0)
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("ALL_DONE", summary["seconds"], "s", flush=True)


if __name__ == "__main__":
    main()
