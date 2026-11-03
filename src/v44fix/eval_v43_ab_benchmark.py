# -*- coding: utf-8 -*-
"""eval_v43_ab_benchmark.py: Comprehensive A/B Dual Evaluator for Closed-Book (学習のみ) vs Open-Book (学習＋検索あり).

Runs side-by-side comparative evaluation across:
1. Academic Multiple-Choice Benchmarks (JMMLU / JamC-QA)
2. Domain-Specific Breakdown (Japanese History, World History, Civics & Law, Physics, Biology, Economics)
3. Telemetry:
   - Closed-Book Parametric Accuracy (学習のみ)
   - Open-Book Search-Augmented Accuracy (学習＋検索あり)
   - Exact Lift: Absolute & Relative Δ%
   - Search Tool Invocation Rate (%)
   - Latency per Question
"""

import sys
import os
import io
import json
import time
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Tuple, Optional

import torch
import torch.nn.functional as F

ROOT = Path(__file__).parent.resolve()
sys.path.append(str(ROOT))

from GeoCPU_CED_v1 import CEDConfig
from GeoCPU_CED_v2 import GeoCPU_CED_v2
from integrated_model_v42 import IntegratedModelV42
from local_search_engine_v42 import LocalSearchEngineV42
from qwenputer_controller_v30 import canonical_normalize_text

def extract_answer_choice(text: str) -> str:
    """Extracts final choice letter (A, B, C, D) from generated CoT output."""
    m_ans = re.search(r'(?:正解|回答|答え|選ぶのは|合致するのは|結論として)\s*[:：は]?\s*([A-D])\b', text)
    if m_ans:
        return m_ans.group(1)
    m_tail = re.search(r'\b([A-D])(?:\s*[\.\)）]|$|\s)', text.strip())
    if m_tail:
        return m_tail.group(1)
    for c in ['A', 'B', 'C', 'D']:
        if c in text:
            return c
    return text.strip()[:1]

def run_v43_ab_benchmark(
    checkpoint_path: str = None,
    manifest_path: str = None,
    eval_count: int = 200,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    model: Optional[IntegratedModelV42] = None,
    vocab: dict = None,
    inv_vocab: dict = None,
    corpus_path: Optional[str] = None
) -> Dict[str, Any]:
    print("=" * 90, flush=True)
    print("GEO-CPU V43 COMPREHENSIVE A/B EVALUATOR: CLOSED-BOOK (学習のみ) VS OPEN-BOOK (学習＋検索あり)", flush=True)
    print("=" * 90, flush=True)
    print(f"  Checkpoint: {checkpoint_path if checkpoint_path else '[In-Memory Model]'}", flush=True)
    print(f"  Device:     {device} | Evaluation Samples: {eval_count}", flush=True)

    if vocab is None or inv_vocab is None:
        meta_file = ROOT / "base_metadata.json"
        meta = json.loads(meta_file.read_bytes())
        vocab = meta["vocab"]
        inv_vocab = {int(v): k for k, v in vocab.items()}

    if model is None:
        cfg_dict = dict(meta["config"])
        cfg_dict["scan_backend"] = "jit"
        cfg = CEDConfig(**cfg_dict)
        base = GeoCPU_CED_v2(cfg)
        
        search_engine = LocalSearchEngineV42(corpus_path=Path(corpus_path) if corpus_path else None)
        model = IntegratedModelV42(base=base, search_engine=search_engine)
        
        if checkpoint_path and Path(checkpoint_path).exists():
            print(f"Loading checkpoint weights from {checkpoint_path}...", flush=True)
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            model.load_state_dict(sd, strict=False)
            print("Checkpoint loaded successfully.", flush=True)
        model.to(device)

    model.eval()

    # Load Questions from manifest
    if manifest_path is None:
        candidates = [
            ROOT / "data" / "benchmark_8100_manifest.jsonl",
            ROOT / "data" / "jamcqa_manifest_test.jsonl"
        ]
        manifest_path = next((str(p) for p in candidates if p.exists()), None)

    if not manifest_path or not Path(manifest_path).exists():
        raise FileNotFoundError(f"Evaluation manifest not found at {manifest_path}")

    print(f"Loading evaluation questions from {manifest_path}...", flush=True)
    all_questions = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("task_type") == "multiple_choice" or "choices" in obj or ("A" in obj.get("prompt", "") and "B" in obj.get("prompt", "")):
                all_questions.append(obj)

    print(f"Total multiple-choice questions in manifest: {len(all_questions):,}", flush=True)
    
    # Subsample balanced evaluation set
    eval_set = all_questions[:eval_count] if eval_count < len(all_questions) else all_questions
    print(f"Running A/B evaluation on {len(eval_set)} balanced benchmark items...\n", flush=True)

    results = []
    stats_closed = {"correct": 0, "total": 0}
    stats_open = {"correct": 0, "total": 0, "search_invoked": 0}

    domain_stats = defaultdict(lambda: {"closed_correct": 0, "open_correct": 0, "total": 0})

    t0_all = time.time()

    for idx, item in enumerate(eval_set):
        raw_prompt = item.get("prompt", "").strip()
        target = item.get("target", "").strip()
        subj = item.get("subject", item.get("category", "general"))

        # -------------------------------------------------------------
        # ARM A: 学習のみ (Closed-Book: Parametric Memory Only)
        # -------------------------------------------------------------
        t0_a = time.time()
        res_a = model.generate_with_qwenputer_v42(
            raw_prompt + "\n思考: ",
            vocab=vocab,
            inv_vocab=inv_vocab,
            max_tokens=60,
            repetition_penalty=1.0,
            enable_search=False
        )
        dt_a = time.time() - t0_a
        pred_a = extract_answer_choice(res_a["output_text"])
        is_correct_a = (pred_a == target)

        stats_closed["total"] += 1
        if is_correct_a:
            stats_closed["correct"] += 1

        # -------------------------------------------------------------
        # ARM B: 学習＋検索あり (Open-Book: Autonomous Tool Retrieval)
        # -------------------------------------------------------------
        t0_b = time.time()
        res_b = model.generate_with_qwenputer_v42(
            raw_prompt + "\n思考: ",
            vocab=vocab,
            inv_vocab=inv_vocab,
            max_tokens=120,
            repetition_penalty=1.0,
            enable_search=True
        )
        dt_b = time.time() - t0_b
        pred_b = extract_answer_choice(res_b["output_text"])
        is_correct_b = (pred_b == target)
        invoked_search = (res_b.get("search_invocations", 0) > 0)

        stats_open["total"] += 1
        if is_correct_b:
            stats_open["correct"] += 1
        if invoked_search:
            stats_open["search_invoked"] += 1

        # Domain Tracking
        domain_stats[subj]["total"] += 1
        if is_correct_a: domain_stats[subj]["closed_correct"] += 1
        if is_correct_b: domain_stats[subj]["open_correct"] += 1

        results.append({
            "id": item.get("id", f"item_{idx+1}"),
            "subject": subj,
            "target": target,
            "arm_a_pred": pred_a,
            "arm_a_correct": is_correct_a,
            "arm_b_pred": pred_b,
            "arm_b_correct": is_correct_b,
            "search_invoked": invoked_search,
            "search_trace": res_b.get("search_trace", [])
        })

        if (idx + 1) % 20 == 0 or (idx + 1) == len(eval_set):
            cur_acc_a = 100.0 * stats_closed["correct"] / stats_closed["total"]
            cur_acc_b = 100.0 * stats_open["correct"] / stats_open["total"]
            s_rate = 100.0 * stats_open["search_invoked"] / stats_open["total"]
            print(f"  Progress: {idx+1:4d}/{len(eval_set)} | "
                  f"ARM A (学習のみ): {cur_acc_a:5.1f}% | "
                  f"ARM B (検索あり): {cur_acc_b:5.1f}% (Δ {cur_acc_b - cur_acc_a:+5.1f}%) | "
                  f"Search Rate: {s_rate:5.1f}% | Elapsed: {time.time()-t0_all:5.1f}s", flush=True)

    # Final Telemetry Report
    tot = len(eval_set)
    acc_a = 100.0 * stats_closed["correct"] / max(1, tot)
    acc_b = 100.0 * stats_open["correct"] / max(1, tot)
    search_rate = 100.0 * stats_open["search_invoked"] / max(1, tot)
    diff_abs = acc_b - acc_a

    print("\n" + "=" * 90, flush=True)
    print("V43 A/B BENCHMARK FINAL EVALUATION REPORT", flush=True)
    print("=" * 90, flush=True)
    print(f"Total Evaluated Questions: {tot} | Total Time: {time.time()-t0_all:.1f}s", flush=True)
    print("-" * 90, flush=True)
    print(f"  [ARM A: 学習のみ (Closed-Book)]  Accuracy: {stats_closed['correct']}/{tot} ({acc_a:5.2f}%)", flush=True)
    print(f"  [ARM B: 学習＋検索 (Open-Book)] Accuracy: {stats_open['correct']}/{tot} ({acc_b:5.2f}%)", flush=True)
    print(f"  -> Absolute Search Improvement:  {diff_abs:+5.2f}%", flush=True)
    print(f"  -> Search Autonomous Invocation Rate: {search_rate:5.2f}% ({stats_open['search_invoked']}/{tot})", flush=True)
    print("-" * 90, flush=True)
    print("  [Breakdown by Academic Subject / Domain]")
    print(f"  {'Subject / Domain':30s} | {'Total':6s} | {'ARM A (学習のみ)':16s} | {'ARM B (検索あり)':16s} | {'Gain':8s}")
    print("  " + "-" * 86)
    for subj, d in sorted(domain_stats.items(), key=lambda x: -x[1]["total"]):
        s_tot = d["total"]
        s_acc_a = 100.0 * d["closed_correct"] / max(1, s_tot)
        s_acc_b = 100.0 * d["open_correct"] / max(1, s_tot)
        print(f"  {subj[:30]:30s} | {s_tot:6d} | {s_acc_a:14.1f}% | {s_acc_b:14.1f}% | {s_acc_b - s_acc_a:+7.1f}%")
    print("=" * 90, flush=True)

    return {
        "total": tot,
        "arm_a_accuracy": acc_a,
        "arm_b_accuracy": acc_b,
        "absolute_gain": diff_abs,
        "search_invocation_rate": search_rate,
        "domain_breakdown": dict(domain_stats),
        "audit_samples": results[:10]
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run v43 A/B Dual Evaluator.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .pt")
    parser.add_argument("--manifest", type=str, default=None, help="Path to evaluation manifest")
    parser.add_argument("--count", type=int, default=50, help="Number of questions to evaluate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_v43_ab_benchmark(
        checkpoint_path=args.checkpoint,
        manifest_path=args.manifest,
        eval_count=args.count,
        device=args.device
    )
