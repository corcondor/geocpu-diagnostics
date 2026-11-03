# -*- coding: utf-8 -*-
"""eval_step_control_heldout.py: Rigorous Step Control Evaluation Runner for v41 (247.5M).

Evaluates trained models on the held-out SWE loop test suite (200 instances with unseen odd loop iterations and unseen constants).

Measures:
1. Final Answer Strict Match (exact numeric equality).
2. Native <EOS> (Token 3) Termination.
3. Strict Match + Native <EOS> Confirmed.
4. Step & Loop Execution Fidelity:
   - Exact-Step Rate: Executed exactly the expected loop iterations.
   - Overshoot Rate: Executed more iterations than expected.
   - Undershoot Rate: Stopped before completing all iterations.
   - Mean Overshoot Step Count.
5. Breakdown by Unseen Loop Iteration Count (1, 3, 5, 7).
"""
import sys
import os
import io
import json
import time
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Optional

import torch

ROOT = Path(__file__).parent.resolve()
sys.path.append(str(ROOT))

from GeoCPU_CED_v1 import CEDConfig
from GeoCPU_CED_v2 import GeoCPU_CED_v2
from integrated_model_v30 import IntegratedModelV30

def run_heldout_evaluation(
    checkpoint_path: Optional[str] = None,
    manifest_path: Optional[str] = None,
    arm: str = 'learned',
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    output_audit_path: Optional[str] = None,
    model: Optional[torch.nn.Module] = None,
    vocab: Optional[Dict[str, int]] = None,
    inv_vocab: Optional[Dict[int, str]] = None,
    max_samples: Optional[int] = None
) -> Dict[str, Any]:
    print("=" * 85, flush=True)
    print("HELDOUT SWE STEP-CONTROL EVALUATION (UNSEEN LOOPS & CONSTANTS)", flush=True)
    print("=" * 85, flush=True)
    print(f"  Checkpoint: {checkpoint_path if checkpoint_path else '[In-Memory Model]'}", flush=True)
    print(f"  Arm: {arm} | Device: {device}", flush=True)

    if vocab is None or inv_vocab is None:
        meta = json.loads((ROOT / "base_metadata.json").read_bytes())
        vocab = meta["vocab"]
        inv_vocab = {int(v): k for k, v in vocab.items()}

    dev = torch.device(device)

    if model is None:
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        meta = json.loads((ROOT / "base_metadata.json").read_bytes())
        cfg_dict = dict(meta["config"])
        cfg_dict["scan_backend"] = "jit"
        config = CEDConfig(**cfg_dict)
        base_model = GeoCPU_CED_v2(config)
        model = IntegratedModelV30(base=base_model)
        ckpt_p = Path(checkpoint_path).resolve()
        state_dict = torch.load(ckpt_p, map_location="cpu")
        sd = state_dict["model_state_dict"] if "model_state_dict" in state_dict else state_dict
        model.load_state_dict(sd, strict=False)
        model.to(dev)

    model.eval()

    # Locate manifest
    m_candidates = [
        manifest_path,
        str(ROOT / "data" / "eval_swe_step_control_heldout.jsonl"),
        str(ROOT / "eval_swe_step_control_heldout.jsonl")
    ]
    target_manifest = None
    for cand in m_candidates:
        if cand and os.path.exists(cand):
            target_manifest = cand
            break

    if not target_manifest:
        raise FileNotFoundError(f"Held-out manifest not found in candidates: {m_candidates}")

    questions = [json.loads(line) for line in open(target_manifest, encoding='utf-8')]
    if max_samples is not None and 0 < max_samples < len(questions):
        # Stratified sample across loop counts (1, 3, 5, 7) for fast milestone probing
        by_lp = defaultdict(list)
        for q in questions:
            by_lp[int(q.get("expected_loops", 1))].append(q)
        per_lp = max(1, max_samples // max(1, len(by_lp)))
        sampled = []
        for lp in sorted(by_lp.keys()):
            sampled.extend(by_lp[lp][:per_lp])
        questions = sampled[:max_samples]
    print(f"  -> Total Held-Out Questions: {len(questions)}", flush=True)

    t0 = time.time()
    audit_records = []

    stats = {
        "total": len(questions),
        "correct_val": 0,
        "ended_with_eos": 0,
        "strict_with_eos": 0,
        "exact_step": 0,
        "overshoot": 0,
        "undershoot": 0,
        "total_extra_steps": 0,
        "cpu_invoked_questions": 0,
        "total_cpu_calls": 0,
        "by_loop_count": defaultdict(lambda: {"total": 0, "correct": 0, "exact_step": 0, "overshoot": 0, "undershoot": 0})
    }

    with torch.no_grad():
        for i, item in enumerate(questions):
            prompt = item["prompt"]
            target = item["target"].strip()
            expected_loops = int(item["expected_loops"])

            gen_res = model.generate_with_qwenputer(
                prompt, vocab, inv_vocab, 
                max_tokens=None, arm=arm, 
                repetition_penalty=1.0
            )
            raw_text = gen_res["output_text"].strip()
            trace = gen_res.get("cpu_trace", [])
            cpu_invocations = gen_res.get("cpu_invocations", 0)
            ended_with_eos = gen_res.get("ended_with_eos", False)

            # Analyze CPU execution
            r0_writes = [step["val"] for step in trace if step.get("dest") == "R0"]
            cpu_final_val = str(r0_writes[-1]) if r0_writes else None

            # Count loop iterations executed
            dec_calls = [step for step in trace if step.get("op") == "DEC"]
            compute_calls = [step for step in trace if step.get("op") in ("ADD", "SUB", "MUL", "MOD", "BXOR")]

            if len(dec_calls) > 0:
                executed_loops = len(dec_calls)
            else:
                executed_loops = len(compute_calls)

            # Resolution of answer
            cleaned = re.sub(r'<[^>]+>(?:\s*=>\s*\[[^\]]+\])?', '', raw_text).strip()
            tokens_emitted = cleaned.split()
            text_emitted_ans = tokens_emitted[-1] if tokens_emitted else ""

            if cpu_invocations > 0 and cpu_final_val is not None:
                final_ans = cpu_final_val
            else:
                final_ans = text_emitted_ans

            is_correct = (final_ans == target)

            # Step classification
            if executed_loops == expected_loops:
                step_category = "exact_step"
                stats["exact_step"] += 1
            elif executed_loops > expected_loops:
                step_category = "overshoot"
                stats["overshoot"] += 1
                stats["total_extra_steps"] += (executed_loops - expected_loops)
            else:
                step_category = "undershoot"
                stats["undershoot"] += 1

            if is_correct: stats["correct_val"] += 1
            if ended_with_eos: stats["ended_with_eos"] += 1
            if is_correct and ended_with_eos: stats["strict_with_eos"] += 1
            if cpu_invocations > 0:
                stats["cpu_invoked_questions"] += 1
                stats["total_cpu_calls"] += cpu_invocations

            loop_group = stats["by_loop_count"][expected_loops]
            loop_group["total"] += 1
            if is_correct: loop_group["correct"] += 1
            if step_category == "exact_step": loop_group["exact_step"] += 1
            elif step_category == "overshoot": loop_group["overshoot"] += 1
            else: loop_group["undershoot"] += 1

            audit_records.append({
                "id": item["id"],
                "expected_loops": expected_loops,
                "executed_loops": executed_loops,
                "step_category": step_category,
                "target": target,
                "final_answer": final_ans,
                "is_correct": is_correct,
                "ended_with_eos": ended_with_eos,
                "cpu_invocations": cpu_invocations,
                "raw_output": raw_text
            })

            if (i + 1) % 50 == 0 or (i + 1) == len(questions):
                print(f"  Progress: {i+1:3d}/{len(questions)} | Correct: {stats['correct_val']:2d} | "
                      f"Exact Steps: {stats['exact_step']:2d} | Overshoot: {stats['overshoot']:2d} | "
                      f"Undershoot: {stats['undershoot']:2d} | Elapsed: {time.time()-t0:.1f}s", flush=True)

    elapsed = time.time() - t0
    total = stats["total"]

    print("\n" + "=" * 85, flush=True)
    print("HELDOUT SWE STEP-CONTROL EVALUATION REPORT", flush=True)
    print("=" * 85, flush=True)
    print(f"Total Evaluated: {total} | Elapsed: {elapsed:.1f}s", flush=True)
    print("-" * 85, flush=True)
    print(f"  Final Answer Strict Match:             {stats['correct_val']:3d} / {total} ({stats['correct_val']/total*100:5.1f}%)", flush=True)
    print(f"  Ended with Native <EOS> (Token 3):     {stats['ended_with_eos']:3d} / {total} ({stats['ended_with_eos']/total*100:5.1f}%)", flush=True)
    print(f"  Strict Match + Native <EOS> Confirmed: {stats['strict_with_eos']:3d} / {total} ({stats['strict_with_eos']/total*100:5.1f}%)", flush=True)
    print("-" * 85, flush=True)
    print("  [Step Count & Termination Fidelity Analysis]", flush=True)
    print(f"  * Exact Step Termination (stopped on time): {stats['exact_step']:3d} / {total} ({stats['exact_step']/total*100:5.1f}%)", flush=True)
    print(f"  * Overshoot Rate (extra instructions):     {stats['overshoot']:3d} / {total} ({stats['overshoot']/total*100:5.1f}%)", flush=True)
    print(f"  * Undershoot Rate (stopped too early):      {stats['undershoot']:3d} / {total} ({stats['undershoot']/total*100:5.1f}%)", flush=True)
    if stats['overshoot'] > 0:
        print(f"  * Mean Extra Steps per Overshoot:           {stats['total_extra_steps']/stats['overshoot']:.2f} extra steps", flush=True)
    print("-" * 85, flush=True)
    print("  [Breakdown by Unseen Loop Iteration Count]", flush=True)
    print(f"  {'Loops':<8} | {'Questions':>9} | {'Exact Steps':>12} | {'Overshoot':>10} | {'Undershoot':>11} | {'Accuracy':>9}", flush=True)
    print("-" * 85, flush=True)
    for l_cnt in sorted(stats["by_loop_count"].keys()):
        ld = stats["by_loop_count"][l_cnt]
        l_tot = ld["total"]
        print(f"  {l_cnt:<8d} | {l_tot:9d} | {ld['exact_step']/l_tot*100:11.1f}% | {ld['overshoot']/l_tot*100:9.1f}% | {ld['undershoot']/l_tot*100:10.1f}% | {ld['correct']/l_tot*100:8.1f}%", flush=True)
    print("=" * 85 + "\n", flush=True)

    if output_audit_path:
        out_p = Path(output_audit_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            for r in audit_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Held-out audit log written to: {out_p}", flush=True)

    return stats

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--arm", type=str, default="learned")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-log", type=str, default=None)
    args = parser.parse_args()

    run_heldout_evaluation(
        checkpoint_path=args.checkpoint,
        manifest_path=args.manifest,
        arm=args.arm,
        device=args.device,
        output_audit_path=args.out_log
    )
