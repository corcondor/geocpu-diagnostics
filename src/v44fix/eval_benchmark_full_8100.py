# -*- coding: utf-8 -*-
"""Canonical 8,100-Question Broad-Spectrum Full Benchmark Runner for v41 (247.5M).

Integrity & Evaluation Principles:
1. 100% Zero-Downsampling:
   - 7,536 JMMLU Questions (All 56 subjects)
   - 400 ARC-AGI Discrete Grid Reasoning Tasks
   - 164 SWE Software & Program State Tracing Tasks
   Total = 8,100 Questions.
2. Strict Final Answer Scoring (Zero Partial-Match Loophole):
   - Eliminates substring partial matches.
   - Evaluates strict exact equality of the model's final emitted answer.
3. Multi-Dimensional Telemetry:
   - Tracks CPU opcode invocations, CPU register R0 correctness, and final answer match separately.
4. Generous Generation Budget (max_tokens=160).
5. In-Memory Model Reuse: Zero reloading overhead when called from co-training runtime.
"""
import sys
import os
import io
import json
import time
import math
import argparse
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).parent.resolve()
sys.path.append(str(ROOT))

from GeoCPU_CED_v1 import CEDConfig
from GeoCPU_CED_v2 import GeoCPU_CED_v2
from integrated_model_v30 import IntegratedModelV30
from qwenputer_controller_v30 import QwenputerExecutionController, canonical_normalize_text

def wilson_score_interval(correct: int, total: int, confidence: float = 0.95) -> Tuple[float, float, float]:
    if total == 0:
        return 0.0, 0.0, 0.0
    p = correct / total
    z = 1.960
    denominator = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denominator
    spread = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
    lower = max(0.0, centre - spread)
    upper = min(1.0, centre + spread)
    return p, lower, upper

def detect_character_repetition(text: str, max_repeat: int = 5) -> Tuple[bool, str]:
    for i in range(len(text) - max_repeat):
        sub = text[i:i + max_repeat]
        if len(set(sub)) == 1:
            return True, sub[0]
    return False, ""

def run_evaluation(
    checkpoint_path: Optional[str] = None,
    manifest_path: Optional[str] = None,
    arm: str = 'learned',
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    batch_size: int = 64,
    output_audit_path: Optional[str] = None,
    model: Optional[torch.nn.Module] = None,
    vocab: Optional[Dict[str, int]] = None,
    inv_vocab: Optional[Dict[int, str]] = None
) -> Dict[str, Any]:
    print("=" * 90, flush=True)
    print("CANONICAL 8,100-QUESTION FULL BENCHMARK RUNNER (ZERO DOWNSAMPLING - STRICT EVAL)", flush=True)
    print("=" * 90, flush=True)
    print(f"  Checkpoint:     {checkpoint_path if checkpoint_path else '[In-Memory Model]'}", flush=True)
    print(f"  Evaluation Arm: {arm}", flush=True)
    print(f"  Execution Dev:  {device}", flush=True)
    print(f"  Batch Size:     {batch_size}", flush=True)

    if vocab is None or inv_vocab is None:
        meta = json.loads((ROOT / "base_metadata.json").read_bytes())
        vocab = meta["vocab"]
        inv_vocab = {int(v): k for k, v in vocab.items()}

    dev = torch.device(device)

    if model is None:
        print("[1/3] Loading Qwenputer 247.5M Model from Checkpoint...", flush=True)
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        meta = json.loads((ROOT / "base_metadata.json").read_bytes())
        cfg_dict = dict(meta["config"])
        cfg_dict["max_positions"] = 2048
        cfg_dict["scan_backend"] = "jit"
        config = CEDConfig(**cfg_dict)
        base_model = GeoCPU_CED_v2(config, select_after=meta.get("select_after", 4))
        model = IntegratedModelV30(base=base_model)
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        sd = state_dict["model_state_dict"] if "model_state_dict" in state_dict else state_dict
        model.load_state_dict(sd, strict=False)
        if "router_bias" in state_dict and state_dict["router_bias"] is not None:
            model.routing_module.router_bias.copy_(torch.tensor(float(state_dict["router_bias"])))
        model.to(dev)
        print("  -> Model loaded successfully.", flush=True)

    model.eval()

    # Locate manifest
    m_candidates = [
        manifest_path,
        str(ROOT / "data" / "benchmark_8100_manifest.jsonl"),
        str(ROOT / "benchmark_8100_manifest.jsonl")
    ]
    target_manifest = None
    for cand in m_candidates:
        if cand and os.path.exists(cand):
            target_manifest = cand
            break

    if not target_manifest:
        raise FileNotFoundError(f"Manifest not found in: {m_candidates}")

    print(f"[2/3] Loading 8,100 Evaluation Questions from {Path(target_manifest).name}...", flush=True)
    questions = [json.loads(line) for line in open(target_manifest, encoding='utf-8')]
    print(f"  -> Total questions loaded: {len(questions)}", flush=True)

    mc_questions = [q for q in questions if q.get("task_type") == "multiple_choice"]
    gen_questions = [q for q in questions if q.get("task_type") != "multiple_choice"]
    print(f"     * JMMLU Multiple Choice (Batched): {len(mc_questions)}", flush=True)
    print(f"     * ARC & SWE Generative:           {len(gen_questions)}", flush=True)

    choice_ids = {letter: vocab[letter] for letter in ['A', 'B', 'C', 'D'] if letter in vocab}
    choice_tokens = list(choice_ids.keys())
    choice_tensor_ids = [choice_ids[c] for c in choice_tokens]

    t0 = time.time()
    audit_records = []
    pillar_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    category_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    subject_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    swe_telemetry = {
        "cpu_invoked_questions": 0,
        "total_cpu_calls": 0,
        "cpu_r0_correct": 0,
        "final_ans_correct": 0
    }
    arc_telemetry = {
        "completed_grids": 0,
        "exact_matches": 0,
        "repetitions": 0
    }

    # Step A: JMMLU Batched
    print(f"[3/3] Executing Batched Evaluation on {len(mc_questions)} JMMLU Questions...", flush=True)
    use_amp = (dev.type == "cuda")
    with torch.no_grad(), torch.amp.autocast(device_type=dev.type, enabled=use_amp, dtype=torch.bfloat16):
        for b_start in range(0, len(mc_questions), batch_size):
            batch_items = mc_questions[b_start : b_start + batch_size]
            seqs = [[vocab[c] for c in canonical_normalize_text(item["prompt"]) if c in vocab] for item in batch_items]
            seqs = [s if len(s) > 0 else [0] for s in seqs]

            max_len = max(32, int(math.ceil(max(len(s) for s in seqs) / 32.0)) * 32)
            B = len(batch_items)
            x_cpu = torch.zeros((B, max_len), dtype=torch.long)
            valid_cpu = torch.zeros((B, max_len), dtype=torch.bool)
            last_indices = []

            for row_idx, s in enumerate(seqs):
                x_cpu[row_idx, :len(s)] = torch.tensor(s, dtype=torch.long)
                valid_cpu[row_idx, :len(s)] = True
                last_indices.append(len(s) - 1)

            x = x_cpu.to(dev, non_blocking=True)
            valid = valid_cpu.to(dev, non_blocking=True)
            out = model(x, valid, arm=arm)

            for row_idx, item in enumerate(batch_items):
                last_idx = last_indices[row_idx]
                token_logits = out.logits[row_idx, last_idx, choice_tensor_ids]
                best_choice_idx = torch.argmax(token_logits).item()
                pred_text = choice_tokens[best_choice_idx]
                target = item["target"].strip()
                is_correct = (pred_text == target)

                pillar = item["pillar"]
                cat = item.get("category", "general")
                subj = item.get("subject", "general")

                pillar_stats[pillar]["total"] += 1
                if is_correct: pillar_stats[pillar]["correct"] += 1
                category_stats[cat]["total"] += 1
                if is_correct: category_stats[cat]["correct"] += 1
                subject_stats[subj]["total"] += 1
                if is_correct: subject_stats[subj]["correct"] += 1

                audit_records.append({
                    "id": item["id"],
                    "pillar": pillar,
                    "category": cat,
                    "subject": subj,
                    "target": target,
                    "prediction": pred_text,
                    "is_correct": is_correct
                })

            if (b_start + batch_size) % 1024 < batch_size or (b_start + batch_size) >= len(mc_questions):
                done_n = min(len(mc_questions), b_start + batch_size)
                cur_acc = pillar_stats["JMMLU_Academic"]["correct"] / max(1, done_n)
                print(f"  JMMLU: {done_n:4d}/{len(mc_questions)} ({100*done_n/len(mc_questions):5.1f}%) | "
                      f"Acc: {cur_acc*100:5.2f}% | Elapsed: {time.time() - t0:4.1f}s", flush=True)

    # Step B: ARC & SWE Generative
    print(f"\nExecuting Generative Evaluation on {len(gen_questions)} ARC & SWE Tasks (Autonomous <EOS> Termination)...", flush=True)
    with torch.no_grad():
        for i, item in enumerate(gen_questions):
            prompt = item["prompt"]
            target = item["target"].strip()
            task_type = item.get("task_type")
            pillar = item["pillar"]
            cat = item.get("category", "general")
            subj = item.get("subject", "general")

            is_arc = (task_type == "grid_transform")
            gen_res = model.generate_with_qwenputer(
                prompt, vocab, inv_vocab,
                max_tokens=None,
                arm=arm,
                stop_strings=["]]"] if is_arc else None
            )
            raw_text = gen_res["output_text"].strip()
            trace = gen_res.get("cpu_trace", [])
            cpu_invocations = gen_res.get("cpu_invocations", 0)

            has_rep, rep_char = detect_character_repetition(raw_text, max_repeat=5)

            r0_writes = [step["val"] for step in trace if step.get("dest") == "R0"]
            cpu_final_val = str(r0_writes[-1]) if r0_writes else None
            cpu_correct = (cpu_final_val == target) if cpu_final_val is not None else False

            final_ans = ""
            is_correct = False

            if task_type == "code_trace":
                if cpu_invocations > 0:
                    swe_telemetry["cpu_invoked_questions"] += 1
                    swe_telemetry["total_cpu_calls"] += cpu_invocations
                if cpu_correct:
                    swe_telemetry["cpu_r0_correct"] += 1

                cleaned = re.sub(r'<[^>]+>(?:\s*=>\s*\[[^\]]+\])?', '', raw_text).strip()
                numbers = re.findall(r'[-+]?\d+', cleaned)
                final_ans = numbers[-1] if numbers else ""
                is_correct = (final_ans == target)
                if is_correct:
                    swe_telemetry["final_ans_correct"] += 1

            elif task_type == "grid_transform":
                m = re.search(r'\[\[.*?\]\]', raw_text)
                final_grid = m.group(0) if m else raw_text
                final_ans = final_grid

                if final_grid.endswith("]]"):
                    arc_telemetry["completed_grids"] += 1
                if has_rep:
                    arc_telemetry["repetitions"] += 1

                is_correct = (final_grid.replace(" ", "") == target.replace(" ", ""))
                if is_correct:
                    arc_telemetry["exact_matches"] += 1
            else:
                final_ans = raw_text.split()[-1] if raw_text.split() else raw_text
                is_correct = (final_ans == target)

            pillar_stats[pillar]["total"] += 1
            if is_correct: pillar_stats[pillar]["correct"] += 1
            category_stats[cat]["total"] += 1
            if is_correct: category_stats[cat]["correct"] += 1
            subject_stats[subj]["total"] += 1
            if is_correct: subject_stats[subj]["correct"] += 1

            audit_records.append({
                "id": item["id"],
                "pillar": pillar,
                "category": cat,
                "subject": subj,
                "target": target,
                "final_answer": final_ans,
                "raw_output": raw_text,
                "is_correct": is_correct,
                "cpu_invocations": cpu_invocations,
                "cpu_final_val": cpu_final_val,
                "cpu_correct": cpu_correct,
                "has_repetition": has_rep
            })

            if (i + 1) % 100 == 0 or (i + 1) == len(gen_questions):
                print(f"  Frontier: {i+1:3d}/{len(gen_questions)} ({100*(i+1)/len(gen_questions):5.1f}%) | "
                      f"Total Elapsed: {time.time() - t0:4.1f}s", flush=True)

    total_elapsed = time.time() - t0
    total_q = len(questions)
    total_c = sum(p["correct"] for p in pillar_stats.values())
    total_p, total_lo, total_hi = wilson_score_interval(total_c, total_q)

    out_log = Path(output_audit_path) if output_audit_path else (ROOT / f"benchmark_8100_audit_arm_{arm}.jsonl")
    with open(out_log, "w", encoding="utf-8") as f:
        for r in audit_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n" + "=" * 90, flush=True)
    print("CANONICAL 8,100-QUESTION FULL BENCHMARK EVALUATION REPORT (STRICT SCORING)", flush=True)
    print("=" * 90, flush=True)
    print(f"Arm: {arm.upper()} | Evaluated: {total_q} | Elapsed: {total_elapsed:.1f}s ({total_q/max(1, total_elapsed):.1f} q/s)", flush=True)
    print(f"Overall Accuracy: {total_p*100:5.2f}%  (95% CI: [{total_lo*100:5.2f}%, {total_hi*100:5.2f}%])", flush=True)
    print("-" * 90, flush=True)
    print(f"{'Major Pillar':<30} | {'Questions':>9} | {'Correct':>7} | {'Accuracy':>9} | {'95% CI':>16}", flush=True)
    print("-" * 90, flush=True)
    for p_name in sorted(pillar_stats.keys()):
        p_d = pillar_stats[p_name]
        p_acc, p_l, p_h = wilson_score_interval(p_d["correct"], p_d["total"])
        print(f"{p_name:<30} | {p_d['total']:9d} | {p_d['correct']:7d} | {p_acc*100:8.2f}% | [{p_l*100:5.1f}%, {p_h*100:5.1f}%]", flush=True)
    print("-" * 90, flush=True)

    print("\n[SWE Software & Program State Tracing Telemetry]", flush=True)
    print("-" * 90, flush=True)
    print(f"  Total SWE Questions Evaluated:         {pillar_stats['SWE_Software']['total']}", flush=True)
    print(f"  Autonomous CPU Invocations Triggered:  {swe_telemetry['cpu_invoked_questions']} / {pillar_stats['SWE_Software']['total']} questions ({swe_telemetry['total_cpu_calls']} total calls)", flush=True)
    print(f"  CPU Register R0 Correct Result:        {swe_telemetry['cpu_r0_correct']} / {pillar_stats['SWE_Software']['total']} ({swe_telemetry['cpu_r0_correct']/max(1, pillar_stats['SWE_Software']['total'])*100:.1f}%)", flush=True)
    print(f"  Final Answer Strict Match:             {swe_telemetry['final_ans_correct']} / {pillar_stats['SWE_Software']['total']} ({swe_telemetry['final_ans_correct']/max(1, pillar_stats['SWE_Software']['total'])*100:.1f}%)", flush=True)

    print("\n[ARC Abstract Grid Reasoning Telemetry]", flush=True)
    print("-" * 90, flush=True)
    print(f"  Total ARC Questions Evaluated:         {pillar_stats['ARC_Reasoning']['total']}", flush=True)
    print(f"  Completed Grids (Closed with ']]'):    {arc_telemetry['completed_grids']} / {pillar_stats['ARC_Reasoning']['total']} ({arc_telemetry['completed_grids']/max(1, pillar_stats['ARC_Reasoning']['total'])*100:.1f}%)", flush=True)
    print(f"  Character Repetition Loops Detected:   {arc_telemetry['repetitions']} / {pillar_stats['ARC_Reasoning']['total']}", flush=True)
    print(f"  Exact Full 2D Grid Matches:            {arc_telemetry['exact_matches']} / {pillar_stats['ARC_Reasoning']['total']} ({arc_telemetry['exact_matches']/max(1, pillar_stats['ARC_Reasoning']['total'])*100:.2f}%)", flush=True)

    print("\n[Granular Subject Breakdown across All 56 JMMLU Subjects]", flush=True)
    print("-" * 90, flush=True)
    print(f"{'Subject':<35} | {'Questions':>9} | {'Correct':>7} | {'Accuracy':>9}", flush=True)
    print("-" * 90, flush=True)
    for s_name in sorted(subject_stats.keys()):
        if s_name.startswith("arc_") or s_name.startswith("swe_"):
            continue
        s_d = subject_stats[s_name]
        s_acc = s_d["correct"] / max(1, s_d["total"])
        print(f"{s_name:<35} | {s_d['total']:9d} | {s_d['correct']:7d} | {s_acc*100:8.2f}%", flush=True)
    print("=" * 90, flush=True)
    print(f"Full 8,100 audit log written to: {out_log}", flush=True)
    print("=" * 90 + "\n", flush=True)

    return {
        "arm": arm,
        "total_questions": total_q,
        "total_correct": total_c,
        "accuracy": total_p,
        "ci_lower": total_lo,
        "ci_upper": total_hi,
        "pillar_stats": {k: dict(v) for k, v in pillar_stats.items()},
        "subject_stats": {k: dict(v) for k, v in subject_stats.items()},
        "swe_telemetry": dict(swe_telemetry),
        "arc_telemetry": dict(arc_telemetry),
        "audit_path": str(out_log),
        "elapsed_seconds": total_elapsed
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--arm", type=str, default="learned", choices=["learned", "dense_all_boundary"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", "--batch-size", type=int, default=64)
    parser.add_argument("--out-log", "--output-audit-path", type=str, default=None)
    args = parser.parse_args()

    run_evaluation(
        checkpoint_path=args.checkpoint,
        manifest_path=args.manifest,
        arm=args.arm,
        device=args.device,
        batch_size=args.batch_size,
        output_audit_path=args.out_log
    )
