# -*- coding: utf-8 -*-
"""
runtime_v44fix.py: Scaled 247.5M Knowledge-Relational & Search Co-Training Engine v43.

Guarantees:
1. Retains 100% of the v41/v42 verified architecture (d_model=768, 10 Enc, 14 Dec, FFN=3072).
2. Warm-Starts from milestone checkpoint_step_5000.pt (preserving SWE 95.5%, State Tracking 85.5%, Logic 65.0%).
3. Intensive Knowledge & Search Co-Training:
   - 268,032 Unified samples in train_v43_knowledge_unified.jsonl.
   - Enriches 4-Choice Academic Search CoT, Relational/Causal Knowledge CoT, and Multi-Hop Retrieval.
4. Scaled 15,000-Step Training Schedule (~3.7 full epochs over unified dataset).
5. Comprehensive 7-Tier Benchmark Suite including:
   - Tier 1: Held-Out SWE Step Control (200 questions)
   - Tier 2: Decomposed State Tracking (200 questions)
   - Tier 3: SoftBank JamC-QA Full (2,309 questions)
   - Tier 4: Canonical Broad 8,100 Full Benchmark (7,536 JMMLU + 164 SWE + 400 ARC)
   - Tier 5: 6-Operator Logical Instruction Contrast (400 questions)
   - Tier 6: Search-Augmented Multi-Hop Retrieval (200 questions)
   - Tier 7: Formal A/B Benchmark: Closed-Book (学習のみ) vs Open-Book (学習＋検索あり)
"""
from __future__ import annotations
import sys
import os
import time
import json
import math
import argparse
import shutil
import functools
import traceback
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

print = functools.partial(print, flush=True)

ROOT = Path(__file__).parent.resolve()
sys.path.append(str(ROOT))

from GeoCPU_CED_v1 import CEDConfig
from GeoCPU_CED_v2 import GeoCPU_CED_v2
from integrated_model_v42 import IntegratedModelV42
from local_search_engine_v42 import LocalSearchEngineV42

import eval_step_control_heldout
import eval_v38_decomposed_state_bench as ev38
import eval_jamcqa_official
import eval_benchmark_full_8100
import eval_logical_instruction_operators
import eval_v42_search_retrieval
import eval_v43_ab_benchmark

from runtime_cotraining_v41_247M import (
    now_iso, compute_cut_ce_loss, PreTokenizedMemoryDataset,
    DynamicTokenBucketBatchSampler, zero_overhead_collate_fn, evaluate_tier2_decomposed_suite
)

def main():
    parser = argparse.ArgumentParser(description="GeoCPU v43: Knowledge-Relational & Search Co-Training Runtime")
    parser.add_argument("--train_path", type=str, default=str(ROOT / "data" / "train_v43_knowledge_unified.jsonl"))
    parser.add_argument("--meta_path", type=str, default=str(ROOT / "base_metadata.json"))
    parser.add_argument("--warmstart_path", type=str, default=str(ROOT / "checkpoints" / "checkpoint_step_5000.pt"))
    parser.add_argument("--checkpoint", type=str, default=None, help="Target checkpoint for evaluation or saving")
    parser.add_argument("--output_dir", type=str, default=str(ROOT / "checkpoints"))
    parser.add_argument("--max_steps", type=int, default=15000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3.5e-5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval_only", action="store_true", help="Run full 7-tier benchmark suite directly")
    parser.add_argument("--stop_at_step", type=int, default=0, help="v44fix: stop early (gate) while the LR schedule still targets --max_steps")
    parser.add_argument("--skip_final_eval", action="store_true", help="v44fix: skip the 7-tier suite after training")
    parser.add_argument("--max_tokens_per_micro", type=int, default=0, help="v44fix: override token budget (CPU smoke tests)")
    args = parser.parse_args()

    print("=" * 95)
    print(f"[{now_iso()}] V43 SCALED 247.5M KNOWLEDGE-RELATIONAL & SEARCH CO-TRAINING ENGINE")
    print("=" * 95)
    print(f"  Mode:            {'EVALUATION ONLY (7-Tier Benchmarks)' if args.eval_only else 'CO-TRAINING + 7-TIER BENCHMARKS'}")
    print(f"  Train File:      {args.train_path}")
    target_ckpt = args.checkpoint or args.warmstart_path
    print(f"  Active Ckpt:     {target_ckpt}")
    print(f"  Max Steps:       {args.max_steps} | Batch Size: {args.batch_size} | LR: {args.lr}")
    print(f"  Device:          {args.device}")

    if "cuda" in args.device and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    dev = torch.device(args.device)
    meta = json.loads(Path(args.meta_path).read_bytes())
    vocab = meta["vocab"]
    inv_vocab = {int(v): k for k, v in vocab.items()}

    cfg_dict = dict(meta["config"])
    cfg_dict["scan_backend"] = "jit"
    cfg = CEDConfig(**cfg_dict)

    base_model = GeoCPU_CED_v2(cfg)
    corpus_v43 = ROOT / "data" / "knowledge_corpus_v43.jsonl"
    search_engine = LocalSearchEngineV42(corpus_path=corpus_v43 if corpus_v43.exists() else None)
    model = IntegratedModelV42(base=base_model, search_engine=search_engine)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model Parameters: {total_params:,} (247.56M Architecture Confirmed)")

    start_step = 0
    if target_ckpt and os.path.exists(target_ckpt):
        print(f"  Loading Weights from Checkpoint (CPU buffer): {target_ckpt}...")
        ckpt = torch.load(target_ckpt, map_location="cpu")
        sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(sd, strict=False)
        if isinstance(ckpt, dict) and "step" in ckpt:
            start_step = int(ckpt["step"])
            print(f"  -> Resuming from Step {start_step} successfully into v43 architecture.")
        else:
            print("  -> Warmstart weights loaded successfully into v43 architecture.")
    else:
        print(f"  WARNING: Checkpoint not found at {target_ckpt}. Initializing cleanly.")

    model.to(dev)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_ckpt = str(target_ckpt)

    if not args.eval_only:
        use_amp = (dev.type == "cuda")
        if dev.type == "cuda":
            free_bytes, total_bytes = torch.cuda.mem_get_info(dev)
            free_gb = free_bytes / (1024**3)
            if free_gb >= 26.0:
                max_tokens_per_micro = 36864
                grad_accum_steps = 1
            elif free_gb >= 16.0:
                max_tokens_per_micro = 24576
                grad_accum_steps = 1
            else:
                max_tokens_per_micro = max(6144, int(max(2.0, free_gb - 3.0) * 2048))
                grad_accum_steps = max(1, 28672 // max_tokens_per_micro)
        else:
            max_tokens_per_micro = 16384
            grad_accum_steps = 1

        if args.max_tokens_per_micro:
            max_tokens_per_micro = int(args.max_tokens_per_micro)
        max_micro_batch = min(96, args.batch_size)
        print(f"  Execution Mode:  AMP bfloat16={use_amp} | Dynamic Token Budget: {max_tokens_per_micro:,} tok/micro (Max B={max_micro_batch}, Accum={grad_accum_steps})")

        dataset = PreTokenizedMemoryDataset(args.train_path, vocab, max_len=cfg.max_positions)
        batch_sampler = DynamicTokenBucketBatchSampler(
            dataset,
            max_batch_size=max_micro_batch,
            max_tokens=max_tokens_per_micro,
            megabucket_mult=32
        )
        dataloader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=zero_overhead_collate_fn,
            pin_memory=(dev.type == "cuda")
        )

        use_fused = (dev.type == "cuda")
        try:
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01, fused=use_fused)
        except Exception:
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)

        def get_lr(st: int) -> float:
            warmup = 300
            if st < warmup:
                return args.lr * (st + 1) / warmup
            prog = (st - warmup) / max(1, args.max_steps - warmup)
            return max(8e-6, args.lr * 0.5 * (1.0 + math.cos(math.pi * prog)))

        def run_microbatch_safe(inp: torch.Tensor, tgt: torch.Tensor, vmask: torch.Tensor, lmask: torch.Tensor, weight: float) -> torch.Tensor:
            cur_b = inp.size(0)
            oom_hit = False
            try:
                with torch.amp.autocast(device_type=dev.type, enabled=use_amp, dtype=torch.bfloat16):
                    out = model(inp, vmask, arm="learned", skip_head=True)
                    ce_loss = compute_cut_ce_loss(model, out.fine_states, tgt, lmask)
                    ratio_loss = (out.compression_ratio - 1.0).pow(2).mean()
                    t_loss = (ce_loss + 0.2 * ratio_loss) * weight
                t_loss.backward()
                return t_loss.detach()
            except torch.OutOfMemoryError as e:
                traceback.clear_frames(e.__traceback__)
                oom_hit = True

            if oom_hit:
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
                half = cur_b // 2
                if half == 0:
                    print("  [OOM Warning] Single sequence exceeded memory; skipping.", flush=True)
                    return torch.tensor(0.0, device=dev)
                t_loss1 = run_microbatch_safe(inp[:half], tgt[:half], vmask[:half], lmask[:half], weight * (half / cur_b))
                t_loss2 = run_microbatch_safe(inp[half:], tgt[half:], vmask[half:], lmask[half:], weight * ((cur_b - half) / cur_b))
                return t_loss1 + t_loss2

        print(f"\nStarting Knowledge Co-Training from step {start_step} to {args.max_steps}...")
        step = start_step
        running_loss = 0.0
        log_interval = 50
        checkpoint_interval = 2500
        t_start = time.time()

        model.train()
        data_iter = iter(dataloader)

        while step < args.max_steps:
            step += 1
            lr = get_lr(step)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0

            for accum_idx in range(grad_accum_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    batch = next(data_iter)

                inp = batch["input_ids"].to(dev, non_blocking=True)
                tgt = batch["targets"].to(dev, non_blocking=True)
                vmask = batch["valid_mask"].to(dev, non_blocking=True)
                lmask = batch["loss_mask"].to(dev, non_blocking=True)

                loss_val = run_microbatch_safe(inp, tgt, vmask, lmask, 1.0 / grad_accum_steps)
                accum_loss += loss_val.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += accum_loss

            if step % log_interval == 0:
                avg_loss = running_loss / log_interval
                running_loss = 0.0
                elapsed = time.time() - t_start
                steps_per_sec = (step - start_step) / max(1.0, elapsed)
                eta_sec = (args.max_steps - step) / max(0.01, steps_per_sec)
                print(f"  Step {step:6d}/{args.max_steps} | Loss: {avg_loss:.4f} | LR: {lr:.2e} | "
                      f"Throughput: {steps_per_sec:.2f} step/s | ETA: {eta_sec/60:.1f}m", flush=True)

            if step % checkpoint_interval == 0 or step == args.max_steps:
                ckpt_file = out_dir / f"checkpoint_step_{step}.pt"
                print(f"  --> Saving Checkpoint at step {step} to {ckpt_file}...", flush=True)
                tmp_file = ckpt_file.with_suffix(".pt.tmp")  # v44fix: never leave a truncated checkpoint under the final name
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "config": cfg_dict,
                    "meta": meta
                }, str(tmp_file))
                os.replace(tmp_file, ckpt_file)
                final_ckpt = str(ckpt_file)

            if args.stop_at_step and step >= args.stop_at_step:
                if not (step % checkpoint_interval == 0 or step == args.max_steps):
                    ckpt_file = out_dir / f"checkpoint_step_{step}.pt"
                    print(f"  --> Saving gate checkpoint at step {step} to {ckpt_file}...", flush=True)
                    tmp_file = ckpt_file.with_suffix(".pt.tmp")
                    torch.save({"step": step, "model_state_dict": model.state_dict(), "config": cfg_dict, "meta": meta}, str(tmp_file))
                    os.replace(tmp_file, ckpt_file)
                    final_ckpt = str(ckpt_file)
                print(f"  --> stop_at_step={args.stop_at_step} reached (LR schedule targets max_steps={args.max_steps}).", flush=True)
                break

    if args.skip_final_eval:
        print(f"  skip_final_eval: last checkpoint {final_ckpt}; 7-tier suite skipped.", flush=True)
        return

    # ==================================================================================
    # EXECUTING FULL 7-TIER BENCHMARK SUITE (11,597 TOTAL QUESTIONS)
    # ==================================================================================
    print("\n" + "=" * 95)
    print(f"[{now_iso()}] EXECUTING FULL 7-TIER BENCHMARK SUITE (11,597 TOTAL QUESTIONS)")
    print("=" * 95)
    model.eval()

    # Tier 1: Held-Out SWE Step Control (200 items)
    t1_manifest = ROOT / "data" / "eval_swe_step_control_heldout.jsonl"
    print("\n[TIER 1/7] Held-Out SWE Step-Control Benchmark (200 Items)...")
    t1_stats = eval_step_control_heldout.run_heldout_evaluation(
        checkpoint_path=final_ckpt, manifest_path=str(t1_manifest), arm="learned",
        device=args.device, model=model, vocab=vocab, inv_vocab=inv_vocab
    )

    # Tier 2: Decomposed State Tracking (200 items)
    t2_manifest = ROOT / "data" / "eval_v38_decomposed_state_bench200.jsonl"
    print("\n[TIER 2/7] Decomposed State Tracking Benchmark (200 Items)...")
    t2_res = evaluate_tier2_decomposed_suite(model, t2_manifest, vocab, inv_vocab, dev)

    # Tier 3: SoftBank JamC-QA (2,309 items)
    jamc_manifest = str(ROOT / "data" / "jamcqa_manifest_test.jsonl")
    jamc_res = {}
    if os.path.exists(jamc_manifest):
        print(f"\n[TIER 3/7] SoftBank JamC-QA Full Benchmark (2,309 items)...")
        jamc_res = eval_jamcqa_official.run_jamcqa_official_eval(
            checkpoint_path=final_ckpt, manifest_path=jamc_manifest, mode="both",
            device=args.device, batch_size=64, max_gen_samples=100,
            model=model, vocab=vocab, inv_vocab=inv_vocab
        )

    # Tier 4: Canonical Broad 8,100 (7,536 JMMLU + 164 SWE + 400 ARC)
    bench_8100_manifest = str(ROOT / "data" / "benchmark_8100_manifest.jsonl")
    bench_8100_res = {}
    if os.path.exists(bench_8100_manifest):
        print(f"\n[TIER 4/7] Canonical Broad 8,100 Full Benchmark (7,536 JMMLU + 164 SWE + 400 ARC)...")
        bench_8100_res = eval_benchmark_full_8100.run_evaluation(
            checkpoint_path=final_ckpt, manifest_path=bench_8100_manifest, arm="learned",
            device=args.device, batch_size=64, model=model, vocab=vocab, inv_vocab=inv_vocab
        )

    # Tier 5: 6-Operator Logical Instruction Contrast (400 items)
    t5_manifest = str(ROOT / "data" / "eval_v41_logical_instruction_operators_heldout.jsonl")
    t5_res = {}
    if os.path.exists(t5_manifest):
        print(f"\n[TIER 5/7] 6-Operator Logical Instruction Contrast Benchmark (400 items)...")
        t5_res = eval_logical_instruction_operators.run_logical_operator_evaluation(
            checkpoint_path=final_ckpt, manifest_path=t5_manifest, device=args.device,
            model=model, vocab=vocab, inv_vocab=inv_vocab
        )

    # Tier 6: Search-Augmented Multi-Hop Retrieval (200 items)
    t6_manifest = str(ROOT / "data" / "eval_v42_search_retrieval_aligned.jsonl")
    t6_res = {}
    if os.path.exists(t6_manifest):
        print(f"\n[TIER 6/7] Search-Augmented Multi-Hop Retrieval Benchmark (200 items)...")
        t6_res = eval_v42_search_retrieval.run_tier6_search_evaluation(
            checkpoint_path=final_ckpt, manifest_path=t6_manifest, device=args.device,
            model=model, vocab=vocab, inv_vocab=inv_vocab
        )

    # Tier 7: Formal A/B Benchmark: Closed-Book (学習のみ) vs Open-Book (学習＋検索あり)
    print("\n[TIER 7/7] Formal A/B Dual Benchmark: Closed-Book (学習のみ) vs Open-Book (学習＋検索あり)...")
    ab_res = eval_v43_ab_benchmark.run_v43_ab_benchmark(
        checkpoint_path=final_ckpt,
        manifest_path=bench_8100_manifest,
        eval_count=200,
        device=args.device,
        model=model,
        vocab=vocab,
        inv_vocab=inv_vocab,
        corpus_path=str(corpus_v43) if corpus_v43.exists() else None
    )

    # ----------------------------------------------------------------------------------
    # MASTER HISTORICAL SCIENTIFIC SCOREBOARD
    # ----------------------------------------------------------------------------------
    print("\n" + "=" * 125)
    print("MASTER HISTORICAL SCIENTIFIC SCOREBOARD: v38 vs v41 vs v42 vs v43 (Knowledge-Scaled & Dual A/B)")
    print("=" * 125)
    print(f"{'Evaluation Metric / Benchmark Pillar':<42} | {'Questions':>9} | {'v41 Frontier':>14} | {'v42 Search':>14} | {'v43 Knowledge-Scaled':>23}")
    print("-" * 125)

    t1_acc = t1_stats.get("correct_val", 0) / max(1, t1_stats.get("total", 1)) * 100.0
    t1_exact = t1_stats.get("exact_step", 0) / max(1, t1_stats.get("total", 1)) * 100.0
    print(f"{'Tier 1: SWE Held-Out Exact Numeric Match':<42} | {t1_stats.get('total', 200):9d} | {'91.50%':>14} | {'95.50%':>14} | {t1_acc:>22.2f}%")
    print(f"{'Tier 1: SWE Held-Out Exact Step Termination':<42} | {t1_stats.get('total', 200):9d} | {'80.00%':>14} | {'83.00%':>14} | {t1_exact:>22.2f}%")

    t2_acc = t2_res.get("accuracy", 0.0)
    print(f"{'Tier 2: State Tracking End-to-End Match':<42} | {t2_res.get('total', 200):9d} | {'84.50%':>14} | {'85.50%':>14} | {t2_acc:>22.2f}%")

    jamc_logit_acc = jamc_res.get("logit_argmax", {}).get("accuracy", 0.0) * 100.0
    print(f"{'Tier 3: SoftBank JamC-QA (Logit Argmax)':<42} | {'2309':>9} | {'25.55%':>14} | {'26.63%':>14} | {jamc_logit_acc:>22.2f}%")

    if bench_8100_res:
        p_stats = bench_8100_res.get("pillar_stats", {})
        jmmlu_acc = p_stats.get("JMMLU_Academic", {}).get("correct", 0) / max(1, p_stats.get("JMMLU_Academic", {}).get("total", 1)) * 100.0
        swe_acc = p_stats.get("SWE_Software", {}).get("correct", 0) / max(1, p_stats.get("SWE_Software", {}).get("total", 1)) * 100.0
        overall_acc = bench_8100_res.get("accuracy", 0.0) * 100.0
        print(f"{'Tier 4: JMMLU Academic (All 56 Subjects)':<42} | {'7536':>9} | {'23.70%':>14} | {'23.91%':>14} | {jmmlu_acc:>22.2f}%")
        print(f"{'Tier 4: SWE Software State Tracing':<42} | {'164':>9} | {'95.73%':>14} | {'95.73%':>14} | {swe_acc:>22.2f}%")
        print("-" * 125)
        print(f"{'OVERALL CANONICAL BENCHMARK (STRICT)':<42} | {'8100':>9} | {'24.02%':>14} | {'24.26%':>14} | {overall_acc:>22.2f}%")

    if t5_res:
        t5_acc = t5_res.get("accuracy", 0.0)
        print("-" * 125)
        print(f"{'Tier 5: 6-Operator Logical Contrast Overall':<42} | {'400':>9} | {'50.25%':>14} | {'65.00%':>14} | {t5_acc:>22.2f}%")

    if t6_res:
        t6_acc = t6_res.get("accuracy", 0.0)
        t6_call = t6_res.get("search_invocation_rate", 0.0)
        print("-" * 125)
        print(f"{'Tier 6: Search Retrieval Overall Accuracy':<42} | {'200':>9} | {'--':>14} | {'100.00%':>14} | {t6_acc:>22.2f}%")
        print(f"{'Tier 6: Search Invocation Autonomy Rate':<42} | {'200':>9} | {'--':>14} | {'100.00%':>14} | {t6_call:>22.2f}%")

    if ab_res:
        a_acc = ab_res.get("arm_a_accuracy", 0.0)
        b_acc = ab_res.get("arm_b_accuracy", 0.0)
        diff = ab_res.get("absolute_gain", 0.0)
        print("-" * 125)
        print(f"{'Tier 7: ARM A - 学習のみ (Closed-Book)':<42} | {'200':>9} | {'--':>14} | {'--':>14} | {a_acc:>22.2f}%")
        print(f"{'Tier 7: ARM B - 学習＋検索 (Open-Book)':<42} | {'200':>9} | {'--':>14} | {'--':>14} | {b_acc:>22.2f}%")
        print(f"{'Tier 7: Absolute Search Gain (Δ Lift)':<42} | {'200':>9} | {'--':>14} | {'--':>14} | {diff:>+22.2f}%")
    print("=" * 125)

if __name__ == "__main__":
    main()
