# -*- coding: utf-8 -*-
"""v44fix training and read-only eight-tier evaluation in an isolated source overlay.

The training forward, losses, optimizer, and sampler are inherited from the
v44fix TIR-consistency candidate. Tier 8 adds four original v44 prompts and
four content-matched v43-format prompts, reported as separate profiles.
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
from integrated_model_v44fix import IntegratedModelV44Fix
from tier8_readonly_v44fix import evaluate_model as evaluate_tier8_model
from local_search_engine_v42 import LocalSearchEngineV42

import eval_step_control_heldout
import eval_v38_decomposed_state_bench as ev38
import eval_jamcqa_official
import eval_benchmark_full_8100
import eval_logical_instruction_operators
import eval_v42_search_retrieval
import eval_v43_ab_benchmark

from runtime_cotraining_v41_247M import (
    now_iso, compute_cut_ce_loss,
    DynamicTokenBucketBatchSampler, evaluate_tier2_decomposed_suite
)
from tir_masking_v44fix import ToolMaskedMemoryDataset, tool_aware_collate_fn
from resume_v44fix import (
    PARENT_BYTES, PARENT_SHA256, PARENT_STEP, ResumableTokenBucketBatchSampler,
    load_checkpoint, restore_rng, save_checkpoint, sha256_file, training_binding,
    verify_checkpoint,
)

def run_tier8_probe(model, vocab, inv_vocab, args, out_dir, checkpoint, step):
    """Run the two distinct read-only Tier 8 profiles and save their raw output."""
    model.eval()
    original = evaluate_tier8_model(model, vocab, inv_vocab,
                                    max_tokens=args.tier8_max_tokens, profile="original_v44")
    aligned = evaluate_tier8_model(model, vocab, inv_vocab,
                                   max_tokens=args.tier8_max_tokens, profile="aligned_v43")
    report = {"schema": "geocpu_v44fix_tier8_readonly_v1", "checkpoint": str(checkpoint),
              "checkpoint_sha256": sha256_file(checkpoint), "step": step,
              "max_tokens": args.tier8_max_tokens,
              "original_v44": original, "aligned_v43": aligned}
    path = Path(args.tier8_output) if args.tier8_output else out_dir / f"tier8_step_{step}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(f"  Tier 8 raw result: {path}")
    return original, aligned

def main():
    parser = argparse.ArgumentParser(description="GeoCPU v44fix training and eight-tier evaluation")
    parser.add_argument("--train_path", type=str, default=str(ROOT / "data" / "train_v43_knowledge_unified.jsonl"))
    parser.add_argument("--meta_path", type=str, default=str(ROOT / "base_metadata.json"))
    parser.add_argument("--warmstart_path", type=str, default=str(ROOT / "checkpoints" / "checkpoint_step_5000.pt"))
    parser.add_argument("--checkpoint", type=str, default=None, help="Target checkpoint for evaluation or saving")
    parser.add_argument("--output_dir", type=str, default=str(ROOT / "checkpoints"))
    parser.add_argument("--max_steps", type=int, default=15000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3.5e-5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval_only", action="store_true", help="Run full 8-tier benchmark suite directly")
    parser.add_argument("--tier8_only", action="store_true",
                        help="Read-only Tier 8 probe without running Tiers 1-7; implies --eval_only")
    parser.add_argument("--stop_at_step", type=int, default=0, help="v44fix: stop early (gate) while the LR schedule still targets --max_steps")
    parser.add_argument("--skip_final_eval", action="store_true", help="v44fix: skip the 8-tier suite after training")
    parser.add_argument("--max_tokens_per_micro", type=int, default=0, help="v44fix: override token budget (CPU smoke tests)")
    parser.add_argument("--tier8_max_tokens", type=int, default=2048,
                        help="Maximum native generated tokens per Tier 8 question; use 128 to reproduce v44's old cap")
    parser.add_argument("--tier8_output", type=str, default=None,
                        help="New JSON result path; defaults to output_dir/tier8_step_STEP.json and never overwrites")
    parser.add_argument("--require_exact_resume", action="store_true",
                        help="Require a verified v44fix checkpoint with optimizer, RNG, and sampler cursor")
    args = parser.parse_args()
    if args.tier8_only:
        if args.skip_final_eval:
            parser.error("--tier8_only cannot be combined with --skip_final_eval")
        args.eval_only = True

    print("=" * 95)
    print(f"[{now_iso()}] V43 SCALED 247.5M KNOWLEDGE-RELATIONAL & SEARCH CO-TRAINING ENGINE")
    print("=" * 95)
    print(f"  Mode:            {'EVALUATION ONLY (8-Tier Benchmarks)' if args.eval_only else 'CO-TRAINING + 8-TIER BENCHMARKS'}")
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
    if not corpus_v43.is_file():
        raise FileNotFoundError("Pinned v43 search corpus is missing: " + str(corpus_v43))
    search_engine = LocalSearchEngineV42(corpus_path=corpus_v43)
    # V44Fix adds only the Python generation method; inherited forward/parameters
    # remain those of V42, so the training computation and checkpoint keys are unchanged.
    model = IntegratedModelV44Fix(base=base_model, search_engine=search_engine)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model Parameters: {total_params:,} (247.56M Architecture Confirmed)")

    if not target_ckpt or not Path(target_ckpt).is_file():
        raise FileNotFoundError(f"Required checkpoint is missing: {target_ckpt}")
    start_step = 0
    if not args.require_exact_resume:
        parent = Path(target_ckpt)
        if parent.stat().st_size != PARENT_BYTES or sha256_file(parent) != PARENT_SHA256:
            raise ValueError("Warmstart is not the pinned v43 step-15000 checkpoint")
        ckpt = torch.load(parent, map_location="cpu", weights_only=True)
        if not isinstance(ckpt, dict) or ckpt.get("step") != PARENT_STEP or "model_state_dict" not in ckpt:
            raise ValueError("Pinned v43 checkpoint lacks the expected step/model state")
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        start_step = PARENT_STEP
        print(f"  -> Strict pinned v43 warm start at step {start_step}; optimizer starts a new v44fix run.")
    elif args.eval_only:
        verify_checkpoint(target_ckpt)
        ckpt = torch.load(target_ckpt, map_location="cpu", weights_only=True)
        if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
            raise ValueError("Exact evaluation checkpoint has no model state")
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        start_step = int(ckpt["step"])

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

        dataset = ToolMaskedMemoryDataset(args.train_path, vocab, max_len=cfg.max_positions)
        batch_sampler = ResumableTokenBucketBatchSampler(
            dataset,
            max_batch_size=max_micro_batch,
            max_tokens=max_tokens_per_micro,
            megabucket_mult=32
        )
        dataloader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=tool_aware_collate_fn,
            pin_memory=(dev.type == "cuda")
        )

        use_fused = (dev.type == "cuda")
        try:
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01, fused=use_fused)
        except Exception:
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)

        binding = training_binding(args.train_path, args.meta_path, ROOT,
            max_steps=args.max_steps, batch_size=args.batch_size, lr=args.lr,
            max_tokens_per_micro=max_tokens_per_micro, grad_accum_steps=grad_accum_steps)
        pending_rng = None
        if args.require_exact_resume:
            start_step, pending_rng = load_checkpoint(target_ckpt, expected_binding=binding,
                model=model, optimizer=optimizer, sampler=batch_sampler)
            if not PARENT_STEP < start_step <= args.max_steps:
                raise ValueError("Exact resume step is outside the declared training run")
            print(f"  -> Strict v44fix resume at step {start_step}, with Adam/RNG/batch cursor restored.")

        def get_lr(st: int) -> float:
            warmup = 300
            if st < warmup:
                return args.lr * (st + 1) / warmup
            prog = (st - warmup) / max(1, args.max_steps - warmup)
            return max(8e-6, args.lr * 0.5 * (1.0 + math.cos(math.pi * prog)))

        def run_microbatch_safe(inp: torch.Tensor, tgt: torch.Tensor, vmask: torch.Tensor, lmask: torch.Tensor, weight: float) -> torch.Tensor:
            try:
                with torch.amp.autocast(device_type=dev.type, enabled=use_amp, dtype=torch.bfloat16):
                    out = model(inp, vmask, arm="learned", skip_head=True)
                    ce_loss = compute_cut_ce_loss(model, out.fine_states, tgt, lmask)
                    ratio_loss = (out.compression_ratio - 1.0).pow(2).mean()
                    t_loss = (ce_loss + 0.2 * ratio_loss) * weight
                t_loss.backward()
                return t_loss.detach()
            except torch.OutOfMemoryError as e:
                # A failed backward can leave partial gradients. Never retry
                # the same batch against those gradients or accept the step.
                raise RuntimeError("OOM: entire update rejected; no checkpoint advanced") from e

        print(f"\nStarting Knowledge Co-Training from step {start_step} to {args.max_steps}...")
        step = start_step
        running_loss = 0.0
        log_interval = 50
        checkpoint_interval = 2500
        t_start = time.time()

        model.train()
        data_iter = iter(dataloader)
        if pending_rng is not None:
            restore_rng(pending_rng)  # DataLoader iterator setup may consume CPU RNG.

        def save_step(accepted_step: int) -> str:
            path = out_dir / f"checkpoint_step_{accepted_step}.pt"
            receipt = save_checkpoint(path, step=accepted_step, model=model,
                optimizer=optimizer, sampler=batch_sampler, binding=binding,
                config=cfg_dict, meta=meta)
            print(f"  --> Verified checkpoint {path.name} SHA256={receipt['sha256']}")
            return str(path)

        while step < args.max_steps:
            step += 1
            lr = get_lr(step)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            accum_loss = torch.zeros((), dtype=torch.float32, device=dev)

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
                if not bool(batch["loss_mask"].any()):
                    raise ValueError("Batch has no supervised target positions")

                loss_val = run_microbatch_safe(inp, tgt, vmask, lmask, 1.0 / grad_accum_steps)
                accum_loss = accum_loss + loss_val

            if not bool(torch.isfinite(accum_loss)):
                raise FloatingPointError("Non-finite loss: update rejected")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()

            running_loss += accum_loss

            if step % log_interval == 0:
                avg_loss = float((running_loss / log_interval).item())
                running_loss = torch.zeros((), dtype=torch.float32, device=dev)
                elapsed = time.time() - t_start
                steps_per_sec = (step - start_step) / max(1.0, elapsed)
                eta_sec = (args.max_steps - step) / max(0.01, steps_per_sec)
                print(f"  Step {step:6d}/{args.max_steps} | Loss: {avg_loss:.4f} | LR: {lr:.2e} | "
                      f"Throughput: {steps_per_sec:.2f} step/s | ETA: {eta_sec/60:.1f}m", flush=True)

            if step % checkpoint_interval == 0 or step == args.max_steps:
                final_ckpt = save_step(step)

            if args.stop_at_step and step >= args.stop_at_step:
                if not (step % checkpoint_interval == 0 or step == args.max_steps):
                    final_ckpt = save_step(step)
                print(f"  --> stop_at_step={args.stop_at_step} reached (LR schedule targets max_steps={args.max_steps}).", flush=True)
                break

    if args.skip_final_eval:
        print(f"  skip_final_eval: last checkpoint {final_ckpt}; 8-tier suite skipped.", flush=True)
        return

    if args.tier8_only:
        print("\n[TIER 8 ONLY] Original v44 and aligned v43 prompts...")
        run_tier8_probe(model, vocab, inv_vocab, args, out_dir, final_ckpt, start_step)
        return

    # ==================================================================================
    # EXECUTING TIERS 1-7 PLUS THE DISTINCT FOUR-PROMPT TIER 8 PROFILES
    # ==================================================================================
    required_evaluation_inputs = (
        "eval_swe_step_control_heldout.jsonl", "eval_v38_decomposed_state_bench200.jsonl",
        "jamcqa_manifest_test.jsonl", "benchmark_8100_manifest.jsonl",
        "eval_v41_logical_instruction_operators_heldout.jsonl",
        "eval_v42_search_retrieval_aligned.jsonl", "knowledge_corpus_v43.jsonl")
    missing_evaluation_inputs = [name for name in required_evaluation_inputs if not (ROOT / "data" / name).is_file()]
    if missing_evaluation_inputs:
        raise FileNotFoundError("Seven-tier evaluation inputs are missing: " + repr(missing_evaluation_inputs))
    print("\n" + "=" * 95)
    print(f"[{now_iso()}] EXECUTING TIERS 1-8 (Tier 8 has two separate four-prompt profiles)")
    print("=" * 95)
    model.eval()

    # Tier 1: Held-Out SWE Step Control (200 items)
    t1_manifest = ROOT / "data" / "eval_swe_step_control_heldout.jsonl"
    print("\n[TIER 1/8] Held-Out SWE Step-Control Benchmark (200 Items)...")
    t1_stats = eval_step_control_heldout.run_heldout_evaluation(
        checkpoint_path=final_ckpt, manifest_path=str(t1_manifest), arm="learned",
        device=args.device, model=model, vocab=vocab, inv_vocab=inv_vocab
    )

    # Tier 2: Decomposed State Tracking (200 items)
    t2_manifest = ROOT / "data" / "eval_v38_decomposed_state_bench200.jsonl"
    print("\n[TIER 2/8] Decomposed State Tracking Benchmark (200 Items)...")
    t2_res = evaluate_tier2_decomposed_suite(model, t2_manifest, vocab, inv_vocab, dev)

    # Tier 3: SoftBank JamC-QA (2,309 items)
    jamc_manifest = str(ROOT / "data" / "jamcqa_manifest_test.jsonl")
    jamc_res = {}
    if os.path.exists(jamc_manifest):
        print(f"\n[TIER 3/8] SoftBank JamC-QA Full Benchmark (2,309 items)...")
        jamc_res = eval_jamcqa_official.run_jamcqa_official_eval(
            checkpoint_path=final_ckpt, manifest_path=jamc_manifest, mode="both",
            device=args.device, batch_size=64, max_gen_samples=100,
            model=model, vocab=vocab, inv_vocab=inv_vocab
        )

    # Tier 4: Canonical Broad 8,100 (7,536 JMMLU + 164 SWE + 400 ARC)
    bench_8100_manifest = str(ROOT / "data" / "benchmark_8100_manifest.jsonl")
    bench_8100_res = {}
    if os.path.exists(bench_8100_manifest):
        print(f"\n[TIER 4/8] Canonical Broad 8,100 Full Benchmark (7,536 JMMLU + 164 SWE + 400 ARC)...")
        bench_8100_res = eval_benchmark_full_8100.run_evaluation(
            checkpoint_path=final_ckpt, manifest_path=bench_8100_manifest, arm="learned",
            device=args.device, batch_size=64, model=model, vocab=vocab, inv_vocab=inv_vocab
        )

    # Tier 5: 6-Operator Logical Instruction Contrast (400 items)
    t5_manifest = str(ROOT / "data" / "eval_v41_logical_instruction_operators_heldout.jsonl")
    t5_res = {}
    if os.path.exists(t5_manifest):
        print(f"\n[TIER 5/8] 6-Operator Logical Instruction Contrast Benchmark (400 items)...")
        t5_res = eval_logical_instruction_operators.run_logical_operator_evaluation(
            checkpoint_path=final_ckpt, manifest_path=t5_manifest, device=args.device,
            model=model, vocab=vocab, inv_vocab=inv_vocab
        )

    # Tier 6: Search-Augmented Multi-Hop Retrieval (200 items)
    t6_manifest = str(ROOT / "data" / "eval_v42_search_retrieval_aligned.jsonl")
    t6_res = {}
    if os.path.exists(t6_manifest):
        print(f"\n[TIER 6/8] Search-Augmented Multi-Hop Retrieval Benchmark (200 items)...")
        t6_res = eval_v42_search_retrieval.run_tier6_search_evaluation(
            checkpoint_path=final_ckpt, manifest_path=t6_manifest, device=args.device,
            model=model, vocab=vocab, inv_vocab=inv_vocab
        )

    # Tier 7: Formal A/B Benchmark: Closed-Book (学習のみ) vs Open-Book (学習＋検索あり)
    print("\n[TIER 7/8] Formal A/B Dual Benchmark: Closed-Book (学習のみ) vs Open-Book (学習＋検索あり)...")
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
    if not all(isinstance(value, dict) and value for value in
               (t1_stats, t2_res, jamc_res, bench_8100_res, t5_res, t6_res, ab_res)):
        raise RuntimeError("Seven-tier evaluation returned an empty or incomplete tier")

    # Tier 8 is a read-only formatting/tool-use probe. Keep the original v44
    # prompts and content-matched v43 prompts separate to expose prompt effects.
    print("\n[TIER 8/8] Python/C++/Rust and explanation probe (original v44 + aligned v43)...")
    eval_step = start_step if args.eval_only else step
    t8_original, t8_aligned = run_tier8_probe(
        model, vocab, inv_vocab, args, out_dir, final_ckpt, eval_step)

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
    for label, result in (("original v44", t8_original), ("aligned v43", t8_aligned)):
        print(f"Tier 8 {label}: code {result['tir_code_rate']:.2f}%, "
              f"narrative {result['narrative_rate']:.2f}%, "
              f"expected-language fence {result['expected_language_fence_rate']:.2f}%")
    print("=" * 125)

if __name__ == "__main__":
    main()
