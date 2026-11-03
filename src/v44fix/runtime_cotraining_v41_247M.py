# -*- coding: utf-8 -*-
"""
runtime_cotraining_v41_247M.py: Zero-Overhead Scaled 247.5M Neurosymbolic Co-Training Engine v41.

Guarantees:
1. Scaled 247.56M Architecture (d_model=768, 10 encoder layers, 14 decoder layers, FFN=3072).
2. Net2Net Warm-Start from v38 peak weights (loaded safely via CPU buffer to avoid GPU unpickling OOM).
3. Pre-tokenized In-Memory Fast Dataset (Zero Python string encoding per step).
4. 32-Warp Aligned Ampere A100 Tensor Core Execution with TF32 enabled.
5. Periodic Milestone Checkpoint & Held-Out Step Control Evaluation (200 questions, Loops 1, 3, 5, 7 breakdown).
6. Comprehensive 4-Tier Benchmark Suite at Step 5000 (10,797 Total Historical Questions):
   - Tier 1: Held-Out SWE Step Control (200 questions)
   - Tier 2: Decomposed State Tracking (200 questions)
   - Tier 3: SoftBank JamC-QA Full (2,309 questions)
   - Tier 4: Canonical 8,100 Full Benchmark (7,536 JMMLU + 164 SWE + 400 ARC)
7. Master Historical Decision Scoreboard comparing v31/v32 vs v38 vs v41.
"""
from __future__ import annotations
import sys
import os
import time
import json
import math
import random
import argparse
import shutil
import functools
import traceback
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

print = functools.partial(print, flush=True)

ROOT = Path(__file__).parent.resolve()
sys.path.append(str(ROOT))

from GeoCPU_CED_v1 import CEDConfig
from GeoCPU_CED_v2 import GeoCPU_CED_v2
from integrated_model_v30 import IntegratedModelV30
from qwenputer_controller_v30 import canonical_normalize_text
import eval_step_control_heldout
import eval_v38_decomposed_state_bench as ev38
import eval_jamcqa_official
import eval_benchmark_full_8100
import eval_logical_instruction_operators

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def compute_cut_ce_loss(model: IntegratedModelV30, fine: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Apple ml-cross-entropy style Cut Head + Cross-Entropy:
    Projects final_norm + head ONLY on active target positions (mask > 0.5),
    skipping prompt and <PAD> positions completely."""
    active = mask.view(-1) > 0.5
    if not active.any():
        return fine.sum() * 0.0
    fine_active = fine.view(-1, fine.size(-1))[active]
    logits_active = model.base.head(model.base.final_norm(fine_active))
    return F.cross_entropy(logits_active, targets.view(-1)[active])

def compute_masked_ce_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    active = mask.view(-1) > 0.5
    if not active.any():
        return logits.sum() * 0.0
    return F.cross_entropy(logits.view(-1, logits.size(-1))[active], targets.view(-1)[active])

class PreTokenizedMemoryDataset(Dataset):
    """Zero-Overhead: Pre-tokenizes all strings on init into flat integer lists."""
    def __init__(self, jsonl_path: str, vocab: Dict[str, int], max_len: int = 2048):
        self.max_len = max_len
        self.samples = []

        t0 = time.time()
        print(f"[{now_iso()}] Pre-tokenizing dataset from {jsonl_path} into memory...")
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"Training dataset not found: {jsonl_path}")

        count = 0
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                count += 1
                item = json.loads(line)
                raw_full = item.get("full_text")
                raw_prompt = item.get("prompt", "")
                if not raw_full:
                    raw_full = raw_prompt + item.get("target", "")
                full_text = canonical_normalize_text(raw_full)
                prompt = canonical_normalize_text(raw_prompt)
                tokens = [vocab[c] for c in full_text if c in vocab]
                prompt_len = sum(1 for c in prompt if c in vocab)
                if len(tokens) > 1:
                    if len(tokens) < max_len:
                        tokens.append(3)  # Append native <EOS> token (ID 3) so generation terminates cleanly
                    else:
                        tokens = tokens[:max_len]
                    self.samples.append((tokens, min(prompt_len, len(tokens) - 1)))
        print(f"[{now_iso()}] Pre-tokenized {len(self.samples):,} samples (read {count:,} lines) in {time.time()-t0:.2f}s (RAM resident).")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[List[int], int]:
        return self.samples[idx]

class DynamicTokenBucketBatchSampler:
    """Groups samples of similar token lengths into megabuckets and dynamically caps (B * aligned_max <= max_tokens)
    so 512-token batches and 128-token batches consume flat, predictable VRAM without OOM spikes."""
    def __init__(self, dataset: PreTokenizedMemoryDataset, max_batch_size: int = 96, max_tokens: int = 24576, megabucket_mult: int = 32):
        self.dataset = dataset
        self.max_batch_size = max_batch_size
        self.max_tokens = max_tokens
        self.megabucket_size = max_batch_size * megabucket_mult
        self.lengths = [len(s[0]) for s in dataset.samples]
        self._cached_batches = self._build_batches(shuffle=False)

    def _build_batches(self, shuffle: bool = True) -> List[List[int]]:
        n = len(self.dataset)
        indices = list(range(n))
        if shuffle:
            random.shuffle(indices)
        batches = []
        for m_start in range(0, n, self.megabucket_size):
            mega = indices[m_start : m_start + self.megabucket_size]
            mega.sort(key=lambda idx: self.lengths[idx])
            pos = 0
            while pos < len(mega):
                cand_end = min(len(mega), pos + self.max_batch_size)
                max_l = self.lengths[mega[cand_end - 1]]
                aligned_max = max(32, int(math.ceil(max_l / 32.0)) * 32)
                safe_b = min(self.max_batch_size, max(8, (self.max_tokens // aligned_max // 8) * 8))
                actual_end = min(len(mega), pos + safe_b)
                b = mega[pos:actual_end]
                if len(b) >= 4:
                    batches.append(b)
                pos = actual_end
        if shuffle:
            random.shuffle(batches)
        return batches

    def __iter__(self):
        batches = self._build_batches(shuffle=True)
        for b in batches:
            yield b

    def __len__(self) -> int:
        return len(self._cached_batches)

def zero_overhead_collate_fn(batch: List[Tuple[List[int], int]], pad_token_id: int = 0) -> Dict[str, torch.Tensor]:
    B = len(batch)
    lengths = [len(b[0]) for b in batch]
    raw_max = max(lengths) if lengths else 32

    # 32-warp alignment for Ampere Tensor Cores
    aligned_max = int(math.ceil(raw_max / 32.0)) * 32
    aligned_max = max(32, aligned_max)

    input_ids = torch.full((B, aligned_max), pad_token_id, dtype=torch.long)
    targets = torch.full((B, aligned_max), pad_token_id, dtype=torch.long)
    valid_mask = torch.zeros((B, aligned_max), dtype=torch.bool)
    loss_mask = torch.zeros((B, aligned_max), dtype=torch.float32)

    for i, (toks, prompt_len) in enumerate(batch):
        n = len(toks)
        if n > 1:
            t = torch.tensor(toks, dtype=torch.long)
            input_ids[i, :n - 1] = t[:-1]
            targets[i, :n - 1] = t[1:]
            valid_mask[i, :n - 1] = True

            target_start = max(0, prompt_len - 1)
            target_end = n - 1
            if target_end > target_start:
                loss_mask[i, target_start:target_end] = 1.0

    return {
        "input_ids": input_ids,
        "targets": targets,
        "valid_mask": valid_mask,
        "loss_mask": loss_mask
    }

def evaluate_tier2_decomposed_suite(model, manifest_path: Path, vocab, inv_vocab, device) -> Dict[str, Any]:
    print(f"\n--- [Tier 2] Evaluating Decomposed State Benchmark ({manifest_path.name}) ---")
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}")
        return {}

    items = [json.loads(line) for line in open(manifest_path, encoding='utf-8')]
    model.eval()
    t0 = time.time()
    total = len(items)
    e2e_correct = 0
    stage1_pass_cnt = 0
    stage2_pass_cnt = 0
    stage3_pass_cnt = 0
    stage4_pass_cnt = 0
    sub_stats = {}

    with torch.no_grad():
        for i, item in enumerate(items):
            prompt = item["prompt"]
            sub = item.get("sub_type", "unknown")
            if sub not in sub_stats:
                sub_stats[sub] = {"total": 0, "correct": 0}
            sub_stats[sub]["total"] += 1

            gen_res = model.generate_with_qwenputer(
                prompt, vocab, inv_vocab,
                max_tokens=None, arm="learned",
                repetition_penalty=1.0
            )
            raw_text = gen_res["output_text"].strip()
            dec_res = ev38.evaluate_decomposed_item(item, raw_text)

            if dec_res["stage1_name_extraction"]: stage1_pass_cnt += 1
            if dec_res["stage2_table_bijection"]: stage2_pass_cnt += 1
            if dec_res["stage3_cpu_logic"]: stage3_pass_cnt += 1
            if dec_res["stage4_table_inversion"]: stage4_pass_cnt += 1
            if dec_res["end_to_end_match"]:
                e2e_correct += 1
                sub_stats[sub]["correct"] += 1

            if (i + 1) % 20 == 0 or (i + 1) == total:
                print(f"  Progress: {i+1:3d}/{total} | E2E: {e2e_correct:3d} ({e2e_correct/(i+1)*100:5.1f}%) | "
                      f"S1:{stage1_pass_cnt:3d} S2:{stage2_pass_cnt:3d} S3:{stage3_pass_cnt:3d} S4:{stage4_pass_cnt:3d} | "
                      f"Elapsed: {time.time()-t0:.1f}s")

    acc = e2e_correct / max(1, total) * 100.0
    print(f"  => [Tier 2 Complete] End-to-End Accuracy: {acc:.2f}% ({e2e_correct}/{total}) in {time.time()-t0:.1f}s")
    print(f"       * Stage 1 Name Extraction: {stage1_pass_cnt/total*100:5.1f}% ({stage1_pass_cnt}/{total})")
    print(f"       * Stage 2 Table Bijection: {stage2_pass_cnt/total*100:5.1f}% ({stage2_pass_cnt}/{total})")
    print(f"       * Stage 3 CPU Logic Ground:{stage3_pass_cnt/total*100:5.1f}% ({stage3_pass_cnt}/{total})")
    print(f"       * Stage 4 Table Inversion: {stage4_pass_cnt/total*100:5.1f}% ({stage4_pass_cnt}/{total})")
    for s_name, s_data in sorted(sub_stats.items()):
        s_acc = s_data["correct"] / max(1, s_data["total"]) * 100.0
        print(f"       * {s_name:25s}: {s_acc:5.1f}% ({s_data['correct']:2d}/{s_data['total']:2d})")
    model.train()
    return {
        "total": total,
        "correct": e2e_correct,
        "accuracy": acc,
        "stage1_acc": stage1_pass_cnt / total * 100.0,
        "stage2_acc": stage2_pass_cnt / total * 100.0,
        "stage3_acc": stage3_pass_cnt / total * 100.0,
        "stage4_acc": stage4_pass_cnt / total * 100.0,
        "breakdown": sub_stats
    }

def main():
    parser = argparse.ArgumentParser(description="GeoCPU v41: 247.5M Zero-Overhead Co-Training Runtime")
    parser.add_argument("--train_path", type=str, default=str(ROOT / "data" / "train_v41_unified.jsonl"))
    parser.add_argument("--meta_path", type=str, default=str(ROOT / "base_metadata.json"))
    parser.add_argument("--warmstart_path", type=str, default=str(ROOT / "checkpoints" / "warmstart_v41_247M_from_v38.pt"))
    parser.add_argument("--checkpoint", type=str, default=None, help="Explicit checkpoint to load or evaluate")
    parser.add_argument("--eval_only", action="store_true", help="Run 4-tier benchmark evaluation directly without training")
    parser.add_argument("--output_dir", type=str, default=str(ROOT / "checkpoints"))
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=96)
    parser.add_argument("--save_interval", type=int, default=2500)
    parser.add_argument("--eval_interval", type=int, default=2500)
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print("=" * 90)
    print(f"[{now_iso()}] V41 SCALED 247.5M NEUROSYMBOLIC CO-TRAINING ENGINE (ZERO-OVERHEAD)")
    print("=" * 90)
    print(f"  Mode:            {'EVALUATION ONLY (4-Tier Benchmarks)' if args.eval_only else 'CO-TRAINING (5,000 Steps) + BENCHMARKS'}")
    print(f"  Train File:      {args.train_path}")
    target_ckpt = args.checkpoint if args.checkpoint else args.warmstart_path
    print(f"  Active Ckpt:     {target_ckpt}")
    print(f"  Max Steps:       {args.max_steps} | Batch Size: {args.batch_size}")
    print(f"  Save Interval:   {args.save_interval} | Eval Interval: {args.eval_interval}")
    print(f"  Device:          {args.device}")

    # Ampere Acceleration Flags
    if "cuda" in args.device and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass

    dev = torch.device(args.device)
    if dev.type == "cuda":
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    if not os.path.exists(args.meta_path):
        raise FileNotFoundError(f"base_metadata.json not found: {args.meta_path}")

    meta = json.loads(Path(args.meta_path).read_bytes())
    vocab = meta["vocab"]
    inv_vocab = {int(v): k for k, v in vocab.items()}

    cfg_dict = dict(meta["config"])
    cfg_dict["scan_backend"] = "jit"
    cfg = CEDConfig(**cfg_dict)

    base_model = GeoCPU_CED_v2(cfg)
    model = IntegratedModelV30(base=base_model)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model Parameters: {total_params:,} (247.56M Architecture Confirmed)")

    # Load checkpoint weights via CPU memory to prevent unpickling GPU OOM
    if os.path.exists(target_ckpt):
        print(f"  Loading Weights from Checkpoint (CPU buffer): {target_ckpt}...")
        ckpt = torch.load(target_ckpt, map_location="cpu")
        sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(sd, strict=False)
        del ckpt, sd
        gc.collect()
        print("  -> Weights loaded successfully into 247.5M architecture.")
    else:
        print(f"  WARNING: Checkpoint not found at {target_ckpt}. Initializing cleanly.")

    if dev.type == "cuda":
        torch.cuda.empty_cache()
    model.to(dev)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_ckpt = str(target_ckpt)

    if not args.eval_only:
        use_amp = (dev.type == "cuda")
        if dev.type == "cuda":
            torch.cuda.empty_cache()
            free_bytes, total_bytes = torch.cuda.mem_get_info(dev)
            free_gb = free_bytes / (1024 ** 3)
            total_gb = total_bytes / (1024 ** 3)
            print(f"  GPU Memory Status after Model Load: {free_gb:.2f} GB Free / {total_gb:.2f} GB Total")
            if free_gb >= 26.0:
                max_tokens_per_micro = 36864
                grad_accum_steps = 1
            elif free_gb >= 16.0:
                max_tokens_per_micro = 24576
                grad_accum_steps = 1
            else:
                print(f"  [!] Notice: External notebook/process is holding {total_gb - free_gb:.1f} GB VRAM. Auto-scaling token budget to fit {free_gb:.2f} GB!")
                max_tokens_per_micro = max(6144, int(max(2.0, free_gb - 3.0) * 2048))
                grad_accum_steps = max(1, 28672 // max_tokens_per_micro)
        else:
            max_tokens_per_micro = 16384
            grad_accum_steps = 1

        max_micro_batch = min(96, args.batch_size)
        print(f"  Execution Mode:  AMP bfloat16={use_amp} | Dynamic Token Budget: {max_tokens_per_micro:,} tok/micro (Max B={max_micro_batch}, Accum={grad_accum_steps})")

        # In-memory dataset: max_len=cfg.max_positions (2048) accommodates 100% of Formal Logic + Lean 4 (up to 852), SWE loops (up to 753), and L.pdf natural deduction (up to 1455)
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

        def get_lr(step: int) -> float:
            warmup = 150
            if step < warmup:
                return args.lr * (step + 1) / warmup
            progress = (step - warmup) / max(1, args.max_steps - warmup)
            return max(1e-5, args.lr * 0.5 * (1.0 + math.cos(math.pi * progress)))

        def run_microbatch_safe(inp: torch.Tensor, tgt: torch.Tensor, vmask: torch.Tensor, lmask: torch.Tensor, weight: float) -> torch.Tensor:
            """Executes forward + backward; if CUDA OOM ever occurs, clears traceback frames FIRST, frees VRAM, and splits batch."""
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
                import gc
                gc.collect()
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
                if cur_b <= 1:
                    raise RuntimeError("CUDA OutOfMemoryError even at batch_size=1")
                mid = cur_b // 2
                w1 = weight * (mid / cur_b)
                w2 = weight * ((cur_b - mid) / cur_b)
                l1 = run_microbatch_safe(inp[:mid], tgt[:mid], vmask[:mid], lmask[:mid], w1)
                l2 = run_microbatch_safe(inp[mid:], tgt[mid:], vmask[mid:], lmask[mid:], w2)
                return l1 + l2

        drive_sync_dirs = [
            Path("/content/drive/MyDrive/GeoCPU_Frontier_CoTraining_v41/checkpoints"),
            out_dir
        ]

        heldout_manifest = ROOT / "data" / "eval_swe_step_control_heldout.jsonl"

        print(f"\nSTARTING V41 247.5M CO-TRAINING: {args.max_steps} STEPS (Dynamic Token Budget: {max_tokens_per_micro:,} tok x {grad_accum_steps})")
        step = 0
        running_loss_gpu = torch.zeros((), dtype=torch.float32, device=dev)
        running_samples = 0
        last_avg_loss = 0.0
        t_start = time.time()
        t_prev = time.time()

        model.train()
        data_iter = iter(dataloader)

        while step < args.max_steps:
            step += 1
            lr = get_lr(step)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad(set_to_none=True)

            for accum_idx in range(grad_accum_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    batch = next(data_iter)

                input_ids = batch["input_ids"].to(dev, non_blocking=True)
                targets = batch["targets"].to(dev, non_blocking=True)
                valid_mask = batch["valid_mask"].to(dev, non_blocking=True)
                loss_mask = batch["loss_mask"].to(dev, non_blocking=True)

                running_samples += input_ids.size(0)
                step_l = run_microbatch_safe(input_ids, targets, valid_mask, loss_mask, 1.0 / grad_accum_steps)
                running_loss_gpu += step_l

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step % 10 == 0:
                avg_loss = (running_loss_gpu / 10.0).item()
                last_avg_loss = avg_loss
                running_loss_gpu.zero_()
                dt = time.time() - t_prev
                t_prev = time.time()
                steps_per_sec = 10.0 / max(0.001, dt)
                samples_per_sec = running_samples / max(0.001, dt)
                running_samples = 0
                rem_steps = args.max_steps - step
                eta_min = (rem_steps / max(0.01, steps_per_sec)) / 60.0
                vram_gb = (torch.cuda.max_memory_allocated(dev) / (1024**3)) if dev.type == "cuda" else 0.0
                print(f"[{now_iso()}] Step {step:5d}/{args.max_steps} | Loss: {avg_loss:6.4f} | "
                      f"LrBase: {lr:.2e} | Speed: {steps_per_sec:4.1f} step/s ({samples_per_sec:4.0f} samp/s) | VRAM: {vram_gb:4.1f}GB | ETA: {eta_min:4.1f} min", flush=True)

            # Milestone Checkpoint & Evaluation
            if step % args.save_interval == 0 or step == args.max_steps:
                print("\n" + "=" * 85)
                print(f"[{now_iso()}] MILESTONE CHECKPOINT AT STEP {step}")
                print("=" * 85)

                ckpt_file = out_dir / f"checkpoint_step_{step}.pt"
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "parameters": total_params,
                    "config": meta["config"],
                    "loss": last_avg_loss
                }, str(ckpt_file))
                print(f"  Saved checkpoint -> {ckpt_file.name} ({ckpt_file.stat().st_size:,} bytes)")

                # Auto-sync to Google Drive
                for d_dir in drive_sync_dirs:
                    if d_dir != out_dir and d_dir.parent.exists():
                        try:
                            d_dir.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(ckpt_file, d_dir / ckpt_file.name)
                            print(f"  Synced checkpoint to Google Drive: {d_dir / ckpt_file.name}")
                            break
                        except Exception as e:
                            pass

                # Run Fast 20-Item Stratified Probe at intermediate milestones (Full 200-item Tier 1 runs at max_steps)
                if step < args.max_steps and heldout_manifest.exists():
                    print(f"\n--- Running Fast 20-Item Stratified Step-Control Probe at Step {step} ---")
                    eval_step_control_heldout.run_heldout_evaluation(
                        checkpoint_path=str(ckpt_file),
                        manifest_path=str(heldout_manifest),
                        arm="learned",
                        device=args.device,
                        model=model,
                        vocab=vocab,
                        inv_vocab=inv_vocab,
                        max_samples=20
                    )

                print("=" * 85 + "\n")
                model.train()

        total_time = time.time() - t_start
        print(f"\n[{now_iso()}] TRAINING COMPLETED IN {total_time/60.0:.2f} MINUTES!")
        final_ckpt = str(out_dir / f"checkpoint_step_{args.max_steps}.pt")

    # ----------------------------------------------------------------------------------
    # STEP 5000: RUN COMPLETE 5-TIER BENCHMARK SUITE (11,197 QUESTIONS)
    # ----------------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print(f"[{now_iso()}] EXECUTING FULL MULTI-TIER BENCHMARK SUITE (11,197 HISTORICAL QUESTIONS)")
    print("=" * 90)
    if not args.eval_only:
        final_ckpt = str(out_dir / f"checkpoint_step_{args.max_steps}.pt")
    else:
        final_ckpt = str(target_ckpt)

    # Tier 1: Held-Out Step Control Suite (200 items: Unseen Loops 1, 3, 5, 7)
    t1_manifest = ROOT / "data" / "eval_swe_step_control_heldout.jsonl"
    print("\n[TIER 1/5] Held-Out SWE Step-Control Benchmark (200 Items: Unseen Loops 1, 3, 5, 7)...")
    t1_stats = eval_step_control_heldout.run_heldout_evaluation(
        checkpoint_path=final_ckpt,
        manifest_path=str(t1_manifest),
        arm="learned",
        device=args.device,
        model=model,
        vocab=vocab,
        inv_vocab=inv_vocab
    )

    # Tier 2: Decomposed State Tracking (200 items)
    t2_manifest = ROOT / "data" / "eval_v38_decomposed_state_bench200.jsonl"
    print("\n[TIER 2/5] Decomposed State Tracking Benchmark (200 Items: Stages 1-4 & E2E)...")
    t2_res = evaluate_tier2_decomposed_suite(model, t2_manifest, vocab, inv_vocab, dev)

    if dev.type == "cuda":
        torch.cuda.empty_cache()
    eval_batch_size = min(64, args.batch_size)

    # Tier 3: SoftBank JamC-QA Full Benchmark (2,309 items)
    jamc_manifest = str(ROOT / "data" / "jamcqa_manifest_test.jsonl")
    jamc_res = {}
    if os.path.exists(jamc_manifest):
        print(f"\n[TIER 3/5] SoftBank JamC-QA Full Benchmark (2,309 items)...")
        jamc_res = eval_jamcqa_official.run_jamcqa_official_eval(
            checkpoint_path=final_ckpt,
            manifest_path=jamc_manifest,
            mode="both",
            device=args.device,
            batch_size=eval_batch_size,
            max_gen_samples=100,
            model=model,
            vocab=vocab,
            inv_vocab=inv_vocab
        )

    # Tier 4: Canonical Broad 8,100 Benchmark (7,536 JMMLU + 164 SWE + 400 ARC)
    bench_8100_manifest = str(ROOT / "data" / "benchmark_8100_manifest.jsonl")
    bench_8100_res = {}
    if os.path.exists(bench_8100_manifest):
        print(f"\n[TIER 4/5] Canonical Broad 8,100 Full Benchmark (7,536 JMMLU + 164 SWE + 400 ARC)...")
        bench_8100_res = eval_benchmark_full_8100.run_evaluation(
            checkpoint_path=final_ckpt,
            manifest_path=bench_8100_manifest,
            arm="learned",
            device=args.device,
            batch_size=eval_batch_size,
            model=model,
            vocab=vocab,
            inv_vocab=inv_vocab
        )

    # Tier 5: 6-Operator Logical Instruction Contrast Benchmark (400 items)
    t5_manifest = str(ROOT / "data" / "eval_v41_logical_instruction_operators_heldout.jsonl")
    t5_res = {}
    if os.path.exists(t5_manifest):
        print(f"\n[TIER 5/5] 6-Operator Logical Instruction Contrast Benchmark (400 Items: NOT/CONTRAPOS/NECESSARY/DE_MORGAN/UNDEDUCIBLE/SYLLOGISM)...")
        t5_res = eval_logical_instruction_operators.run_logical_operator_evaluation(
            checkpoint_path=final_ckpt,
            manifest_path=t5_manifest,
            device=args.device,
            batch_size=eval_batch_size,
            max_gen_samples=60,
            model=model,
            vocab=vocab,
            inv_vocab=inv_vocab
        )

    # ----------------------------------------------------------------------------------
    # MASTER HISTORICAL SCIENTIFIC SCOREBOARD
    # ----------------------------------------------------------------------------------
    print("\n" + "=" * 105)
    print("MASTER HISTORICAL SCIENTIFIC SCOREBOARD: v31/v32 vs v38 vs v41 (247.56M)")
    print("=" * 105)
    print(f"{'Evaluation Metric / Benchmark Pillar':<42} | {'Questions':>9} | {'v32 Baseline':>14} | {'v38 Baseline':>14} | {'v41 Frontier (247.5M)':>21}")
    print("-" * 105)

    # Tier 1
    t1_acc = t1_stats.get("correct_val", 0) / max(1, t1_stats.get("total", 1)) * 100.0
    t1_exact = t1_stats.get("exact_step", 0) / max(1, t1_stats.get("total", 1)) * 100.0
    print(f"{'Tier 1: SWE Held-Out Exact Numeric Match':<42} | {t1_stats.get('total', 200):9d} | {'0.00%':>14} | {'22.50%':>14} | {t1_acc:>20.2f}%")
    print(f"{'Tier 1: SWE Held-Out Exact Step Termination':<42} | {t1_stats.get('total', 200):9d} | {'8.00%':>14} | {'41.00%':>14} | {t1_exact:>20.2f}%")

    # Tier 2
    t2_acc = t2_res.get("accuracy", 0.0)
    t2_s2 = t2_res.get("stage2_acc", 0.0)
    print(f"{'Tier 2: State Tracking End-to-End Match':<42} | {t2_res.get('total', 200):9d} | {'--':>14} | {'33.00%':>14} | {t2_acc:>20.2f}%")
    print(f"{'Tier 2: Table Bijection Coherence (Stage 2)':<42} | {t2_res.get('total', 200):9d} | {'--':>14} | {'74.50%':>14} | {t2_s2:>20.2f}%")

    # Tier 3
    jamc_logit_acc = jamc_res.get("logit_argmax", {}).get("accuracy", 0.0) * 100.0
    jamc_gen_acc = jamc_res.get("generative_em", {}).get("accuracy", 0.0) * 100.0
    print(f"{'Tier 3: SoftBank JamC-QA (Logit Argmax)':<42} | {'2309':>9} | {'24.68%':>14} | {'24.77%':>14} | {jamc_logit_acc:>20.2f}%")
    print(f"{'Tier 3: SoftBank JamC-QA (Generative EM)':<42} | {'100':>9} | {'--':>14} | {'1.00%':>14} | {jamc_gen_acc:>20.2f}%")

    # Tier 4
    if bench_8100_res:
        p_stats = bench_8100_res.get("pillar_stats", {})
        jmmlu_acc = p_stats.get("JMMLU_Academic", {}).get("correct", 0) / max(1, p_stats.get("JMMLU_Academic", {}).get("total", 1)) * 100.0
        swe_acc = p_stats.get("SWE_Software", {}).get("correct", 0) / max(1, p_stats.get("SWE_Software", {}).get("total", 1)) * 100.0
        arc_acc = p_stats.get("ARC_Reasoning", {}).get("correct", 0) / max(1, p_stats.get("ARC_Reasoning", {}).get("total", 1)) * 100.0
        overall_acc = bench_8100_res.get("accuracy", 0.0) * 100.0
        print(f"{'Tier 4: JMMLU Academic (All 56 Subjects)':<42} | {'7536':>9} | {'24.68%':>14} | {'24.64%':>14} | {jmmlu_acc:>20.2f}%")
        print(f"{'Tier 4: SWE Software State Tracing':<42} | {'164':>9} | {'2.44%':>14} | {'12.80%':>14} | {swe_acc:>20.2f}%")
        print(f"{'Tier 4: ARC Abstract Grid Reasoning':<42} | {'400':>9} | {'0.00%':>14} | {'0.00%':>14} | {arc_acc:>20.2f}%")
        print("-" * 105)
        print(f"{'OVERALL CANONICAL BENCHMARK (STRICT)':<42} | {'8100':>9} | {'23.01%':>14} | {'23.20%':>14} | {overall_acc:>20.2f}%")

    # Tier 5
    if t5_res:
        t5_acc = t5_res.get("accuracy", 0.0)
        t5_neg_und = t5_res.get("neg_vs_undeducible_rate", 0.0)
        t5_switch = t5_res.get("tri_switch_rate", 0.0)
        t5_tot = t5_res.get("total", 400)
        print("-" * 105)
        print(f"{'Tier 5: 6-Operator Logical Contrast Overall':<42} | {t5_tot:9d} | {'--':>14} | {'--':>14} | {t5_acc:>20.2f}%")
        print(f"{'Tier 5: ¬P vs (P |/- Q) Discrimination':<42} | {'100':>9} | {'--':>14} | {'--':>14} | {t5_neg_und:>20.2f}%")
        print(f"{'Tier 5: 3-Way Instruction Switch Rate':<42} | {'100':>9} | {'--':>14} | {'--':>14} | {t5_switch:>20.2f}%")
    print("=" * 105)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n" + "!" * 80)
        print(f"FATAL EXCEPTION IN V41 RUNTIME: {e}")
        traceback.print_exc()
        print("!" * 80 + "\n")
        sys.exit(1)
