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
import random
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
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

from qwenputer_controller_v30 import canonical_normalize_text
from runtime_cotraining_v41_247M import (
    now_iso, compute_cut_ce_loss, PreTokenizedMemoryDataset,
    DynamicTokenBucketBatchSampler, zero_overhead_collate_fn, evaluate_tier2_decomposed_suite
)


class ResumableTokenBucketSampler(DynamicTokenBucketBatchSampler):
    """The original bucket math with a private RNG and serializable epoch/cursor."""
    def __init__(self, *args, seed: int = 460006, **kwargs):
        super().__init__(*args, **kwargs)
        self.rng = random.Random(seed)
        self.batches = None
        self.cursor = 0
        self.epoch = 0
        self.last_batch = None

    def __iter__(self):
        if self.batches is None or self.cursor >= len(self.batches):
            # _build_batches uses module-global random; isolate its state.
            outer = random.getstate()
            random.setstate(self.rng.getstate())
            try:
                self.batches = self._build_batches(shuffle=True)
                self.rng.setstate(random.getstate())
            finally:
                random.setstate(outer)
            self.cursor = 0
            self.epoch += 1
        while self.cursor < len(self.batches):
            batch = self.batches[self.cursor]
            self.cursor += 1
            self.last_batch = batch
            yield batch

    def state_dict(self):
        return {"batches": self.batches, "cursor": self.cursor, "epoch": self.epoch,
                "rng_state": self.rng.getstate(), "n": len(self.dataset),
                "max_batch_size": self.max_batch_size, "max_tokens": self.max_tokens,
                "last_batch": self.last_batch}

    def load_state_dict(self, state):
        if (state["n"] != len(self.dataset) or state["max_batch_size"] != self.max_batch_size
                or state["max_tokens"] != self.max_tokens):
            raise RuntimeError("sampler/dataset configuration mismatch on exact resume")
        self.batches = state["batches"]
        self.cursor = int(state["cursor"])
        self.epoch = int(state["epoch"])
        self.rng.setstate(state["rng_state"])
        self.last_batch = state["last_batch"]
        if self.batches is None or not 0 <= self.cursor <= len(self.batches):
            raise RuntimeError("invalid sampler cursor on exact resume")


class ToolMaskedDataset(Dataset):
    """PreTokenizedMemoryDataset + per-row character spans that are excluded from the loss.

    The gate after v44b showed the model predicting tool observations instead of reading them (29% of marked SWE
    runs ended with the model writing its own ```output block). Observations are written by the runtime, so the
    model should only learn to CONDITION on them. v43 rows have no spans and are tokenized exactly as before.
    Spans are character offsets into full_text; they are used only when every character survives tokenization
    (normalization is a no-op and all characters are in the vocabulary), so character index == token index."""

    @staticmethod
    def split_sequence(tokens, prompt_len, spans, max_len):
        """Cover each answer character and the one true EOS exactly once.

        Later windows retain up to 512 previous tokens as context. Earlier
        windows never acquire extra loss when they overlap a later window.
        """
        sequence = list(tokens) + [3]
        first_target = max(1, min(int(prompt_len), len(tokens)))
        target = first_target
        first = True
        while target < len(sequence):
            start = max(0, target - (max_len - 1 if first else 512))
            end = min(len(sequence), start + max_len)
            if end <= target:
                raise RuntimeError("cannot cover supervised target within context window")
            local_spans = tuple((max(s, start) - start, min(e, end) - start)
                                for s, e in spans if max(s, start) < min(e, end))
            yield (sequence[start:end], target - start, local_spans)
            target = end
            first = False

    def __init__(self, jsonl_path, vocab, max_len=2048):
        self.max_len = max_len
        self.samples = []
        self.groups = []
        self.origins = []
        self.supervised_counts = []
        t0 = time.time()
        n_rows = n_spans = n_dropped = masked = target_tok = long_rows = 0
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                n_rows += 1
                raw_full = item.get("full_text") or (item.get("prompt", "") + item.get("target", ""))
                full_text = canonical_normalize_text(raw_full)
                prompt = canonical_normalize_text(item.get("prompt", ""))
                tokens = [vocab[c] for c in full_text if c in vocab]
                prompt_len = sum(1 for c in prompt if c in vocab)
                spans = item.get("loss_mask_spans") or []
                if spans and (full_text != raw_full or len(tokens) != len(full_text)):
                    n_dropped += 1
                    spans = []
                if len(tokens) <= 1:
                    raise RuntimeError(f"row {n_rows} has fewer than two tokens")
                spans = tuple((int(s), int(e)) for s, e in spans if int(s) < len(tokens))
                long_rows += int(len(tokens) + 1 > max_len)
                is_v43 = str(item.get("_v46_group", "legacy")) != "practical_multistep_v1"
                for chunk, pl, local_spans in self.split_sequence(tokens, prompt_len, spans, max_len):
                    n_spans += len(local_spans)
                    target_tok += len(chunk) - pl
                    masked += sum(min(e, len(chunk)) - max(s, pl) for s, e in local_spans
                                  if min(e, len(chunk)) > max(s, pl))
                    self.samples.append((chunk, pl, local_spans, is_v43))
                    self.groups.append(str(item.get("_v46_group", "legacy")))
                    self.origins.append(str(item.get("_practical_source", item.get("_v46_group", "legacy"))))
                    self.supervised_counts.append(practical_supervised_count(chunk, pl, local_spans))
        self.stats = {"rows": n_rows, "spans": n_spans, "rows_with_unaligned_spans_dropped": n_dropped,
                      "long_rows_chunked": long_rows, "samples_after_chunking": len(self.samples),
                      "masked_target_token_share": masked / max(1, target_tok)}
        print(f"[{now_iso()}] ToolMaskedDataset: {len(self.samples):,} samples in {time.time() - t0:.1f}s | {self.stats}", flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def masked_collate_fn(batch, pad_token_id=0):
    out = zero_overhead_collate_fn([(s[0], s[1]) for s in batch], pad_token_id)
    lm = out["loss_mask"]
    for i, sample in enumerate(batch):
        toks, spans = sample[0], sample[2]
        n = len(toks)
        for s, e in spans:
            # position j predicts token j+1: excluding tokens [s, e) clears positions [s-1, e-1)
            a, b = max(0, s - 1), min(n - 1, e - 1)
            if b > a:
                lm[i, a:b] = 0.0
    # r4: which rows are v43's own (distillation toward v43 applies only to them)
    out["v43_row"] = torch.tensor([bool(s[3]) if len(s) > 3 else True for s in batch], dtype=torch.bool)
    return out


def distill_kl(model, teacher, fine_s, fine_t, mask, temperature=1.0):
    """KL(teacher || student) per supervised position, averaged over positions (LwF-style retention term).
    Logits are computed only at mask>0.5 positions (same cut-head trick as compute_cut_ce_loss)."""
    active = mask.reshape(-1) > 0.5
    if not bool(active.any()):
        return fine_s.sum() * 0.0
    T = float(temperature)
    s = model.base.head(model.base.final_norm(fine_s.reshape(-1, fine_s.size(-1))[active])).float() / T
    with torch.no_grad():
        t = teacher.base.head(teacher.base.final_norm(fine_t.reshape(-1, fine_t.size(-1))[active])).float() / T
    return F.kl_div(F.log_softmax(s, -1), F.log_softmax(t, -1), log_target=True, reduction="batchmean") * (T * T)


# r5b: what one optimizer step sees must not depend on the GPU Colab hands out.
# run_7_r5 got a smaller GPU than every earlier run: the sampler budget fell from 36,864 to 24,576 tokens, batches of
# long rows ran out of memory, and the old fallback kept the failed batch's graph alive while it retried the halves, so
# every half failed too and was skipped: 4,064 rows (LogiQA2/FOLIO first-order logic, r5's own predicate-logic rows) in
# 1,000 steps, plus the partial gradient of each failed backward. Now the sampler budget is fixed (36,864, as in v43 and
# all A100 runs), each sampled batch is run in chunks that fit, and an out-of-memory error redoes the whole step with
# smaller chunks. Nothing is ever skipped; a single sequence that does not fit stops the run.
DEFAULT_STEP_TOKENS = 36864


class ChunkPolicy:
    """Largest chunk (in tokens) per padded-length bucket. Starts at `start_tokens`; an OOM halves the bucket's limit."""

    def __init__(self, start_tokens, bucket=256):
        self.start = int(start_tokens)
        self.bucket = int(bucket)
        self.limit = {}
        self.oom_retries = 0
        self.last = None

    def _key(self, L):
        return int(math.ceil(L / self.bucket))

    def seqs_per_chunk(self, L):
        return max(1, self.limit.get(self._key(L), self.start) // max(1, L))

    def on_oom(self, n, L):
        """n sequences of padded length L did not fit. Raises SystemExit when n == 1 (a single sequence)."""
        if n <= 1:
            raise SystemExit(f"[chunk] one sequence of {L} tokens does not fit on this GPU; use a larger GPU "
                             f"(nothing was skipped; the run stops here)")
        self.limit[self._key(L)] = max(1, n // 2) * L
        self.oom_retries += 1

    def describe(self):
        return {"start_tokens": self.start, "bucket": self.bucket, "oom_retries": self.oom_retries,
                "limits": {f"len<={k * self.bucket}": v for k, v in sorted(self.limit.items())}}


def forward_backward_chunked(model, batch, policy, dev, use_amp, weight=1.0, teacher=None, distill_weight=1.0,
                             distill_temperature=1.0):
    """Gradients of one sampled batch, accumulated chunk by chunk. Weights make the sum equal the whole-batch loss:
    cross-entropy by each chunk's share of supervised tokens, the routing term by its share of valid tokens, the KL by
    its share of supervised v43 tokens. Returns (loss, kl or None)."""
    inp = batch["input_ids"].to(dev, non_blocking=True)
    tgt = batch["targets"].to(dev, non_blocking=True)
    vmask = batch["valid_mask"].to(dev, non_blocking=True)
    lmask = batch["loss_mask"].to(dev, non_blocking=True)
    v43row = batch["v43_row"].to(dev, non_blocking=True) if "v43_row" in batch else None
    B, L = inp.shape
    act = lmask > 0.5
    n_ce = max(1, int(act.sum()))
    n_val = max(1, int(vmask.sum()))
    use_kl = teacher is not None and v43row is not None
    n_kl = int((act & v43row[:, None]).sum()) if use_kl else 0
    per = policy.seqs_per_chunk(L)
    total, kl_sum = 0.0, 0.0
    for a in range(0, B, per):
        b = min(B, a + per)
        policy.last = (b - a, L)
        c_ce, c_val = int(act[a:b].sum()), int(vmask[a:b].sum())
        with torch.amp.autocast(device_type=dev.type, enabled=use_amp, dtype=torch.bfloat16):
            out = model(inp[a:b], vmask[a:b], arm="learned", skip_head=True)
            loss = compute_cut_ce_loss(model, out.fine_states, tgt[a:b], lmask[a:b]) * (c_ce / n_ce)
            loss = loss + 0.2 * (out.compression_ratio - 1.0).pow(2).mean() * (c_val / n_val)
            if n_kl:
                dmask = lmask[a:b] * v43row[a:b, None].to(lmask.dtype)
                c_kl = int((dmask > 0.5).sum())
                if c_kl:
                    with torch.no_grad():
                        t_out = teacher(inp[a:b], vmask[a:b], arm="learned", skip_head=True)
                    kl = distill_kl(model, teacher, out.fine_states, t_out.fine_states, dmask, distill_temperature)
                    loss = loss + distill_weight * kl * (c_kl / n_kl)
                    kl_sum += float(kl.detach()) * c_kl / n_kl
            loss = loss * weight
        loss.backward()
        total += float(loss.detach())
    return total, (kl_sum if n_kl else None)


def step_with_retry(model, optimizer, batch, policy, dev, use_amp, teacher=None, distill_weight=1.0,
                    distill_temperature=1.0, kl_log=None, step=0):
    """The whole step or nothing: on OOM the partial gradients are dropped, the chunk limit for that length is halved and
    the same batch is run again. (The old fallback skipped sequences and kept the failed graph alive.)"""
    while True:
        optimizer.zero_grad(set_to_none=True)
        try:
            loss_val, kl_val = forward_backward_chunked(model, batch, policy, dev, use_amp, 1.0, teacher,
                                                        distill_weight, distill_temperature)
        except torch.OutOfMemoryError as e:
            traceback.clear_frames(e.__traceback__)
            loss_val = None
        if loss_val is not None:
            if kl_val is not None and kl_log is not None:
                kl_log.append(kl_val)
            return loss_val
        optimizer.zero_grad(set_to_none=True)
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        n_fail, L_fail = policy.last
        policy.on_oom(n_fail, L_fail)
        print(f"  [chunk] OOM on {n_fail} x {L_fail} tokens at step {step}: up to {policy.seqs_per_chunk(L_fail)} "
              f"sequences of this length per pass from now on; the step is redone (retry #{policy.oom_retries}).", flush=True)


def phase_lr(st, start_step, phase_start, max_steps, peak, floor, warmup):
    """v44b phase schedule: cosine peak->floor over [phase_start, max_steps]; times a linear ramp over `warmup`
    steps counted from THIS launch's start_step (warmup=0 when Adam moments were restored)."""
    prog = min(1.0, max(0.0, (st - phase_start) / max(1, max_steps - phase_start)))
    lr = floor + (peak - floor) * 0.5 * (1.0 + math.cos(math.pi * prog))
    if warmup > 0:
        lr *= min(1.0, (st - start_step) / warmup)
    return lr


def practical_supervised_count(chunk, prompt_len, spans):
    first = max(1, int(prompt_len))
    return len(chunk) - first - sum(max(0, min(e, len(chunk)) - max(s, first)) for s, e in spans)


def practical_telemetry(path, event):
    record = dict(event, timestamp_utc=datetime.now(timezone.utc).isoformat())
    line = json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False)
    with open(path, "a", encoding="utf-8") as sink:
        sink.write(line + "\n")
        sink.flush()
        os.fsync(sink.fileno())
    print("GECPU_TRAIN_EVENT " + line, flush=True)


def main():
    parser = argparse.ArgumentParser(description="GeoCPU v43: Knowledge-Relational & Search Co-Training Runtime")
    parser.add_argument("--train_path", type=str, default=str(ROOT / "data" / "train_v43_knowledge_unified.jsonl"))
    parser.add_argument("--corpus_path", type=str, default="", help="Pinned local retrieval index; remote search is disabled")
    parser.add_argument("--required_start_step", type=int, default=-1, help="Fail if the warmstart checkpoint is from another step")
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
    parser.add_argument("--max_tokens_per_micro", type=int, default=0, help="v44fix: override token budget (CPU smoke tests); same as --step_tokens")
    parser.add_argument("--step_tokens", type=int, default=0,
                        help=f"r5b: sampler budget per optimizer step, independent of the GPU (default {DEFAULT_STEP_TOKENS} on CUDA)")
    parser.add_argument("--chunk_tokens", type=int, default=0,
                        help="r5b: starting chunk size per forward/backward (default: from free GPU memory); OOM halves it per length")
    # v44b: v43 ended its cosine at the 8e-6 floor (max_steps=15000) and its checkpoints hold no optimizer state.
    # Resuming with max_steps=25000 jumped to 1.24e-5 with fresh AdamW moments and no warmup: every weight moves
    # ~lr per step for the first updates. These flags start a new phase that continues from v43's final LR instead.
    parser.add_argument("--phase_lr", type=float, default=0.0, help="v44b: peak LR of the new phase (0 = v43 schedule)")
    parser.add_argument("--phase_floor", type=float, default=0.0, help="v44b: LR at max_steps for the new phase")
    parser.add_argument("--phase_start", type=int, default=-1, help="v44b: step where the phase began (default: resume step)")
    parser.add_argument("--phase_warmup", type=int, default=300, help="v44b: linear warmup from the resume step whenever Adam moments start fresh")
    parser.add_argument("--save_optimizer", action="store_true", help="v44b: keep output_dir/optimizer_latest.pt so a resume continues Adam")
    parser.add_argument("--resume_state", type=str, default="", help="v46 exact sampler/RNG state for a matching saved step")
    parser.add_argument("--numina_repeat", type=int, default=1, help="planned per-phase presentation multiplier 1..4")
    parser.add_argument("--new_phase_sampler", action="store_true", help="intentionally reset sampler at a 2500-step curriculum boundary")
    parser.add_argument("--new_dataset_sampler", action="store_true", help="explicitly reset sampler after a SHA-pinned append; restore model, Adam and all RNG")
    parser.add_argument("--new_curriculum_repeat", type=int, default=1, help="presentation multiplier for appended practical_multistep_v1 rows")
    parser.add_argument("--new_group_min_exposures", type=int, default=1000, help="minimum deterministic new-row presentations before first update")
    parser.add_argument("--distill_teacher", type=str, default="",
                        help="v44b r4: frozen teacher checkpoint (v43); adds weight*KL(teacher||student) on v43 rows only")
    parser.add_argument("--distill_weight", type=float, default=1.0)
    parser.add_argument("--distill_temperature", type=float, default=1.0)
    parser.add_argument("--mask_tool_output", action="store_true",
                        help="v44b r2: exclude each row's loss_mask_spans (injected ```output blocks, failed first attempts) from the loss")
    parser.add_argument("--practical_variant", choices=("multistep", "answer_only", "replay_only"), default="multistep")
    parser.add_argument("--telemetry_jsonl", required=True)
    parser.add_argument("--min_replay_share", type=float, default=0.75)
    args = parser.parse_args()
    if not 0.75 <= args.min_replay_share <= 1.0:
        raise RuntimeError("replay supervised-token floor must be at least 0.75")

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
    corpus_path = Path(args.corpus_path) if args.corpus_path else corpus_v43
    if not corpus_path.is_file():
        raise SystemExit(f"retrieval index missing: {corpus_path}")
    search_engine = LocalSearchEngineV42(corpus_path=corpus_path, geosearch_url="")
    model = IntegratedModelV42(base=base_model, search_engine=search_engine)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model Parameters: {total_params:,} (247.56M Architecture Confirmed)")

    start_step = 0
    if target_ckpt and os.path.exists(target_ckpt):
        print(f"  Loading Weights from Checkpoint (CPU buffer): {target_ckpt}...")
        ckpt = torch.load(target_ckpt, map_location="cpu")
        sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        incompatible = model.load_state_dict(sd, strict=False)
        bad_missing = [k for k in incompatible.missing_keys if not k.startswith(("alu.", "ram."))]
        if bad_missing or incompatible.unexpected_keys:
            raise SystemExit(f"checkpoint/model mismatch: missing={bad_missing[:8]} unexpected={incompatible.unexpected_keys[:8]}")
        if isinstance(ckpt, dict) and "step" in ckpt:
            start_step = int(ckpt["step"])
            print(f"  -> Resuming from Step {start_step} successfully into v43 architecture.")
        else:
            print("  -> Warmstart weights loaded successfully into v43 architecture.")
    else:
        raise SystemExit(f"required warmstart checkpoint not found: {target_ckpt}")
    if args.required_start_step >= 0 and start_step != args.required_start_step:
        raise SystemExit(f"wrong warmstart step: got {start_step}, required {args.required_start_step}")

    model.to(dev)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_ckpt = str(target_ckpt)

    if not args.eval_only:
        use_amp = (dev.type == "cuda")
        gpu_name, free_gb = "cpu", 0.0
        if dev.type == "cuda":
            free_bytes, total_bytes = torch.cuda.mem_get_info(dev)
            free_gb = free_bytes / (1024**3)
            gpu_name = torch.cuda.get_device_name(dev)
            # what one step sees is fixed; only how it is split depends on the GPU (free memory picks the first chunk size)
            max_tokens_per_micro = DEFAULT_STEP_TOKENS
            chunk_start = DEFAULT_STEP_TOKENS if free_gb >= 26.0 else (24576 if free_gb >= 16.0 else
                                                                         max(4096, int(max(2.0, free_gb - 3.0) * 2048)))
        else:
            max_tokens_per_micro = 16384
            chunk_start = 16384
        grad_accum_steps = 1

        if args.max_tokens_per_micro:
            max_tokens_per_micro = int(args.max_tokens_per_micro)
        if args.step_tokens:
            max_tokens_per_micro = int(args.step_tokens)
        chunk_start = min(int(args.chunk_tokens) if args.chunk_tokens else chunk_start, max_tokens_per_micro)
        policy = ChunkPolicy(chunk_start)
        max_micro_batch = min(96, args.batch_size)
        print(f"  Execution Mode:  AMP bfloat16={use_amp} | Dynamic Token Budget: {max_tokens_per_micro:,} tok/micro (Max B={max_micro_batch}, Accum={grad_accum_steps})")
        print(f"  r5b GPU: {gpu_name} | free {free_gb:.1f} GB | step budget {max_tokens_per_micro:,} tokens (fixed) | "
              f"first chunk {chunk_start:,} tokens (OOM halves it per length; nothing is skipped)", flush=True)

        def write_train_meta(st: int):
            meta_out = {"gpu": gpu_name, "free_gb_at_start": round(free_gb, 2), "step_tokens": max_tokens_per_micro,
                        "start_step": start_step, "last_step": st, "skipped_sequences": 0,
                        "source_exposures": exposure_counts, "chunk": policy.describe()}
            (out_dir / f"train_meta_from_{start_step}.json").write_text(json.dumps(meta_out, indent=1), encoding="utf-8")

        if args.mask_tool_output:
            dataset = ToolMaskedDataset(args.train_path, vocab, max_len=cfg.max_positions)
            collate = masked_collate_fn
            if dataset.stats["spans"] == 0 or dataset.stats["rows_with_unaligned_spans_dropped"] > 0:
                # e.g. a stale v44b mixture (no loss_mask_spans): training would silently ignore the r2 change
                raise SystemExit(f"--mask_tool_output: unusable spans in {args.train_path}: {dataset.stats}")
        else:
            dataset = PreTokenizedMemoryDataset(args.train_path, vocab, max_len=cfg.max_positions)
            collate = zero_overhead_collate_fn
        if not 1 <= args.numina_repeat <= 4:
            raise RuntimeError("numina repeat must be 1..4")
        if args.numina_repeat > 1:
            if not isinstance(dataset, ToolMaskedDataset):
                raise RuntimeError("curriculum requires grouped masked dataset")
            numina_indices = [i for i, group in enumerate(dataset.groups) if group == "numina_tir"]
            if not numina_indices:
                raise RuntimeError("Numina rows absent from verified dataset")
            originals = [dataset.samples[i] for i in numina_indices]
            for _ in range(args.numina_repeat - 1):
                dataset.samples.extend(originals)
                dataset.groups.extend(["numina_tir"] * len(originals))
                dataset.origins.extend([dataset.origins[i] for i in numina_indices])
                dataset.supervised_counts.extend([dataset.supervised_counts[i] for i in numina_indices])
        print(f"[v46 curriculum] numina_presentation_multiplier={args.numina_repeat} "
              f"unique_numina={len(set(i for i,g in enumerate(dataset.groups) if g == 'numina_tir')) // args.numina_repeat}", flush=True)
        if not 1 <= args.new_curriculum_repeat <= 16:
            raise RuntimeError("new curriculum repeat must be 1..16")
        new_indices = [i for i, group in enumerate(dataset.groups) if group == "practical_multistep_v1"]
        if args.new_dataset_sampler:
            if not args.resume_state or not args.save_optimizer or not isinstance(dataset, ToolMaskedDataset):
                raise RuntimeError("new dataset phase requires masked data and exact model/Adam/RNG resume")
            if not new_indices and args.practical_variant != "replay_only":
                raise RuntimeError("explicit new dataset phase has no practical_multistep_v1 rows")
            if args.practical_variant == "replay_only" and (new_indices or args.new_curriculum_repeat != 1 or args.new_group_min_exposures != 0):
                raise RuntimeError("replay-only requires no new rows, repeat=1 and exposure floor=0")
        if args.new_curriculum_repeat > 1:
            if not new_indices:
                raise RuntimeError("new curriculum rows absent from verified dataset")
            originals = [dataset.samples[i] for i in new_indices]
            for _ in range(args.new_curriculum_repeat - 1):
                dataset.samples.extend(originals)
                dataset.groups.extend(["practical_multistep_v1"] * len(originals))
                dataset.origins.extend([dataset.origins[i] for i in new_indices])
                dataset.supervised_counts.extend([dataset.supervised_counts[i] for i in new_indices])
        print(f"[new curriculum] unique_rows={len(new_indices)} presentation_multiplier={args.new_curriculum_repeat}", flush=True)
        batch_sampler = ResumableTokenBucketSampler(
            dataset,
            max_batch_size=max_micro_batch,
            max_tokens=max_tokens_per_micro,
            megabucket_mult=32, seed=460006 + args.numina_repeat
        )
        loader_generator = torch.Generator().manual_seed(460006)
        dataloader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate,
            pin_memory=(dev.type == "cuda"), generator=loader_generator, num_workers=0
        )

        # The student is the entire original v43 model. A frozen base would make
        # this continuation a different experiment, so fail before spending GPU.
        named_student = list(model.named_parameters())
        circuit_parameters = [name for name, _ in named_student if name.startswith(("alu.", "ram."))]
        if circuit_parameters:
            raise RuntimeError(f"fixed ALU/RAM unexpectedly expose trainable parameters: {circuit_parameters[:8]}")
        previously_frozen = [name for name, p in named_student if not p.requires_grad]
        for _, parameter in named_student:
            parameter.requires_grad_(True)
        frozen_student = [name for name, p in named_student if not p.requires_grad]
        if frozen_student:
            raise RuntimeError(f"v46 requires full-model updates; frozen parameters: {frozen_student[:8]}")
        trainable_count = sum(p.numel() for _, p in named_student)
        total_count = sum(p.numel() for p in model.parameters())
        if trainable_count != total_count:
            raise RuntimeError(f"student parameter mismatch: {trainable_count} != {total_count}")
        print(f"[v46 full student] trainable={trainable_count} total={total_count} "
              f"newly_unfrozen={previously_frozen[:8]} count={len(previously_frozen)}", flush=True)
        use_fused = (dev.type == "cuda")
        try:
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01, fused=use_fused)
        except Exception:
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
        optimizer_count = sum(p.numel() for g in optimizer.param_groups for p in g["params"])
        if optimizer_count != total_count:
            raise RuntimeError(f"optimizer omits student parameters: {optimizer_count} != {total_count}")

        opt_file = out_dir / "optimizer_latest.pt"
        if args.resume_state and not opt_file.exists():
            raise RuntimeError("exact resume requires matching optimizer_latest.pt")
        opt_restored = False
        if args.save_optimizer and opt_file.exists():
            oc = torch.load(str(opt_file), map_location="cpu", weights_only=True)
            if int(oc.get("step", -1)) == start_step:
                optimizer.load_state_dict(oc["optimizer_state_dict"])
                opt_restored = True
                print(f"  -> AdamW state restored from {opt_file.name} (step {start_step}); no re-warmup.", flush=True)
            else:
                raise RuntimeError(f"optimizer step {oc.get('step')} != checkpoint step {start_step}")
            del oc
        phase_start = args.phase_start if args.phase_start >= 0 else start_step

        def get_lr(st: int) -> float:
            if args.phase_lr > 0:
                return phase_lr(st, start_step, phase_start, args.max_steps, args.phase_lr, args.phase_floor,
                                0 if opt_restored else args.phase_warmup)
            warmup = 300
            if st < warmup:
                return args.lr * (st + 1) / warmup
            prog = (st - warmup) / max(1, args.max_steps - warmup)
            return max(8e-6, args.lr * 0.5 * (1.0 + math.cos(math.pi * prog)))

        def save_optimizer(st: int):
            if args.save_optimizer:
                record = {"step": st, "optimizer_state_dict": optimizer.state_dict()}
                target = out_dir / f"optimizer_step_{st}.pt"
                tmp = target.with_suffix(target.suffix + ".tmp")
                torch.save(record, str(tmp))
                os.replace(tmp, target)
                latest_tmp = opt_file.with_suffix(".pt.tmp")
                try:
                    os.link(target, latest_tmp)
                except OSError:
                    shutil.copyfile(target, latest_tmp)
                os.replace(latest_tmp, opt_file)

        def save_train_state(st: int):
            ns = np.random.get_state()
            state = {"step": st, "sampler": batch_sampler.state_dict(),
                     "torch_cpu": torch.get_rng_state(),
                     "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                     "python": random.getstate(),
                     "numpy": (ns[0], ns[1].tolist(), ns[2], ns[3], ns[4]),
                     "loader_generator": loader_generator.get_state(),
                     "source_exposures": exposure_counts,
                     "practical_phase_sources": phase_sources,
                     "practical_supervised_tokens": phase_supervised,
                     "practical_replay_supervised_tokens": phase_replay_supervised,
                     "practical_variant": args.practical_variant,
                     "numina_repeat": args.numina_repeat,
                     "new_curriculum_repeat": args.new_curriculum_repeat,
                     "new_dataset_sampler_started": bool(args.new_dataset_sampler),
                     "cuda_devices": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                     "gpu_name": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
                     "chunk_policy": {"start": policy.start, "limit": policy.limit,
                                      "oom_retries": policy.oom_retries}}
            target = out_dir / f"train_state_step_{st}.pt"
            tmp = target.with_suffix(target.suffix + ".tmp")
            torch.save(state, str(tmp))
            os.replace(tmp, target)

        if args.phase_lr > 0:
            print(f"  v44b phase LR: peak {args.phase_lr:.2e} -> floor {args.phase_floor:.2e} over [{phase_start}, {args.max_steps}]"
                  f" | warmup {0 if opt_restored else args.phase_warmup} steps from {start_step} | first LR {get_lr(start_step + 1):.2e}", flush=True)

        teacher = None
        kl_log = []
        if args.distill_teacher:
            if not args.mask_tool_output:
                raise SystemExit("--distill_teacher needs --mask_tool_output (the masked dataset marks which rows are v43's)")
            teacher = IntegratedModelV42(base=GeoCPU_CED_v2(cfg), search_engine=search_engine)
            tck = torch.load(args.distill_teacher, map_location="cpu", weights_only=True)
            incompatible = teacher.load_state_dict(tck["model_state_dict"] if "model_state_dict" in tck else tck, strict=False)
            bad_missing = [k for k in incompatible.missing_keys if not k.startswith(("alu.", "ram."))]
            if bad_missing or incompatible.unexpected_keys:
                raise SystemExit(f"teacher/model mismatch: missing={bad_missing[:8]} unexpected={incompatible.unexpected_keys[:8]}")
            del tck
            teacher.to(dev).eval()
            for p_ in teacher.parameters():
                p_.requires_grad_(False)
            print(f"  v44b r4 distillation: teacher {args.distill_teacher} | weight {args.distill_weight} | "
                  f"T {args.distill_temperature} | applied to ALL replay rows; new practical rows excluded", flush=True)

        print(f"\nStarting Knowledge Co-Training from step {start_step} to {args.max_steps}...")
        step = start_step
        running_loss = 0.0
        log_interval = 10
        checkpoint_interval = 2500
        t_start = time.time()

        model.train()
        exposure_counts = {key: 0 for key in set(dataset.groups)}
        phase_sources = {}
        phase_source_tokens = {}
        phase_source_supervised = {}
        phase_supervised = 0
        phase_replay_supervised = 0
        phase_input_tokens = 0
        phase_padded_tokens = 0
        # Construct the loader iterator before restoring the recorded RNG: even
        # a zero-worker DataLoader may draw a private base seed in __iter__.
        data_iter = iter(dataloader)
        if args.resume_state:
            if not opt_restored:
                raise RuntimeError("exact resume requires restored Adam state")
            state = torch.load(args.resume_state, map_location="cpu", weights_only=True)
            if int(state["step"]) != start_step:
                raise RuntimeError("resume state/checkpoint step mismatch")
            if state["cuda_devices"] != (torch.cuda.device_count() if torch.cuda.is_available() else 0):
                raise RuntimeError("CUDA device count changed on exact resume")
            if state["gpu_name"] != (torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu"):
                raise RuntimeError("GPU model changed on exact resume")
            if args.new_dataset_sampler:
                if args.new_phase_sampler or int(state["numina_repeat"]) != args.numina_repeat:
                    raise RuntimeError("new dataset phase must retain the recorded Numina multiplier")
                if args.stop_at_step <= start_step or (args.new_group_min_exposures < 1000 and args.practical_variant != "replay_only"):
                    raise RuntimeError("new dataset phase needs a stop step and exposure floor >=1000")
                print("[new dataset] parent sampler cursor deliberately retired; new sampler begins at epoch 1", flush=True)
            elif args.new_phase_sampler:
                if int(state["numina_repeat"]) + 1 != args.numina_repeat:
                    raise RuntimeError("invalid curriculum phase transition")
            else:
                if int(state["numina_repeat"]) != args.numina_repeat:
                    raise RuntimeError("curriculum multiplier changed without phase transition")
                if int(state.get("new_curriculum_repeat", 1)) != args.new_curriculum_repeat:
                    raise RuntimeError("new curriculum multiplier changed without a new dataset phase")
                batch_sampler.load_state_dict(state["sampler"])
            exposure_counts = {str(k): int(v) for k, v in state["source_exposures"].items()}
            policy.start = int(state["chunk_policy"]["start"])
            policy.limit = state["chunk_policy"]["limit"]
            policy.oom_retries = int(state["chunk_policy"]["oom_retries"])
            random.setstate(state["python"])
            nps = state["numpy"]
            np.random.set_state((nps[0], np.asarray(nps[1], dtype=np.uint32), nps[2], nps[3], nps[4]))
            torch.set_rng_state(state["torch_cpu"])
            if torch.cuda.is_available():
                if len(state["torch_cuda"]) != torch.cuda.device_count():
                    raise RuntimeError("CUDA device count changed on exact resume")
                torch.cuda.set_rng_state_all(state["torch_cuda"])
            loader_generator.set_state(state["loader_generator"])
        planned_new_exposures = None
        new_group_start_exposures = exposure_counts.get("practical_multistep_v1", 0)
        if args.new_dataset_sampler:
            # Preview the private sampler without consuming the training sampler or global RNG.
            preview = ResumableTokenBucketSampler(
                dataset, max_batch_size=max_micro_batch, max_tokens=max_tokens_per_micro,
                megabucket_mult=32, seed=460006 + args.numina_repeat)
            iterator = iter(preview)
            planned = 0
            planned_sources = {}
            planned_supervised = 0
            planned_replay = 0
            for _ in range(args.stop_at_step - start_step):
                try:
                    indices = next(iterator)
                except StopIteration:
                    iterator = iter(preview)
                    indices = next(iterator)
                planned += sum(dataset.groups[index] == "practical_multistep_v1" for index in indices)
                for index in indices:
                    tokens = dataset.supervised_counts[index]
                    planned_supervised += tokens
                    planned_replay += tokens if dataset.groups[index] != "practical_multistep_v1" else 0
                    origin = dataset.origins[index]
                    planned_sources[origin] = planned_sources.get(origin, 0) + 1
            print(f"[new dataset] planned new-row presentations={planned} minimum={args.new_group_min_exposures}", flush=True)
            if planned < args.new_group_min_exposures:
                raise RuntimeError("appended curriculum exposure floor cannot be met")
            planned_new_exposures = planned
            planned_replay_share = planned_replay / max(1, planned_supervised)
            practical_telemetry(args.telemetry_jsonl, {"event": "preview", "new_presentations": planned,
                "source_presentations": planned_sources, "supervised_tokens": planned_supervised,
                "replay_supervised_tokens": planned_replay, "replay_supervised_share": planned_replay_share})
            if planned_replay_share < args.min_replay_share:
                raise RuntimeError("deterministic phase replay share is below 75%; no update performed")
        while step < args.max_steps:
            step += 1
            lr = get_lr(step)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
            accum_loss = step_with_retry(model, optimizer, batch, policy, dev, use_amp, teacher, args.distill_weight,
                                         args.distill_temperature, kl_log, step)

            if not math.isfinite(accum_loss):
                raise RuntimeError(f"nonfinite training loss at step {step}; optimizer update rejected")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            for idx in batch_sampler.last_batch:
                group = dataset.groups[idx]
                exposure_counts[group] = exposure_counts.get(group, 0) + 1
                origin = dataset.origins[idx]
                tokens = dataset.supervised_counts[idx]
                inputs = len(dataset.samples[idx][0]) - 1
                phase_sources[origin] = phase_sources.get(origin, 0) + 1
                phase_source_tokens[origin] = phase_source_tokens.get(origin, 0) + inputs
                phase_source_supervised[origin] = phase_source_supervised.get(origin, 0) + tokens
                phase_supervised += tokens
                phase_replay_supervised += tokens if group != "practical_multistep_v1" else 0
                phase_input_tokens += inputs
            phase_padded_tokens += int(batch["input_ids"].numel())

            running_loss += accum_loss

            if step % log_interval == 0 or step == args.stop_at_step:
                window_steps = min(log_interval, step - start_step) if step % log_interval == 0 else max(1, step % log_interval)
                avg_loss = running_loss / window_steps
                running_loss = 0.0
                elapsed = time.time() - t_start
                steps_per_sec = (step - start_step) / max(1.0, elapsed)
                eta_sec = (args.max_steps - step) / max(0.01, steps_per_sec)
                kl_txt = ""
                mean_kl = sum(kl_log) / len(kl_log) if kl_log else None
                if kl_log:
                    kl_txt = f" | KL(parent30500/replay): {sum(kl_log) / len(kl_log):.4f}"
                    kl_log.clear()
                ch_txt = f" | OOM retries {policy.oom_retries}" if policy.oom_retries else ""
                print(f"  Step {step:6d}/{args.max_steps} | Loss: {avg_loss:.4f}{kl_txt} | LR: {lr:.2e} | "
                      f"Throughput: {steps_per_sec:.2f} step/s | ETA: {eta_sec/60:.1f}m{ch_txt}", flush=True)
                practical_telemetry(args.telemetry_jsonl, {"event": "train_step", "step": step,
                    "start_step": start_step, "stop_step": args.stop_at_step, "loss": avg_loss, "kl": mean_kl,
                    "learning_rate": lr, "elapsed_seconds": elapsed, "eta_seconds": eta_sec,
                    "source_presentations": phase_sources, "source_input_tokens": phase_source_tokens,
                    "source_supervised_tokens": phase_source_supervised, "supervised_tokens": phase_supervised,
                    "replay_supervised_tokens": phase_replay_supervised, "input_tokens": phase_input_tokens,
                    "padded_tokens": phase_padded_tokens, "replay_supervised_share": phase_replay_supervised/max(1,phase_supervised),
                    "vram_allocated_bytes": torch.cuda.memory_allocated(dev) if dev.type == "cuda" else 0,
                    "vram_peak_bytes": torch.cuda.max_memory_allocated(dev) if dev.type == "cuda" else 0,
                    "oom_retries": policy.oom_retries, "variant": args.practical_variant})

            if args.new_dataset_sampler and step == args.stop_at_step:
                actual = exposure_counts.get("practical_multistep_v1", 0) - new_group_start_exposures
                print(f"[new dataset] actual new-row presentations={actual} planned={planned_new_exposures}", flush=True)
                if actual != planned_new_exposures or actual < args.new_group_min_exposures:
                    raise RuntimeError("new curriculum sampler exposure differed from the verified plan")
            if (step - phase_start) % checkpoint_interval == 0 or step == args.max_steps:
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
                save_optimizer(step)
                save_train_state(step)
                write_train_meta(step)
                final_ckpt = str(ckpt_file)

            if args.stop_at_step and step >= args.stop_at_step:
                if not (step % checkpoint_interval == 0 or step == args.max_steps):
                    ckpt_file = out_dir / f"checkpoint_step_{step}.pt"
                    print(f"  --> Saving gate checkpoint at step {step} to {ckpt_file}...", flush=True)
                    tmp_file = ckpt_file.with_suffix(".pt.tmp")
                    torch.save({"step": step, "model_state_dict": model.state_dict(), "config": cfg_dict, "meta": meta}, str(tmp_file))
                    os.replace(tmp_file, ckpt_file)
                    save_optimizer(step)
                    save_train_state(step)
                    final_ckpt = str(ckpt_file)
                write_train_meta(step)
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
