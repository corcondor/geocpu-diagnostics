"""Token-exact v43/Python teaching masks and staged exposure plans.

This module deliberately has no model or GPU side effects.  The original v43
collator predicts token k+1 at input position k, supervises from the final
prompt token, and includes a native EOS when the sequence fits 2048.  For
Python rows, only externally supplied tool observations are removed from that
same target mask.  Printed Python code and Japanese explanation stay active.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import random
import re
from typing import Callable, Mapping, Sequence


MAX_POSITIONS = 2048
STAGE_PYTHON_CAPS = (0.01, 0.02, 0.05, 0.10)


@dataclass(frozen=True)
class EncodedExample:
    tokens: tuple[int, ...]
    prompt_length: int
    tool_target_ranges: tuple[tuple[int, int], ...]
    active_positions: int
    kind: str
    row_sha256: str
    source_family: str
    japanese_target: bool

    @property
    def sequence_length(self) -> int:
        return len(self.tokens)

    def target_mask(self) -> tuple[bool, ...]:
        active = [False] * (len(self.tokens) - 1)
        active[max(0, self.prompt_length - 1):] = [True] * (
            len(active) - max(0, self.prompt_length - 1))
        for start, end in self.tool_target_ranges:
            active[start:end] = [False] * (end - start)
        return tuple(active)


def _raw_row_sha(row: Mapping) -> str:
    return sha256(json.dumps(row, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":")).encode("utf-8")).hexdigest()


def encode_example(
    row: Mapping,
    vocab: Mapping[str, int],
    *,
    kind: str,
    normalize: Callable[[str], str] = lambda value: value,
    max_positions: int = MAX_POSITIONS,
) -> EncodedExample:
    if kind not in ("v43", "python"):
        raise ValueError("kind must be v43 or python")
    if max_positions != MAX_POSITIONS:
        raise ValueError("The pinned v43 maximum context is 2048")
    raw_prompt = row.get("prompt", "")
    raw_full = row.get("full_text") or raw_prompt + row.get("target", "")
    if not isinstance(raw_prompt, str) or not isinstance(raw_full, str):
        raise ValueError("prompt/full_text must be strings")
    prompt, full = normalize(raw_prompt), normalize(raw_full)
    tokens = [vocab[char] for char in full if char in vocab]
    prompt_length = sum(char in vocab for char in prompt)
    if len(tokens) <= 1:
        raise ValueError("No supervised target tokens")
    if kind == "python":
        target = row.get("target")
        if (not isinstance(target, str) or prompt_length >= len(tokens)
                or raw_full != raw_prompt + target
                or full != raw_full or prompt != raw_prompt
                or len(tokens) != len(full) or len(tokens) >= max_positions):
            raise ValueError("Python row changes offsets, loses vocabulary, or has no room for EOS")
    if len(tokens) < max_positions:
        tokens.append(3)  # Original v43 native EOS ID.
    else:
        tokens = tokens[:max_positions]
    prompt_length = min(prompt_length, len(tokens) - 1)
    active = [False] * (len(tokens) - 1)
    # Exactly zero_overhead_collate_fn: targets[k] = tokens[k + 1].
    for index in range(max(0, prompt_length - 1), len(tokens) - 1):
        active[index] = True
    tool_ranges: list[tuple[int, int]] = []
    if kind == "python":
        spans = row.get("target_segments")
        if not isinstance(spans, list) or not spans:
            raise ValueError("Python row requires verified tool observation ranges")
        previous_end = 0
        tool_count = 0
        for span in spans:
            if not isinstance(span, dict):
                raise ValueError("Invalid tool segment")
            start, end = span.get("start"), span.get("end")
            if (type(start) is not int or type(end) is not int
                    or not previous_end <= start < end <= len(row["target"])
                    or (span.get("role"), span.get("train")) not in
                    (("tool", False), ("assistant", True))):
                raise ValueError("Invalid or overlapping Python tool segment")
            if span["role"] == "tool":
                absolute_start, absolute_end = prompt_length + start, prompt_length + end
                # Slice labels at k+1, not input tokens at k.
                mask_start, mask_end = absolute_start - 1, absolute_end - 1
                if not 0 <= mask_start < mask_end <= len(active):
                    raise ValueError("Tool observation falls outside the target mask")
                active[mask_start:mask_end] = [False] * (mask_end - mask_start)
                tool_ranges.append((mask_start, mask_end))
                tool_count += 1
            previous_end = end
        if not tool_count or not any(active) or not active[-1]:
            raise ValueError("Python row lost assistant labels or true EOS")
    count = sum(active)
    if not count:
        raise ValueError("No supervised target positions remain")
    family = row.get("original_source") if kind == "python" else row.get("source")
    return EncodedExample(tuple(tokens), prompt_length, tuple(tool_ranges),
                          count, kind, _raw_row_sha(row),
                          str(family) if family is not None else "unclassified",
                          bool(re.search(r"[ぁ-んァ-ン一-龯]", row.get("target", ""))))


def collate_exact(batch: Sequence[EncodedExample]):
    """Return original four v43 tensors with Python tool targets set to zero."""
    import torch
    if not batch:
        raise ValueError("Empty batch")
    aligned = max(32, ((max(len(row.tokens) for row in batch) + 31) // 32) * 32)
    b = len(batch)
    inp = torch.zeros((b, aligned), dtype=torch.long)
    tgt = torch.zeros((b, aligned), dtype=torch.long)
    valid = torch.zeros((b, aligned), dtype=torch.bool)
    mask = torch.zeros((b, aligned), dtype=torch.float32)
    for row_id, row in enumerate(batch):
        n = len(row.tokens)
        values = torch.tensor(row.tokens, dtype=torch.long)
        inp[row_id, :n-1] = values[:-1]
        tgt[row_id, :n-1] = values[1:]
        valid[row_id, :n-1] = True
        mask[row_id, max(0, row.prompt_length-1):n-1] = 1.0
        for start, end in row.tool_target_ranges:
            mask[row_id, start:end] = 0.0
        if int(mask[row_id].sum().item()) != row.active_positions:
            raise AssertionError("Actual target mask differs from preflight token count")
    return {"input_ids": inp, "targets": tgt, "valid_mask": valid,
            "loss_mask": mask}


@dataclass(frozen=True)
class ExposurePlan:
    cap: float
    batches: tuple[tuple[tuple[str, int], ...], ...]
    paired_old_batches: tuple[tuple[int, ...], ...]
    old_active_positions: int
    python_active_positions: int
    seed: int

    @property
    def measured_fraction(self) -> float:
        return self.python_active_positions / (self.python_active_positions + self.old_active_positions)

    def as_json(self) -> dict:
        return {"schema": "geocpu_v43_python_actual_target_plan_v1", "cap": self.cap,
                "seed": self.seed, "batches": self.batches,
                "paired_old_batches": self.paired_old_batches,
                "old_active_positions": self.old_active_positions,
                "python_active_positions": self.python_active_positions,
                "measured_fraction": self.measured_fraction}


def plan_exposure(old: Sequence[EncodedExample], python: Sequence[EncodedExample],
                  *, cap: float, updates: int, batch_size: int, seed: int,
                  min_fraction_factor: float = 0.70) -> ExposurePlan:
    """Predeclare all selected rows and bound Python's actual active labels.

    A stage cannot begin when its complete token-level schedule misses its
    declared exposure interval.  No row-count proxy or after-the-fact tuning.
    """
    if cap not in (0.0, *STAGE_PYTHON_CAPS) or updates < 1 or batch_size < 1:
        raise ValueError("Unapproved stage fraction or invalid schedule size")
    if not old or not python or any(row.kind != "v43" for row in old) or any(row.kind != "python" for row in python):
        raise ValueError("Both verified source pools are required")
    old_rng = random.Random(seed)
    python_rng = random.Random(seed ^ 0x5A17C0DE)
    old_ids, py_ids = list(range(len(old))), list(range(len(python)))
    old_rng.shuffle(old_ids)
    python_rng.shuffle(py_ids)
    oi = pi = old_active = py_active = 0
    batches: list[tuple[tuple[str, int], ...]] = []
    paired_old_batches: list[tuple[int, ...]] = []
    for _ in range(updates):
        chosen: list[tuple[str, int]] = []
        paired_old: list[int] = []
        for _ in range(batch_size):
            if oi >= len(old_ids):
                old_rng.shuffle(old_ids)
                oi = 0
            if pi >= len(py_ids):
                python_rng.shuffle(py_ids)
                pi = 0
            old_index = old_ids[oi]
            oi += 1  # Both arms consume this exact old slot even if Python replaces it.
            paired_old.append(old_index)
            old_count = old[old_index].active_positions
            py_count = python[py_ids[pi]].active_positions
            # Select Python only when it moves the cumulative target share
            # closer to the fixed cap without crossing it.
            old_gap = abs(cap - py_active / (old_active + old_count + py_active))
            py_fraction = (py_active + py_count) / (old_active + py_active + py_count)
            choose_python = py_fraction <= cap and abs(cap - py_fraction) < old_gap
            if choose_python:
                chosen.append(("python", py_ids[pi]))
                pi += 1
                py_active += py_count
            else:
                chosen.append(("v43", old_index))
                old_active += old_count
        batches.append(tuple(chosen))
        paired_old_batches.append(tuple(paired_old))
    plan = ExposurePlan(cap, tuple(batches), tuple(paired_old_batches), old_active, py_active, seed)
    if not cap * min_fraction_factor <= plan.measured_fraction <= cap:
        raise ValueError("Stage does not deliver its declared actual target-position share")
    return plan


def verify_realized_update(plan: ExposurePlan, step: int, rows: Sequence[EncodedExample],
                           packed) -> tuple[int, int]:
    """Check the live GPU-bound mask before a gradient step can be accepted."""
    selected = plan.batches[step]
    if len(selected) != len(rows):
        raise ValueError("Update differs from the admitted schedule")
    old_count = py_count = 0
    for (kind, index), row, mask in zip(selected, rows, packed["loss_mask"]):
        if row.kind != kind or row.active_positions != int(mask.sum().item()):
            raise ValueError("Live supervised target positions differ from schedule")
        if kind == "python":
            py_count += row.active_positions
        else:
            old_count += row.active_positions
    if old_count + py_count == 0:
        raise ValueError("Empty supervised update")
    return old_count, py_count
