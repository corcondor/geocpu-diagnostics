"""V44fix-only next-token mask for externally executed Python observations.

Tool text remains in the input sequence so the model can continue after it.
Only its target positions are excluded from cross-entropy.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from qwenputer_controller_v30 import canonical_normalize_text
from runtime_cotraining_v41_247M import zero_overhead_collate_fn


class ToolMaskedMemoryDataset(Dataset):
    def __init__(self, jsonl_path: str, vocab: dict[str, int], max_len: int = 2048):
        self.max_len = max_len
        self.samples = []
        with Path(jsonl_path).open(encoding="utf-8") as stream:
            for row_number, line in enumerate(stream, 1):
                item = json.loads(line)
                raw_prompt = item.get("prompt", "")
                raw_full = item.get("full_text") or raw_prompt + item.get("target", "")
                full_text = canonical_normalize_text(raw_full)
                prompt = canonical_normalize_text(raw_prompt)
                tokens = [vocab[c] for c in full_text if c in vocab]
                prompt_len = sum(c in vocab for c in prompt)
                if len(tokens) <= 1:
                    continue
                if item.get("bucket") == "tir":
                    if (raw_full != raw_prompt + item.get("target", "")
                            or len(tokens) != len(full_text)
                            or len(tokens) >= max_len):
                        raise ValueError(f"TIR row {row_number} changes character positions or loses EOS")
                    tool_ranges = self._tool_ranges(item, prompt_len, len(tokens), row_number)
                else:
                    tool_ranges = ()
                if len(tokens) < max_len:
                    tokens.append(3)  # The same native EOS ID as the v43 loader.
                else:
                    tokens = tokens[:max_len]
                self.samples.append((tokens, min(prompt_len, len(tokens) - 1), tool_ranges))

    @staticmethod
    def _tool_ranges(item: dict, prompt_len: int, full_len: int, row_number: int):
        segments = item.get("target_segments")
        if not isinstance(segments, list) or not segments:
            raise ValueError(f"TIR row {row_number} has no tool/assistant segments")
        cursor, ranges = 0, []
        target_len = len(item["target"])
        for segment in segments:
            if not isinstance(segment, dict):
                raise ValueError(f"TIR row {row_number} has an invalid segment")
            start, end = segment.get("start"), segment.get("end")
            role, train = segment.get("role"), segment.get("train")
            if (type(start) is not int or type(end) is not int or start != cursor
                    or not start < end <= target_len
                    or (role, train) not in (("assistant", True), ("tool", False))):
                raise ValueError(f"TIR row {row_number} has an invalid segment")
            if role == "tool":
                # targets[k] = tokens[k + 1], hence collate masks k = absolute - 1.
                absolute_start, absolute_end = prompt_len + start, prompt_len + end
                if not 0 < absolute_start < absolute_end <= full_len:
                    raise ValueError(f"TIR row {row_number} has an invalid tool range")
                ranges.append((absolute_start - 1, absolute_end - 1))
            cursor = end
        if cursor != target_len or not ranges:
            raise ValueError(f"TIR row {row_number} lacks a complete tool mask")
        return tuple(ranges)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def tool_aware_collate_fn(batch, pad_token_id: int = 0):
    packed = zero_overhead_collate_fn(
        [(tokens, prompt_len) for tokens, prompt_len, _ in batch], pad_token_id)
    mask = packed["loss_mask"]
    for row, (_, _, ranges) in enumerate(batch):
        for start, end in ranges:
            if end > mask.shape[1]:
                raise ValueError("Tool mask exceeds the packed sequence")
            mask[row, start:end] = 0.0
    return packed
