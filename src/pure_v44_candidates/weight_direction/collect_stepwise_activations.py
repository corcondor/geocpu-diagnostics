"""Read-only paired activation collection through the pure-v44 generation path.

This module never loads a checkpoint or writes weights.  A caller must pass an
already loaded, eval-mode model and practice-only paired continuations.  It
uses the same prefill/step_state path as TorchModelAdapter, forcing branch text
instead of sampling it.  Hidden states are measurements, not a model fix.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn

from weight_direction import PairedActivation, resolve_ffn_output


@dataclass(frozen=True)
class PairedContinuation:
    case_id: str
    source: str  # "train" or "practice"; never held-out evaluation
    prompt: str
    shared_prefix: str
    productive_suffix: str
    repetitive_suffix: str


@dataclass(frozen=True)
class BranchCapture:
    residual_mean: Tensor  # CPU float32, [d_model]
    ffn_input_mean: Tensor  # CPU float32, [ffn_dim]
    suffix_tokens: int
    site_calls: int
    tail_site_calls: int
    tail_window_tokens: int
    mean_forced_logprob: float


def _strict_ids(text: str, vocab: Mapping[str, int]) -> list[int]:
    if not text:
        raise ValueError("Empty text cannot be teacher-forced")
    missing = sorted(set(text) - set(vocab))
    if missing:
        raise ValueError("Text contains characters outside the pinned vocabulary: " + repr(missing[:8]))
    ids = [int(vocab[char]) for char in text]
    if any(token <= 3 for token in ids):
        raise ValueError("A special token cannot appear as a normal character")
    return ids


def _single_vector(value: Tensor, width: int, label: str) -> Tensor:
    if not isinstance(value, Tensor) or tuple(value.shape) != (1, 1, width):
        raise ValueError(f"{label} must have shape [1,1,{width}]")
    return value.detach().float().reshape(width).clone()


def capture_branch(
    model: nn.Module,
    vocab: Mapping[str, int],
    prefix: str,
    suffix: str,
    site: str,
    *,
    tail_window: int = 36,
    min_tail_calls: int = 4,
    expected_shape: tuple[int, int] = (768, 3072),
) -> BranchCapture:
    """Force suffix tokens one at a time and collect late FFN writer states.

    In the actual pure-v44 generator, TorchModelAdapter.prefill calls model.prefill
    with arm='learned', and advance calls model.step_state(compute_logits=True).
    Hooks are registered only after prefill, so prompt states cannot enter the
    branch mean.  A decoder FFN can be skipped on a non-boundary token; those
    missing calls are counted and never replaced by zero vectors.
    """
    if model.training:
        raise ValueError("Use a separate eval-mode model for read-only activation collection")
    if tail_window < 4 or min_tail_calls < 1 or min_tail_calls > tail_window:
        raise ValueError("Invalid tail-window or minimum-call setting")
    output = resolve_ffn_output(model, site, expected_shape=expected_shape)
    block = model.get_submodule(site)
    if not isinstance(block.norm2, nn.LayerNorm):
        raise TypeError("Expected the original GeoCPU block.norm2")
    d_model, ffn_dim = expected_shape
    prefix_ids, suffix_ids = _strict_ids(prefix, vocab), _strict_ids(suffix, vocab)
    max_positions = int(model.config.max_positions)
    if len(prefix_ids) + len(suffix_ids) > max_positions:
        raise ValueError("Paired continuation exceeds the model's configured context")
    if len(suffix_ids) > 2048:
        raise ValueError("Paired continuation exceeds the original generation budget")
    device = next(model.parameters()).device
    last_events: deque[tuple[int, Tensor, Tensor]] = deque(maxlen=tail_window)
    forced_logprobs: list[Tensor] = []
    site_calls = 0
    frame: dict[str, Tensor] = {}

    def before_norm(_module: nn.Module, inputs: tuple[Tensor, ...]) -> None:
        frame["pre_residual"] = _single_vector(inputs[0], d_model, "norm2 input")

    def before_output(_module: nn.Module, inputs: tuple[Tensor, ...]) -> None:
        frame["ffn_input"] = _single_vector(inputs[0], ffn_dim, "FFN output projection input")

    def after_output(_module: nn.Module, _inputs: tuple[Tensor, ...], result: Tensor) -> None:
        frame["ffn_output"] = _single_vector(result, d_model, "FFN output projection output")

    with torch.inference_mode():
        state = model.prefill(torch.tensor([prefix_ids], dtype=torch.long, device=device), arm="learned")
        if getattr(state, "iterations", None) != 1:
            raise ValueError("This collector requires the original single-iteration generation path")
        handles = (
            block.norm2.register_forward_pre_hook(before_norm),
            output.register_forward_pre_hook(before_output),
            output.register_forward_hook(after_output),
        )
        try:
            for index, token in enumerate(suffix_ids):
                frame.clear()
                logits = state.last_logits.reshape(-1, state.last_logits.shape[-1])[-1].float()
                forced_logprobs.append(torch.log_softmax(logits, dim=-1)[token].detach())
                state = model.step_state(
                    torch.tensor([[token]], dtype=torch.long, device=device),
                    state, compute_logits=True,
                )
                if not frame:
                    continue  # Main decoder skipped this non-boundary token.
                if set(frame) != {"pre_residual", "ffn_input", "ffn_output"}:
                    raise RuntimeError(f"Incomplete FFN hook capture at suffix token {index}")
                residual = frame["pre_residual"] + frame["ffn_output"]
                last_events.append((index, residual, frame["ffn_input"]))
                site_calls += 1
        finally:
            for handle in handles:
                handle.remove()
    start = max(0, len(suffix_ids) - tail_window)
    tail = [(residual, ffn_input) for index, residual, ffn_input in last_events if index >= start]
    if len(tail) < min_tail_calls:
        raise ValueError(
            f"Only {len(tail)} FFN calls in the last {min(tail_window, len(suffix_ids))} "
            f"suffix tokens at {site}; skipped calls are not zeros"
        )
    residual_mean = torch.stack([item[0] for item in tail]).mean(0).cpu()
    ffn_input_mean = torch.stack([item[1] for item in tail]).mean(0).cpu()
    mean_logprob = float(torch.stack(forced_logprobs).mean().cpu())
    if not torch.isfinite(residual_mean).all() or not torch.isfinite(ffn_input_mean).all():
        raise ValueError("Non-finite captured activation")
    return BranchCapture(
        residual_mean, ffn_input_mean, len(suffix_ids), site_calls,
        len(tail), min(tail_window, len(suffix_ids)), mean_logprob,
    )


def collect_discovery_pairs(
    model: nn.Module,
    vocab: Mapping[str, int],
    pairs: Sequence[PairedContinuation],
    site: str,
    *,
    tail_window: int = 36,
    min_tail_calls: int = 4,
    expected_shape: tuple[int, int] = (768, 3072),
) -> tuple[list[PairedActivation], list[dict]]:
    """Return original-model contrasts and explicit coverage diagnostics.

    The caller must independently verify that the productive branch is useful
    and the repetitive branch actually repeats.  No held-out IDs are accepted
    as discovery data; held-out cases remain for the later intervention gate.
    """
    if len(pairs) < 4 or len({row.case_id for row in pairs}) != len(pairs):
        raise ValueError("At least four distinct discovery pairs are required")
    if any(row.source not in {"train", "practice"} or not row.case_id for row in pairs):
        raise ValueError("Discovery pairs must be identified train/practice cases")
    results, diagnostics = [], []
    for row in pairs:
        prefix = row.prompt + row.shared_prefix
        if not row.prompt or not row.productive_suffix or not row.repetitive_suffix:
            raise ValueError(f"Empty prompt or continuation in {row.case_id}")
        if row.productive_suffix == row.repetitive_suffix:
            raise ValueError(f"Identical continuations in {row.case_id}")
        if "```output" in row.productive_suffix or "```output" in row.repetitive_suffix:
            raise ValueError("Tool observations cannot be treated as model-authored continuations")
        good = capture_branch(model, vocab, prefix, row.productive_suffix, site,
                              tail_window=tail_window, min_tail_calls=min_tail_calls,
                              expected_shape=expected_shape)
        bad = capture_branch(model, vocab, prefix, row.repetitive_suffix, site,
                             tail_window=tail_window, min_tail_calls=min_tail_calls,
                             expected_shape=expected_shape)
        results.append(PairedActivation(
            case_id=row.case_id, source=row.source,
            productive_residual=good.residual_mean,
            repetitive_residual=bad.residual_mean,
            productive_ffn_input=good.ffn_input_mean,
            repetitive_ffn_input=bad.ffn_input_mean,
        ))
        diagnostics.append({
            "case_id": row.case_id, "site": site,
            "shared_prefix_characters": len(prefix),
            "productive": {"suffix_tokens": good.suffix_tokens,
                           "site_calls": good.site_calls,
                           "tail_site_calls": good.tail_site_calls,
                           "tail_window_tokens": good.tail_window_tokens,
                           "mean_forced_logprob": good.mean_forced_logprob},
            "repetitive": {"suffix_tokens": bad.suffix_tokens,
                           "site_calls": bad.site_calls,
                           "tail_site_calls": bad.tail_site_calls,
                           "tail_window_tokens": bad.tail_window_tokens,
                           "mean_forced_logprob": bad.mean_forced_logprob},
        })
    return results, diagnostics
