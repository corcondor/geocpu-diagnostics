"""Bounded, read-only candidate for GeoCPU pure-v44 FFN direction interventions.

This module does not load a checkpoint, train a model, or save weights.  Evidence
must come from discovery-only paired continuations collected by a separate,
read-only runner.  The published Qwen edit motivates the formula; its directions,
layer numbers, and column IDs are not used here.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
import re
from typing import Callable, Iterator, Mapping, Sequence

import torch
from torch import Tensor, nn


# The pinned pure-v44 architecture has 768 residual channels and 3,072 FFN
# intermediate channels.  A real-model intervention checks this shape again.
GEOCPU_FFN_SHAPE = (768, 3072)
SITE_PATTERN = re.compile(r"^(base\.encoder|base\.decoder|fine_decoder)\.(\d+)$")


@dataclass(frozen=True)
class PairedActivation:
    """Means over matched branch tails for one discovery prompt and one site.

    The two continuations must share the same prompt and prefix.  The caller
    supplies the means from the model's original, unedited weights.  `residual`
    is the block output after the FFN; `ffn_input` is the input to ffn[2].
    """

    case_id: str
    source: str  # "train" or "practice"; never a held-out evaluation item
    productive_residual: Tensor  # [d_model]
    repetitive_residual: Tensor  # [d_model]
    productive_ffn_input: Tensor  # [ffn_dim]
    repetitive_ffn_input: Tensor  # [ffn_dim]


@dataclass(frozen=True)
class Proposal:
    site: str
    direction: Tensor  # unit vector in GeoCPU residual space
    columns: tuple[int, ...]
    alpha: float
    weight_shape: tuple[int, int]
    discovery_ids: tuple[str, ...]
    scores: Tensor  # score for every FFN input column


def resolve_ffn_output(model: nn.Module, site: str, *, expected_shape=GEOCPU_FFN_SHAPE) -> nn.Linear:
    """Find only a known GeoCPU FFN output projection, checking its real shape."""
    match = SITE_PATTERN.fullmatch(site)
    if match is None:
        raise ValueError(f"Unsupported GeoCPU site: {site!r}")
    owner: nn.Module = model
    for part in match.group(1).split("."):
        owner = getattr(owner, part)
    index = int(match.group(2))
    if index >= len(owner):
        raise ValueError(f"Site index out of range: {site}")
    block = owner[index]
    if not isinstance(block.ffn, nn.Sequential) or len(block.ffn) != 3:
        raise TypeError(f"Expected GeoCPU three-part FFN at {site}")
    output = block.ffn[2]
    if not isinstance(output, nn.Linear):
        raise TypeError(f"Expected nn.Linear FFN output at {site}")
    if tuple(output.weight.shape) != tuple(expected_shape):
        raise ValueError(f"FFN shape at {site} is {tuple(output.weight.shape)}, expected {expected_shape}")
    return output


def propose_from_pairs(
    weight: Tensor,
    site: str,
    pairs: Sequence[PairedActivation],
    *,
    top_k: int,
) -> Proposal:
    """Estimate a repetition direction and rank columns from discovery pairs.

    The direction is the normalized mean of (repetitive - productive) residual
    differences.  Column j is scored by the mean over pairs of
    (repetitive - productive FFN input)_j * (u @ W[:, j]).  Only positive
    scores, meaning writing *toward* the measured direction, are eligible.
    """
    if weight.ndim != 2 or not torch.isfinite(weight).all():
        raise ValueError("weight must be a finite 2-D tensor")
    if len(pairs) < 4 or top_k < 1 or top_k > weight.shape[1]:
        raise ValueError("Need at least four discovery pairs and a valid top_k")
    ids = [p.case_id for p in pairs]
    if len(ids) != len(set(ids)) or any(not x for x in ids):
        raise ValueError("Discovery IDs must be unique and nonempty")
    if any(p.source not in ("train", "practice") for p in pairs):
        raise ValueError("Every direction pair must come from train/practice data")
    d, f = weight.shape
    residual_diffs = []
    input_diffs = []
    for pair in pairs:
        fields = (
            (pair.productive_residual, d), (pair.repetitive_residual, d),
            (pair.productive_ffn_input, f), (pair.repetitive_ffn_input, f),
        )
        if any(t.ndim != 1 or t.numel() != width or not torch.isfinite(t).all()
               for t, width in fields):
            raise ValueError(f"Invalid activation shape or value for {pair.case_id}")
        residual_diffs.append(pair.repetitive_residual.float() - pair.productive_residual.float())
        input_diffs.append(pair.repetitive_ffn_input.float() - pair.productive_ffn_input.float())
    contrast = torch.stack(residual_diffs).mean(0)
    norm = torch.linalg.vector_norm(contrast)
    if not torch.isfinite(norm) or norm <= 1e-8:
        raise ValueError("No measurable residual contrast")
    direction = (contrast / norm).to(device=weight.device)
    activation_delta = torch.stack(input_diffs).to(device=weight.device)
    writing = direction @ weight.detach().float()
    scores = (activation_delta * writing).mean(0)
    eligible = torch.nonzero(scores > 0, as_tuple=False).flatten()
    if len(eligible) < top_k:
        raise ValueError(f"Only {len(eligible)} columns write toward the measured direction")
    ranked = eligible[torch.argsort(scores[eligible], descending=True, stable=True)[:top_k]]
    return Proposal(site, direction.detach().clone(), tuple(int(i) for i in ranked),
                    1.0, (d, f), tuple(ids), scores.detach().cpu().clone())


def edited_weight(weight: Tensor, proposal: Proposal) -> Tensor:
    """Return a NEW tensor with W' = W - alpha*u*(mask*(u@W))."""
    if weight.ndim != 2 or tuple(weight.shape) != proposal.weight_shape:
        raise ValueError("Proposal and weight shape differ")
    if not torch.isfinite(weight).all() or not torch.isfinite(proposal.direction).all():
        raise ValueError("Non-finite input")
    if not (0 <= proposal.alpha <= 16):
        raise ValueError("Intervention dose must be between 0 and 16")
    u = proposal.direction.to(device=weight.device, dtype=torch.float32)
    if u.ndim != 1 or u.numel() != weight.shape[0] or not torch.isclose(u.norm(), u.new_tensor(1.0), atol=1e-5):
        raise ValueError("Direction must be a unit vector of residual width")
    columns = proposal.columns
    if not columns or len(set(columns)) != len(columns) or min(columns) < 0 or max(columns) >= weight.shape[1]:
        raise ValueError("Invalid selected columns")
    result = weight.detach().clone()
    if proposal.alpha == 0:
        return result
    selected = weight.detach().float()[:, list(columns)]
    delta = proposal.alpha * torch.outer(u, u @ selected)
    result[:, list(columns)] = (selected - delta).to(dtype=weight.dtype)
    return result


def orthogonal_null(weight: Tensor, proposal: Proposal) -> Proposal:
    """Deterministic orthogonal direction with matched float32 edit norm."""
    u = proposal.direction.to(device=weight.device, dtype=torch.float32)
    if u.numel() < 2:
        raise ValueError("Orthogonal control requires residual width >= 2")
    axis = int(torch.argmin(u.abs()))
    other = torch.zeros_like(u)
    other[axis] = 1
    other -= (other @ u) * u
    other /= other.norm()
    selected = weight.detach().float()[:, list(proposal.columns)]
    target_norm = proposal.alpha * (u @ selected).norm()
    null_norm = (other @ selected).norm()
    if null_norm <= 1e-8:
        raise ValueError("Cannot norm-match an orthogonal control for these columns")
    alpha = float(target_norm / null_norm)
    if alpha > 16:
        raise ValueError("Norm-matched null dose exceeds the bounded intervention range")
    return replace(proposal, direction=other.detach().clone(), alpha=alpha)


@contextmanager
def temporary_output_hook(output: nn.Linear, proposal: Proposal) -> Iterator[None]:
    """Apply the projected weight edit during inference without changing weights.

    For y = Wa+b, the hook adds (W'-W)a.  Removal in `finally` makes the
    intervention reversible even when generation raises an exception.
    """
    if tuple(output.weight.shape) != proposal.weight_shape:
        raise ValueError("Hook target shape differs from proposal")
    if output.training:
        raise ValueError("Call model.eval() before installing the inference hook")
    # Validate the proposal and obtain exactly the rounded delta that would be
    # produced in the module's native weight dtype.
    delta = (edited_weight(output.weight.detach(), proposal).float()
             - output.weight.detach().float()).detach()

    def hook(_module: nn.Module, inputs: tuple[Tensor, ...], result: Tensor) -> Tensor:
        if torch.is_grad_enabled():
            raise RuntimeError("The weight intervention requires torch.inference_mode()")
        correction = torch.nn.functional.linear(inputs[0].float(), delta)
        return (result.float() + correction).to(result.dtype)

    handle = output.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def assert_disjoint(discovery_ids: Sequence[str], heldout_ids: Sequence[str]) -> None:
    if not heldout_ids or len(set(heldout_ids)) != len(heldout_ids):
        raise ValueError("Held-out IDs must be nonempty and unique")
    overlap = set(discovery_ids) & set(heldout_ids)
    if overlap:
        raise ValueError(f"Discovery/held-out overlap: {sorted(overlap)[:4]}")


def repeated_suffix(text: str, *, minimum_period: int = 1, maximum_period: int = 64) -> dict:
    """Measure exact repeated character phrases at the end of generated text."""
    best = (0, 0, 0)  # span, period, repetitions
    for period in range(minimum_period, min(maximum_period, len(text) // 3) + 1):
        unit = text[-period:]
        count = 1
        while text[-(count + 1) * period:-count * period] == unit:
            count += 1
        if count >= 3 and count * period > best[0]:
            best = (count * period, period, count)
    return {"span_chars": best[0], "period_chars": best[1],
            "repetitions": best[2], "fraction_of_generated_chars": best[0] / max(1, len(text))}


def record_generation(case_id: str, prompt: str, reference: str,
                      generation: Mapping, correct: bool) -> dict:
    """Keep original answer and stop facts; score only model-generated text."""
    if not isinstance(correct, bool):
        raise TypeError("Task correctness must be supplied by a real task grader")
    required = ("output_text", "model_generated_text", "stop_reason", "ended_with_eos",
                "sampled_token_count", "tool_invocations")
    if any(k not in generation for k in required):
        raise ValueError("Missing original generator fields")
    if generation["sampled_token_count"] > 2048:
        raise ValueError("Generation exceeded the fixed 2,048-token cap")
    text = generation["model_generated_text"]
    return {"case_id": case_id, "prompt": prompt, "reference": reference,
            "raw_answer": generation["output_text"], "model_generated_text": text,
            "generated_tokens": generation.get("generated_tokens"),
            "stop_reason": generation["stop_reason"], "ended_with_eos": generation["ended_with_eos"],
            "sampled_token_count": generation["sampled_token_count"],
            "tool_invocations": generation["tool_invocations"], "error": generation.get("error"),
            "correct": correct,
            "repetition": repeated_suffix(text)}


def evaluate_heldout(
    model: nn.Module, proposal: Proposal, heldout_cases: Sequence[Mapping],
    generate: Callable[[nn.Module, str], Mapping],
    grade: Callable[[Mapping, Mapping], bool],
) -> list[dict]:
    """Optional bounded four-arm evaluation; the caller supplies the original generator.

    `generate` must call the unmodified pure-v44 generation method with identical
    prompt, vocabulary, decoding, and 2,048-token settings in every arm.  This
    function never loads or saves a model.  Do not run on held-out cases while
    choosing the site, columns, direction, or dose.
    """
    assert_disjoint(proposal.discovery_ids, [str(row["case_id"]) for row in heldout_cases])
    output = resolve_ffn_output(model, proposal.site)
    model.eval()
    null = orthogonal_null(output.weight.detach(), proposal)
    controls = (("baseline", None), ("zero_dose", replace(proposal, alpha=0.0)),
                ("orthogonal_null", null), ("candidate", proposal))
    records = []
    with torch.inference_mode():
        for case in heldout_cases:
            arms = {}
            for label, candidate in controls:
                if candidate is None:
                    result = generate(model, case["prompt"])
                else:
                    with temporary_output_hook(output, candidate):
                        result = generate(model, case["prompt"])
                arms[label] = record_generation(str(case["case_id"]), case["prompt"],
                                                case["reference"], result, grade(case, result))
            baseline = arms["baseline"]
            zero = arms["zero_dose"]
            for field in ("raw_answer", "model_generated_text", "generated_tokens",
                          "stop_reason", "ended_with_eos", "sampled_token_count",
                          "tool_invocations", "correct"):
                if baseline[field] != zero[field]:
                    raise RuntimeError(f"Zero-dose hook changed {field} on {case['case_id']}")
            records.append({"case_id": case["case_id"], "arms": arms})
    return records
