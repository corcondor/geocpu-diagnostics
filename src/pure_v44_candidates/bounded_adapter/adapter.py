"""Reversible low-rank residuals for a strictly loaded v43 neural parent.

This module contains no training loop. The caller must first load the pinned
v43 weights into the pure-Python neural model, then install the adapters.
Only the residual matrices are trainable; the parent weights stay unchanged.
"""

from __future__ import annotations

import copy
import hashlib
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4

import torch
from torch import nn


SCHEMA = "geocpu_v43_parent_bounded_python_adapter_v1"


@dataclass(frozen=True)
class AdapterSpec:
    rank: int = 8
    alpha: float = 8.0
    include_fine_decoder: bool = True

    def validate(self) -> None:
        if not isinstance(self.rank, int) or self.rank <= 0:
            raise ValueError("rank must be positive")
        if not isinstance(self.alpha, (int, float)) or not 0 < self.alpha <= 1024:
            raise ValueError("alpha must be finite and positive")


class LowRankResidualLinear(nn.Module):
    """W_parent x + alpha/rank * B(Ax), with B initially zero."""

    def __init__(self, parent: nn.Linear, *, rank: int, alpha: float):
        super().__init__()
        if not isinstance(parent, nn.Linear):
            raise TypeError("Only nn.Linear is supported")
        if rank > min(parent.in_features, parent.out_features):
            raise ValueError("rank exceeds target dimensions")
        self.parent = parent
        self.parent.requires_grad_(False)
        self.down = nn.Linear(parent.in_features, rank, bias=False,
                              device=parent.weight.device, dtype=parent.weight.dtype)
        self.up = nn.Linear(rank, parent.out_features, bias=False,
                            device=parent.weight.device, dtype=parent.weight.dtype)
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        nn.init.zeros_(self.up.weight)
        self.scale = float(alpha) / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.parent(x) + self.scale * self.up(self.down(x))


def _target_names(model: nn.Module, spec: AdapterSpec) -> tuple[str, ...]:
    spec.validate()
    if not isinstance(model.get_submodule("base.head"), nn.Linear):
        raise TypeError("Expected plain base.head before adapter installation")
    names = ["base.head"]
    if spec.include_fine_decoder:
        decoder = model.get_submodule("fine_decoder")
        if not isinstance(decoder, nn.ModuleList) or not decoder:
            raise TypeError("Expected a nonempty fine_decoder ModuleList")
        names.extend(f"fine_decoder.{i}.ffn.2" for i in range(len(decoder)))
    for name in names:
        if not isinstance(model.get_submodule(name), nn.Linear):
            raise TypeError(f"Expected plain linear target: {name}")
    return tuple(names)


def install_adapters(model: nn.Module, spec: AdapterSpec) -> tuple[str, ...]:
    """Freeze the loaded parent and install exact-identity residuals.

    Initial outputs are unchanged because every B matrix starts at zero.
    The same wrapped head is used by the verified training and generation AST.
    """
    names = _target_names(model, spec)
    model.requires_grad_(False)
    for name in names:
        parent_name, leaf = name.rsplit(".", 1)
        container = model.get_submodule(parent_name)
        old = model.get_submodule(name)
        new = LowRankResidualLinear(old, rank=spec.rank, alpha=spec.alpha)
        if isinstance(container, (nn.Sequential, nn.ModuleList)):
            container[int(leaf)] = new
        else:
            setattr(container, leaf, new)
    assert_only_adapters_trainable(model, names)
    return names


def assert_only_adapters_trainable(model: nn.Module, names: tuple[str, ...]) -> None:
    allowed = {f"{name}.{part}.weight" for name in names for part in ("down", "up")}
    found = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if found != allowed:
        raise AssertionError(f"Unexpected trainable parameters: {sorted(found ^ allowed)}")


def adapter_parameters(model: nn.Module, names: tuple[str, ...]):
    assert_only_adapters_trainable(model, names)
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def adapter_state(model: nn.Module, names: tuple[str, ...]) -> dict[str, torch.Tensor]:
    assert_only_adapters_trainable(model, names)
    return {f"{name}.{part}.weight": getattr(model.get_submodule(name), part).weight.detach().cpu().clone()
            for name in names for part in ("down", "up")}


def _validate_adapter_state(model: nn.Module, names: tuple[str, ...], state: dict) -> None:
    expected = adapter_state(model, names)
    if set(state) != set(expected):
        raise ValueError("Adapter keys differ")
    for key, value in state.items():
        if not isinstance(value, torch.Tensor) or value.shape != expected[key].shape or value.dtype != expected[key].dtype:
            raise ValueError(f"Adapter tensor shape or dtype differs: {key}")


def load_adapter_state(model: nn.Module, names: tuple[str, ...], state: dict) -> None:
    _validate_adapter_state(model, names, state)
    with torch.no_grad():
        for name in names:
            module = model.get_submodule(name)
            for part in ("down", "up"):
                target = getattr(module, part).weight
                target.copy_(state[f"{name}.{part}.weight"].to(target.device))


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save_adapter_checkpoint(path: str | Path, *, model: nn.Module, names: tuple[str, ...],
                            spec: AdapterSpec, parent_sha256: str, train_manifest_sha256: str,
                            optimizer: torch.optim.Optimizer, step: int, rng: dict | None = None) -> str:
    """Save adapter and optimizer only; never rewrite the parent checkpoint."""
    if len(parent_sha256) != 64 or len(train_manifest_sha256) != 64 or step < 0:
        raise ValueError("Missing lineage or invalid step")
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema": SCHEMA, "parent_sha256": parent_sha256,
               "train_manifest_sha256": train_manifest_sha256, "step": step,
               "spec": asdict(spec), "target_names": names,
               "adapter_state_dict": adapter_state(model, names),
               "optimizer_state_dict": optimizer.state_dict(), "rng": rng or {}}
    temporary = path.with_name(path.name + "." + uuid4().hex + ".partial")
    try:
        with temporary.open("xb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        # A corrupt or truncated write must be rejected before publication.
        saved = torch.load(temporary, map_location="cpu", weights_only=True)
        if saved["schema"] != SCHEMA or saved["step"] != step:
            raise RuntimeError("Adapter checkpoint readback failed")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return file_sha256(path)


def resume_adapter_checkpoint(path: str | Path, *, expected_sha256: str,
                              expected_parent_sha256: str, expected_train_manifest_sha256: str,
                              model: nn.Module, names: tuple[str, ...], spec: AdapterSpec,
                              optimizer: torch.optim.Optimizer) -> dict:
    path = Path(path)
    if file_sha256(path) != expected_sha256:
        raise ValueError("Adapter checkpoint SHA differs")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != SCHEMA or payload.get("parent_sha256") != expected_parent_sha256:
        raise ValueError("Parent lineage differs")
    if payload.get("train_manifest_sha256") != expected_train_manifest_sha256:
        raise ValueError("Training data lineage differs")
    if payload.get("spec") != asdict(spec) or tuple(payload.get("target_names", ())) != names:
        raise ValueError("Adapter architecture differs")
    _validate_adapter_state(model, names, payload["adapter_state_dict"])
    before = adapter_state(model, names)
    before_optimizer = copy.deepcopy(optimizer.state_dict())
    try:
        load_adapter_state(model, names, payload["adapter_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    except BaseException:
        load_adapter_state(model, names, before)
        optimizer.load_state_dict(before_optimizer)
        raise
    return {"step": payload["step"], "rng": payload["rng"], "checkpoint_sha256": expected_sha256}
