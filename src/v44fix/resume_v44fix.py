"""Fail-closed, exact-boundary checkpointing for the isolated v44fix branch.

The v43 step-15000 file is a pinned *warm start*: its optimizer and data order
are not claimed to continue. Checkpoints produced here contain the complete
state needed to resume this branch at the next optimizer update.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
import uuid

import numpy as np
import torch


PARENT_SHA256 = "a4b95d7d0f486146196271b3b1fbc6a1b6b3342d8fadd590320fc3842c1e80e5"
PARENT_BYTES = 990545719
PARENT_STEP = 15000
SCHEMA = "geocpu_v44fix_exact_resume_v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def training_binding(train_path: str | Path, meta_path: str | Path, root: str | Path,
                     *, max_steps: int, batch_size: int, lr: float,
                     max_tokens_per_micro: int, grad_accum_steps: int) -> dict:
    root = Path(root)
    source_names = ("runtime_v44fix.py", "resume_v44fix.py", "tir_masking_v44fix.py",
                    "runtime_cotraining_v41_247M.py", "integrated_model_v42.py",
                    "integrated_model_v30.py", "GeoCPU_CED_v1.py", "GeoCPU_CED_v2.py",
                    "fast_encoder_scan.py", "qwenputer_controller_v30.py",
                    "cpu_circuit_v30.py", "local_search_engine_v42.py",
                    "official_dc_reference_v22.py", "tir_engine_v44r1.py",
                    "executor_worker_v44r1.py", "resource_limited_worker_v44r1.py")
    return {"schema": SCHEMA, "parent_sha256": PARENT_SHA256,
            "train_sha256": sha256_file(train_path), "meta_sha256": sha256_file(meta_path),
            "source_sha256": {name: sha256_file(root / name) for name in source_names},
            "max_steps": int(max_steps), "batch_size": int(batch_size), "lr": float(lr),
            "max_tokens_per_micro": int(max_tokens_per_micro),
            "grad_accum_steps": int(grad_accum_steps), "torch_version": str(torch.__version__),
            "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
                            if torch.cuda.is_available() else []}


def capture_rng() -> dict:
    numpy = np.random.get_state()
    return {"torch_cpu": torch.get_rng_state().clone(),
            "torch_cuda": [state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
            "python": random.getstate(),
            "numpy": (numpy[0], numpy[1].tolist(), int(numpy[2]), int(numpy[3]), float(numpy[4]))}


def restore_rng(state: dict) -> None:
    required = {"torch_cpu", "torch_cuda", "python", "numpy"}
    if not isinstance(state, dict) or set(state) != required:
        raise ValueError("Checkpoint RNG state is incomplete")
    cuda = state["torch_cuda"]
    if torch.cuda.is_available() and len(cuda) != torch.cuda.device_count():
        raise ValueError("Checkpoint CUDA RNG device count differs")
    if not torch.cuda.is_available() and cuda:
        raise ValueError("Checkpoint CUDA RNG requires CUDA")
    np_state = state["numpy"]
    if not isinstance(np_state, (tuple, list)) or len(np_state) != 5:
        raise ValueError("Checkpoint NumPy RNG state is invalid")
    torch.set_rng_state(state["torch_cpu"])
    if cuda:
        torch.cuda.set_rng_state_all(cuda)
    random.setstate(tuple(state["python"]))
    np.random.set_state((np_state[0], np.array(np_state[1], dtype=np.uint32),
                         int(np_state[2]), int(np_state[3]), float(np_state[4])))


class ResumableTokenBucketBatchSampler:
    """Original megabucket policy, with its exact pending batches/cursor saved."""
    def __init__(self, dataset, max_batch_size: int = 96, max_tokens: int = 24576,
                 megabucket_mult: int = 32):
        self.dataset = dataset
        self.max_batch_size = int(max_batch_size)
        self.max_tokens = int(max_tokens)
        self.megabucket_size = self.max_batch_size * int(megabucket_mult)
        self.lengths = [len(sample[0]) for sample in dataset.samples]
        self.lengths_sha256 = hashlib.sha256(json.dumps(self.lengths, separators=(",", ":")).encode()).hexdigest()
        self._cached_batches = self._build_batches(shuffle=False)
        self.batches = self._build_batches(shuffle=True)
        self.cursor = 0
        self.epoch = 0
        self._resume_pending_exhaustion = False

    def _build_batches(self, shuffle: bool = True):
        indices = list(range(len(self.dataset)))
        if shuffle:
            random.shuffle(indices)
        batches = []
        for m_start in range(0, len(indices), self.megabucket_size):
            mega = indices[m_start:m_start + self.megabucket_size]
            mega.sort(key=lambda index: self.lengths[index])
            pos = 0
            while pos < len(mega):
                candidate_end = min(len(mega), pos + self.max_batch_size)
                max_length = self.lengths[mega[candidate_end - 1]]
                aligned_max = max(32, int(math.ceil(max_length / 32.0)) * 32)
                safe_size = min(self.max_batch_size, max(8, (self.max_tokens // aligned_max // 8) * 8))
                end = min(len(mega), pos + safe_size)
                batch = mega[pos:end]
                if len(batch) >= 4:
                    batches.append(batch)
                pos = end
        if shuffle:
            random.shuffle(batches)
        return batches

    def __iter__(self):
        if self._resume_pending_exhaustion:
            # The uninterrupted DataLoader still has to observe StopIteration
            # from its old iterator before constructing the next one. Preserve
            # that extra iterator construction (and its torch RNG consumption).
            self._resume_pending_exhaustion = False
            return
        if self.cursor >= len(self.batches):
            self.batches = self._build_batches(shuffle=True)
            self.cursor = 0
            self.epoch += 1
        while self.cursor < len(self.batches):
            batch = self.batches[self.cursor]
            self.cursor += 1  # Advance before yielding so a saved boundary is exact.
            yield batch

    def __len__(self):
        return len(self._cached_batches)

    def state_dict(self) -> dict:
        return {"schema": SCHEMA, "lengths_sha256": self.lengths_sha256,
                "dataset_count": len(self.dataset), "max_batch_size": self.max_batch_size,
                "max_tokens": self.max_tokens, "megabucket_size": self.megabucket_size,
                "epoch": self.epoch, "cursor": self.cursor,
                "batches": [list(batch) for batch in self.batches]}

    def load_state_dict(self, state: dict) -> None:
        expected = (SCHEMA, self.lengths_sha256, len(self.dataset), self.max_batch_size,
                    self.max_tokens, self.megabucket_size)
        actual = tuple(state.get(key) for key in ("schema", "lengths_sha256", "dataset_count",
                                               "max_batch_size", "max_tokens", "megabucket_size"))
        if actual != expected:
            raise ValueError("Checkpoint sampler parameters or dataset lengths differ")
        batches, cursor, epoch = state.get("batches"), state.get("cursor"), state.get("epoch")
        if (not isinstance(batches, list) or not batches or type(cursor) is not int
                or not 0 <= cursor <= len(batches) or type(epoch) is not int or epoch < 0
                or any(not isinstance(batch, list) or len(batch) < 4 or
                       any(type(index) is not int or not 0 <= index < len(self.dataset) for index in batch)
                       for batch in batches)):
            raise ValueError("Checkpoint sampler cursor or batches are invalid")
        self.batches, self.cursor, self.epoch = batches, cursor, epoch
        self._resume_pending_exhaustion = cursor == len(batches)


def receipt_path(path: str | Path) -> Path:
    path = Path(path)
    return path.with_name(path.name + ".receipt.json")


def verify_checkpoint(path: str | Path, expected_binding: dict | None = None,
                      expected_step: int | None = None) -> dict:
    path = Path(path)
    receipt_file = receipt_path(path)
    if not path.is_file() or not receipt_file.is_file():
        raise FileNotFoundError("Complete checkpoint and receipt are required: " + str(path))
    receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
    if (receipt.get("schema") != SCHEMA or type(receipt.get("step")) is not int
            or receipt.get("bytes") != path.stat().st_size or
            receipt.get("sha256") != sha256_file(path)):
        raise ValueError("Checkpoint receipt integrity failed: " + str(path))
    if expected_step is not None and receipt["step"] != expected_step:
        raise ValueError("Checkpoint step differs")
    if expected_binding is not None and receipt.get("binding") != expected_binding:
        raise ValueError("Checkpoint training binding differs")
    return receipt


def save_checkpoint(path: str | Path, *, step: int, model, optimizer, sampler,
                    binding: dict, config: dict, meta: dict) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    receipt_file = receipt_path(path)
    if path.exists() or receipt_file.exists():
        raise FileExistsError("Refusing to overwrite a checkpoint or receipt: " + str(path))
    payload = {"schema": SCHEMA, "step": int(step), "model_state_dict": model.state_dict(),
               "optimizer_state_dict": optimizer.state_dict(), "sampler_state_dict": sampler.state_dict(),
               "rng_state_dict": capture_rng(), "binding": binding, "config": config, "meta": meta}
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".partial")
    temp_receipt = receipt_file.with_name(receipt_file.name + "." + uuid.uuid4().hex + ".partial")
    with temp.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    receipt = {"schema": SCHEMA, "step": int(step), "bytes": temp.stat().st_size,
               "sha256": sha256_file(temp), "binding": binding}
    with temp_receipt.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    os.replace(temp_receipt, receipt_file)
    verify_checkpoint(path, binding, step)
    return receipt


def load_checkpoint(path: str | Path, *, expected_binding: dict,
                    model, optimizer, sampler) -> tuple[int, dict]:
    receipt = verify_checkpoint(path, expected_binding)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (checkpoint.get("schema") != SCHEMA or checkpoint.get("step") != receipt["step"]
            or checkpoint.get("binding") != expected_binding
            or not all(key in checkpoint for key in
                       ("model_state_dict", "optimizer_state_dict", "sampler_state_dict", "rng_state_dict"))):
        raise ValueError("Checkpoint state closure is incomplete")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    for parameter, values in optimizer.state.items():
        for key in ("exp_avg", "exp_avg_sq"):
            if key in values and tuple(values[key].shape) != tuple(parameter.shape):
                raise ValueError("Restored Adam state shape differs: " + key)
    sampler.load_state_dict(checkpoint["sampler_state_dict"])
    # The caller constructs its DataLoader iterator before restoring this state.
    return checkpoint["step"], checkpoint["rng_state_dict"]
