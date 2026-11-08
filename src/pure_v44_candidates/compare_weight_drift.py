#!/usr/bin/env python3
"""Read-only, low-RAM drift measurement of two PyTorch ZIP checkpoints.

Only a small allowlist of pickle constructors is accepted. No model or optimizer
is constructed; model-state ZIP members are streamed and CRC-checked by zipfile.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import pickle
import time
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
PARENT = Path(r"G:\マイドライブ\GeoCPU_Frontier_CoTraining_v43\checkpoints\checkpoint_step_15000.pt")
TRAINING = Path(r"G:\マイドライブ\GeoCPU_Frontier_CoTraining_v44_PythonTIR_Hotfix_20261103\run_1\training")
PARENT_MIGRATION = TRAINING / "strict_neural_migration.json"
PARENT_INSPECTION = ROOT.parent / "Grand_cotraining_v44_python_tir_repair_v1_20261103" / "checkpoint_container_inspection.json"
OUTPUT = ROOT / "weight_drift.json"
CHUNK_BYTES = 1 << 20
BUFFER_KEY = "routing_module.router_bias"  # registered buffer, not a parameter


@dataclass(frozen=True)
class StorageRef:
    kind: str
    key: str
    location: str
    values: int


@dataclass(frozen=True)
class TensorRef:
    storage: StorageRef
    offset: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]

    @property
    def values(self) -> int:
        return math.prod(self.shape)


def rebuild_tensor(storage, offset, shape, stride, requires_grad, hooks):
    if not isinstance(storage, StorageRef) or requires_grad not in (False, True):
        raise ValueError("Unexpected tensor reconstruction")
    return TensorRef(storage, int(offset), tuple(shape), tuple(stride))


class RestrictedMetadataUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {
            ("collections", "OrderedDict"): OrderedDict,
            ("torch._utils", "_rebuild_tensor_v2"): rebuild_tensor,
            ("torch", "FloatStorage"): "FloatStorage",
            ("torch", "ByteStorage"): "ByteStorage",
        }
        try:
            return allowed[(module, name)]
        except KeyError as exc:
            raise pickle.UnpicklingError(f"Disallowed pickle global: {module}.{name}") from exc

    def persistent_load(self, pid):
        if not isinstance(pid, tuple) or len(pid) != 5 or pid[0] != "storage":
            raise pickle.UnpicklingError("Unexpected persistent reference")
        _, kind, key, location, values = pid
        if kind not in ("FloatStorage", "ByteStorage") or location not in ("cpu", "cuda:0"):
            raise pickle.UnpicklingError(f"Unexpected storage type/location: {pid!r}")
        if not str(key).isdigit() or not isinstance(values, int) or values < 0:
            raise pickle.UnpicklingError("Invalid storage reference")
        return StorageRef(kind, str(key), location, values)


def read_checkpoint_index(path: Path):
    z = zipfile.ZipFile(path, "r")
    names = set(z.namelist())
    pickles = [name for name in names if name.endswith("/data.pkl")]
    if len(pickles) != 1:
        raise ValueError("Expected exactly one checkpoint data.pkl")
    prefix = pickles[0][:-len("data.pkl")]
    z._geocpu_prefix = prefix
    state = RestrictedMetadataUnpickler(io.BytesIO(z.read(pickles[0]))).load()
    if not isinstance(state, dict) or not isinstance(state.get("model_state_dict"), (dict, OrderedDict)):
        raise ValueError("Unexpected checkpoint structure")
    refs = state["model_state_dict"]
    for name, ref in refs.items():
        if not isinstance(name, str) or not isinstance(ref, TensorRef):
            raise ValueError("Unexpected model-state entry")
        if ref.storage.kind != "FloatStorage":
            raise ValueError(f"Non-float32 model-state entry: {name}")
        if ref.offset != 0 or ref.storage.values != ref.values:
            raise ValueError(f"Shared/sliced tensor storage: {name}")
        expected_stride = tuple(math.prod(ref.shape[i + 1:]) for i in range(len(ref.shape)))
        if ref.stride != expected_stride:
            raise ValueError(f"Non-contiguous tensor: {name}")
        member = f"{prefix}data/{ref.storage.key}"
        if member not in names or z.getinfo(member).file_size != ref.values * 4:
            raise ValueError(f"Missing or wrong-sized storage: {name}")
    return z, state, refs


def family(name: str) -> str:
    if name.startswith("base.token_embed."):
        return "base_embedding"
    if name.startswith("base.encoder."):
        return "encoder"
    if name.startswith("base.decoder."):
        return "decoder"
    if name.startswith("base.head."):
        return "head"
    if name.startswith("fine_decoder."):
        return "fine_decoder"
    if name.startswith("routing_module."):
        return "router"
    if name.startswith("residual_proj."):
        return "residual"
    if name.startswith(("base.writer.", "base.final_norm.", "base.local_select.")):
        return "base_other"
    raise ValueError(f"Unclassified neural tensor: {name}")


def fresh_stats():
    return {"tensors": 0, "values": 0, "changed_values": 0,
            "delta_l2_squared": 0.0, "parent_l2_squared": 0.0,
            "child_l2_squared": 0.0, "max_abs_delta": 0.0}


def add_stats(dst, src):
    for name in ("tensors", "values", "changed_values", "delta_l2_squared", "parent_l2_squared", "child_l2_squared"):
        dst[name] += src[name]
    dst["max_abs_delta"] = max(dst["max_abs_delta"], src["max_abs_delta"])


def finalize(stats):
    den = stats["parent_l2_squared"]
    stats["relative_l2_to_parent"] = math.sqrt(stats["delta_l2_squared"] / den) if den else None
    stats["delta_l2"] = math.sqrt(stats["delta_l2_squared"])
    stats["parent_l2"] = math.sqrt(den)
    stats["child_l2"] = math.sqrt(stats["child_l2_squared"])
    stats["changed_fraction"] = stats["changed_values"] / stats["values"] if stats["values"] else None
    return stats


def read_exact(stream, n):
    chunks = []
    remaining = n
    while remaining:
        block = stream.read(remaining)
        if not block:
            raise EOFError("Tensor storage ended early")
        chunks.append(block)
        remaining -= len(block)
    return b"".join(chunks)


def compare_tensor(pa_zip, ch_zip, pa_ref, ch_ref, parent_expected_sha):
    stats = fresh_stats()
    stats["tensors"] = 1
    stats["values"] = pa_ref.values
    parent_sha = hashlib.sha256()
    pa_member = f"{pa_zip._geocpu_prefix}data/{pa_ref.storage.key}"
    ch_member = f"{ch_zip._geocpu_prefix}data/{ch_ref.storage.key}"
    with pa_zip.open(pa_member, "r") as pa_stream, ch_zip.open(ch_member, "r") as ch_stream:
        remaining = pa_ref.values * 4
        while remaining:
            n = min(CHUNK_BYTES, remaining)
            old_bytes = read_exact(pa_stream, n)
            new_bytes = read_exact(ch_stream, n)
            parent_sha.update(old_bytes)
            old = np.frombuffer(old_bytes, dtype="<f4")
            new = np.frombuffer(new_bytes, dtype="<f4")
            if not bool(np.isfinite(old).all() and np.isfinite(new).all()):
                raise ValueError("Nonfinite model parameter")
            delta = new.astype(np.float64) - old.astype(np.float64)
            stats["changed_values"] += int(np.count_nonzero(old != new))
            stats["delta_l2_squared"] += float(np.dot(delta, delta))
            old64 = old.astype(np.float64)
            new64 = new.astype(np.float64)
            stats["parent_l2_squared"] += float(np.dot(old64, old64))
            stats["child_l2_squared"] += float(np.dot(new64, new64))
            if len(delta):
                stats["max_abs_delta"] = max(stats["max_abs_delta"], float(np.max(np.abs(delta))))
            remaining -= n
        if pa_stream.read(1) or ch_stream.read(1):
            raise ValueError("Tensor storage unexpectedly larger than metadata")
    if parent_sha.hexdigest() != parent_expected_sha:
        raise ValueError(f"Parent tensor SHA mismatch: {pa_member}")
    return stats


def main():
    start = time.monotonic()
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    children = list(TRAINING.glob("checkpoint_step_25000_*.pt"))
    if len(children) != 1:
        raise ValueError(f"Expected one step-25000 child; found {len(children)}")
    child = children[0]
    receipt_path = Path(str(child) + ".receipt.json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    migration = json.loads(PARENT_MIGRATION.read_text(encoding="utf-8"))
    inspection = json.loads(PARENT_INSPECTION.read_text(encoding="utf-8"))
    parent_record = next(row for row in inspection["files"] if row["path"] == str(PARENT))
    if receipt["status"] != "COMPLETE" or receipt["step"] != 25000 or receipt["bytes"] != child.stat().st_size:
        raise ValueError("Child file does not match receipt")
    if PARENT.stat().st_size != parent_record["bytes"] or migration["parent_sha256"] != parent_record["sha256"]:
        raise ValueError("Parent file does not match prior inspection")
    if migration["parent_step"] != 15000 or not migration["all_fp32_neural_bytes_identical"]:
        raise ValueError("Parent migration receipt not suitable")

    pa_zip, pa_state, pa_refs = read_checkpoint_index(PARENT)
    ch_zip, ch_state, ch_refs = read_checkpoint_index(child)
    parent_members = len(pa_zip.namelist())
    child_members = len(ch_zip.namelist())
    if (parent_members != parent_record["members"]
            or child_members != receipt["container"]["members"]):
        raise ValueError("ZIP member count differs from receipt")
    if pa_state.get("step") != 15000 or ch_state.get("step") != 25000:
        raise ValueError("Checkpoint internal steps do not match expected steps")
    if set(ch_refs) != set(receipt["state_manifest"]):
        raise ValueError("Child neural keys differ from receipt")
    if len(ch_refs) != migration["neural_tensor_count"] or not set(ch_refs).issubset(pa_refs):
        raise ValueError("Parent and child neural key mismatch")
    dropped = sorted(set(pa_refs) - set(ch_refs))
    expected_dropped = sorted(row["name"] for row in migration["dropped_fixed_buffers"])
    if dropped != expected_dropped:
        raise ValueError("Unexplained parent-only tensor keys")
    families = {}
    all_stats = fresh_stats()
    parameter_stats = fresh_stats()
    tensor_rows = []
    for i, name in enumerate(sorted(ch_refs), 1):
        pa_ref, ch_ref = pa_refs[name], ch_refs[name]
        if pa_ref.shape != ch_ref.shape or pa_ref.stride != ch_ref.stride:
            raise ValueError(f"Tensor geometry mismatch: {name}")
        manifest = receipt["state_manifest"][name]
        if manifest["shape"] != list(ch_ref.shape) or manifest["dtype"] != "torch.float32":
            raise ValueError(f"Child tensor metadata mismatch: {name}")
        row = compare_tensor(pa_zip, ch_zip, pa_ref, ch_ref,
                             migration["neural_tensor_sha256"][name])
        fam = family(name)
        add_stats(all_stats, row)
        if name != BUFFER_KEY:
            add_stats(families.setdefault(fam, fresh_stats()), row)
            add_stats(parameter_stats, row)
        tensor_rows.append({"name": name, "family": fam, **finalize(row)})
        if i % 25 == 0 or i == len(ch_refs):
            print(f"compared {i}/{len(ch_refs)} neural tensors; {time.monotonic()-start:.1f} s", flush=True)
    pa_zip.close()
    ch_zip.close()
    buffer_rows = [row for row in tensor_rows if row["name"] == BUFFER_KEY]
    if (len(buffer_rows) != 1 or buffer_rows[0]["values"] != 1
            or buffer_rows[0]["changed_values"] != 0
            or parameter_stats["values"] != migration["neural_parameter_count"]):
        raise ValueError("Parameter/buffer accounting mismatch")
    result = {
        "schema": "geocpu_v43_to_v44_pure_weight_drift_v1",
        "method": "restricted metadata unpickler + streamed ZIP storage reads; float64 chunkwise sums; no model or optimizer load",
        "source_identity": {
            "parent_path": str(PARENT), "parent_bytes": PARENT.stat().st_size,
            "parent_sha256_prior_receipt": parent_record["sha256"],
            "parent_tensor_sha256_checked": True,
            "parent_zip_members_current_and_receipt": parent_members,
            "parent_all_member_crc_prior_inspection": parent_record["crc_bad_member"] is None,
            "child_path": str(child), "child_bytes": child.stat().st_size,
            "child_sha256_release_receipt": receipt["sha256"],
            "child_model_storage_zip_crcs_checked": True,
            "child_zip_members_current_and_receipt": child_members,
            "child_all_member_crc_release_receipt": receipt["container"]["all_member_CRCs_passed"],
            "whole_file_sha256_recomputed": False,
        },
        "parent_step": 15000, "child_step": 25000, "update_gap": 10000,
        "parent_only_fixed_buffers": len(dropped),
        "parent_only_fixed_buffer_names": dropped,
        "model_state_tensors": len(ch_refs),
        "trainable_parameter_tensors": len(ch_refs) - 1,
        "all_model_state": finalize(all_stats),
        "trainable_parameters": finalize(parameter_stats),
        "nonparameter_buffer": {**buffer_rows[0], "shape": [], "dtype": "torch.float32", "role": "registered_buffer"},
        "families": {name: finalize(stats) for name, stats in sorted(families.items())},
        "largest_tensors_by_delta_l2": sorted(tensor_rows, key=lambda row: row["delta_l2"], reverse=True)[:20],
        "largest_tensors_by_relative_l2": sorted(tensor_rows, key=lambda row: row["relative_l2_to_parent"] or 0, reverse=True)[:20],
        "elapsed_seconds": time.monotonic() - start,
        "interpretation_limit": "Weight drift is descriptive and does not establish which changes caused output repetition.",
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"all_model_state": result["all_model_state"], "trainable_parameters": result["trainable_parameters"], "families": result["families"]}, ensure_ascii=False), flush=True)
    print("Wrote", OUTPUT, flush=True)


if __name__ == "__main__":
    main()
