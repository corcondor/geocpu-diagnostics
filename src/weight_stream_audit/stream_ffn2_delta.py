#!/usr/bin/env python3
"""Read-only, low-RSS comparison of FFN output columns in two PyTorch ZIPs.

This ranks parameter changes. It does not identify a causal repetition circuit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import psutil
import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--child", type=Path, required=True)
    parser.add_argument("--parent-sha256", required=True)
    parser.add_argument("--child-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-free-mib", type=int, default=256)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.parent.resolve() == args.child.resolve():
        raise ValueError("Parent and child must be distinct")
    for path, expected in ((args.parent, args.parent_sha256), (args.child, args.child_sha256)):
        if not path.is_file() or sha256(path) != expected.lower():
            raise ValueError(f"Missing or changed checkpoint: {path}")
    if psutil.virtual_memory().available < args.minimum_free_mib * 1024 * 1024:
        raise MemoryError("Insufficient free physical memory before mapping")

    parent = torch.load(args.parent, map_location="cpu", mmap=True, weights_only=True)
    child = torch.load(args.child, map_location="cpu", mmap=True, weights_only=True)
    pa, ch = parent["model_state_dict"], child["model_state_dict"]
    keys = sorted(key for key in pa if key.endswith(".ffn.2.weight"))
    if not keys or keys != sorted(key for key in ch if key.endswith(".ffn.2.weight")):
        raise ValueError("FFN output tensor keys differ")
    rows = []
    for key in keys:
        if psutil.virtual_memory().available < args.minimum_free_mib * 1024 * 1024:
            raise MemoryError(f"Stopped before {key}: free physical memory below threshold")
        old, new = pa[key], ch[key]
        if old.ndim != 2 or old.shape != new.shape or old.dtype != new.dtype or old.dtype != torch.float32:
            raise ValueError(f"FFN tensor mismatch: {key}")
        x, y = old.detach().numpy(), new.detach().numpy()
        energy = np.zeros(x.shape[1], dtype=np.float64)
        for start in range(0, x.shape[0], 32):
            stop = min(start + 32, x.shape[0])
            diff = y[start:stop].astype(np.float64) - x[start:stop].astype(np.float64)
            energy += np.einsum("ij,ij->j", diff, diff)
        total = float(energy.sum())
        order = np.argsort(energy)[::-1]
        rows.append({
            "key": key,
            "shape": list(old.shape),
            "squared_delta_total": total,
            "top64_fraction": float(energy[order[:64]].sum() / total) if total else 0.0,
            "top_columns": [
                {"column": int(index), "squared_delta": float(energy[index])}
                for index in order[:8]
            ],
        })
        print(key, f"squared_delta={total:.8g}", flush=True)
    ranked = sorted(rows, key=lambda item: item["squared_delta_total"], reverse=True)
    result = {
        "schema": "geocpu_ffn2_column_delta_readonly_v1",
        "parent": {"path": str(args.parent), "sha256": args.parent_sha256.lower(), "bytes": os.path.getsize(args.parent)},
        "child": {"path": str(args.child), "sha256": args.child_sha256.lower(), "bytes": os.path.getsize(args.child)},
        "tensor_count": len(rows),
        "ranking_is_causal_evidence": False,
        "layers": ranked,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Wrote", args.output, flush=True)


if __name__ == "__main__":
    main()
