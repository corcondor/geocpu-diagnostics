"""Read-only full-forward versus cached-step probe for the pure v44 source.

This tiny random fixture tests the real model methods without loading a saved
checkpoint. It is a code-path probe, not a quality or checkpoint evaluation.
"""

import argparse
import importlib
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_dir.resolve()))

    import torch

    torch.set_num_threads(1)
    config_type = importlib.import_module("GeoCPU_CED_v1").CEDConfig
    base_type = importlib.import_module("GeoCPU_CED_v2").GeoCPU_CED_v2
    model_type = importlib.import_module("integrated_model_v44_python").IntegratedModelV44Python
    config = config_type(
        vocab_size=32,
        d_model=8,
        encoder_layers=1,
        decoder_layers=1,
        ffn_dim=16,
        heads=2,
        key_dim=4,
        value_dim=4,
        max_positions=64,
        chunk_size=4,
        scan_backend="jit",
        position_mode="none",
        head_bias=True,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        model = model_type(base_type(config, select_after=0))
    model.eval()
    tokens = torch.tensor([[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]], dtype=torch.long)
    results = []
    with torch.inference_mode():
        for arm in ("dense_all_boundary", "learned"):
            full = model(tokens, arm=arm).logits[0]
            state = model.prefill(tokens[:, :1], arm=arm)
            cached = [state.last_logits[0, 0].clone()]
            for position in range(1, tokens.shape[1]):
                state = model.step_state(tokens[:, position : position + 1], state)
                cached.append(state.last_logits[0, 0].clone())
            stepped = torch.stack(cached)
            diff = (full - stepped).abs()
            results.append(
                {
                    "arm": arm,
                    "positions": int(tokens.shape[1]),
                    "max_abs_logit_difference": float(diff.max()),
                    "per_position_max_abs_logit_difference": [float(value) for value in diff.max(dim=1).values],
                    "argmax_mismatch_positions": [
                        index
                        for index, (a, b) in enumerate(zip(full.argmax(dim=1).tolist(), stepped.argmax(dim=1).tolist()))
                        if a != b
                    ],
                }
            )
    payload = {
        "schema": "geocpu_pure_v44_tiny_full_step_readonly_v1",
        "scope": "random 8-wide CPU fixture; not pretrained checkpoint",
        "updates": 0,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
