# -*- coding: utf-8 -*-
"""
eval_v42_search_retrieval.py: Tier 6 Search-Augmented Multi-Hop Held-Out Evaluator (200 Questions).
"""
import sys
import json
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, Optional

ROOT = Path(__file__).parent.resolve()
sys.path.append(str(ROOT))

import torch
from GeoCPU_CED_v1 import CEDConfig
from GeoCPU_CED_v2 import GeoCPU_CED_v2
from integrated_model_v42 import IntegratedModelV42
from local_search_engine_v42 import LocalSearchEngineV42

def run_tier6_search_evaluation(
    checkpoint_path: Optional[str] = None,
    manifest_path: Optional[Any] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    model: Optional[torch.nn.Module] = None,
    vocab: Optional[Dict[str, int]] = None,
    inv_vocab: Optional[Dict[int, str]] = None,
    arm: str = "learned",
    max_samples: Optional[int] = None,
    **kwargs: Any
) -> Dict[str, Any]:
    print("=" * 90, flush=True)
    print("TIER 6: SEARCH-AUGMENTED MULTI-HOP RETRIEVAL BENCHMARK (200 QUESTIONS)", flush=True)
    print("=" * 90, flush=True)

    if manifest_path is None:
        p_aligned = ROOT / "data" / "eval_v42_search_retrieval_aligned.jsonl"
        p_heldout = ROOT / "data" / "eval_v42_search_retrieval_heldout.jsonl"
        manifest_path = p_aligned if p_aligned.exists() else p_heldout
    else:
        manifest_path = Path(manifest_path)

    if not manifest_path.exists():
        print(f"  [Tier 6] Manifest not found: {manifest_path}")
        return {}

    dev = torch.device(device)
    if vocab is None or inv_vocab is None:
        meta = json.loads((ROOT / "base_metadata.json").read_bytes())
        vocab = meta["vocab"]
        inv_vocab = {int(v): k for k, v in vocab.items()}

    if model is None:
        if not checkpoint_path or not Path(checkpoint_path).exists():
            print(f"  [Tier 6] Checkpoint not found: {checkpoint_path}")
            return {}
        cfg_dict = dict(meta["config"])
        cfg_dict["scan_backend"] = "jit"
        cfg = CEDConfig(**cfg_dict)
        base = GeoCPU_CED_v2(cfg)
        model = IntegratedModelV42(base=base)
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(sd, strict=False)
        model.to(dev)

    model.eval()
    items = [json.loads(line) for line in open(manifest_path, encoding="utf-8")]
    if max_samples is not None and 0 < max_samples < len(items):
        items = items[:max_samples]

    total = len(items)
    correct = 0
    search_called_cnt = 0
    multihop_called_cnt = 0
    budget_exceeded_cnt = 0
    by_hop = defaultdict(lambda: {"total": 0, "correct": 0, "search_calls": 0})
    t0 = time.time()

    with torch.no_grad():
        for i, item in enumerate(items):
            prompt = item["prompt"]
            if "\n思考:" in prompt:
                prompt = prompt.split("\n思考:")[0] + "\n思考:"
            elif not prompt.endswith("思考:"):
                prompt = prompt.strip() + "\n思考:"

            target = item["target"].strip()
            hop_type = item.get("hop_type", "1_hop")

            if hasattr(model, "generate_with_qwenputer_v42"):
                gen_res = model.generate_with_qwenputer_v42(
                    prompt, vocab, inv_vocab,
                    max_tokens=None, arm=arm,
                    repetition_penalty=1.0,
                    enable_search=True
                )
            else:
                gen_res = model.generate_with_qwenputer(
                    prompt, vocab, inv_vocab,
                    max_tokens=None, arm=arm,
                    repetition_penalty=1.0
                )

            out_text = gen_res["output_text"].strip()
            full_text = gen_res["full_text"]
            s_calls = gen_res.get("search_invocations", 0)

            if s_calls > 0:
                search_called_cnt += 1
            if s_calls >= 2:
                multihop_called_cnt += 1

            # Check if 2048 budget was respected
            if len(full_text) > 2000:
                budget_exceeded_cnt += 1

            is_corr = (target in out_text)
            if is_corr:
                correct += 1

            by_hop[hop_type]["total"] += 1
            if is_corr:
                by_hop[hop_type]["correct"] += 1
            if s_calls > 0:
                by_hop[hop_type]["search_calls"] += 1

            if (i + 1) % 40 == 0 or (i + 1) == total:
                print(f"  Progress: {i+1:3d}/{total} | Acc: {correct/(i+1)*100:5.1f}% ({correct}/{i+1}) | "
                      f"SearchCalls: {search_called_cnt}/{i+1} | Elapsed: {time.time()-t0:.1f}s", flush=True)

    acc = correct / max(1, total) * 100.0
    search_rate = search_called_cnt / max(1, total) * 100.0
    multihop_rate = multihop_called_cnt / max(1, total) * 100.0

    print("-" * 90, flush=True)
    print(f"  Overall Tier 6 Search Accuracy:      {correct:3d} / {total} ({acc:5.2f}%)", flush=True)
    print(f"  Search Invocation Autonomy Rate:     {search_called_cnt:3d} / {total} ({search_rate:5.2f}%)", flush=True)
    print(f"  Multi-Hop 2-Step Execution Rate:     {multihop_called_cnt:3d} / {total} ({multihop_rate:5.2f}%)", flush=True)
    print(f"  2048 Context Budget Adherence:       {total - budget_exceeded_cnt:3d} / {total} ({(total-budget_exceeded_cnt)/total*100:5.2f}%)", flush=True)
    print("-" * 90, flush=True)
    for h_type, h_d in sorted(by_hop.items()):
        h_acc = h_d["correct"] / max(1, h_d["total"]) * 100.0
        h_call = h_d["search_calls"] / max(1, h_d["total"]) * 100.0
        print(f"  * {h_type:15s}: Acc: {h_acc:5.1f}% ({h_d['correct']:2d}/{h_d['total']:2d}) | SearchUsed: {h_call:5.1f}% ({h_d['search_calls']:2d}/{h_d['total']:2d})", flush=True)
    print("=" * 90 + "\n", flush=True)

    return {
        "total": total,
        "correct": correct,
        "accuracy": acc,
        "search_invocation_rate": search_rate,
        "multihop_rate": multihop_rate,
        "budget_adherence_rate": (total - budget_exceeded_cnt) / max(1, total) * 100.0,
        "by_hop": dict(by_hop)
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    run_tier6_search_evaluation(
        checkpoint_path=args.checkpoint,
        manifest_path=args.manifest,
        device=args.device,
        max_samples=args.max_samples
    )
