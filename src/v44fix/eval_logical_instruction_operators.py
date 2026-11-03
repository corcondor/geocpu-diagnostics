# -*- coding: utf-8 -*-
"""
eval_logical_instruction_operators.py:
Tier 5 Held-Out Evaluation Suite for Natural-Language-to-Logical-Operator Compilation (400 Items).

Tests whether GeoCPU v41 converts distinct natural-language instructions on the EXACT SAME premise context
into distinct internal logical operations:
1. positive_entailment        (P |- Q          : 正しいものを選べ)
2. negation_contradiction     (P |- ¬Q         : 誤っている・矛盾するものを選べ)
3. non_entailment_undeducible (P |/- Q         : 本文からは導けない・真偽未確定のものを選べ)
4. not_necessarily_true       (¬∀x P(x)        : 必ずしも正しいとはいえないものを選べ)

Measures:
- Overall Generative Accuracy (400 items)
- Per-Operator Accuracy
- ¬P vs (P |/- Q) Discrimination Rate (Distinguishing Falsehood/Contradiction from Undeducible/Neutral)
- Tri-Instruction Switch Consistency (All 3 distinct answers T, F, U correctly selected on the same context)
"""
import json
import re
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, Optional
import torch

ROOT = Path(__file__).parent.resolve()

def extract_choice_abc(text: str) -> str:
    m = re.search(r"正解[:：]\s*([ABC])", text)
    if m:
        return m.group(1)
    m2 = re.findall(r"\b([ABC])\b", text[-40:])
    if m2:
        return m2[-1]
    for ch in reversed(text):
        if ch in ("A", "B", "C"):
            return ch
    return ""

def run_logical_operator_evaluation(
    checkpoint_path: Optional[str] = None,
    manifest_path: Optional[Any] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 64,
    max_gen_samples: Optional[int] = None,
    max_samples: Optional[int] = None,
    model: Optional[torch.nn.Module] = None,
    vocab: Optional[Dict[str, int]] = None,
    inv_vocab: Optional[Dict[int, str]] = None,
    arm: str = "learned",
    **kwargs: Any
) -> Dict[str, Any]:
    if manifest_path is None:
        manifest_path = ROOT / "data" / "eval_v41_logical_instruction_operators_heldout.jsonl"
    else:
        manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        print(f"  [Tier 5] Manifest not found: {manifest_path}")
        return {}

    dev = torch.device(device)
    if vocab is None or inv_vocab is None:
        meta_p = ROOT / "base_metadata.json"
        if meta_p.exists():
            meta = json.loads(meta_p.read_bytes())
            vocab = meta["vocab"]
            inv_vocab = {int(v): k for k, v in vocab.items()}

    if model is None:
        if not checkpoint_path or not Path(checkpoint_path).exists():
            print(f"  [Tier 5] Model and checkpoint not found: {checkpoint_path}")
            return {}
        from GeoCPU_CED_v1 import CEDConfig
        from GeoCPU_CED_v2 import GeoCPU_CED_v2
        from integrated_model_v30 import IntegratedModelV30
        meta = json.loads((ROOT / "base_metadata.json").read_bytes())
        cfg_dict = dict(meta["config"])
        cfg_dict["scan_backend"] = "jit"
        cfg = CEDConfig(**cfg_dict)
        base_m = GeoCPU_CED_v2(cfg)
        model = IntegratedModelV30(base=base_m)
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(sd, strict=False)
        model.to(dev)

    items = [json.loads(line) for line in open(manifest_path, encoding="utf-8")]
    limit = max_samples or max_gen_samples
    if limit is not None and 0 < limit < len(items):
        items = items[:limit]

    print("=" * 90, flush=True)
    print("TIER 5: LOGICAL INSTRUCTION OPERATOR CONTRAST BENCHMARK (P |- Q vs ¬P vs P |/- Q vs ¬∀x)", flush=True)
    print("=" * 90, flush=True)
    print(f"  Manifest: {manifest_path.name} ({len(items)} items across {len(items)//4} identical contexts)", flush=True)

    model.eval()
    t0 = time.time()
    total = len(items)
    correct = 0
    by_op = defaultdict(lambda: {"total": 0, "correct": 0})
    by_group = defaultdict(dict)

    with torch.no_grad():
        for i, item in enumerate(items):
            prompt = item["prompt"]
            target = item["target"].strip()
            op_type = item.get("operator_type", "unknown")
            gid = item.get("group_id", f"g_{i//4}")

            gen_res = model.generate_with_qwenputer(
                prompt, vocab, inv_vocab,
                max_tokens=None,
                arm=arm,
                repetition_penalty=1.0
            )
            raw_text = gen_res["output_text"].strip()
            pred = extract_choice_abc(raw_text)
            is_corr = (pred == target)

            if is_corr:
                correct += 1
            by_op[op_type]["total"] += 1
            if is_corr:
                by_op[op_type]["correct"] += 1

            by_group[gid][op_type] = {
                "pred": pred,
                "target": target,
                "correct": is_corr
            }

            if (i + 1) % 40 == 0 or (i + 1) == total:
                print(f"  Progress: {i+1:3d}/{total} | Acc: {correct/(i+1)*100:5.1f}% ({correct}/{i+1}) | Elapsed: {time.time()-t0:.1f}s", flush=True)

    # Compute Contrast Consistency across identical contexts
    tri_switch_total = 0
    tri_switch_exact = 0
    neg_vs_undeducible_total = 0
    neg_vs_undeducible_distinct_correct = 0

    for gid, gdict in by_group.items():
        if "negation_contradiction" in gdict and "non_entailment_undeducible" in gdict:
            neg_vs_undeducible_total += 1
            if gdict["negation_contradiction"]["correct"] and gdict["non_entailment_undeducible"]["correct"]:
                neg_vs_undeducible_distinct_correct += 1
        if all(k in gdict for k in ("positive_entailment", "negation_contradiction", "non_entailment_undeducible")):
            tri_switch_total += 1
            if (gdict["positive_entailment"]["correct"] and
                gdict["negation_contradiction"]["correct"] and
                gdict["non_entailment_undeducible"]["correct"]):
                tri_switch_exact += 1

    overall_acc = correct / max(1, total) * 100.0
    neg_und_acc = neg_vs_undeducible_distinct_correct / max(1, neg_vs_undeducible_total) * 100.0
    tri_acc = tri_switch_exact / max(1, tri_switch_total) * 100.0

    print("-" * 90, flush=True)
    print(f"  Overall Logical Operator Accuracy:          {correct:3d} / {total} ({overall_acc:5.2f}%)", flush=True)
    print(f"  ¬P vs (P |/- Q) Joint Discrimination Rate:  {neg_vs_undeducible_distinct_correct:3d} / {neg_vs_undeducible_total} ({neg_und_acc:5.2f}%)", flush=True)
    print(f"  Full 3-Way Instruction Switch Consistency:  {tri_switch_exact:3d} / {tri_switch_total} ({tri_acc:5.2f}%)", flush=True)
    print("-" * 90, flush=True)
    for op_k, op_d in by_op.items():
        op_acc = op_d["correct"] / max(1, op_d["total"]) * 100.0
        print(f"  * {op_k:34s}: {op_d['correct']:3d} / {op_d['total']:3d} ({op_acc:5.1f}%)", flush=True)
    print("=" * 90 + "\n", flush=True)

    return {
        "total": total,
        "correct": correct,
        "accuracy": overall_acc,
        "neg_vs_undeducible_rate": neg_und_acc,
        "tri_switch_rate": tri_acc,
        "by_operator": dict(by_op)
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Tier 5 6-Operator Logical Evaluation")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    run_logical_operator_evaluation(
        checkpoint_path=args.checkpoint,
        manifest_path=args.manifest,
        device=args.device,
        max_samples=args.max_samples
    )
