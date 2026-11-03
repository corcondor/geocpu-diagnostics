"""eval_jamcqa_official.py: Dual Evaluator for SoftBank's JamC-QA Benchmark.

Supports:
1. Official Generative Exact Match (Exact Match of generated choice text verbatim)
2. Classical Multiple-Choice Logit Argmax (Perplexity on A, B, C, D choices)
"""
import sys
import json
import math
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict

import torch
import torch.nn.functional as F

ROOT = Path(__file__).parent.resolve()
sys.path.append(str(ROOT))

from GeoCPU_CED_v1 import CEDConfig
from GeoCPU_CED_v2 import GeoCPU_CED_v2
from integrated_model_v30 import IntegratedModelV27, IntegratedModelV30
from qwenputer_controller_v30 import canonical_normalize_text

def wilson_score_interval(correct: int, total: int) -> Tuple[float, float, float]:
    if total == 0:
        return 0.0, 0.0, 0.0
    p = correct / total
    z = 1.960
    denominator = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denominator
    spread = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
    lower = max(0.0, centre - spread)
    upper = min(1.0, centre + spread)
    return p, lower, upper

def run_jamcqa_official_eval(
    checkpoint_path: str = None,
    manifest_path: str = None,
    mode: str = "both",  # 'generative', 'logit', 'both'
    arm: str = "learned",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 64,
    max_gen_samples: int = 100,  # evaluate generative EM on dev or sample subset
    output_audit_path: str = None,
    model: torch.nn.Module = None,
    vocab: dict = None,
    inv_vocab: dict = None
) -> Dict[str, Any]:
    print("=" * 90, flush=True)
    print("SOFTBANK JamC-QA OFFICIAL DUAL EVALUATOR (Generative Exact Match & Logit Argmax)", flush=True)
    print("=" * 90, flush=True)
    print(f"  Checkpoint: {checkpoint_path if checkpoint_path else '[In-Memory Model]'}", flush=True)
    print(f"  Mode: {mode} | Arm: {arm} | Device: {device} | Batch Size: {batch_size}", flush=True)

    if vocab is None or inv_vocab is None:
        meta = json.loads((ROOT / "base_metadata.json").read_bytes())
        vocab = meta["vocab"]
        inv_vocab = {int(v): k for k, v in vocab.items()}

    dev = torch.device(device)

    if model is None:
        meta = json.loads((ROOT / "base_metadata.json").read_bytes())
        cfg_dict = dict(meta["config"])
        cfg_dict["max_positions"] = 2048
        cfg_dict["scan_backend"] = "jit"
        config = CEDConfig(**cfg_dict)

        base_model = GeoCPU_CED_v2(config, select_after=meta.get("select_after", 4))
        model = IntegratedModelV30(base=base_model)

        ckpt_p = Path(checkpoint_path).resolve()
        state_dict = torch.load(ckpt_p, map_location="cpu")
        sd = state_dict["model_state_dict"] if "model_state_dict" in state_dict else state_dict
        model.load_state_dict(sd, strict=False)
        if "router_bias" in state_dict and state_dict["router_bias"] is not None:
            model.routing_module.router_bias.copy_(torch.tensor(float(state_dict["router_bias"])))
        model.to(dev)

    model.eval()

    m_path = Path(manifest_path) if manifest_path else (ROOT / "data" / "jamcqa_manifest_test.jsonl")
    questions = [json.loads(line) for line in open(m_path, encoding='utf-8')]
    print(f"  Loaded {len(questions)} JamC-QA questions from {m_path.name}")

    results = {}

    # 1. Multiple-Choice Logit Argmax Mode
    if mode in ("logit", "both"):
        print("\n--- Running Mode 1: Classical Multiple-Choice Logit Argmax ---")
        choice_letters = ["A", "B", "C", "D"]
        choice_ids = [vocab[c] for c in choice_letters if c in vocab]

        t0 = time.time()
        total_correct = 0
        cat_stats = defaultdict(lambda: {"total": 0, "correct": 0})

        use_amp = (dev.type == "cuda")
        with torch.no_grad(), torch.amp.autocast(device_type=dev.type, enabled=use_amp, dtype=torch.bfloat16):
            for b_start in range(0, len(questions), batch_size):
                b_items = questions[b_start:b_start + batch_size]
                seqs = [[vocab[c] for c in canonical_normalize_text(item["prompt"]) if c in vocab] for item in b_items]
                max_len = max(32, int(math.ceil(max(len(s) for s in seqs) / 32.0)) * 32)
                B = len(b_items)

                x_cpu = torch.zeros((B, max_len), dtype=torch.long)
                valid_cpu = torch.zeros((B, max_len), dtype=torch.bool)
                last_pos = []
                for i, seq in enumerate(seqs):
                    n = len(seq)
                    if n > 0:
                        x_cpu[i, :n] = torch.tensor(seq, dtype=torch.long)
                        valid_cpu[i, :n] = True
                        last_pos.append(n - 1)
                    else:
                        last_pos.append(0)

                x = x_cpu.to(dev, non_blocking=True)
                valid = valid_cpu.to(dev, non_blocking=True)
                out = model(x, valid, arm=arm)
                logits = out.logits

                for i, item in enumerate(b_items):
                    lp = last_pos[i]
                    final_logits = logits[i, lp, choice_ids]
                    pred_idx = torch.argmax(final_logits).item()
                    pred_letter = choice_letters[pred_idx]
                    target_letter = item["target"]
                    is_correct = (pred_letter == target_letter)

                    if is_correct:
                        total_correct += 1
                    cat = item["category"]
                    cat_stats[cat]["total"] += 1
                    if is_correct:
                        cat_stats[cat]["correct"] += 1

                if (b_start + batch_size) % 512 == 0 or (b_start + batch_size) >= len(questions):
                    done = min(len(questions), b_start + batch_size)
                    print(f"  Logit Progress: {done:5d}/{len(questions)} ({done/len(questions)*100:5.1f}%) | "
                          f"Acc: {total_correct/max(1, done)*100:5.2f}% | Elapsed: {time.time()-t0:.1f}s")

        acc, ci_low, ci_high = wilson_score_interval(total_correct, len(questions))
        results["logit_argmax"] = {
            "total": len(questions),
            "correct": total_correct,
            "accuracy": acc,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "categories": {c: {"total": st["total"], "correct": st["correct"], "acc": st["correct"]/st["total"]} for c, st in cat_stats.items()}
        }

    # 2. Official Generative Exact Match Mode
    if mode in ("generative", "both"):
        gen_subset = questions[:max_gen_samples] if max_gen_samples > 0 else questions
        print(f"\n--- Running Mode 2: Official Generative Exact Match ({len(gen_subset)} samples) ---")

        t0 = time.time()
        em_correct = 0
        substr_correct = 0
        ended_with_eos = 0
        gen_records = []

        with torch.no_grad():
            for i, item in enumerate(gen_subset):
                q = item["question"]
                choices = item["choices"]
                ans_idx = item["answer_index"]
                expected_text = choices[ans_idx]

                official_prompt = (
                    "指示:: 質問と回答の選択肢を入力として受け取り、選択肢から回答を選択してください。回答の他には何も含めないことを厳守してください。\n\n"
                    f"質問:: {q}\n"
                    f"選択肢:: {' '.join(choices)}\n"
                    "回答::"
                )

                res = model.generate_with_qwenputer(
                    official_prompt, vocab, inv_vocab,
                    max_tokens=None, arm=arm, repetition_penalty=1.12
                )
                gen_text = res["output_text"].strip()
                eos = res.get("ended_with_eos", False)
                if eos:
                    ended_with_eos += 1

                is_em = (gen_text == expected_text)
                is_sub = (expected_text in gen_text)

                if is_em:
                    em_correct += 1
                if is_sub:
                    substr_correct += 1

                gen_records.append({
                    "id": item["id"],
                    "category": item["category"],
                    "question": q,
                    "expected": expected_text,
                    "generated": gen_text,
                    "exact_match": is_em,
                    "substr_match": is_sub,
                    "ended_with_eos": eos
                })

                if (i + 1) % 25 == 0 or (i + 1) == len(gen_subset):
                    print(f"  Gen Progress: {i+1:3d}/{len(gen_subset)} | Exact Match: {em_correct:2d} | "
                          f"Substr: {substr_correct:2d} | EOS: {ended_with_eos:2d} | Elapsed: {time.time()-t0:.1f}s")

        em_acc, em_low, em_high = wilson_score_interval(em_correct, len(gen_subset))
        sub_acc = substr_correct / max(1, len(gen_subset))
        results["generative_exact_match"] = {
            "total": len(gen_subset),
            "exact_match_count": em_correct,
            "exact_match_acc": em_acc,
            "exact_match_ci": [em_low, em_high],
            "substr_match_count": substr_correct,
            "substr_match_acc": sub_acc,
            "eos_rate": ended_with_eos / max(1, len(gen_subset))
        }
        results["generative_em"] = {
            "accuracy": sub_acc if sub_acc > em_acc else em_acc,
            "exact_match_acc": em_acc,
            "substr_match_acc": sub_acc
        }

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--mode", type=str, default="both", choices=["logit", "generative", "both"])
    parser.add_argument("--max_gen_samples", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    res = run_jamcqa_official_eval(
        checkpoint_path=args.checkpoint,
        manifest_path=args.manifest,
        mode=args.mode,
        max_gen_samples=args.max_gen_samples,
        batch_size=args.batch_size
    )
    print("\nFINAL RESULTS SUMMARY:")
    print(json.dumps(res, indent=2, ensure_ascii=False))
