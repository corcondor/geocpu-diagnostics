"""All eight original v44 question tiers, evaluated through language/Python TIR.

No ALU controller is imported or invoked. Old ALU-specific execution measures
are explicitly not applicable, never renamed as Python success. This module
does not load a checkpoint, construct a model, or start training.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import dataclass
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
import uuid

MANIFESTS = {
    1: "eval_swe_step_control_heldout.jsonl",
    2: "eval_v38_decomposed_state_bench200.jsonl",
    3: "jamcqa_manifest_test.jsonl",
    4: "benchmark_8100_manifest.jsonl",
    5: "eval_v41_logical_instruction_operators_heldout.jsonl",
    6: "eval_v42_search_retrieval_aligned.jsonl",
    7: "benchmark_8100_manifest.jsonl",
}
TIER8_PROMPTS = [
    "問題: 関数 f(x) = 3x^3 + 2sin(x) - 5 について、導関数 f'(x) を求め、x = 0 における微分係数 f'(0) を計算してください。\n思考:",
    "問題: 初速 20 m/s、仰角 45° で斜方投射された物体の滞空時間と水平到達距離を求めてください（重力加速度 g = 9.8 m/s^2、空気抵抗は無視）。\n思考:",
    "指示: C++ を用いて、効率的な『二分探索 (Binary Search)』の完全な実装コードを作成し、その計算量と動作原理を解説してください。\n回答:",
    "指示: Rust を用いて、効率的な『フィボナッチ数列のメモ化DP』の完全な実装コードを作成し、その計算量と動作原理を解説してください。\n回答:",
]
_GLYPHS = dict(zip("ϵϕϑϰϱςοΑΒΕΖΗΙΚΜΝΟΡΤΥΧ", "εφθκρσoABEZHIKMNOPTYX"))
_GLYPHS.update({"＞": ">", "＜": "<", "⇔": "↔", "↦": "→", "⊨": "⊢", "⊤": "T", "∅": "φ", "≢": "≠", "÷": "/", "⋅": "・", "∙": "・", "⟨": "(", "⟩": ")", "⟪": "(", "⟫": ")", "⟦": "[", "⟧": "]", "⌈": "[", "⌉": "]", "⌊": "[", "⌋": "]", "−": "-"})
for _code in range(0xFF01, 0xFF5F):
    if chr(_code) not in "（）！？：～":
        _GLYPHS[chr(_code)] = chr(_code - 0xFEE0)
_TRANSLATE = str.maketrans(_GLYPHS)

def normalize_text(text):
    """Same legacy glyph map, separated from every CPU/ALU implementation."""
    return text.translate(_TRANSLATE) if text else ""

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()

def _json(path, data):
    path = Path(path); part = path.with_name(path.name + "." + uuid.uuid4().hex + ".partial")
    with part.open("x", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2); f.flush(); os.fsync(f.fileno())
    os.replace(part, path)

def load_rows(path):
    with Path(path).open(encoding="utf-8-sig") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if any(not isinstance(row, dict) or not isinstance(row.get("prompt"), str) for row in rows):
        raise ValueError("Invalid evaluation row: " + str(path))
    return rows

def assistant_text(generation):
    """Never give a model answer credit for a value printed by the environment."""
    if "model_generated_text" in generation:
        return generation["model_generated_text"]
    # A trace-less result may be supplied by a test fixture, not an old executor.
    if generation.get("tool_invocations", 0):
        raise ValueError("Tool output present but assistant-only text is missing")
    return generation.get("output_text", "")

def prose_text(text):
    return re.sub(r"(?ms)^```[^\n]*\n.*?^```[^\n]*(?:\n|$)", "", text).strip()

def final_number(text):
    values = re.findall(r"[-+]?\d+(?:\.\d+)?", prose_text(text))
    return values[-1] if values else None

def extract_choice(text, letters="ABCD"):
    # The original Tier5 ABC extractor, and Tier7 generated ABCD extractor.
    if letters == "ABC":
        found = re.search(r"正解[:：]\s*([ABC])", text)
        if found: return found.group(1)
        found = re.findall(r"\b([ABC])\b", text[-40:])
        if found: return found[-1]
        return next((c for c in reversed(text) if c in letters), "")
    found = re.search(r"(?:正解|回答|答え|選ぶのは|合致するのは|結論として)\s*[:：は]?\s*([A-D])\b", text)
    if found: return found.group(1)
    found = re.search(r"\b([A-D])(?:\s*[\.\)）]|$|\s)", text.strip())
    if found: return found.group(1)
    return next((c for c in letters if c in text), text.strip()[:1])

def score_generation(tier, row, generation, mode="generation"):
    text = assistant_text(generation).strip()
    target = str(row.get("final_answer", row.get("target", ""))).strip()
    score = {"prediction": None, "correct": None, "semantic_reasoning_quality": None}
    if tier == 1 or (tier == 4 and row.get("task_type") == "code_trace"):
        score.update(prediction=final_number(text), answer_metric="assistant_final_numeric_exact_match")
        score["correct"] = score["prediction"] == target
    elif tier == 2:
        score.update(prediction=text, correct=bool(target) and target in text,
                     answer_metric="legacy_final_answer_substring_in_assistant_text")
    elif tier == 3:
        choices = row.get("choices", []); index = row.get("answer_index")
        if not isinstance(index, int) or not 0 <= index < len(choices):
            raise ValueError("JamC-QA answer_index/choices invalid")
        answer = str(choices[index]); score.update(prediction=text, correct=text == answer,
            expected_choice_text=answer, substring_match=answer in text, answer_metric="official_generative_choice_text_exact_match")
    elif tier == 4 and row.get("task_type") == "grid_transform":
        found = re.search(r"\[\[.*?\]\]", text)
        grid = found.group(0) if found else text
        score.update(prediction=grid, correct=grid.replace(" ", "") == target.replace(" ", ""),
                     answer_metric="legacy_grid_text_exact_match", completed_grid=grid.endswith("]]"))
    elif tier in (5, 7):
        prediction = extract_choice(text, "ABC" if tier == 5 else "ABCD")
        score.update(prediction=prediction, correct=prediction == target, answer_metric="legacy_generated_choice_extraction")
    elif tier == 6:
        score.update(prediction=text, correct=bool(target) and target in text,
                     answer_metric="legacy_target_substring_not_semantic_grading")
    elif tier == 8:
        score.update(prediction=text,
                     has_code=any(s in text for s in ("```", "def ", "int ", "fn ")),
                     has_narrative=any(s in text for s in ("起", "承", "転", "結", "求")),
                     answer_metric="legacy_format_probes_plus_actual_Python_trace_not_answer_correctness")
    else:
        score.update(prediction=text.split()[-1] if text.split() else text,
                     correct=(text.split()[-1] if text.split() else text) == target,
                     answer_metric="legacy_final_word_exact_match")
    score.update(ended_with_eos=generation.get("ended_with_eos"),
                 actual_tool_invocations=generation.get("tool_invocations"),
                 successful_tool_invocations=generation.get("successful_tool_invocations"),
                 search_invocations=generation.get("search_invocations"),
                 legacy_alu_invocations=None, legacy_alu_step_termination=None,
                 legacy_alu_register_correctness=None,
                 legacy_alu_metric_status="not_applicable_ALU_removed")
    return score

def torch_choice_predictions(model, rows, vocab, *, batch_size=64, arm="learned", letters="ABCD"):
    """Preserve the old batched, last-valid-position choice-logit argmax."""
    import torch
    device = next(model.parameters()).device
    choices = [c for c in letters if c in vocab]
    if not choices: raise ValueError("No legal choice tokens")
    predictions = []
    amp = device.type == "cuda"
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=amp, dtype=torch.bfloat16):
        for offset in range(0, len(rows), batch_size):
            chunk = rows[offset:offset + batch_size]
            ids = [[vocab[c] for c in normalize_text(row["prompt"]) if c in vocab] for row in chunk]
            if any(not seq for seq in ids): raise ValueError("Empty encoded evaluation prompt")
            width = max(32, math.ceil(max(map(len, ids)) / 32) * 32)
            x = torch.zeros((len(chunk), width), dtype=torch.long, device=device)
            valid = torch.zeros_like(x, dtype=torch.bool)
            for i, seq in enumerate(ids):
                x[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device); valid[i, :len(seq)] = True
            output = model(x, valid, arm=arm)
            for i, seq in enumerate(ids):
                logits = output.logits[i, len(seq) - 1, [vocab[c] for c in choices]]
                predictions.append({"prediction": choices[int(logits.argmax().item())],
                                    "choice_logits": {c: float(logits[j].float().item()) for j, c in enumerate(choices)}})
    return predictions

def _deadline(value):
    if value is None: return None
    if isinstance(value, str): value = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None: raise ValueError("Evaluation deadline must include timezone")
    return time.monotonic() + (value - dt.datetime.now(dt.timezone.utc)).total_seconds()

def _tasks(tier, rows):
    if tier == 3:
        tasks = [(row, "logit", row["prompt"], False) for row in rows]
        for row in rows[:100]:
            prompt = ("指示:: 質問と回答の選択肢を入力として受け取り、選択肢から回答を選択してください。回答の他には何も含めないことを厳守してください。\n\n"
                + f"質問:: {row['question']}\n選択肢:: {' '.join(row['choices'])}\n回答::")
            tasks.append((row, "official_generation", prompt, False))
        return tasks
    if tier == 7:
        eligible = [row for row in rows if row.get("task_type") == "multiple_choice" or "choices" in row or ("A" in row.get("prompt", "") and "B" in row.get("prompt", ""))][:200]
        return [(row, mode, row["prompt"].strip() + "\n思考: ", enable)
                for row in eligible for mode, enable in (("closed_book", False), ("open_book", True))]
    tasks = []
    for row in rows:
        prompt = row["prompt"]
        if tier == 6:
            prompt = prompt.split("\n思考:")[0] + "\n思考:" if "\n思考:" in prompt else prompt.rstrip() + "\n思考:"
        mode = "logit" if tier == 4 and row.get("task_type") == "multiple_choice" else "generation"
        tasks.append((row, mode, prompt, tier == 6))
    return tasks

class Journal:
    def __init__(self, path):
        self.file = Path(path).open("x", encoding="utf-8")
    def append(self, record):
        self.file.write(json.dumps(record, ensure_ascii=False) + "\n"); self.file.flush(); os.fsync(self.file.fileno())
    def close(self): self.file.close()

def _summarize(records, planned):
    measured = [r for r in records if r["status"] == "measured"]
    scored = [r for r in measured if isinstance(r.get("score", {}).get("correct"), bool)]
    groups = {}
    for mode in sorted({r["mode"] for r in records}):
        group = [r for r in records if r["mode"] == mode]
        observed = [r for r in group if r["status"] == "measured"]
        graded = [r for r in observed if isinstance(r.get("score", {}).get("correct"), bool)]
        groups[mode] = {"planned_records": len(group), "measured": len(observed), "answer_scored": len(graded),
                       "correct": sum(r["score"]["correct"] for r in graded),
                       "accuracy_measured_only": sum(r["score"]["correct"] for r in graded) / len(graded) if graded else None}
    generated = [r for r in measured if isinstance(r.get("generation"), dict)]
    return {"status": "COMPLETE" if len(measured) == planned else "PARTIAL", "planned_records": planned,
            "durable_records": len(records), "measured": len(measured), "missing": planned - len(measured),
            "status_counts": dict(Counter(r["status"] for r in records)), "modes": groups,
            "correct": sum(r["score"]["correct"] for r in scored), "answer_scored": len(scored),
            "accuracy_measured_only": sum(r["score"]["correct"] for r in scored) / len(scored) if scored else None,
            "generated_records_measured": len(generated),
            "natural_eos_count": sum(r["generation"].get("ended_with_eos") is True for r in generated),
            "actual_python_call_count": sum(int(r["generation"].get("tool_invocations") or 0) for r in generated),
            "successful_python_call_count": sum(int(r["generation"].get("successful_tool_invocations") or 0) for r in generated),
            "search_invoked_records": sum(int(r["generation"].get("search_invocations") or 0) > 0 for r in generated),
            "multihop_search_records": sum(int(r["generation"].get("search_invocations") or 0) >= 2 for r in generated),
            "missing_counted_as_wrong": False, "legacy_alu_step_termination": None,
            "legacy_alu_metric_status": "not_applicable_ALU_removed"}

def evaluate_eight_tiers(model, vocab, inv_vocab, data_dir, output_dir, *, batch_size=64,
                         max_new_tokens=2048, deadline_utc=None, per_case_seconds=180.,
                         generate_fn=None, choice_fn=None, arm="learned", tiers=None):
    """Evaluate all original tiers by default; explicit subset is a smoke probe only.

    callbacks are for tiny tests: generate_fn(model,prompt,vocab,inv_vocab,**kw),
    choice_fn(model,rows,vocab,**kw). Neither callback receives reference answers.
    """
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    claim = output_dir / "eight_tier_evaluation_claim.json"
    with claim.open("x", encoding="utf-8") as f:
        json.dump({"started_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "training_updates": 0}, f)
    selected = list(range(1, 9)) if tiers is None else list(tiers)
    if len(set(selected)) != len(selected) or any(t not in range(1, 9) for t in selected): raise ValueError("Invalid tier selection")
    end = _deadline(deadline_utc)
    suite = {"schema": "geocpu_python_tir_eight_tiers_v44_release_20261103", "status": "RUNNING", "tiers": {},
             "all_eight_requested": selected == list(range(1, 9)), "training_updates": 0, "ALU_enabled": False,
             "model_reloaded": False, "max_new_tokens": max_new_tokens,
             "generation_budget_change": "Tier8 uses 2048 sampled tokens by default instead of old 128; Tier7 retains sampled caps 60 closed/120 open; other tiers retain native context-limited sampling. Injected observations count against context, not sampled-token caps.",
             "old_historical_score_constants_reused": False,
             "prior_checkpoint_evaluation_exposure": "unknown; old v44 curriculum exposed Canonical8100, so historical exposure is not erased",
             "unmeasured_values": None}
    choice_fn = choice_fn or torch_choice_predictions
    previous_mode = getattr(model, "training", None)
    if hasattr(model, "eval"): model.eval()
    try:
        for tier in selected:
            path = data_dir / MANIFESTS[tier] if tier in MANIFESTS else None
            try:
                rows = load_rows(path) if path is not None else [{"id": f"tir_original_{i+1}", "prompt": prompt} for i, prompt in enumerate(TIER8_PROMPTS)]
                tasks = _tasks(tier, rows)
            except Exception as exc:
                suite["tiers"][str(tier)] = {"status": "UNAVAILABLE", "error": repr(exc), "planned_records": None,
                    "measured": 0, "accuracy_measured_only": None}
                _json(output_dir / "eight_tier_summary.json", suite); continue
            records = []; journal = Journal(output_dir / f"tier{tier}_raw.jsonl")
            try:
                index = 0
                while index < len(tasks):
                    row, mode, original_prompt, enable_search = tasks[index]
                    remaining = float("inf") if end is None else end - time.monotonic() - 5.
                    base = {"tier": tier, "index": index, "id": row.get("id", str(index)), "mode": mode,
                            "input_row": row, "prompt": original_prompt, "enable_search": enable_search,
                            "reference_given_to_generation": False}
                    if remaining <= 0:
                        record = dict(base, status="unmeasured_deadline", generation=None, score=None)
                        journal.append(record); records.append(record); index += 1; continue
                    if mode == "logit":
                        stop = index
                        while stop < len(tasks) and tasks[stop][1] == "logit" and stop - index < batch_size: stop += 1
                        group = tasks[index:stop]
                        # Only prompts/IDs cross into prediction; labels stay in grading.
                        prediction_rows = [{"prompt": r[2], "id": r[0].get("id")} for r in group]
                        try:
                            predictions = choice_fn(model, prediction_rows, vocab, batch_size=batch_size, arm=arm)
                            if len(predictions) != len(group): raise ValueError("Incomplete choice batch")
                            for offset, ((item, imode, prompt, search), prediction) in enumerate(zip(group, predictions)):
                                rec = {"tier": tier, "index": index + offset, "id": item.get("id", str(index + offset)), "mode": imode,
                                       "input_row": item, "prompt": prompt, "enable_search": search,
                                       "reference_given_to_generation": False, "status": "measured", "generation": None,
                                       "score": dict(prediction, correct=prediction["prediction"] == str(item.get("target", "")).strip(),
                                                     answer_metric="legacy_last_valid_position_choice_logit_argmax")}
                                journal.append(rec); records.append(rec)
                        except Exception as exc:
                            for offset, (item, imode, prompt, search) in enumerate(group):
                                rec = {"tier": tier, "index": index + offset, "id": item.get("id", str(index + offset)), "mode": imode,
                                       "input_row": item, "prompt": prompt, "status": "unmeasured_prediction_error", "error": repr(exc), "score": None,
                                       "generation": None, "reference_given_to_generation": False}
                                journal.append(rec); records.append(rec)
                        index = stop; continue
                    normalized = normalize_text(original_prompt)
                    removed = Counter(c for c in normalized if c not in vocab)
                    used_prompt = "".join(c for c in normalized if c in vocab)
                    base.update(used_prompt=used_prompt, legacy_prompt_preprocessing={"normalized": normalized != original_prompt,
                        "removed_unknown_characters": dict(removed), "policy": "legacy glyph normalization and unknown-character removal, explicitly recorded"})
                    repetition_penalty = 1.12 if tier == 3 and mode == "official_generation" else 1.0
                    base["repetition_penalty"] = repetition_penalty
                    try:
                        from tir_engine_v44r1 import TIRLimits
                        cap = int(getattr(getattr(model, "config", None), "max_positions", 8192))
                        sampled_cap = (60 if mode == "closed_book" else 120) if tier == 7 else (max_new_tokens if tier == 8 else cap)
                        limits = TIRLimits(max_new_tokens=max(0, cap - len(used_prompt)), max_context_tokens=cap,
                                           max_sampled_tokens=sampled_cap,
                                           max_wall_seconds=min(per_case_seconds, remaining), max_tool_calls=8)
                        base.update(max_sampled_tokens=sampled_cap, max_total_context_tokens=cap,
                                    observations_count_against_sampled_cap=False)
                        if generate_fn is not None:
                            generation = generate_fn(model, used_prompt, vocab, inv_vocab, limits=limits, arm=arm,
                                enable_search=enable_search, repetition_penalty=repetition_penalty)
                        else:
                            generation = model.generate_with_python_tir(used_prompt, vocab, inv_vocab,
                                max_new_tokens=sampled_cap, arm=arm, enable_search=enable_search, limits=limits,
                                repetition_penalty=repetition_penalty)
                        if not isinstance(generation, dict): raise TypeError("Generation result must be an object")
                        if generation.get("error"):
                            record = dict(base, status="unmeasured_generation_error", generation=generation, score=None, error=generation["error"])
                        elif generation.get("stop_reason") in ("wall_time_limit", "tool_worker_startup_error"):
                            record = dict(base, status="unmeasured_generation_partial", generation=generation, score=None)
                        else:
                            record = dict(base, status="measured", generation=generation, score=score_generation(tier, row, generation, mode))
                    except Exception as exc:
                        record = dict(base, status="unmeasured_generation_error", generation=None, score=None, error=repr(exc))
                    journal.append(record); records.append(record); index += 1
            finally:
                journal.close()
            summary = _summarize(records, len(tasks))
            question_count = sum(mode == "closed_book" for _, mode, _, _ in tasks) if tier == 7 else len(rows)
            summary.update(question_count=question_count, manifest={"path": str(path), "sha256": sha256(path)} if path is not None else None,
                           raw_path=str(output_dir / f"tier{tier}_raw.jsonl"), raw_sha256=sha256(output_dir / f"tier{tier}_raw.jsonl"))
            if tier in (3, 7):
                summary["accuracy_measured_only"] = None
                summary["accuracy_scope"] = "Separate per-mode denominators; no pooled accuracy between distinct evaluation modes"
            if tier == 4:
                pillars = defaultdict(lambda: {"measured": 0, "correct": 0})
                for rec in records:
                    if rec["status"] == "measured" and isinstance(rec.get("score", {}).get("correct"), bool):
                        stats = pillars[rec["input_row"].get("pillar", "unspecified")]
                        stats["measured"] += 1; stats["correct"] += rec["score"]["correct"]
                summary["pillar_stats"] = {key: dict(value, accuracy_measured_only=value["correct"] / value["measured"])
                                             for key, value in pillars.items()}
            if tier == 5:
                grouped = defaultdict(dict)
                operators = defaultdict(lambda: {"measured": 0, "correct": 0})
                for rec in records:
                    if rec["status"] == "measured":
                        kind = rec["input_row"].get("operator_type", "unknown")
                        grouped[rec["input_row"].get("group_id")][kind] = rec["score"]["correct"]
                        operators[kind]["measured"] += 1; operators[kind]["correct"] += rec["score"]["correct"]
                needed = ("positive_entailment", "negation_contradiction", "non_entailment_undeducible")
                complete = [group for group in grouped.values() if all(k in group for k in needed)]
                summary["tri_switch"] = {"measured_groups": len(complete), "correct": sum(all(g[k] for k in needed) for g in complete),
                                         "accuracy": sum(all(g[k] for k in needed) for g in complete) / len(complete) if complete else None}
                pair = needed[1:]
                complete_pair = [group for group in grouped.values() if all(k in group for k in pair)]
                summary["negation_vs_undeducible"] = {"measured_groups": len(complete_pair),
                    "correct": sum(all(g[k] for k in pair) for g in complete_pair),
                    "accuracy": sum(all(g[k] for k in pair) for g in complete_pair) / len(complete_pair) if complete_pair else None}
                summary["by_operator"] = dict(operators)
            if tier == 7:
                modes = summary["modes"]; closed = modes.get("closed_book", {}); opened = modes.get("open_book", {})
                a, b = closed.get("accuracy_measured_only"), opened.get("accuracy_measured_only")
                summary["absolute_search_lift"] = b - a if a is not None and b is not None and closed.get("measured") == opened.get("measured") == question_count else None
                summary["comparison_budget"] = "Original unequal sampled caps retained: closed=60/open=120; injected observations separate within native context. Search lift is budget-confounded and not isolated search causality."
            if tier == 8:
                observed = [r["score"] for r in records if r["status"] == "measured"]
                summary.update(tir_code_rate=sum(r["has_code"] for r in observed) / len(observed) if observed else None,
                               narrative_rate=sum(r["has_narrative"] for r in observed) / len(observed) if observed else None,
                               format_rate_denominator=len(observed), format_rates_are_not_answer_accuracy=True)
            suite["tiers"][str(tier)] = summary
            _json(output_dir / "eight_tier_summary.json", suite)
            print(f"Tier {tier}/8: {summary['status']}; measured={summary['measured']}/{summary['planned_records']}", flush=True)
    finally:
        if previous_mode is not None and hasattr(model, "train"): model.train(previous_mode)
    suite["status"] = "COMPLETE" if len(suite["tiers"]) == 8 and all(r["status"] == "COMPLETE" for r in suite["tiers"].values()) else "PARTIAL"
    suite["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    _json(output_dir / "eight_tier_summary.json", suite)
    return suite

run_eight_tiers = evaluate_eight_tiers
