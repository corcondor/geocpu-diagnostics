"""Collect all measured scores without pooling unlike tests or hiding missing runs."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys

from live_logging import canonical, file_sha, comparison_rows, read_logged_run

FIELDS = ["suite", "mode", "metric", "before", "after", "delta_pp", "before_correct", "after_correct",
          "before_denominator", "after_denominator", "before_state", "after_state", "comparable", "note"]

def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()

def _native_validation(folder, binding):
    """Require the same read-only completeness checks as native evidence reuse."""
    import frozen_eval as frozen
    from eval_candidate import verify_original_stage, verify_reused_original
    arguments = dict(checkpoint_sha=binding.get("checkpoint_sha256"),
                     run=binding.get("run"), step=binding.get("step"))
    try:
        verifier = verify_reused_original if (folder / "reuse_receipt.json").exists() else verify_original_stage
        verifier(folder, **arguments)
        original_output = str(folder.resolve())
        if (folder / "reuse_receipt.json").exists():
            original_output = json.loads((folder / "reuse_receipt.json").read_text())["source_directory"]
            if not Path(original_output).is_absolute():
                raise ValueError("Reused original directory must be absolute")
        stage_sha = file_sha(frozen.source_path("continue/tier_stage/stage_official_tier_v47.py"))
        for label in ("original_tier1_7", "original_tier8"):
            # The original marker binds the pinned stage (which fixes native
            # generation budgets) and its ORIGINAL output path. Reused copies
            # keep that path and marker intact; no hash is rebased or rewritten.
            expected = hashlib.sha256(json.dumps([label, stage_sha, original_output],
                                                 sort_keys=True).encode()).hexdigest()
            marker = json.loads((folder / (label + ".complete.json")).read_text())
            if marker.get("command_sha256") != expected:
                raise ValueError(f"Native stage source/original-path binding differs: {label}")
    except (OSError, ValueError, KeyError, TypeError) as error:
        return False, f"Native evidence is incomplete or unverified: {type(error).__name__}: {error}"
    return True, ""

def _tier_sections(text):
    """Do not let T4's identically named final-answer line overwrite T1."""
    headers = list(re.finditer(r"\[TIER ([1-7])/7\]", text))
    sections, repeated = {}, set()
    for index, match in enumerate(headers):
        tier = "T" + match.group(1)
        if tier in sections:
            repeated.add(tier)
        end = headers[index+1].start() if index+1 < len(headers) else len(text)
        sections[tier] = text[match.end():end]
    # Appended/repeated runs are not one immutable measurement.
    for tier in repeated:
        sections.pop(tier, None)
    return sections

def legacy_rows(root):
    import frozen_eval as frozen
    root = Path(root)
    folder = root / "original_tiers" if (root / "original_tiers").is_dir() else root
    if not (folder / "input_binding.json").is_file():
        return []
    binding = json.loads((folder / "input_binding.json").read_text())
    verified, validation_note = _native_validation(folder, binding)
    state = "complete" if verified else "unverified_or_incomplete"
    source_keys = ("bundle_sha256", "r6_sha256", "tier_manifest_counts", "retrieval_corpus_sha256", "tier8_main_sha256", "tier8_helper_sha256")
    conditions = {k: binding.get(k) for k in source_keys}
    comparable_binding = digest(conditions) if all(v is not None for v in conditions.values()) else None
    log = (folder / "original_tier1_7.log").read_text(encoding="utf-8") if (folder / "original_tier1_7.log").exists() else ""
    sections = _tier_sections(log)
    expected_counts = {suite: frozen.get_suite(suite)["selected_counts"]["rows"]
                       for suite in ("T1", "T2", "T3", "T5", "T6", "T7")}
    rows = []
    def put(suite, metric, n, correct=None, accuracy=None, mode="original", note="", valid=True):
        valid &= (type(n) is int and n > 0 and
                  (correct is None or (type(correct) is int and 0 <= correct <= n)) and
                  (accuracy is None or (math.isfinite(accuracy) and 0 <= accuracy <= 1)))
        expected = 100 if mode == "original_first100" else expected_counts.get(suite)
        if expected is not None and n != expected:
            valid = False
            note += f" Expected {expected} original cases; observed denominator={n}."
        rows.append({"suite": suite, "mode": mode, "metric": metric, "correct": correct, "answered": n,
                     "accuracy": (correct/n if correct is not None else accuracy) if valid else None,
                     "state": state if valid else "invalid_or_incomplete_measurement",
                     "condition_sha256": comparable_binding,
                     "note": " | ".join(x for x in (note.strip(), validation_note) if x),
                     "checkpoint_sha256": binding.get("checkpoint_sha256")})
    patterns = [
        ("T1", "final_answer_strict_match", r"Final Answer Strict Match:\s*(\d+)\s*/\s*(\d+)"),
        ("T1", "exact_step_termination", r"Exact Step Termination \(stopped on time\):\s*(\d+)\s*/\s*(\d+)"),
        ("T2", "end_to_end", r"End-to-End Accuracy:.*?\((\d+)/(\d+)\)"),
        ("T5", "logical_operator_accuracy", r"Overall Logical Operator Accuracy:\s*(\d+)\s*/\s*(\d+)"),
        ("T6", "legacy_search_accuracy", r"Overall Tier 6 Search Accuracy:\s*(\d+)\s*/\s*(\d+)"),
        ("T7", "closed_accuracy", r"\[ARM A:.*?Accuracy:\s*(\d+)\s*/\s*(\d+)"),
        ("T7", "open_accuracy", r"\[ARM B:.*?Accuracy:\s*(\d+)\s*/\s*(\d+)"),
    ]
    for suite, metric, pattern in patterns:
        hits = re.findall(pattern, sections.get(suite, ""))
        if hits:
            correct, n = map(int, hits[-1]); put(suite, metric, n, correct, valid=len(hits) == 1,
                note="Duplicate score lines; measurement is ambiguous." if len(hits) != 1 else "")
    for n, total, rate in re.findall(r"Logit Progress:\s*(\d+)/(\d+)\s*\([^\n]*?\| Acc:\s*([\d.]+)%", sections.get("T3", "")):
        if n == total:
            put("T3", "logit_argmax", int(total), accuracy=float(rate)/100,
                note="Original log reports a rounded percentage; numerator is not reconstructed.")
    hits = re.findall(r"Gen Progress:\s*(\d+)/(\d+)\s*\| Exact Match:\s*(\d+)", sections.get("T3", ""))
    for n, total, correct in hits:
        if n == total:
            put("T3", "generative_exact", int(total), int(correct), mode="original_first100")
    raw = folder / "tier4_raw.jsonl"
    if raw.exists():
        buckets = defaultdict(list)
        expected4 = {row["id"]: row for row in frozen.iter_cases("T4")}
        for line in raw.read_text(encoding="utf-8").splitlines():
            r = json.loads(line); buckets[r["pillar"]].append(r)
        for pillar, cases in sorted(buckets.items()):
            valid = all(type(r.get("is_correct")) is bool and
                        expected4.get(r.get("id"), {}).get("pillar") == pillar for r in cases)
            put("T4", pillar, len(cases), sum(r["is_correct"] for r in cases) if valid else None,
                valid=valid, note="Original raw per-case booleans; no rounded-log numerator reconstruction.")
    raw8 = folder / "tier8_raw.json"
    if raw8.exists():
        for profile, record in json.loads(raw8.read_text(encoding="utf-8")).items():
            for key, detail_key in (("tir_code_rate", "has_code"), ("narrative_rate", "has_narrative"),
                                    ("expected_language_fence_rate", "expected_language_fence"),
                                    ("prose_outside_code_rate", "has_prose_outside_code")):
                if key in record:
                    details = record.get("details", [])
                    valid = (len(details) == record["n"] == 4 and
                             all(type(r.get(detail_key)) is bool for r in details))
                    valid &= (valid and record[key] == 100 * sum(r[detail_key] for r in details) / record["n"])
                    put("T8", key, record["n"], accuracy=record[key]/100, mode=profile,
                        valid=valid, note="Format/prose rate; not a mathematical correctness score.")
    return rows

def collect(root):
    root = Path(root)
    rows = legacy_rows(root)
    for path in sorted(root.rglob("status.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        identity = record.get("identity")
        if not identity or "suite" not in identity:
            continue
        conditions = {k: identity.get(k) for k in ("suite", "mode", "metric", "dataset_sha256", "scorer_sha256", "runtime_sha256", "budget", "seed")}
        try:
            verified = read_logged_run(path.parent)
        except (OSError, ValueError, KeyError, TypeError) as error:
            rows.append({"suite": identity["suite"], "mode": identity.get("mode", "unknown"),
                         "metric": identity.get("metric", "unknown"), "correct": None,
                         "answered": None, "accuracy": None, "state": "unverified_result",
                         "condition_sha256": None, "logger_dir": None,
                         "note": f"Durable event verification failed: {type(error).__name__}: {error}"})
            continue
        record = verified["status"]
        main = {"suite": identity["suite"], "mode": identity["mode"], "metric": identity["metric"],
                     "correct": record["correct"], "answered": record["answered"], "accuracy": record["accuracy"],
                     "state": record["state"], "condition_sha256": digest(conditions),
                     "checkpoint_sha256": identity["checkpoint_sha256"], "logger_dir": str(path.parent),
                     "note": f"processed={record['processed']}/{record['total']}; invalid={record['invalid']}; unmeasured={record['unmeasured']}; errors={record['errors']}"}
        rows.append(main)
        # All measured functional submetrics remain visible, with their original denominator.
        for key, value in record.get("metric_totals", {}).items():
            if key in ("python_execution_success", "tool_and_terminal_exact", "explicit_final_numeric_exact", "ended_with_eos") and key != identity["metric"]:
                observed = {case_id: row["metrics"][key] for case_id, row in verified["rows"].items()
                            if key in row["metrics"] and row["status"] in ("scored", "invalid")}
                valid = bool(observed) and all(type(v) in (bool, int) and v in (0, 1) for v in observed.values())
                complete = valid and len(observed) == record["total"] and record["state"] == "complete"
                count = sum(observed.values()) if valid else None
                rows.append({**main, "metric": key, "correct": count, "answered": len(observed),
                             "accuracy": count/len(observed) if valid else None,
                             "state": "complete" if complete else "incomplete_metric_coverage",
                             "note": main["note"] + f"; metric_observations={len(observed)}/{record['total']}",
                             "condition_sha256": digest(dict(conditions, metric=key,
                                 measured_case_ids=sorted(observed))), "logger_dir": None})
    for path in sorted(root.glob("new_public_*/summary.json")):
        record = json.loads(path.read_text())
        group = path.parent.name.removeprefix("new_public_")
        result = record["summary"]
        rows.append({"suite": "new_public:"+group, "mode": "reference_loss", "metric": "conditional_ce",
                     "correct": None, "answered": result["nll_rows"], "accuracy": result["conditional_ce"],
                     "state": record["state"], "condition_sha256": digest({k:v for k,v in record["binding"].items() if k != "checkpoint_sha256"}),
                     "checkpoint_sha256": record["binding"]["checkpoint_sha256"], "note": "Conditional cross entropy, lower is better; not an accuracy."})
    keyed = {}
    for row in rows:
        key = (row["suite"], row["mode"], row["metric"])
        if key in keyed:
            raise ValueError(f"Duplicate result series under one root: {key}; choose one run directory")
        keyed[key] = row
    return keyed

def compare(before_root, after_root):
    before, after = collect(before_root), collect(after_root)
    result = []
    for key in sorted(before.keys() | after.keys()):
        a, b = before.get(key, {}), after.get(key, {})
        matched = bool(a and b and a.get("state") == b.get("state") == "complete" and
                       a.get("condition_sha256") and a["condition_sha256"] == b.get("condition_sha256") and
                       a.get("answered") == b.get("answered"))
        note = " | ".join(x for x in [a.get("note"), b.get("note")] if x)
        if matched and a.get("logger_dir") and b.get("logger_dir"):
            try:
                paired = comparison_rows([a["logger_dir"], b["logger_dir"]])[-1]
            except (OSError, ValueError, KeyError, TypeError) as error:
                matched = False
                note += f" | Paired comparison rejected: {type(error).__name__}: {error}"
            else:
                note += f" | wrong→right={paired['paired_improved']}; right→wrong={paired['paired_regressed']}"
        if not matched:
            note += " | Missing, incomplete or unmatched conditions; no improvement claim."
        delta = (b["accuracy"]-a["accuracy"])*100 if matched and a["accuracy"] is not None and b["accuracy"] is not None and key[2] != "conditional_ce" else None
        result.append(dict(zip(("suite", "mode", "metric"), key),
                     before=a.get("accuracy"), after=b.get("accuracy"), delta_pp=delta,
                     before_correct=a.get("correct"), after_correct=b.get("correct"),
                     before_denominator=a.get("answered"), after_denominator=b.get("answered"),
                     before_state=a.get("state", "not_measured"), after_state=b.get("state", "not_measured"),
                     comparable=matched, note=note.strip(" |")))
    return result

def retention_decision(rows):
    """Predeclared practical acceptance thresholds, not statistical proof."""
    def completed_delta(item):
        return bool(item and item.get("comparable") is True and
                    item.get("before_state") == item.get("after_state") == "complete" and
                    isinstance(item.get("delta_pp"), (int, float)) and math.isfinite(item["delta_pp"]) and
                    item.get("before_denominator") == item.get("after_denominator") and
                    isinstance(item.get("before_denominator"), int) and item["before_denominator"] > 0)
    required = [("T1", "final_answer_strict_match"), ("T1", "exact_step_termination"),
                ("T2", "end_to_end"), ("T5", "logical_operator_accuracy"), ("T4", "SWE_Software")]
    checks, complete = [], True
    for suite, metric in required:
        found = [x for x in rows if x["suite"] == suite and x["metric"] == metric and x["mode"] == "original"]
        item = found[0] if len(found) == 1 else None
        valid = completed_delta(item)
        complete &= valid
        checks.append({"suite": suite, "metric": metric, "measured_comparably": valid,
                       "delta_pp": item["delta_pp"] if valid else None,
                       "max_drop_pp": 2.0, "passed": item["delta_pp"] >= -2.0 - 1e-9 if valid else None})
    holds = complete and all(x["passed"] for x in checks)
    improvements = [x for x in rows if completed_delta(x) and
                    ((x["suite"] == "python120" and x["mode"] == "formal" and x["metric"] == "terminal_explicit_exact" and x["delta_pp"] >= 5.0 - 1e-9) or
                     (x["suite"] == "T3" and x["mode"] == "original" and x["metric"] == "logit_argmax" and x["delta_pp"] >= 2.0 - 1e-9) or
                     (x["suite"] == "T4" and x["mode"] == "original" and x["metric"] == "JMMLU_Academic" and x["delta_pp"] >= 2.0 - 1e-9))]
    return {"criteria_status": "predeclared_engineering_thresholds_not_significance_test", "retention_complete": complete,
            "retention_passed": holds, "checks": checks, "target_improvement_seen": bool(improvements),
            "proceed_to_next_small_experiment": bool(holds and improvements),
            "automatic_checkpoint_promotion": False,
            "limits": "This covers named regression metrics only. T8, search behavior and all raw failures require review; tiny suites do not establish general capability."}

def write_reports(before, after, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    rows = compare(before, after)
    if not rows:
        raise ValueError("No measured results found; a plan is not a model result")
    with (output / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS); writer.writeheader(); writer.writerows(rows)
    payload = {"rows": rows, "retention": retention_decision(rows), "before": str(before), "after": str(after)}
    (output / "comparison.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    lines = ["# GeoCPU 比較結果", "", "同じ条件で完了した測定だけに差分を表示します。未評価・計測エラーは0点にしません。", "",
             "| 評価 | モード・指標 | 前 | 後 | 差(pp) | 状態 |", "|---|---|---:|---:|---:|---|"]
    for r in rows:
        def val(v): return "未評価" if v is None else (f"{v:.4f}" if r["metric"] == "conditional_ce" else f"{100*v:.2f}%")
        lines.append(f"| {r['suite']} | {r['mode']} / {r['metric']} | {val(r['before'])} | {val(r['after'])} | {r['delta_pp'] if r['delta_pp'] is not None else '—'} | {r['before_state']} → {r['after_state']} |")
    lines += ["", "T8は形式、conditional_ceは参照文の損失です。タスク正答率とは別の指標です。", "",
              "## 保持判定", "", "```json", json.dumps(payload["retention"], ensure_ascii=False, indent=2), "```"]
    (output / "comparison.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return payload

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--before", type=Path, required=True); p.add_argument("--after", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(); result = write_reports(a.before, a.after, a.output)
    print(json.dumps(result["retention"], ensure_ascii=False, indent=2))
