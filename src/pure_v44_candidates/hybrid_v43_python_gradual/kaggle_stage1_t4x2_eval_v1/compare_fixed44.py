"""Compare complete raw generations on identical fixed case/profile keys only."""
from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path

from eval_fixed_diagnostic import _validate_saved_rows
from source_pools import PINNED_V43_ZIP_SHA, file_sha
from train_hybrid_stage import PARENT_CHECKPOINT_SHA, SOURCE_OVERLAY_SHA

CASES_SHA = "4032a2c17de2ce56a7758cd35c16db2c2b357ce76b394522411e86c0aa682b45"
EVALUATOR_SHA = "bfce8f14f28e8e3714d30234da445e3d2c1e47a97e8666ca0a959e0dad630d8b"
PROFILE = "eval_fp16_native_v1"
COUNTS = {"native": 44, "python_cued": 20, "python_uncued": 20}
SCORE_FIELDS = (
    "old_original_evaluator_correct", "old_cpu_native_rule_correct",
    "old_cpu_native_rule_and_natural_eos_correct", "native_natural_eos",
    "python_executed_and_numeric_correct", "python_final_answer_numeric_correct",
    "python_japanese_explanation_present", "python_execution_and_final_answer_correct",
    "python_natural_eos", "python_execution_final_and_eos_correct",
)


def _cases(path: Path) -> list[dict]:
    if file_sha(path) != CASES_SHA:
        raise ValueError("Fixed 44-case source differs")
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    runs = [(case, profile) for case in cases for profile in case["profiles"]]
    if len(cases) != 44 or len(runs) != 84 or len({(case["id"], profile)
                                                  for case, profile in runs}) != 84:
        raise ValueError("Fixed case/profile universe is incomplete")
    return runs


def _load_arm(directory: Path, arm: str, checkpoint_sha: str, runs: list) -> tuple[dict, list[dict]]:
    complete_path = directory / "COMPLETE.json"
    raw_path = directory / "raw_answers_44_profiles.jsonl"
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    overlay_sha = sha256(json.dumps(SOURCE_OVERLAY_SHA, sort_keys=True).encode()).hexdigest()
    if (complete.get("schema") != "geocpu_v43_python_gradual_raw_diagnostic_v1"
            or complete.get("status") != "COMPLETE" or complete.get("arm") != arm
            or complete.get("checkpoint_kind") != ("parent" if arm == "parent" else "hybrid_250")
            or complete.get("checkpoint_sha256") != checkpoint_sha
            or complete.get("v43_zip_sha256") != PINNED_V43_ZIP_SHA
            or complete.get("cases_sha256") != CASES_SHA
            or complete.get("source_sha256") != EVALUATOR_SHA
            or complete.get("overlay_sha256") != overlay_sha
            or complete.get("precision_profile") != PROFILE
            or complete.get("profile_counts") != COUNTS
            or complete.get("max_generated_tokens") != 2048
            or complete.get("training_updates") != 0
            or complete.get("raw_answers_sha256") != file_sha(raw_path)):
        raise ValueError("Incomplete or unmatched raw-answer closure: " + arm)
    raw = raw_path.read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError("Raw-answer file has a torn final row: " + arm)
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    if len(rows) != len(runs):
        raise ValueError("Raw-answer case/profile count differs: " + arm)
    _validate_saved_rows(rows, runs, arm=arm, checkpoint_sha=checkpoint_sha,
                         evaluator_sha=EVALUATOR_SHA, overlay_sha=overlay_sha,
                         precision_profile=PROFILE)
    return complete, rows


def compare(*, parent_dir: Path, control_dir: Path, python_dir: Path,
            checkpoint_sha: dict[str, str], cases_path: Path, output_dir: Path) -> dict:
    if (output_dir / "comparison.json").exists() or (output_dir / "matched_rows_84.jsonl").exists():
        raise FileExistsError("Comparison output is one-use")
    runs = _cases(cases_path)
    expected_sha = {"parent": PARENT_CHECKPOINT_SHA, **checkpoint_sha}
    inputs = {"parent": parent_dir, "control": control_dir, "python": python_dir}
    loaded = {arm: _load_arm(directory, arm, expected_sha[arm], runs)
              for arm, directory in inputs.items()}
    metrics = defaultdict(lambda: {"cases": 0, "parent_true": 0,
                                   "control_true": 0, "python_true": 0,
                                   "control_gain_vs_parent": 0, "control_loss_vs_parent": 0,
                                   "python_gain_vs_parent": 0, "python_loss_vs_parent": 0,
                                   "python_gain_vs_control": 0, "python_loss_vs_control": 0})
    matched_path = output_dir / "matched_rows_84.jsonl"
    with matched_path.open("x", encoding="utf-8", newline="\n") as sink:
        for index, (case, profile) in enumerate(runs):
            rows = {arm: loaded[arm][1][index] for arm in inputs}
            if any((row["case_id"], row["profile"], row["prompt"], row["expected_answer"])
                   != (rows["parent"]["case_id"], rows["parent"]["profile"],
                       rows["parent"]["prompt"], rows["parent"]["expected_answer"])
                   for row in rows.values()):
                raise ValueError("Case/profile/prompt or target differs between arms")
            item = {"case_id": case["id"], "group": case["group"], "profile": profile,
                    "prompt_sha256": sha256(rows["parent"]["prompt"].encode()).hexdigest(),
                    "expected_answer": case["target"], "arms": {}}
            for arm, row in rows.items():
                item["arms"][arm] = {
                    "raw_answer": row["raw_answer"],
                    "raw_answer_sha256": row["raw_answer_sha256"],
                    "model_generated_text": row["model_generated_text"],
                    "generated_tokens": row["generated_tokens"],
                    "ended_with_eos": row["ended_with_eos"],
                    "scores": {name: row.get(name) for name in SCORE_FIELDS
                               if type(row.get(name)) is bool},
                }
            for field in SCORE_FIELDS:
                values = {arm: rows[arm].get(field) for arm in inputs}
                if not all(type(value) is bool for value in values.values()):
                    continue
                key = "|".join((case["group"], profile, field))
                result = metrics[key]
                result["cases"] += 1
                for arm in inputs:
                    result[arm + "_true"] += int(values[arm])
                for arm in ("control", "python"):
                    result[arm + "_gain_vs_parent"] += int(values[arm] and not values["parent"])
                    result[arm + "_loss_vs_parent"] += int(values["parent"] and not values[arm])
                result["python_gain_vs_control"] += int(values["python"] and not values["control"])
                result["python_loss_vs_control"] += int(values["control"] and not values["python"])
            sink.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    record = {
        "schema": "geocpu_stage1_paired_fixed44_comparison_v1", "status": "COMPLETE",
        "matched_case_profiles": 84, "cases_sha256": CASES_SHA,
        "precision_profile": PROFILE, "max_generated_tokens": 2048,
        "checkpoint_sha256": expected_sha,
        "complete_sha256": {arm: file_sha(inputs[arm] / "COMPLETE.json") for arm in inputs},
        "raw_answers_sha256": {arm: file_sha(inputs[arm] / "raw_answers_44_profiles.jsonl")
                               for arm in inputs},
        "matched_rows_sha256": file_sha(matched_path),
        "metrics": dict(sorted(metrics.items())),
        "ungraded_groups": ["search8", "logic8", "japanese_explanation8"],
        "stage_promotion": False,
    }
    with (output_dir / "comparison.json").open("x", encoding="utf-8") as sink:
        json.dump(record, sink, ensure_ascii=False, sort_keys=True, indent=2)
        sink.write("\n")
    return record
