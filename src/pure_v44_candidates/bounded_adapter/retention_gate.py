"""Promote only after matched raw answers and an independent per-answer grade."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from adapter import file_sha256
from admission import (require_independent_execution_verification,
                       require_v43_native_baseline, verify_dataset)
from blind_packet import GRADE_SCHEMA, PACKET_SCHEMA, PIN_SCHEMA


SCHEMA = "geocpu_bounded_adapter_retention_gate_v1"


def _read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _index(records):
    result = {}
    for row in records:
        key = (row.get("arm"), row.get("split"), row.get("id"))
        if None in key or key in result:
            raise ValueError("Duplicate or incomplete paired answer/grade key")
        result[key] = row
    return result


def build_gate(*, data_dir, data_manifest_sha256, paired_dir, packet_dir,
               grades_path, grades_sha256, grade_pin_path, grade_pin_sha256,
               adapter_sha256, verification_proof_dir, verification_proof_sha256,
               meta_sha256, source_manifest_sha256, v43_native_proof_dir,
               v43_native_capture_sha256, v43_native_grade_sha256):
    admission = verify_dataset(data_dir, data_manifest_sha256)
    verification = require_independent_execution_verification(
        admission, data_dir=data_dir, proof_dir=verification_proof_dir,
        expected_proof_sha256=verification_proof_sha256,
        expected_meta_sha256=meta_sha256,
        expected_source_manifest_sha256=source_manifest_sha256)
    v43_native = require_v43_native_baseline(
        data_dir, proof_dir=v43_native_proof_dir,
        expected_capture_sha256=v43_native_capture_sha256,
        expected_grade_sha256=v43_native_grade_sha256)
    paired_dir = Path(paired_dir)
    paired = json.loads((paired_dir / "paired_result.json").read_text(encoding="utf-8"))
    raw_path = paired_dir / "paired_raw_answers.jsonl"
    raw_sha = file_sha256(raw_path)
    if (paired.get("status") != "COMPLETE"
            or paired.get("schema") != "geocpu_bounded_adapter_paired_eval_v1"
            or paired.get("adapter_sha256") != adapter_sha256
            or paired.get("data_manifest_sha256") != data_manifest_sha256
            or paired.get("raw_answer_sha256") != raw_sha
            or paired.get("independent_verification") != verification
            or paired.get("v43_native_baseline") != v43_native
            or paired.get("max_new_tokens") != 2048):
        raise ValueError("Paired raw-answer result is incomplete or unbound")
    raw = _index(_read_jsonl(raw_path))
    expected_ids = {(arm, split, row["id"])
                    for split in ("old_skill_eval", "unseen_eval")
                    for row in _read_jsonl(Path(data_dir) / (split + ".jsonl"))
                    for arm in ("parent", "adapter")}
    if set(raw) != expected_ids or len(raw) != paired.get("records"):
        raise ValueError("Paired answers do not cover the fixed evaluation suites exactly")
    packet_dir = Path(packet_dir)
    if file_sha256(grade_pin_path) != grade_pin_sha256:
        raise ValueError("Blind grade pin SHA differs")
    pin = json.loads(Path(grade_pin_path).read_text(encoding="utf-8"))
    packet_manifest_path = packet_dir / "packet_manifest.json"
    packet = json.loads(packet_manifest_path.read_text(encoding="utf-8"))
    if (pin.get("schema") != PIN_SCHEMA or packet.get("schema") != PACKET_SCHEMA
            or packet.get("data_manifest_sha256") != data_manifest_sha256
            or pin.get("grade_sha256") != grades_sha256
            or pin.get("packet_manifest_sha256") != file_sha256(packet_manifest_path)
            or pin.get("packet_sha256") != packet.get("packet_sha256")
            or pin.get("raw_answer_sha256") != raw_sha
            or packet.get("raw_answer_sha256") != raw_sha
            or file_sha256(packet_dir / "packet.jsonl") != packet.get("packet_sha256")
            or file_sha256(packet_dir / "rolemap.jsonl") != packet.get("rolemap_sha256")
            or pin.get("rolemap_opened_by_pin") is not False):
        raise ValueError("Anonymous packet or grade-first pin differs")
    if file_sha256(grades_path) != grades_sha256:
        raise ValueError("Independent grades SHA differs")
    grade_file = json.loads(Path(grades_path).read_text(encoding="utf-8"))
    if (grade_file.get("schema") != GRADE_SCHEMA
            or not isinstance(grade_file.get("grader_id"), str) or not grade_file["grader_id"]):
        raise ValueError("Missing externally pinned anonymous grading provenance")
    roles = {row["anonymous_id"]: row for row in _read_jsonl(packet_dir / "rolemap.jsonl")}
    packets = {row["anonymous_id"]: row for row in _read_jsonl(packet_dir / "packet.jsonl")}
    if len(roles) != packet.get("records") or len(packets) != packet.get("records") or set(roles) != set(packets):
        raise ValueError("Anonymous packet IDs are not unique or complete")
    eval_rows = {(split, row["id"]): row for split in ("old_skill_eval", "unseen_eval")
                 for row in _read_jsonl(Path(data_dir) / (split + ".jsonl"))}
    for anon, role in roles.items():
        key = (role.get("arm"), role.get("split"), role.get("id"))
        case = eval_rows.get((role.get("split"), role.get("id")))
        answer = raw.get(key)
        visible = packets[anon]
        if (case is None or answer is None
                or visible.get("prompt") != case["prompt"]
                or visible.get("expected_answer") != case["expected_answer"]
                or visible.get("raw_output") != answer.get("raw_output")
                or visible.get("output_sha256") != answer.get("output_sha256")):
            raise ValueError("Anonymous packet content differs from authenticated evaluation cases")
    normalized = []
    seen_anonymous = set()
    for grade in grade_file.get("records", []):
        anon = grade.get("anonymous_id")
        if anon in seen_anonymous or anon not in roles or grade.get("output_sha256") != packets[anon].get("output_sha256"):
            raise ValueError("Anonymous grade ID or raw answer differs")
        seen_anonymous.add(anon)
        normalized.append({**grade, "arm": roles[anon]["arm"], "split": roles[anon]["split"], "id": roles[anon]["id"]})
    if seen_anonymous != set(roles):
        raise ValueError("Anonymous grades do not cover all sealed answers")
    grades = _index(normalized)
    if set(grades) != expected_ids:
        raise ValueError("Pinned anonymous grades do not cover every raw answer exactly")
    for key, row in raw.items():
        grade = grades[key]
        if roles[grade["anonymous_id"]].get("output_sha256") != row.get("output_sha256"):
            raise ValueError("Role map does not bind to the paired raw answer")
        if row.get("status") != "observed" or row.get("error") is not None:
            raise ValueError("An answer is partial or failed")
        if grade.get("output_sha256") != row.get("output_sha256"):
            raise ValueError("A grade is not bound to its raw answer")
        if type(grade.get("correct")) is not bool:
            raise ValueError("Correctness grade must be a boolean")
        for axis in ("japanese_fluency", "reasoning", "repetition_control"):
            if type(grade.get(axis)) is not int or grade[axis] not in (0, 1, 2):
                raise ValueError("Grade axes must be integers from 0 to 2")

    old_keys = [key for key in expected_ids if key[0] == "adapter" and key[1] == "old_skill_eval"]
    old_rows = {row["id"]: row for row in _read_jsonl(Path(data_dir) / "old_skill_eval.jsonl")}
    old_cpu_keys = [key for key in old_keys if old_rows[key[2]]["v43_cpu_task"] is True]
    unseen_keys = [key for key in expected_ids if key[0] == "adapter" and key[1] == "unseen_eval"]
    old_regressions = sum(not grades[key]["correct"] for key in old_keys)
    unseen_parent_correct = sum(grades[("parent", key[1], key[2])]["correct"] for key in unseen_keys)
    unseen_adapter_correct = sum(grades[key]["correct"] for key in unseen_keys)
    eos_parent = sum(raw[("parent", key[1], key[2])]["ended_with_eos"] is True for key in unseen_keys)
    eos_adapter = sum(raw[key]["ended_with_eos"] is True for key in unseen_keys)
    python_success = sum(int(raw[key].get("successful_tool_invocations") or 0) for key in unseen_keys)
    old_cpu_python_success = sum(int(raw[key].get("successful_tool_invocations") or 0) for key in old_cpu_keys)
    old_cpu_correct = sum(grades[key]["correct"] for key in old_cpu_keys)
    # Japanese quality is assessed only on the separately identified prose rows.
    unseen_rows = {row["id"]: row for row in _read_jsonl(Path(data_dir) / "unseen_eval.jsonl")}
    japanese_keys = [key for key in unseen_keys if unseen_rows[key[2]]["bucket"] == "japanese_prose"]
    japanese = {}
    for axis in ("japanese_fluency", "reasoning", "repetition_control"):
        japanese[axis] = {arm: sum(grades[(arm, key[1], key[2])][axis] for key in japanese_keys)
                          for arm in ("parent", "adapter")}
    checks = {
        "all_original_v43_solved_old_skill_items_preserved": old_regressions == 0,
        "old_v43_cpu_tasks_semantically_correct_with_actual_python": (
            len(old_cpu_keys) >= 12 and old_cpu_correct == len(old_cpu_keys)
            and all(int(raw[key].get("successful_tool_invocations") or 0) > 0
                    for key in old_cpu_keys)),
        "unseen_correct_answers_increase": unseen_adapter_correct > unseen_parent_correct,
        "unseen_natural_eos_not_lower": eos_adapter >= eos_parent,
        "actual_python_execution_succeeds": paired.get("model_python_execution") is True and python_success > 0,
        "japanese_fluency_not_lower": japanese["japanese_fluency"]["adapter"] >= japanese["japanese_fluency"]["parent"],
        "reasoning_not_lower": japanese["reasoning"]["adapter"] >= japanese["reasoning"]["parent"],
        "repetition_control_not_lower": japanese["repetition_control"]["adapter"] >= japanese["repetition_control"]["parent"],
        "at_least_one_japanese_reasoning_answer": any(grades[key]["reasoning"] > 0 for key in japanese_keys),
    }
    return {"schema": SCHEMA, "status": "PASS" if all(checks.values()) else "FAIL",
            "scope": "NEXT_BOUNDED_STAGE_ONLY", "public_adoption_approved": False,
            "tier_1_to_8_full_evaluation_status": "NOT_RUN",
            "checkpoint_sha256": adapter_sha256, "data_manifest_sha256": data_manifest_sha256,
            "raw_answer_sha256": raw_sha, "grades_sha256": grades_sha256,
            "grade_pin_sha256": grade_pin_sha256,
            "packet_manifest_sha256": file_sha256(packet_manifest_path),
            "grader_id": grade_file["grader_id"], "old_skill_evaluated": len(old_keys),
            "old_skill_regressions": old_regressions, "unseen_evaluated": len(unseen_keys),
            "old_v43_cpu_tasks_evaluated": len(old_cpu_keys),
            "old_v43_cpu_tasks_correct": old_cpu_correct,
            "old_v43_cpu_tasks_successful_python_calls": old_cpu_python_success,
            "unseen_parent_correct": unseen_parent_correct,
            "unseen_adapter_correct": unseen_adapter_correct,
            "unseen_natural_eos": {"parent": eos_parent, "adapter": eos_adapter},
            "unseen_successful_python_calls": python_success,
            "japanese_prose_evaluated": len(japanese_keys), "japanese_grade_sums": japanese,
            "checks": checks, "dataset_admission": admission,
            "independent_execution_verification": verification,
            "v43_native_baseline": v43_native}


def main(argv=None):
    parser = argparse.ArgumentParser()
    for name in ("data_dir", "data_manifest_sha256", "paired_dir", "packet_dir",
                 "grades", "grades_sha256", "grade_pin", "grade_pin_sha256",
                 "adapter_sha256", "verification_proof_dir", "verification_proof_sha256",
                 "meta_sha256", "source_manifest_sha256", "output"):
        parser.add_argument("--" + name.replace("_", "-"), required=True)
    # Separately captured native-v43 full answers, not an authored correct flag.
    for name in ("v43_native_proof_dir", "v43_native_capture_sha256", "v43_native_grade_sha256"):
        parser.add_argument("--" + name.replace("_", "-"), required=True)
    args = parser.parse_args(argv)
    proof = build_gate(data_dir=args.data_dir, data_manifest_sha256=args.data_manifest_sha256,
                       paired_dir=args.paired_dir, packet_dir=args.packet_dir,
                       grades_path=args.grades, grades_sha256=args.grades_sha256,
                       grade_pin_path=args.grade_pin, grade_pin_sha256=args.grade_pin_sha256,
                       adapter_sha256=args.adapter_sha256,
                       verification_proof_dir=args.verification_proof_dir,
                       verification_proof_sha256=args.verification_proof_sha256,
                       meta_sha256=args.meta_sha256,
                       source_manifest_sha256=args.source_manifest_sha256,
                       v43_native_proof_dir=args.v43_native_proof_dir,
                       v43_native_capture_sha256=args.v43_native_capture_sha256,
                       v43_native_grade_sha256=args.v43_native_grade_sha256)
    with Path(args.output).open("x", encoding="utf-8") as stream:
        json.dump(proof, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"status": proof["status"], "checks": proof["checks"],
                      "proof_sha256": file_sha256(args.output)}, ensure_ascii=False))
    return proof


if __name__ == "__main__":
    main()
