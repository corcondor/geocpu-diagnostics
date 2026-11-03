"""CPU-only admission checks on tiny synthetic files; no model opened."""

from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import admission
from adapter import file_sha256


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.train = [
            {"id": "t1", "family_id": "f1", "bucket": "converted_legacy_python",
             "prompt": "問題 1", "target": "答え 1",
             "source_ref": {"kind": "v43_zip_member", "file": "data/train_v43_knowledge_unified.jsonl",
                            "line": 1, "raw_sha256": "1" * 64}},
            {"id": "t2", "family_id": "f2", "bucket": "japanese_prose",
             "prompt": "日本語 2", "target": "答え 2",
             "source_ref": {"kind": "public_json", "file": "japanese/public.json", "line": 2,
                            "raw_sha256": "2" * 64}},
            {"id": "t3", "family_id": "f3", "bucket": "v43_tool_free_anchor",
             "prompt": "日本の歴史 3", "target": "説明 3",
             "source_ref": {"kind": "v43_zip_member", "file": "data/train_v43_knowledge_unified.jsonl",
                            "line": 3, "raw_sha256": "3" * 64}},
        ]
        self.old = [{"id": "o1", "family_id": "fo", "bucket": "old_skill",
                     "prompt": "昔の問題", "expected_answer": "3",
                     "v43_cpu_task": True,
                     "v43_original_correct": True,
                     "v43_original_raw_sha256": hashlib.sha256("3".encode()).hexdigest()}]
        self.unseen = [{"id": "u1", "family_id": "fu", "bucket": "japanese_prose",
                        "prompt": "新しい問題", "expected_answer": "4"}]
        for row in self.train + self.old + self.unseen:
            row["family_id"] = admission.family_id_for_prompt(row["prompt"])

    def _materialize(self):
        files = {"train.jsonl": self.train, "old_skill_eval.jsonl": self.old,
                 "unseen_eval.jsonl": self.unseen,
                 "v43_original_answers.jsonl": [{"id": "o1", "profile": "v43_cpu_native",
                     "parent_sha256": "a4b95d7d0f486146196271b3b1fbc6a1b6b3342d8fadd590320fc3842c1e80e5",
                     "prompt_sha256": hashlib.sha256("昔の問題".encode()).hexdigest(),
                     "raw_output": "3", "raw_output_sha256": hashlib.sha256("3".encode()).hexdigest(),
                     "expected_answer": "3", "correct": True}]}
        for name, rows in files.items():
            _write_jsonl(self.root / name, rows)
        manifest = {"schema": admission.SCHEMA,
                    "files": {name: {"sha256": file_sha256(self.root / name)} for name in files},
                    "quality_audit": {
                        "all_tool_observations_executed_and_matched": True,
                        "all_legacy_cpu_python_traces_matched": True,
                        "source_family_separation_checked": True},
                    "source_files": {"v43_zip_sha256": admission.PINNED_V43_DATA_ZIP_SHA256},
                    "evaluation_input_pins": [
                        {"path": name, "sha256": sha}
                        for name, sha in admission.REQUIRED_V43_EVAL_SHA256.items()],
                    "verification": {"converter_source_sha256": "e" * 64,
                                     "verifier_source_sha256": admission.VERIFIER_SOURCE_SHA256},
                    "counts": {"train_rows": 3, "train_converted_legacy_python": 1,
                               "train_japanese_prose": 1, "python_candidate_rows": 1,
                               "train_v43_tool_free_anchor": 1,
                               "old_skill_eval_rows": 1, "old_skill_eval_old_skill": 1,
                               "old_skill_eval_original_cpu_tasks": 1,
                               "unseen_eval_rows": 1, "unseen_eval_japanese_prose": 1,
                               "v43_original_answers_rows": 1}}
        (self.root / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return file_sha256(self.root / "dataset_manifest.json")

    def _verify(self, sha):
        with (patch.object(admission, "MIN_UNIQUE_TRAIN", 3),
              patch.object(admission, "MIN_PYTHON_CANDIDATES", 1),
              patch.object(admission, "MIN_PUBLIC_TIR", 0),
              patch.object(admission, "MIN_CONVERTED_CPU", 1),
              patch.object(admission, "MIN_PUBLIC_MULTILINGUAL_CODE", 0),
              patch.object(admission, "MIN_JAPANESE", 1),
              patch.object(admission, "MIN_V43_TOOL_FREE_ANCHORS", 1),
              patch.object(admission, "MIN_EVAL_PER_SPLIT", 1),
              patch.object(admission, "MIN_UNSEEN_JAPANESE", 1),
              patch.object(admission, "MIN_OLD_CPU_CASES", 1)):
            return admission.verify_dataset(self.root, sha)

    def test_sealed_data_and_missing_verification(self):
        sha = self._materialize()
        structural = self._verify(sha)
        self.assertEqual(structural["counts"]["train_rows"], 3)
        self.assertEqual(structural["execution_verification"], "UNVERIFIED_SELF_ATTESTATION_ONLY")
        with self.assertRaisesRegex(RuntimeError, "Independent full-row"):
            admission.require_independent_execution_verification(
                structural, data_dir=self.root, proof_dir=None,
                expected_proof_sha256=None, expected_meta_sha256=admission.PINNED_V43_META_SHA256,
                expected_source_manifest_sha256="a" * 64)
        self.train[0].pop("source_ref")
        sha = self._materialize()
        with self.assertRaisesRegex(ValueError, "original source reference"):
            self._verify(sha)

    def test_prompt_overlap_rejected(self):
        self.unseen[0]["prompt"] = "ユーザー: 問題 1\nアシスタント:"
        self.unseen[0]["family_id"] = admission.family_id_for_prompt(self.unseen[0]["prompt"])
        sha = self._materialize()
        with self.assertRaisesRegex(ValueError, "Evaluation prompt overlap"):
            self._verify(sha)

    def test_independent_receipts_bind_every_row(self):
        sha = self._materialize()
        data = self._verify(sha)
        proof_dir = self.root / "proof"
        proof_dir.mkdir()
        with (self.root / "train.jsonl").open("rb") as source, (proof_dir / "receipts.jsonl").open("wb") as out:
            for line in source:
                row = json.loads(line)
                receipt = {"id": row["id"], "train_row_sha256": hashlib.sha256(line).hexdigest(),
                           "source_row_sha256": row["source_ref"]["raw_sha256"],
                           "bucket": row["bucket"],
                           "tool_calls_replayed": int(row["bucket"] == "converted_legacy_python"),
                           "observations_sha256": "f" * 64}
                out.write((json.dumps(receipt) + "\n").encode())
        proof = {"schema": admission.VERIFIER_SCHEMA, "status": "ALL_ROWS_REPLAYED",
                 "verifier_source_sha256": admission.VERIFIER_SOURCE_SHA256,
                 "converter_source_sha256": "e" * 64,
                 "v43_zip_sha256": admission.PINNED_V43_DATA_ZIP_SHA256,
                 "data_manifest_sha256": sha, "train_sha256": data["files_sha256"]["train.jsonl"],
                 "meta_sha256": admission.PINNED_V43_META_SHA256,
                 "pure_source_manifest_sha256": "a" * 64,
                 "train_rows": 3,
                 "receipts_sha256": file_sha256(proof_dir / "receipts.jsonl"), "replayed_tool_calls": 1}
        (proof_dir / "proof.json").write_text(json.dumps(proof), encoding="utf-8")
        proof_sha = file_sha256(proof_dir / "proof.json")
        kwargs = dict(data_dir=self.root, proof_dir=proof_dir,
                      expected_proof_sha256=proof_sha,
                      expected_meta_sha256=admission.PINNED_V43_META_SHA256,
                      expected_source_manifest_sha256="a" * 64)
        self.assertEqual(admission.require_independent_execution_verification(data, **kwargs)["verified_rows"], 3)
        receipts = (proof_dir / "receipts.jsonl").read_text(encoding="utf-8").replace('"id": "t1"', '"id": "wrong"')
        (proof_dir / "receipts.jsonl").write_text(receipts, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "incomplete or has wrong lineage"):
            admission.require_independent_execution_verification(data, **kwargs)

    def test_native_v43_full_answer_and_external_grade_are_required(self):
        self._materialize()
        with self.assertRaisesRegex(RuntimeError, "Complete pinned"):
            admission.require_v43_native_baseline(self.root, proof_dir=None,
                expected_capture_sha256=None, expected_grade_sha256=None)
        proof_dir = self.root / "original_native"
        proof_dir.mkdir()
        full, audits, grades = [], [], []
        for index in range(200):
            ident = "o1" if index == 0 else f"other_{index}"
            prompt = "昔の問題" if index == 0 else f"別の問題 {index}"
            raw = "3" if index == 0 else "0"
            correct = index == 0
            answer = {"id": ident, "source_index": index, "profile": "v43_cpu_native",
                      "source_member_sha256": admission.PINNED_V43_SWE_SOURCE_SHA256,
                      "parent_sha256": admission.PINNED_V43_PARENT_SHA256,
                      "full_answer_captured": True, "reference_given_to_generation": False,
                      "generation_limit": 2048, "prompt": prompt,
                      "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                      "raw_output": raw, "raw_output_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                      "expected_answer": "3", "cpu_invocations": 1,
                      "cpu_trace": [], "ended_with_eos": True}
            audit = {"id": ident, "raw_output": raw, "target": "3",
                     "cpu_invocations": 1, "ended_with_eos": True,
                     "is_correct": correct, "final_answer": raw,
                     "expected_loops": 1, "executed_loops": 1,
                     "step_category": "fixture"}
            grade = {"id": ident, "prompt_sha256": answer["prompt_sha256"],
                     "raw_output_sha256": answer["raw_output_sha256"],
                     "expected_answer": "3", "correct": correct,
                     "scorer_sha256": admission.PINNED_V43_SWE_EVALUATOR_SHA256,
                     "final_answer": raw, "expected_loops": 1, "executed_loops": 1,
                     "step_category": "fixture", "reason": "exact fixture"}
            full.append(answer)
            audits.append(audit)
            grades.append(grade)
        capture, grade = full[0], grades[0]
        _write_jsonl(proof_dir / "swe200_full_answers.jsonl", full)
        _write_jsonl(proof_dir / "original_swe_audit.jsonl", audits)
        _write_jsonl(proof_dir / "swe200_original_grades.jsonl", grades)
        (proof_dir / "claim.json").write_text(json.dumps({
            "source_zip_sha256": admission.PINNED_V43_DATA_ZIP_SHA256,
            "parent_sha256": admission.PINNED_V43_PARENT_SHA256,
            "meta_sha256": admission.PINNED_V43_META_SHA256,
            "eval_member_sha256": admission.PINNED_V43_SWE_SOURCE_SHA256,
            "original_evaluator_sha256": admission.PINNED_V43_SWE_EVALUATOR_SHA256,
            "script_sha256": admission.PINNED_V43_CAPTURE_SOURCE_SHA256,
            "suite": "heldout_swe200", "generation_limit": 2048,
            "training_updates": 0, "parent_checkpoint_modified": False}), encoding="utf-8")
        _write_jsonl(proof_dir / "v43_native_full_answers.jsonl", [capture])
        _write_jsonl(proof_dir / "v43_native_external_grades.jsonl", [grade])
        for name in ("old_skill_eval.jsonl", "v43_original_answers.jsonl"):
            (proof_dir / name).write_bytes((self.root / name).read_bytes())
        accepted = admission.require_v43_native_baseline(
            self.root, proof_dir=proof_dir,
            expected_capture_sha256=file_sha256(proof_dir / "v43_native_full_answers.jsonl"),
            expected_grade_sha256=file_sha256(proof_dir / "v43_native_external_grades.jsonl"))
        self.assertEqual(accepted["old_skill_originally_correct"], 1)


if __name__ == "__main__":
    unittest.main()
