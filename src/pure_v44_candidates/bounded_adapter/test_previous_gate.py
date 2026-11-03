"""CPU-only checks that a prior-stage PASS cannot bypass raw-evidence replay."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import retention_gate
import train_stage
from adapter import file_sha256


PATH_KEYS = ("data_dir", "paired_dir", "packet_dir", "grades_path",
             "grade_pin_path", "verification_proof_dir", "v43_native_proof_dir")


class PreviousGateTests(unittest.TestCase):
    def test_relocated_evidence_is_replayed_and_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            moved = {}
            for key in PATH_KEYS:
                path = root / key
                if key.endswith("_path"):
                    path.write_text("fixture", encoding="utf-8")
                else:
                    path.mkdir()
                moved[key] = str(path)
            relocation = root / "relocation.json"
            relocation.write_text(json.dumps(moved), encoding="utf-8")
            input_paths = {key: "/previous/kaggle/" + key for key in PATH_KEYS}
            proof = {
                "schema": retention_gate.SCHEMA, "status": "PASS",
                "scope": "NEXT_BOUNDED_STAGE_ONLY", "public_adoption_approved": False,
                "tier_1_to_8_full_evaluation_status": "NOT_RUN",
                "checkpoint_sha256": "a" * 64, "data_manifest_sha256": "b" * 64,
                "old_skill_regressions": 0, "unseen_evaluated": 64,
                "raw_answer_sha256": "c" * 64,
                "checks": {"old_v43_natural_eos_retained_on_every_item": True,
                           "old_v43_cpu_tasks_semantically_correct_with_actual_python": True},
                "retention_gate_source_sha256": file_sha256(retention_gate.__file__),
                "evidence_inputs": {**input_paths, "data_manifest_sha256": "b" * 64},
            }
            gate_file = root / "gate.json"
            gate_file.write_text(json.dumps(proof), encoding="utf-8")
            args = (gate_file, file_sha256(gate_file), "a" * 64, "b" * 64,
                    relocation, file_sha256(relocation))
            replayed = {**proof, "evidence_inputs": {**proof["evidence_inputs"], **moved}}
            with patch.object(retention_gate, "build_gate", return_value=replayed) as verifier:
                train_stage._verify_previous_gate(*args)
                self.assertEqual(verifier.call_args.kwargs["data_dir"], moved["data_dir"])
            false_pass = {**replayed, "raw_answer_sha256": "d" * 64}
            with patch.object(retention_gate, "build_gate", return_value=false_pass):
                with self.assertRaisesRegex(ValueError, "cannot be reproduced"):
                    train_stage._verify_previous_gate(*args)
            with self.assertRaisesRegex(ValueError, "relocation map"):
                train_stage._verify_previous_gate(*args[:-1], "0" * 64)


if __name__ == "__main__":
    unittest.main()
