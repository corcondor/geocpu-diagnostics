"""Small packet and grade-pin test; no model, checkpoint or GPU."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from adapter import file_sha256
from blind_packet import GRADE_SCHEMA, create_packet, pin_grades


def _jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class BlindPacketTests(unittest.TestCase):
    def test_anonymous_packet_is_pinned_before_role_lookup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, paired = root / "data", root / "paired"
            data.mkdir()
            paired.mkdir()
            _jsonl(data / "old_skill_eval.jsonl", [{"id": "o", "prompt": "Q?", "expected_answer": "A"}])
            _jsonl(data / "unseen_eval.jsonl", [])
            answers = []
            for arm in ("parent", "adapter"):
                raw_output = "A" if arm == "parent" else "B"
                answers.append({"arm": arm, "split": "old_skill_eval", "id": "o",
                                "status": "observed", "raw_output": raw_output,
                                "output_sha256": hashlib.sha256(raw_output.encode()).hexdigest(),
                                "ended_with_eos": True, "tool_trace": []})
            _jsonl(paired / "paired_raw_answers.jsonl", answers)
            (paired / "paired_result.json").write_text(json.dumps({
                "status": "COMPLETE", "raw_answer_sha256": file_sha256(paired / "paired_raw_answers.jsonl"),
                "data_manifest_sha256": "d" * 64}),
                encoding="utf-8")
            packet_dir = root / "packet"
            with patch("blind_packet.verify_dataset", return_value={}):
                created = create_packet(data_dir=data, data_manifest_sha256="d" * 64,
                                        paired_dir=paired, output_dir=packet_dir)
            packet = [json.loads(line) for line in (packet_dir / "packet.jsonl").read_text().splitlines()]
            self.assertEqual(len(packet), 2)
            self.assertTrue(all("arm" not in row and "split" not in row and "id" not in row for row in packet))
            grades = {"schema": GRADE_SCHEMA, "grader_id": "test-independent-grader",
                      "records": [{"anonymous_id": row["anonymous_id"],
                                   "output_sha256": row["output_sha256"], "correct": row["raw_output"] == "A",
                                   "japanese_fluency": 1, "reasoning": 0, "repetition_control": 2,
                                   "reason": "tiny fixture"} for row in packet]}
            grade_path = root / "grade.json"
            grade_path.write_text(json.dumps(grades), encoding="utf-8")
            pinned = pin_grades(packet_dir=packet_dir, grades_path=grade_path,
                                output_path=root / "grade_pin.json")
            self.assertEqual(pinned["grade_sha256"], file_sha256(grade_path))
            self.assertEqual(created["rolemap_sha256"], file_sha256(packet_dir / "rolemap.jsonl"))


if __name__ == "__main__":
    unittest.main()
