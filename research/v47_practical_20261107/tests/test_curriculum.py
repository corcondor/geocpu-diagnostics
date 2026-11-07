"""Checks of boundaries and provenance; these are not model capability tests."""
import copy
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import build_curriculum as curriculum


class CurriculumChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((ROOT / "data/practical_manifest.json").read_text(encoding="utf-8"))
        cls.rows = {split: [json.loads(line) for line in (ROOT / "data" / meta["file"]).read_text(encoding="utf-8").splitlines()]
                    for split, meta in cls.manifest["files"].items()}
        cls.all_rows = [row for rows in cls.rows.values() for row in rows]
        cls.vocab = json.loads((ROOT / "frozen/r6/base_metadata.json").read_text(encoding="utf-8"))["vocab"]

    def test_counts_and_hashes_correspond_to_actual_files(self):
        self.assertEqual({key: len(rows) for key, rows in self.rows.items()},
                         {"train": 512, "dev": 128, "expanded_eval": 240})
        for split, meta in self.manifest["files"].items():
            data = (ROOT / "data" / meta["file"]).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), meta["sha256"])
            self.assertEqual(len(data), meta["size_bytes"])
            self.assertEqual(len(self.rows[split]), meta["rows"])

    def test_only_observations_are_masked(self):
        for row in self.all_rows:
            curriculum.validate_row(row, self.vocab, {})
            full = row["full_text"]
            for (start, end), step in zip(row["loss_mask_spans"], row["tool_steps"]):
                self.assertEqual(full[start:end], step["observation"])
                self.assertEqual(step["observation"], "\n```output\n" + step["stdout"] + "```\n")
                self.assertNotIn("```python", full[start:end])
            self.assertEqual(row["final_answer_line"], full[row["loss_mask_spans"][-1][1]:])
            self.assertTrue(row["final_answer_line"].startswith("答え: "))

    def test_missing_observation_mask_is_rejected(self):
        row = copy.deepcopy(self.rows["train"][0])
        row["loss_mask_spans"] = row["loss_mask_spans"][:-1]
        with self.assertRaisesRegex(ValueError, "observation count"):
            curriculum.validate_row(row, self.vocab, {})

    def test_mask_cannot_include_final_answer(self):
        row = copy.deepcopy(self.rows["train"][0])
        row["loss_mask_spans"][-1][1] = len(row["full_text"])
        with self.assertRaisesRegex(ValueError, "observation/mask"):
            curriculum.validate_row(row, self.vocab, {})

    def test_prompt_and_scenario_groups_are_disjoint(self):
        seen_prompts = set()
        seen_groups = set()
        for row in self.all_rows:
            digest = curriculum.fingerprint(row["prompt"])
            self.assertNotIn(digest, seen_prompts)
            self.assertNotIn(row["group_id"], seen_groups)
            seen_prompts.add(digest)
            seen_groups.add(row["group_id"])
        self.assertEqual(len(seen_prompts), 880)
        train_families = {row["family"] for row in self.rows["train"]}
        unseen = [row for row in self.rows["expanded_eval"] if row["family_partition"] == "unseen_family"]
        self.assertEqual(len(unseen), 48)
        self.assertFalse(train_families & {row["family"] for row in unseen})

    def test_reference_matches_executed_final_output(self):
        for row in self.all_rows:
            self.assertEqual(Decimal(row["reference"]["value"]), Decimal(row["tool_steps"][-1]["stdout"].strip()))
            self.assertTrue(row["verification"]["final_stdout_matches_reference"])
            self.assertEqual(row["verification"]["gpu_model_evaluations"], 0)

    def test_repair_failure_is_in_prompt_and_not_teacher_target(self):
        repairs = [row for row in self.all_rows if row["family"].startswith("repair_")]
        self.assertEqual(len(repairs), 156)
        for row in repairs:
            self.assertIn("AssertionError", row["prompt"])
            self.assertNotIn("AssertionError", row["target"])
            self.assertIn("tests: 3 passed", row["tool_steps"][0]["stdout"])
            self.assertEqual(row["verification"]["prompt_diagnostic_calls"], 1)

    def test_normalization_detects_width_and_space_variants(self):
        self.assertEqual(curriculum.fingerprint("Question: Ａ  １２\n B"),
                         curriculum.fingerprint("question: A 12 B"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frozen.json"
            path.write_text(json.dumps({"cases": [{"question": "Question: Ａ  １２\n B"}]}), encoding="utf-8")
            hashes, audit = curriculum.heldout_fingerprints([path])
            self.assertIn(curriculum.fingerprint("question: A 12 B"), hashes)
            self.assertEqual(audit[str(path)]["prompt_fields"], 1)

    def test_original_bundle_hash_is_required(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "modified.zip"
            import zipfile
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("wrong.txt", "modified source")
            with self.assertRaisesRegex(ValueError, "hash differs"):
                curriculum.read_r6(path)


if __name__ == "__main__":
    unittest.main()
