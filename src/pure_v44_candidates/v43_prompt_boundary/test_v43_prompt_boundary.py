"""CPU-only shape checks using pinned public and saved model-facing records."""
import importlib.util
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
ORIGINAL = Path(r"C:\Users\81808\.openclaw\reports\jmmlu_v1\ag_fused_fullqa_training_v1\v44_hotfix_source_readonly_20261103")
V43_META = Path(r"C:\Users\81808\.openclaw\reports\jmmlu_v1\ag_fused_fullqa_training_v1\v43_source_readonly_pure_audit_20261103\base_metadata.json")
SAVED_PUBLIC = Path(r"G:\マイドライブ\GeoCPU_Frontier_CoTraining_v44_PythonTIR_Hotfix_20261103\run_1\training\fixed_public_cases.json")
sys.path[:0] = [str(HERE), str(ORIGINAL)]
import prepare_public_v44 as prepare
from training_data_v44 import encode_supervision
from evaluate_v44_eight_tiers import TIER8_PROMPTS


def record(prompt, raw_target, segments, vocab):
    target, segments = prepare.add_v43_public_boundary(raw_target, segments)
    item = {"prompt": prompt, "target": target, "full_text": prompt + target,
            "target_segments": segments}
    ids, tags = encode_supervision(item, vocab)
    assert ids[len(prompt)] == vocab[" "]
    assert tags[len(prompt)] == 1  # y[len(prompt)-1] is the first supervised space
    assert ids[-1] == 3 and tags[-1] == 1  # real EOS remains supervised
    assert item["full_text"] == prompt + " " + raw_target
    for part in segments:
        if part["role"] == "tool":
            assert all(tag == 0 for tag in tags[len(prompt)+part["start"]:len(prompt)+part["end"]])
    return item


class V43PromptBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vocab = json.loads(V43_META.read_text(encoding="utf-8"))["vocab"]

    def test_actual_v43_source_boundaries(self):
        evidence = json.loads((HERE / "v43_boundary_examples.json").read_text(encoding="utf-8"))
        rows = evidence["rows"]
        self.assertEqual(len(rows), 4)
        self.assertTrue(rows["Algebraic_Composite_Aligned"]["prompt"].startswith("問題: "))
        self.assertTrue(rows["Algebraic_Composite_Aligned"]["prompt"].endswith("\n思考:"))
        self.assertTrue(rows["Databricks_Dolly_Full"]["prompt"].startswith("指示: "))
        self.assertIn("\n文脈: ", rows["Databricks_Dolly_Full"]["prompt"])
        self.assertTrue(rows["Databricks_Dolly_Full"]["prompt"].endswith("\n回答:"))
        self.assertTrue(rows["GSM8K_Japanese_Full"]["prompt"].endswith("\n思考:"))
        self.assertTrue(all(row["separator"] == " " for row in rows.values()))

    def test_real_public_source_rows_and_target_mask(self):
        samples = json.loads((HERE / "public_format_samples_v44.json").read_text(encoding="utf-8"))
        expected = {"numina": "問題: ", "code": "指示: ", "japanese": "指示: "}
        for name, bucket in (("numina", "tir"), ("code", "code"), ("japanese", "japanese")):
            with self.subTest(bucket=bucket):
                raw = samples[name]["sample"]
                candidate = prepare.public_candidate(bucket, raw)
                self.assertTrue(candidate["prompt"].startswith(expected[name]))
                self.assertNotIn("アシスタント:", candidate["prompt"])
                if bucket == "tir":
                    normalized, segments, receipts = prepare.normalize_tir(candidate["target"])
                    self.assertTrue(receipts)
                    target = normalized
                else:
                    target = candidate["target"]
                    segments = prepare._segments(len(target), [])
                admitted = record(candidate["prompt"], target, segments, self.vocab)
                self.assertEqual(admitted["target"][1:], target)
                self.assertEqual(admitted["target_segments"][-1]["end"], len(admitted["target"]))
                if bucket == "japanese":
                    self.assertEqual(candidate["prompt"], "指示: " + raw["instruction"] + "\n文脈: " + raw["input"] + "\n回答:")

    def test_saved_public_test_rows_keep_answer_and_split(self):
        saved = json.loads(SAVED_PUBLIC.read_text(encoding="utf-8"))["cases"]
        self.assertEqual(len(saved), 4)
        for case in saved:
            self.assertEqual(case["provenance"]["split"], "test")
            old_prompt = case["prompt"]
            self.assertTrue(old_prompt.startswith("ユーザー: "))
            self.assertTrue(old_prompt.endswith("\nアシスタント: "))
            content = old_prompt[len("ユーザー: "):-len("\nアシスタント: ")]
            new_prompt = (("問題: " + content + "\n思考:") if case["bucket"] == "tir"
                          else ("指示: " + content + "\n回答:"))
            changed = record(new_prompt, case["target"], case["target_segments"], self.vocab)
            self.assertEqual(changed["target"][1:], case["target"])
            self.assertEqual(changed["target_segments"][-1]["end"], len(changed["target"]))
            self.assertLessEqual(len(changed["full_text"]) + 1, 8192)

    def test_tier8_generation_uses_matching_boundaries(self):
        self.assertEqual(len(TIER8_PROMPTS), 4)
        self.assertTrue(all(prompt.endswith(("\n思考:", "\n回答:")) for prompt in TIER8_PROMPTS))
        self.assertTrue(all("ユーザー:" not in prompt and "アシスタント:" not in prompt for prompt in TIER8_PROMPTS))
        self.assertTrue(all(set(prompt) <= set(self.vocab) for prompt in TIER8_PROMPTS))

    def test_curriculum_serializes_real_public_samples_with_space(self):
        """Admission fixture: three actual public rows, one actual v43 anchor."""
        samples = json.loads((HERE / "public_format_samples_v44.json").read_text(encoding="utf-8"))
        anchors = json.loads((HERE / "v43_dolly_anchor_samples.json").read_text(encoding="utf-8"))
        anchor = anchors["train"]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            inputs = {}
            for name in ("numina", "code", "japanese"):
                path = root / (name + ".jsonl")
                path.write_text(json.dumps(samples[name]["sample"], ensure_ascii=False) + "\n", encoding="utf-8")
                inputs[(name, "train")] = path
            numina_test = root / "numina_test.jsonl"
            numina_test.write_text(json.dumps({"problem": "separate official test fixture"}) + "\n", encoding="utf-8")
            inputs[("numina", "test")] = numina_test
            anchor_path = root / "anchor.jsonl"
            anchor_path.write_text(
                "".join(json.dumps(anchors[split], ensure_ascii=False) + "\n"
                        for split in ("train", "dev", "test")), encoding="utf-8")
            lock = {"repos": {}}
            for (name, split), path in inputs.items():
                repo = lock["repos"].setdefault(name, {"repo_id": "fixture/" + name,
                                                     "revision": "0" * 40, "files": {}})
                data = path.read_bytes()
                repo["files"][split] = {"path": path.name, "bytes": len(data),
                                         "sha256": hashlib.sha256(data).hexdigest()}
            lock_path = root / "lock.json"
            lock_path.write_text(json.dumps(lock), encoding="utf-8")
            evaluation_dir = root / "eval"
            evaluation_dir.mkdir()
            (evaluation_dir / "benchmark_8100_manifest.jsonl").write_text(
                json.dumps({"id": "separate-eval", "question": "unrelated evaluation fixture"}) + "\n",
                encoding="utf-8")
            output = root / "prepared"
            manifest = prepare.prepare_public_curriculum(
                output, V43_META, evaluation_dir, anchor_path,
                source_files=inputs, source_lock_path=lock_path,
                target_counts={bucket: 1 for bucket in prepare.TARGET_COUNTS},
                verify_sample_count=0)
            self.assertEqual(manifest["counts"]["actual_train_exposures"],
                             {bucket: 1 for bucket in prepare.TARGET_COUNTS})
            rows = list(prepare.json_records(output / "train.jsonl"))
            self.assertEqual(len(rows), 4)
            for row in rows:
                self.assertEqual(row["full_text"], row["prompt"] + row["target"])
                encode_supervision(row, self.vocab)
                if row["bucket"] != "anchor":
                    self.assertTrue(row["target"].startswith(" "))
                    self.assertEqual(row["target_segments"][0]["end"], 1)
                    source_name = {"tir": "numina", "code": "code", "japanese": "japanese"}[row["bucket"]]
                    raw = samples[source_name]["sample"]
                    original_answer = raw["output"] if source_name == "japanese" else raw["solution"]
                    self.assertEqual(row["provenance"]["source_target_sha256"],
                                     hashlib.sha256(original_answer.encode()).hexdigest())
                else:
                    self.assertEqual(row["prompt"], anchor["prompt"])
                    self.assertEqual(row["full_text"], anchor["full_text"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
