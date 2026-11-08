import json
from hashlib import sha256
from pathlib import Path
import tempfile
import unittest

from source_pools import PINNED_V43_ZIP_SHA, file_sha
from train_hybrid_stage import (_microbatches, _verify_original_swe_baseline,
                                _verify_pretrain_raw_diagnostic,
                                PARENT_CHECKPOINT_SHA, PINNED_SWE_EVAL_SHA,
                                PINNED_SWE_EVALUATOR_SHA,
                                PINNED_INTERIM_EVALUATOR_SHA, PINNED_INTERIM_CASES_SHA,
                                SOURCE_OVERLAY_SHA)


NAMES = ("swe200_full_answers.jsonl", "swe200_original_grades.jsonl",
         "original_swe_audit.jsonl", "v43_native_full_answers.jsonl",
         "v43_native_external_grades.jsonl", "v43_original_answers.jsonl",
         "old_skill_eval.jsonl")


class StageAdmissionTests(unittest.TestCase):
    def test_parent_fixed_raw_diagnostic_binding(self):
        cases_path = Path(__file__).with_name("fixed_interim_diagnostic_44_v2.jsonl")
        cases = [json.loads(line) for line in cases_path.read_text(encoding="utf-8").splitlines()]
        overlay_sha = sha256(json.dumps(SOURCE_OVERLAY_SHA, sort_keys=True).encode()).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for case in cases:
                for profile in case["profiles"]:
                    raw = "fixture original output"
                    rows.append({"case_id": case["id"], "group": case["group"],
                                 "profile": profile, "arm":"parent",
                                 "prompt": ("Pythonで計算し、実行結果を根拠に日本語で説明してください。\n"
                                            if profile == "python_cued" else "") + case["prompt"],
                                 "expected_answer": case["target"], "raw_answer": raw,
                                 "raw_answer_sha256": sha256(raw.encode()).hexdigest(),
                                 "model_generated_text":raw,
                                 "model_generated_text_sha256":sha256(raw.encode()).hexdigest(),
                                 "checkpoint_sha256":PARENT_CHECKPOINT_SHA,
                                 "v43_zip_sha256":PINNED_V43_ZIP_SHA,
                                 "cases_sha256":PINNED_INTERIM_CASES_SHA,
                                 "source_sha256":PINNED_INTERIM_EVALUATOR_SHA,
                                 "precision_profile":"eval_bf16_native",
                                 "overlay_sha256":overlay_sha,
                                 "ended_with_eos": True})
            raw_path = root / "raw_answers_44_profiles.jsonl"
            raw_path.write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in rows), encoding="utf-8")
            manifest = {"schema":"geocpu_v43_python_gradual_raw_diagnostic_v1",
                        "status":"COMPLETE", "arm":"parent", "checkpoint_kind":"parent",
                        "checkpoint_sha256":PARENT_CHECKPOINT_SHA, "v43_zip_sha256":PINNED_V43_ZIP_SHA,
                        "cases_sha256":PINNED_INTERIM_CASES_SHA,
                        "source_sha256":PINNED_INTERIM_EVALUATOR_SHA,
                        "precision_profile":"eval_bf16_native",
                        "overlay_sha256":overlay_sha,
                        "raw_answers_sha256":file_sha(raw_path),
                        "profile_counts":{"native":44,"python_cued":20,"python_uncued":20},
                        "max_generated_tokens":2048,"training_updates":0}
            (root / "COMPLETE.json").write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(_verify_pretrain_raw_diagnostic(root), file_sha(root / "COMPLETE.json"))
            rows[0]["prompt"] += "changed"
            raw_path.write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in rows), encoding="utf-8")
            manifest["raw_answers_sha256"] = file_sha(raw_path)
            (root / "COMPLETE.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ValueError):
                _verify_pretrain_raw_diagnostic(root)

    def test_complete_baseline_files_are_byte_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pins = {}
            for name in NAMES:
                (root / name).write_text('{"id":"fixture"}\n', encoding="utf-8")
                pins[name] = file_sha(root / name)
            complete = {"status": "COMPLETE", "suite": "heldout_swe200",
                        "cases": 200, "generation_limit": 2048, "training_updates": 0,
                        "source_zip_sha256": PINNED_V43_ZIP_SHA,
                        "parent_sha256": PARENT_CHECKPOINT_SHA,
                        "eval_member_sha256": PINNED_SWE_EVAL_SHA,
                        "original_evaluator_sha256": PINNED_SWE_EVALUATOR_SHA,
                        "source_and_parent_unchanged": True,
                        "eligible_old_skill_rows": 64, "files_sha256": pins}
            (root / "COMPLETE.json").write_text(json.dumps(complete), encoding="utf-8")
            self.assertEqual(_verify_original_swe_baseline(root), file_sha(root / "COMPLETE.json"))
            (root / NAMES[0]).write_text('{"id":"mutated"}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                _verify_original_swe_baseline(root)

    def test_microbatch_token_budget_is_hard(self):
        class Row:
            def __init__(self, length):
                self.tokens = [0] * length
        lengths = [100, 100, 250, 250, 100]
        chunks = _microbatches([Row(n) for n in lengths], lengths, budget=512)
        self.assertEqual([len(chunk) for chunk in chunks], [2, 2, 1])
        for chunk in chunks:
            padded = ((max(len(row.tokens) for row in chunk)+31)//32)*32
            self.assertLessEqual(padded * len(chunk), 512)
        with self.assertRaises(ValueError):
            _microbatches([Row(600)], [600], budget=512)

    def test_paired_arms_share_microbatch_boundaries(self):
        class Row:
            def __init__(self, length):
                self.tokens = [0] * length
        old = [Row(n) for n in (100, 100, 250, 250, 100)]
        intervention = [Row(n) for n in (180, 80, 220, 300, 100)]
        common = [max(len(a.tokens), len(b.tokens)) for a, b in zip(old, intervention)]
        control_chunks = _microbatches(old, common, budget=512)
        intervention_chunks = _microbatches(intervention, common, budget=512)
        self.assertEqual([len(c) for c in control_chunks], [len(c) for c in intervention_chunks])
        self.assertEqual([len(c) for c in control_chunks], [2, 1, 1, 1])


if __name__ == "__main__":
    unittest.main()
