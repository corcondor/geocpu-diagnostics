"""CPU checks of frozen inputs and adapter semantics; not model evaluation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import frozen_eval as frozen


class FrozenEvaluationTests(unittest.TestCase):
    def test_source_hashes_and_formal_counts(self):
        report = frozen.verify_frozen()
        self.assertFalse(report["model_evaluation_executed"])
        expected = {"T1": 200, "T2": 200, "T3": 2309, "T4": 8100,
                    "T5": 400, "T6": 200, "T7": 200, "T8": 8,
                    "python120": 120, "engineering64": 64, "visual10": 10,
                    "search_decision200": 200, "search_final198": 198,
                    "public_dev80": 80}
        for suite_id, count in expected.items():
            with self.subTest(suite=suite_id):
                self.assertEqual(len(list(frozen.iter_cases(suite_id))), count)
        self.assertEqual(len(list(frozen.iter_cases("T3", "generative"))), 100)
        self.assertEqual(len(list(frozen.iter_cases("T8", "original_v44"))), 4)
        self.assertEqual(frozen.get_suite("T6")["hop_prompt_counts"], {"1_hop": 17, "2_hop": 4})

    def test_selection_order_matches_pinned_registry(self):
        for suite in frozen.registry()["suites"]:
            if suite["availability"] != "frozen":
                continue
            for mode in suite["formal_modes"]:
                with self.subTest(suite=suite["id"], mode=mode["mode"]):
                    rows = list(frozen.iter_cases(suite["id"], mode["mode"]))
                    ids = "\n".join(str(row.get("id", "")) for row in rows)
                    prompts = json.dumps([row["prompt"] for row in rows],
                                         ensure_ascii=False, separators=(",", ":"))
                    self.assertEqual(hashlib.sha256(ids.encode()).hexdigest(), mode["ordered_ids_sha256"])
                    self.assertEqual(hashlib.sha256(prompts.encode()).hexdigest(), mode["ordered_prompts_sha256"])

    def test_tampered_source_and_path_escape_are_rejected(self):
        relative = "continue/engineering/scorer.py"
        with tempfile.TemporaryDirectory() as folder:
            temp = Path(folder)
            shutil.copyfile(frozen.FROZEN_ROOT / "source_files.json", temp / "source_files.json")
            destination = temp / relative
            destination.parent.mkdir(parents=True)
            destination.write_bytes(frozen.source_path(relative).read_bytes() + b"\n# altered\n")
            with self.assertRaisesRegex(ValueError, "changed"):
                frozen.source_path(relative, temp)
            with self.assertRaisesRegex(ValueError, "escapes"):
                frozen.source_path("../outside.py", temp)

    def test_original_python_scorer_requires_model_answer(self):
        case = next(frozen.iter_cases("python120"))
        scorer = frozen.load_original_scorer("python120")
        answer = case["expected_numeric"]
        generation = {"model_generated_text": "計算しました。", "output_text": answer,
                      "python_trace": [{"success": True, "stdout": answer}],
                      "python_invocations": 1, "search_invocations": 0,
                      "ended_with_eos": True}
        result = scorer.score_generation(case, generation)
        self.assertTrue(result["expected_numeric_in_successful_stdout"])
        self.assertFalse(result["explicit_final_numeric_exact"])
        result = scorer.score_generation(case, {**generation, "model_generated_text": "答え: " + answer})
        self.assertTrue(result["terminal_explicit_exact"])

    def test_native_search_scoring_distinguishes_observation_and_answer(self):
        case = next(frozen.iter_cases("search_final198"))
        answer = case["reference_answer"]
        output = {"model_generated_text": "分かりました。", "output_text": "[RESULT " + answer + "]",
                  "search_invocations": 0, "ended_with_eos": True}
        result = frozen.score_search_final(case, output)
        self.assertTrue(result["no_new_search"])
        self.assertFalse(result["terminal_exact"])
        valid = {**output, "model_generated_text": "答えは" + answer + "。"}
        self.assertTrue(frozen.score_search_final(case, valid)["no_search_and_terminal_exact"])
        trailing = {**valid, "model_generated_text": valid["model_generated_text"] + "追加説明。"}
        self.assertFalse(frozen.score_search_final(case, trailing)["terminal_exact"])
        decision = next(frozen.iter_cases("search_decision200"))
        result = frozen.score_search_decision(decision, {**output,
            "search_trace": [{"query": decision["reference_next_search_query"]}], "search_invocations": 1})
        self.assertTrue(result["exact_query_match"])

    def test_all_selected_prompts_are_in_leakage_corpus(self):
        hashes = frozen.heldout_prompt_hashes()
        for suite in frozen.registry()["suites"]:
            if suite["availability"] != "frozen":
                continue
            for mode in suite["formal_modes"]:
                for row in frozen.iter_cases(suite["id"], mode["mode"]):
                    self.assertIn(hashlib.sha256(row["prompt"].encode()).hexdigest(), hashes)
        self.assertGreater(len(hashes), 25000)

    def test_no_unavailable_suite_can_be_reported_complete(self):
        for suite in ("gate_v44b", "sentinel_v46"):
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                list(frozen.iter_cases(suite))
        with self.assertRaises(ValueError):
            frozen.generation_kwargs("T4")

    def test_original_tier_command_never_invokes_training(self):
        command = frozen.build_official_tier_command(study="/tmp/study", run="candidate",
            step=33000, checkpoint_sha="a" * 64, output="/tmp/new-output")
        self.assertTrue(command[2].endswith("stage_official_tier_v47.py"))
        self.assertIn("33000", command)
        self.assertNotIn("launch_suite_30500.py", " ".join(command))
        with self.assertRaises(ValueError):
            frozen.build_official_tier_command(study="/tmp/study", run="../bad", step=1,
                checkpoint_sha="a" * 64, output="/tmp/new-output")


if __name__ == "__main__":
    unittest.main()
