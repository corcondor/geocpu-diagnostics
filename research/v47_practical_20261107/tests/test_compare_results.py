import io
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from live_logging import RunLogger, file_sha, atomic_json
from compare_results import collect, compare, retention_decision


class CompareTests(unittest.TestCase):
    def make_result(self, root, checkpoint, *, correct=True, budget=1024, case_id="case1"):
        identity = {"suite": "python120", "mode": "formal", "checkpoint_id": checkpoint,
                    "checkpoint_sha256": ("a" if checkpoint == "before" else "b")*64,
                    "dataset_sha256": "c"*64, "scorer_sha256": "d"*64,
                    "runtime_sha256": "e"*64, "budget": {"max_tokens": budget},
                    "seed": 0, "metric": "terminal_explicit_exact"}
        with RunLogger(Path(root)/"python", identity, total=1, stream=io.StringIO()) as log:
            log.result(case_id, correct=correct, metrics={"python_execution_success": True})

    def native_fixture(self, root):
        """CPU parser fixture with frozen IDs; these are not model results."""
        import frozen_eval as frozen
        from eval_candidate import _original_source_binding
        folder = Path(root) / "original_tiers"
        folder.mkdir(parents=True)
        binding = {**_original_source_binding(), "run": "cpu_fixture", "step": 1,
                   "checkpoint_sha256": "a"*64, "training_data_sha256": "b"*64,
                   "receipt_sha256": "c"*64, "stage_order": "original Tier1-7, then original Tier8"}
        atomic_json(folder / "input_binding.json", binding)
        cases = list(frozen.iter_cases("T4"))
        ordered = [r for r in cases if r["task_type"] == "multiple_choice"]
        ordered += [r for r in cases if r["task_type"] != "multiple_choice"]
        swe = 0
        with (folder / "tier4_raw.jsonl").open("w", encoding="utf-8") as stream:
            for row in ordered:
                swe += row["pillar"] == "SWE_Software"
                stream.write(json.dumps({"id": row["id"], "target": row["target"].strip(),
                    "pillar": row["pillar"], "is_correct": row["pillar"] != "SWE_Software" or swe <= 158,
                    "cpu_fixture": True}) + "\n")
        raw8 = {}
        for profile in ("original_v44", "aligned_v43"):
            raw8[profile] = {"n": 4, "tir_code_rate": 75.0, "narrative_rate": 50.0,
                "expected_language_fence_rate": 75.0, "prose_outside_code_rate": 25.0,
                "details": [{"case_id": row["id"], "has_code": i < 3,
                    "has_narrative": i < 2, "expected_language_fence": i < 3,
                    "has_prose_outside_code": i < 1, "cpu_fixture": True}
                    for i, row in enumerate(frozen.iter_cases("T8", profile))]}
        atomic_json(folder / "tier8_raw.json", raw8)
        bodies = {
            1: "Final Answer Strict Match: 172 / 200 (86.0%)\nExact Step Termination (stopped on time): 174 / 200\n",
            2: "End-to-End Accuracy: 89.0% (178/200)\n",
            3: "Logit Progress: 2309/2309 (100.0%) | Acc: 41.62%\nGen Progress: 100/100 | Exact Match: 20\n",
            4: "Final Answer Strict Match: 158 / 164 (96.3%)\n",
            5: "Overall Logical Operator Accuracy: 280 / 400\n",
            6: "Overall Tier 6 Search Accuracy: 171 / 200\n",
            7: "[ARM A: Closed-Book] Accuracy: 21/200\n[ARM B: Open-Book] Accuracy: 88/200\n"}
        joined = "CPU parser fixture; no model was run.\n"
        for tier, body in bodies.items():
            section = f"[TIER {tier}/7]\n" + body
            (folder / f"tier_{tier}.log").write_text(section, encoding="utf-8")
            joined += section
        (folder / "original_tier1_7.log").write_text(joined, encoding="utf-8")
        (folder / "original_tier8.log").write_text("CPU format fixture; no model was run.\n")
        for label in ("original_tier1_7", "original_tier8"):
            command_sha = hashlib.sha256(json.dumps([label,
                file_sha(frozen.source_path("continue/tier_stage/stage_official_tier_v47.py")),
                str(folder.resolve())], sort_keys=True).encode()).hexdigest()
            atomic_json(folder / (label + ".complete.json"), {"returncode": 0,
                "command_sha256": command_sha, "log_sha256": file_sha(folder / (label+".log")),
                "observed_tiers": list(range(1, 8)) if label == "original_tier1_7" else []})
        atomic_json(folder / "complete.json", {"checkpoint_sha256": binding["checkpoint_sha256"],
            "tier4_raw_sha256": file_sha(folder / "tier4_raw.jsonl"),
            "tier8_raw_sha256": file_sha(folder / "tier8_raw.json"),
            "tier1_7_log_sha256": file_sha(folder / "original_tier1_7.log"),
            "tier8_log_sha256": file_sha(folder / "original_tier8.log")})
        return folder

    def test_missing_candidate_is_not_zero(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root/"after").mkdir()
            self.make_result(root/"before", "before")
            rows = compare(root/"before", root/"after")
            self.assertTrue(all(r["after"] is None and r["delta_pp"] is None and not r["comparable"] for r in rows))

    def test_changed_generation_budget_is_not_compared(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_result(root/"before", "before", budget=100)
            self.make_result(root/"after", "after", budget=200)
            self.assertTrue(all(not r["comparable"] for r in compare(root/"before", root/"after")))

    def test_matched_comparison_keeps_paired_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_result(root/"before", "before", correct=False)
            self.make_result(root/"after", "after", correct=True)
            rows = compare(root/"before", root/"after")
            main = next(r for r in rows if r["metric"] == "terminal_explicit_exact")
            self.assertEqual(main["delta_pp"], 100)
            self.assertIn("wrong→right=1", main["note"])
            self.assertFalse(retention_decision(rows)["retention_complete"])

    def test_progress_only_is_not_a_score(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/"suite_status.json").write_text(json.dumps({"state":"running", "progress":"90/120"}))
            self.assertEqual(collect(root), {})
            self.assertFalse(retention_decision([])["proceed_to_next_small_experiment"])

    def test_native_tier_scopes_keep_t1_t4_t3_and_t8_distinct(self):
        with tempfile.TemporaryDirectory() as temp:
            self.native_fixture(temp)
            rows = collect(temp)
            t1 = rows["T1", "original", "final_answer_strict_match"]
            self.assertEqual((t1["correct"], t1["answered"], t1["state"]), (172, 200, "complete"))
            t4 = rows["T4", "original", "SWE_Software"]
            self.assertEqual((t4["correct"], t4["answered"]), (158, 164))
            t3 = rows["T3", "original", "logit_argmax"]
            self.assertIsNone(t3["correct"])
            self.assertAlmostEqual(t3["accuracy"], .4162)
            self.assertIn("rounded", t3["note"])
            self.assertEqual(rows["T3", "original_first100", "generative_exact"]["answered"], 100)
            t8 = rows["T8", "original_v44", "tir_code_rate"]
            self.assertEqual((t8["accuracy"], t8["answered"]), (.75, 4))
            self.assertIn("not a mathematical correctness", t8["note"])
            self.assertEqual({key[0] for key in rows}, {f"T{i}" for i in range(1, 9)})

    def test_native_partial_raw_cannot_be_made_complete_by_updating_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self.native_fixture(root / "before")
            target = root / "after/original_tiers"
            shutil.copytree(source, target)
            raw = target / "tier4_raw.jsonl"
            raw.write_text("\n".join(raw.read_text().splitlines()[:-1]) + "\n")
            marker = json.loads((target / "complete.json").read_text())
            marker["tier4_raw_sha256"] = file_sha(raw)
            atomic_json(target / "complete.json", marker)
            rows = compare(root / "before", root / "after")
            self.assertTrue(rows)
            self.assertTrue(all(row["delta_pp"] is None and not row["comparable"] for row in rows))
            self.assertFalse(retention_decision(rows)["retention_complete"])

    def test_complete_native_and_verified_copy_compare_under_same_conditions(self):
        from eval_candidate import reuse_original_stage
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self.native_fixture(root / "before")
            self.native_fixture(root / "after")
            rows = compare(root / "before", root / "after")
            self.assertTrue(rows)
            self.assertTrue(all(row["comparable"] and row["delta_pp"] == 0 for row in rows))
            reuse_original_stage(source, root / "copied/original_tiers", checkpoint_sha="a"*64,
                                 run="cpu_fixture", step=1, training_sha="b"*64, receipt_sha="c"*64)
            copied = compare(root / "before", root / "copied")
            self.assertEqual([(r["suite"], r["metric"], r["delta_pp"]) for r in copied],
                             [(r["suite"], r["metric"], r["delta_pp"]) for r in rows])
            self.assertTrue(all(row["comparable"] for row in copied))

    def test_native_changed_stage_marker_never_produces_delta(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.native_fixture(root / "before")
            target = self.native_fixture(root / "after")
            path = target / "original_tier1_7.complete.json"
            marker = json.loads(path.read_text())
            marker["command_sha256"] = "f"*64
            atomic_json(path, marker)
            rows = compare(root / "before", root / "after")
            self.assertTrue(all(not row["comparable"] and row["delta_pp"] is None for row in rows))

    def test_native_changed_source_binding_is_not_compared(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self.native_fixture(root / "before")
            target = root / "after/original_tiers"
            shutil.copytree(source, target)
            binding = json.loads((target / "input_binding.json").read_text())
            binding["retrieval_corpus_sha256"] = "f"*64
            atomic_json(target / "input_binding.json", binding)
            rows = compare(root / "before", root / "after")
            self.assertTrue(all(not row["comparable"] and row["delta_pp"] is None for row in rows))

    def test_inconsistent_status_score_is_unverified_not_a_gain(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_result(root / "before", "before", correct=False)
            self.make_result(root / "after", "after", correct=False)
            status = root / "after/python/status.json"
            record = json.loads(status.read_text())
            record.update(correct=1, incorrect=0, accuracy=1.0)
            atomic_json(status, record)
            row = next(r for r in compare(root / "before", root / "after") if r["metric"] == "terminal_explicit_exact")
            self.assertIsNone(row["after"])
            self.assertIsNone(row["delta_pp"])
            self.assertEqual(row["after_state"], "unverified_result")

    def test_same_count_different_case_ids_has_no_delta_for_any_metric(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_result(root / "before", "before", correct=False, case_id="first")
            self.make_result(root / "after", "after", case_id="different")
            rows = compare(root / "before", root / "after")
            self.assertTrue(all(r["delta_pp"] is None and not r["comparable"] for r in rows))

    def test_evaluator_error_is_not_an_incorrect_answer(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_result(root / "before", "before")
            manifest = json.loads((root / "before/python/run.json").read_text())
            identity = {**manifest["binding"]["identity"], "checkpoint_id": "after", "checkpoint_sha256": "b"*64}
            with RunLogger(root / "after/python", identity, total=1, stream=io.StringIO()) as log:
                log.result("case1", status="error", detail="CPU fixture evaluator exception")
            row = next(r for r in compare(root / "before", root / "after") if r["metric"] == "terminal_explicit_exact")
            self.assertEqual(row["after_denominator"], 0)
            self.assertIsNone(row["after"])
            self.assertIsNone(row["delta_pp"])
            self.assertEqual(row["after_state"], "failed")
            self.assertIn("errors=1", row["note"])

    def test_partial_submetric_uses_actual_observations_and_never_compares(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_result(root / "identity_seed", "before")
            manifest = json.loads((root / "identity_seed/python/run.json").read_text())
            identity = manifest["binding"]["identity"]
            for label in ("before", "after"):
                with RunLogger(root / label / "python", {**identity, "checkpoint_id": label},
                               total=2, stream=io.StringIO()) as log:
                    log.result("case1", correct=True, metrics={"python_execution_success": True})
                    log.result("case2", correct=True)
            rows = compare(root / "before", root / "after")
            sub = next(row for row in rows if row["metric"] == "python_execution_success")
            self.assertEqual((sub["before"], sub["before_denominator"]), (1.0, 1))
            self.assertFalse(sub["comparable"])
            self.assertIsNone(sub["delta_pp"])
            self.assertEqual(sub["after_state"], "incomplete_metric_coverage")

    def complete_gate_rows(self):
        metrics = [("T1", "final_answer_strict_match"), ("T1", "exact_step_termination"),
                   ("T2", "end_to_end"), ("T5", "logical_operator_accuracy"),
                   ("T4", "SWE_Software")]
        rows = [{"suite": suite, "metric": metric, "mode": "original", "comparable": True,
                 "delta_pp": -2.000000000000001, "before_state": "complete", "after_state": "complete",
                 "before_denominator": 200, "after_denominator": 200} for suite, metric in metrics]
        rows.append({"suite": "python120", "metric": "terminal_explicit_exact", "mode": "formal",
                     "comparable": True, "delta_pp": 5.0, "before_state": "complete", "after_state": "complete",
                     "before_denominator": 120, "after_denominator": 120})
        return rows

    def test_retention_requires_every_named_basic_metric_completed(self):
        rows = self.complete_gate_rows()
        decision = retention_decision(rows)
        self.assertTrue(decision["retention_complete"])
        self.assertTrue(decision["retention_passed"])
        self.assertTrue(decision["proceed_to_next_small_experiment"])
        self.assertFalse(decision["automatic_checkpoint_promotion"])
        for index in range(5):
            with self.subTest(missing=index):
                missing = retention_decision(rows[:index] + rows[index+1:])
                self.assertFalse(missing["retention_complete"])
                self.assertFalse(missing["proceed_to_next_small_experiment"])
            with self.subTest(incomplete=index):
                incomplete = [dict(row) for row in rows]
                incomplete[index]["after_state"] = "incomplete"
                self.assertFalse(retention_decision(incomplete)["retention_complete"])

    def test_retention_drop_and_wrong_improvement_mode_fail(self):
        rows = self.complete_gate_rows()
        rows[2]["delta_pp"] = -2.01
        self.assertFalse(retention_decision(rows)["retention_passed"])
        rows = self.complete_gate_rows()
        rows[-1]["mode"] = "changed_budget_diagnostic"
        decision = retention_decision(rows)
        self.assertTrue(decision["retention_passed"])
        self.assertFalse(decision["target_improvement_seen"])
        self.assertFalse(decision["proceed_to_next_small_experiment"])


if __name__ == "__main__":
    unittest.main()
