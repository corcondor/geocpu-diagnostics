"""CPU mocks of runner contracts and reuse; these are NOT model/GPU results.

Synthetic generation dictionaries and synthetic checkpoint bytes test plumbing
only. Original scoring code is called where it does not load a model.
"""
from __future__ import annotations

from argparse import Namespace
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import eval_candidate as evaluation
import frozen_eval as frozen
from live_logging import RunLogger as RealRunLogger, atomic_json, file_sha


def _frozen_data_available():
    try:
        return all((frozen.FROZEN_ROOT / name).is_file() for name in frozen.source_inventory())
    except (OSError, ValueError):
        return False


requires_frozen_data = unittest.skipUnless(_frozen_data_available(),
    "requires the private frozen-data bundle; CPU runner contract tests remain available")


def generation(text="答え: Oslo", **updates):
    result = {"model_generated_text": text, "output_text": text,
              "full_text": "CPU_FIXTURE_NOT_MODEL_EVALUATION\n"+text,
              "generated_tokens": [], "cpu_invocations": 0, "cpu_trace": [],
              "python_invocations": 0, "python_trace": [],
              "search_invocations": 0, "search_trace": [],
              "ended_with_eos": True, "stop_reason": "eos"}
    result.update(updates)
    return result


def knowledge_case(index=0):
    return {"id": f"cpu_mock_{index}", "prompt": f"CPU mock question {index}",
            "answer_format": "free_text", "reference_final_answer": "Oslo",
            "reference_observations": ["[RESULT CPU mock evidence]"]}


def identity(suite, mode, cases):
    fixture_sha = hashlib.sha256(b"CPU mock; no model was evaluated").hexdigest()
    return {"suite": suite, "mode": mode, "checkpoint_id": "CPU_MOCK_NOT_A_MODEL_RUN",
            "checkpoint_sha256": fixture_sha, "dataset_sha256": evaluation.hash_object(cases),
            "scorer_sha256": fixture_sha, "runtime_sha256": fixture_sha,
            "budget": {"max_tokens": 1024, "cpu_mock": True}, "seed": 0,
            "metric": evaluation.metric_for(suite)}


class CandidateRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.stream = io.StringIO()
        self.logger_patch = patch.object(evaluation, "RunLogger", side_effect=lambda *args, **kwargs:
            RealRunLogger(*args, **kwargs, stream=self.stream))
        self.logger_patch.start()

    def tearDown(self):
        self.logger_patch.stop()
        self.temp.cleanup()

    def run_mock(self, cases=None, generate=None, output=None, resume=False, **kwargs):
        cases = cases or [knowledge_case(0), knowledge_case(1)]
        return evaluation.run_questions("knowledge_diagnostic", "closed", cases,
            generate or (lambda case: generation()), output or self.root / "run",
            identity("knowledge_diagnostic", "closed", cases), resume=resume, **kwargs)

    def status(self, output=None):
        return json.loads(((output or self.root / "run") / "status.json").read_text())

    @requires_frozen_data
    def test_native_conditions_and_primary_metrics_are_preserved(self):
        for suite in evaluation.AUX:
            self.assertEqual(evaluation.generation_settings(suite, "formal"), frozen.generation_kwargs(suite))
        self.assertEqual(evaluation.generation_settings("knowledge_diagnostic", "tool")["max_tokens"], 2048)
        self.assertEqual(evaluation.generation_settings("practical_unseen_families", "python")["max_tokens"], 2048)
        self.assertEqual(evaluation.metric_for("search_final198"), "terminal_exact")
        cases = [
            ("python120", next(frozen.iter_cases("python120"))),
            ("engineering64", next(case for case in frozen.iter_cases("engineering64") if case["reference"]["unit"] != "min")),
            ("search_decision200", next(frozen.iter_cases("search_decision200"))),
            ("search_final198", next(frozen.iter_cases("search_final198"))),
            ("public_dev80", next(frozen.iter_cases("public_dev80"))),
        ]
        for suite, case in cases:
            with self.subTest(suite=suite):
                work = self.root / suite; work.mkdir()
                result, metric, state = evaluation.score_case(suite, case, generation("不明です。"), work)
                self.assertEqual(metric, evaluation.metric_for(suite))
                self.assertEqual(state, "scored")
                self.assertIs(type(result[metric]), bool)
        with patch.object(frozen, "load_original_scorer", return_value=SimpleNamespace(
                evaluate=lambda case, text, work: {"status": "unmeasured", "note": "CPU mock; render not executed"})):
            result, metric, state = evaluation.score_case("visual10", {"id": "mock"}, generation(), self.root)
            self.assertEqual((metric, state, result[metric]), ("correct", "unmeasured", False))

    @requires_frozen_data
    def test_tool_observations_alone_never_supply_the_scored_answer(self):
        k = knowledge_case()
        self.assertFalse(evaluation.knowledge_score(k, generation("", output_text="答え: Oslo"))["terminal_exact"])
        self.assertTrue(evaluation.knowledge_score(k, generation())["format_valid"])
        py = next(frozen.iter_cases("python120"))
        value = py["expected_numeric"]
        result, _, _ = evaluation.score_case("python120", py, generation(
            "計算しました。", output_text="```output\n"+value+"\n```",
            python_trace=[{"success": True, "stdout": value}], python_invocations=1), self.root)
        self.assertTrue(result["expected_numeric_in_successful_stdout"])
        self.assertFalse(result["terminal_explicit_exact"])
        final = next(frozen.iter_cases("search_final198"))
        result, metric, _ = evaluation.score_case("search_final198", final,
            generation("不明です。", output_text="答えは"+final["reference_answer"]+"。"), self.root)
        self.assertFalse(result[metric])
        public = next(case for case in frozen.iter_cases("public_dev80") if case["tool_search_count"] > 0)
        observation = "[RESULT 答えは"+public["answer"]+"。]"
        raw = generation("まだ判断していません。", output_text="<SEARCH query> => "+observation+" まだ判断していません。",
            search_trace=[{"query": "query", "result": observation}], search_invocations=1)
        result, metric, _ = evaluation.score_case("public_dev80", public, raw, self.root)
        self.assertFalse(result[metric])

    def test_supplemental_parser_rejects_fenced_or_pre_code_answers(self):
        case = knowledge_case()
        for text in ("```output\nOslo", "```python\nOslo\n```", "答え: Oslo\n```python\npass\n```"):
            self.assertFalse(evaluation.knowledge_score(case, generation(text))["terminal_exact"])
        self.assertTrue(evaluation.knowledge_score(case, generation("```python\npass\n```\n答え: Oslo"))["terminal_exact"])

    def test_cpu_state_is_omitted_but_native_trace_and_text_survive(self):
        native = generation(cpu_state=object(), cpu_trace=[{"op": "SET", "val": 1}])
        cases = [knowledge_case()]
        status = self.run_mock(cases, lambda case: native)
        self.assertEqual((status["state"], status["correct"], status["answered"]), ("complete", 1, 1))
        raw = json.loads(evaluation._raw_path(self.root / "run", cases[0]["id"]).read_text())
        self.assertEqual(raw["generation_omitted_fields"], ["cpu_state"])
        self.assertNotIn("cpu_state", raw["generation"])
        self.assertEqual(raw["generation"]["cpu_trace"], native["cpu_trace"])
        self.assertIn("cpu_state", native)  # Caller evidence object was not mutated.

    def test_resume_true_can_start_a_new_suite_then_reuse_completed_raw(self):
        calls = Mock(return_value=generation())
        status = self.run_mock(generate=calls, resume=True)
        self.assertEqual(status["state"], "complete")
        self.assertEqual(calls.call_count, 2)
        no_generate = Mock(side_effect=AssertionError("completed raw must not regenerate"))
        resumed = self.run_mock(generate=no_generate, resume=True)
        self.assertEqual((resumed["processed"], resumed["state"]), (2, "complete"))
        no_generate.assert_not_called()

    def test_interrupted_after_raw_resumes_without_repeating_generation(self):
        first = Mock(return_value=generation())
        with patch.object(evaluation, "score_case", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.run_mock(generate=first)
        self.assertEqual(first.call_count, 1)
        self.assertEqual(self.status()["state"], "interrupted")
        second = Mock(return_value=generation())
        result = self.run_mock(generate=second, resume=True)
        self.assertEqual(second.call_count, 1)
        self.assertEqual((result["state"], result["correct"]), ("complete", 2))

    def test_changed_completed_raw_is_rejected_before_generation(self):
        self.run_mock()
        path = evaluation._raw_path(self.root / "run", "cpu_mock_0")
        raw = json.loads(path.read_text()); raw["generation"]["model_generated_text"] = "tampered"
        atomic_json(path, raw)
        before = path.read_bytes()
        generator = Mock(return_value=generation())
        with self.assertRaisesRegex(evaluation.EvaluationContractError, "evidence differs"):
            self.run_mock(generate=generator, resume=True)
        generator.assert_not_called()
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(self.status()["state"], "failed")

    def test_wrong_pending_raw_id_is_not_overwritten(self):
        with patch.object(evaluation, "score_case", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.run_mock()
        path = evaluation._raw_path(self.root / "run", "cpu_mock_0")
        raw = json.loads(path.read_text()); raw["id"] = "different_case"
        atomic_json(path, raw); before = path.read_bytes()
        generator = Mock(return_value=generation())
        with self.assertRaisesRegex(evaluation.EvaluationContractError, "binding differs"):
            self.run_mock(generate=generator, resume=True)
        self.assertEqual(path.read_bytes(), before)
        generator.assert_not_called()

    def test_configuration_errors_fail_before_any_model_call(self):
        cases = [knowledge_case()]
        generator = Mock(return_value=generation())
        bound = identity("knowledge_diagnostic", "closed", cases)
        invalid_cases = ([], cases+cases)
        for index, rows in enumerate(invalid_cases):
            with self.assertRaises(evaluation.EvaluationContractError):
                evaluation.run_questions("knowledge_diagnostic", "closed", rows, generator,
                    self.root / f"bad{index}", bound)
        for change in ({"metric": "wrong"}, {"dataset_sha256": "f"*64}):
            with self.assertRaises(evaluation.EvaluationContractError):
                evaluation.run_questions("knowledge_diagnostic", "closed", cases, generator,
                    self.root / "bad_binding", {**bound, **change})
        generator.assert_not_called()

    def test_generation_failure_records_error_and_stops_the_suite(self):
        generator = Mock(side_effect=RuntimeError("CPU injected model/runtime failure"))
        with self.assertRaisesRegex(RuntimeError, "stopped before further"):
            self.run_mock(generate=generator)
        self.assertEqual(generator.call_count, 1)
        status = self.status()
        self.assertEqual((status["state"], status["errors"], status["answered"], status["pending"]), ("failed", 1, 0, 1))
        raw = json.loads(evaluation._raw_path(self.root / "run", "cpu_mock_0").read_text())
        self.assertEqual(raw["error_phase"], "generation_or_serialization")

    def test_unknown_non_json_generation_field_is_a_visible_fatal_error(self):
        generator = Mock(return_value=generation(unexpected_tensor=object()))
        with self.assertRaises(RuntimeError):
            self.run_mock(generate=generator)
        self.assertEqual(generator.call_count, 1)
        self.assertEqual(self.status()["state"], "failed")

    def test_isolated_scorer_failure_keeps_raw_and_is_not_a_wrong_answer(self):
        original = evaluation.score_case
        def sometimes(suite, case, generated, work):
            if case["id"] == "cpu_mock_0":
                raise RuntimeError("CPU injected scorer failure")
            return original(suite, case, generated, work)
        with patch.object(evaluation, "score_case", side_effect=sometimes):
            status = self.run_mock()
        self.assertEqual((status["state"], status["errors"], status["answered"], status["correct"]), ("failed", 1, 1, 1))
        raw = json.loads(evaluation._raw_path(self.root / "run", "cpu_mock_0").read_text())
        self.assertEqual(raw["generation"]["model_generated_text"], "答え: Oslo")
        self.assertEqual(raw["error_phase"], "scoring")

    def test_non_json_score_is_recorded_as_measurement_error(self):
        with patch.object(evaluation, "score_case", return_value=(
                {"terminal_exact": False, "unsupported_value": object()}, "terminal_exact", "scored")):
            status = self.run_mock([knowledge_case()])
        self.assertEqual((status["state"], status["errors"], status["answered"]), ("failed", 1, 0))
        raw = json.loads(evaluation._raw_path(self.root / "run", "cpu_mock_0").read_text())
        self.assertIn("generation", raw)
        self.assertEqual(raw["status"], "error")

    def test_repeated_scorer_failures_stop_before_all_cases(self):
        calls = Mock(return_value=generation())
        with patch.object(evaluation, "score_case", side_effect=RuntimeError("CPU injected scorer bug")):
            with self.assertRaisesRegex(RuntimeError, "repeated scoring"):
                self.run_mock([knowledge_case(i) for i in range(8)], calls)
        self.assertEqual(calls.call_count, 3)
        status = self.status()
        self.assertEqual((status["state"], status["errors"], status["pending"]), ("failed", 3, 5))

    @requires_frozen_data
    def test_known_original_engineering_min_bug_is_unmeasured(self):
        case = next(case for case in frozen.iter_cases("engineering64") if case["reference"]["unit"] == "min")
        text = "答え: "+str(case["reference"]["value"])+" min"
        result = evaluation.run_questions("engineering64", "formal", [case], lambda _: generation(text),
            self.root / "run", identity("engineering64", "formal", [case]))
        self.assertEqual((result["state"], result["unmeasured"], result["errors"], result["answered"]),
                         ("complete_with_unmeasured", 1, 0, 0))

    def test_primary_metric_contract_error_stops_after_one_case(self):
        generator = Mock(return_value=generation())
        with patch.object(evaluation, "score_case", return_value=({"terminal_exact": 1}, "terminal_exact", "scored")):
            with self.assertRaisesRegex(RuntimeError, "contract"):
                self.run_mock(generate=generator)
        self.assertEqual(generator.call_count, 1)
        self.assertEqual(self.status()["errors"], 1)

    def test_public_row_receipt_reuses_native_detail_and_detects_changed_bytes(self):
        case, binding = knowledge_case(), {"checkpoint_sha256": "a"*64, "cpu_mock": True}
        compute = Mock(return_value={"id": case["id"], "ce": 1.25, "positions": 8})
        result = evaluation._public_row(self.root, "nll", 0, case, binding, compute, resume=True)
        self.assertEqual(evaluation._public_row(self.root, "nll", 0, case, binding, compute, resume=True), result)
        self.assertEqual(compute.call_count, 1)
        atomic_json(self.root / "nll_0000.json", {**result, "ce": 0.0})
        with self.assertRaisesRegex(evaluation.EvaluationContractError, "hash differs"):
            evaluation._public_row(self.root, "nll", 0, case, binding, compute, resume=True)
        self.assertEqual(compute.call_count, 1)

    def test_default_knowledge_reads_free_text_file_and_oracle_only_adds_evidence(self):
        data = self.root / "data"; data.mkdir()
        case = knowledge_case()
        (data / "knowledge_free_dev.jsonl").write_text(json.dumps(case)+"\n")
        (data / "knowledge_dev.jsonl").write_text(json.dumps({"id": "must_not_be_used"})+"\n")
        with patch.object(evaluation, "ROOT", self.root):
            self.assertEqual(evaluation.suite_cases("knowledge_diagnostic", "closed"), [case])
            oracle = evaluation.suite_cases("knowledge_diagnostic", "oracle")[0]
        self.assertEqual(oracle["id"], case["id"])
        self.assertTrue(oracle["prompt"].endswith(case["prompt"]))
        self.assertIn(case["reference_observations"][0], oracle["prompt"])

    @requires_frozen_data
    @unittest.skipUnless((ROOT / "data/practical_expanded_eval.jsonl").is_file(), "requires practical evaluation data")
    def test_full_plan_uses_all_native_mc_rows_in_separate_modes(self):
        original = evaluation.suite_cases
        def cases(suite, mode="formal"):
            if suite == "knowledge_diagnostic":
                return [knowledge_case(i) for i in range(64)]  # Explicit CPU fixture for this plan test.
            return original(suite, mode)
        with patch.object(evaluation, "suite_cases", side_effect=cases):
            plan = evaluation.make_plan(True)
        self.assertFalse(plan["model_evaluation_executed"])
        specs = {(row["suite"], row["mode"]): row for row in plan["plan"]}
        for mode in ("closed", "tool"):
            self.assertEqual(specs["benchmark_tool_jmmlu", mode]["rows"], 7536)
            self.assertEqual(specs["benchmark_tool_jamcqa", mode]["rows"], 2309)
        self.assertEqual(specs["T1-T8", "original"]["implementation"], "unchanged_original_stage")
        self.assertEqual(specs["practical_known_families", "python"]["rows"], 192)
        self.assertEqual(specs["practical_unseen_families", "python"]["rows"], 48)
        self.assertEqual(specs["search_final198", "formal"]["metric"], "terminal_exact")

    def saved_fixture(self):
        study = self.root / "study"
        branch = study / "cpu_mock_run"; checkpoints = branch / "checkpoints"
        checkpoints.mkdir(parents=True)
        checkpoint = checkpoints / "checkpoint_step_1.pt"
        checkpoint.write_bytes(b"CPU fixture bytes; not a model checkpoint")
        cksha, trainsha = file_sha(checkpoint), hashlib.sha256(b"CPU fixture train manifest").hexdigest()
        receipt = {"step": 1, "checkpoint_sha256": cksha, "training_data_sha256": trainsha}
        atomic_json(checkpoints / "receipt_step_1.json", receipt)
        atomic_json(branch / "training_saved.json", {**receipt, "full_model": True})
        atomic_json(branch / "input_manifest.json", {"output_sha256": trainsha})
        return study, branch, receipt

    def test_saved_checkpoint_requires_matching_training_manifest(self):
        study, branch, receipt = self.saved_fixture()
        path, observed = evaluation.validate_checkpoint(study, "cpu_mock_run", 1, receipt["checkpoint_sha256"].upper())
        self.assertTrue(path.is_file())
        self.assertEqual(observed, receipt)
        atomic_json(branch / "input_manifest.json", {"output_sha256": "f"*64})
        with self.assertRaisesRegex(ValueError, "disagree"):
            evaluation.validate_checkpoint(study, "cpu_mock_run", 1)

    def native_evidence_fixture(self, checkpoint_sha, training_sha, receipt_sha):
        source = self.root / "original_source"; source.mkdir()
        bound = {**evaluation._original_source_binding(), "run": "cpu_mock_run", "step": 1,
                 "checkpoint_sha256": checkpoint_sha, "training_data_sha256": training_sha,
                 "receipt_sha256": receipt_sha, "stage_order": "original Tier1-7, then original Tier8"}
        atomic_json(source / "input_binding.json", bound)
        rows = list(frozen.iter_cases("T4"))
        ordered = [r for r in rows if r["task_type"] == "multiple_choice"] + [r for r in rows if r["task_type"] != "multiple_choice"]
        # No correctness values or generated predictions: these synthetic rows
        # only exercise completeness and ID-binding checks, not model scoring.
        with (source / "tier4_raw.jsonl").open("w") as stream:
            for row in ordered:
                stream.write(json.dumps({"id": row["id"], "target": row["target"].strip(), "cpu_test_fixture": True})+"\n")
        atomic_json(source / "tier8_raw.json", {profile: {"n": 4,
            "details": [{"case_id": row["id"], "cpu_test_fixture": True} for row in frozen.iter_cases("T8", profile)]}
            for profile in ("original_v44", "aligned_v43")})
        for tier in range(1, 8):
            (source / f"tier_{tier}.log").write_text("CPU protocol fixture, no model evaluation\n")
        for label in ("original_tier1_7", "original_tier8"):
            log = source / (label+".log"); log.write_text("CPU protocol fixture, no model evaluation\n")
            atomic_json(source / (label+".complete.json"), {"returncode": 0,
                "log_sha256": file_sha(log), "command_sha256": hashlib.sha256(b"CPU mock command").hexdigest(),
                "observed_tiers": list(range(1, 8)) if label == "original_tier1_7" else []})
        atomic_json(source / "complete.json", {"checkpoint_sha256": checkpoint_sha,
            "tier4_raw_sha256": file_sha(source / "tier4_raw.jsonl"),
            "tier8_raw_sha256": file_sha(source / "tier8_raw.json"),
            "tier1_7_log_sha256": file_sha(source / "original_tier1_7.log"),
            "tier8_log_sha256": file_sha(source / "original_tier8.log")})
        (source / "unrelated_lock.json").write_text("must never be copied")
        return source

    def original_fixture(self):
        study, branch, receipt = self.saved_fixture()
        binding = {"checkpoint_sha": receipt["checkpoint_sha256"], "run": "cpu_mock_run", "step": 1,
                   "training_sha": receipt["training_data_sha256"],
                   "receipt_sha": file_sha(branch / "checkpoints/receipt_step_1.json")}
        source = self.native_evidence_fixture(receipt["checkpoint_sha256"], binding["training_sha"], binding["receipt_sha"])
        return study, source, binding

    @requires_frozen_data
    def test_original_reuse_copies_only_verified_evidence_without_source_writes(self):
        _, source, binding = self.original_fixture()
        before = {p.name: file_sha(p) for p in source.iterdir()}
        output = self.root / "new_results" / "original_tiers"
        receipt = evaluation.reuse_original_stage(source, output, **binding)
        self.assertFalse(receipt["model_evaluation_executed"])
        self.assertFalse(receipt["native_stage_executed"])
        self.assertEqual(len(receipt["copied_files_sha256"]), 15)
        self.assertFalse((output / "unrelated_lock.json").exists())
        self.assertEqual(before, {p.name: file_sha(p) for p in source.iterdir()})
        self.assertEqual(evaluation.verify_reused_original(output, **binding), receipt)
        self.assertEqual(evaluation.reuse_original_stage(source, output, **binding), receipt)

    @requires_frozen_data
    def test_original_reuse_rejects_active_incomplete_or_changed_evidence(self):
        _, source, binding = self.original_fixture()
        output = self.root / "copy"
        atomic_json(source / "tier_stage_active.json", {"cpu_test": True})
        with self.assertRaisesRegex(evaluation.EvaluationContractError, "lock"):
            evaluation.reuse_original_stage(source, output, **binding)
        (source / "tier_stage_active.json").unlink()
        with self.assertRaisesRegex(evaluation.EvaluationContractError, "binding"):
            evaluation.reuse_original_stage(source, output, **{**binding, "checkpoint_sha": "f"*64})
        (source / "original_tier1_7.log").write_text("tampered")
        with self.assertRaisesRegex(evaluation.EvaluationContractError, "hash differs"):
            evaluation.reuse_original_stage(source, output, **binding)
        self.assertFalse(output.exists())

    @requires_frozen_data
    def test_run_with_original_reuse_never_starts_native_stage(self):
        study, source, binding = self.original_fixture()
        model_source = self.root / "model_source_fixture"; model_source.mkdir()
        args = Namespace(output=self.root / "evaluation", resume=False, study=study,
                         run="cpu_mock_run", step=1, checkpoint_sha=binding["checkpoint_sha"],
                         full_knowledge=False, aux_only=False, reuse_original_from=source)
        plan = {"model_evaluation_executed": False, "plan": [
            {"suite": "T1-T8", "mode": "original", "implementation": "unchanged_original_stage"}]}
        fake_module = SimpleNamespace(LocalSearchEngineV42=lambda *args, **kwargs: object())
        fake_model = SimpleNamespace(search_engine=None)
        with patch.object(evaluation, "make_plan", return_value=plan), \
             patch.object(evaluation, "prepare_source", return_value=model_source), \
             patch.object(evaluation, "load_model", return_value=(fake_model, {}, {})), \
             patch.object(evaluation, "run_new_public") as public, \
             patch.object(evaluation, "run_with_logging") as native, \
             patch.dict(sys.modules, {"local_search_engine_v42": fake_module}):
            evaluation.run(args)
            args.resume = True; args.reuse_original_from = None
            evaluation.run(args)
        native.assert_not_called()
        self.assertEqual(public.call_count, 2)
        self.assertTrue((args.output / "original_tiers/reuse_receipt.json").exists())


if __name__ == "__main__":
    unittest.main()
