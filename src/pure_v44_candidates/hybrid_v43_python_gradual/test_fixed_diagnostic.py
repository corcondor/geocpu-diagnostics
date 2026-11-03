import json
from pathlib import Path
import unittest
import zipfile

from build_fixed_diagnostic_cases import build
from eval_fixed_diagnostic import (CASES_NAME, CASES_SHA256, _score_swe,
                                   _validate_saved_rows, _partial_record,
                                   _derived_fp16_source, _verify_hybrid_precision_binding,
                                   FP16_DERIVED_SOURCE_SHA)
from source_pools import file_sha
from hashlib import sha256


ZIP = Path(r"C:\Users\81808\.openclaw\reports\jmmlu_v1\ag_fused_fullqa_training_v1\v43_v44fix_paired_eval_assets\v43_source.zip")
ACTUAL_GENERATIONS = Path(r"C:\Users\81808\.openclaw\reports\jmmlu_v1\ag_fused_fullqa_training_v1\paired_v43_v44fix_continuation_20261103\static_work_v43_resume_v4\results\v43\raw_japanese_answers.jsonl")
ACTUAL_GENERATIONS_SHA = "36b32aa59d74c77f99335393585b22bc8c1c680b51668bf557e052dcda869197"


class DiagnosticTests(unittest.TestCase):
    def test_hybrid_checkpoint_rejects_cross_precision_evaluation(self):
        first = "a" * 64
        second = "b" * 64
        binding = {"precision_profile":"train_fp16_gradscaler_v1",
                   "original_swe200_baseline_complete_sha256":first,
                   "parent_fixed44_raw_diagnostic_complete_sha256":second}
        contract = {"precision_profile":"eval_fp16_native_v1",
                    "parent_swe200_baseline_complete_sha256":first,
                    "parent_fixed44_raw_diagnostic_complete_sha256":second}
        _verify_hybrid_precision_binding(binding, contract)
        contract["precision_profile"] = "eval_bf16_native"
        with self.assertRaises(ValueError):
            _verify_hybrid_precision_binding(binding, contract)
        contract["precision_profile"] = "eval_fp16_native_v1"
        contract["parent_swe200_baseline_complete_sha256"] = "c" * 64
        with self.assertRaises(ValueError):
            _verify_hybrid_precision_binding(binding, contract)

    @unittest.skipUnless(ZIP.is_file(), "Pinned original v43 ZIP unavailable")
    def test_fixed_source_cases(self):
        cases = build(ZIP)
        self.assertEqual(len(cases), 44)
        self.assertEqual(len({row["id"] for row in cases}), 44)
        self.assertEqual(sum(row["group"] == "swe20" for row in cases), 20)
        self.assertEqual(sum(row["group"] == "search8" for row in cases), 8)
        self.assertEqual(sum(row["group"] == "logic8" for row in cases), 8)
        self.assertEqual(sum(row["group"] == "japanese_explanation8" for row in cases), 8)
        self.assertEqual(file_sha(Path(__file__).with_name(CASES_NAME)), CASES_SHA256)

    @unittest.skipUnless(ZIP.is_file(), "Pinned original v43 ZIP unavailable")
    def test_explicit_fp16_generator_source_derivation(self):
        with zipfile.ZipFile(ZIP) as archive:
            original = archive.read("integrated_model_v42.py")
        from hashlib import sha256
        changed = _derived_fp16_source(original, "integrated_model_v42.py")
        self.assertEqual(sha256(changed).hexdigest(), FP16_DERIVED_SOURCE_SHA["integrated_model_v42.py"])
        self.assertEqual(original.count(b"dtype=torch.bfloat16"), 1)
        overlay = Path(__file__).resolve().parents[2] / "v44fix" / "integrated_model_v44fix.py"
        changed = _derived_fp16_source(overlay.read_bytes(), "integrated_model_v44fix.py")
        self.assertEqual(sha256(changed).hexdigest(), FP16_DERIVED_SOURCE_SHA["integrated_model_v44fix.py"])

    def test_cpu_and_python_execution_are_separate_scores(self):
        native = {"output_text": "<SET R0 5> => [R0=5] 違う", "cpu_trace": [{"dest":"R0", "val":5}],
                  "cpu_invocations": 1, "ended_with_eos": True}
        self.assertTrue(_score_swe(native, "5", "native", 1)["old_cpu_native_rule_correct"])
        self.assertTrue(_score_swe(native, "5", "native", 1)["old_cpu_native_rule_and_natural_eos_correct"])
        text_only = {"output_text":"5", "cpu_trace":[], "cpu_invocations":0,
                     "ended_with_eos":True}
        self.assertTrue(_score_swe(text_only, "5", "native", 1)["old_original_evaluator_correct"])
        self.assertFalse(_score_swe(text_only, "5", "native", 1)["old_cpu_native_rule_correct"])
        self.assertIsNone(_score_swe(text_only, "5", "native", 1)["cpu_final"])
        self.assertIsNone(_score_swe(native, "5", "native", 1)["python_executed_and_numeric_correct"])
        python = {"output_text": "```python\nprint(5)\n```\n答えは5です。",
                  "model_generated_text": "```python\nprint(5)\n```\n答えは5です。",
                  "cpu_trace": [], "cpu_invocations": 0,
                  "python_trace": [{"code":"print(5)", "success": True, "stdout":"[R0=5]\n"}],
                  "ended_with_eos": True}
        self.assertTrue(_score_swe(python, "5", "python_cued", 1)["python_executed_and_numeric_correct"])
        self.assertTrue(_score_swe(python, "5", "python_cued", 1)["python_execution_and_final_answer_correct"])
        self.assertTrue(_score_swe(python, "5", "python_cued", 1)["python_execution_final_and_eos_correct"])
        python["ended_with_eos"] = False
        self.assertFalse(_score_swe(python, "5", "python_cued", 1)["python_execution_final_and_eos_correct"])
        python["ended_with_eos"] = True
        self.assertTrue(_score_swe(python, "5", "python_cued", 1)["python_japanese_explanation_present"])
        self.assertIsNone(_score_swe(python, "5", "python_cued", 1)["python_grounded_explanation_grade"])
        python["model_generated_text"] = "```python\nprint(5)\n```\n答えは7です。"
        self.assertTrue(_score_swe(python, "5", "python_cued", 1)["python_executed_and_numeric_correct"])
        self.assertFalse(_score_swe(python, "5", "python_cued", 1)["python_execution_and_final_answer_correct"])
        python["model_generated_text"] = "```python\nprint(5)\n```\n5"
        self.assertFalse(_score_swe(python, "5", "python_cued", 1)["python_japanese_explanation_present"])
        python["model_generated_text"] = "```python\nprint(5)\n```"
        self.assertFalse(_score_swe(python, "5", "python_cued", 1)["python_final_answer_numeric_correct"])
        python["python_trace"] = [{"code":"print(5)", "success": True, "stdout":"unrelated\n"}]
        self.assertFalse(_score_swe(python, "5", "python_cued", 1)["python_executed_and_numeric_correct"])
        python["python_trace"] = []
        self.assertFalse(_score_swe(python, "5", "python_cued", 1)["python_executed_and_numeric_correct"])

    def test_partial_rows_reject_cross_checkpoint_mix(self):
        case = {"id":"x", "group":"swe20", "prompt":"問題", "target":"5",
                "profiles":["python_cued"]}
        prompt = "Pythonで計算し、実行結果を根拠に日本語で説明してください。\n問題"
        row = {"case_id":"x", "profile":"python_cued", "prompt":prompt,
               "arm":"python", "checkpoint_sha256":"a"*64,
               "cases_sha256":CASES_SHA256,"source_sha256":"b"*64,
               "overlay_sha256":"c"*64,
               "precision_profile":"eval_bf16_native",
               "v43_zip_sha256":"0102dc9b01b4b795c3123b18cefc7a8734e47a06dd85292af464d3232704d198",
               "expected_answer":"5", "group":"swe20", "generated_tokens":10,
               "generation_iteration_limit":2048,
               "generated_token_ids":list(range(10)),
               "raw_answer":"実行結果は5", "model_generated_text":"答えは5"}
        row["generated_token_ids_sha256"] = sha256(json.dumps(row["generated_token_ids"],
            separators=(",", ":")).encode()).hexdigest()
        row["raw_answer_sha256"] = sha256(row["raw_answer"].encode()).hexdigest()
        row["model_generated_text_sha256"] = sha256(row["model_generated_text"].encode()).hexdigest()
        expected = [(case, "python_cued")]
        _validate_saved_rows([row], expected, arm="python", checkpoint_sha="a"*64,
                             evaluator_sha="b"*64, overlay_sha="c"*64,
                             precision_profile="eval_bf16_native")
        with self.assertRaises(ValueError):
            _validate_saved_rows([row], expected, arm="python", checkpoint_sha="d"*64,
                                 evaluator_sha="b"*64, overlay_sha="c"*64,
                                 precision_profile="eval_bf16_native")
        row["raw_answer"] = "変更"
        with self.assertRaises(ValueError):
            _validate_saved_rows([row], expected, arm="python", checkpoint_sha="a"*64,
                                 evaluator_sha="b"*64, overlay_sha="c"*64,
                                 precision_profile="eval_bf16_native")
        self.assertEqual(_partial_record(arm="python", checkpoint_sha="a"*64,
                         evaluator_sha="b"*64, overlay_sha="c"*64,
                         raw_sha="d"*64, completed=1, required=84,
                         precision_profile="eval_bf16_native")["max_generated_tokens"], 2048)

    @unittest.skipUnless(ACTUAL_GENERATIONS.is_file(), "Pinned v43 saved generation unavailable")
    def test_actual_v43_generation_list_and_observation_length(self):
        self.assertEqual(file_sha(ACTUAL_GENERATIONS), ACTUAL_GENERATIONS_SHA)
        saved = [json.loads(line) for line in ACTUAL_GENERATIONS.read_text(encoding="utf-8").splitlines()]
        item = max(saved, key=lambda row: len(row["generation"]["generated_tokens"]))
        generated = item["generation"]
        self.assertIsInstance(generated["generated_tokens"], list)
        self.assertGreater(len(generated["generated_tokens"]), 2048)
        case = {"id":"actual_v43_saved_one", "group":"japanese_explanation8",
                "prompt":item["prompt"], "target":"unused", "profiles":["native"]}
        row = {"case_id":case["id"], "group":case["group"], "profile":"native",
               "prompt":case["prompt"], "expected_answer":case["target"],
               "arm":"parent", "checkpoint_sha256":"a"*64,
               "cases_sha256":CASES_SHA256, "source_sha256":"b"*64,
               "overlay_sha256":"c"*64,
               "precision_profile":"eval_bf16_native",
               "v43_zip_sha256":"0102dc9b01b4b795c3123b18cefc7a8734e47a06dd85292af464d3232704d198",
               "generation_iteration_limit":2048,
               "generated_tokens":len(generated["generated_tokens"]),
               "generated_token_ids":generated["generated_tokens"],
               "raw_answer":generated["output_text"],
               "model_generated_text":generated["model_generated_text"]}
        row["generated_token_ids_sha256"] = sha256(json.dumps(row["generated_token_ids"],
            separators=(",", ":")).encode()).hexdigest()
        row["raw_answer_sha256"] = sha256(row["raw_answer"].encode()).hexdigest()
        row["model_generated_text_sha256"] = sha256(row["model_generated_text"].encode()).hexdigest()
        _validate_saved_rows([row], [(case,"native")], arm="parent",
                             checkpoint_sha="a"*64, evaluator_sha="b"*64, overlay_sha="c"*64,
                             precision_profile="eval_bf16_native")


if __name__ == "__main__":
    unittest.main()
