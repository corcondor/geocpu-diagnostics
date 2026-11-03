import json
from pathlib import Path
import unittest

from build_fixed_diagnostic_cases import build
from eval_fixed_diagnostic import (CASES_NAME, CASES_SHA256, _score_swe,
                                   _validate_saved_rows, _partial_record)
from source_pools import file_sha
from hashlib import sha256


ZIP = Path(r"C:\Users\81808\.openclaw\reports\jmmlu_v1\ag_fused_fullqa_training_v1\v43_v44fix_paired_eval_assets\v43_source.zip")


class DiagnosticTests(unittest.TestCase):
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

    def test_cpu_and_python_execution_are_separate_scores(self):
        native = {"output_text": "<SET R0 5> => [R0=5] 違う", "cpu_trace": [{"dest":"R0", "val":5}],
                  "cpu_invocations": 1}
        self.assertTrue(_score_swe(native, "5", "native", 1)["old_cpu_native_rule_correct"])
        self.assertIsNone(_score_swe(native, "5", "native", 1)["python_executed_and_numeric_correct"])
        python = {"output_text": "```python\nprint(5)\n```\n答えは5です。",
                  "model_generated_text": "```python\nprint(5)\n```\n答えは5です。",
                  "cpu_trace": [], "cpu_invocations": 0,
                  "python_trace": [{"code":"print(5)", "success": True, "stdout":"[R0=5]\n"}]}
        self.assertTrue(_score_swe(python, "5", "python_cued", 1)["python_executed_and_numeric_correct"])
        self.assertTrue(_score_swe(python, "5", "python_cued", 1)["python_execution_and_final_answer_correct"])
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
               "v43_zip_sha256":"0102dc9b01b4b795c3123b18cefc7a8734e47a06dd85292af464d3232704d198",
               "expected_answer":"5", "group":"swe20", "generated_tokens":10,
               "raw_answer":"実行結果は5", "model_generated_text":"答えは5"}
        row["raw_answer_sha256"] = sha256(row["raw_answer"].encode()).hexdigest()
        row["model_generated_text_sha256"] = sha256(row["model_generated_text"].encode()).hexdigest()
        expected = [(case, "python_cued")]
        _validate_saved_rows([row], expected, arm="python", checkpoint_sha="a"*64,
                             evaluator_sha="b"*64, overlay_sha="c"*64)
        with self.assertRaises(ValueError):
            _validate_saved_rows([row], expected, arm="python", checkpoint_sha="d"*64,
                                 evaluator_sha="b"*64, overlay_sha="c"*64)
        row["raw_answer"] = "変更"
        with self.assertRaises(ValueError):
            _validate_saved_rows([row], expected, arm="python", checkpoint_sha="a"*64,
                                 evaluator_sha="b"*64, overlay_sha="c"*64)
        self.assertEqual(_partial_record(arm="python", checkpoint_sha="a"*64,
                         evaluator_sha="b"*64, overlay_sha="c"*64,
                         raw_sha="d"*64, completed=1, required=84)["max_generated_tokens"], 2048)


if __name__ == "__main__":
    unittest.main()
