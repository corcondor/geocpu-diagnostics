"""CPU-only Tier 8 call/schema checks; no model loads or training."""

import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from tier8_readonly_v44fix import CASES, ALIGNED_CASES, evaluate_tier8, evaluate_both_profiles


class Tier8ReadOnlyTests(unittest.TestCase):
    def test_calls_and_output_schema_with_injected_observation_excluded(self):
        called = []

        def fake_generate(prompt):
            called.append(prompt)
            if len(called) == 1:
                return {"model_generated_text": "```python\nprint(2+2)\n```\n結論は4です。",
                        "output_text": "```python\nprint(2+2)\n```\n```output\n4\n```\n結論は4です。",
                        "python_invocations": 1, "python_trace": [{"success": True}],
                        "ended_with_eos": True}
            if len(called) == 2:
                return {"model_generated_text": "計算を求めます。", "output_text": "計算を求めます。",
                        "python_invocations": 0, "python_trace": []}
            if len(called) == 3:
                return {"model_generated_text": "```cpp\nint main() {}\n```\n動作を説明します。",
                        "output_text": "```cpp\nint main() {}\n```\n動作を説明します。"}
            return {"model_generated_text": "```rust\nfn main() {}\n```\n動作を説明します。",
                    "output_text": "```rust\nfn main() {}\n```\n動作を説明します。"}

        result = evaluate_tier8(fake_generate)
        self.assertEqual(called, [case[1] for case in CASES])
        self.assertEqual((result["tier"], result["n"]), (8, 4))
        self.assertEqual(result["python_invocation_count"], 1)
        self.assertEqual(result["successful_python_invocation_count"], 1)
        self.assertEqual(result["expected_language_fence_rate"], 75.0)
        self.assertEqual(result["tir_code_rate"], 75.0)
        self.assertEqual(result["narrative_rate"], 50.0)
        self.assertNotIn("```output", result["details"][0]["model_generated_text"])
        self.assertIn("```output", result["details"][0]["output_text"])

    def test_missing_model_generated_text_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "model_generated_text"):
            evaluate_tier8(lambda prompt: {"output_text": "```python\nprint(1)\n```"})

    def test_original_and_v43_aligned_prompts_are_separate_profiles(self):
        self.assertEqual([case[0] for case in CASES], [case[0] for case in ALIGNED_CASES])
        for original, aligned in zip(CASES, ALIGNED_CASES):
            self.assertNotEqual(original[1], aligned[1])
            self.assertIn(original[1].removeprefix("ユーザー: ").removesuffix("\nアシスタント: "), aligned[1])
        called = []

        def fake_generate(prompt):
            called.append(prompt)
            return {"model_generated_text": "答えを求めます。", "output_text": "答えを求めます。"}

        result = evaluate_both_profiles(fake_generate)
        self.assertEqual(set(result), {"original_v44", "aligned_v43"})
        self.assertEqual(result["original_v44"]["profile"], "original_v44")
        self.assertEqual(result["aligned_v43"]["profile"], "aligned_v43")
        self.assertEqual(called[:4], [case[1] for case in CASES])
        self.assertEqual(called[4:], [case[1] for case in ALIGNED_CASES])


if __name__ == "__main__":
    unittest.main()
