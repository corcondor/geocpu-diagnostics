import unittest

from hybrid_data import (STAGE_PYTHON_CAPS, encode_example, plan_exposure)


class HybridDataTests(unittest.TestCase):
    def setUp(self):
        text = "問題: 1+2?\n答え: 3```python\nprint(3)\n```\n```output\n3\n```\n以上。"
        self.vocab = {char: index + 4 for index, char in enumerate(sorted(set(text)))}

    def test_v43_native_boundary_and_eos(self):
        row = {"prompt": "問題: 1+2?\n答え:", "target": " 3"}
        example = encode_example(row, self.vocab, kind="v43")
        mask = example.target_mask()
        self.assertEqual(example.active_positions, 3)  # space, 3, true EOS
        self.assertEqual(sum(mask), 3)
        self.assertEqual(example.tokens[example.prompt_length], self.vocab[" "])
        self.assertTrue(mask[example.prompt_length - 1])
        self.assertTrue(mask[-1])
        self.assertEqual(example.tokens[-1], 3)

    def test_python_tool_targets_only_are_masked(self):
        prompt = "問題: 1+2?\n答え:"
        target = " 3```python\nprint(3)\n```\n```output\n3\n```\n以上。"
        tool_text = "```output\n3\n```\n"
        start = target.index(tool_text)
        row = {"prompt": prompt, "target": target, "full_text": prompt + target,
               "target_segments": [{"start": start, "end": start + len(tool_text),
                                    "role": "tool", "train": False}]}
        example = encode_example(row, self.vocab, kind="python")
        mask = example.target_mask()
        self.assertEqual(example.active_positions, len(target) + 1 - len(tool_text))
        self.assertTrue(mask[len(prompt) - 1])  # first assistant character
        self.assertTrue(mask[len(prompt) + target.index("p") - 1])  # code generation
        self.assertFalse(any(mask[len(prompt)+start-1:len(prompt)+start+len(tool_text)-1]))
        self.assertTrue(mask[-1])  # EOS is still learned
        self.assertEqual(example.tokens[-1], 3)

    def test_python_missing_character_rejected(self):
        row = {"prompt": "問題: 1+2?\n答え:", "target": "未知",
               "target_segments": [{"start": 0, "end": 1, "role": "tool", "train": False}]}
        with self.assertRaises(ValueError):
            encode_example(row, self.vocab, kind="python")

    def test_stage_plan_uses_active_positions_not_row_counts(self):
        old = [encode_example({"prompt": "問題: 1+2?\n答え:", "target": " 3"},
                              self.vocab, kind="v43") for _ in range(200)]
        prompt = "問題: 1+2?\n答え:"
        target = " 3```python\nprint(3)\n```\n```output\n3\n```\n以上。"
        start = target.index("```output")
        pyrow = {"prompt": prompt, "target": target,
                 "target_segments": [{"start": start, "end": target.index("以上。"),
                                      "role": "tool", "train": False}]}
        python = [encode_example(pyrow, self.vocab, kind="python") for _ in range(100)]
        self.assertNotEqual(old[0].active_positions, python[0].active_positions)
        for cap in STAGE_PYTHON_CAPS:
            plan = plan_exposure(old, python, cap=cap, updates=300, batch_size=16, seed=7)
            self.assertGreater(plan.measured_fraction, cap * 0.70)
            self.assertLessEqual(plan.measured_fraction, cap)
            self.assertEqual(len(plan.batches), 300)
        control = plan_exposure(old, python, cap=0.0, updates=300, batch_size=16, seed=7)
        self.assertEqual(control.measured_fraction, 0.0)
        self.assertTrue(all(kind == "v43" for batch in control.batches for kind, _ in batch))
        treatment = plan_exposure(old, python, cap=0.05, updates=300, batch_size=16, seed=7)
        self.assertEqual(control.paired_old_batches, treatment.paired_old_batches)
        self.assertEqual(tuple(tuple(index for _, index in batch) for batch in control.batches),
                         control.paired_old_batches)
        self.assertTrue(all(index == paired for batch, paired_batch in
                            zip(treatment.batches, treatment.paired_old_batches)
                            for (kind, index), paired in zip(batch, paired_batch) if kind == "v43"))

    def test_invalid_python_span_rejected(self):
        prompt = "問題: 1+2?\n答え:"
        row = {"prompt": prompt, "target": " 3",
               "target_segments": [{"start": 1, "end": 3, "role": "tool", "train": False}]}
        with self.assertRaises(ValueError):
            encode_example(row, self.vocab, kind="python")


if __name__ == "__main__":
    unittest.main()
