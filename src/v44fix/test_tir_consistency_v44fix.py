"""Small CPU-only fixtures for Python action/observation training alignment."""
import json
from pathlib import Path
import tempfile
import unittest

import torch

from build_v44fix_data import convert
from gpt.prepare_public_v44 import RowRejected, normalize_tir
from runtime_cotraining_v41_247M import zero_overhead_collate_fn
from tir_engine_v44r1 import PYTHON_BLOCK
from tir_masking_v44fix import ToolMaskedMemoryDataset, tool_aware_collate_fn


ROOT = Path(__file__).resolve().parent
VOCAB = json.loads((ROOT / "base_metadata.json").read_text(encoding="utf-8"))["vocab"]


class ArithmeticExecutor:
    def reset(self):
        return {"reset": True}

    def execute(self, code):
        expression = code.strip().removeprefix("print(").removesuffix(")")
        answer = eval(expression, {"__builtins__": {}}, {})
        return {"success": True, "stdout": f"{answer}\n", "stderr": "", "error": None}


def public_tir_record(code_header="PY", leading=True):
    intro = "" if leading else "まず計算します。\n"
    published = (intro + "```" + code_header + "\nprint(2+3)\n```\n"
                 "```output\n5\n```\nしたがって $\\boxed{5}$ です。")
    target, segments, receipts = normalize_tir(published, executor=ArithmeticExecutor())
    prompt = "ユーザー: 2+3を計算して説明。\nアシスタント: "
    return {"bucket": "tir", "prompt": prompt, "target": target,
            "full_text": prompt + target, "target_segments": segments,
            "provenance": {"verification": "executed_all_blocks_in_this_problem", "tool_blocks": receipts}}


class TIRConsistencyTests(unittest.TestCase):
    def test_leading_alias_fence_is_executable_and_observation_is_not_a_label(self):
        record = public_tir_record()
        converted, reason = convert(record)
        self.assertIsNone(reason)
        (prompt, target, segments), _ = converted
        self.assertTrue(target.startswith(" \n```python\n"))
        self.assertEqual(len(list(PYTHON_BLOCK.finditer(target))), 1)
        tool = [s for s in segments if s["role"] == "tool"]
        self.assertEqual(len(tool), 1)
        self.assertFalse(tool[0]["train"])
        self.assertIn("```output\n5\n```", target[tool[0]["start"]:tool[0]["end"]])
        self.assertTrue(target.endswith("\n答え: 5"))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.jsonl"
            row = {"prompt": prompt, "target": target, "full_text": prompt + target,
                   "bucket": "tir", "target_segments": segments}
            path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            dataset = ToolMaskedMemoryDataset(str(path), VOCAB)
            tokens, prompt_len, ranges = dataset[0]
            self.assertEqual(len(ranges), 1)
            packed = tool_aware_collate_fn([dataset[0]])
            original = zero_overhead_collate_fn([(tokens, prompt_len)])
            start, end = ranges[0]
            self.assertTrue(bool((original["loss_mask"][0, start:end] == 1).all()))
            self.assertTrue(bool((packed["loss_mask"][0, start:end] == 0).all()))
            self.assertEqual(packed["loss_mask"][0, prompt_len - 1].item(), 1)
            self.assertEqual(packed["loss_mask"][0, start - 1].item(), 1)
            self.assertEqual(packed["loss_mask"][0, end].item(), 1)
            self.assertEqual(packed["loss_mask"][0, len(tokens) - 2].item(), 1)  # EOS
            self.assertTrue(torch.equal(packed["input_ids"], original["input_ids"]))
            self.assertTrue(torch.equal(packed["targets"], original["targets"]))
            self.assertTrue(torch.equal(packed["valid_mask"], original["valid_mask"]))

    def test_prose_then_python_uses_same_header_and_preserves_tool_span(self):
        converted, reason = convert(public_tir_record("Python", leading=False))
        self.assertIsNone(reason)
        (_, target, segments), _ = converted
        self.assertIn("\n```python\n", target)
        self.assertEqual(len(list(PYTHON_BLOCK.finditer(target))), 1)
        self.assertEqual(len([s for s in segments if s["role"] == "tool"]), 1)

    def test_two_code_blocks_keep_both_observation_spans(self):
        published = ("```py\nprint(1)\n```\n```output\n1\n```\n次を計算します。\n"
                     "```Python\nprint(2)\n```\n```output\n2\n```\n"
                     "結論は $\\boxed{3}$ です。")
        target, segments, receipts = normalize_tir(published, executor=ArithmeticExecutor())
        record = {"bucket": "tir", "prompt": "ユーザー: 1+2を計算。\nアシスタント: ",
                  "target": target, "target_segments": segments,
                  "provenance": {"verification": "executed_all_blocks_in_this_problem", "tool_blocks": receipts}}
        converted, reason = convert(record)
        self.assertIsNone(reason)
        (_, answer, converted_segments), _ = converted
        self.assertEqual(len(list(PYTHON_BLOCK.finditer(answer))), 2)
        tool_texts = [answer[s["start"]:s["end"]] for s in converted_segments
                      if s["role"] == "tool"]
        self.assertEqual(len(tool_texts), 2)
        self.assertIn("```output\n1\n```", tool_texts[0])
        self.assertIn("```output\n2\n```", tool_texts[1])

    def test_missing_tool_segments_are_rejected_before_training(self):
        record = public_tir_record()
        record["target_segments"] = []
        converted, reason = convert(record)
        self.assertIsNone(converted)
        self.assertEqual(reason, "tir_missing_tool_segments")

    def test_author_recorded_or_missing_execution_receipt_is_rejected(self):
        record = public_tir_record()
        record["provenance"]["verification"] = "public_author_recorded_not_executed_here"
        self.assertEqual(convert(record)[1], "tir_observation_not_locally_executed")
        record["provenance"]["verification"] = "executed_all_blocks_in_this_problem"
        record["provenance"]["tool_blocks"] = []
        self.assertEqual(convert(record)[1], "tir_missing_or_unverified_tool_receipt")

    def test_unavailable_dependency_is_a_countable_row_rejection(self):
        class MissingDependency(ArithmeticExecutor):
            def execute(self, code):
                return {"exception_type": "ModuleNotFoundError", "error": "missing fixture module"}

        solution = "```python\nimport missing_fixture_module\n```\n```output\n\n```\n$\\boxed{1}$"
        with self.assertRaisesRegex(RowRejected, "executed_python_dependency_unavailable"):
            normalize_tir(solution, executor=MissingDependency())

    def test_legacy_anchor_mask_is_unchanged(self):
        row = {"prompt": "指示: 季節を答えて。\n回答:", "target": " 春です。", "bucket": "anchor"}
        row["full_text"] = row["prompt"] + row["target"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.jsonl"
            path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            dataset = ToolMaskedMemoryDataset(str(path), VOCAB)
            tokens, prompt_len, ranges = dataset[0]
            self.assertEqual(ranges, ())
            packed = tool_aware_collate_fn([dataset[0]])
            original = zero_overhead_collate_fn([(tokens, prompt_len)])
            for key in ("input_ids", "targets", "valid_mask", "loss_mask"):
                self.assertTrue(torch.equal(packed[key], original[key]))


if __name__ == "__main__":
    unittest.main()
