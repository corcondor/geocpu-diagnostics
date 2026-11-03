"""Meaningful CPU, tool, and masking checks for the bounded conversion prototype."""

import json
import re
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import convert_sample as candidate

sys.path.insert(0, str(candidate.SOURCE_DIR))
sys.path.insert(0, str(candidate.V44_DIR))
import torch
from cpu_circuit_v30 import CompiledOperators32, ExactRAM8x32
from qwenputer_controller_v30 import QwenputerExecutionController
from tir_engine_v44r1 import PYTHON_BLOCK, PythonExecutor, observation_text
from training_data_v44 import encode_supervision


class ConverterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.set_grad_enabled(False)
        cls.cpu = QwenputerExecutionController(ExactRAM8x32(), CompiledOperators32())
        cls.tool = PythonExecutor()

    @classmethod
    def tearDownClass(cls):
        cls.tool.close()

    def test_real_cpu_and_python_agree_on_unsigned_wrap_and_zero_divisor(self):
        # Original controller decodes 32-bit words as unsigned. Negative literals
        # are represented modulo 2**32; DIV/MOD by zero substitute divisor 1.
        cases = [
            ("<SET R0 -1>", "[R0=4294967295]"),
            ("<DIV R1 R0 2>", "[R1=2147483647]"),
            ("<MOD R2 R0 2>", "[R2=1]"),
            ("<DIV R3 R0 0>", "[R3=4294967295]"),
            ("<SUB R4 0 1>", "[R4=4294967295]"),
            ("<SET R10 7>", "[R2=7]"),
        ]
        state = self.cpu.initial_cpu_state()
        self.tool.reset()
        for index, (command, expected) in enumerate(cases):
            parsed = self.cpu.parse_command(command)
            self.assertIsNotNone(parsed)
            state, _, original = self.cpu.execute_step(state, parsed)
            self.assertEqual(original, expected)
            pair = candidate.PAIR.fullmatch(command + " => " + expected)
            self.assertIsNotNone(pair)
            result = self.tool.execute(candidate.instruction_code(pair, index == 0))
            self.assertTrue(result["success"], result)
            self.assertEqual(result["stdout"], original + "\n")

    def test_rejects_unpaired_commands_and_preexisting_tool_fence(self):
        with self.assertRaisesRegex(candidate.Reject, "unpaired_or_unsupported_trace"):
            candidate.parse_pairs("<SET R0 1> => [R0=1] <ADD R0 2>")
        row = {"source": "Algebraic_Composite_Aligned", "prompt": "問題: 数式 (1 * 1 + 0) を 2 で割った余りを求めよ。\n思考:",
               "target": "```python\nprint(1)\n``` <SET R0 1> => [R0=1] 答え: 1"}
        row["full_text"] = row["prompt"] + " " + row["target"]
        with self.assertRaisesRegex(candidate.Reject, "preexisting_code_or_observation_fence"):
            candidate.convert_row(row, self.cpu, self.tool, {" ": 5})

    def test_actual_converted_sample_prompt_boundary_tool_mask_and_eos(self):
        sample = HERE / "sample_128" / "converted_sample.jsonl"
        self.assertTrue(sample.exists(), "Run convert_sample.py first")
        with sample.open(encoding="utf-8") as source:
            items = [json.loads(line) for line in source]
        self.assertGreaterEqual(len(items), 3)
        # A char-to-id mapping is sufficient for the exact training mask test.
        characters = sorted(set("".join(item["full_text"] for item in items)))
        vocab = {char: index + 5 for index, char in enumerate(characters)}
        for item in items:
            self.assertEqual(item["full_text"], item["prompt"] + item["target"])
            self.assertTrue(item["target"].startswith(" ") or item["original_source"] == "swe_alu_accumulate_sum")
            ids, tags = encode_supervision(item, vocab)
            self.assertEqual(ids[-1], 3)
            self.assertEqual(tags[-1], 1)
            for span in item["target_segments"]:
                self.assertEqual(span["role"], "tool")
                self.assertIs(span["train"], False)
                self.assertEqual(set(tags[len(item["prompt"]) + span["start"]:len(item["prompt"]) + span["end"]]), {0})
                self.assertTrue(item["target"][span["start"]:span["end"]].startswith("\n```output\n"))
            self.tool.reset()
            blocks = list(PYTHON_BLOCK.finditer(item["target"]))
            self.assertEqual(len(blocks), len(item["target_segments"]))
            for block, span in zip(blocks, item["target_segments"]):
                result = self.tool.execute(block.group(1))
                self.assertTrue(result["success"], result)
                self.assertEqual(observation_text(result), item["target"][span["start"]:span["end"]])


if __name__ == "__main__":
    unittest.main()
