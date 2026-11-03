import json
import sys
import unittest
import zipfile

from src.pure_v44_candidates.v43_swe_to_python.convert_swe import (
    prompt_program, task_result, parse_swe_suffix, expected_commands,
)
from src.pure_v44_candidates.v43_alu_to_python.convert_sample import V44_DIR

ZIP = r"G:\マイドライブ\GeoCPU_Frontier_CoTraining_v43\Grand_cotraining_v43.zip"
MEMBER = "data/train_v43_knowledge_unified.jsonl"


class SourceRowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with zipfile.ZipFile(ZIP) as archive, archive.open(MEMBER) as lines:
            cls.rows = {}
            for index, raw in enumerate(lines, 1):
                if index in (10, 44, 71, 113, 144):
                    cls.rows[index] = json.loads(raw)
                if index >= 144: break

    def test_real_accumulator_full_instruction_plan(self):
        row = self.rows[10]
        code = prompt_program(row["prompt"])
        final, writes = task_result(row["source"], code)
        pairs, stated = parse_swe_suffix(row["full_text"][len(row["prompt"]):])
        self.assertEqual(final, stated)
        self.assertEqual(writes, [73,73])
        self.assertEqual([p.group(0).split("=>",1)[0].rstrip() for p in pairs],
                         expected_commands(row["source"], code))

    def test_real_conditional_mismatch_rejected_before_cpu(self):
        row = self.rows[71]
        code = prompt_program(row["prompt"])
        final, _ = task_result(row["source"], code)
        _, stated = parse_swe_suffix(row["full_text"][len(row["prompt"]):])
        self.assertEqual(final, 1067)
        self.assertEqual(stated, 1059)

    def test_real_bxor_branch_and_modulo(self):
        for index in (44, 113, 144):
            row = self.rows[index]
            code = prompt_program(row["prompt"])
            final, _ = task_result(row["source"], code)
            pairs, stated = parse_swe_suffix(row["full_text"][len(row["prompt"]):])
            self.assertEqual(final, stated)
            self.assertEqual([p.group(0).split("=>",1)[0].rstrip() for p in pairs],
                             expected_commands(row["source"], code))

    def test_native_tir_requires_generated_fence_at_line_start(self):
        sys.path.insert(0,str(V44_DIR))
        from tir_engine_v44r1 import PYTHON_BLOCK
        code="print(7)\n"
        self.assertIsNone(PYTHON_BLOCK.search(" ```python\n"+code+"```\n"))
        match=PYTHON_BLOCK.search(" \n```python\n"+code+"```\n")
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1),code)


if __name__ == "__main__": unittest.main()
