"""CPU-only checks for the isolated v44fix overlay. No model weights are loaded."""
from __future__ import annotations

import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


HERE = Path(__file__).resolve().parent
OVERLAY = HERE
BASE = HERE
sys.path.insert(0, str(HERE))

from build_v44fix_data import convert
from gpt.prepare_public_v44 import normalize_tir
from integrated_model_v44fix import IntegratedModelV44Fix
from runtime_cotraining_v41_247M import zero_overhead_collate_fn
from tir_engine_v44r1 import PYTHON_BLOCK, observation_text
from tir_masking_v44fix import ToolMaskedMemoryDataset, tool_aware_collate_fn


VOCAB = json.loads((BASE / "base_metadata.json").read_text(encoding="utf-8"))["vocab"]


class TinyExecutor:
    def reset(self):
        return {"reset": True}

    def execute(self, code):
        number = int(code.strip().removeprefix("print(").removesuffix(")"))
        return {"success": True, "stdout": f"{number}\n", "stderr": "", "error": None}


class OverlayEndToEndTests(unittest.TestCase):
    def test_two_python_observations_mask_exact_targets_and_leave_eos(self):
        source = ("まず調べます。\n```py\nprint(1)\n```\n```output\n1\n```\n"
                  "次に調べます。\n```Python\nprint(2)\n```\n```output\n2\n```\n"
                  "したがって $\\boxed{3}$ です。")
        target, segments, receipts = normalize_tir(source, executor=TinyExecutor())
        prompt = "ユーザー: 1と2を調べて説明。\nアシスタント: "
        record = {"bucket": "tir", "prompt": prompt, "target": target,
                  "target_segments": segments,
                  "provenance": {"verification": "executed_all_blocks_in_this_problem",
                                 "tool_blocks": receipts}}
        converted, reason = convert(record)
        self.assertIsNone(reason)
        (prompt, answer, spans), _ = converted
        tools = [span for span in spans if span["role"] == "tool"]
        self.assertEqual(len(tools), 2)
        matches = list(PYTHON_BLOCK.finditer(answer))
        self.assertEqual(len(matches), 2)
        for number, span in enumerate(tools, 1):
            expected = observation_text({"success": True, "stdout": f"{number}\n"})
            self.assertEqual(answer[span["start"]:span["end"]], expected)
            # On a *complete* target the regex consumes the tool's first newline.
            # During incremental generation, $ matches immediately after the
            # third closing backtick, before that newline is injected.
            self.assertEqual(matches[number - 1].end(), span["start"] + 1)
            action_prefix = answer[:span["start"]]
            action_start = action_prefix.rfind("```python\n")
            self.assertTrue(action_prefix.endswith("```"))
            self.assertIsNone(PYTHON_BLOCK.search(action_prefix[:-1], action_start))
            incremental_match = PYTHON_BLOCK.search(action_prefix, action_start)
            self.assertIsNotNone(incremental_match)
            self.assertEqual(incremental_match.end(), len(action_prefix))
            self.assertEqual(answer[span["start"]], "\n")
        self.assertTrue(answer.endswith("\n答え: 3"))

        tampered = json.loads(json.dumps(record, ensure_ascii=False))
        tampered["provenance"]["tool_blocks"][1]["observation_sha256"] = "0" * 64
        self.assertEqual(convert(tampered)[1], "tir_receipt_text_mismatch")

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "train.jsonl"
            path.write_text(json.dumps({"prompt": prompt, "target": answer,
                                        "full_text": prompt + answer, "bucket": "tir",
                                        "target_segments": spans}, ensure_ascii=False) + "\n", encoding="utf-8")
            row = ToolMaskedMemoryDataset(str(path), VOCAB)[0]
            tokens, prompt_len, ranges = row
            packed = tool_aware_collate_fn([row])
            original = zero_overhead_collate_fn([(tokens, prompt_len)])
            self.assertEqual(len(ranges), 2)
            self.assertEqual(len(tokens), len(prompt + answer) + 1)
            for key in ("input_ids", "targets", "valid_mask"):
                self.assertTrue((packed[key] == original[key]).all().item())
            mask = packed["loss_mask"][0]
            self.assertEqual(mask[prompt_len - 1].item(), 1)
            for start, end in ranges:
                self.assertTrue((mask[start:end] == 0).all().item())
                self.assertEqual(mask[start - 1].item(), 1)
                self.assertEqual(mask[end].item(), 1)
            self.assertEqual(mask[len(tokens) - 2].item(), 1)  # true EOS

    def test_tier8_only_path_precedes_all_other_tiers(self):
        text = (OVERLAY / "runtime_v44fix.py").read_text(encoding="utf-8")
        self.assertIn('parser.add_argument("--tier8_only"', text)
        self.assertLess(text.index('if args.tier8_only:\n        print("\\n[TIER 8 ONLY]'),
                        text.index("required_evaluation_inputs ="))
        self.assertIn("args.eval_only = True", text)
        self.assertIn("run_tier8_probe(model, vocab, inv_vocab, args, out_dir, final_ckpt, start_step)", text)

    def test_source_binding_contains_generation_and_corpus(self):
        text = (OVERLAY / "resume_v44fix.py").read_text(encoding="utf-8")
        tree = ast.parse(text)
        binding = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                       and n.name == "training_binding")
        names = {value.value for n in ast.walk(binding) if isinstance(n, ast.Constant)
                 and isinstance(n.value, str) for value in [n]}
        self.assertTrue({"integrated_model_v44fix.py", "tier8_readonly_v44fix.py",
                         "build_v44fix_data.py", "gpt/prepare_public_v44.py",
                         "knowledge_corpus_v43.jsonl", "search_corpus_sha256"} <= names)

    def test_v43_architecture_has_no_new_trainable_layer(self):
        tree = ast.parse((OVERLAY / "integrated_model_v44fix.py").read_text(encoding="utf-8"))
        model = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                     and n.name == "IntegratedModelV44Fix")
        method_names = {n.name for n in model.body if isinstance(n, ast.FunctionDef)}
        self.assertEqual(method_names, {"generate_v44fix"})

    def test_unencodable_python_observation_is_not_silently_truncated(self):
        from test_generate_v44fix import ScriptedNet, INV

        class UnknownCharacterExecutor:
            def execute(self, _code):
                return {"success": True, "stdout": "😀\n", "stderr": "", "error": None}

        script = "確認します。\n```python\nprint(1)\n```\n"
        result = IntegratedModelV44Fix.generate_v44fix(
            ScriptedNet(script), "問題: 計算してください。\n思考:", VOCAB, INV,
            max_tokens=100, executor=UnknownCharacterExecutor())
        self.assertEqual(result["stop_reason"], "tool_output_unencodable")
        self.assertEqual(result["python_invocations"], 1)
        self.assertNotIn("😀", result["output_text"])

    def test_two_python_actions_trigger_before_tool_newline(self):
        from test_generate_v44fix import ScriptedNet, INV

        class TwoResults:
            def execute(self, code):
                number = int(code.strip().removeprefix("print(").removesuffix(")"))
                return {"success": True, "stdout": f"{number}\n", "stderr": "",
                        "error": None, "worker_alive": True}

        script = ("確認します。\n```python\nprint(1)\n```\n```output\n1\n```\n"
                  "次に進みます。\n```python\nprint(2)\n```\n```output\n2\n```\n"
                  "答え: 3")
        result = IntegratedModelV44Fix.generate_v44fix(
            ScriptedNet(script), "問題: 1と2を調べて説明。\n思考:", VOCAB, INV,
            max_tokens=200, executor=TwoResults())
        self.assertEqual(result["python_invocations"], 2)
        self.assertEqual(result["output_text"], script)
        self.assertTrue(result["ended_with_eos"])


if __name__ == "__main__":
    unittest.main()
