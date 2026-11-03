#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Control-flow tests for IntegratedModelV44Fix.generate_v44fix with a scripted stand-in network.
Real ALU circuits, real controller, real Python worker; only the 247M network is replaced by a script.
"""
import json, sys, unittest
from pathlib import Path

import torch

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))
from integrated_model_v42 import IntegratedModelV42
from integrated_model_v44fix import IntegratedModelV44Fix
from cpu_circuit_v30 import ExactRAM8x32, CompiledOperators32
from qwenputer_controller_v30 import QwenputerExecutionController
from local_search_engine_v42 import LocalSearchEngineV42

META = json.loads((ROOT / "base_metadata.json").read_bytes())
VOCAB = META["vocab"]
INV = {int(v): k for k, v in VOCAB.items()}
V = max(VOCAB.values()) + 1


class FakeState:
    def __init__(self, toks, logits):
        self.toks, self.last_logits = toks, logits
        self.compact_prefix = torch.zeros(1, len(toks))


class ScriptedNet:
    """Emits SCRIPT one char at a time as long as everything generated so far is a prefix of it; else EOS."""
    def __init__(self, script):
        missing = sorted({c for c in script if c not in VOCAB})
        assert not missing, f"script chars not in vocab: {missing}"
        self.script = script
        self.ram, self.alu = ExactRAM8x32(), CompiledOperators32()
        self.controller = QwenputerExecutionController(self.ram, self.alu)
        self.search_engine = LocalSearchEngineV42(corpus_path=ROOT / "data" / "knowledge_corpus_v43.jsonl")
        self.config = type("Cfg", (), {"max_positions": 2048})()
        self._p = torch.zeros(1)
        self.n_prompt = 0

    def parameters(self):
        yield self._p

    def _logits(self, toks):
        gen = "".join(INV.get(t, "") for t in toks[self.n_prompt:])
        nxt = self.script[len(gen)] if self.script.startswith(gen) and len(gen) < len(self.script) else None
        lg = torch.full((1, 1, V), -1e9)
        lg[0, 0, VOCAB[nxt] if nxt is not None else 3] = 0.0
        return lg

    def prefill(self, t, arm="learned"):
        toks = t[0].tolist()
        self.n_prompt = len(toks)
        return FakeState(toks, self._logits(toks))

    def step_state(self, tok, state, compute_logits=True):
        toks = state.toks + [int(tok.item())]
        return FakeState(toks, self._logits(toks) if compute_logits else state.last_logits)


EXECUTOR = None  # one real worker shared by the tests (slow to start on Windows with the stripped env)


def v44fix(script, prompt, **kw):
    kw.setdefault("executor", EXECUTOR)
    return IntegratedModelV44Fix.generate_v44fix(ScriptedNet(script), prompt, VOCAB, INV, max_tokens=400, **kw)


class GenerateV44FixTests(unittest.TestCase):
    PROMPT = "問題: テスト\n思考:"

    @classmethod
    def setUpClass(cls):
        global EXECUTOR
        from tir_engine_v44r1 import PythonExecutor
        EXECUTOR = PythonExecutor(5.0, 8192, 32768, startup_timeout_seconds=300.0)

    @classmethod
    def tearDownClass(cls):
        EXECUTOR.close()

    def test_alu_and_python_in_one_answer(self):
        script = "まず<SET R0 7> => [R0=7] 次に計算します。\n```python\nprint(2+3)\n```\n```output\n5\n```\n答え: 5"
        g = v44fix(script, self.PROMPT)
        self.assertEqual(g["cpu_invocations"], 1)
        self.assertEqual(g["python_invocations"], 1)
        self.assertEqual(g["python_trace"][0]["stdout"], "5\n")
        self.assertTrue(g["ended_with_eos"])
        self.assertEqual(g["output_text"], script)

    def test_back_to_back_python_blocks(self):
        script = "確認します。\n```python\nprint(1)\n```\n```output\n1\n```\n```python\nprint(2)\n```\n```output\n2\n```\n答え: 2"
        g = v44fix(script, self.PROMPT)
        self.assertEqual(g["python_invocations"], 2)
        self.assertTrue(g["ended_with_eos"])
        self.assertEqual(g["output_text"], script)

    def test_huge_python_output_is_bounded(self):
        g = v44fix("確認します。\n```python\nprint('7'*20000)\n```", self.PROMPT, executor=None,
                   python_startup_timeout=300.0)
        self.assertEqual(g["python_invocations"], 1)
        self.assertLess(len(g["output_text"]), 1400)

    def test_code_in_prompt_is_never_executed(self):
        prompt = "タスク: SWE\n```python\nprint('PROMPT')\n```\n最終値 R0 ="
        g = v44fix(" 3", prompt)
        self.assertEqual(g["python_invocations"], 0)
        self.assertEqual(g["output_text"], " 3")

    def test_identical_to_v42_when_no_python(self):
        script = "演算します。<MUL R0 87 14> => [R0=1218] <ADD R0 R0 23> => [R0=1241] 計算結果は1241です。答え: 1241"
        a = IntegratedModelV42.generate_with_qwenputer_v42(ScriptedNet(script), self.PROMPT, VOCAB, INV, max_tokens=400)
        for py in (True, False):
            b = v44fix(script, self.PROMPT, enable_python=py)
            self.assertEqual(a["generated_tokens"], b["generated_tokens"])
            self.assertEqual(a["output_text"], b["output_text"])
            self.assertEqual(a["cpu_invocations"], b["cpu_invocations"])

    def test_loop_guard_optional(self):
        script = "ABCDEFGH" * 6
        self.assertEqual(v44fix(script, self.PROMPT, loop_guard=True)["stop_reason"], "loop_guard")
        g = v44fix(script, self.PROMPT, loop_guard=False)
        self.assertTrue(g["ended_with_eos"])
        self.assertEqual(g["output_text"], script)

    def test_model_written_observation_stops(self):
        g = v44fix("結果:\n```output\n5\n```\n答え: 5", self.PROMPT)
        self.assertEqual(g["stop_reason"], "model_generated_observation")
        self.assertEqual(g["python_invocations"], 0)

    def test_python_error_is_observed(self):
        g = v44fix("確認します。\n```python\nprint(1/0)\n```", self.PROMPT)
        self.assertEqual(g["python_invocations"], 1)
        self.assertFalse(g["python_trace"][0]["success"])
        self.assertIn("ZeroDivisionError", g["output_text"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
