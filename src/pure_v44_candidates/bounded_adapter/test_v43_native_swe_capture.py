"""CPU-only tests for the original-v43 read-only SWE capture entrypoint."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

from v43_native_swe_capture_readonly import (CaptureProxy, SOURCE_SHA256,
                                             SWE_MEMBER, read_jsonl, sha256_file)


class FakeV43Model:
    def __init__(self, expected_prompt: str, answer: str):
        self.expected_prompt = expected_prompt
        self.answer = answer

    def eval(self):
        return self

    def generate_with_qwenputer(self, prompt, _vocab, _inverse, *, max_tokens,
                                 arm, repetition_penalty):
        assert prompt == self.expected_prompt
        assert (max_tokens, arm, repetition_penalty) == (2048, "learned", 1.0)
        value = int(self.answer)
        return {
            "output_text": "  ignored prose  ", "tokens": [42, 43],
            "cpu_trace": [{"step": 1, "cmd": "<SET R0 0>", "op": "SET",
                           "dest": "R0", "val": value, "res_str": f"[R0={value}]"}],
            "cpu_invocations": 1, "ended_with_eos": True,
        }


class NativeCaptureTests(unittest.TestCase):
    def setUp(self):
        root = Path(tempfile.gettempdir()).resolve()
        self.directory = Path(tempfile.mkdtemp(prefix="geocpu_native_capture_test_", dir=root))
        self.addCleanup(self._clean)

    def _clean(self):
        root = Path(tempfile.gettempdir()).resolve()
        path = self.directory.resolve()
        if path == root or not path.is_relative_to(root) or not path.name.startswith("geocpu_native_capture_test_"):
            raise RuntimeError("Refusing unsafe test cleanup")
        shutil.rmtree(path)

    def test_proxy_preserves_unstripped_output_and_cpu_trace(self):
        case = {"id": "fake_swe_1", "prompt": "問題: 5\n思考: ", "target": "5",
                "source_index": 0, "source_raw_line_sha256": "0" * 64}
        proxy = CaptureProxy(FakeV43Model(case["prompt"], "5"), [case], self.directory / "capture.jsonl")
        try:
            proxy.eval()
            proxy.generate_with_qwenputer(case["prompt"], {}, {}, max_tokens=None,
                                          arm="learned", repetition_penalty=1.0)
        finally:
            proxy.close()
        row = read_jsonl(self.directory / "capture.jsonl")[0]
        self.assertEqual(row["raw_output"], "  ignored prose  ")
        self.assertEqual(row["cpu_trace"][0]["val"], 5)
        self.assertTrue(row["natural_eos"])
        self.assertEqual(row["prompt_sha256"], hashlib.sha256(case["prompt"].encode()).hexdigest())

    def test_original_evaluator_scores_the_captured_result(self):
        source = Path(os.environ.get("GEOCPU_V43_SOURCE_ZIP",
            r"G:\マイドライブ\GeoCPU_Frontier_CoTraining_v43\Grand_cotraining_v43.zip"))
        if not source.is_file():
            self.skipTest("Pinned original v43 source ZIP is unavailable")
        self.assertEqual(sha256_file(source), SOURCE_SHA256)
        with zipfile.ZipFile(source) as archive:
            line = archive.read(SWE_MEMBER).splitlines(keepends=True)[0]
        case = json.loads(line)
        case.update(source_index=0, source_raw_line_sha256=hashlib.sha256(line).hexdigest())
        sys.path.insert(0, str(source))
        from eval_step_control_heldout import run_heldout_evaluation
        manifest = self.directory / "one.jsonl"
        manifest.write_text(json.dumps({key: value for key, value in case.items()
                                        if key not in ("source_index", "source_raw_line_sha256")},
                                       ensure_ascii=False) + "\n", encoding="utf-8")
        proxy = CaptureProxy(FakeV43Model(case["prompt"], case["target"].strip()),
                             [case], self.directory / "capture.jsonl")
        try:
            stats = run_heldout_evaluation(
                model=proxy, vocab={}, inv_vocab={}, manifest_path=str(manifest),
                arm="learned", device="cpu", output_audit_path=str(self.directory / "audit.jsonl"))
        finally:
            proxy.close()
        answer = read_jsonl(self.directory / "capture.jsonl")[0]
        grade = read_jsonl(self.directory / "audit.jsonl")[0]
        self.assertEqual(stats["correct_val"], 1)
        self.assertEqual(stats["strict_with_eos"], 1)
        self.assertTrue(grade["is_correct"])
        self.assertEqual(grade["raw_output"], answer["raw_output"].strip())
        self.assertEqual(grade["final_answer"], case["target"].strip())


if __name__ == "__main__":
    unittest.main()
