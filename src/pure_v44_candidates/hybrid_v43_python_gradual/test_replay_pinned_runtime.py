"""Small real-source preflight for the independent replay, without training."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
import zipfile

from replay_python_full import _load_module, verify_runtime_sources


ASSETS = Path(r"C:\Users\81808\.openclaw\reports\jmmlu_v1\ag_fused_fullqa_training_v1")
V43_ZIP = ASSETS / "v43_v44fix_paired_eval_assets" / "v43_source.zip"
PURE_SOURCE = ASSETS / "v44_hotfix_source_readonly_20261103"


class PinnedRuntimeTest(unittest.TestCase):
    @unittest.skipUnless(V43_ZIP.is_file() and PURE_SOURCE.is_dir(),
                         "Pinned original source files unavailable")
    def test_actual_manifest_cpu_and_python_execution(self):
        with tempfile.TemporaryDirectory(prefix="geocpu_replay_preflight_") as temporary:
            root = Path(temporary)
            v43_code = root / "v43_code"
            pure_code = root / "pure_code"
            v43_code.mkdir()
            pure_code.mkdir()
            manifest = json.loads((PURE_SOURCE / "hotfix_manifest.json").read_text(encoding="utf-8"))
            for name in manifest["files"]:
                target = pure_code / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(PURE_SOURCE / name, target)
            shutil.copyfile(PURE_SOURCE / "hotfix_manifest.json", pure_code / "hotfix_manifest.json")
            with zipfile.ZipFile(V43_ZIP) as archive:
                for name in ("qwenputer_controller_v30.py", "cpu_circuit_v30.py"):
                    (v43_code / name).write_bytes(archive.read(name))
            verify_runtime_sources(v43_zip=V43_ZIP, v43_code_dir=v43_code,
                                   python_code_dir=pure_code)
            sys.dont_write_bytecode = True
            sys.path.insert(0, str(v43_code))
            sys.path.insert(0, str(pure_code))
            try:
                cpu_module = _load_module(v43_code / "cpu_circuit_v30.py", "cpu_circuit_v30")
                controller_module = _load_module(v43_code / "qwenputer_controller_v30.py",
                                                 "qwenputer_controller_v30")
                tir_module = _load_module(pure_code / "tir_engine_v44r1.py", "tir_engine_v44r1")
                cpu = controller_module.QwenputerExecutionController(
                    cpu_module.ExactRAM8x32(), cpu_module.CompiledOperators32())
                state = cpu.initial_cpu_state()
                command = cpu.parse_command("<ADD R0 2 3>")
                self.assertIsNotNone(command)
                _, value, observed = cpu.execute_step(state, command)
                self.assertEqual((value, observed), (5, "[R0=5]"))
                with tir_module.PythonExecutor() as executor:
                    result = executor.execute("print(2 + 3)")
                    self.assertTrue(result["success"], result)
                    self.assertEqual(result["stdout"], "5\n")
            finally:
                sys.path.remove(str(pure_code))
                sys.path.remove(str(v43_code))


if __name__ == "__main__":
    unittest.main()
