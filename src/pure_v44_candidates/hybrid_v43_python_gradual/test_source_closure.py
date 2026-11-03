"""Actual pinned-source extraction and resume integrity gate, with no model load."""
from pathlib import Path
import tempfile
import unittest

from train_hybrid_stage import _prepare_source, _verify_extracted_source
from eval_fixed_diagnostic import _prepare_eval_source, _verify_eval_source
from replay_python_full import _verify_importable_tree
from source_pools import file_sha


ROOT = Path(__file__).resolve().parents[3]
ZIP = Path(r"C:\Users\81808\.openclaw\reports\jmmlu_v1\ag_fused_fullqa_training_v1\v43_v44fix_paired_eval_assets\v43_source.zip")
OVERLAY = ROOT / "src" / "v44fix"


class SourceClosureTests(unittest.TestCase):
    @unittest.skipUnless(ZIP.is_file(), "Pinned original v43 ZIP unavailable")
    def test_actual_extracted_source_and_changed_overlay_rejection(self):
        with tempfile.TemporaryDirectory(prefix="hybrid_source_closure_", dir=ROOT) as temporary:
            scratch = Path(temporary).resolve()
            if not scratch.is_relative_to(ROOT):
                raise ValueError("Temporary source test escaped the intended workspace")
            extracted = scratch / "source"
            _prepare_source(ZIP, OVERLAY, extracted)
            _verify_extracted_source(ZIP, OVERLAY, extracted)
            injected = extracted / "torch.py"
            injected.write_text("raise RuntimeError('import shadow')\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                _verify_extracted_source(ZIP, OVERLAY, extracted)
            injected.unlink()
            changed = extracted / "tir_engine_v44r1.py"
            changed.write_bytes(changed.read_bytes() + b"\n# mutation fixture\n")
            with self.assertRaises(ValueError):
                _verify_extracted_source(ZIP, OVERLAY, extracted)

    @unittest.skipUnless(ZIP.is_file(), "Pinned original v43 ZIP unavailable")
    def test_explicit_fp16_eval_copy_and_changed_source_rejection(self):
        with tempfile.TemporaryDirectory(prefix="hybrid_fp16_source_", dir=ROOT) as temporary:
            scratch = Path(temporary).resolve()
            if not scratch.is_relative_to(ROOT):
                raise ValueError("Temporary source test escaped the intended workspace")
            extracted = scratch / "source"
            _prepare_eval_source(ZIP, OVERLAY, extracted, "eval_fp16_native_v1")
            _verify_eval_source(ZIP, OVERLAY, extracted, "eval_fp16_native_v1")
            injected = extracted / "torch.py"
            injected.write_text("raise RuntimeError('import shadow')\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                _verify_eval_source(ZIP, OVERLAY, extracted, "eval_fp16_native_v1")
            injected.unlink()
            changed = extracted / "integrated_model_v42.py"
            changed.write_bytes(changed.read_bytes() + b"\n# mutation fixture\n")
            with self.assertRaises(ValueError):
                _verify_eval_source(ZIP, OVERLAY, extracted, "eval_fp16_native_v1")

    def test_replay_rejects_unpinned_importable_module(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            known = root / "known.py"
            known.write_text("value = 1\n", encoding="utf-8")
            _verify_importable_tree(root, {"known.py": file_sha(known)})
            injected = root / "torch.py"
            injected.write_text("raise RuntimeError('import shadow')\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                _verify_importable_tree(root, {"known.py": file_sha(known)})


if __name__ == "__main__":
    unittest.main()
