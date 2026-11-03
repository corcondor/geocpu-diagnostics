"""Actual pinned-source extraction and resume integrity gate, with no model load."""
from pathlib import Path
import tempfile
import unittest

from train_hybrid_stage import _prepare_source, _verify_extracted_source


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
            changed = extracted / "tir_engine_v44r1.py"
            changed.write_bytes(changed.read_bytes() + b"\n# mutation fixture\n")
            with self.assertRaises(ValueError):
                _verify_extracted_source(ZIP, OVERLAY, extracted)


if __name__ == "__main__":
    unittest.main()
