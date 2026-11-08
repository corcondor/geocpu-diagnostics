"""The generated Kaggle notebook must be valid JSON, every code cell must parse, and the embedded files must equal the sources."""
import ast
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


class Notebook(unittest.TestCase):
    def build(self, td):
        subprocess.run([sys.executable, "-I", str(HERE / "make_kaggle_notebook.py"), td], check=True, capture_output=True)
        meta = json.loads((Path(td) / "kernel-metadata.json").read_text(encoding="utf-8"))
        nb = json.loads((Path(td) / meta["code_file"]).read_text(encoding="utf-8"))
        return meta, nb

    def test_generated_kernel_is_consistent(self):
        with tempfile.TemporaryDirectory() as td:
            meta, nb = self.build(td)
            self.assertTrue(meta["enable_gpu"] and not meta["enable_internet"] and meta["is_private"])
            self.assertEqual(meta["machine_shape"], "NvidiaTeslaT4")
            self.assertEqual(meta["dataset_sources"], ["geocpuresearch/geocpu-v48-eval-assets-20261108",
                                                       "geocpuresearch/geocpu-v48-eval-code-20261108"])
            sources = {}
            for cell in nb["cells"]:
                if cell["cell_type"] != "code":
                    continue
                src = "".join(cell["source"])
                ast.parse(src)
                sources[cell["id"]] = src
            self.assertEqual(list(sources), ["setup", "smoke", "run", "report"])
            # run only the assignment of FILES from the setup cell and compare with the files on disk
            files = None
            for node in ast.parse(sources["setup"]).body:
                if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "FILES":
                    files = ast.literal_eval(node.value)
            self.assertEqual(set(files), {"robust_eval.py", "kaggle_runner.py", "kaggle_compare.py"})
            for name, text in files.items():
                self.assertEqual(text, (HERE / name).read_text(encoding="utf-8"), name)

    def test_stages_guard_the_expensive_part(self):
        with tempfile.TemporaryDirectory() as td:
            _, nb = self.build(td)
            src = {c["id"]: "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"}
            self.assertIn("K.BASE_BUNDLE_SHA256", src["setup"])                       # pins are checked before any child starts
            self.assertIn("K.CODE_ZIP_SHA256", src["setup"])
            self.assertIn('".bin"', src["setup"])                                     # zips come as *.zip.bin (Kaggle unpacks *.zip)
            self.assertIn("--limit", src["smoke"])
            self.assertIn("SMOKE OK", src["smoke"])
            self.assertIn("assert not bad", src["smoke"])                             # a failing smoke test stops the kernel
            self.assertIn("DEADLINE", src["run"])
            self.assertIn("STATUS.txt", src["run"])
            self.assertIn("r.stderr", src["report"])                                  # the compare traceback is never discarded
            self.assertIn("assert not bad", src["report"])                            # a failed child makes the kernel fail visibly

    def test_parent40500_variant_measures_only_the_parent_and_needs_the_right_datasets(self):
        with tempfile.TemporaryDirectory() as td:
            subprocess.run([sys.executable, "-I", str(HERE / "make_kaggle_notebook.py"), td, "parent40500"], check=True, capture_output=True)
            meta = json.loads((Path(td) / "kernel-metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["dataset_sources"], ["geocpuresearch/geocpu-v47-eval-parent40500-20261109",
                                                       "geocpuresearch/geocpu-v48-eval-code-20261108"])
            self.assertEqual(meta["id"], "geocpuresearch/geocpu-v47-parent40500-robust-eval-20261109")
            nb = json.loads((Path(td) / meta["code_file"]).read_text(encoding="utf-8"))
            src = {c["id"]: "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"}
            for s in src.values():
                ast.parse(s)
            self.assertIn("v47_40500_checkpoint_step_40500.pt", src["setup"])
            self.assertIn("[('v47_40500', 'practical_20261107_10k_multistep_40500', 40500)]", src["setup"])
            self.assertNotIn("kaggle_compare.py\"), \"--out\"", src["report"])            # nothing to compare inside this kernel
            jobs_line = [l for l in src["setup"].splitlines() if l.startswith("JOBS = ")]
            self.assertEqual(len(jobs_line), 1)
            self.assertNotIn("v48_61500", jobs_line[0])                                       # (the embedded compare script mentions it)

    def test_the_two_jobs_use_the_checkpoints_the_runner_tests_rehearse(self):
        text = (HERE / "make_kaggle_notebook.py").read_text(encoding="utf-8")
        for run, step in (("v47_from28000_to30500_curriculum_v1", 30500), ("v48_A100_from40500_20261108T045428Z", 61500)):
            self.assertIn(f'"{run}", {step}', text)


if __name__ == "__main__":
    unittest.main()
