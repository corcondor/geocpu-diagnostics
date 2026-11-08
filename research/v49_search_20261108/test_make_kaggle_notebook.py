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
    def test_generated_kernel_is_consistent(self):
        with tempfile.TemporaryDirectory() as td:
            subprocess.run([sys.executable, "-I", str(HERE / "make_kaggle_notebook.py"), td], check=True, capture_output=True)
            meta = json.loads((Path(td) / "kernel-metadata.json").read_text(encoding="utf-8"))
            nb = json.loads((Path(td) / meta["code_file"]).read_text(encoding="utf-8"))
            self.assertTrue(meta["enable_gpu"] and not meta["enable_internet"] and meta["is_private"])
            self.assertEqual(meta["machine_shape"], "NvidiaTeslaT4")
            self.assertEqual(meta["dataset_sources"], ["geocpuresearch/geocpu-v48-eval-assets-20261108"])
            ns = {}
            for cell in nb["cells"]:
                if cell["cell_type"] != "code":
                    continue
                src = "".join(cell["source"])
                ast.parse(src)
                if cell["id"] == "setup":
                    ns["FILES_SRC"] = src
            # run only the assignment of FILES from the setup cell and compare with the files on disk
            tree = ast.parse(ns["FILES_SRC"])
            files = None
            for node in tree.body:
                if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "FILES":
                    files = ast.literal_eval(node.value)
            self.assertEqual(set(files), {"robust_eval.py", "kaggle_runner.py", "kaggle_compare.py"})
            for name, text in files.items():
                self.assertEqual(text, (HERE / name).read_text(encoding="utf-8"), name)

    def test_the_two_jobs_use_the_checkpoints_the_runner_tests_rehearse(self):
        text = (HERE / "make_kaggle_notebook.py").read_text(encoding="utf-8")
        for run, step in (("v47_from28000_to30500_curriculum_v1", 30500), ("v48_A100_from40500_20261108T045428Z", 61500)):
            self.assertIn(f'"{run}", {step}', text)


if __name__ == "__main__":
    unittest.main()
