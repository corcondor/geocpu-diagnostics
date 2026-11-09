"""The generated Colab notebook must be valid nbformat-4 JSON, every code cell must parse, the embedded files must equal the sources,
no GPU may be requested, no secret may be present or read, and nothing may open a network connection other than the Hub client.

    python -I -X utf8 test_make_notebook.py        # builds the notebook into a temp folder; nothing else is read
"""
import ast
import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODULES = ("acquire.py", "checks.py", "convert.py", "pipeline.py")
URL = "https://colab.research.google.com/github/corcondor/geocpu-diagnostics/blob/research/v49-acquire-20261109/research/v49_acquire_20261109/colab_acquire_check.ipynb"
EXPECTED_CELLS = ["intro", "pip", "config", "modules", "md-dry", "dryrun", "md-acquire", "acquire", "md-raw", "mirror", "vocab", "raw", "md-convert",
                  "smoke", "full", "md-filters", "filters", "md-final", "final", "md-finish", "finish"]
STEP_CELLS = {"dryrun": "step_dry_run", "acquire": "step_acquire", "mirror": "step_mirror", "vocab": "step_vocab_eval", "raw": "step_raw_checks",
              "smoke": "step_convert_smoke", "full": "step_convert_full", "filters": "step_filters", "final": "step_finalize"}
NETWORK_MODULES = {"urllib", "urllib3", "requests", "http", "socket", "ftplib", "smtplib", "telnetlib", "aiohttp", "httpx", "ssl", "websocket", "imaplib", "poplib"}
SECRET_PATTERNS = [r"hf_[A-Za-z0-9]{30,}", r"ghp_[A-Za-z0-9]{30,}", r"gho_[A-Za-z0-9]{30,}", r"github_pat_[A-Za-z0-9_]{30,}", r"sk-[A-Za-z0-9]{20,}",
                   r"AKIA[0-9A-Z]{16}", r"-----BEGIN [A-Z ]*PRIVATE KEY-----", r"xox[abp]-[A-Za-z0-9-]{10,}", r"AIza[0-9A-Za-z_-]{30,}"]
SECRET_READS = ["userdata.get", "notebook_login", "getpass", "HF_TOKEN", "GITHUB_TOKEN", "huggingface-cli login", "Authorization:", "Bearer "]


def build(td):
    subprocess.run([sys.executable, "-I", "-X", "utf8", str(HERE / "make_notebook.py"), td], check=True, capture_output=True)
    raw = (Path(td) / "colab_acquire_check.ipynb").read_bytes()
    return raw, json.loads(raw.decode("utf-8"))


def code_sources(nb):
    return {c["id"]: "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"}


def literal(src, name):
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == name:
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not assigned")


class Notebook(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.td = tempfile.TemporaryDirectory()
        cls.raw, cls.nb = build(cls.td.name)
        cls.src = code_sources(cls.nb)

    @classmethod
    def tearDownClass(cls):
        cls.td.cleanup()

    def test_valid_nbformat4_structure(self):
        nb = self.nb
        self.assertEqual(nb["nbformat"], 4)
        self.assertGreaterEqual(nb["nbformat_minor"], 5)                                  # cell ids need minor 5
        self.assertEqual([c["id"] for c in nb["cells"]], EXPECTED_CELLS)
        ids = [c["id"] for c in nb["cells"]]
        self.assertEqual(len(ids), len(set(ids)))
        for c in nb["cells"]:
            self.assertIn(c["cell_type"], ("code", "markdown"))
            self.assertIsInstance(c["metadata"], dict)
            self.assertTrue(isinstance(c["source"], list) and all(isinstance(x, str) for x in c["source"]))
            if c["cell_type"] == "code":
                self.assertIsNone(c["execution_count"])
                self.assertEqual(c["outputs"], [])
        self.assertLess(len(self.raw), 1_000_000, "the notebook must stay below 1 MB")
        self.assertEqual(self.raw.decode("utf-8"), self.raw.decode("utf-8").replace("\r", ""))   # no CR in the file

    def test_notebook_build_is_deterministic(self):
        with tempfile.TemporaryDirectory() as td2:
            raw2, _ = build(td2)
        self.assertEqual(self.raw, raw2)

    def test_the_notebook_file_next_to_the_sources_is_up_to_date(self):
        """Colab opens THAT file from GitHub: it must be exactly what make_notebook.py builds from the current sources."""
        committed = HERE / "colab_acquire_check.ipynb"
        self.assertTrue(committed.is_file(), "run: python make_notebook.py")
        on_disk = committed.read_bytes().replace(bytes([13, 10]), bytes([10]))                      # a Windows checkout may have CRLF (autocrlf)
        self.assertEqual(on_disk, self.raw, "colab_acquire_check.ipynb is stale: run python make_notebook.py")

    def test_every_code_cell_parses(self):
        self.assertEqual(len(self.src), 13)
        for cid, s in self.src.items():
            ast.parse(s, filename=cid)

    def test_embedded_files_equal_the_sources(self):
        files = literal(self.src["modules"], "FILES")
        self.assertEqual(list(files), list(MODULES))
        for name, text in files.items():
            self.assertEqual(text, (HERE / name).read_text(encoding="utf-8"), name)
            ast.parse(text, filename=name)
        self.assertTrue(all("\r" not in t for t in files.values()))

    def test_embedded_v48_unseen_eval_equals_the_repo_files(self):
        extra = literal(self.src["modules"], "EVAL_EXTRA")
        data = HERE.parent / "v48_balanced_augmentation_20261108" / "data"
        self.assertEqual(sorted(extra), [f"{f}_unseen_eval.jsonl" for f in ("evidence", "japanese", "logic", "python")])
        for name, text in extra.items():
            rows = [json.loads(l) for l in text.splitlines()]
            full = [json.loads(l) for l in (data / name).read_text(encoding="utf-8").splitlines() if l.strip()]
            self.assertEqual(len(rows), len(full), name)
            self.assertEqual(len(rows), 128)
            for a, b in zip(rows, full):
                self.assertEqual(a["id"], b["id"])
                self.assertEqual(a["prompt"], b["prompt"])
                self.assertEqual(a["target"], b["target"])
                self.assertEqual(a.get("options"), b.get("options"))
                self.assertEqual(a.get("python_code"), b.get("python_code"))
                self.assertLessEqual(set(a), {"id", "prompt", "target", "options", "python_code"})

    def test_the_modules_cell_is_a_collapsed_form_cell(self):
        cell = next(c for c in self.nb["cells"] if c["id"] == "modules")
        self.assertEqual(cell["metadata"].get("cellView"), "form")
        self.assertTrue(self.src["modules"].startswith("#@title 3b"))

    def test_no_gpu_is_requested_or_used(self):
        meta = json.dumps(self.nb["metadata"]).lower()
        for bad in ("accelerator", "gputype", "gpu", "tpu"):
            self.assertNotIn(bad, meta)
        text = json.dumps(self.nb).lower()
        for bad in ("nvidia-smi", "cuda", "import torch", "tensorflow", "\"enable_gpu\""):
            self.assertNotIn(bad, text)
        self.assertIn("GPU は使いません", "".join(self.nb["cells"][0]["source"]))

    def test_no_secret_is_present_or_read(self):
        text = json.dumps(self.nb, ensure_ascii=False)
        for pat in SECRET_PATTERNS:
            self.assertIsNone(re.search(pat, text), pat)
        for s in SECRET_READS:
            self.assertNotIn(s, text, s)
        files = literal(self.src["modules"], "FILES")
        self.assertIn("token=False", files["acquire.py"])                                  # every Hub call opts out of any ambient token

    def test_no_network_library_is_imported_anywhere(self):
        files = literal(self.src["modules"], "FILES")
        parsed = {name: ast.parse(text) for name, text in files.items()}
        parsed.update({cid: ast.parse(s) for cid, s in self.src.items() if cid != "modules"})
        for name, tree in parsed.items():
            for node in ast.walk(tree):
                mods = []
                if isinstance(node, ast.Import):
                    mods = [a.name.split(".")[0] for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    mods = [node.module.split(".")[0]]
                self.assertFalse(set(mods) & NETWORK_MODULES, f"{name} imports {mods}")
        allcode = "\n".join(s for cid, s in self.src.items() if cid != "modules")
        for bad in ("wget", "curl ", "git clone", "requests.", "urlopen", "!pip", "!apt"):
            self.assertNotIn(bad, allcode)
        # the only installer call is pip with the two libraries of the task
        self.assertIn('"pip", "install", "-q", "huggingface_hub", "pyarrow"', self.src["pip"])

    def test_drive_is_touched_only_by_the_mount_the_root_and_the_final_flush(self):
        for cid, s in self.src.items():
            if cid in ("config", "modules", "finish"):
                continue
            self.assertNotIn("/content/drive", s, cid)
            self.assertNotIn("open(", s, cid)
        self.assertIn("drive.mount(\"/content/drive\")", self.src["config"])
        self.assertIn("GeoCPU_v47_fullmodel_20261106", self.src["config"])
        self.assertIn("drive.flush_and_unmount()", self.src["finish"])

    def test_each_cell_prints_a_pass_or_fail_and_stops_on_fail(self):
        sys.path.insert(0, str(HERE))
        import pipeline as P
        for cid, fn in STEP_CELLS.items():
            s = self.src[cid]
            self.assertIn(f"P.{fn}(CFG)", s, cid)
            self.assertRegex(s, r'assert R\w+\["pass"\]', cid)
            f = getattr(P, fn)
            self.assertTrue(hasattr(f, "step_name"), f"{fn} is not a @step (it would not print PASS/FAIL)")
        for cid in ("pip", "config", "finish"):
            self.assertIn("PASS [", self.src[cid], cid)
            self.assertIn("assert", self.src[cid], cid)
        self.assertIn("assert R3[\"pass\"]", self.src["modules"])
        # the step functions print "PASS [name]" / "FAIL [name]" (checked by running them in test_pipeline.py)
        text = (HERE / "pipeline.py").read_text(encoding="utf-8")
        self.assertIn('cfg.log(f"PASS [{name}]', text)
        self.assertIn('cfg.log(f"FAIL [{name}]', text)

    def test_the_run_order_matches_the_task(self):
        order = [c["id"] for c in self.nb["cells"] if c["cell_type"] == "code"]
        self.assertEqual(order, ["pip", "config", "modules", "dryrun", "acquire", "mirror", "vocab", "raw", "smoke", "full", "filters", "final", "finish"])
        # smoke (2,000 records) comes before the full conversion; the raw checks come before both
        self.assertLess(order.index("raw"), order.index("smoke"))
        self.assertLess(order.index("smoke"), order.index("full"))
        self.assertLess(order.index("full"), order.index("filters"))
        self.assertIn("2,000", "".join(c["source"][0] for c in self.nb["cells"] if c["id"] == "smoke"))

    def test_intro_says_what_runs_and_what_is_written(self):
        intro = "".join(self.nb["cells"][0]["source"])
        for must in ("GPU は使いません", "新しいフォルダにだけ", "上書きも削除もしません", "PASS", "FAIL", "MISMATCH", "REPORT_JA.md", "423,348", "RUN_SUFFIX"):
            self.assertIn(must, intro)


class Readme(unittest.TestCase):
    def test_readme_covers_the_required_points(self):
        p = HERE / "README_JA.md"
        self.assertTrue(p.is_file())
        t = p.read_text(encoding="utf-8")
        for must in (URL, "REPORT_JA.md", "registry.json", "external_data_v49_rows", "ATTRIBUTION.md", "CC BY-SA", "Apache", "ShareAlike", "RUN_SUFFIX",
                     "GPU", "MISMATCH", "目視"):
            self.assertIn(must, t, must)


if __name__ == "__main__":
    unittest.main(verbosity=2)
