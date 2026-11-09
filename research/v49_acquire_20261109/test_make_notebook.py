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
URL_ENCODED = URL.replace("blob/research/v49-acquire-20261109", "blob/research%2Fv49-acquire-20261109")          # Colab's github/ route may need the branch slash encoded
EXPECTED_CELLS = ["intro", "pip", "config", "modules", "md-probe", "probe", "sandbox", "md-dry", "dryrun", "md-acquire", "acquire", "md-raw", "mirror", "vocab",
                  "raw", "md-convert", "smoke", "full", "md-filters", "filters", "reviews", "md-final", "final", "md-finish", "finish"]
STEP_CELLS = {"probe": "step_drive_probe", "dryrun": "step_dry_run", "acquire": "step_acquire", "mirror": "step_mirror", "vocab": "step_vocab_eval", "raw": "step_raw_checks",
              "sandbox": "step_sandbox_selftest", "smoke": "step_convert_smoke", "full": "step_convert_full", "filters": "step_filters",
              "reviews": "step_reviews", "final": "step_finalize"}
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
        self.assertLess(len(self.raw), 1_100_000, "the notebook must stay about 1 MB (it embeds four modules)")
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
        self.assertEqual(len(self.src), 16)
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
        # a runtime that HAS a GPU (e.g. left over from an A100 session) stops the run instead of burning paid compute units
        self.assertIn("/proc/driver/nvidia", self.src["config"])
        self.assertRegex(self.src["config"], r"assert not \(os\.path\.exists\(\"/proc/driver/nvidia\"\)")
        self.assertIn("FAIL [3a cpu]", self.src["config"])

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
        # the only installer call is pip with the two libraries of the task (huggingface_hub pinned from below; the versions are printed)
        self.assertIn('"pip", "install", "-q", "huggingface_hub>=0.24", "pyarrow"', self.src["pip"])
        self.assertIn("md.version", self.src["pip"])
        self.assertIn("セッションを再起動", self.src["pip"])                                 # an already imported older copy is detected

    def test_drive_is_touched_only_by_the_mount_the_root_and_the_final_flush(self):
        for cid, s in self.src.items():
            if cid in ("config", "modules", "finish"):
                continue
            self.assertNotIn("/content/drive", s, cid)
            self.assertNotIn("open(", s, cid)
        self.assertIn("drive.mount(\"/content/drive\")", self.src["config"])
        self.assertIn("GeoCPU_v47_fullmodel_20261106", self.src["config"])
        self.assertIn("drive.flush_and_unmount()", self.src["finish"])
        # the hooks that unmount / flush Drive live in the modules cell and are switched by two settings of the config cell
        self.assertIn("UNMOUNT_DRIVE_DURING_CODE_RUNS = True", self.src["config"])
        self.assertIn("FLUSH_DRIVE_AFTER_STEPS = True", self.src["config"])
        for needle in ("def _detach", "def _attach", "def _sync", "drive_detach=_detach", "drive_attach=_attach", "drive_sync=_sync"):
            self.assertIn(needle, self.src["modules"].split("R3 = ")[0].split("CODE = Path")[1], needle)

    def test_the_modules_cell_loads_this_text_not_an_older_copy(self):
        m = self.src["modules"]
        tail = m.split("sys.path.insert(0, str(CODE))")[1]
        self.assertIn("sys.modules.pop(_m, None)", tail)
        self.assertLess(tail.index("sys.modules.pop"), tail.index("import acquire"))
        self.assertIn("hashlib.sha256(_file.read_bytes())", tail)                            # the loaded file equals the embedded text

    def test_config_exposes_the_new_settings(self):
        for name in ("RUN_SUFFIX", "RAW_SUFFIX", "STOP_ON_COUNT_MISMATCH", "ALLOW_VOCAB_MISMATCH", "ONLY", "RUN_SANDBOX", "SHARD_RECORDS", "SHARD_KEYS", "MAX_UNSHARDED_MINUTES"):
            self.assertRegex(self.src["config"], rf"(?m)^{name} = ", name)
        for kw in ("raw_suffix=RAW_SUFFIX", "allow_vocab_mismatch=ALLOW_VOCAB_MISMATCH", "suffix=RUN_SUFFIX", "run_sandbox=RUN_SANDBOX", "shard_records=SHARD_RECORDS",
                   "shard_keys=tuple(SHARD_KEYS)", "max_unsharded_minutes=MAX_UNSHARDED_MINUTES"):
            self.assertIn(kw, self.src["modules"], kw)
        # the messages of the pipeline name settings the owner can really reach from the notebook
        pipe = (HERE / "pipeline.py").read_text(encoding="utf-8")
        for name in ("SHARD_KEYS", "MAX_UNSHARDED_MINUTES"):
            self.assertIn(name, pipe)
            self.assertRegex(self.src["config"], rf"(?m)^{name} = ")
        self.assertNotIn("Cfg.shard_keys", pipe)

    def test_every_setting_passed_to_the_cfg_is_a_field_of_it(self):
        sys.path.insert(0, str(HERE))
        import pipeline as P
        tree = ast.parse(self.src["modules"])
        call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "make_cfg")
        fields = {f.name for f in P.dataclasses.fields(P.Cfg)}
        self.assertTrue(call.keywords)
        for kw in call.keywords:
            self.assertIn(kw.arg, fields, kw.arg)
        # and the names the keywords take are defined by the config cell (or are the hooks of the modules cell)
        defined = {t.id for n in ast.parse(self.src["config"]).body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
        defined |= {"_detach", "_attach", "_sync", "EXTRA", "DRIVE_ROOT"}
        for kw in call.keywords:
            for node in ast.walk(kw.value):
                if isinstance(node, ast.Name) and node.id not in ("str", "tuple", "P"):
                    self.assertIn(node.id, defined | {"UNMOUNT_DRIVE_DURING_CODE_RUNS", "FLUSH_DRIVE_AFTER_STEPS"}, f"{kw.arg}: {node.id}")

    def test_cell_10_prints_the_whole_report_not_its_first_8000_characters(self):
        # review round 3: sections 5-7 (limits of the checks, licence duties, output locations) were cut off
        self.assertNotIn("[:8000]", self.src["finish"])
        self.assertIn('print(rep.read_text(encoding="utf-8"))', self.src["finish"])

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
        self.assertEqual(order, ["pip", "config", "modules", "probe", "sandbox", "dryrun", "acquire", "mirror", "vocab", "raw", "smoke", "full", "filters", "reviews",
                                 "final", "finish"])
        # review round 3: the Drive probe and the sandbox self-test come BEFORE the 30-80 minutes of download and raw checks (a Linux failure shows up at once);
        # smoke (2,000 records) comes before the full conversion
        self.assertLess(order.index("modules"), order.index("probe"))
        self.assertLess(order.index("probe"), order.index("sandbox"))
        self.assertLess(order.index("sandbox"), order.index("acquire"))
        self.assertLess(order.index("raw"), order.index("smoke"))
        self.assertLess(order.index("smoke"), order.index("full"))
        self.assertLess(order.index("full"), order.index("filters"))
        self.assertLess(order.index("filters"), order.index("reviews"))                      # the optional reviews never come before the drop lists exist
        self.assertLess(order.index("reviews"), order.index("final"))
        self.assertIn("2,000", "".join(c["source"][0] for c in self.nb["cells"] if c["id"] == "smoke"))

    def test_intro_says_what_runs_and_what_is_written(self):
        intro = "".join(self.nb["cells"][0]["source"])
        for must in ("GPU は使いません", "新しいフォルダにだけ", "上書きも削除もしません", "PASS", "FAIL", "MISMATCH", "REPORT_JA.md", "423,348", "RUN_SUFFIX",
                     "RAW_SUFFIX", "nobody", "アンマウント", "_HOLD_do_not_mix", "タブを開いたまま", "シャード", "保存出力", "弱い代わりの方法には切り替えません",
                     "あなたが押すもの", "このまま実行", "許可", "すべてのセルを実行", "セッションを再起動", "3c", "3d", "約20回", "再接続"):
            self.assertIn(must, intro)


class Readme(unittest.TestCase):
    def test_readme_covers_the_required_points(self):
        p = HERE / "README_JA.md"
        self.assertTrue(p.is_file())
        t = p.read_text(encoding="utf-8")
        self.assertLess(t.index(URL_ENCODED), t.index(URL))                                   # the encoded form is the one that most probably works: listed first
        self.assertIn("<コミットのsha>", t)                                                    # the commit form needs the sha of the pushed commit: never a stale one
        self.assertNotIn("b60f1e38e13f092506ec8d6bbc68582d18950b0f/research", t)
        for must in (URL, URL_ENCODED, "File > Upload", "REPORT_JA.md", "registry.json", "external_data_v49_rows", "ATTRIBUTION.md", "CC BY-SA", "Apache",
                     "ShareAlike", "RUN_SUFFIX", "RAW_SUFFIX", "GPU", "MISMATCH", "目視", "_HOLD_do_not_mix", "external_data_v49_stageA", "nobody", "LICENSE_NOTICE.txt"):
            self.assertIn(must, t, must)
        for wrong in ("分割は文章の単位（グループ）で、heldout 3%・dev 1%・train 96% です。",):                      # the old, misleading claim about every dataset
            self.assertNotIn(wrong, t, wrong)

    def test_the_owner_clicks_list_is_there_and_names_every_click(self):
        t = (HERE / "README_JA.md").read_text(encoding="utf-8")
        self.assertIn("## あなたが押すもの", t)
        section = t.split("## あなたが押すもの")[1].split("\n## ")[0]
        for must in ("このまま実行", "ハードウェア アクセラレータ", "すべてのセルを実行", "アクセス許可", "3c", "セッションを再起動", "File > Upload", "RUN_SUFFIX",
                     "タブを開いたまま", "FAIL", "再接続"):
            self.assertIn(must, section, must)
        # what it does NOT ask for: no password, key or token
        self.assertIn("パスワード・API キー・トークンも要りません", t)

    def test_the_readme_states_what_was_never_run_on_linux(self):
        t = (HERE / "README_JA.md").read_text(encoding="utf-8")
        for must in ("まだ実行していません", "3d", "match"):
            self.assertIn(must, t, must)


if __name__ == "__main__":
    unittest.main(verbosity=2)
