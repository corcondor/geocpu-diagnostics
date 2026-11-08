"""CPU rehearsal of everything in kaggle_runner.py that does not need the 1 GB weights or a CUDA device.

Uses the REAL small files from the Drive study folder (receipts, training_saved, input_manifest, base bundle, code zip) when the
drive is present, so a wrong path or layout assumption fails here and not on the Kaggle GPU.
"""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import kaggle_runner as K  # noqa: E402
import robust_eval as R  # noqa: E402

STUDY = Path("G:/マイドライブ/GeoCPU_v47_fullmodel_20261106")
REAL = (STUDY / "GeoCPU_v47_fullmodel_bundle_v1_READY.zip").exists()
RUNS = {"v48_61500": ("v48_A100_from40500_20261108T045428Z", 61500),
        "v47_30500": ("v47_from28000_to30500_curriculum_v1", 30500)}


def make_input(td: Path) -> Path:
    """A flat 'Kaggle dataset' with real small files and a 1 KB stand-in for each checkpoint."""
    inp = td / "input"
    inp.mkdir()
    # the two pinned ZIPs live in their own dataset as *.zip.bin: Kaggle unpacks every *.zip it is given
    (td / "zips").mkdir()
    shutil.copyfile(STUDY / "GeoCPU_v47_fullmodel_bundle_v1_READY.zip", td / "zips" / (K.BUNDLE + ".bin"))
    shutil.copyfile(STUDY / "GeoCPU_practical_training_logging_20261107" / K.CODE_ZIP, td / "zips" / (K.CODE_ZIP + ".bin"))
    for prefix, (run, step) in RUNS.items():
        shutil.copyfile(STUDY / run / "checkpoints" / f"receipt_step_{step}.json", inp / f"{prefix}_receipt_step_{step}.json")
        shutil.copyfile(STUDY / run / "training_saved.json", inp / f"{prefix}_training_saved.json")
        shutil.copyfile(STUDY / run / "input_manifest.json", inp / f"{prefix}_input_manifest.json")
        (inp / f"{prefix}_checkpoint_step_{step}.pt").write_bytes(b"x" * 1024)
    return inp


@unittest.skipUnless(REAL, "Drive study folder not mounted")
class Layout(unittest.TestCase):
    def test_study_layout_passes_the_original_validator_for_both_checkpoints(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            inp = make_input(td)
            work = K.extract_code(td / "zips" / (K.CODE_ZIP + ".bin"), td / "code")
            sys.path.insert(0, str(work))
            import eval_candidate as E
            for prefix, (run, step) in RUNS.items():
                study = K.build_study(inp, td / f"study_{prefix}", run, step, prefix, zips=[td / "zips"])
                ckpt, receipt = E.validate_checkpoint(study, run, step)
                self.assertTrue(ckpt.is_file())
                self.assertRegex(receipt["checkpoint_sha256"], r"^[0-9a-f]{64}$")

    def test_code_zip_sha_is_enforced(self):
        with tempfile.TemporaryDirectory() as td:
            bad = Path(td) / "x.zip"
            bad.write_bytes(b"PK\x05\x06" + b"\0" * 18)
            with self.assertRaisesRegex(RuntimeError, "pinned"):
                K.extract_code(bad, Path(td) / "d")

    def test_prepared_source_has_every_data_file_the_runner_reads(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            inp = make_input(td)
            work = K.extract_code(td / "zips" / (K.CODE_ZIP + ".bin"), td / "code")
            sys.path.insert(0, str(work))
            import eval_candidate as E
            study = K.build_study(inp, td / "study", *RUNS["v48_61500"][:2], "v48_61500", zips=[td / "zips"])
            (td / "src").mkdir()
            source = E.prepare_source(study, td / "src")
            data = source / "data"
            for name in ("benchmark_8100_manifest.jsonl", "eval_v41_logical_instruction_operators_heldout.jsonl",
                         "eval_v42_search_retrieval_aligned.jsonl", "knowledge_corpus_v42.jsonl"):
                self.assertTrue((data / name).is_file(), name)
            self.assertTrue((source / "qwenputer_controller_v30.py").is_file())
            self.assertTrue((source / "local_search_engine_v42.py").is_file())
            # the exact data the robust evaluations will use
            t6 = K.dedupe_by_prompt(R.read_jsonl(data / "eval_v42_search_retrieval_aligned.jsonl"))
            self.assertEqual(len(t6), 21)
            self.assertEqual(sum(i["weight"] for i in t6), 200)
            t5 = R.read_jsonl(data / "eval_v41_logical_instruction_operators_heldout.jsonl")
            self.assertEqual(len(t5), 400)
            mc = [r for r in R.read_jsonl(data / "benchmark_8100_manifest.jsonl") if r.get("task_type") == "multiple_choice"]
            self.assertEqual(len(mc), 7536)
            self.assertGreaterEqual(sum(R.parse_mcq(r["prompt"]) is not None for r in mc), 7500)
            # v42 corpus really contains every T6 answer (the point of re-running T6 against it)
            corpus = "\n".join(d.get("title", "") + " " + d.get("text", "") for d in R.read_jsonl(data / "knowledge_corpus_v42.jsonl"))
            self.assertEqual(sum(1 for i in t6 if i["target"].strip() in corpus), 21)


class Scorer(unittest.TestCase):
    def test_scorer_reads_the_last_real_token_not_the_padding(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        if not REAL:
            self.skipTest("needs the model source for canonical_normalize_text")
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            inp = make_input(td)
            work = K.extract_code(td / "zips" / (K.CODE_ZIP + ".bin"), td / "code")
            sys.path.insert(0, str(work))
            import eval_candidate as E
            study = K.build_study(inp, td / "study", *RUNS["v48_61500"][:2], "v48_61500", zips=[td / "zips"])
            (td / "src").mkdir()
            source = E.prepare_source(study, td / "src")
            sys.path.insert(0, str(source))
            meta = json.loads((source / "base_metadata.json").read_text(encoding="utf-8"))
            vocab = meta["vocab"]

            class Out:
                pass

            class Fake(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.p = torch.nn.Parameter(torch.zeros(1))

                def forward(self, x, valid, arm="learned"):
                    B, L = x.shape
                    o = Out()
                    pos = torch.arange(L, dtype=torch.float32).view(1, L, 1).expand(B, L, 4000)
                    o.logits = pos + torch.arange(4000, dtype=torch.float32).view(1, 1, 4000) * 0.001
                    return o

            scorer = K.make_mcq_scorer(Fake(), vocab)
            prompts = ["分野: a\n問題: x\nA: 1\nB: 2\nC: 3\nD: 4\n正解:", "短い\nA: 1\nB: 2\nC: 3\nD: 4\n正解:"]
            out = scorer(prompts)
            from qwenputer_controller_v30 import canonical_normalize_text
            for p, vec in zip(prompts, out):
                n = len([c for c in canonical_normalize_text(p) if c in vocab])
                for j, letter in enumerate("ABCD"):
                    self.assertAlmostEqual(vec[j], (n - 1) + vocab[letter] * 0.001, places=2)


class Inputs(unittest.TestCase):
    def test_link_refuses_a_missing_source_instead_of_creating_a_dangling_symlink(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError):
                K.link(Path(td) / "nope.zip", Path(td) / "out" / "nope.zip")
            self.assertFalse((Path(td) / "out" / "nope.zip").is_symlink())

    def test_locate_finds_the_bin_name_and_never_a_directory(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            (td / "a" / "GeoCPU_v47_fullmodel_bundle_v1_READY.zip").mkdir(parents=True)             # Kaggle's unpacked directory
            (td / "b").mkdir()
            (td / "b" / "x.zip.bin").write_bytes(b"1")
            self.assertEqual(K.locate([td / "a", td / "b"], "x.zip"), td / "b" / "x.zip.bin")
            with self.assertRaises(FileNotFoundError):
                K.locate([td / "a"], "GeoCPU_v47_fullmodel_bundle_v1_READY.zip")                  # only a directory is there

    def test_an_unpacked_dataset_fails_loudly_at_setup(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            inp = td / "input"
            inp.mkdir()
            (inp / K.BUNDLE.replace(".zip", "")).mkdir()                                            # what Kaggle did to the zips
            for name in ("v47_30500_checkpoint_step_30500.pt",):
                (inp / name).write_bytes(b"x")
            with self.assertRaises(FileNotFoundError):
                K.build_study(inp, td / "study", "run", 30500, "v47_30500")

    def test_pins_are_the_ones_the_evaluation_code_uses(self):
        self.assertEqual(K.BASE_BUNDLE_SHA256, "a25d312429d5b2ee43ec4b812e7a1fb948d0625fbfdfbe9cdf7fac54c075fb70")
        if REAL:
            self.assertEqual(K.file_sha256(STUDY / "GeoCPU_v47_fullmodel_bundle_v1_READY.zip"), K.BASE_BUNDLE_SHA256)
            self.assertEqual(K.file_sha256(STUDY / "GeoCPU_practical_training_logging_20261107" / K.CODE_ZIP), K.CODE_ZIP_SHA256)


class Dedupe(unittest.TestCase):
    def test_dedupe_counts_rows_per_prompt(self):
        items = [{"prompt": "a", "target": "x"}, {"prompt": "b", "target": "y"}, {"prompt": "a", "target": "x"}]
        out = K.dedupe_by_prompt(items)
        self.assertEqual([(i["prompt"], i["weight"]) for i in out], [("a", 2), ("b", 1)])


if __name__ == "__main__":
    unittest.main()
