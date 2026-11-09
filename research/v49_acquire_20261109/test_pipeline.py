#!/usr/bin/env python3
"""Tests of pipeline.py: the notebook's steps run end to end on the tiny fixtures (< 100 KB) in a fake Drive tree.  Nothing is downloaded,
nothing larger than the fixtures is read:

    python -I -X utf8 test_pipeline.py        # one process; it starts the converters one after the other (parallel=1)

The only step that is not run for real is step 5 (it needs the Hugging Face Hub; acquire.py has its own tests with fake Hub functions):
the layout it leaves behind (provenance.json per dataset, files under external_data_v49/) is built by hand here, and verify_acquired()
is run on it.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import acquire as A                                              # noqa: E402
import checks as C                                               # noqa: E402
import convert as CV                                             # noqa: E402
import pipeline as P                                             # noqa: E402
import test_checks as TC                                         # noqa: E402  (make_bundle: a Drive-like tree with the eval manifests)

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception:                                                # pragma: no cover
    pa = pq = None

FIX = HERE / "fixtures"
VOCAB = HERE.parents[1] / "src" / "v44fix" / "base_metadata.json"
COMMIT = "0123456789abcdef0123456789abcdef01234567"


def fixture(name):
    return json.loads((FIX / name).read_text(encoding="utf-8"))


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def fake_provenance(ds: dict, files: list) -> dict:
    """What acquire.build_provenance writes (only the fields the pipeline reads)."""
    recs = []
    for rel, data, role in files:
        recs.append({"path": rel.split("/", 1)[-1] if role in ("dev", "validation") else rel, "role": role, "may_train": role == "train",
                     "dest_rel": ("do_not_train_%s/" % role if role in ("dev", "validation") else "") + A.folder_of(ds["repo"]) + "/" + rel, "size": len(data), "sha256": sha(data)})
    return {"schema": A.SCHEMA, "repo_id": ds["repo"], "dataset_key": ds["key"], "commit_sha": COMMIT, "retrieved_utc": "2026-11-09T12:00:00Z",
            "license": {"expected_ids": list(ds["licenses"]), "ids_from_card": list(ds["licenses"]), "name": ds["license_name"], "url": ds["license_url"],
                        "share_alike": ds["share_alike"]},
            "attribution": {"lines": A.attribution_lines(ds)}, "row_defaults": {"source": ds["row_source"], "license": ds["row_license"]},
            "language": ds["language"], "rows_per_card": ds["rows_card"], "generator_notes": list(ds["generator_notes"]), "quality_notes": list(ds["quality_notes"]),
            "adoption": dict(ds["adoption"]), "files": recs}


def parquet_bytes(records) -> bytes:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "x.parquet"
        pq.write_table(pa.Table.from_pylist(records), p)
        return p.read_bytes()


def fixture_layout() -> dict:
    """{acquire key: [(path inside the repo, bytes, role)]}: the fixtures as the real files would be laid out in the repos."""
    hp = fixture("hotpot_fixture.json")
    jq = (FIX / "jaquad_train_fixture.json").read_bytes()
    dev = json.dumps({"data": [{"title": "DEV", "paragraphs": [{"context": "甲は乙である。丙は丁である。", "qas": [
        {"id": "de-001-00-000", "question": "甲は何ですか。", "question_type": "x", "answers": [{"text": "乙", "answer_start": 2, "answer_type": "O"}]}]}]}]},
        ensure_ascii=False).encode("utf-8")
    layout = {
        "coding": [("data/train-00000-of-00001.parquet", parquet_bytes(fixture("coding_fixture.json")), "train")],
        "tengentoppa": [("completed_sets.jsonl", (FIX / "tengentoppa_fixture.jsonl").read_bytes(), "train")],
        "jaquad": [("data/train/jaquad_train_0000.json", jq, "train"), ("data/dev/jaquad_dev_0000.json", dev, "dev")],
        "hotpotqa": [("distractor/train-00000-of-00002.parquet", parquet_bytes(hp), "train"),
                     ("distractor/validation-00000-of-00001.parquet", parquet_bytes(hp[:1]), "validation")],
        "toolace": [("data.json", json.dumps(fixture("toolace_fixture.json"), ensure_ascii=False).encode("utf-8"), "train")],
        "wizardlm": [("data/train-00000-of-00002.parquet", parquet_bytes(fixture("wizard_fixture.json")), "train")],
    }
    return layout


def build_root(base: Path) -> Path:
    """A fake Drive folder in the layout acquire.py leaves behind, filled with the fixtures (hand-made provenance.json)."""
    root = base / "drive" / "GeoCPU_v47_fullmodel_20261106"
    layout = fixture_layout()
    for ds in A.DATASETS:
        files = [("README.md", b"---\nlicense: %s\n---\ncard\n" % ds["licenses"][0].encode(), "card")] + layout[ds["key"]]
        for rel, data, role in files:
            dr = ("do_not_train_%s/" % role if role in ("dev", "validation") else "") + A.folder_of(ds["repo"]) + "/" + rel
            p = root / A.EXTERNAL_DIRNAME / dr
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
        prov = fake_provenance(ds, files)
        (root / A.EXTERNAL_DIRNAME / A.folder_of(ds["repo"]) / "provenance.json").write_text(json.dumps(prov, ensure_ascii=False), encoding="utf-8")
    TC.make_bundle(str(root))                                                   # eval manifests inside the bundle zip
    return root


def write_eval_extra(base: Path) -> Path:
    d = base / "eval_extra"
    d.mkdir(parents=True, exist_ok=True)
    for n in P.V48_EVAL_NAMES:
        (d / n).write_text(json.dumps({"id": n, "prompt": "問題: v48の評価用の問題文はここに入ります。十分に長い文字列です。", "target": "\n答え: 1。"}, ensure_ascii=False) + "\n", encoding="utf-8")
    return d


@unittest.skipIf(pa is None or not VOCAB.is_file(), "pyarrow or the vocabulary file is missing")
class TestRealAcquireLayout(unittest.TestCase):
    """step 5 with the REAL acquire.py (fake Hub functions): what it writes is what the next steps read."""

    def test_acquire_then_mirror_raw_checks_and_conversion_read_what_acquire_wrote(self):
        import test_acquire as TA
        from unittest import mock
        layout = fixture_layout()
        with tempfile.TemporaryDirectory(prefix="test_pipeline_acq_", ignore_cleanup_errors=True) as td, mock.patch("os.fsync"):
            base = Path(td)
            root = base / "drive" / "GeoCPU_v47_fullmodel_20261106"
            root.mkdir(parents=True)
            TC.make_bundle(str(root))
            table, pairs = [], []
            for ds in A.DATASETS:
                card = b"---" + bytes([10]) + b"license: " + ds["licenses"][0].encode() + bytes([10]) + b"---" + bytes([10]) + b"card text" + bytes([10])
                content, entries = {"README.md": card}, [A.F("README.md", len(card), "card", blob=TA.git_blob(card))]
                for rel, data, role in layout[ds["key"]]:
                    content[rel] = data
                    if rel.endswith((".parquet", ".jsonl")):
                        entries.append(A.F(rel, len(data), role, sha256=TA.sha(data), pointer_blob=TA.git_blob(b"pointer:" + data)))
                    else:
                        entries.append(A.F(rel, len(data), role, blob=TA.git_blob(data)))
                d = dict(ds)
                d["files"], d["planning_sha"] = entries, hashlib.sha1(ds["repo"].encode()).hexdigest()
                table.append(d)
                pairs.append((d, content))
            hub = TA.FakeHub(pairs)
            logs = []
            with mock.patch.object(A, "DATASETS", table), mock.patch.object(A, "HfApi", hub.HfApi), mock.patch.object(A, "hf_hub_download", hub.hf_hub_download):
                cfg = P.Cfg(root=str(root), work=str(base / "work"), stage_dir=str(base / "stage"), allow_local=True, vocab_override=str(VOCAB), require_v48=False,
                            only=("jaquad", "hotpot", "toolace"), parallel=1, poll_s=0.2, heartbeat_s=1e9, check_disk=False, expected_counts=None,
                            run_semantic=False, run_cross=False, log=logs.append)
                R = {"dry": P.step_dry_run(cfg), "acq": P.step_acquire(cfg), "mirror": P.step_mirror(cfg), "vocab": P.step_vocab_eval(cfg),
                     "raw": P.step_raw_checks(cfg), "smoke": P.step_convert_smoke(cfg), "full": P.step_convert_full(cfg)}
                self.assertTrue(all(r["pass"] for r in R.values()))
                self.assertEqual(R["dry"]["expected_total_bytes"], A.total_bytes([d for d in table if d["key"] in ("jaquad", "hotpotqa", "toolace")]))
                # acquire wrote the layout; only train files were mirrored; dev / validation reach the eval index only
                self.assertTrue((root / A.EXTERNAL_DIRNAME / "do_not_train_dev" / "SkelterLabsInc__JaQuAD" / "data" / "dev" / "jaquad_dev_0000.json").is_file())
                self.assertFalse((cfg.mirror / "SkelterLabsInc__JaQuAD" / "data" / "dev").exists())
                self.assertEqual([p.name for p in P.mirror_train_files(cfg, "hotpot")], ["train-00000-of-00002.parquet"])
                st = R["vocab"]["eval"]["index"]["per_source"]
                self.assertGreater(st["jaquad_dev"]["items"], 0)
                self.assertGreater(st["hotpot_validation"]["items"], 0)
                prov = json.loads((root / A.EXTERNAL_DIRNAME / "SkelterLabsInc__JaQuAD" / "provenance.json").read_text(encoding="utf-8"))
                self.assertEqual(prov["row_defaults"]["license"], "CC-BY-SA-3.0")
                rows = [json.loads(l) for _, l in P.iter_stage_a(cfg, "jaquad")]
                self.assertTrue(rows and all(r["license"] == "CC-BY-SA-3.0" and "JaQuAD" in r["source"] for r in rows))   # convert.py applied provenance row_defaults
                for ds in table:
                    if ds["key"] in ("jaquad", "hotpotqa", "toolace"):
                        self.assertTrue((cfg.stage_a_drive({"hotpotqa": "hotpot"}.get(ds["key"], ds["key"])) / "stageA_meta.json").is_file())
                self.assertEqual(sorted(os.listdir(root)), sorted([A.EXTERNAL_DIRNAME, "GeoCPU_v47_fullmodel_bundle_v1.zip", P.OUT_NAME]))
                self.assertEqual([l for l in logs if l.startswith("FAIL [")], [])


@unittest.skipIf(pa is None or not VOCAB.is_file(), "pyarrow or the vocabulary file is missing")
class TestEndToEnd(unittest.TestCase):
    td = None
    results: dict = {}
    failure = None
    logs: list = []

    @classmethod
    def setUpClass(cls):
        cls.td = tempfile.TemporaryDirectory(prefix="test_pipeline_", ignore_cleanup_errors=True)
        base = Path(cls.td.name)
        cls.logs = []
        try:
            cls.root = build_root(base)
            extra = write_eval_extra(base)
            jq = fixture("jaquad_train_fixture.json")
            passage = jq["data"][0]["paragraphs"][0]["context"]
            assert len(passage) > 40
            user = cls.root / P.USER_EVAL_EXTRA                                    # the owner's own extra eval file: overlaps the first JaQuAD passage
            user.mkdir()
            (user / "mine.jsonl").write_text(json.dumps({"id": "x1", "prompt": passage[:60]}, ensure_ascii=False) + "\n", encoding="utf-8")
            cls.n_jq_qas = sum(len(p["qas"]) for a in jq["data"] for p in a["paragraphs"])
            cls.cfg = P.Cfg(root=str(cls.root), work=str(base / "work"), vocab_override=str(VOCAB), eval_extra_dir=str(extra), parallel=1, poll_s=0.2,
                            heartbeat_s=1e9, check_disk=False, expected_old_rows=None, log=cls.logs.append,
                            expected_counts={"jaquad": cls.n_jq_qas, "coding": 10, "tengentoppa": 2, "hotpot": 5, "toolace": 999, "wizard": 5})
            R = cls.results
            R["modules"] = P.step_modules(cls.cfg)
            R["dry"] = P.step_dry_run(cls.cfg)
            P.save_state(cls.cfg, "step5_acquire", P.verify_acquired(cls.cfg))
            R["mirror"] = P.step_mirror(cls.cfg)
            R["vocab_eval"] = P.step_vocab_eval(cls.cfg)
            R["raw"] = P.step_raw_checks(cls.cfg)
            R["smoke"] = P.step_convert_smoke(cls.cfg)
            R["full"] = P.step_convert_full(cls.cfg)
            # the old corpus: one exact copy of a converted row, one near copy (a few characters changed), and filler
            rows = {k: [json.loads(l) for _, l in P.iter_stage_a(cls.cfg, k)] for k in ("coding", "tengentoppa", "wizard")}
            cls.exact_src = rows["wizard"][0]
            cls.near_src = max(rows["coding"], key=lambda r: len(r["full_text"]))
            near_text = cls.near_src["full_text"][:-6] + "改変済み。"
            old = [{"id": "o_exact", "full_text": cls.exact_src["full_text"]}, {"id": "o_near", "full_text": near_text}]
            old += [{"id": f"o{i}", "full_text": "旧コーパスの行 %d: " % i + "".join(chr(0x3042 + (i * 7 + j * 3) % 80) for j in range(300))} for i in range(20)]
            oc = cls.root / C.OLD_CORPUS_REL
            oc.parent.mkdir(parents=True)
            oc.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in old), encoding="utf-8")
            cls.cfg.expected_old_rows = len(old)
            R["filters"] = P.step_filters(cls.cfg)
            R["final"] = P.step_finalize(cls.cfg)
        except BaseException as e:                                                  # noqa: BLE001
            cls.failure = e
        cls.pipeline_logs = list(cls.logs)                                          # later tests add their own (intended) FAIL lines
        if os.environ.get("V49_SHOW_LOGS"):
            sys.stdout.write(chr(10).join(cls.pipeline_logs) + chr(10))

    @classmethod
    def tearDownClass(cls):
        cls.td.cleanup()

    def need(self, name):
        if self.failure is not None:
            self.fail(f"pipeline did not finish: {type(self.failure).__name__}: {self.failure}\n" + "\n".join(self.logs[-30:]))
        return self.results[name]

    # ---------------------------------------------------------------- the steps
    def test_every_step_prints_pass_and_reports_it(self):
        for n in ("modules", "dry", "mirror", "vocab_eval", "raw", "smoke", "full", "filters", "final"):
            r = self.need(n)
            self.assertTrue(r["pass"], n)
        passes = [l for l in self.pipeline_logs if l.startswith("PASS [")]
        self.assertGreaterEqual(len(passes), 9)
        self.assertFalse([l for l in self.pipeline_logs if l.startswith("FAIL [")])

    def test_dry_run_states_the_expected_total(self):
        r = self.need("dry")
        self.assertEqual(r["expected_total_bytes"], A.total_bytes(A.DATASETS))
        self.assertTrue(any(l.startswith("TOTAL: ") for l in self.logs))

    def test_count_mismatch_is_loud_and_recorded_but_does_not_stop(self):
        r = self.need("raw")
        self.assertEqual([m["dataset"] for m in r["mismatches"]], ["toolace"])
        self.assertTrue(any("MISMATCH toolace" in l and l.startswith("!") for l in self.logs))
        self.assertEqual(r["datasets"]["jaquad"]["n_raw"], self.n_jq_qas)
        self.assertTrue(r["datasets"]["jaquad"]["count_match"])
        self.assertEqual(sorted(r["datasets"]), sorted(P.KEYS))

    def test_stop_on_count_mismatch_fails_the_step(self):
        self.need("raw")
        cfg = P.Cfg(**{**{f.name: getattr(self.cfg, f.name) for f in P.dataclasses.fields(P.Cfg) if f.name not in ("state", "timings", "log")},
                       "stop_on_count_mismatch": True, "log": lambda s: None})
        P._CACHE["vocab"] = P._CACHE["vocab"]
        with self.assertRaises(P.StepFailed) as cm:
            P.step_raw_checks(cfg)
        self.assertIn("toolace", str(cm.exception))

    def test_reading_files_exist_for_every_dataset(self):
        self.need("final")
        for k in P.KEYS:
            for kind in ("raw", "final"):
                p = self.cfg.out / "reading" / f"{kind}_{k}_100.txt"
                self.assertTrue(p.is_file(), p)
                self.assertGreater(len(p.read_text(encoding="utf-8")), 100)

    def test_smoke_ran_the_same_converters_with_a_limit(self):
        r = self.need("smoke")
        self.assertEqual(sorted(r["datasets"]), sorted(P.KEYS))
        for k, v in r["datasets"].items():
            self.assertGreater(v["n_out"], 0, k)

    def test_funnel_is_monotone_and_old_corpus_and_eval_overlap_drop_rows(self):
        r = self.need("filters")
        st = r["datasets"]
        for k, v in st.items():
            c = v["counts"]
            seq = [c["converted"], c["after_exact_dedup"], c["after_near_dedup"], c["after_eval_decontam"], c["after_old_corpus_dedup"]]
            self.assertEqual(seq, sorted(seq, reverse=True), k)
            self.assertEqual(c["final_rows"], seq[-1])
        old = r["global"]["old_corpus"]
        self.assertEqual(old["rows_found"], 22)
        self.assertEqual(old["exact_dropped_by_dataset"].get("wizard"), 1)
        self.assertEqual(old["near_dropped_by_dataset"].get("coding"), 1)
        c = st["jaquad"]["counts"]
        self.assertLess(c["after_eval_decontam"], c["after_near_dedup"])           # the owner's extra eval file overlaps the first passage
        self.assertEqual(r["global"]["decontam"]["flagged_by_dataset"].get("jaquad"), c["after_near_dedup"] - c["after_eval_decontam"])
        self.assertGreater(c["final_rows"], 0)
        why = {k: v["dropped_by_reason"] for k, v in st.items()}
        self.assertIn("exact_duplicate_of_old_corpus", why["wizard"])
        self.assertIn("near_duplicate_of_old_corpus", why["coding"])
        self.assertIn("eval_overlap_13gram", why["jaquad"])

    def test_dropped_lists_name_every_dropped_row(self):
        r = self.need("filters")
        for k in P.KEYS:
            lines = (self.cfg.out / "filters" / f"{k}_dropped.tsv").read_text(encoding="utf-8").splitlines()
            c = r["datasets"][k]["counts"]
            self.assertEqual(len(lines) - 1, c["converted"] - c["final_rows"], k)

    def test_final_rows_are_on_drive_valid_and_match_the_registry(self):
        res = self.need("final")
        reg = json.loads((self.cfg.out / "registry.json").read_text(encoding="utf-8"))
        self.assertEqual(reg["schema"], "v49_acquire_registry/1")
        ck = P._convert_ck(self.cfg)
        for k in P.KEYS:
            d = self.cfg.rows / k
            self.assertEqual(sorted(p.name for p in d.iterdir()), ["ATTRIBUTION.md", "dev.jsonl", "heldout.jsonl", "manifest.json", "train.jsonl"])
            n = 0
            for sp in P.SPLITS:
                for line in (d / f"{sp}.jsonl").read_text(encoding="utf-8").splitlines():
                    row = json.loads(line)
                    self.assertIsNone(CV.validate_row(row, ck))
                    self.assertEqual(row["split"], sp)
                    n += 1
            self.assertEqual(n, reg["datasets"][k]["counts"]["final_rows"])
            for name, e in reg["datasets"][k]["final"]["files"].items():
                self.assertEqual(C.sha256_file(str(d / name)), e["sha256"])
            self.assertTrue((d / "ATTRIBUTION.md").read_text(encoding="utf-8").startswith("# Attribution"))
        self.assertEqual(reg["old_corpus"]["rows_found"], 22)
        self.assertEqual([m["dataset"] for m in reg["count_mismatches"]], ["toolace"])
        self.assertTrue(res["registry"]["vocab"]["n_single_chars"] == 3893)

    def test_registry_verdicts_follow_the_rules(self):
        self.need("final")
        reg = json.loads((self.cfg.out / "registry.json").read_text(encoding="utf-8"))
        v = {k: d["adoption"]["verdict"]["code"] for k, d in reg["datasets"].items()}
        self.assertEqual(v["toolace"], "hold")
        self.assertEqual(v["jaquad"], "candidate")
        self.assertEqual(v["tengentoppa"], "conditional")
        self.assertIn(v["wizard"], ("hold", "low_priority_candidate", "reject"))
        for d in reg["datasets"].values():
            self.assertIn("adoption", d["adoption"]["at_acquire"]["text"])

    def test_report_is_japanese_with_a_table_and_the_mismatch(self):
        self.need("final")
        text = (self.cfg.out / "REPORT_JA.md").read_text(encoding="utf-8")
        self.assertIn("| データセット | ライセンス | 元の件数 | 変換後 |", text)
        for k in P.KEYS:
            self.assertIn(f"| {k} |", text)
        self.assertIn("MISMATCH", text)
        self.assertIn("旧コーパス: 22 行を確認", text)
        self.assertIn("ShareAlike", text)

    def test_rerun_is_idempotent_and_never_overwrites(self):
        self.need("final")
        before = {p: C.sha256_file(str(p)) for p in self.cfg.out.rglob("*") if p.is_file() and not p.name.startswith("run_info_")}
        before_rows = {p: C.sha256_file(str(p)) for p in self.cfg.rows.rglob("*") if p.is_file()}
        r = P.step_finalize(self.cfg)                                              # same inputs, same code: identical files, no refusal
        self.assertTrue(r["pass"])
        after = {p: C.sha256_file(str(p)) for p in self.cfg.out.rglob("*") if p.is_file() and not p.name.startswith("run_info_")}
        self.assertEqual(before, after)
        self.assertEqual(before_rows, {p: C.sha256_file(str(p)) for p in self.cfg.rows.rglob("*") if p.is_file()})

    def test_rerunning_the_earlier_steps_changes_nothing_and_is_not_refused(self):
        self.need("final")

        def tree():
            return {p: C.sha256_file(str(p)) for p in list(self.cfg.out.rglob("*")) + list(self.cfg.rows.rglob("*"))
                    if p.is_file() and not p.name.startswith("run_info_")}
        before = tree()
        for fn in (P.step_vocab_eval, P.step_raw_checks, P.step_convert_full, P.step_filters, P.step_finalize):
            self.assertTrue(fn(self.cfg)["pass"], fn.step_name)
        self.assertEqual(before, tree())

    def test_changed_output_is_refused_with_the_run_suffix_hint(self):
        self.need("final")
        p = self.cfg.rows / "jaquad" / "train.jsonl"
        data = p.read_bytes()
        p.write_bytes(data + b"\n")                                                 # somebody edited an output on Drive
        try:
            with self.assertRaises(P.StepFailed) as cm:
                P.step_finalize(self.cfg)
            self.assertIn("RUN_SUFFIX", str(cm.exception))
            self.assertEqual(p.read_bytes(), data + b"\n")                          # the edited file is untouched
        finally:
            p.write_bytes(data)

    def test_stage_a_on_drive_resumes_and_refuses_other_code(self):
        self.need("full")
        d = self.cfg.stage_a_drive("toolace")
        self.assertTrue((d / "stageA_meta.json").is_file())
        # a second config with an empty local disk restores the rows from Drive instead of converting again
        with tempfile.TemporaryDirectory() as w2:
            cfg2 = P.Cfg(root=str(self.root), work=w2, vocab_override=str(VOCAB), parallel=1, poll_s=0.2, check_disk=False, log=lambda s: None, only=("toolace",))
            self.assertTrue(P.restore_stage_a(cfg2, "toolace"))
            self.assertEqual(C.sha256_file(str(cfg2.conv_dir("toolace") / "train.jsonl")), C.sha256_file(str(d / "train.jsonl")))
            meta = json.loads((d / "stageA_meta.json").read_text(encoding="utf-8"))
            meta["code"]["convert.py"] = "0" * 64
            (d / "stageA_meta.json").write_text(json.dumps(meta), encoding="utf-8")
            try:
                with tempfile.TemporaryDirectory() as w3:
                    cfg3 = P.Cfg(root=str(self.root), work=w3, parallel=1, log=lambda s: None, only=("toolace",))
                    with self.assertRaises(P.StepFailed) as cm:
                        P.restore_stage_a(cfg3, "toolace")
                    self.assertIn("different code", str(cm.exception))
            finally:
                meta["code"] = P.Cfg(root=str(self.root)).code_hashes()
                (d / "stageA_meta.json").write_text(json.dumps(meta), encoding="utf-8")

    def test_dev_and_validation_files_only_reach_the_eval_index(self):
        self.need("vocab_eval")
        st = self.results["vocab_eval"]["eval"]["index"]["per_source"]
        self.assertGreater(st["jaquad_dev"]["items"], 0)
        self.assertGreater(st["hotpot_validation"]["items"], 0)
        for k in P.KEYS:
            for f in P.mirror_train_files(self.cfg, k):
                self.assertNotIn("do_not_train", f.as_posix())
                self.assertNotRegex(f.name, r"dev|validation")
        # and no row id of the converted datasets comes from the dev file
        ids = {json.loads(l)["id"] for _, l in P.iter_stage_a(self.cfg, "jaquad")}
        self.assertNotIn("v49acq_jaquad_de-001-00-000", ids)


@unittest.skipIf(not VOCAB.is_file(), "the vocabulary file is missing")
class TestVocabFromBundleAndCache(unittest.TestCase):
    def test_vocabulary_is_read_from_the_bundle_zip_then_from_the_cache_with_identical_state(self):
        vocab = json.loads(VOCAB.read_text(encoding="utf-8"))["vocab"]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "drive" / "GeoCPU_v47_fullmodel_20261106"
            TC.make_bundle(str(root), vocab=vocab)
            logs = []
            cfg = P.Cfg(root=str(root), work=str(Path(td) / "work"), only=("toolace",), require_v48=False, check_disk=False, log=logs.append)
            r1 = P.step_vocab_eval(cfg)
            self.assertIn("base_metadata.json", r1["vocab"]["source"])
            self.assertFalse(r1["vocab"]["derived"])
            cfg2 = P.Cfg(root=str(root), work=str(Path(td) / "work2"), only=("toolace",), require_v48=False, check_disk=False, log=logs.append)
            r2 = P.step_vocab_eval(cfg2)                                            # reads vocab_cache.json now; the saved state must be identical
            self.assertEqual(r1["vocab"], r2["vocab"])
            self.assertTrue((cfg.out / "state" / "vocab_cache.json").is_file())
            self.assertEqual(len(json.loads(cfg.vocab_file.read_text(encoding="utf-8"))["vocab"]), 3893)

    def test_a_vocabulary_derived_from_a_corpus_is_refused(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "drive" / "GeoCPU_v47_fullmodel_20261106"
            root.mkdir(parents=True)
            cfg = P.Cfg(root=str(root), work=str(Path(td) / "work"), only=("toolace",), require_v48=False, check_disk=False, log=lambda s: None)
            with self.assertRaises(P.StepFailed):
                P.step_vocab_eval(cfg)


class TestUnits(unittest.TestCase):
    def test_dry_run_checks_the_free_disk_space(self):
        from unittest import mock
        from collections import namedtuple
        du = namedtuple("du", "total used free")
        with tempfile.TemporaryDirectory() as td:
            cfg = P.Cfg(root=td, work=str(Path(td) / "w"), log=lambda s: None)
            with mock.patch.object(P.shutil, "disk_usage", return_value=du(10, 9, 1 << 30)):
                with self.assertRaises(P.StepFailed) as cm:
                    P.step_dry_run(cfg)
                self.assertIn("local disk", str(cm.exception))
            with mock.patch.object(P.shutil, "disk_usage", return_value=du(10, 0, 100 << 30)):
                self.assertTrue(P.step_dry_run(cfg)["pass"])


    def test_gate_helpers(self):
        with self.assertRaises(P.StepFailed):
            P.require(False, "x")
        self.assertTrue(issubclass(P.StepFailed, AssertionError))

    def test_step_decorator_prints_pass_and_fail_and_wraps_exceptions(self):
        logs = []
        cfg = P.Cfg(log=logs.append)

        @P.step("t ok")
        def ok(c):
            return "fine", {"x": 1}

        @P.step("t bad")
        def bad(c):
            P.require(False, "nope")

        @P.step("t boom")
        def boom(c):
            raise KeyError("k")
        r = ok(cfg)
        self.assertTrue(r["pass"] and r["x"] == 1)
        with self.assertRaises(P.StepFailed):
            bad(cfg)
        with self.assertRaises(P.StepFailed):
            boom(cfg)
        self.assertTrue(any(l.startswith("PASS [t ok]") for l in logs))
        self.assertTrue(any(l.startswith("FAIL [t bad] nope") for l in logs))
        self.assertTrue(any(l.startswith("FAIL [t boom] KeyError") for l in logs))

    def test_publish_file_is_create_only(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            (td / "a.txt").write_text("one", encoding="utf-8")
            self.assertEqual(P.publish_file(td / "a.txt", td / "d" / "a.txt"), "created")
            self.assertEqual(P.publish_file(td / "a.txt", td / "d" / "a.txt"), "identical")
            (td / "b.txt").write_text("two", encoding="utf-8")
            with self.assertRaises(P.StepFailed) as cm:
                P.publish_file(td / "b.txt", td / "d" / "a.txt")
            self.assertIn("RUN_SUFFIX", str(cm.exception))
            self.assertEqual((td / "d" / "a.txt").read_text(encoding="utf-8"), "one")
            self.assertEqual(sorted(os.listdir(td / "d")), ["a.txt"])

    def test_write_once_refuses_other_content(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "x" / "r.json"
            self.assertEqual(P.write_once(p, "{}"), "created")
            self.assertEqual(P.write_once(p, "{}"), "identical")
            with self.assertRaises(P.StepFailed):
                P.write_once(p, "{ }")

    def test_combined_stream_maps_positions_back(self):
        class Fake(P.Combined):
            def __init__(self, alive):
                self.keys = list(alive)
                self.alive = alive
                self.maps, self.starts, tot = [], [], 0
                for k in self.keys:
                    idx = [i for i, a in enumerate(alive[k]) if a]
                    self.maps.append(idx)
                    self.starts.append(tot)
                    tot += len(idx)
                self.total = tot
        c = Fake({"a": bytearray([1, 0, 1]), "empty": bytearray([0, 0]), "b": bytearray([0, 1, 1, 1])})
        self.assertEqual([c.locate(g) for g in range(c.total)], [("a", 0), ("a", 2), ("b", 1), ("b", 2), ("b", 3)])

    def test_dedupe_text_compares_the_question_part_for_passage_datasets(self):
        a = {"family": "jaquad_human", "prompt": "問題: 長い文章。\n質問: 甲は何か\n条件", "target": "\n答え: 乙。", "full_text": "x"}
        b = {"family": "jaquad_human", "prompt": "問題: 全く別の文章。\n質問: 甲は何か\n条件", "target": "\n答え: 乙。", "full_text": "y"}
        c = {"family": "coding_pytool", "prompt": "p", "target": "t", "full_text": "pt"}
        self.assertEqual(P.dedupe_text(a), P.dedupe_text(b))
        self.assertEqual(P.dedupe_text(c), "pt")

    def test_wizard_verdict_depends_on_the_measured_error_rate(self):
        cfg = P.Cfg()
        good = {"arith_error_rate": {"statement_error_rate": 0.004, "statements_checked": 5000}}
        bad = {"arith_error_rate": {"statement_error_rate": 0.08, "statements_checked": 5000}}
        few = {"arith_error_rate": {"statement_error_rate": 0.0, "statements_checked": 10}}
        self.assertEqual(P.adoption_verdict(cfg, "wizard", 100, good)["code"], "low_priority_candidate")
        self.assertEqual(P.adoption_verdict(cfg, "wizard", 100, bad)["code"], "hold")
        self.assertEqual(P.adoption_verdict(cfg, "wizard", 100, few)["code"], "hold")
        self.assertEqual(P.adoption_verdict(cfg, "wizard", 0, good)["code"], "reject")
        self.assertEqual(P.adoption_verdict(cfg, "toolace", 100, {})["code"], "hold")

    def test_coding_verdict_counts_the_verified_python_rows_of_the_final_set(self):
        cfg = P.Cfg()
        v = P.adoption_verdict(cfg, "coding", 10, {}, {"stored_output_equals_sandbox": 7, "sandbox_only_no_stored_output": 2})
        self.assertEqual(v["code"], "candidate")
        self.assertIn("出力一致 7行", v["ja"])
        self.assertIn("実行のみ 2行", v["ja"])

    def _scheduler(self, parallel, rc_of=None):
        """Run _run_converters with a fake Popen (no process is started): returns (start order, peak concurrency)."""
        from unittest import mock
        started, live, peak = [], set(), [0]
        rc_of = rc_of or {}

        class FakeProc:
            def __init__(self, cmd, **kw):
                self.key = cmd[cmd.index("--dataset") + 1]
                self.polls = 0
                started.append(self.key)
                live.add(self.key)
                peak[0] = max(peak[0], len(live))

            def poll(self):
                self.polls += 1
                if self.polls < 2:
                    return None
                live.discard(self.key)
                return rc_of.get(self.key, 0)
        with tempfile.TemporaryDirectory() as td:
            cfg = P.Cfg(root=td, work=str(Path(td) / "w"), parallel=parallel, poll_s=0, heartbeat_s=1e9, log=lambda s: None)
            with mock.patch.object(P.subprocess, "Popen", FakeProc):
                try:
                    P._run_converters(cfg, P.KEYS, 2000, smoke=True)
                    err = None
                except P.StepFailed as e:
                    err = e
        return started, peak[0], err

    def test_converters_are_started_slowest_first_and_never_more_than_parallel_at_once(self):
        started, peak, err = self._scheduler(2)
        self.assertIsNone(err)
        self.assertEqual(started, list(P.CONVERT_ORDER))
        self.assertEqual(started[0], "coding")
        self.assertEqual(peak, 2)
        started, peak, err = self._scheduler(1)
        self.assertEqual(peak, 1)

    def test_a_failing_converter_stops_the_step_and_names_it(self):
        _, _, err = self._scheduler(2, {"jaquad": 3})
        self.assertIsInstance(err, P.StepFailed)
        self.assertIn("jaquad rc=3", str(err))

    def test_verify_acquired_fails_loudly_when_a_dataset_is_missing(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = P.Cfg(root=td, log=lambda s: None, only=("jaquad",))
            with self.assertRaises(P.StepFailed):
                P.verify_acquired(cfg)

    def test_cfg_paths_use_the_new_folder_names_and_the_suffix(self):
        cfg = P.Cfg(root="/r", suffix="_2")
        self.assertEqual(cfg.out, Path("/r/v49_acquire_20261109_2"))
        self.assertEqual(cfg.rows, Path("/r/external_data_v49_rows_2"))
        self.assertEqual(cfg.ext, Path("/r/external_data_v49"))
        self.assertEqual(P.Cfg().acq_only(), None)
        self.assertEqual(P.Cfg(only=("hotpot", "wizard")).acq_only(), ["hotpotqa", "wizardlm"])
        with self.assertRaises(ValueError):
            P.Cfg(only=("nope",))


if __name__ == "__main__":
    unittest.main(verbosity=2)
