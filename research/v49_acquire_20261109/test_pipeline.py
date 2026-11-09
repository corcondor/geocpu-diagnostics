#!/usr/bin/env python3
"""Tests of pipeline.py: the notebook's steps run end to end on the tiny fixtures (< 100 KB) in a fake Drive tree.  Nothing is downloaded,
nothing larger than the fixtures is read:

    python -I -X utf8 test_pipeline.py        # one process; it starts the converters one after the other (parallel=1)

The only step that is not run for real is step 5 (it needs the Hugging Face Hub; acquire.py has its own tests with fake Hub functions):
the layout it leaves behind (provenance.json per dataset, files under external_data_v49/) is built by hand here, and verify_acquired()
is run on it.
"""
from __future__ import annotations

import collections
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


FENCE = CV.FENCE
JQ_ART = [("山田太郎は東京出身の作家である。", "山田太郎の出身地はどこ?", "東京"), ("鈴木花子は大阪で生まれた画家である。", "鈴木花子はどこで生まれた?", "大阪"),
          ("佐藤次郎は北海道の農家に育った。", "佐藤次郎はどこで育った?", "北海道"), ("高橋美咲は京都にある大学を卒業した。", "高橋美咲はどこの大学を卒業した?", "京都"),
          ("伊藤健一は福岡市の市長を務めた。", "伊藤健一は何市の市長を務めた?", "福岡"), ("渡辺由美は仙台で会社を設立した。", "渡辺由美はどこで会社を設立した?", "仙台"),
          ("中村拓也は札幌の病院で働いている。", "中村拓也はどこの病院で働いている?", "札幌"), ("小林真理は名古屋で店を開いた。", "小林真理はどこで店を開いた?", "名古屋")]
JQ_TAIL = "別の文が続きます。"
EXTRA_CODING = [
    {"messages": [{"role": "user", "content": "1から10までの合計を表示するPythonコードを書いて。"},
                  {"role": "assistant", "content": f"以下のPythonコードは、1から10までの合計を計算して表示します。\n{FENCE}python\nprint(sum(range(1, 11)))\n{FENCE}\n出力:\n{FENCE}\n55\n{FENCE}\n結果は55です。"}]},
    {"messages": [{"role": "user", "content": "Rustで挨拶を表示する関数を書いてください。"},
                  {"role": "assistant", "content": f"次のように書けます。\n{FENCE}rust\nfn main() {{\n    println!(\"こんにちは\");\n}}\n{FENCE}\nこれで標準出力に表示されます。"}]},
]


def jaquad_json() -> dict:
    jq = fixture("jaquad_train_fixture.json")
    for i, (ctx, q, a) in enumerate(JQ_ART):
        jq["data"].append({"title": f"人物{i}", "paragraphs": [{"context": ctx + JQ_TAIL, "qas": [
            {"id": f"tr-ok-{i:02d}-000", "question": q, "question_type": "Syntactic variation",
             "answers": [{"text": a, "answer_start": ctx.index(a), "answer_type": "Place"}]}]}]})
    return jq


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
    jq = json.dumps(jaquad_json(), ensure_ascii=False).encode("utf-8")
    dev = json.dumps({"data": [{"title": "DEV", "paragraphs": [{"context": "甲は乙である。丙は丁である。", "qas": [
        {"id": "de-001-00-000", "question": "甲は何ですか。", "question_type": "x", "answers": [{"text": "乙", "answer_start": 2, "answer_type": "O"}]}]}]}]},
        ensure_ascii=False).encode("utf-8")
    layout = {
        "coding": [("data/train-00000-of-00001.parquet", parquet_bytes(fixture("coding_fixture.json") + EXTRA_CODING), "train")],
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
                events, after_acq = [], []
                cfg = P.Cfg(root=str(root), work=str(base / "work"), stage_dir=str(base / "stage"), allow_local=True, vocab_override=str(VOCAB), require_v48=False,
                            only=("jaquad", "hotpot", "toolace"), parallel=1, poll_s=0.2, heartbeat_s=1e9, check_disk=False, expected_counts=None,
                            run_semantic=False, run_cross=False, log=logs.append, drive_sync=lambda: events.append("sync"))

                def acquire_step():
                    r = P.step_acquire(cfg)
                    after_acq.append(len(events))
                    return r
                R = {"dry": P.step_dry_run(cfg), "acq": acquire_step(), "mirror": P.step_mirror(cfg), "vocab": P.step_vocab_eval(cfg),
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
                self.assertEqual(sorted(os.listdir(root)), sorted([A.EXTERNAL_DIRNAME, "GeoCPU_v47_fullmodel_bundle_v1.zip", cfg.out.name, P.STAGE_A_NAME]))
                self.assertEqual(cfg.out.name, P.OUT_NAME + "_only_jaquad-hotpot-toolace")                 # a trial run has its own output folder
                self.assertEqual(after_acq, [3 + 1])                                         # one Drive flush per acquired dataset (before its provenance.json) + one after step 5
                self.assertEqual([l for l in logs if l.startswith("FAIL [")], [])


@unittest.skipIf(pa is None or not VOCAB.is_file(), "pyarrow or the vocabulary file is missing")
class TestEndToEnd(unittest.TestCase):
    td = None
    results: dict = {}
    failure = None
    logs: list = []

    @classmethod
    def setUpClass(cls):
        from unittest import mock
        cls._apache = mock.patch.object(P, "find_apache_license_text", return_value="APACHE LICENSE TEXT\n")        # the same text on every run, as on one machine
        cls._apache.start()
        cls.td = tempfile.TemporaryDirectory(prefix="test_pipeline_", ignore_cleanup_errors=True)
        base = Path(cls.td.name)
        cls.logs = []
        try:
            cls.root = build_root(base)
            extra = write_eval_extra(base)
            jq = jaquad_json()
            passage = JQ_ART[0][0]                                                  # the first sentence of a clean passage (the five real fixture rows are all rejected)
            assert len(passage) > 13
            user = cls.root / P.USER_EVAL_EXTRA                                    # the owner's own extra eval file: overlaps the first clean JaQuAD passage
            user.mkdir()
            (user / "mine.jsonl").write_text(json.dumps({"id": "x1", "prompt": passage[:60]}, ensure_ascii=False) + "\n", encoding="utf-8")
            cls.n_jq_qas = sum(len(p["qas"]) for a in jq["data"] for p in a["paragraphs"])
            cls.events = []                                                         # what the Drive hooks saw, in order
            cls.cfg = P.Cfg(root=str(cls.root), work=str(base / "work"), vocab_override=str(VOCAB), eval_extra_dir=str(extra), parallel=1, poll_s=0.2,
                            heartbeat_s=1e9, check_disk=False, expected_old_rows=None, log=cls.logs.append, shard_records=3,
                            drive_detach=lambda: cls.events.append("detach"), drive_attach=lambda: cls.events.append("attach"),
                            drive_sync=lambda: cls.events.append("sync"),
                            expected_counts={"jaquad": cls.n_jq_qas, "coding": 12, "tengentoppa": 2, "hotpot": 5, "toolace": 999, "wizard": 5})
            R = cls.results
            R["modules"] = P.step_modules(cls.cfg)
            R["dry"] = P.step_dry_run(cls.cfg)
            P.save_state(cls.cfg, "step5_acquire", P.verify_acquired(cls.cfg))
            R["mirror"] = P.step_mirror(cls.cfg)
            R["vocab_eval"] = P.step_vocab_eval(cls.cfg)
            R["raw"] = P.step_raw_checks(cls.cfg)
            R["sandbox"] = P.step_sandbox_selftest(cls.cfg)
            cls.events.clear()
            R["smoke"] = P.step_convert_smoke(cls.cfg)
            cls.smoke_events = list(cls.events)
            cls.events.clear()
            R["full"] = P.step_convert_full(cls.cfg)
            cls.full_events = list(cls.events)
            # the old corpus: one exact copy of a converted row, one near copy (a few characters changed), and filler
            rows = {k: [json.loads(l) for _, l in P.iter_stage_a(cls.cfg, k)] for k in ("coding", "tengentoppa", "wizard")}
            cls.exact_src = rows["wizard"][0]
            cls.near_src = next(r for r in rows["coding"] if r["family"] == "code_text")      # the python row must survive: it is the verified one
            near_text = cls.near_src["full_text"][:-6] + "改変済み。"
            old = [{"id": "o_exact", "full_text": cls.exact_src["full_text"]}, {"id": "o_near", "full_text": near_text}]
            old += [{"id": f"o{i}", "full_text": "旧コーパスの行 %d: " % i + "".join(chr(0x3042 + (i * 7 + j * 3) % 80) for j in range(300))} for i in range(20)]
            oc = cls.root / C.OLD_CORPUS_REL
            oc.parent.mkdir(parents=True)
            oc.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in old), encoding="utf-8")
            cls.cfg.expected_old_rows = len(old)
            R["filters"] = P.step_filters(cls.cfg)
            R["reviews"] = P.step_reviews(cls.cfg)
            cls.publish_log = []                                                    # (destination, does the hold marker exist yet?) of every publish in step 9
            real_publish = P.publish_file

            def spy_publish(src, dst, *a, **kw):
                cls.publish_log.append((str(dst), (Path(dst).parent / "DO_NOT_TRAIN_UNTIL_APPROVED.txt").exists()))
                return real_publish(src, dst, *a, **kw)
            with mock.patch.object(P, "publish_file", spy_publish):
                R["final"] = P.step_finalize(cls.cfg)
        except BaseException as e:                                                  # noqa: BLE001
            cls.failure = e
        cls.pipeline_logs = list(cls.logs)                                          # later tests add their own (intended) FAIL lines
        if os.environ.get("V49_SHOW_LOGS"):
            sys.stdout.write(chr(10).join(cls.pipeline_logs) + chr(10))

    @classmethod
    def tearDownClass(cls):
        cls._apache.stop()
        cls.td.cleanup()

    def need(self, name):
        if self.failure is not None:
            self.fail(f"pipeline did not finish: {type(self.failure).__name__}: {self.failure}\n" + "\n".join(self.logs[-30:]))
        return self.results[name]

    # ---------------------------------------------------------------- the steps
    def test_every_step_prints_pass_and_reports_it(self):
        for n in ("modules", "dry", "mirror", "vocab_eval", "raw", "sandbox", "smoke", "full", "filters", "reviews", "final"):
            r = self.need(n)
            self.assertTrue(r["pass"], n)
        passes = [l for l in self.pipeline_logs if l.startswith("PASS [")]
        self.assertGreaterEqual(len(passes), 11)
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
            seq = [c[x] for x in P.FUNNEL]
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
        self.assertEqual(reg["schema"], "v49_acquire_registry/2")
        ck = P._convert_ck(self.cfg)
        for k in P.KEYS:
            hold = k in P.HOLD_KEYS
            d = self.cfg.rows / P.HOLD_DIRNAME / k if hold else self.cfg.rows / k
            apache = ["LICENSE-Apache-2.0.txt"] if k != "jaquad" and k != "hotpot" else []
            self.assertEqual(sorted(p.name for p in d.iterdir()),
                             sorted(["ATTRIBUTION.md", "LICENSE_NOTICE.txt", "dev.jsonl", "heldout.jsonl", "manifest.json", "train.jsonl"] + apache
                                    + (["DO_NOT_TRAIN_UNTIL_APPROVED.txt"] if hold else [])), k)
            self.assertFalse((self.cfg.rows / k).exists() and hold, k)             # a held dataset is never next to the candidates
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
            notice = (d / "LICENSE_NOTICE.txt").read_text(encoding="utf-8")
            self.assertIn(reg["datasets"][k]["license"]["url"], notice)
            if k in ("jaquad", "hotpot"):
                self.assertIn("DO NOT merge ShareAlike rows", notice)
                self.assertIn("DO NOT merge", (d / "ATTRIBUTION.md").read_text(encoding="utf-8").replace("Do not", "DO NOT"))
            else:
                self.assertEqual((d / "LICENSE-Apache-2.0.txt").read_text(encoding="utf-8"), "APACHE LICENSE TEXT\n")
        self.assertEqual(reg["held"], ["toolace", "wizard"])
        self.assertEqual(reg["trainable"], ["coding", "tengentoppa", "jaquad", "hotpot"])
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
        self.assertEqual(v["wizard"], "hold")                                       # never promoted automatically
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
        self.assertIn("T1〜T8 の評価データの照合状況", text)
        self.assertIn("T8 code/explanation form probe", text)
        self.assertIn("行単位。dev / heldout をグループ独立の評価として使わないでください", text)      # the four synthetic datasets
        self.assertIn("検証済み", text)
        self.assertIn("見つかる確率は計算上", text)
        self.assertIn("真のJaccard 0.8 の組で約93%", text)
        self.assertIn(P.HOLD_DIRNAME, text)
        self.assertIn("英語の行:", text)
        self.assertIn("NOT found on Drive", text)                                    # the registry suites that are not there are named in section 5
        self.assertIn("丸ごと入っていた行", text)                                        # whole eval items (templated items that differ only in digits) are counted apart
        self.assertIn("データセット別の丸ごと一致", text)                                  # ... per dataset, with the warning to read it before trusting the drop counts
        self.assertIn("この行数を読んでから除外数を信じてください", text)
        self.assertIn("13〜39文字の評価文は13文字の断片でしか照合しない", text)            # the band between 'shorter than 13' and the 40-character whole items
        self.assertIn("13 to 39 normalised characters", text)
        self.assertNotIn("行 - 行", text)                                              # ... with a real number, not a dash
        self.assertIn("長さ（最終のtrain行）", text)
        self.assertIn("落とすのは、すでに残した行に似ている行だけです", text)                    # near duplicates: no chains

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
        for fn in (P.step_vocab_eval, P.step_raw_checks, P.step_convert_smoke, P.step_convert_full, P.step_filters, P.step_reviews, P.step_finalize):
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

    def fresh_cfg(self, work, **kw):
        """A config with an EMPTY local disk (a restarted runtime) that has the same vocabulary file as the original run."""
        cfg = P.Cfg(root=str(self.root), work=work, vocab_override=str(VOCAB), parallel=1, poll_s=0.2, check_disk=False, log=lambda s: None, **kw)
        Path(work).mkdir(parents=True, exist_ok=True)
        cfg.vocab_file.write_bytes(self.cfg.vocab_file.read_bytes())
        return cfg

    def test_stage_a_on_drive_resumes_and_refuses_other_code(self):
        self.need("full")
        d = self.cfg.stage_a_drive("toolace")
        self.assertTrue((d / "stageA_meta.json").is_file())
        self.assertEqual(d.parent.parent.name, P.STAGE_A_NAME)                      # not under the RUN_SUFFIX-dependent output folder
        # a second config with an empty local disk restores the rows from Drive instead of converting again
        with tempfile.TemporaryDirectory() as w2:
            cfg2 = self.fresh_cfg(w2, only=("toolace",))
            self.assertTrue(P.restore_stage_a(cfg2, "toolace"))
            self.assertEqual(C.sha256_file(str(cfg2.conv_dir("toolace") / "train.jsonl")), C.sha256_file(str(d / "train.jsonl")))
            meta = json.loads((d / "stageA_meta.json").read_text(encoding="utf-8"))
            good = json.dumps(meta)
            meta["converter"]["convert.py"] = "0" * 64
            (d / "stageA_meta.json").write_text(json.dumps(meta), encoding="utf-8")
            try:
                with tempfile.TemporaryDirectory() as w3:
                    cfg3 = self.fresh_cfg(w3, only=("toolace",))
                    with self.assertRaises(P.StepFailed) as cm:
                        P.restore_stage_a(cfg3, "toolace")
                    self.assertIn("different converter code", str(cm.exception))
            finally:
                (d / "stageA_meta.json").write_text(good, encoding="utf-8")
            # review round 3: rows made from OTHER RAW DATA (another commit, other train files) are refused too - they would be labelled with the
            # new commit in registry.json while holding the old data
            for what in ("train", "commit"):
                meta = json.loads(good)
                if what == "train":
                    meta["raw"]["train_sha256"][0][1] = "0" * 64
                else:
                    meta["raw"]["commit_sha"] = "f" * 40
                (d / "stageA_meta.json").write_text(json.dumps(meta), encoding="utf-8")
                try:
                    with tempfile.TemporaryDirectory() as w4:
                        with self.assertRaises(P.StepFailed) as cm:
                            P.restore_stage_a(self.fresh_cfg(w4, only=("toolace",)), "toolace")
                        self.assertIn("other raw data", str(cm.exception), what)
                finally:
                    (d / "stageA_meta.json").write_text(good, encoding="utf-8")
            self.assertIn("raw", json.loads(good))
            # the same for a shard
            sd = self.cfg.shard_drive("coding", *P.shard_ranges(self.results["raw"]["datasets"]["coding"]["n_raw"], self.cfg.shard_records)[0])
            smeta_good = (sd / "shard_meta.json").read_text(encoding="utf-8")
            smeta = json.loads(smeta_good)
            smeta["raw"]["commit_sha"] = "e" * 40
            (sd / "shard_meta.json").write_text(json.dumps(smeta), encoding="utf-8")
            try:
                with tempfile.TemporaryDirectory() as w5:
                    cfg5 = self.fresh_cfg(w5, only=("coding",), shard_records=self.cfg.shard_records)
                    a, b = P.shard_ranges(self.results["raw"]["datasets"]["coding"]["n_raw"], cfg5.shard_records)[0]
                    with self.assertRaises(P.StepFailed) as cm:
                        P.restore_shard(cfg5, "coding", a, b)
                    self.assertIn("other raw data", str(cm.exception))
            finally:
                (sd / "shard_meta.json").write_text(smeta_good, encoding="utf-8")

    def test_a_restarted_runtime_regenerates_nothing_different_and_a_finished_smoke_step_is_not_repeated(self):
        # review round 3 (blocker): 7a saved wall-clock estimates through write-once and a re-run was refused; (major) REPORT_JA.md changed
        # its dict order after a restart (state reloaded from sorted JSON) and cell 9 was refused too
        self.need("final")
        from unittest import mock

        def tree():
            return {p: C.sha256_file(str(p)) for p in list(self.cfg.out.rglob("*")) + list(self.cfg.rows.rglob("*"))
                    if p.is_file() and not p.name.startswith("run_info_")}
        before = tree()
        cfg2 = P.dataclasses.replace(self.cfg, state={}, timings=[], log=lambda s: None)       # a restarted notebook: nothing in memory but the vocabulary
        self.assertEqual(cfg2.state, {})
        with mock.patch.object(P, "_run_converters", side_effect=AssertionError("the smoke converters were started again")):
            r = P.step_convert_smoke(cfg2)
        self.assertIn("already done", r["detail"])
        self.assertEqual(r["estimate_minutes"], self.results["smoke"]["estimate_minutes"])
        self.assertTrue(P.step_finalize(cfg2)["pass"])
        self.assertEqual(before, tree())

    def test_registry_carries_the_length_information_the_training_plan_reads(self):
        self.need("final")
        reg = json.loads((self.cfg.out / "registry.json").read_text(encoding="utf-8"))
        seen_train_rows = 0
        for k in P.KEYS:
            d = reg["datasets"][k]
            rd = self.cfg.rows / P.HOLD_DIRNAME / k if k in P.HOLD_KEYS else self.cfg.rows / k
            lens, masked, sup = [], [], {sp: 0 for sp in P.SPLITS}
            for sp in P.SPLITS:
                for line in (rd / f"{sp}.jsonl").read_text(encoding="utf-8").splitlines():
                    r = json.loads(line)
                    m = sum(b - a for a, b in r["loss_mask_spans"])
                    sup[sp] += len(r["target"]) - m
                    if sp == "train":
                        lens.append(len(r["full_text"]) + 1)
                        masked.append(m)
            self.assertEqual(d["supervised_chars_by_split"], sup, k)
            self.assertEqual(d["supervised_chars"], {"train": sup["train"], "all": sum(sup.values())}, k)
            self.assertEqual(d["length_hist_bucket"], 64)
            self.assertIn("midpoint", d["length_note"])
            man = json.loads((rd / "manifest.json").read_text(encoding="utf-8"))
            for key in ("mean_len", "mean_masked", "length_hist", "supervised_chars_by_split"):
                self.assertEqual(man[key], d[key], (k, key))
            if lens:
                seen_train_rows += len(lens)
                self.assertAlmostEqual(d["mean_len"], sum(lens) / len(lens), places=6, msg=k)            # mean of len(prompt + target) + 1 over the TRAIN rows
                self.assertAlmostEqual(d["mean_masked"], sum(masked) / len(masked), places=6, msg=k)
                self.assertEqual(d["length_hist"], dict(collections.Counter(str(64 * (n // 64) + 32) for n in lens)), k)   # 64-character buckets, key = midpoint
                self.assertGreater(d["mean_len"], sup["train"] / len(lens), k)                           # the plan needs len_mean > sup_mean
            else:
                self.assertIsNone(d["mean_len"], k)
                self.assertEqual(d["length_hist"], {}, k)
        self.assertGreater(seen_train_rows, 0)
        self.assertIsNone(reg["only"])

    def test_the_hold_marker_exists_before_any_held_row_is_published(self):
        self.need("final")
        held = [(dst, marker) for dst, marker in self.publish_log if P.HOLD_DIRNAME in dst]
        self.assertTrue(held)
        self.assertTrue(all(marker for _, marker in held), held)

    def test_a_new_run_suffix_redoes_the_filters_but_not_the_conversion(self):
        self.need("final")
        from unittest import mock
        with tempfile.TemporaryDirectory() as w2:
            cfg2 = self.fresh_cfg(w2, suffix="_2", expected_old_rows=22, expected_counts=self.cfg.expected_counts)
            for name in ("step6c_raw_checks", "step7a_smoke"):                          # (6c and 7a run before 7b in the notebook: their numbers guard 7b; 7b refuses to run without them)
                P.save_state(cfg2, name, self.cfg.state[name])
            self.assertNotEqual(cfg2.out, self.cfg.out)
            for k in P.KEYS:
                self.assertEqual(cfg2.stage_a_drive(k), self.cfg.stage_a_drive(k))
            P.ensure_sandbox_ok(cfg2)                                               # (already proven by this process; a fresh one would start processes)
            with mock.patch.object(P.subprocess, "Popen", side_effect=AssertionError("a converter was started")):
                r = P.step_convert_full(cfg2)
            self.assertTrue(r["pass"])
            self.assertIn("0 now", r["detail"])
            self.assertFalse((cfg2.out / "state").exists() and any((cfg2.out / "state").glob("step8*")))

    def test_shards_were_published_one_by_one_and_a_lost_shard_alone_is_redone(self):
        self.need("full")
        from unittest import mock
        key = "coding"
        n = self.results["raw"]["datasets"][key]["n_raw"]
        ranges = P.shard_ranges(n, self.cfg.shard_records)
        self.assertGreater(len(ranges), 2)
        for a, b in ranges:
            sd = self.cfg.shard_drive(key, a, b)
            self.assertTrue((sd / "shard_meta.json").is_file(), (a, b))
            self.assertTrue((sd / "train.jsonl").is_file())
        want = {n_: C.sha256_file(str(self.cfg.stage_a_drive(key) / n_)) for n_ in ("train.jsonl", "dev.jsonl", "heldout.jsonl")}
        # a runtime that died after the other shards were published: with the merged rows missing and one shard missing, only that shard is converted
        with tempfile.TemporaryDirectory() as drive2, tempfile.TemporaryDirectory() as w2:
            import shutil
            root2 = Path(drive2) / "d"
            shutil.copytree(self.root, root2)
            cfg2 = P.Cfg(root=str(root2), work=w2, vocab_override=str(VOCAB), parallel=1, poll_s=0.2, check_disk=False, log=lambda s: None, only=(key,),
                         shard_records=self.cfg.shard_records, expected_counts=self.cfg.expected_counts)
            cfg2.vocab_file.parent.mkdir(parents=True, exist_ok=True)
            cfg2.vocab_file.write_bytes(self.cfg.vocab_file.read_bytes())
            P.step_mirror(cfg2)                                                     # a restarted runtime runs 6a again before 7b
            for name in ("step6c_raw_checks", "step7a_smoke"):                      # (a trial run (only=) has its own output folder: 6c and 7a ran in it first)
                P.save_state(cfg2, name, self.cfg.state[name])
            lost = ranges[1]
            shutil.rmtree(cfg2.stage_a_drive(key))                                  # (this is a COPY in a temp dir; the notebook never deletes on Drive)
            shutil.rmtree(cfg2.shard_drive(key, *lost))
            started = []
            real = P.subprocess.Popen

            def spy(cmd, **kw):
                started.append(cmd)
                return real(cmd, **kw)
            with mock.patch.object(P.subprocess, "Popen", spy):
                r = P.step_convert_full(cfg2)
            self.assertTrue(r["pass"])
            self.assertEqual(len(started), 1)
            self.assertIn(str(lost[0]), started[0][started[0].index("--start") + 1])
            self.assertEqual(started[0][started[0].index("--limit") + 1], str(lost[1]))
            got = {n_: C.sha256_file(str(cfg2.stage_a_drive(key) / n_)) for n_ in want}
            self.assertEqual(got, want)                                             # identical to the rows of the uninterrupted run

    def test_an_unsharded_dataset_estimated_to_take_too_long_is_refused_before_it_starts(self):
        self.need("full")
        import shutil
        with tempfile.TemporaryDirectory() as drive2, tempfile.TemporaryDirectory() as w2:
            root2 = Path(drive2) / "d"
            shutil.copytree(self.root, root2)
            cfg2 = P.Cfg(root=str(root2), work=w2, vocab_override=str(VOCAB), parallel=1, poll_s=0.2, check_disk=False, log=lambda s: None, only=("toolace",),
                         shard_keys=(), max_unsharded_minutes=60.0, expected_counts=self.cfg.expected_counts)
            cfg2.vocab_file.parent.mkdir(parents=True, exist_ok=True)
            cfg2.vocab_file.write_bytes(self.cfg.vocab_file.read_bytes())
            shutil.rmtree(cfg2.stage_a_drive("toolace"))                                      # (a COPY in a temp dir)
            cfg2.state["step6c_raw_checks"] = self.cfg.state["step6c_raw_checks"]                  # (7b needs the 6c counts first)
            cfg2.state["step7a_smoke"] = {"estimate_minutes": {"toolace": 300.0}, "est_drive_bytes": 1}
            with self.assertRaises(P.StepFailed) as cm:
                P.step_convert_full(cfg2)
            self.assertIn("300", str(cm.exception))
            self.assertIn("shard", str(cm.exception).lower())
            self.assertIn("MAX_UNSHARDED_MINUTES", str(cm.exception))                      # a setting the owner can reach from cell 3a
            # no estimate for this dataset (7a had nothing to extrapolate from): the guard is off, it must not crash on formatting `None`
            cfg2.state["step7a_smoke"] = {"estimate_minutes": {}, "est_drive_bytes": 1}
            P.step_mirror(cfg2)                                                                # (a fresh local disk: the converter reads the mirrored train files)
            self.assertTrue(P.step_convert_full(cfg2)["pass"])

    def test_drive_is_unmounted_while_code_runs_and_flushed_after_the_writing_steps(self):
        self.need("full")
        # 7a: one window around all converters; 7b: a window per wave of converters; the hooks always come in pairs, in order
        self.assertEqual(self.smoke_events[0], "detach")
        self.assertEqual(self.smoke_events.count("detach"), self.smoke_events.count("attach"))
        ev = self.full_events
        self.assertGreater(ev.count("detach"), 3)
        self.assertEqual(ev.count("detach"), ev.count("attach"))
        self.assertTrue(all(ev[i] == "detach" and ev[i + 1] == "attach" for i in range(len(ev)) if ev[i] == "detach"))
        self.assertEqual(ev[-1], "sync")                                            # step 7b ends with a flush

    def test_the_sandbox_selftest_state_names_the_backend(self):
        r = self.need("sandbox")
        self.assertEqual(r["selftest"]["backend"], C.SANDBOX_BACKEND)
        recs = sorted((self.cfg.out / "state").glob("sandbox_selftest_*.json"))
        self.assertEqual(len(recs), 1)                                                  # a record named by its content: re-runs never conflict
        st = json.loads(recs[0].read_text(encoding="utf-8"))
        self.assertEqual(st["backend"], "checks.code_sandbox_check(profile=executor)")
        self.assertTrue(st["drive_unmounted_during_code_runs"])
        self.assertEqual(P.last_sandbox_record(self.cfg), P._CACHE["sandbox"])
        self.assertTrue(all(c["ok"] for c in st["checks"]))
        man = json.loads((self.cfg.conv_dir("coding") / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(man["backends"]["sandbox"], C.SANDBOX_BACKEND)
        verified = {json.loads(l)["verification"] for _, l in P.iter_stage_a(self.cfg, "coding") if json.loads(l).get("python_code")}
        self.assertEqual(verified, {"stored_output_equals_sandbox"})

    def test_reviews_are_saved_apart_and_say_when_they_were_cut_short(self):
        r = self.need("reviews")
        rv = r["reviews"]
        self.assertFalse(rv["semantic"].get("truncated"))
        self.assertTrue(Path(rv["semantic"]["file"]).is_file())
        self.assertIn("matrix", json.dumps(rv["cross_dataset"]) + "matrix")
        # a budget of zero cuts both short and the result says so (a fresh suffix: the saved result of the first run is returned otherwise)
        with tempfile.TemporaryDirectory() as w2:
            cfg2 = self.fresh_cfg(w2, suffix="_cut", semantic_budget_s=0.0, cross_budget_s=0.0, expected_counts=self.cfg.expected_counts)
            P.save_state(cfg2, "step8_filters", self.results["filters"] and {"datasets": self.results["filters"]["datasets"], "global": self.results["filters"]["global"]})
            for k in P.KEYS:                                                          # the drop lists of the first run, copied to the new output folder
                (cfg2.out / "filters").mkdir(parents=True, exist_ok=True)
                (cfg2.out / "filters" / f"{k}_dropped.tsv").write_bytes((self.cfg.out / "filters" / f"{k}_dropped.tsv").read_bytes())
            P._CACHE["eval_sources"] = P._CACHE["eval_sources"]
            out = P.step_reviews(cfg2)["reviews"]
            self.assertTrue(out["cross_dataset"]["truncated"])
            self.assertTrue(out["semantic"]["truncated"] or out["semantic"]["queries"] == 0)

    def test_split_hygiene_only_touches_the_row_level_datasets(self):
        r = self.need("filters")
        sh = r["global"]["split_hygiene"]
        self.assertEqual(sorted(sh["datasets"]), ["coding", "tengentoppa", "toolace", "wizard"])
        for k in ("jaquad", "hotpot"):
            self.assertIsNone(r["datasets"][k]["split_hygiene"])
        for k, v in sh["datasets"].items():
            self.assertGreaterEqual(v["dev_heldout_rows"], 0)

    def test_registry_documents_the_split_unit_and_the_held_datasets(self):
        self.need("final")
        reg = json.loads((self.cfg.out / "registry.json").read_text(encoding="utf-8"))
        for k in ("coding", "tengentoppa", "toolace", "wizard"):
            self.assertFalse(reg["datasets"][k]["convert"]["group_independent_dev_heldout"], k)
            self.assertIn("NOT group-independent", reg["datasets"][k]["convert"]["split_unit"])
        for k in ("jaquad", "hotpot"):
            self.assertTrue(reg["datasets"][k]["convert"]["group_independent_dev_heldout"], k)
        self.assertIsInstance(reg["decontam"]["rows_flagged_whole_item"], int)
        self.assertEqual(reg["decontam"]["whole_item_min_chars"], 40)
        by = reg["decontam"]["rows_flagged_whole_item_by_dataset"]                      # review round 4: the number to read before trusting the drop counts
        self.assertEqual(sum(by.values()), reg["decontam"]["rows_flagged_whole_item"])
        self.assertTrue(set(by) <= set(reg["datasets"]))
        self.assertTrue(any("13 to 39 normalised characters" in x and "rows_flagged_whole_item" in x for x in reg["not_covered"]))
        self.assertEqual(reg["params"]["near_estimate_cut"], 0.75)
        self.assertEqual(reg["params"]["minhash_k"], 128)
        self.assertEqual(reg["params"]["minhash_detection_probability"]["0.8"], 0.933)
        self.assertEqual(reg["params"]["boilerplate_eval_df"], 20)
        self.assertTrue(reg["params"]["python_requires_stored_output"])
        self.assertTrue(any("registry" in x for x in reg["not_covered"]) or reg["eval"]["missing_optional"] == [])
        self.assertTrue(any(x.startswith("registry:") for x in reg["eval"]["missing_optional"]))       # no registry suite exists in this fake Drive: named, not silent
        self.assertTrue(any("NOT found on Drive" in x for x in reg["not_covered"]))
        self.assertEqual(len(reg["eval"]["tiers"]), 8)
        self.assertTrue(all(v["shingles"] > 0 for t in reg["eval"]["tiers"] for v in t["files"].values()))
        man = json.loads((self.cfg.rows / P.HOLD_DIRNAME / "toolace" / "manifest.json").read_text(encoding="utf-8"))
        self.assertTrue(man["held"])
        coding = json.loads((self.cfg.rows / "coding" / "manifest.json").read_text(encoding="utf-8"))
        self.assertFalse(coding["held"])
        self.assertEqual(coding["verification"].get("stored_output_equals_sandbox"), 1)

    def test_dev_and_validation_files_only_reach_the_eval_index(self):
        self.need("vocab_eval")
        st = self.results["vocab_eval"]["eval"]["index"]["per_source"]
        self.assertGreater(st["jaquad_dev"]["items"], 0)
        self.assertGreater(st["hotpot_validation"]["items"], 0)
        for k in P.KEYS:
            for f in P.mirror_train_files(self.cfg, k):
                self.assertNotIn("do_not_train", f.as_posix())
                self.assertNotRegex(f.name, r"dev|validation")
        self.assertEqual(self.results["vocab_eval"]["eval"]["index"]["max_eval_items_per_shingle"] >= 1, True)
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

    def test_a_vocabulary_with_another_character_set_is_refused_unless_allowed(self):
        vocab = json.loads(VOCAB.read_text(encoding="utf-8"))["vocab"]
        swapped = dict(vocab)
        victim = next(k for k in vocab if len(k) == 1 and ord(k) > 0x4e00)                    # one rare kanji becomes another character: same size, other set
        swapped.pop(victim)
        swapped["☃"] = vocab[victim]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "drive" / "GeoCPU_v47_fullmodel_20261106"
            TC.make_bundle(str(root), vocab=swapped)
            logs = []
            cfg = P.Cfg(root=str(root), work=str(Path(td) / "work"), only=("toolace",), require_v48=False, check_disk=False, log=logs.append)
            with self.assertRaises(P.StepFailed) as cm:
                P.step_vocab_eval(cfg)
            self.assertIn("ALLOW_VOCAB_MISMATCH", str(cm.exception))
            self.assertTrue(any("MISMATCH" in l and l.startswith("!") for l in logs))
            cfg2 = P.Cfg(root=str(root), work=str(Path(td) / "work2"), only=("toolace",), require_v48=False, check_disk=False, log=logs.append, allow_vocab_mismatch=True)
            r = P.step_vocab_eval(cfg2)
            self.assertFalse(r["vocab"]["sha_matches_owner_pc_copy"])

    def test_an_eval_manifest_without_any_text_is_refused(self):
        vocab = json.loads(VOCAB.read_text(encoding="utf-8"))["vocab"]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "drive" / "GeoCPU_v47_fullmodel_20261106"
            TC.make_bundle(str(root), vocab=vocab)
            (root / "jamcqa_manifest_test.jsonl").write_text(json.dumps({"id": "x", "prompt": "短い"}, ensure_ascii=False) + "\n", encoding="utf-8")     # a loose file wins
            cfg = P.Cfg(root=str(root), work=str(Path(td) / "work"), only=("toolace",), require_v48=False, check_disk=False, log=lambda s: None)
            with self.assertRaises(P.StepFailed) as cm:
                P.step_vocab_eval(cfg)
            self.assertIn("no 13-gram", str(cm.exception).replace("contributed no 13-gram at all", "no 13-gram"))
            self.assertIn("jamcqa_manifest_test.jsonl", str(cm.exception))

    def test_a_required_eval_file_that_repeats_the_prompts_of_another_is_not_refused(self):
        # review round 3 (blocker): eval_swe_step_control_heldout_v43labels.jsonl has the prompts of eval_swe_step_control_heldout.jsonl, so it added
        # 0 NEW shingles and step 6b stopped on the real files ('eval sources that contributed no 13-gram at all')
        vocab = json.loads(VOCAB.read_text(encoding="utf-8"))["vocab"]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "drive" / "GeoCPU_v47_fullmodel_20261106"
            TC.make_bundle(str(root), vocab=vocab)
            twin = "eval_swe_step_control_heldout_v43labels.jsonl"
            self.assertIn(twin, C.EVAL_REQUIRED_JSONL)
            text = "問題: eval_swe_step_control_heldout.jsonl の評価用の質問文です。ここは十分に長い文章です。"        # the text make_bundle wrote for the first file
            (root / twin).write_text(json.dumps({"id": "y", "prompt": text, "target": "A"}, ensure_ascii=False) + "\n", encoding="utf-8")
            cfg = P.Cfg(root=str(root), work=str(Path(td) / "work"), only=("toolace",), require_v48=False, check_disk=False, log=lambda s: None)
            r = P.step_vocab_eval(cfg)
            ps = r["eval"]["index"]["per_source"]
            self.assertEqual(ps[twin]["new_shingles"], 0)
            self.assertGreater(ps[twin]["shingles"], 0)
            self.assertEqual(P.eval_source_problems(ps, {twin}), {"no_items": [], "no_shingles": []})
            self.assertEqual(P.eval_source_problems({"a": {"items": 0, "indexable_texts": 3, "shingles": 5}, "b": {"items": 2, "indexable_texts": 0, "shingles": 0}}, {"a", "b", "c"}),
                             {"no_items": ["a", "c"], "no_shingles": ["b", "c"]})

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


    def test_dry_run_lists_stale_files_and_needs_room_for_the_converted_rows(self):
        from unittest import mock
        from collections import namedtuple
        du = namedtuple("du", "total used free")
        logs = []
        with tempfile.TemporaryDirectory() as td:
            cfg = P.Cfg(root=td, work=str(Path(td) / "w"), log=logs.append)
            stale = Path(td) / A.EXTERNAL_DIRNAME / "x" / "data.parquet.part.stale-1"
            stale.parent.mkdir(parents=True)
            stale.write_bytes(b"x" * (1 << 20))
            with mock.patch.object(P.shutil, "disk_usage", return_value=du(10, 0, 100 << 30)):
                r = P.step_dry_run(cfg)
            self.assertEqual(r["stale_files"], 1)
            self.assertTrue(any("never deleted" in l and "stale" in l for l in logs), logs)
            need, train = r["need_drive_bytes"], A.total_bytes(A.DATASETS, ("train",))
            self.assertEqual(need, r["expected_total_bytes"] + 2 * train + (1 << 30))        # raw files + converted rows + final rows + dumps
            self.assertGreater(need, 3 * r["expected_total_bytes"] // 2)                      # the old 'raw + 1.5 GiB' check was far too low
            with mock.patch.object(P.shutil, "disk_usage", return_value=du(10, 0, need - 1)):
                with self.assertRaises(P.StepFailed) as cm:
                    P.step_dry_run(cfg)
            self.assertIn("Drive reports", str(cm.exception))
            P.preflight_drive(cfg, {"est_drive_bytes": 1 << 20})
            with self.assertRaises(P.StepFailed):
                P.preflight_drive(cfg, {"est_drive_bytes": 1 << 62})

    def test_writing_steps_flush_drive_also_when_they_fail_and_a_failing_flush_hides_nothing(self):
        calls = []
        cfg = P.Cfg(log=lambda s: None, drive_sync=lambda: calls.append(1))

        @P.step("w", flush_on_failure=True)
        def writes(c):
            P.require(False, "nope")

        @P.step("r")
        def reads(c):
            P.require(False, "nope")

        @P.step("boom", flush_on_failure=True)
        def boom(c):
            raise KeyError("k")
        for fn in (writes, boom):
            with self.assertRaises(P.StepFailed):
                fn(cfg)
        with self.assertRaises(P.StepFailed):
            reads(cfg)
        self.assertEqual(calls, [1, 1])                                                       # not after the read-only step
        bad = P.Cfg(log=lambda s: None, drive_sync=lambda: (_ for _ in ()).throw(OSError("gone")))
        with self.assertRaises(P.StepFailed) as cm:
            writes(bad)
        self.assertIn("nope", str(cm.exception))
        for name in ("5 acquire", "7b convert full", "8 filters", "8b optional reviews", "9 final rows and report"):
            self.assertIn(f'@step("{name}", flush_on_failure=True)', Path(P.__file__).read_text(encoding="utf-8"), name)

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

    def test_wizard_is_never_promoted_automatically_whatever_the_measured_rate(self):
        # review finding: one narrow measure (a lower bound) with thresholds nobody set must not lift a dataset the owner's plan says to avoid
        cfg = P.Cfg()
        good = {"arith_error_rate": {"statement_error_rate": 0.004, "statements_checked": 5000}}
        bad = {"arith_error_rate": {"statement_error_rate": 0.08, "statements_checked": 5000}}
        few = {"arith_error_rate": {"statement_error_rate": 0.0, "statements_checked": 10}}
        for conv in (good, bad, few, {}):
            v = P.adoption_verdict(cfg, "wizard", 100, conv)
            self.assertEqual(v["code"], "hold", conv)
            self.assertIn("あなたが決めてください", v["ja"])
        self.assertIn("0.40%", P.adoption_verdict(cfg, "wizard", 100, good)["ja"])               # the measured rate is shown
        self.assertIn("8.00%", P.adoption_verdict(cfg, "wizard", 100, bad)["ja"])
        self.assertEqual(P.adoption_verdict(cfg, "wizard", 0, good)["code"], "reject")
        self.assertEqual(P.adoption_verdict(cfg, "toolace", 100, {})["code"], "hold")

    def test_coding_verdict_counts_only_the_verified_python_rows_of_the_final_set(self):
        cfg = P.Cfg()
        v = P.adoption_verdict(cfg, "coding", 10, {}, {"stored_output_equals_sandbox": 7, "unverified_no_execution": 2})
        self.assertEqual(v["code"], "candidate")
        self.assertIn("保存出力と一致したpython行 7行のみ", v["ja"])
        self.assertIn("code_text行 2行は未実行・未検証", v["ja"])

    def test_a_new_runtime_proves_the_sandbox_again_and_a_saved_result_is_never_trusted(self):
        from unittest import mock
        with tempfile.TemporaryDirectory() as td:
            cfg = P.Cfg(root=td, work=str(Path(td) / "w"), log=lambda s: None)
            (cfg.out / "state").mkdir(parents=True)
            (cfg.out / "state" / "sandbox_selftest_00000000.json").write_text(json.dumps({"backend": C.SANDBOX_BACKEND, "run_as": "nobody", "checks": []}), encoding="utf-8")
            saved = P._CACHE.pop("sandbox", None)
            try:
                with mock.patch.object(P.checks, "sandbox_selftest", wraps=C.sandbox_selftest) as spy:
                    P.ensure_sandbox_ok(cfg)                                               # a file from an earlier runtime does not count
                    P.ensure_sandbox_ok(cfg)                                               # this process has proven it: not again
                self.assertEqual(spy.call_count, 1)
                self.assertEqual(len(list((cfg.out / "state").glob("sandbox_selftest_*.json"))), 2)
                # on Colab (containment required) a cached result without an OS barrier is not enough
                P._CACHE["sandbox"] = {"backend": C.SANDBOX_BACKEND, "run_as": None}
                strict = P.Cfg(root=td, work=str(Path(td) / "w"), log=lambda s: None, require_os_containment=True)
                with self.assertRaises(P.StepFailed):
                    P.ensure_sandbox_ok(strict)                                            # (Windows / non-root here: the self-test itself fails)
            finally:
                if saved is not None:
                    P._CACHE["sandbox"] = saved
                else:
                    P._CACHE.pop("sandbox", None)

    def test_converter_tag_covers_the_converter_not_the_filters(self):
        with tempfile.TemporaryDirectory() as td:
            a, b = P.Cfg(root=td, work=str(Path(td) / "a")), P.Cfg(root=td, work=str(Path(td) / "b"), suffix="_2")
            self.assertEqual(a.converter_tag(), b.converter_tag())                       # RUN_SUFFIX does not change it
            self.assertNotEqual(a.code_tag()[:8], a.converter_tag()[:8])
            self.assertEqual(sorted(a.converter_hashes()), ["checks.sandbox_and_normaliser", "convert.py", "parameters", "vocabulary_chars"])
            self.assertEqual(a.converter_hashes()["vocabulary_chars"], "none")
            before = a.converter_tag()
            Path(a.work).mkdir(parents=True)
            a.vocab_file.write_text(json.dumps({"vocab": {"あ": 0, "い": 1}}), encoding="utf-8")
            self.assertNotEqual(a.converter_tag(), before)                               # another vocabulary = other converted rows
            self.assertEqual(P.Cfg(root=td, suffix="_9").stage_a_drive("x").parent.parent, Path(td) / P.STAGE_A_NAME)

    def test_shard_ranges_tile_the_records_and_the_last_runs_to_the_end(self):
        self.assertEqual(P.shard_ranges(10, 4), [(0, 4), (4, 8), (8, None)])
        self.assertEqual(P.shard_ranges(8, 4), [(0, 4), (4, None)])
        self.assertEqual(P.shard_ranges(3, 4), [(0, None)])
        r = P.shard_ranges(113796, 20000)
        self.assertEqual((len(r), r[0], r[-1]), (6, (0, 20000), (100000, None)))
        self.assertTrue(all(a[1] == b[0] for a, b in zip(r, r[1:])))

    def test_detection_table_matches_the_binomial_claims(self):
        pr = P.detection_probs(128, 0.75)
        self.assertEqual({j: round(p, 3) for j, p in pr.items()}, {0.9: 1.0, 0.85: 0.999, 0.8: 0.933, 0.75: 0.547, 0.7: 0.127})

    def test_raw_rows_are_lenient_and_count_what_they_skip(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = P.Cfg(root=td, work=str(Path(td) / "w"))
            d = cfg.mirror_dir("tengentoppa")
            d.mkdir(parents=True)
            (d / "completed_sets.jsonl").write_bytes('{"instruction": "a", "refined_answer": "b。"}\nnot json\n'.encode("utf-8") + b'{"instruction": "\xff"}\n')
            (d / "provenance.json").write_text(json.dumps({"files": [{"dest_rel": "DeL-TaiseiOzaki__Tengentoppa-sft-qwen2.5-32b-reasoning-100k/completed_sets.jsonl",
                                                                       "role": "train", "may_train": True}]}), encoding="utf-8")
            stats = {}
            rows = list(P.raw_rows(cfg, "tengentoppa", stats))
            self.assertEqual(len(rows), 2)
            self.assertEqual(stats, {"bad_lines": 1, "decode_replacements": 1})
            self.assertEqual(P.n_records({"datasets": {"tengentoppa": {"n_raw": 2, "bad_lines": 1}}}, "tengentoppa"), 3)

    def test_raw_suffix_gives_a_fresh_raw_folder_and_run_suffix_does_not(self):
        cfg = P.Cfg(root="/r", suffix="_2", raw_suffix="_raw2")
        self.assertEqual(cfg.ext, Path("/r/external_data_v49_raw2"))
        self.assertEqual(cfg.out, Path("/r/v49_acquire_20261109_2"))
        self.assertEqual(P.Cfg(root="/r", suffix="_2").ext, Path("/r/external_data_v49"))
        # review round 3: converted rows of the old raw data are never looked up for a fresh raw folder (RAW_SUFFIX); RUN_SUFFIX does not matter
        self.assertEqual(cfg.stage_a_drive("coding").parent.parent, Path("/r/external_data_v49_stageA_raw2"))
        self.assertEqual(cfg.shard_drive("coding", 0, None).parent.parent.parent, Path("/r/external_data_v49_stageA_raw2"))
        self.assertEqual(P.Cfg(root="/r", suffix="_2").stage_a_drive("coding").parent.parent, Path("/r/external_data_v49_stageA"))
        self.assertEqual(Path(cfg.stage_a_drive("coding")).parents[1], cfg.stage_a_root)

    def _scheduler(self, parallel, rc_of=None, wave_size=None, hooks=None, on_wave=None, raise_in_sleep=None):
        """Run _run_converters with a fake Popen (no process is started): returns (start order, peak concurrency, error, fake processes)."""
        from unittest import mock
        started, live, peak, procs = [], set(), [0], []
        rc_of = rc_of or {}

        class FakeProc:
            def __init__(self, cmd, **kw):
                self.key = cmd[cmd.index("--dataset") + 1]
                self.polls = 0
                self.stopped = []
                started.append(self.key)
                live.add(self.key)
                peak[0] = max(peak[0], len(live))
                procs.append(self)

            def poll(self):
                if self.stopped:
                    return -15
                self.polls += 1
                if self.polls < 2:
                    return None
                live.discard(self.key)
                return rc_of.get(self.key, 0)

            def terminate(self):
                self.stopped.append("terminate")
                live.discard(self.key)
        with tempfile.TemporaryDirectory() as td:
            kw = {} if hooks is None else {"drive_detach": hooks[0], "drive_attach": hooks[1]}
            cfg = P.Cfg(root=td, work=str(Path(td) / "w"), parallel=parallel, poll_s=0, heartbeat_s=1e9, log=lambda s: None, **kw)
            jobs = [P.new_job(k, stop=2000, out_root=cfg.conv_dir(k, smoke=True).parent) for k in P.CONVERT_ORDER]
            ctx = [mock.patch.object(P.subprocess, "Popen", FakeProc)]
            if raise_in_sleep is not None:
                ctx.append(mock.patch.object(P.time, "sleep", side_effect=raise_in_sleep))
            with ctx[0]:
                try:
                    if len(ctx) > 1:
                        with ctx[1]:
                            P._run_converters(cfg, jobs, smoke=True, limit=2000, wave_size=wave_size, on_wave=on_wave)
                    else:
                        P._run_converters(cfg, jobs, smoke=True, limit=2000, wave_size=wave_size, on_wave=on_wave)
                    err = None
                except (P.StepFailed, KeyboardInterrupt) as e:
                    err = e
            self.locks_left = sorted(p.name for p in (Path(td) / "w" / "locks").glob("*.lock")) if (Path(td) / "w" / "locks").exists() else []
        return started, peak[0], err, procs

    def test_converters_are_started_slowest_first_and_never_more_than_parallel_at_once(self):
        started, peak, err, _ = self._scheduler(2)
        self.assertIsNone(err)
        self.assertEqual(started, list(P.CONVERT_ORDER))
        self.assertEqual(started[0], "coding")
        self.assertEqual(peak, 2)
        started, peak, err, _ = self._scheduler(1)
        self.assertEqual(peak, 1)
        self.assertEqual(self.locks_left, [])                                       # every lock file is removed when its converter has finished

    def test_a_failing_converter_stops_the_step_and_names_it(self):
        _, _, err, _ = self._scheduler(2, {"jaquad": 3})
        self.assertIsInstance(err, P.StepFailed)
        self.assertIn("jaquad rc=3", str(err))

    def test_an_interrupt_stops_every_converter_and_removes_the_locks(self):
        # review finding: Colab 'Interrupt execution' (KeyboardInterrupt) used to leave the converters running behind the next 'Run all'
        started, peak, err, procs = self._scheduler(2, raise_in_sleep=KeyboardInterrupt())
        self.assertIsInstance(err, KeyboardInterrupt)
        self.assertEqual(len(procs), 2)
        self.assertTrue(all(p.stopped == ["terminate"] for p in procs))
        self.assertEqual(self.locks_left, [])

    def test_a_second_converter_for_the_same_job_is_refused_while_the_first_is_alive(self):
        from unittest import mock
        with tempfile.TemporaryDirectory() as td:
            cfg = P.Cfg(root=td, work=str(Path(td) / "w"), parallel=1, poll_s=0, log=lambda s: None)
            lockdir = Path(td) / "w" / "locks"
            lockdir.mkdir(parents=True)
            (lockdir / "convert_toolace_full.lock").write_text(str(os.getpid()), encoding="utf-8")          # this very process is 'alive'
            job = P.new_job("toolace", out_root=cfg.conv_dir("toolace").parent)
            with mock.patch.object(P.subprocess, "Popen", side_effect=AssertionError("must not start")):
                with self.assertRaises(P.StepFailed) as cm:
                    P._run_window(cfg, [job], None, smoke=False)
            self.assertIn("still running", str(cm.exception))
            (lockdir / "convert_toolace_full.lock").write_text("0", encoding="utf-8")                         # a dead / stale lock does not block
            self.assertFalse(P._pid_alive(0))
            self.assertTrue(P._pid_alive(os.getpid()))

    def test_waves_publish_what_is_done_and_unmount_drive_only_around_the_converters(self):
        events = []
        waves = []
        started, peak, err, _ = self._scheduler(2, wave_size=2, hooks=(lambda: events.append("detach"), lambda: events.append("attach")),
                                                on_wave=lambda ok, res: (waves.append([j["id"] for j in ok]), events.append("publish")))
        self.assertIsNone(err)
        self.assertEqual(waves, [["coding", "tengentoppa"], ["hotpot", "wizard"], ["jaquad", "toolace"]])
        self.assertEqual(events, ["detach", "attach", "publish"] * 3)               # unmounted while converting, mounted again before anything is published
        self.assertEqual(peak, 2)

    def test_hooks_reattach_drive_even_when_a_converter_fails_and_the_finished_ones_are_still_published(self):
        events, waves = [], []
        started, peak, err, _ = self._scheduler(2, {"tengentoppa": 3}, wave_size=2, hooks=(lambda: events.append("detach"), lambda: events.append("attach")),
                                                on_wave=lambda ok, res: waves.append([j["id"] for j in ok]))
        self.assertIsInstance(err, P.StepFailed)
        self.assertIn("tengentoppa rc=3", str(err))
        self.assertEqual(events, ["detach", "attach"])                              # re-mounted although the window failed
        self.assertEqual(waves, [["coding"]])                                       # the converter that finished is handed over before the step fails

    def test_detach_and_attach_come_together(self):
        with self.assertRaises(ValueError):
            P.Cfg(drive_detach=lambda: None)

    def test_verify_acquired_fails_loudly_when_a_dataset_is_missing(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = P.Cfg(root=td, log=lambda s: None, only=("jaquad",))
            with self.assertRaises(P.StepFailed):
                P.verify_acquired(cfg)

    def test_a_rerun_of_the_smoke_step_returns_the_saved_estimates_instead_of_being_refused(self):
        # review round 3 (blocker): cell 7a measured seconds, saved estimate_minutes through write-once and a re-run (new runtime, 'Run all' again)
        # measured slightly different seconds: 'refusing to overwrite state/step7a_smoke.json' - and 'Run all' stopped before 7b
        from unittest import mock
        with tempfile.TemporaryDirectory() as td:
            def make():
                c = P.Cfg(root=td, work=str(Path(td) / "w"), only=("coding",), check_disk=False, log=lambda s: None)
                Path(c.work).mkdir(exist_ok=True)
                c.vocab_file.write_text("{}", encoding="utf-8")
                return c
            first, second = make(), make()
            saved = {"estimate_minutes": {"coding": 3.4}, "est_converted_bytes": 10, "est_drive_bytes": 20}
            P.save_state(first, "step7a_smoke", saved)
            with self.assertRaises(P.StepFailed):                                     # the write-once rule itself is unchanged: other numbers are refused
                P.save_state(second, "step7a_smoke", dict(saved, estimate_minutes={"coding": 3.5}))
            with mock.patch.object(P, "_run_converters", side_effect=AssertionError("the converters were started again")), \
                    mock.patch.object(P, "ensure_sandbox_ok", side_effect=AssertionError("the sandbox self-test ran for a finished step")):
                r = P.step_convert_smoke(second)
            self.assertTrue(r["pass"])
            self.assertIn("already done", r["detail"])
            self.assertEqual((r["estimate_minutes"], r["est_drive_bytes"]), ({"coding": 3.4}, 20))

    def test_a_trial_run_with_ONLY_writes_to_its_own_output_folder(self):
        # review round 3: a partial run and a later full run with the same RUN_SUFFIX collided at cell 5 ('refusing to overwrite step5_acquire.json')
        full, trial = P.Cfg(root="/r"), P.Cfg(root="/r", only=("toolace", "jaquad"))
        self.assertEqual(full.out.name, P.OUT_NAME)
        self.assertEqual(trial.out.name, P.OUT_NAME + "_only_jaquad-toolace")                 # in KEYS order
        self.assertEqual(P.Cfg(root="/r", only=("toolace",), suffix="_2").out.name, P.OUT_NAME + "_2_only_toolace")
        self.assertEqual(P.Cfg(root="/r", only=P.KEYS).out, full.out)                          # all six = a full run
        self.assertEqual((trial.rows, trial.stage_a_drive("jaquad")), (full.rows, full.stage_a_drive("jaquad")))     # converted / final rows are shared: reused

    def test_only_the_coding_dataset_needs_the_sandbox(self):
        # review round 3: a failing sandbox self-test stopped all six datasets, although only 'coding' ever runs code
        from unittest import mock
        self.assertFalse(P.Cfg(only=("jaquad", "hotpot", "toolace", "wizard", "tengentoppa")).needs_sandbox)
        self.assertTrue(P.Cfg(only=("coding",)).needs_sandbox)
        self.assertTrue(P.Cfg().needs_sandbox)
        self.assertFalse(P.Cfg(run_sandbox=False).needs_sandbox)
        cfg = P.Cfg(root="/r", work="/w")
        for key in P.KEYS:
            cmd = P._converter_cmd(cfg, P.new_job(key, out_root=Path("/o")), None)
            self.assertEqual("--no-sandbox" in cmd, key != "coding", key)
        strict = P.Cfg(root="/r", work="/w", run_sandbox=True, require_os_containment=True)
        self.assertNotIn("--no-sandbox", P._converter_cmd(strict, P.new_job("coding", out_root=Path("/o")), None))
        self.assertIn("--require-os-containment", P._converter_cmd(strict, P.new_job("coding", out_root=Path("/o")), None))
        self.assertIn("--no-sandbox", P._converter_cmd(P.Cfg(root="/r", work="/w", run_sandbox=False), P.new_job("coding", out_root=Path("/o")), None))
        with tempfile.TemporaryDirectory() as td:
            no_code = P.Cfg(root=td, work=str(Path(td) / "w"), only=("jaquad", "toolace"), log=lambda s: None)
            with mock.patch.object(P.checks, "sandbox_selftest", side_effect=AssertionError("the self-test must not run")):
                r = P.step_sandbox_selftest(no_code)
            self.assertTrue(r["pass"])
            self.assertIn("not needed", r["detail"])

    def test_the_drive_probe_remounts_a_few_times_and_says_what_to_expect(self):
        logs, events = [], []
        cfg = P.Cfg(log=logs.append, drive_detach=lambda: events.append("detach"), drive_attach=lambda: events.append("attach"),
                    drive_sync=lambda: events.append("sync"))
        r = P.step_drive_probe(cfg)
        self.assertTrue(r["pass"])
        self.assertEqual(events, ["sync", "sync", "detach", "attach"])
        text = "\n".join(logs)
        self.assertIn("authorisation", text)
        self.assertIn(str(P.expected_remounts(cfg)), text)
        self.assertGreaterEqual(P.expected_remounts(cfg), 15)
        none = P.Cfg(log=logs.append)
        self.assertTrue(P.step_drive_probe(none)["pass"])                                    # the hooks are off: nothing is probed

    def test_verify_acquired_hashes_the_dev_validation_and_card_files_after_the_remount(self):
        if pa is None:
            self.skipTest("pyarrow missing")
        with tempfile.TemporaryDirectory() as td:
            root = build_root(Path(td))
            cfg = P.Cfg(root=str(root), work=str(Path(td) / "w"), check_disk=False, log=lambda s: None)
            self.assertEqual(sorted(P.verify_acquired(cfg)), sorted(P.KEYS))
            dev = root / A.EXTERNAL_DIRNAME / "do_not_train_dev" / "SkelterLabsInc__JaQuAD" / "data" / "dev" / "jaquad_dev_0000.json"
            data = dev.read_bytes()
            dev.write_bytes(data[:-3] + b"XYZ")                                              # same size, other bytes (Drive lost / mangled the write)
            with self.assertRaises(P.StepFailed) as cm:
                P.verify_acquired(cfg)
            self.assertIn("jaquad_dev_0000.json", str(cm.exception))
            dev.write_bytes(data)
            card = root / A.EXTERNAL_DIRNAME / "SkelterLabsInc__JaQuAD" / "README.md"
            cdata = card.read_bytes()
            card.write_bytes(cdata[:-2] + b"ZZ")
            with self.assertRaises(P.StepFailed):
                P.verify_acquired(cfg)
            card.write_bytes(cdata)
            self.assertTrue(P.verify_acquired(cfg))

    def test_converter_tag_covers_the_parameters_that_change_converted_rows(self):
        with tempfile.TemporaryDirectory() as td:
            a = P.Cfg(root=td, work=str(Path(td) / "a"))
            for kw in ({"max_chars": 1200}, {"python_allow_unverified": True}, {"run_sandbox": False}):
                self.assertNotEqual(P.Cfg(root=td, work=str(Path(td) / "b"), **kw).converter_tag(), a.converter_tag(), kw)
            for kw in ({"near_threshold": 0.9}, {"ngram": 11}, {"suffix": "_3"}, {"parallel": 1}):                  # filter / run parameters do not
                self.assertEqual(P.Cfg(root=td, work=str(Path(td) / "b"), **kw).converter_tag(), a.converter_tag(), kw)

    def test_a_refused_stage_a_file_does_not_send_the_owner_to_RUN_SUFFIX(self):
        from unittest import mock
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            (td / "a.txt").write_text("one", encoding="utf-8")
            (td / "b.txt").write_text("two", encoding="utf-8")
            P.publish_file(td / "a.txt", td / "d" / "a.txt")
            with self.assertRaises(P.StepFailed) as cm:
                P.publish_file(td / "b.txt", td / "d" / "a.txt", hint="rename the folder by hand")
            self.assertIn("rename the folder by hand", str(cm.exception))
            self.assertNotIn("RUN_SUFFIX", str(cm.exception))
            cfg = P.Cfg(root=str(td / "drive"), work=str(td / "w"), only=("toolace",), log=lambda s: None)
            src = cfg.conv_dir("toolace")
            src.mkdir(parents=True)
            (src / "manifest.json").write_text(json.dumps({"provenance": {"commit_sha": "c"}}), encoding="utf-8")
            calls = []
            with mock.patch.object(P, "publish_file", side_effect=lambda s, d, hint=None: calls.append(hint) or "created"), \
                    mock.patch.object(P.Cfg, "raw_binding", return_value={"commit_sha": "c", "train_sha256": []}):
                P.publish_stage_a(cfg, "toolace", SimpleNamespace(sha256_chars="s"))
            self.assertTrue(calls and all(h and "by hand" in h and "RUN_SUFFIX" not in h.split("RAW_SUFFIX")[0] for h in calls), calls)

    def test_7b_needs_the_7a_numbers_when_it_has_to_convert(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = P.Cfg(root=td, work=str(Path(td) / "w"), only=("jaquad",), check_disk=False, log=lambda s: None)
            Path(cfg.work).mkdir()
            cfg.vocab_file.write_text("{}", encoding="utf-8")
            P.save_state(cfg, "step6c_raw_checks", {"datasets": {"jaquad": {"n_raw": 5, "bad_lines": 0}}, "mismatches": [], "binding": {}})       # 6c ran (7b needs it first)
            saved = P._CACHE.get("vocab")
            P._CACHE["vocab"] = saved or object()
            try:
                with self.assertRaises(P.StepFailed) as cm:
                    P.step_convert_full(cfg)
            finally:
                if saved is None:
                    P._CACHE.pop("vocab", None)
            self.assertIn("7a", str(cm.exception))

    def test_7a_and_7b_need_the_6c_record_counts(self):
        # review round 4: both steps accepted a missing 6c state and silently switched off the record-count check, the sharding decision and the
        # guard for the time of an unsharded conversion
        from unittest import mock
        with tempfile.TemporaryDirectory() as td:
            cfg = P.Cfg(root=td, work=str(Path(td) / "w"), only=("jaquad",), check_disk=False, log=lambda s: None)
            Path(cfg.work).mkdir()
            cfg.vocab_file.write_text("{}", encoding="utf-8")
            saved = P._CACHE.get("vocab")
            P._CACHE["vocab"] = saved or object()
            try:
                with mock.patch.object(P, "_run_converters", side_effect=AssertionError("a converter was started without the 6c counts")):
                    for fn in (P.step_convert_smoke, P.step_convert_full):
                        with self.assertRaises(P.StepFailed) as cm:
                            fn(cfg)
                        self.assertIn("run step 6c first", str(cm.exception), fn.__name__)
                    P.save_state(cfg, "step6c_raw_checks", {"datasets": {"jaquad": {"n_raw": 5, "bad_lines": 0}}, "mismatches": [], "binding": {}})
                    with self.assertRaises(P.StepFailed) as cm:                                    # with the counts the next requirement (the 7a numbers) is reached
                        P.step_convert_full(cfg)
                    self.assertIn("7a", str(cm.exception))
            finally:
                if saved is None:
                    P._CACHE.pop("vocab", None)

    def test_a_balanced_manifest_that_hides_sandbox_failures_fails_the_check(self):
        # review round 4 (major): a sandbox failure in the middle of a long run became 'sandbox_failed:crash:...' rejects; the manifest stayed balanced
        # and 7a / 7b passed with the verified python rows gone.  Second net behind convert.SandboxUnavailable.
        cfg = P.Cfg(root="/r", work="/w", only=("coding",), require_os_containment=False)
        good = {"backends": {"sandbox": C.SANDBOX_BACKEND, "sandbox_run_as": "nobody"},
                "rejects": {"sandbox_failed:exception:ZeroDivisionError: division by zero": 3, "sandbox_failed:timeout:5.0s": 2, "sandbox_failed:output_limit:x": 1,
                            "sandbox_failed:static:import:os": 1, "static:import:os": 4, "no_stored_output_to_corroborate": 1, "stored_output_mismatch": 2}}
        P.check_sandbox_backend("coding", good, cfg)                                         # ordinary code failures are fine
        self.assertEqual(P.sandbox_infrastructure_rejects(good), {})
        for reason in ("sandbox_failed:crash:spawn:OSError: [Errno 28] No space left on device", "sandbox_failed:crash:returncode=-9:",
                       "sandbox_failed:spawn:x", "sandbox_failed:checks_error:SandboxError", "sandbox_failed:checks_error:OSError"):
            bad = dict(good, rejects=dict(good["rejects"], **{reason: 7}))
            self.assertEqual(P.sandbox_infrastructure_rejects(bad), {reason: 7})
            with self.assertRaises(P.StepFailed) as cm:
                P.check_sandbox_backend("coding", bad, cfg)
            self.assertIn("SANDBOX failed", str(cm.exception))
            self.assertIn(reason, str(cm.exception))                                          # named ...
            self.assertIn("7 row(s)", str(cm.exception))                                      # ... and counted
        two = dict(good, rejects={"sandbox_failed:crash:spawn:OSError": 5, "sandbox_failed:checks_error:SandboxError": 2})
        with self.assertRaises(P.StepFailed) as cm:
            P.check_sandbox_backend("coding", two, cfg)
        self.assertIn("7 row(s)", str(cm.exception))
        self.assertIn("sandbox_failed:crash:spawn:OSError x 5", str(cm.exception))
        self.assertIn("sandbox_failed:checks_error:SandboxError x 2", str(cm.exception))
        with self.assertRaises(P.StepFailed):                                                 # also for a run without the sandbox switch / another dataset's manifest
            P.check_sandbox_backend("jaquad", two, P.Cfg(root="/r", work="/w", run_sandbox=False))

    def _balanced_manifest_with(self, rejects):
        return {"balanced": True, "n_in": 10, "n_out": 4, "reject_total": 6, "limit": None, "rejects": rejects, "splits": {}, "files": {},
                "backends": {"sandbox": C.SANDBOX_BACKEND, "sandbox_run_as": "nobody"}, "keep_rate": 0.4}

    def test_7a_and_7b_fail_when_the_manifest_hides_sandbox_failures(self):
        from unittest import mock
        with tempfile.TemporaryDirectory() as td:
            cfg = P.Cfg(root=td, work=str(Path(td) / "w"), only=("coding",), check_disk=False, require_os_containment=False, log=lambda s: None)
            Path(cfg.work).mkdir()
            cfg.vocab_file.write_text("{}", encoding="utf-8")
            P.save_state(cfg, "step6c_raw_checks", {"datasets": {"coding": {"n_raw": 10, "bad_lines": 0}}, "mismatches": [], "binding": {}})
            man = self._balanced_manifest_with({"sandbox_failed:crash:spawn:OSError: [Errno 28] No space left on device": 6})
            saved = P._CACHE.get("vocab")
            P._CACHE["vocab"] = saved or object()
            try:
                with mock.patch.object(P, "ensure_sandbox_ok", return_value={}), mock.patch.object(P, "_convert_ck", return_value=None), \
                        mock.patch.object(P, "_check_manifest", return_value=man), \
                        mock.patch.object(P, "_run_converters", return_value={"coding": {"rc": 0, "seconds": 1.0, "log": "x"}}):
                    with self.assertRaises(P.StepFailed) as cm:
                        P.step_convert_smoke(cfg)                                                  # 7a
                    self.assertIn("6 row(s)", str(cm.exception))
                    self.assertIn("sandbox_failed:crash:spawn", str(cm.exception))
                    P.save_state(cfg, "step7a_smoke", {"estimate_minutes": {}, "est_converted_bytes": 1, "est_drive_bytes": 1})
                    d = cfg.conv_dir("coding")                                                     # rows of an earlier run (made by an older converter) are 'resumed'
                    d.mkdir(parents=True)
                    (d / "manifest.json").write_text("{}", encoding="utf-8")
                    with self.assertRaises(P.StepFailed) as cm:
                        P.step_convert_full(cfg)                                                   # 7b
                    self.assertIn("6 row(s)", str(cm.exception))
                    self.assertIn("sandbox_failed:crash:spawn", str(cm.exception))
            finally:
                if saved is None:
                    P._CACHE.pop("vocab", None)

    def test_a_shard_manifest_that_hides_sandbox_failures_is_refused_before_it_is_published(self):
        from unittest import mock
        with tempfile.TemporaryDirectory() as td:
            cfg = P.Cfg(root=td, work=str(Path(td) / "w"), only=("coding",), check_disk=False, require_os_containment=False, shard_records=3, log=lambda s: None)
            Path(cfg.work).mkdir()
            cfg.vocab_file.write_text("{}", encoding="utf-8")
            P.save_state(cfg, "step6c_raw_checks", {"datasets": {"coding": {"n_raw": 10, "bad_lines": 0}}, "mismatches": [], "binding": {}})
            P.save_state(cfg, "step7a_smoke", {"estimate_minutes": {}, "est_converted_bytes": 1, "est_drive_bytes": 1})
            man = self._balanced_manifest_with({"sandbox_failed:checks_error:SandboxError": 2})
            published = []

            def fake_run(cfg_, jobs, smoke, limit=None, wave_size=None, on_wave=None):
                res = {j["id"]: {"rc": 0, "seconds": 1.0, "log": "x"} for j in jobs}
                on_wave(jobs[:1], res)
                return res
            saved = P._CACHE.get("vocab")
            P._CACHE["vocab"] = saved or object()
            try:
                with mock.patch.object(P, "ensure_sandbox_ok", return_value={}), mock.patch.object(P, "_check_manifest", return_value=man), \
                        mock.patch.object(P, "publish_shard", side_effect=lambda *a, **k: published.append(a)), mock.patch.object(P, "restore_shard", return_value=False), \
                        mock.patch.object(P, "_run_converters", fake_run):
                    with self.assertRaises(P.StepFailed) as cm:
                        P.step_convert_full(cfg)
                self.assertIn("2 row(s)", str(cm.exception))
                self.assertEqual(published, [])
            finally:
                if saved is None:
                    P._CACHE.pop("vocab", None)

    def test_the_selftest_step_does_not_count_an_information_line_as_a_pass(self):
        from unittest import mock
        with tempfile.TemporaryDirectory() as td:
            cfg = P.Cfg(root=td, work=str(Path(td) / "w"), only=("coding",), check_disk=False, require_os_containment=False, log=lambda s: None,
                        drive_detach=lambda: None, drive_attach=lambda: None)
            fake = {"ok": True, "backend": C.SANDBOX_BACKEND, "run_as": "nobody",
                    "checks": [{"name": "a", "ok": True, "detail": ""}, {"name": "b", "ok": True, "detail": ""},
                               {"name": "info: what the child could see of Drive at 3d time (x)", "ok": True, "detail": "it saw nothing", "info": True}]}
            saved = P._CACHE.get("sandbox")
            try:
                with mock.patch.object(P.checks, "sandbox_selftest", return_value=fake):
                    r = P.step_sandbox_selftest(cfg)
                self.assertIn("2 checks passed", r["detail"])
                self.assertIn("1 information line", r["detail"])
                rec = P._CACHE["sandbox"]
                self.assertEqual((rec["n_checks"], rec["n_info"]), (2, 1))
                self.assertEqual([c.get("info", False) for c in rec["checks"]], [False, False, True])
            finally:
                if saved is None:
                    P._CACHE.pop("sandbox", None)
                else:
                    P._CACHE["sandbox"] = saved

    def test_cfg_paths_use_the_new_folder_names_and_the_suffix(self):
        cfg = P.Cfg(root="/r", suffix="_2")
        self.assertEqual(cfg.stage_a_drive("coding").parent.parent, Path("/r/external_data_v49_stageA"))
        self.assertEqual(cfg.out, Path("/r/v49_acquire_20261109_2"))
        self.assertEqual(cfg.rows, Path("/r/external_data_v49_rows_2"))
        self.assertEqual(cfg.ext, Path("/r/external_data_v49"))
        self.assertEqual(P.Cfg().acq_only(), None)
        self.assertEqual(P.Cfg(only=("hotpot", "wizard")).acq_only(), ["hotpotqa", "wizardlm"])
        with self.assertRaises(ValueError):
            P.Cfg(only=("nope",))


if __name__ == "__main__":
    unittest.main(verbosity=2)
