"""CPU-only tests for eval_v48_oneshot.py (no GPU, no Drive, no model).

They run the REAL v47 question loop (eval_candidate.run_questions + live_logging) with a fake generator, the real
v47 checkpoint validation and lock check, and a recording stand-in for the subprocess stages, so the parts that
can fail on a paid GPU (case binding, scoring, resume, stage order, stop-on-failure) are exercised here first.
"""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
import zipfile

HERE = Path(__file__).resolve().parent
V47 = HERE.parents[1] / "research/v47_practical_20261107"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(V47))

import eval_v48_oneshot as M  # noqa: E402
import eval_candidate as E  # noqa: E402
import frozen_eval as frozen  # noqa: E402
import train_candidate as OLD  # noqa: E402
import train_from_40500 as T  # noqa: E402

ALL = {M.suite_name(s, d): M.load_cases(s, d) for s in M.SPLITS for d in M.DOMAINS}
BY_PROMPT = {c["prompt"]: c for cases in ALL.values() for c in cases}


def native(text, calls=0, eos=True):
    return {"model_generated_text": text, "output_text": text, "ended_with_eos": eos,
            "python_invocations": calls, "python_trace": [{}] * calls,
            "search_invocations": 0, "search_trace": []}


def perfect(case):
    return native(case["target"], len(case.get("tool_steps", [])))


def wrong(case):
    return native("\n答え: 99999。", 0)


class Cases(unittest.TestCase):
    def test_eight_suites_of_128_unique_rows(self):
        self.assertEqual(len(ALL), 8)
        for suite, rows in ALL.items():
            self.assertEqual(len(rows), 128, suite)
        ids = [c["id"] for rows in ALL.values() for c in rows]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(sum(len(r) for s, r in ALL.items() if s.startswith("v48_dev_")), 512)
        self.assertEqual(sum(len(r) for s, r in ALL.items() if s.startswith("v48_unseen_eval_")), 512)

    def test_tampered_file_is_refused(self):
        with tempfile.TemporaryDirectory() as td:
            data = Path(td) / "data"
            shutil.copytree(M.DATA, data)
            path = data / "python_dev.jsonl"
            path.write_bytes(path.read_bytes().replace("答え".encode(), "答X".encode(), 1))
            with self.assertRaisesRegex(ValueError, "differs from its manifest"):
                M.load_cases("dev", "python", data)

    def test_wrong_split_rows_are_refused(self):
        with tempfile.TemporaryDirectory() as td:
            data = Path(td) / "data"
            shutil.copytree(M.DATA, data)
            manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
            (data / "python_dev.jsonl").write_bytes((data / "python_unseen_eval.jsonl").read_bytes())
            manifest["files"]["python_dev.jsonl"]["sha256"] = M.sha256_file(data / "python_dev.jsonl")
            (data / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "expected"):
                M.load_cases("dev", "python", data)


class Scoring(unittest.TestCase):
    def test_every_reference_target_scores_correct_with_eos(self):
        count = 0
        for suite, rows in ALL.items():
            for row in rows:
                result = M.score_v48(row, perfect(row))
                self.assertIs(result["terminal_exact"], True, (suite, row["id"], result))
                count += 1
        self.assertEqual(count, 1024)

    def test_every_target_scores_correct_as_the_model_really_samples_it(self):
        """The runtime injects the tool observation; model_text never contains it (found on the first A100 run:
        a perfect python answer scored 0/128 because the closing fence is followed directly by the answer)."""
        count = 0
        for suite, rows in ALL.items():
            for row in rows:
                text = row["target"]
                for step in row.get("tool_steps", []):
                    assert step["observation"] in text
                    text = text.replace(step["observation"], "", 1)
                gen = native(text, len(row.get("tool_steps", [])))
                result = M.score_v48(row, gen)
                self.assertIs(result["terminal_exact"], True, (suite, row["id"], text[-90:], result))
                count += 1
        self.assertEqual(count, 1024)

    def test_python_rows_are_tool_grounded_only_when_python_ran(self):
        row = ALL["v48_dev_python"][0]
        self.assertTrue(M.score_v48(row, perfect(row))["tool_grounded_exact"])
        no_tool = native("\n答え: %s。" % row["reference_final_answer"], 0)
        result = M.score_v48(row, no_tool)
        self.assertTrue(result["terminal_exact"])
        self.assertFalse(result["tool_grounded_exact"])

    def test_rejections(self):
        row = ALL["v48_dev_logic"][0]
        good = row["reference_final_answer"]
        self.assertFalse(M.score_v48(row, native("\n答え: 99999。"))["terminal_exact"])
        self.assertFalse(M.score_v48(row, native("\n答え: %s。" % good, eos=False))["terminal_exact"])
        self.assertTrue(M.score_v48(row, native("\n答え: %s。" % good, eos=False))["exact_final"])
        self.assertIsNone(M.final_answer("```python\n答え: %s。\n```" % good))          # inside code
        self.assertEqual(M.final_answer("```python\nprint(1)\n```答え: %s。実行結果を確認しました。" % good), good)  # no newline after the fence
        self.assertIsNone(M.final_answer("```python\nprint(1)\n```答え: %s。```" % good))   # odd fence count
        self.assertIsNone(M.final_answer("説明\n```python\nprint(1)\n答え: %s。" % good))  # unterminated fence
        self.assertEqual(M.final_answer("答え: 1。\n考え直す\n答え: %s。" % good), good)    # last answer wins
        self.assertIsNone(M.final_answer("答えは%s" % good))                               # not the required form
        self.assertEqual(M.final_answer("答え：%s。" % good), good)                        # full-width colon


LIMIT = 2     # the real question loop fsyncs several files per question (seconds each on a slow disk)


class QuestionLoop(unittest.TestCase):
    def run_sets(self, out, generator, **kw):
        return M.run_sets(E, frozen, generator, Path(out), checkpoint_id="x:1", checkpoint_sha="a" * 64,
                          runtime_sha="b" * 64, corpus_sha="c" * 64, limit=LIMIT, **kw)

    def test_perfect_then_resume_never_regenerates(self):
        with tempfile.TemporaryDirectory() as td:
            first = self.run_sets(td, perfect)
            self.assertEqual(len(first), 8)
            for suite, status in first.items():
                self.assertEqual((status["correct"], status["answered"], status["state"]), (LIMIT, LIMIT, "complete"), suite)
            calls = []
            self.run_sets(td, lambda case: calls.append(case["id"]) or perfect(case))
            self.assertEqual(calls, [], "completed suites must be verified and skipped, not regenerated")

    def test_only_runs_the_named_suites(self):
        with tempfile.TemporaryDirectory() as td:
            only = ["v48_dev_python", "v48_unseen_eval_python"]
            results = self.run_sets(td, perfect, only=only)
            self.assertEqual(sorted(results), sorted(only))
            self.assertEqual(sorted(p.name for p in Path(td).iterdir()),
                             sorted(s + "__generate" for s in only))

    def test_breakdown_explains_a_zero(self):
        """Right answers without EOS must show up as exact_final, not vanish into terminal_exact = 0."""
        def right_but_no_eos(case):
            return native(case["target"], len(case.get("tool_steps", [])), eos=False)
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            self.run_sets(a, right_but_no_eos)
            self.run_sets(b, perfect)
            summary = M.summarize_sets(Path(a), Path(b))
            n = 8 * LIMIT
            self.assertEqual(summary["totals"]["dev_before"], {"correct": 0, "answered": 4 * LIMIT})
            d = summary["detail_totals"]
            self.assertEqual(d["before_exact_final"], n)
            self.assertEqual(d["before_ended_with_eos"], 0)
            self.assertEqual(d["before_format_valid"], n)
            self.assertEqual(d["after_ended_with_eos"], n)
            self.assertIn("EOSを問わない", M.sets_markdown(summary))

    def test_wrong_generator_scores_zero_but_completes(self):
        with tempfile.TemporaryDirectory() as td:
            for suite, status in self.run_sets(td, wrong).items():
                self.assertEqual((status["correct"], status["state"]), (0, "complete"), suite)

    def test_interrupted_run_resumes_where_it_stopped(self):
        with tempfile.TemporaryDirectory() as td:
            seen = []

            def dies_after_five(case):
                seen.append(case["id"])
                if len(seen) > 5:
                    raise KeyboardInterrupt
                return perfect(case)

            with self.assertRaises(KeyboardInterrupt):
                self.run_sets(td, dies_after_five)
            again = []
            self.run_sets(td, lambda case: again.append(case["id"]) or perfect(case))
            self.assertEqual(len(set(again) & set(seen[:5])), 0, "the five saved answers are not generated again")
            self.assertEqual(len(again), 8 * LIMIT - 5)

    def test_a_changed_checkpoint_cannot_reuse_old_raw_files(self):
        with tempfile.TemporaryDirectory() as td:
            self.run_sets(td, perfect)
            with self.assertRaises(Exception):
                M.run_sets(E, frozen, perfect, Path(td), checkpoint_id="x:1", checkpoint_sha="d" * 64,
                           runtime_sha="b" * 64, corpus_sha="c" * 64, limit=LIMIT)

    def test_summary_table_before_after(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            self.run_sets(a, wrong)
            self.run_sets(b, perfect)
            summary = M.summarize_sets(Path(a), Path(b))
            n = 4 * LIMIT
            self.assertEqual(summary["totals"]["dev_before"], {"correct": 0, "answered": n})
            self.assertEqual(summary["totals"]["unseen_eval_after"], {"correct": n, "answered": n})
            text = M.sets_markdown(summary)
            self.assertIn(f"{n}/{n} (100.0%)", text)
            self.assertIn(f"0/{n} (0.0%)", text)

    def test_mirrored_run_publishes_each_suite_and_resumes_from_drive(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "drive" / "step61500"
            kw = dict(checkpoint_id="x:1", checkpoint_sha="a" * 64, runtime_sha="b" * 64, corpus_sha="c" * 64,
                      limit=LIMIT)
            local = M.mirror_root(out)
            self.addCleanup(shutil.rmtree, local, True)
            shutil.rmtree(local, True)
            first = M.run_sets_mirrored(E, frozen, perfect, out, **kw)
            self.assertEqual(len(first), 8)
            for suite in ALL:
                self.assertTrue((out / (suite + "__generate") / "status.json").is_file(), suite)
                self.assertFalse((out / (suite + "__generate") / "logger_active.json").exists(), suite)
            # a new VM: the local copy is gone, the Drive copy alone must be enough
            shutil.rmtree(local)
            calls = []
            M.run_sets_mirrored(E, frozen, lambda c: calls.append(c["id"]) or perfect(c), out, **kw)
            self.assertEqual(calls, [])
            # and the Drive copy is a complete, verifiable record (what compare_results reads)
            import live_logging
            for suite in ALL:
                self.assertEqual(live_logging.read_logged_run(out / (suite + "__generate"))["status"]["state"],
                                 "complete")

    def test_rescore_fixes_scores_from_saved_raw_without_touching_the_logs(self):
        """A run scored with a wrong scorer can be re-scored from raw/*.json; the logs stay byte-identical."""
        def as_sampled(case):
            text = case["target"]
            for step in case.get("tool_steps", []):
                text = text.replace(step["observation"], "", 1)
            return native(text, len(case.get("tool_steps", [])))
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for folder in ("step61500", "step40500_v48sets"):
                local = root / folder
                M.run_sets(E, frozen, as_sampled, local, checkpoint_id="x:1", checkpoint_sha="a" * 64,
                           runtime_sha="b" * 64, corpus_sha="c" * 64, limit=LIMIT)
            before_bytes = {p: p.read_bytes() for p in root.rglob("status.json")}
            summary = M.rescore(root, step=61500, expected=LIMIT)
            self.assertEqual(summary["totals"]["dev_after"], {"correct": 4 * LIMIT, "answered": 4 * LIMIT})
            self.assertEqual(summary["detail_totals"]["after_tool_grounded_exact"], 2 * LIMIT)   # the python suites
            self.assertTrue((root / "v48_sets_rescored.md").is_file())
            self.assertEqual(before_bytes, {p: p.read_bytes() for p in root.rglob("status.json")})

    def test_missing_side_is_unmeasured_not_zero(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            self.run_sets(b, perfect)
            summary = M.summarize_sets(Path(a), Path(b))
            self.assertIsNone(summary["totals"]["dev_before"])
            self.assertIn("未評価", M.sets_markdown(summary))


class FakeFrozen:
    def build_official_tier_command(self, *, study, run, step, checkpoint_sha, output, preflight_only=False, **kw):
        return ["STAGE", run, str(step), checkpoint_sha, str(output)] + (["--preflight-only"] if preflight_only else [])


def make_run(study, run, step, sha):
    training_sha = ("%064x" % (hash(run) & 0xffffffff))
    folder = Path(study) / run
    (folder / "checkpoints").mkdir(parents=True)
    (folder / f"checkpoints/checkpoint_step_{step}.pt").write_bytes(b"x")
    (folder / f"checkpoints/receipt_step_{step}.json").write_text(json.dumps(
        {"step": step, "checkpoint_sha256": sha, "training_data_sha256": training_sha}), encoding="utf-8")
    (folder / "training_saved.json").write_text(json.dumps(
        {"step": step, "full_model": True, "training_data_sha256": training_sha}), encoding="utf-8")
    (folder / "input_manifest.json").write_text(json.dumps({"output_sha256": training_sha}), encoding="utf-8")


class Orchestration(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.addCleanup(self.td.cleanup)
        self.study = Path(self.td.name)
        make_run(self.study, "v48_A100_from40500_20261108T045428Z", 61500, "1" * 64)
        make_run(self.study, M.BASE_RUN, 40500, "2" * 64)
        (self.study / "v48_A100_from40500_20261108T044231Z").mkdir()    # failed launch: no training_saved.json
        self.commands, self.reports = [], []
        self.fail_label = None

    def args(self, **kw):
        base = dict(study=self.study, run="auto", step=61500, stamp="S1", resume=False, aux=False,
                    full_knowledge=False, code_dir=self.study / "code", shards=4)
        base.update(kw)
        return types.SimpleNamespace(**base)

    def run_child(self, command, label):
        self.commands.append((label, command))
        return 2 if label == self.fail_label else 0

    def run_group(self, group):
        codes = {}
        for label, command in group:
            self.commands.append((label, command))
            codes[label] = 2 if label == self.fail_label else 0
            if label == "original_tier_1_8" and codes[label] == 0:
                out = Path(command[-1])
                out.mkdir(parents=True)
                (out / "complete.json").write_text("{}", encoding="utf-8")
        return codes

    def report(self, before, after, output):
        self.reports.append((Path(before), Path(after), Path(output)))
        return {"written": True}

    def go(self, **kw):
        return M.oneshot(self.args(**kw), run_child=self.run_child, run_group=self.run_group,
                         report_originals=self.report, E=E, frozen=FakeFrozen(), OLD=OLD, T=T)

    def labels(self):
        return [label for label, _ in self.commands]

    def test_order_and_arguments(self):
        state = self.go()
        sets = [f"v48_sets_{tag}_{i}" for tag in ("new", "base") for i in range(4)]
        self.assertEqual(self.labels(), ["original_stage_preflight", "original_tier_1_8"] + sets)
        root = self.study / "v48_eval_S1"
        self.assertTrue(state["finished"])
        self.assertEqual(self.commands[0][1][-1], "--preflight-only")
        self.assertEqual(self.commands[0][1][1], "v48_A100_from40500_20261108T045428Z")   # the finished run, not the failed one
        by = dict(self.commands)
        new0, base3 = by["v48_sets_new_0"], by["v48_sets_base_3"]
        self.assertIn("v48_A100_from40500_20261108T045428Z", new0)
        self.assertEqual(new0[new0.index("--step") + 1], "61500")
        self.assertEqual(new0[new0.index("--output") + 1], str(root / "step61500"))
        self.assertEqual(new0[new0.index("--only") + 1], "v48_dev_python,v48_unseen_eval_python")
        self.assertIn(M.BASE_RUN, base3)
        self.assertEqual(base3[base3.index("--output") + 1], str(root / "step40500_v48sets"))
        self.assertEqual(base3[base3.index("--only") + 1], "v48_dev_japanese,v48_unseen_eval_japanese")
        self.assertEqual(by["original_tier_1_8"][-1], str(root / "step61500/original_tiers"))
        self.assertEqual(self.reports, [(self.study / M.BASE_EVAL_DIR, root / "step61500",
                                         root / "reports/original_tiers_vs_40500")])
        self.assertTrue((root / "oneshot_state.json").is_file())

    def test_shards_cover_every_suite_exactly_once(self):
        for shards in (1, 2, 4):
            parts = M.shard_suites(shards)
            self.assertEqual(len(parts), shards)
            flat = [s for p in parts for s in p]
            self.assertEqual(sorted(flat), sorted(M.ALL_SUITES))
        with self.assertRaises(ValueError):
            M.shard_suites(3)

    def test_baseline_sets_are_outside_the_after_root(self):
        """compare_results rglobs the 'after' root; baseline v48 suites inside it would collide with the new ones."""
        self.go()
        root = self.study / "v48_eval_S1"
        self.assertFalse((root / "step61500").joinpath("step40500_v48sets").exists())
        self.assertNotIn(root / "step40500_v48sets", (root / "step61500").parents)

    def test_failed_preflight_stops_before_any_generation(self):
        self.fail_label = "original_stage_preflight"
        with self.assertRaisesRegex(RuntimeError, "original_stage_preflight"):
            self.go()
        self.assertEqual(self.labels(), ["original_stage_preflight"])
        self.assertEqual(self.reports, [])

    def test_a_failed_worker_does_not_hide_the_others_and_blocks_the_report(self):
        self.fail_label = "v48_sets_new_1"
        with self.assertRaisesRegex(RuntimeError, "v48_sets_new_1"):
            self.go()
        self.assertIn("original_tier_1_8", self.labels())      # the sibling stages were still launched
        self.assertEqual(self.reports, [])
        self.assertTrue((self.study / "v48_eval_S1/v48_sets_summary.md").is_file())   # partial table is still written

    def test_failed_original_stage_blocks_the_report(self):
        self.fail_label = "original_tier_1_8"
        with self.assertRaisesRegex(RuntimeError, "original_tier_1_8"):
            self.go()
        self.assertEqual(self.reports, [])

    def test_resume_skips_completed_original_stage(self):
        self.go()
        self.commands.clear()
        self.go(resume=True)
        self.assertEqual(self.labels(), [f"v48_sets_{tag}_{i}" for tag in ("new", "base") for i in range(4)])

    def test_existing_root_needs_resume(self):
        self.go()
        with self.assertRaises(FileExistsError):
            self.go()

    def test_incomplete_original_dir_is_never_overwritten(self):
        (self.study / "v48_eval_S1/step61500/original_tiers").mkdir(parents=True)
        with self.assertRaisesRegex(RuntimeError, "not complete"):
            self.go(resume=True)
        self.assertEqual(self.commands, [])

    def test_live_training_lock_blocks_everything(self):
        run = self.study / "v48_A100_from40500_20261108T045428Z"
        (run / "training_active.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(RuntimeError):
            self.go()
        self.assertEqual(self.commands, [])

    def test_failed_launch_lock_is_ignored_but_kept(self):
        import os, time
        lock = self.study / "v48_A100_from40500_20261108T044231Z/training_active.json"
        lock.write_text("{}", encoding="utf-8")
        old = time.time() - 3600
        os.utime(lock, (old, old))
        self.go()
        self.assertTrue(lock.exists())

    def test_wrong_checkpoint_receipt_stops_on_cpu(self):
        path = self.study / "v48_A100_from40500_20261108T045428Z/training_saved.json"
        saved = json.loads(path.read_text())
        saved["training_data_sha256"] = "0" * 64
        path.write_text(json.dumps(saved), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "disagree"):
            self.go()
        self.assertEqual(self.commands, [])

    def test_aux_is_opt_in(self):
        self.go()
        self.assertNotIn("aux_suites", self.labels())
        self.commands.clear()
        self.go(resume=True, aux=True, full_knowledge=True)
        aux = dict(self.commands)["aux_suites"]
        for flag in ("--resume", "--aux-only", "--full-knowledge"):
            self.assertIn(flag, aux)

    def test_no_run_found(self):
        shutil.rmtree(self.study / "v48_A100_from40500_20261108T045428Z")
        with self.assertRaises(FileNotFoundError):
            self.go()


class CodeZip(unittest.TestCase):
    def test_extract_checks_sha(self):
        with tempfile.TemporaryDirectory() as td:
            z = Path(td) / "code.zip"
            with zipfile.ZipFile(z, "w") as archive:
                archive.writestr("frozen_eval.py", "x=1\n")
            Path(td, "dest").mkdir()
            with self.assertRaisesRegex(RuntimeError, "SHA-256"):
                M.extract_code(z, "0" * 64, Path(td) / "dest")
            work = M.extract_code(z, M.sha256_file(z), Path(td) / "dest")
            self.assertTrue((work / "frozen_eval.py").is_file())


if __name__ == "__main__":
    unittest.main()
