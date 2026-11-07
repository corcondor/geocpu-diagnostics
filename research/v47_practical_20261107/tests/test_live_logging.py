"""CPU checks for incorrect score claims, lost progress, and unsafe comparisons."""

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from live_logging import (ExistingSuiteMonitor, RunLogger, TextProgressParser,
                          comparison_rows, export_comparison, run_with_logging)


def identity(**changes):
    value = dict(suite="Python120", mode="python-enabled", checkpoint_id="baseline",
                 checkpoint_sha256="a" * 64, dataset_sha256="b" * 64,
                 scorer_sha256="c" * 64, runtime_sha256="d" * 64,
                 budget={"max_tokens": 2048, "enable_search": False}, seed=42,
                 metric="terminal_explicit_exact")
    value.update(changes)
    return value


class LiveLoggerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.output = StringIO()

    def tearDown(self):
        self.tmp.cleanup()

    def logger(self, name="run", *, total=5, ident=None, resume=False, heartbeat_s=5):
        return RunLogger(self.root / name, ident or identity(), total=total,
                         resume=resume, stream=self.output, heartbeat_s=heartbeat_s)

    def test_accuracy_and_coverage_keep_invalid_unmeasured_and_errors_distinct(self):
        with self.logger() as log:
            log.result("correct", correct=True)
            log.result("incorrect", correct=False)
            log.result("invalid-format", status="invalid")
            log.result("missing-cad-dependency", status="unmeasured")
            log.result("scorer-error", status="error", detail="KeyError: min")
            s = log.snapshot()
            self.assertEqual((s["processed"], s["answered"], s["correct"]), (5, 3, 1))
            self.assertEqual((s["incorrect"], s["invalid"], s["unmeasured"], s["errors"]), (1, 1, 1, 1))
            self.assertAlmostEqual(s["accuracy"], 1 / 3)
            self.assertAlmostEqual(s["coverage"], 3 / 5)
        self.assertEqual(json.loads((self.root / "run/status.json").read_text())["state"], "failed")

    def test_progress_is_not_a_correct_score_or_a_completed_case(self):
        with self.logger(total=120) as log:
            log.progress(90, 120)
            s = log.snapshot()
            self.assertIsNone(s["accuracy"])
            self.assertEqual(s["correct"], 0)
            self.assertEqual(s["processed"], 0)
            self.assertEqual(s["progress_hint"]["processed"], 90)
        parser = TextProgressParser()
        line = "[2026-11-07T00:01:00] PYTHON_TIR_STEP_28000_PROGRESS 90/120"
        parsed = parser.feed(line)
        self.assertEqual(parsed["processed"], 90)
        self.assertIsNone(parsed["accuracy"])
        self.assertNotIn("correct", parsed)
        self.assertIsNone(parser.feed("EXTENDED_CASE_START 64/74 engineering any"))
        explicit = parser.feed("Progress: 20/200 | Acc: 25.0% (5/20) | SearchCalls: 18/20")
        self.assertEqual((explicit["correct"], explicit["answered"]), (5, 20))

    def test_resume_preserves_results_and_keeps_unfinished_case_pending(self):
        log = self.logger(total=2)
        log.start_case("one")
        log.result("one", correct=True, metrics={"python_calls": 1})
        log.start_case("two")
        status = log.finish(interrupted=True)
        self.assertEqual(status["processed"], 1)
        self.assertEqual(status["state"], "interrupted")
        initial_bytes = (self.root / "run/events.jsonl").read_bytes()
        with self.logger(total=2, resume=True) as resumed:
            self.assertFalse(resumed.result("one", correct=True, metrics={"python_calls": 1}))
            self.assertEqual(resumed.snapshot()["processed"], 1)
            self.assertEqual(resumed.snapshot()["active_cases"], [])
            self.assertTrue(resumed.start_case("two"))
            resumed.result("two", correct=False)
        self.assertTrue((self.root / "run/events.jsonl").read_bytes().startswith(initial_bytes))
        saved = json.loads((self.root / "run/status.json").read_text())
        self.assertEqual((saved["state"], saved["correct"], saved["answered"]), ("complete", 1, 2))
        events = [json.loads(l) for l in (self.root / "run/events.jsonl").read_text().splitlines()]
        resumed_event = next(e for e in events if e["event"] == "run_resumed")
        self.assertEqual(resumed_event["pending_interrupted_cases"], ["two"])
        self.assertEqual(len([e for e in events if e["event"] == "case_result"]), 2)

    def test_changed_settings_and_conflicting_duplicate_are_rejected(self):
        log = self.logger(total=1)
        log.result("one", correct=True)
        with self.assertRaisesRegex(ValueError, "contradictory"):
            log.result("one", correct=False)
        log.finish()
        with self.assertRaisesRegex(ValueError, "cannot resume"):
            self.logger(total=1, resume=True, ident=identity(budget={"max_tokens": 1024}))
        with self.assertRaises(FileExistsError):
            self.logger(total=1)

    def test_heartbeat_records_during_a_long_case(self):
        with self.logger(total=1, heartbeat_s=0.05) as log:
            log.start_case("slow")
            time.sleep(0.18)
            events = [json.loads(l) for l in (self.root / "run/events.jsonl").read_text().splitlines()]
            self.assertGreaterEqual(len([e for e in events if e["event"] == "heartbeat"]), 2)
            saved = json.loads((self.root / "run/status.json").read_text())
            self.assertEqual(saved["active_cases"], ["slow"])
            self.assertEqual(saved["processed"], 0)
            log.result("slow", correct=True)

    def test_exception_is_recorded_without_inventing_wrong_answer(self):
        with self.assertRaisesRegex(RuntimeError, "scorer"):
            with self.logger(total=1) as log:
                log.start_case("unfinished")
                raise RuntimeError("scorer crashed")
        s = json.loads((self.root / "run/status.json").read_text())
        self.assertEqual(s["state"], "failed")
        self.assertEqual((s["processed"], s["answered"], s["correct"]), (0, 0, 0))
        self.assertEqual(s["pending"], 1)
        self.assertFalse((self.root / "run/logger_active.json").exists())

    def test_comparison_pairs_identical_cases_and_rejects_mode_changes(self):
        for name, scores in (("old", [True, False]), ("new", [False, True])):
            ident = identity(checkpoint_id=name, checkpoint_sha256=("e" if name == "new" else "a") * 64)
            with self.logger(name, total=2, ident=ident) as log:
                for i, score in enumerate(scores):
                    log.result(str(i), correct=score)
        rows = export_comparison([self.root / "old", self.root / "new"], self.root / "compare.csv")
        self.assertEqual((rows[1]["paired_improved"], rows[1]["paired_regressed"]), (1, 1))
        self.assertEqual(rows[1]["delta_percentage_points"], 0)
        with self.logger("other-mode", total=2, ident=identity(mode="search-enabled")) as log:
            log.result("0", correct=True)
            log.result("1", correct=True)
        with self.assertRaisesRegex(ValueError, "comparison rejected"):
            comparison_rows([self.root / "old", self.root / "other-mode"])

    def test_comparison_does_not_hide_unmeasured_cases(self):
        with self.logger("partial", total=2) as log:
            log.result("one", correct=True)
            log.result("two", status="unmeasured")
        with self.assertRaisesRegex(ValueError, "complete measured"):
            comparison_rows([self.root / "partial"])

    def test_incomplete_event_tail_is_not_silently_dropped(self):
        log = self.logger(total=1)
        log.finish(interrupted=True)
        with (self.root / "run/events.jsonl").open("ab") as f:
            f.write(b'{"event":"case_result"')
        with self.assertRaisesRegex(ValueError, "incomplete event tail"):
            self.logger(total=1, resume=True)

    def test_existing_python_partials_monitor_is_read_only_and_deduplicates(self):
        parent = self.root / "active-suite/python_120"
        partials = parent / "partials_step_30500"
        partials.mkdir(parents=True)
        selection = {"provenance": {"selection_sha256": "b" * 64}, "cases": [{"id": "q1"}, {"id": "q2"}]}
        (parent / "selection_120.json").write_text(json.dumps(selection))
        row = {"id": "q1", "explicit_final_numeric_exact": True,
               "terminal_explicit_exact": False, "python_execution_success": True,
               "tool_and_terminal_exact": False, "python_only_tool_and_terminal_exact": False,
               "ended_with_eos": False, "python_invocations": 2,
               "successful_python_invocations": 1}
        item = {"checkpoint_sha256": "a" * 64, "selection_sha256": "b" * 64,
                "max_tokens": 2048, "result": row}
        (partials / "000_q1.json").write_text(json.dumps(item))
        (partials / "duplicate_q1.json").write_text(json.dumps(item))
        suite = parent.parent
        (suite / "suite_abc.log").write_text("PYTHON_TIR_STEP_30500_PROGRESS 90/120\n")
        before = {str(p.relative_to(suite)): p.read_bytes() for p in suite.rglob("*") if p.is_file()}
        monitor = ExistingSuiteMonitor(suite)
        s = monitor.snapshot()
        self.assertEqual(s["source_writes"], 0)
        self.assertEqual(s["python"][0]["processed"], 1)
        self.assertEqual(s["python"][0]["metric_counts"]["explicit_final_numeric_exact"], 1)
        self.assertEqual(s["python"][0]["total"], 2)
        self.assertEqual(s["last_progress_hint"]["processed"], 90)
        after = {str(p.relative_to(suite)): p.read_bytes() for p in suite.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_subprocess_tee_remains_live_while_child_is_quiet(self):
        command = [sys.executable, "-c", "import time; print('PYTHON_TIR_STEP_30500_PROGRESS 90/120'); time.sleep(.25)"]
        code = run_with_logging(command, self.root / "tee", heartbeat_s=0.05, stream=self.output)
        self.assertEqual(code, 0)
        status = json.loads((self.root / "tee/process_status.json").read_text())
        self.assertEqual(status["state"], "complete")
        self.assertEqual(status["last_progress_hint"]["processed"], 90)
        self.assertNotIn("correct", status["last_progress_hint"])
        self.assertGreaterEqual((self.root / "tee/console.log").read_text().count("HEARTBEAT"), 2)
        stored_command = json.loads((self.root / "tee/command.json").read_text())["command"]
        self.assertIn("-u", stored_command)

    def test_live_render_reports_python_metrics_only_when_observed(self):
        with self.logger(total=2) as log:
            log.result("a", correct=False, metrics={"python_execution_success": True,
                "explicit_final_numeric_exact": False, "tool_and_terminal_exact": False})
            log.result("b", correct=True, metrics={"python_execution_success": True,
                "explicit_final_numeric_exact": True, "tool_and_terminal_exact": True})
            log.render(force=True)
        rendered = self.output.getvalue()
        self.assertIn("Python実行成功 2/2", rendered)
        self.assertIn("最終数値一致 1/2", rendered)
        self.assertIn("ツール＋最終一致 1/2", rendered)

    def test_wrapper_waits_for_child_created_watch_directory(self):
        watched = self.root / "child_results"
        command = [sys.executable, "-c", "from pathlib import Path; import sys; Path(sys.argv[1]).mkdir()", str(watched)]
        code = run_with_logging(command, self.root / "tee_late", watch_root=watched,
                                heartbeat_s=0.05, stream=self.output)
        self.assertEqual(code, 0)
        self.assertTrue(watched.is_dir())
        self.assertEqual(list(watched.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
