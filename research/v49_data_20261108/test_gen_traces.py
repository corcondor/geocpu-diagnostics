"""CPU tests for gen_traces.py: three-way answer agreement, vocabulary, length, and 'the trace really contains the work'."""
import collections
import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gen_traces as G  # noqa: E402
import solvers as S  # noqa: E402

VOCAB = set(json.loads((HERE.parents[1] / "src/v44fix/base_metadata.json").read_text(encoding="utf-8"))["vocab"])
ROWS = [G.make_row(f, i, "train") for f in range(32) for i in range(32)]


class Traces(unittest.TestCase):
    def test_every_row_agrees_three_ways(self):
        for r in ROWS:
            q = r["prompt"][len("問題: "):].split("\n条件と根拠を順に考えてください。")[0]
            self.assertEqual(S.solve_py(r["prompt"]), r["reference_final_answer"], r["id"])      # independent solver
            self.assertTrue(r["target"].endswith(f"答え: {r['reference_final_answer']}。"), r["id"])
            self.assertTrue(q)                                                                      # (trace == v48 answer is asserted in make_row)

    def test_all_32_families_are_covered_and_distinct(self):
        self.assertEqual(len({r["family"] for r in ROWS}), 32)
        self.assertEqual(len({r["prompt"] for r in ROWS}), len(ROWS))

    def test_vocabulary_and_length(self):
        for r in ROWS:
            self.assertLessEqual(set(r["full_text"]), VOCAB, (r["id"], set(r["full_text"]) - VOCAB))
            self.assertLessEqual(len(r["full_text"]) + 1, 2048, r["id"])

    def test_the_trace_shows_intermediate_work(self):
        const = collections.Counter()
        for r in ROWS:
            body = r["target"].rsplit("\n答え:", 1)[0]
            self.assertGreaterEqual(r["trace_lines"], 2, r["id"])
            self.assertGreater(len(body), 25, r["id"])
        # the decisive result is now one step of a visible chain, not the only unexplained characters
        longest = max(len(r["target"]) for r in ROWS)
        avg = sum(len(r["target"]) for r in ROWS) / len(ROWS)
        self.assertGreater(avg, 60)            # v48 logic targets averaged ~31 characters
        self.assertLess(longest, 600)

    def test_dev_split_also_builds(self):
        rows = [G.make_row(f, i, "dev") for f in range(32) for i in range(1000, 1004)]
        self.assertEqual(len(rows), 128)
        for r in rows:
            self.assertEqual(S.solve_py(r["prompt"]), r["reference_final_answer"])

    def test_a_wrong_trace_is_caught(self):
        import gen_traces
        orig = gen_traces.trace_for
        try:
            gen_traces.trace_for = lambda q: (["x"], -1)
            with self.assertRaises(AssertionError):
                gen_traces.make_row(0, 0, "train")
        finally:
            gen_traces.trace_for = orig


JROWS = [G.make_japanese_row(f, i, "train") for f in range(32) for i in range(32)]


class JapaneseTraces(unittest.TestCase):
    def test_three_way_agreement_on_all_rows(self):
        for r in JROWS:
            letter, n = S.solve_japanese(r["prompt"])
            self.assertEqual(n, 1, r["id"])                      # exactly one option satisfies the question
            self.assertEqual(letter, r["reference_final_answer"], r["id"])
            self.assertTrue(r["target"].endswith(f"答え: {r['reference_final_answer']}。"), r["id"])

    def test_every_option_is_judged_on_the_page(self):
        for r in JROWS:
            body = r["target"]
            for letter in "ABCD":
                self.assertIn(f"\n{letter}=", "\n" + body.lstrip("\n"), r["id"])
            self.assertEqual(body.count("→"), 4, r["id"])
            self.assertEqual(r["trace_lines"], 5)

    def test_all_conditions_covered_and_vocab_ok(self):
        self.assertEqual(len({r["family"] for r in JROWS}), 32)
        for r in JROWS:
            self.assertLessEqual(set(r["full_text"]), VOCAB, (r["id"], set(r["full_text"]) - VOCAB))
            self.assertLessEqual(len(r["full_text"]) + 1, 2048)

    def test_dev_and_unseen_build(self):
        for split, fams, idxs in (("dev", range(32), range(1000, 1004)), ("unseen_eval", range(32, 36), range(2000, 2032))):
            rows = [G.make_japanese_row(f, i, split) for f in fams for i in idxs]
            self.assertEqual(len(rows), 128)
            for r in rows:
                self.assertEqual(S.solve_japanese(r["prompt"])[0], r["reference_final_answer"])


if __name__ == "__main__":
    unittest.main()
