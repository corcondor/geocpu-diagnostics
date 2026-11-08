"""CPU tests for robust_eval.py (fake scorers/generators; real JMMLU/T5/T6 manifests when the handoff copy is present)."""
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import robust_eval as R  # noqa: E402

DATA = None
for cand in (HERE.parents[2] / "v47_handoff/bundle/r6/data",
             Path(r"C:/Users/81808/AppData/Local/Temp/claude/ff74aa4e8a98970d/f80753de-c283-4765-a931-f974c2b54295/scratchpad/v47_handoff/bundle/r6/data")):
    if (cand / "benchmark_8100_manifest.jsonl").exists():
        DATA = cand
        break


def synthetic_items(n=200):
    items = []
    for i in range(n):
        gold = i % 4
        texts = [f"選択肢{i}-{j}" for j in range(4)]
        prompt = f"分野: x (y)\n問題: 問{i}\n" + "\n".join(f"{R.LETTERS[j]}: {texts[j]}" for j in range(4)) + "\n正解:"
        items.append({"id": f"s{i}", "prompt": prompt, "target": R.LETTERS[gold], "_gold_text": texts[gold]})
    return items


def content_scorer(items):
    gold = {it["prompt"].split("\n")[1]: it["_gold_text"] for it in items}

    def score(prompts):
        out = []
        for p in prompts:
            stem = p.split("\n")[1]
            parsed = R.parse_mcq(p)
            vec = [0.0] * 4
            for j, t in enumerate(parsed[1]):
                if t == gold[stem]:
                    vec[j] = 6.0
            out.append(vec)
        return out
    return score


class Stats(unittest.TestCase):
    def test_wilson_and_mcnemar_known_values(self):
        lo, hi = R.wilson(50, 100)
        self.assertAlmostEqual(lo, 40.4, 1)
        self.assertAlmostEqual(hi, 59.6, 1)
        self.assertAlmostEqual(R.mcnemar_exact(0, 10), 2 / 1024, 6)
        self.assertEqual(R.mcnemar_exact(5, 5), 1.0)
        self.assertEqual(R.mcnemar_exact(0, 0), 1.0)

    def test_paired_report(self):
        a = [True] * 60 + [False] * 40
        b = [True] * 50 + [False] * 10 + [True] * 25 + [False] * 15      # 10 only-A, 25 only-B
        rep = R.paired_report(a, b)
        self.assertEqual((rep["only_a"], rep["only_b"], rep["a"], rep["b"]), (10, 25, 60, 75))
        self.assertLess(rep["mcnemar_p"], 0.02)
        with self.assertRaises(ValueError):
            R.paired_report([True], [True, False])


class Rotation(unittest.TestCase):
    def test_rotation_keeps_the_gold_option_text(self):
        for it in synthetic_items(40):
            parsed = R.parse_mcq(it["prompt"])
            gold = R.LETTERS.index(it["target"])
            for k in range(4):
                new = R.parse_mcq(R.rotated_prompt(parsed, k))
                self.assertEqual(new[1][R.gold_after_rotation(gold, k)], it["_gold_text"])
                self.assertEqual(sorted(new[1]), sorted(parsed[1]))

    def test_content_aware_scorer_is_perfect_on_every_rotation(self):
        items = synthetic_items(120)
        res = R.mcq_rotation_eval(items, content_scorer(items))
        self.assertEqual(res["n"], 120)
        for acc in res["rotation_accuracy"]:
            self.assertEqual(acc, 100.0)
        self.assertEqual(res["debiased_accuracy"], 100.0)

    def test_position_biased_scorer_is_exposed(self):
        items = synthetic_items(200)
        res = R.mcq_rotation_eval(items, lambda ps: [[5.0, 1.0, 0.0, 0.0] for _ in ps])      # always 'A'
        self.assertEqual(res["picked_letter_share"]["A"], 1.0)
        for acc in res["rotation_accuracy"]:
            self.assertAlmostEqual(acc, 25.0, delta=0.01)       # gold is balanced across positions
        self.assertAlmostEqual(res["debiased_accuracy"], 25.0, delta=0.01)

    def test_partly_informed_scorer_gains_from_debiasing(self):
        items = synthetic_items(400)
        content = content_scorer(items)

        def mix(ps):
            base = content(ps)
            return [[v + (3.0 if j == 0 else 0.0) * 1.0 for j, v in enumerate(vec)] if (hash(p) % 3 == 0) else vec
                    for p, vec in zip(ps, base)]
        res = R.mcq_rotation_eval(items, mix)
        self.assertGreaterEqual(res["debiased_accuracy"], max(res["rotation_accuracy"]) - 1e-9)

    def test_bad_prompts_are_skipped_not_crashed(self):
        items = [{"id": "x", "prompt": "選択肢の無い問題\n正解:", "target": "A"}] + synthetic_items(8)
        res = R.mcq_rotation_eval(items, lambda ps: [[1, 0, 0, 0]] * len(ps))
        self.assertEqual(res["n"], 8)

    @unittest.skipUnless(DATA, "handoff manifests not available")
    def test_real_jmmlu_prompts_parse_and_rotate(self):
        rows = [json.loads(l) for l in (DATA / "benchmark_8100_manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        mc = [r for r in rows if r.get("task_type") == "multiple_choice"]
        ok = [r for r in mc if R.parse_mcq(r["prompt"])]
        self.assertEqual(len(mc), 7536)
        self.assertGreaterEqual(len(ok), 7500)                # a handful of prompts contain extra 'A:'-style lines and are skipped
        self.assertLessEqual(len(mc) - len(ok), 10)
        for r in ok[:300]:
            parsed = R.parse_mcq(r["prompt"])
            gold = R.LETTERS.index(r["target"].strip())
            for k in range(4):
                self.assertEqual(R.parse_mcq(R.rotated_prompt(parsed, k))[1][R.gold_after_rotation(gold, k)], parsed[1][gold])
            self.assertTrue(R.rotated_prompt(parsed, 0) == r["prompt"])      # rotation 0 reproduces the prompt byte for byte


class Runners(unittest.TestCase):
    def test_resume_never_regenerates(self):
        items = [{"id": f"i{n}", "prompt": "p", "target": "B", "operator_type": "o", "group_id": "g"} for n in range(6)]
        calls = []

        def gen(item):
            calls.append(item["id"])
            return {"output_text": "考え 正解: B"}
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "t5.jsonl"
            R.run_items(items, gen, R.t5_score, out, limit=3)
            self.assertEqual(len(calls), 3)
            rows = R.run_items(items, gen, R.t5_score, out)
            self.assertEqual(len(calls), 6)                   # only the 3 new ones were generated
            self.assertEqual(len(rows), 6)
            self.assertTrue(all(r["correct"] for r in rows))

    def test_t5_extractor_is_the_original(self):
        self.assertEqual(R.extract_choice_abc("理由…正解: C"), "C")
        self.assertEqual(R.extract_choice_abc("長い" * 50 + "ので A です。"), "A")
        self.assertEqual(R.extract_choice_abc("答えなし"), "")

    def test_t5_score_records_whether_the_marker_was_there(self):
        s = R.t5_score({"target": "B"}, {"output_text": "…正解: B"})
        self.assertTrue(s["correct"] and s["has_answer_marker"])
        s = R.t5_score({"target": "B"}, {"output_text": "…ということで B"})
        self.assertTrue(s["correct"] and not s["has_answer_marker"])        # tail letter, no marker: flagged

    def test_t6_separates_retrieval_from_use(self):
        item = {"target": "math.isqrt", "hop_type": "1_hop"}
        hit = R.t6_score(item, {"output_text": "答えは math.isqrt", "search_invocations": 1,
                                "search_trace": [{"query": "q", "result": "[RESULT: math.isqrt | 整数平方根]"}]})
        self.assertTrue(hit["answer_retrieved"] and hit["used_after_retrieval"] and hit["answer_in_output"])
        missed = R.t6_score(item, {"output_text": "分かりません", "search_invocations": 1,
                                   "search_trace": [{"query": "q", "result": "[RESULT: 別の文書]"}]})
        self.assertFalse(missed["answer_retrieved"] or missed["answer_in_output"])
        retrieved_not_used = R.t6_score(item, {"output_text": "不明", "search_invocations": 1,
                                               "search_trace": [{"query": "q", "result": "math.isqrt の説明"}]})
        self.assertTrue(retrieved_not_used["answer_retrieved"] and not retrieved_not_used["used_after_retrieval"])


if __name__ == "__main__":
    unittest.main()
