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

    @staticmethod
    def real_generation(before, query, result, after, eos=True, n=40):
        """Shaped exactly like generate_with_qwenputer_v42: the runtime appends ' => {result} ' to the output text."""
        text = f"{before}<SEARCH {query}> => {result} {after}"
        return {"output_text": text, "search_invocations": 1, "search_trace": [{"query": query, "result": result}],
                "ended_with_eos": eos, "generated_tokens": list(range(n))}

    def test_t6_injected_result_does_not_count_as_the_models_answer(self):
        item = {"target": "math.isqrt", "hop_type": "1_hop"}
        result = "[RESULT (src: local, date: 常駐): math.isqrt | 整数平方根を返す]"
        junk = R.t6_score(item, self.real_generation("思考: 調べる ", "math.isqrt", result, "ほげほげ"))
        self.assertTrue(junk["answer_retrieved"])
        self.assertTrue(junk["answer_in_output"])                 # the legacy rule is fooled by the injected text ...
        self.assertFalse(junk["answer_written"])                  # ... the honest measures are not
        self.assertFalse(junk["used_after_retrieval"])
        copy = R.t6_score(item, self.real_generation("思考: 調べる ", "x", result, "答え: math.isqrt。"))
        self.assertTrue(copy["answer_retrieved"] and copy["answer_written"] and copy["used_after_retrieval"])

    def test_t6_a_query_that_spells_the_answer_is_not_an_answer(self):
        item = {"target": "math.isqrt", "hop_type": "1_hop"}
        r = R.t6_score(item, self.real_generation("", "math.isqrt", "[RESULT: 別の文書]", "不明"))
        self.assertFalse(r["answer_written"] or r["answer_retrieved"] or r["used_after_retrieval"])

    def test_t6_answer_without_searching_and_without_retrieval(self):
        item = {"target": "math.isqrt", "hop_type": "1_hop"}
        r = R.t6_score(item, {"output_text": "答え: math.isqrt。", "search_invocations": 0, "search_trace": [], "ended_with_eos": True,
                              "generated_tokens": [1, 2, 3]})
        self.assertTrue(r["answer_written"] and r["answer_in_output"])
        self.assertFalse(r["answer_retrieved"] or r["used_after_retrieval"])
        self.assertEqual(r["n_tokens"], 3)

    def test_t6_the_whole_generation_and_trace_are_saved(self):
        item = {"target": "t", "hop_type": "1_hop"}
        r = R.t6_score(item, self.real_generation("あ" * 500, "q", "[RESULT: t]", "い" * 500))
        self.assertGreater(len(r["text"]), 1000)
        self.assertEqual(r["trace"][0]["query"], "q")

    def test_t5_records_how_the_generation_ended(self):
        capped = R.t5_score({"target": "B"}, {"output_text": "長い考え", "tokens": list(range(192)), "ended_with_eos": False}, cap=192)
        self.assertTrue(capped["hit_cap"] and not capped["ended_with_eos"] and capped["n_tokens"] == 192)
        done = R.t5_score({"target": "B"}, {"output_text": "正解: B", "tokens": list(range(30)), "ended_with_eos": True}, cap=192)
        self.assertFalse(done["hit_cap"])
        self.assertFalse(R.t5_score({"target": "B"}, {"output_text": "x"}, cap=None)["hit_cap"])


class McqRobustness(unittest.TestCase):
    def test_cache_makes_a_restart_free_and_identical(self):
        items = synthetic_items(40)
        calls = []

        def counted(ps):
            calls.append(len(ps))
            return content_scorer(items)(ps)
        with tempfile.TemporaryDirectory() as td:
            cache = Path(td) / "mcq_logits.jsonl"
            first = R.mcq_rotation_eval(items, counted, cache_path=cache)
            n_first = sum(calls)
            self.assertEqual(n_first, 40 * 4)
            second = R.mcq_rotation_eval(items, counted, cache_path=cache)
            self.assertEqual(sum(calls), n_first)                          # nothing was scored again
            self.assertEqual(first["debiased_correct"], second["debiased_correct"])
            self.assertEqual(first["letter_logits"], second["letter_logits"])

    def test_a_crash_midway_keeps_the_finished_batches(self):
        items = synthetic_items(40)
        good = content_scorer(items)
        state = {"n": 0}

        def flaky(ps):
            state["n"] += 1
            if state["n"] == 7:
                raise RuntimeError("session died")
            return good(ps)
        with tempfile.TemporaryDirectory() as td:
            cache = Path(td) / "c.jsonl"
            with self.assertRaises(RuntimeError):
                R.mcq_rotation_eval(items, flaky, batch=4, cache_path=cache)
            kept = len(cache.read_text(encoding="utf-8").splitlines())
            self.assertEqual(kept, 6 * 4)
            calls = []
            res = R.mcq_rotation_eval(items, lambda ps: (calls.append(len(ps)), good(ps))[1], batch=4, cache_path=cache)
            self.assertEqual(sum(calls), 40 * 4 - kept)
            self.assertEqual(res["debiased_accuracy"], 100.0)

    def test_progress_is_reported(self):
        items = synthetic_items(40)
        seen = []
        R.mcq_rotation_eval(items, content_scorer(items), batch=4, progress=lambda d, t: seen.append((d, t)), progress_every=10)
        self.assertEqual(seen[-1], (160, 160))
        self.assertTrue(all(t == 160 for _, t in seen))

    def test_non_finite_rows_are_counted_wrong_not_turned_into_A(self):
        items = synthetic_items(40)
        good = content_scorer(items)

        def broken(ps):
            out = good(ps)
            out[0] = [float("nan")] * 4
            return out
        res = R.mcq_rotation_eval(items, broken, batch=4)
        self.assertGreater(res["nonfinite"], 0)
        self.assertEqual(res["picked_letter_share"]["A"], res["picked_letter_share"]["A"])     # no NaN leaked into the shares
        self.assertLess(res["debiased_accuracy"], 100.0)

    def test_ties_are_counted(self):
        items = synthetic_items(8)
        res = R.mcq_rotation_eval(items, lambda ps: [[2.0, 2.0, 0.0, 0.0] for _ in ps])
        self.assertEqual(res["ties"], 8 * 4)
        self.assertEqual(res["picked_letter_share"]["A"], 1.0)           # ties resolve to the first letter, like argmax


if __name__ == "__main__":
    unittest.main()
