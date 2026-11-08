"""CPU tests for the teacher-forced T5 probe (real T5 manifest + the templates extracted from real training rows)."""
import json
import re
import sys
import unittest
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import t5_probe as P  # noqa: E402

MANIFEST = None
for cand in (HERE.parents[2] / "v47_handoff/bundle/r6/data/eval_v41_logical_instruction_operators_heldout.jsonl",
             Path(r"C:/Users/81808/AppData/Local/Temp/claude/ff74aa4e8a98970d/f80753de-c283-4765-a931-f974c2b54295/scratchpad/v47_handoff/bundle/r6/data/eval_v41_logical_instruction_operators_heldout.jsonl")):
    if cand.exists():
        MANIFEST = cand
        break
TEMPLATES = HERE / "t5_templates.json"
READY = MANIFEST is not None and TEMPLATES.exists()


def items():
    return [json.loads(x) for x in MANIFEST.read_text(encoding="utf-8").splitlines() if x.strip()]


@unittest.skipUnless(READY, "T5 manifest / templates not available")
class Probe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.items = items()
        cls.templates = P.load_templates(TEMPLATES)

    def test_orig_variant_with_identity_renaming_reproduces_the_original_prompt_byte_for_byte(self):
        for it in self.items:
            p = P.parse_prompt(it["prompt"])
            self.assertEqual(P.render_prompt(p["head"], p["premises"], p["instruction"], p["options"]), it["prompt"], it["id"])

    def test_every_decisive_offset_points_at_the_gold_letter_in_every_variant(self):
        probes = P.build_all(self.items, self.templates)
        self.assertEqual(len(probes), 400 * 4 * 3)
        for pr in probes:
            for pos, off in pr["positions"].items():
                self.assertEqual(pr["text"][off], pr["gold"][pos])
            self.assertLessEqual(len(pr["text"]) + 1, 2048)

    def test_gold_response_has_the_trained_shape(self):
        pr = P.build_variant(self.items[0], self.templates, "orig", P.RENAMINGS[0])
        t = pr["text"]
        self.assertIn("【2. 各選択肢の論理状態検証】\n- ", t)
        self.assertTrue(t.endswith("\n正解: " + pr["gold"]["COPY"]))
        self.assertEqual(pr["gold"]["FIN"], pr["gold"]["COPY"])
        # the decisive letters of the three state lines are three different letters
        self.assertEqual(len({pr["gold"]["S1"], pr["gold"]["S2"], pr["gold"]["S3"]}), 3)

    def test_content_labels_are_preserved_in_every_variant_and_renaming(self):
        for it in self.items:
            p0 = P.parse_prompt(it["prompt"])
            slot_of_text = {text: {v: k for k, v in it["map_type_to_letter"].items()}[old] for old, (_k, text) in p0["options"].items()}
            for variant in P.VARIANTS:
                for ren in P.RENAMINGS:
                    pr = P.build_variant(it, self.templates, variant, ren)
                    p = P.parse_prompt(pr["text"].split("思考の連鎖と回答:")[0] + P.PROMPT_END) if variant != "nocite" else None
                    if p is None:
                        continue
                    for letter, (_k, text) in p["options"].items():
                        slot = slot_of_text[text]
                        self.assertEqual(pr["gold"]["S" + str({"T": 1, "F": 2, "U": 3}[slot])], letter)

    def test_renumbering_breaks_the_number_label_correspondence_and_orig_keeps_it(self):
        for it in self.items:
            for ren in P.RENAMINGS:
                orig = P.build_variant(it, self.templates, "orig", ren)
                for pos in P.POSITIONS:
                    self.assertEqual(orig["number_rule"][pos], orig["gold"][pos])
                for variant in ("renum1", "renum2"):
                    pr = P.build_variant(it, self.templates, variant, ren)
                    for pos in ("S1", "S2", "S3", "FIN", "COPY"):
                        self.assertNotEqual(pr["number_rule"][pos], pr["gold"][pos], (it["id"], variant, pos))
                self.assertTrue(all(v is None for v in P.build_variant(it, self.templates, "nocite", ren)["number_rule"].values()))

    def test_letter_renaming_does_not_leave_a_letter_prior_in_the_gold(self):
        probes = P.build_all(self.items, self.templates, variants=("orig",))
        for pos in ("S1", "S2", "S3", "FIN"):
            c = Counter(pr["gold"][pos] for pr in probes)
            self.assertLess(max(c.values()) / sum(c.values()), 0.40, (pos, c))

    # ---- scorers that behave like known solvers
    @staticmethod
    def fake_scorer(kind):
        def score(texts, offsets):
            out = []
            for text, offs in zip(texts, offsets):
                prompt = text.split(P.PROMPT_END)[0] + P.PROMPT_END
                try:
                    p = P.parse_prompt(prompt)
                except ValueError:                                  # 'nocite': no number to follow
                    p = None
                ins = re.search(r"【指示】\n(.*?)\n【選択肢】", prompt, re.S).group(1).strip()
                ans_slot = P.TYPE_SLOT[next(t for t, v in KNOWN_INSTR.items() if v == ins)]
                res = []
                for pos, off in zip(P.POSITIONS, offs):
                    if kind == "number_rule":
                        want = {"S1": 1, "S2": 2, "S3": 3, "FIN": P.SLOT_NUMBER[ans_slot], "COPY": P.SLOT_NUMBER[ans_slot]}[pos]
                        letter = next(L for L, (k, _t) in p["options"].items() if k == want) if p else "A"
                    elif kind == "content":
                        letter = text[off]                       # cheats with the gold letter: a perfect content reader
                    else:                                         # 'always A'
                        letter = "A"
                    res.append([5.0 if L == letter else 0.0 for L in P.LETTERS])
                out.append(res)
            return out
        return score

    def test_number_rule_solver_is_perfect_on_orig_and_collapses_on_renumbered(self):
        global KNOWN_INSTR
        KNOWN_INSTR = {}
        for it in self.items:
            KNOWN_INSTR[it["operator_type"]] = P.parse_prompt(it["prompt"])["instruction"]
        probes = P.build_all(self.items[:80], self.templates)
        rows = P.score_all(probes, self.fake_scorer("number_rule"))
        s = P.summarize(rows)
        for pos in P.POSITIONS:
            self.assertEqual(s["accuracy"][f"orig/{pos}"], 100.0)
            self.assertEqual(s["accuracy"][f"renum1/{pos}"], 0.0)
            self.assertEqual(s["accuracy"][f"renum2/{pos}"], 0.0)
            self.assertEqual(s["follows_number_rule"][f"renum1/{pos}"], 100.0)

    def test_content_reader_keeps_its_score_and_always_A_sits_at_chance(self):
        global KNOWN_INSTR
        KNOWN_INSTR = {}
        for it in self.items:
            KNOWN_INSTR[it["operator_type"]] = P.parse_prompt(it["prompt"])["instruction"]
        probes = P.build_all(self.items[:80], self.templates)
        s = P.summarize(P.score_all(probes, self.fake_scorer("content")))
        self.assertTrue(all(v == 100.0 for v in s["accuracy"].values()))
        a = P.summarize(P.score_all(probes, self.fake_scorer("always_A")))
        for k, v in a["accuracy"].items():
            self.assertAlmostEqual(v, 100 / 3, delta=6, msg=k)           # letters were renamed evenly -> position prior is neutralised

    def test_scoring_is_resumable(self):
        import tempfile
        global KNOWN_INSTR
        KNOWN_INSTR = {}
        for it in self.items:
            KNOWN_INSTR[it["operator_type"]] = P.parse_prompt(it["prompt"])["instruction"]
        probes = P.build_all(self.items[:10], self.templates)
        calls = []

        def counted(texts, offsets):
            calls.append(len(texts))
            return self.fake_scorer("content")(texts, offsets)
        with tempfile.TemporaryDirectory() as td:
            cache = Path(td) / "c.jsonl"
            first = P.score_all(probes, counted, cache_path=cache)
            n = sum(calls)
            second = P.score_all(probes, counted, cache_path=cache)
            self.assertEqual(sum(calls), n)
            self.assertEqual([r["logits"] for r in first], [r["logits"] for r in second])


KNOWN_INSTR = {}

if __name__ == "__main__":
    unittest.main()
