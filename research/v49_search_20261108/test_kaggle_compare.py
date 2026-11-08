"""CPU test: kaggle_compare builds a report from two fake measurement folders, and refuses to judge unfinished ones."""
import json, sys, tempfile, unittest
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import robust_eval as R
import kaggle_compare as C


def fake_measure(out: Path, prefix: str, skill: float, complete: bool = True, t5_rows: int = 400, with_summary: bool = True):
    d = out / prefix; d.mkdir(parents=True)
    items = []
    for i in range(300):
        gold = i % 4
        t = [f"o{i}{j}" for j in range(4)]
        items.append({"id": f"s{i}", "prompt": "分野\n問題: q%d\n" % i + "\n".join(f"{R.LETTERS[j]}: {t[j]}" for j in range(4)) + "\n正解:", "target": R.LETTERS[gold], "_t": t[gold]})
    gold_text = {it["prompt"].split("\n")[1]: it["_t"] for it in items}
    def scorer(ps):
        res = []
        for p in ps:
            parsed = R.parse_mcq(p); stem = p.split("\n")[1]
            vec = [3.0 if j == 0 else 0.0 for j in range(4)]
            if (hash(p) % 100) / 100 < skill:
                vec = [6.0 if parsed[1][j] == gold_text[stem] else 0.0 for j in range(4)]
            res.append(vec)
        return res
    (d / "mcq.json").write_text(json.dumps(R.mcq_rotation_eval(items, scorer), ensure_ascii=False), encoding="utf-8")
    t5 = [{"id": f"t{i}", "correct": (i % 100) / 100 < skill, "has_answer_marker": i % 2 == 0, "hit_cap": i % 5 == 0, "pred": "A", "target": "A"}
          for i in range(t5_rows)]
    (d / "t5.jsonl").write_text("\n".join(json.dumps(r) for r in t5), encoding="utf-8")
    t6 = [{"id": f"u{i}", "weight": 10 if i < 20 else 0, "answer_retrieved": True, "answer_in_output": True,
           "answer_written": i < skill * 21, "used_after_retrieval": i < skill * 21, "search_calls": 1} for i in range(21)]
    (d / "t6.jsonl").write_text("\n".join(json.dumps(r) for r in t6), encoding="utf-8")
    if with_summary:
        (d / "summary.json").write_text(json.dumps({"prefix": prefix, "complete": complete, "caps": {"t5": 192, "t6": 256}, "tasks": {}}), encoding="utf-8")


class Compare(unittest.TestCase):
    def test_report_flags_a_real_difference_and_not_noise(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            fake_measure(out, "base", 0.30)
            fake_measure(out, "new", 0.80)
            text = C.report(C.load(out, "base"), C.load(out, "new"), "base", "new")
            self.assertIn("JMMLU", text); self.assertIn("T5", text); self.assertIn("T6", text)
            self.assertIn("有意に向上", text)
            self.assertIn("偏りを除いた正解率", text)
            self.assertNotIn("!!", text)                                  # complete runs: no banners
            self.assertIn("答えをモデル自身が書いた", text)
            self.assertIn("上限に当たった", text)
            same = C.report(C.load(out, "base"), C.load(out, "base"), "base", "base")
            self.assertIn("有意ではない", same)

    def test_a_drop_is_called_out(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td); fake_measure(out, "base", 0.80); fake_measure(out, "new", 0.30)
            self.assertIn("有意に低下（要調査）", C.report(C.load(out, "base"), C.load(out, "new"), "base", "new"))

    def test_a_side_without_summary_is_incomplete_and_a_partial_t5_gets_no_verdict(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            fake_measure(out, "base", 0.30)
            fake_measure(out, "new", 0.80, complete=False, t5_rows=200, with_summary=False)
            text = C.report(C.load(out, "base"), C.load(out, "new"), "base", "new")
            self.assertIn("!! 未完了：new", text)
            self.assertIn("!! T5：問題の集合が一致しません", text)
            self.assertNotIn("## T5", text)                                # no T5 statistics at all for the partial side

    def test_a_failed_task_is_named_in_the_banner(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            fake_measure(out, "base", 0.30)
            fake_measure(out, "new", 0.80)
            s = json.loads((out / "new" / "summary.json").read_text())
            s["complete"] = False; s["tasks"] = {"t5": {"error": "RuntimeError('boom')"}}
            (out / "new" / "summary.json").write_text(json.dumps(s))
            text = C.report(C.load(out, "base"), C.load(out, "new"), "base", "new")
            self.assertIn("t5=RuntimeError('boom')", text)

    def test_empty_and_missing_sides_do_not_crash(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            fake_measure(out, "base", 0.30)
            (out / "new").mkdir()
            (out / "new" / "t5.jsonl").write_text("", encoding="utf-8")       # crashed on item 0 of T5
            text = C.report(C.load(out, "base"), C.load(out, "new"), "base", "new")
            self.assertIn("!! 未完了：new", text)
            self.assertIn("T5 は片方が空か欠けている", text)
            C.main(["--out", str(out), "--base", "base", "--new", "new"])       # must not raise
            self.assertTrue((out / "report.md").exists())

    def test_smoke_runs_are_labelled(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            fake_measure(out, "base", 0.30); fake_measure(out, "new", 0.30)
            s = json.loads((out / "new" / "summary.json").read_text()); s["limit"] = 2
            (out / "new" / "summary.json").write_text(json.dumps(s))
            self.assertIn("試験実行", C.report(C.load(out, "base"), C.load(out, "new"), "base", "new"))


if __name__ == "__main__":
    unittest.main()
