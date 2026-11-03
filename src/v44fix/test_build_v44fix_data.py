#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Data-builder tests. Public fixtures use exactly the record keys gpt/prepare_public_v44.py writes
(prompt/target/full_text/source/bucket/language/target_segments/provenance; NO "question" field)."""
import json, re, subprocess, sys, tempfile, unittest
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
V43_SAMPLE = ROOT / "test_assets" / "v43_train_sample.jsonl"


def gpt_row(bucket, content, target):
    prompt = "ユーザー: " + content + "\nアシスタント: "
    return {"prompt": prompt, "target": target, "full_text": prompt + target, "source": "public_x", "bucket": bucket,
            "language": "ja", "target_segments": [], "provenance": {"public_repo": "test"}}


class BuildV44FixDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        pub, ev = cls.tmp / "public", cls.tmp / "eval"
        pub.mkdir(); ev.mkdir()
        tir = [gpt_row("tir", f"Compute {i} + {i}.", f"We use Python.\n```python\nprint({i}+{i})\n```\n```output\n{2*i}\n```\nSo the answer is $\\boxed{{{2*i}}}$.")
               for i in range(1, 6)]
        rows = tir + tir  # GPT's mixture repeats rows; the builder must dedup public rows
        rows.append(gpt_row("tir", "No boxed here.", "Just text."))
        rows.append(gpt_row("code", "Write a C function.", "```c\nint f(void){return 1;}\n```"))
        rows.append(gpt_row("japanese", "本文:\n富士山は日本一高い山です。\n質問:\n日本一高い山は？", "富士山です。"))
        rows.append(gpt_row("japanese", "季節を一つ挙げて。", "春です。"))
        rows.append(gpt_row("japanese", "評価に使う質問です。", "答え"))
        rows.append(gpt_row("japanese", "語彙外の文字です。\U0001F600", "x"))
        rows.append(gpt_row("anchor", "これは捨てる", "x"))
        bad = gpt_row("japanese", "壊れた行", "x"); bad["prompt"] = "壊れた行"
        rows.append(bad)
        (pub / "train.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
        (pub / "dev.jsonl").write_text(json.dumps(gpt_row("tir", "Held out 7 * 3.", "x\n$\\boxed{21}$"), ensure_ascii=False) + "\n", encoding="utf-8")
        (pub / "test.jsonl").write_text("", encoding="utf-8")
        (ev / "eval_x.jsonl").write_text(json.dumps({"prompt": "評価に使う質問です。"}, ensure_ascii=False) + "\n", encoding="utf-8")
        cls.v43 = [json.loads(l) for l in V43_SAMPLE.read_text(encoding="utf-8").splitlines()]
        cls.out = cls.tmp / "out"
        r = subprocess.run([sys.executable, str(ROOT / "build_v44fix_data.py"), "--public-dir", str(pub), "--v43-train", str(V43_SAMPLE),
                            "--eval-dir", str(ev), "--vocab", str(ROOT / "base_metadata.json"), "--out-dir", str(cls.out),
                            "--mix", "anchor=2000,tir=100,code=100,japanese=100", "--max-repeat", "2.0"],
                           capture_output=True, text=True, encoding="utf-8", cwd=str(ROOT))
        assert r.returncode == 0, r.stdout + r.stderr
        cls.manifest = json.loads((cls.out / "manifest.json").read_text(encoding="utf-8"))
        cls.rows = [json.loads(l) for l in (cls.out / "train.jsonl").read_text(encoding="utf-8").splitlines()]

    def test_no_v44_format_and_v43_heads(self):
        self.assertEqual(self.manifest["rows_in_v44_user_assistant_format"], 0)
        pub = [r for r in self.rows if r["bucket"] != "anchor"]
        self.assertTrue(pub)
        self.assertTrue(all(r["prompt"].startswith(("問題: ", "指示: ")) for r in pub))
        self.assertTrue(all(not r["prompt"].startswith("ユーザー:") for r in self.rows))

    def test_v43_space_after_prompt_everywhere(self):
        self.assertEqual(self.manifest["rows_without_v43_space_after_prompt"], 0)
        self.assertTrue(all(r["full_text"][len(r["prompt"])] == " " for r in self.rows))
        self.assertTrue(all(r["full_text"] == r["prompt"] + r["target"] for r in self.rows if r["bucket"] != "anchor"))

    def test_tir_conversion(self):
        r = next(r for r in self.rows if r["bucket"] == "tir")
        self.assertTrue(r["prompt"].startswith("問題: Compute ") and r["prompt"].endswith("\n思考:"))
        self.assertTrue(r["target"].startswith(" We use Python."))
        self.assertRegex(r["target"], r"\n答え: \d+$")
        self.assertEqual(self.manifest["rejections"].get("tir:tir_no_boxed_answer"), 1)

    def test_japanese_context_uses_v43_dolly_format(self):
        ja = {r["prompt"] for r in self.rows if r["bucket"] == "japanese"}
        self.assertIn("指示: 日本一高い山は？\n文脈: 富士山は日本一高い山です。\n回答:", ja)
        self.assertIn("指示: 季節を一つ挙げて。\n回答:", ja)

    def test_rejections_and_public_repeat_cap(self):
        rj = self.manifest["rejections"]
        self.assertEqual(rj.get("japanese:evaluation_overlap"), 1)
        self.assertEqual(rj.get("japanese:out_of_vocab_char"), 1)
        self.assertEqual(rj.get("japanese:schema"), 1)
        b = self.manifest["buckets"]
        self.assertEqual(b["tir"]["unique_rows"], 5)
        self.assertEqual(b["tir"]["exposures"], 10)           # 100 requested, capped at 5 * 2.0
        for k in ("tir", "code", "japanese"):
            self.assertLessEqual(b[k]["repeat_factor_vs_pool"], 2.0)

    def test_anchor_keeps_v43_weighting_incl_search_and_alu(self):
        a = self.manifest["buckets"]["anchor"]
        self.assertEqual(a["exposures"], 2000)
        self.assertGreater(a["alu_rows"], 0)
        v43_share = sum(bool(re.search(r"<SEARCH[ \t]", r["target"])) for r in self.v43) / len(self.v43)
        share = self.manifest["search_share"]["anchor"]
        self.assertGreater(v43_share, 0)
        self.assertGreaterEqual(share, 0.8 * v43_share)
        anchors = [r for r in self.rows if r["bucket"] == "anchor"]
        self.assertTrue(all(r["full_text"] in {x.get("full_text") or x["prompt"] + x["target"] for x in self.v43} for r in anchors))

    def test_heldout_tir(self):
        held = [json.loads(l) for l in (self.out / "tir_heldout.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(held[0]["answer"], "21")
        self.assertTrue(held[0]["prompt"].startswith("問題: Held out"))
        self.assertTrue(held[0]["reference"].startswith(" "))


if __name__ == "__main__":
    unittest.main(verbosity=2)
