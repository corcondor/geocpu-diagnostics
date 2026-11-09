#!/usr/bin/env python3
"""Tests of convert.py on tiny fixtures (fixtures/, < 100 KB, real public rows) plus hand-made records.  One process, local PC safe:

    python -I test_convert.py

Checks per converter: row schema, full_text == prompt + target, length cap, loss_mask spans cover the injected observation text for
tool rows (python stdout / tool results), quotes verbatim in the prompt, rejects counted by reason (n_in == n_out + rejects), no
validation/dev files reachable, nothing overwritten.
"""
from __future__ import annotations

import ast
import json
import os
import random
import re
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import convert as C                                              # noqa: E402

FIX = HERE / "fixtures"
CK = C.Checks(use_external=False, run_code=False)               # vocabulary / OOV / row checks; never runs code
CKX = C.Checks(vocab=CK.vocab)                                   # the real executor-profile sandbox (light self-test passed, or this line raises)
FENCE = C.FENCE

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception:                                                # pragma: no cover
    pa = pq = None


def fixture(name):
    return json.loads((FIX / name).read_text(encoding="utf-8"))


def run(key, records, ck=CK):
    spec = C.DATASETS[key]
    st = C.Stats(key)
    rows = list(spec.convert(list(enumerate(records)), st, ck, spec))
    return rows, st


def msg(user, asst):
    return {"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": asst}]}


class RowAsserts(unittest.TestCase):
    def check_rows(self, rows, ck=CK):
        seen = set()
        for r in rows:
            for k in C.REQUIRED_FIELDS:
                self.assertIn(k, r, k)
            self.assertEqual(r["full_text"], r["prompt"] + r["target"])
            self.assertLessEqual(len(r["full_text"]) + 1, C.MAX_LEN)
            self.assertLessEqual(len(r["full_text"]), ck.max_chars)
            self.assertIn(r["split"], ("train", "dev", "heldout"))
            self.assertTrue(r["id"].startswith("v49acq_"))
            self.assertNotIn(r["id"], seen)
            seen.add(r["id"])
            self.assertTrue(r["license"] and r["source"] and r["attribution"])
            self.assertIn(r["adoption"], ("candidate", "hold"))
            self.assertIsNone(C.validate_row(r, ck), r["id"])
            for q in r.get("quotes", []):
                self.assertIn(q, r["prompt"])
            for (a, b), step in zip(r["loss_mask_spans"], r["tool_steps"]):
                self.assertEqual(r["full_text"][a:b], step["observation"])
                self.assertGreaterEqual(a, len(r["prompt"]))
            self.assertEqual(len(r["loss_mask_spans"]), len(r["tool_steps"]))
            self.assertEqual(r["n_oov"], ck.oov_count(r["full_text"]))
        return rows

    def check_stats(self, st):
        s = C.summarize(st, CK)
        self.assertTrue(s["balanced"], s)
        self.assertEqual(st.n_in, st.n_out + sum(st.rejects.values()))
        json.dumps(s)
        return s


# ====================================================================================================== helpers
class TestHelpers(unittest.TestCase):
    def test_split_is_deterministic_and_about_96_1_3(self):
        c = Counter(C.split_of(f"k{i}") for i in range(20000))
        self.assertEqual(C.split_of("jaquad:手塚治虫"), C.split_of("jaquad:手塚治虫"))
        self.assertAlmostEqual(c["heldout"] / 20000, 0.03, delta=0.006)
        self.assertAlmostEqual(c["dev"] / 20000, 0.01, delta=0.004)
        self.assertAlmostEqual(c["train"] / 20000, 0.96, delta=0.01)

    def test_normalizer_matches_runtime_source_when_reachable(self):
        src = next((up / "src" / "v44fix" / "qwenputer_controller_v30.py" for up in HERE.parents
                    if (up / "src" / "v44fix" / "qwenputer_controller_v30.py").exists()), None)
        if src is None:
            self.skipTest("controller source not reachable")
        tree = ast.parse(src.read_text(encoding="utf-8"))
        lit = [n.value for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_BASE_NORM_DICT" for t in n.targets)]
        mapping = ast.literal_eval(lit[0])
        for code in range(0xFF01, 0xFF5F):
            if chr(code) not in ("（", "）", "！", "？", "：", "～"):
                mapping[chr(code)] = chr(code - 0xFEE0)
        self.assertEqual(str.maketrans(mapping), C.NORMALIZE)

    def test_vocab_and_oov(self):
        self.assertGreater(len(CK.vocab), 3000)
        self.assertEqual(CK.oov_count("問題: 答え 1+1=2。\n"), 0)
        self.assertEqual(CK.oov_count("a\U0001F600b"), 1)                      # emoji is outside the vocabulary
        self.assertEqual(CK.oov_count("ＡＢＣ１２３"), 0)                         # full-width ASCII is normalised by the runtime first
        st = CK.oov_stats("abc\U0001F600")
        self.assertAlmostEqual(st["frac"], 0.25)

    def test_iter_json_array_streams_across_chunk_boundaries(self):
        objs = [{"i": i, "t": "日本語" * (i % 7), "n": {"a": [1, 2, {"b": "]"}]}} for i in range(40)]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.json"
            p.write_text("[\n" + ",\n".join(json.dumps(o, ensure_ascii=False) for o in objs) + "\n]", encoding="utf-8")
            for chunk in (7, 64, 1 << 20):
                self.assertEqual(list(C.iter_json_array(p, chunk=chunk)), objs)
            q = Path(d) / "b.jsonl"
            q.write_text("\n".join(json.dumps(o) for o in objs[:3]) + "\n\n", encoding="utf-8")
            self.assertEqual(list(C.iter_json_array(q)), objs[:3])

    def test_static_check_is_the_executor_profile(self):
        ok = lambda code: CKX.static_check(code)                         # noqa: E731
        self.assertEqual(ok("import math\nprint(math.sqrt(4))"), (True, ""))
        # what the model's own executor refuses is refused here too (the earlier 'wide' profile accepted all of these)
        for code, why in [("import os\nprint(1)", "import:os"), ("import subprocess", "import:subprocess"), ("x = input()", "name:input"),
                          ("open('a')", "name:open"), ("eval('1')", "name:eval"), ("exec('1')", "name:exec"),
                          ("import random\nprint(random.random())", "import:random"), ("import re\nprint(re.compile('a'))", "import:re"),
                          ("import json\nprint(json.dumps(1))", "import:json"), ("import string\nprint(string.digits)", "import:string"),
                          ("import dataclasses", "import:dataclasses"), ("class A:\n    pass\n", "class_def:"),
                          ("import math\nmath.x = 1", "attr_store:"), ("print().__class__.__subclasses__()", "attr:__subclasses__"),
                          ("def (", "syntax:"), ("from . import x", "import:None"), ("__import__('os')", "name:__import__"),
                          ("from operator import attrgetter", "import_name:attrgetter"), ("g = (1 for _ in [1])\nprint(g.gi_frame)", "attr:gi_frame"),
                          ("s = '{0._' + '_x}'\nprint(s.format(1))", "format_string:")]:
            r = ok(code)
            self.assertFalse(r[0], code)
            self.assertTrue(r[1].startswith(why), (code, r[1]))
        self.assertEqual(ok("from fractions import Fraction\nprint(Fraction(1, 3))"), (True, ""))
        self.assertEqual(ok("print('{:>4}|{n}'.format(1, n=2))"), (True, ""))                  # str.format on a literal with plain fields

    def test_sandbox_runs_and_limits(self):
        r = CKX.run_python("print('こんにちは', 1 + 2)")
        self.assertEqual((r["ok"], r["stdout"]), (True, "こんにちは 3\n"))
        r = CKX.run_python("print(sorted({'b', 'a', 'c'}))")                          # fixed hash seed -> reproducible
        self.assertEqual(r["stdout"], "['a', 'b', 'c']\n")
        r = CKX.run_python("print(1/0)")
        self.assertFalse(r["ok"])
        self.assertTrue(r["reason"].startswith("exception:ZeroDivisionError"), r)
        slow = C.Checks(vocab=CK.vocab, sandbox_timeout=1.0)
        r = slow.run_python("while True:\n    pass\n")
        self.assertEqual((r["ok"], r["reason"]), (False, "timeout:1.0s"))
        r = CKX.run_python("import os\nprint(os.getcwd())")
        self.assertFalse(r["ok"])
        self.assertTrue(r["reason"].startswith("static:import:os"), r)

    def test_no_weaker_fallback(self):
        # without the sibling sandbox, or with a failing self-test, running code is impossible - never silently weaker
        with self.assertRaises(C.SandboxUnavailable):
            C.Checks(vocab=CK.vocab, use_external=False, run_code=True)
        ext = C._load_external_checks()
        real = ext.sandbox_selftest
        try:
            ext.sandbox_selftest = lambda **kw: {"ok": False, "run_as": None, "checks": [{"name": "print(1 + 1)", "ok": False, "detail": "boom"}]}
            with self.assertRaises(C.SandboxUnavailable) as cm:
                C.Checks(vocab=CK.vocab)
            self.assertIn("boom", str(cm.exception))
        finally:
            ext.sandbox_selftest = real
        off = C.Checks(vocab=CK.vocab, use_external=False, run_code=False)
        self.assertEqual((off.backends["sandbox"], off.run_python("print(1)")["reason"]), ("disabled", "sandbox_disabled"))
        for name in ("run_python_sandbox", "static_code_check", "ALLOWED_MODULES"):
            self.assertFalse(hasattr(C, name), name)

    def test_check_arithmetic(self):
        res = lambda t: [(s, ok) for s, ok in C.check_arithmetic(t)]      # noqa: E731
        self.assertEqual(res("合計は3 + 4 = 7です。"), [("3 + 4 = 7", True)])
        self.assertEqual(res("8＋2＝10 と 10−3＝7"), [("8+2 = 10", True), ("10-3 = 7", True)])
        self.assertEqual([ok for _, ok in res("5 + 5 = 11")], [False])
        self.assertEqual([ok for _, ok in res("10 / 3 = 3.33")], [True])       # rounded result accepted
        self.assertEqual([ok for _, ok in res("10 / 3 = 3.5")], [False])
        self.assertEqual([ok for _, ok in res("1,200 × 3 = 3,600")], [True])
        self.assertEqual(res("x + 3 + 4 = 10"), [])                           # equation in a variable: skipped
        self.assertEqual(res("2 + 3 = 5 + 1 = 6"), [])                        # chain: skipped
        self.assertEqual(res("a = 3 + 4 = 7"), [])
        self.assertEqual([ok for _, ok in res("(3 + 4) * 2 = 14")], [True])
        self.assertIsNone(C.safe_eval_fraction("__import__('os')"))
        self.assertIsNone(C.safe_eval_fraction("2 ** 100"))
        self.assertIsNone(C.safe_eval_fraction("1 / 0"))

    def test_parse_fences_and_sentences(self):
        t = "a\n```python\nprint(1)\n```\nb\n  ```C++\n  int x;\n  ```\nc"
        b = C.parse_fences(t)
        self.assertEqual([(x["lang"], x["code"]) for x in b], [("python", "print(1)"), ("cpp", "int x;")])
        self.assertEqual(t[b[0]["start"]:b[0]["end"]], "```python\nprint(1)\n```")
        self.assertIsNone(C.parse_fences("```python\nprint(1)"))
        self.assertIsNone(C.parse_fences("```python\n```python\nx\n```"))
        self.assertEqual(C.parse_fences("no code"), [])
        body = "AはBである。\nCはDだ（E）。」\nF"
        sp = C.sentence_spans(body)
        self.assertEqual([body[a:b].strip() for a, b in sp], ["AはBである。", "CはDだ（E）。」", "F"])

    def test_find_never_returns_dev_validation_or_do_not_train(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for rel in ["data/train/jaquad_train_0000.json", "data/train/jaquad_train_0001.json", "data/dev/jaquad_dev_0000.json",
                        "do_not_train_dev/x/data/train/jaquad_train_0009.json",
                        "distractor/train-00000-of-00002.parquet", "distractor/validation-00000-of-00001.parquet",
                        "fullwiki/train-00000-of-00002.parquet", "completed_sets.jsonl", "data.json"]:
                f = root / rel
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_bytes(b"[]")
            rel = lambda fs: sorted("/".join(p.relative_to(root).parts) for p in fs)      # noqa: E731
            self.assertEqual(rel(C.find_jaquad(root)), ["data/train/jaquad_train_0000.json", "data/train/jaquad_train_0001.json"])
            self.assertEqual(rel(C.find_hotpot(root)), ["distractor/train-00000-of-00002.parquet"])
            self.assertEqual(rel(C.find_tengentoppa(root)), ["completed_sets.jsonl"])
            self.assertEqual(rel(C.find_toolace(root)), ["data.json"])


# ====================================================================================================== (a) JaQuAD
JQ_OK = {"id": "tr-ok-00-000", "title": "題", "question_type": "Syntactic variation", "question": "山田太郎の出身地はどこ?",
         "context": "山田太郎は東京出身の作家である。花子は大阪で育った。", "answers": [{"text": "東京", "answer_start": 5, "answer_type": "Place"}]}


class TestJaquad(RowAsserts):
    def test_fixture(self):
        files = [FIX / "jaquad_train_fixture.json"]
        st = C.Stats("jaquad")
        spec = C.DATASETS["jaquad"]
        recs = list(C.read_jaquad(files))
        rows = self.check_rows(list(spec.convert(recs, st, CK, spec)))
        s = self.check_stats(st)
        # the five real rows: three are 'Multiple sentence reasoning' (the answer sentence is not the whole evidence), two have a character
        # outside the model vocabulary - none is a clean single-sentence row, so nothing is emitted from this fixture
        self.assertEqual(rows, [])
        self.assertEqual(s["rejects"], {"question_type_multiple_sentence": 3, "oov_in_quote_or_answer": 2})
        self.assertEqual(s["question_types"], {"Multiple sentence reasoning": 3, "Syntactic variation": 2})
        self.assertEqual(len(list(C.read_jaquad(files, limit=2))), 2)
        self.assertEqual([i for i, _ in C.read_jaquad(files, start=3)], [3, 4])                  # a shard keeps the global record index

    def test_happy_path_rows(self):
        rows, st = run("jaquad", [JQ_OK])
        self.check_rows(rows)
        r = rows[0]
        self.assertEqual(r["family"], "jaquad_human")
        self.assertTrue(r["id"].startswith("v49acq_jaquad_tr-"))
        self.assertIn(r["reference_final_answer"], r["quotes"][0])                      # answer inside the one evidence sentence
        self.assertTrue(r["target"].endswith(f"答え: {r['reference_final_answer']}。"))
        self.assertEqual(r["loss_mask_spans"], [])
        self.assertEqual(r["split"], C.split_of(r["group_key"]))

    def test_card_example_and_evidence_checks(self):
        # the dataset card's own validation example: the question names ビワタナゴ, the evidence sentence never does
        card = {"id": "tr-card", "title": "イタセンパラ", "question": "ビワタナゴの正式名称は何?", "question_type": "Multiple sentence reasoning",
                "context": "イタセンパラ(板鮮腹、Acheilognathuslongipinnis)は、コイ科のタナゴ亜科タナゴ属に分類される淡水魚の一種。\n別名はビワタナゴ(琵琶鱮、琵琶鰱)。",
                "answers": [{"text": "イタセンパラ", "answer_start": 0, "answer_type": "Object"}]}
        rows, st = run("jaquad", [card])
        self.assertEqual((rows, dict(st.rejects)), ([], {"question_type_multiple_sentence": 1}))
        # a type that is not filtered: the content n-gram check still stops a question whose words the evidence sentence does not contain
        no_type = dict(card, question_type="Lexical variation (synonymy)", question="琵琶湖固有の鯉の名称は何?")
        rows, st = run("jaquad", [no_type])
        self.assertEqual((rows, dict(st.rejects)), ([], {"quote_shares_no_content_with_question": 1}))
        ambiguous = dict(JQ_OK, question_type="Ambiguous")
        rows, st = run("jaquad", [ambiguous])
        self.assertEqual(dict(st.rejects), {"question_type_ambiguous": 1})
        self.assertTrue(C.shares_content("太郎はどこで生まれた?", "太郎は東京で生まれた。"))
        self.assertFalse(C.shares_content("何ですか?", "太郎は東京で生まれた。"))
        self.assertFalse(C.shares_content("ビワタナゴの正式名称は何?", "イタセンパラは淡水魚の一種。"))

    def test_guards_and_rejects(self):
        base = {"id": "tr-000-00-000", "title": "題", "context": "太郎は東京で生まれた。\n花子は大阪で育った。", "question": "太郎はどこで生まれた?",
                "question_type": "x", "answers": [{"text": "東京", "answer_start": 3, "answer_type": "Place"}]}
        rows, st = run("jaquad", [base])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["quotes"], ["太郎は東京で生まれた。"])
        dev = dict(base, id="de-001-00-000")
        long_ctx = dict(base, id="tr-9", context=base["context"] + "あ" * 701)
        bad_off = dict(base, id="tr-8", answers=[{"text": "東京", "answer_start": 4, "answer_type": "Place"}])
        no_ans = dict(base, id="tr-7", answers=[])
        newline = dict(base, id="tr-6", answers=[{"text": "れた。\n花子", "answer_start": 8, "answer_type": "x"}])
        cross = dict(base, id="tr-4", context="甲は乙である。丙は丁である。", answers=[{"text": "る。丙", "answer_start": 5, "answer_type": "x"}])
        view = dict(base, id="tr-5", answers={"text": ["東京"], "answer_start": [3], "answer_type": ["Place"]})   # datasets-server layout
        rows, st = run("jaquad", [dev, long_ctx, bad_off, no_ans, newline, cross, view])
        s = self.check_stats(st)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["id"], "v49acq_jaquad_tr-5")
        self.assertEqual(s["rejects"], {"not_train_id": 1, "passage_too_long": 1, "offset_mismatch": 1, "no_answer": 1, "answer_has_newline": 1, "answer_spans_sentences": 1})

    def test_group_split_keeps_an_article_together(self):
        recs = []
        for art in range(30):
            for q in range(3):
                recs.append({"id": f"tr-{art:03d}-00-{q:03d}", "title": f"記事{art}", "context": "山田太郎は東京出身の作家である。花子は大阪で育った。",
                             "question": "山田太郎の出身地はどこ?", "question_type": "x", "answers": [{"text": "東京", "answer_start": 5, "answer_type": "O"}]})
        rows, _ = run("jaquad", recs)
        self.assertEqual(len(rows), 90)
        by = {}
        for r in rows:
            by.setdefault(r["group_key"], set()).add(r["split"])
        self.assertEqual(len(by), 30)
        self.assertTrue(all(len(v) == 1 for v in by.values()))


# ====================================================================================================== (b) coding
PY_PLAN = "以下のPythonコードは、1から10までの合計を計算して表示します。"


PY_OK = msg("1から10までの合計を表示するPythonコードを書いて。", f"{PY_PLAN}\n{FENCE}python\nprint(sum(range(1, 11)))\n{FENCE}\n出力:\n{FENCE}\n55\n{FENCE}\n結果は55です。")
RUST_OK = msg("Rustで挨拶を表示する関数を書いてください。", f"次のように書けます。\n{FENCE}rust\nfn main() {{\n    println!(\"こんにちは\");\n}}\n{FENCE}\nこれで標準出力に表示されます。")


class TestCoding(RowAsserts):
    def test_fixture_through_parquet(self):
        if pa is None:
            self.skipTest("pyarrow missing")
        recs = fixture("coding_fixture.json") + [PY_OK, RUST_OK]            # the real python row of the fixture has no stored output (dropped); two clean rows added
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "data" / "train-00000-of-00001.parquet"
            p.parent.mkdir()
            pq.write_table(pa.Table.from_pylist(recs), p)
            files = C.find_coding(d)
            self.assertEqual(len(files), 1)
            spec = C.DATASETS["coding"]
            st = C.Stats("coding")
            rows = self.check_rows(list(spec.convert(C.read_messages_parquet(files), st, CKX, spec)), CKX)
            self.assertEqual(len(list(C.read_messages_parquet(files, limit=3))), 3)
            self.assertEqual([i for i, _ in C.read_messages_parquet(files, start=len(recs) - 2)], [len(recs) - 2, len(recs) - 1])
        s = self.check_stats(st)
        fam = Counter(r["family"] for r in rows)
        self.assertGreaterEqual(fam["coding_pytool"], 1)
        self.assertGreaterEqual(fam["code_text"], 1)
        for reason in ("row_too_long", "static:name:input", "untagged_code_fence", "no_code_block", "english_row_too_long",
                       "no_stored_output_to_corroborate"):
            self.assertIn(reason, s["rejects"], s["rejects"])
        for r in rows:
            if r["family"] == "coding_pytool":
                a, b = r["loss_mask_spans"][0]
                obs = r["tool_steps"][0]["observation"]
                self.assertEqual(r["full_text"][a:b], obs)
                self.assertEqual(obs, f"\n{FENCE}output\n{r['expected_stdout']}{FENCE}\n")
                self.assertEqual(r["tool_steps"][0]["stdout"], r["expected_stdout"])
                # the observation really is what the stored code prints (re-run it)
                again = CKX.run_python(r["python_code"])
                self.assertTrue(again["ok"])
                self.assertEqual(again["stdout"].replace("\r\n", "\n"), r["expected_stdout"])
                self.assertEqual(r["verification"], "stored_output_equals_sandbox")
                self.assertEqual(r["sandbox_backend"], "checks.code_sandbox_check(profile=executor)")
                self.assertIn(f"{FENCE}python\n{r['python_code']}\n{FENCE}", r["target"])
                self.assertTrue(r["target"].endswith(f"答え: {r['reference_final_answer']}。"))
                self.assertEqual(r["answer_type"], "output_last_line")
                self.assertEqual(C.NORMALIZE and r["full_text"].translate(C.NORMALIZE), r["full_text"])
                self.assertEqual(r["n_oov"], 0)
            else:
                self.assertEqual(r["loss_mask_spans"], [])
                self.assertEqual(r["tool_steps"], [])
                self.assertEqual(r["verification"], "unverified_no_execution")
                blocks = C.parse_fences(r["target"])
                self.assertTrue(blocks and all(b["lang"] for b in blocks))
                self.assertNotIn("python", [b["lang"] for b in blocks])
                self.assertIn(r["code_lang"], C.CODE_LANGS)
        self.assertTrue(set(s["code_text_languages"]))

    def test_python_row_variants(self):
        ok = msg("1から10までの合計を表示するPythonコードを書いて。", f"{PY_PLAN}\n{FENCE}python\nprint(sum(range(1, 11)))\n{FENCE}\n出力:\n{FENCE}\n55\n{FENCE}\n結果は55です。")
        mism = msg("sum 1..10 in Python", f"Code:\n{FENCE}python\nprint(sum(range(1, 11)))\n{FENCE}\nOutput:\n{FENCE}\n56\n{FENCE}")
        nostore = msg("sum 1..10 in Python", f"Code:\n{FENCE}python\nprint(sum(range(1, 11)))\n{FENCE}\nThat prints the sum.")
        contra = msg("sum 1..10 in Python", f"Code:\n{FENCE}python\nprint(sum(range(1, 11)))\n{FENCE}\nThe output is 66.")
        inp = msg("read a number", f"{FENCE}python\nn = int(input())\nprint(n)\n{FENCE}")
        osimp = msg("list files", f"{FENCE}python\nimport os\nprint(os.listdir('.'))\n{FENCE}")
        err = msg("divide", f"{FENCE}python\nprint(1 / 0)\n{FENCE}")
        silent = msg("define f", f"{FENCE}python\nx = 1\n{FENCE}")
        quiet = msg("quiet", f"{FENCE}python\nprint('x') if False else None\n{FENCE}")
        addr = msg("obj", f"{FENCE}python\nclass A: pass\nprint(A())\n{FENCE}")
        multi = msg("two lines", f"{FENCE}python\nprint('first')\nprint('last line')\n{FENCE}")
        multi = msg("two lines", f"{FENCE}python\nprint('first')\nprint('last line')\n{FENCE}\n出力:\n{FENCE}\nfirst\nlast line\n{FENCE}")
        rows, st = run("coding", [ok, mism, nostore, contra, inp, osimp, err, silent, quiet, addr, multi], ck=CKX)
        s = self.check_stats(st)
        self.check_rows(rows, CKX)
        by_id = {r["id"]: r for r in rows}
        r0, r10 = by_id["v49acq_coding_0000000"], by_id["v49acq_coding_0000010"]
        self.assertEqual(r0["verification"], "stored_output_equals_sandbox")
        self.assertEqual(r0["plan_line"], PY_PLAN[:-1])
        self.assertEqual(r0["reference_final_answer"], "55")
        self.assertEqual(r10["reference_final_answer"], "last line")
        self.assertEqual(r10["plan_line"], C.PLAN_FALLBACK)
        self.assertEqual(sorted(by_id), ["v49acq_coding_0000000", "v49acq_coding_0000010"])
        # a row that runs but has no stored output is NOT kept (default) - it is counted, so the report can say how many there were
        self.assertEqual(s["rejects"], {"stored_output_mismatch": 1, "no_stored_output_to_corroborate": 2, "static:name:input": 1,
                                        "static:import:os": 1, "sandbox_failed:exception:ZeroDivisionError: division by zero": 1,
                                        "static:no_print": 1, "static:class_def:class definitions are disabled": 1, "empty_stdout": 1})
        self.assertEqual(s["verification"], {"stored_output_equals_sandbox": 2})
        self.assertEqual(s["python"], {"rows_kept_stored_output_equals_sandbox": 2, "ran_in_sandbox_but_had_no_stored_output": 2,
                                       "rows_rejected_code_crash": 0, "note": s["python"]["note"]})
        self.assertEqual(C.summarize(st, CKX)["backends"]["sandbox"], "checks.code_sandbox_check(profile=executor)")

    def test_policy_switches(self):
        nostore = msg("sum 1..10 in Python", f"Code:\n{FENCE}python\nprint(sum(range(1, 11)))\n{FENCE}\nThat prints the sum.")
        rows, st = run("coding", [nostore], ck=CKX)                                   # the default: stored output required
        self.assertEqual((len(rows), dict(st.rejects)), (0, {"no_stored_output_to_corroborate": 1}))
        lenient = C.Checks(vocab=CK.vocab, python_require_stored_output=False)
        rows, st = run("coding", [nostore], ck=lenient)
        self.assertEqual([r["verification"] for r in rows], ["sandbox_only_no_stored_output"])
        contra = msg("sum 1..10 in Python", f"Code:\n{FENCE}python\nprint(sum(range(1, 11)))\n{FENCE}\nThe output is 66.")
        rows, st = run("coding", [contra], ck=lenient)                                # no stored output, but the prose contradicts the output
        self.assertEqual(dict(st.rejects), {"prose_number_contradicts_output": 1})
        off = C.Checks(vocab=CK.vocab, use_external=False, run_code=False)
        rows, st = run("coding", [nostore], ck=off)
        self.assertEqual(dict(st.rejects), {"sandbox_failed:sandbox_disabled": 1})
        self.assertEqual(off.backends["sandbox"], "disabled")

    def test_english_rows_are_short_only(self):
        short_en = msg("Write Rust code that prints hello.", f"Here it is:\n{FENCE}rust\nfn main() {{ println!(\"hello\"); }}\n{FENCE}")
        long_en = msg("Write Rust code that prints hello.", "This program is explained in a lot of detail. " * 20 + f"\n{FENCE}rust\nfn main() {{ println!(\"hello\"); }}\n{FENCE}")
        long_en_user = msg("Please write a Rust program " + "that prints hello and explains each step in detail " * 14, f"{FENCE}rust\nfn main() {{ println!(\"hello\"); }}\n{FENCE}")
        ja_long = msg("Rustで挨拶を表示するプログラムを書いてください。", "このプログラムは詳しい説明つきです。" * 30 + f"\n{FENCE}rust\nfn main() {{ println!(\"hello\"); }}\n{FENCE}")
        rows, st = run("coding", [short_en, long_en, long_en_user, ja_long])
        s = self.check_stats(st)
        self.assertEqual([r["id"] for r in rows], ["v49acq_coding_0000000", "v49acq_coding_0000003"])
        self.assertEqual(s["rejects"], {"english_row_too_long": 2})
        self.assertEqual((rows[0]["user_lang"], rows[0]["answer_lang"]), ("en", "en"))
        self.assertEqual((rows[1]["user_lang"], rows[1]["answer_lang"]), ("ja", "ja"))              # language of the prose, not of the code
        self.assertEqual(s["english_rows_total"], 1)
        self.assertEqual(sorted(s["english"]), ["code_text"])
        self.assertGreater(s["english"]["code_text"]["supervised_chars"], 0)
        self.assertEqual(sum(s["supervised_chars"].values()), sum(len(r["target"]) for r in rows))

    def test_code_text_rows(self):
        rust = RUST_OK
        untagged = msg("C", f"{FENCE}\nint x;\n{FENCE}")
        too_long = msg("long", f"{FENCE}cpp\n" + "int x;\n" * 400 + FENCE)
        pyfail = msg("p", f"{FENCE}python\nimport numpy\nprint(1)\n{FENCE}")
        text_only = msg("hi", "ただの説明です。")
        multiturn = {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": f"{FENCE}java\nint a;\n{FENCE}"},
                                  {"role": "user", "content": "more"}, {"role": "assistant", "content": "ok"}]}
        rows, st = run("coding", [rust, untagged, too_long, pyfail, text_only, multiturn], ck=CKX)
        s = self.check_stats(st)
        self.check_rows(rows, CKX)
        self.assertEqual([r["code_lang"] for r in rows], ["rust", "java"])
        self.assertEqual(rows[0]["answer_lang"], "ja")
        self.assertEqual(s["extra"]["multiturn_first_pair_only"], 1)
        self.assertEqual(s["rejects"], {"untagged_code_fence": 1, "row_too_long": 1, "static:import:numpy": 1, "no_code_block": 1})
        self.assertEqual(s["code_text_languages"], {"rust": 1, "java": 1})
        self.assertEqual(rows[0]["target"], "\n" + rust["messages"][1]["content"])

    def test_a_pathological_python_row_is_one_reject_and_not_the_end_of_the_run(self):
        # round 6.  The old checker recursed once per link of an attribute chain, so a chain of about 1,000 links made checks.static_check RAISE
        # RecursionError.  What that did to the converter depends on the stack the converter is called from, and the row cap (1,900 characters, about 570
        # links) keeps a chain from reaching the sandbox when the stack is shallow.  Measured on Python 3.12 with the old checker (the sweep is kept in the
        # r6 notes): a chain that passes the first static_check (convert._coding_python_row) and overflows in the SECOND one (code_sandbox_check, called
        # from Checks._sandbox) becomes SandboxUnavailable and STOPS the whole run; with 520 frames already on the stack that was the 470-link chain
        # (1,453 bytes).  Others only became 'exception:RecursionError' rows (wrong category).  Here the converter is called from a stack that leaves
        # about 480 frames, and every chain length from 380 to 570 links is converted: each must be an ordinary 'static:too_deep' reject.
        def py(code):
            return msg("p", f"{FENCE}python\n{code}\n{FENCE}")
        chains = [py("import math\nx = math" + ".pi" * n + "\nprint(x)") for n in range(380, 571)]
        spec = py("print('" + "{:" * 1100 + "{}" + "}" * 1100 + "'.format(1))")
        deep = py("x = " + "+".join(["1"] * 1000) + "\nprint(x)")
        frame, used = sys._getframe(), 0
        while frame is not None:
            used, frame = used + 1, frame.f_back

        def from_a_deep_stack(k):
            if k > 0:
                return from_a_deep_stack(k - 1)
            return run("coding", chains[:100] + [RUST_OK] + chains[100:] + [spec, deep], ck=CKX)      # run() raises SandboxUnavailable if the run would stop
        rows, st = from_a_deep_stack(max(0, sys.getrecursionlimit() - used - 480))
        s = self.check_stats(st)
        self.assertEqual([r["code_lang"] for r in rows], ["rust"])
        self.assertTrue(all(k.startswith("static:") for k in s["rejects"]), s["rejects"])
        self.assertEqual({k.split(":")[1]: v for k, v in s["rejects"].items()}, {"too_deep": len(chains) + 1, "format_string": 1}, s["rejects"])
        # the same rows from a normal stack
        rows, st = run("coding", [chains[0], RUST_OK, chains[-1], spec, deep], ck=CKX)
        self.assertEqual([r["code_lang"] for r in rows], ["rust"])
        self.assertEqual({k.split(":")[1]: v for k, v in dict(st.rejects).items()}, {"too_deep": 3, "format_string": 1})

    def test_masked_row_must_be_clean(self):
        emoji_user = msg("print 1 \U0001F600", f"{FENCE}python\nprint(1)\n{FENCE}\n出力:\n{FENCE}\n1\n{FENCE}")
        fullwidth_out = msg("print", f"{FENCE}python\nprint('１２')\n{FENCE}\n出力:\n{FENCE}\n１２\n{FENCE}")
        rows, st = run("coding", [emoji_user, fullwidth_out], ck=CKX)
        self.assertEqual(len(rows), 0)
        self.assertEqual(dict(st.rejects), {"invalid_row:masked_row_not_clean": 2})


# ====================================================================================================== (c) Tengentoppa
class TestTengentoppa(RowAsserts):
    def test_fixture(self):
        recs = [json.loads(x) for x in (FIX / "tengentoppa_fixture.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        rows, st = run("tengentoppa", recs)
        self.check_rows(rows)
        s = self.check_stats(st)
        self.assertGreaterEqual(len(rows), 1)
        for r in rows:
            self.assertTrue(r["target"].endswith(f"答え: {r['reference_final_answer']}"))
            self.assertEqual(r["quality_flag"], "qwen2.5-32b generated, unverified")
            self.assertNotIn("initial_answer", r["full_text"])
            self.assertIn(r["family"], ("tengentoppa_answer", "tengentoppa_steps"))
        self.assertEqual(s["families"].get("tengentoppa_steps", 0) + s["families"].get("tengentoppa_answer", 0), len(rows))

    def test_rules(self):
        steps = "1. **まず** 条件を整理します。\n\n2. **次に** 理由を述べます。"
        good = {"instruction": "日本の首都について説明してください。", "reasoning": steps, "initial_answer": "初期回答です。", "refined_answer": "日本の首都は東京です。"}
        no_steps = dict(good, reasoning="ただの段落でステップではない。")
        rnd = random.Random(1)
        too_long = dict(good, refined_answer="".join(chr(0x3041 + rnd.randrange(80)) for _ in range(1900)) + "。")
        refusal = dict(good, refined_answer="申し訳ありませんが、お答えできません。")
        refusal2 = dict(good, refined_answer="その件については、お答えできません。")
        boiler = dict(good, refined_answer="もちろんです。日本の首都は東京です。")
        latin = dict(good, refined_answer="Tokyo is the capital of Japan and it is a very large city.")
        unterminated = dict(good, refined_answer="日本の首都は東京で")
        empty = dict(good, instruction="")
        repetitive = dict(good, refined_answer="東京は大きい。" * 200)
        long_steps = dict(good, reasoning="1. 手順一\n2. 手順二\n" + "あ" * 1850)
        fenced = dict(good, instruction="JSONで答えてください。", reasoning="", refined_answer="```json\n{\"首都\": \"東京\"}\n```")
        as_json = dict(good, instruction="JSONで答えて。", reasoning="", refined_answer="{\"首都\": \"東京\"}")
        cut_fence = dict(good, reasoning="", refined_answer="コードです。\n```json\n{\"首都\": \"東")
        cut_json = dict(good, reasoning="", refined_answer="{\"首都\": \"東京\", \"人口\": [1, 2}")
        rows, st = run("tengentoppa", [good, no_steps, too_long, refusal, refusal2, boiler, latin, unterminated, empty, repetitive, long_steps, {"__bad_json__": "x"},
                                       fenced, as_json, cut_fence, cut_json])
        s = self.check_stats(st)
        self.check_rows(rows)
        fam = {r["id"]: r["family"] for r in rows}
        self.assertEqual(fam, {"v49acq_tengentoppa_0000000": "tengentoppa_steps", "v49acq_tengentoppa_0000001": "tengentoppa_answer",
                               "v49acq_tengentoppa_0000010": "tengentoppa_answer", "v49acq_tengentoppa_0000012": "tengentoppa_answer",
                               "v49acq_tengentoppa_0000013": "tengentoppa_answer"})
        fmt = {r["id"][-2:]: r["answer_format"] for r in rows}
        self.assertEqual(fmt, {"00": "prose", "01": "prose", "10": "prose", "12": "fenced", "13": "json"})
        self.assertEqual(rows[0]["target"], f"\n手順: {steps}\n答え: 日本の首都は東京です。")
        self.assertEqual(rows[1]["target"], "\n答え: 日本の首都は東京です。")
        self.assertEqual(s["rejects"], {"row_too_long": 1, "refusal_opening": 2, "boilerplate_opening": 1, "latin_heavy": 1,
                                        "answer_not_terminated": 3, "empty_field": 1, "repetitive": 1, "bad_json_line": 1})
        never = C.Checks(vocab=CK.vocab, use_external=False, run_code=False, tengentoppa_steps=False)
        rows, _ = run("tengentoppa", [good], ck=never)
        self.assertEqual(rows[0]["family"], "tengentoppa_answer")

    def test_python_blocks_are_dropped_not_kept_unverified(self):
        steps = "1. **まず** 条件を整理します。\n\n2. **次に** 理由を述べます。"
        good = {"instruction": "合計を求めるコードを書いてください。", "reasoning": steps, "initial_answer": "x", "refined_answer": "合計は次のとおりです。"}
        in_answer = dict(good, refined_answer=f"次のコードです。\n{FENCE}python\nprint(1 + 2)\n{FENCE}\n出力は3です。")
        in_steps = dict(good, reasoning="1. 手順一\n" + f"{FENCE}py\nprint(1)\n{FENCE}\n2. 手順二")
        rows, st = run("tengentoppa", [good, in_answer, in_steps])
        self.assertEqual(dict(st.rejects), {"python_block_not_executed": 1})
        self.assertEqual({r["id"]: r["family"] for r in rows}, {"v49acq_tengentoppa_0000000": "tengentoppa_steps", "v49acq_tengentoppa_0000002": "tengentoppa_answer"})
        self.assertNotIn("print(1)", rows[1]["full_text"])                       # the steps with the python block were left out, the answer kept

    def test_jsonl_reader_streams(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "completed_sets.jsonl"
            p.write_text('{"instruction": "a", "refined_answer": "b。"}\nnot json\n\n{"instruction": "c"}\n', encoding="utf-8")
            recs = list(C.read_tengentoppa([p]))
            self.assertEqual([i for i, _ in recs], [0, 1, 2])
            self.assertIn("__bad_json__", recs[1][1])


# ====================================================================================================== (d) HotpotQA
def hp(rid="x1", q="Which is older, A or B?", ans="A", supp=(("A", 0), ("B", 0)), extra=None, a_sents=None):
    paras = {"A": a_sents or ["A is an old town in Spain.", " It was founded in 1100."], "B": ["B is a new town in Spain.", " It was founded in 1900."],
             "C": ["C is a river.", " It is long."], "D": ["D is a lake."], "E": ["E is a hill.", " It is green.", " It is high."]}
    if extra:
        paras.update(extra)
    return {"id": rid, "question": q, "answer": ans, "type": "comparison", "level": "easy",
            "supporting_facts": {"title": [t for t, _ in supp], "sent_id": [i for _, i in supp]},
            "context": {"title": list(paras), "sentences": list(paras.values())}}


class TestHotpot(RowAsserts):
    def test_fixture_through_parquet(self):
        if pa is None:
            self.skipTest("pyarrow missing")
        recs = fixture("hotpot_fixture.json")
        with tempfile.TemporaryDirectory() as d:
            for rel in ("distractor/train-00000-of-00001.parquet", "distractor/validation-00000-of-00001.parquet", "fullwiki/train-00000-of-00001.parquet"):
                p = Path(d) / rel
                p.parent.mkdir(exist_ok=True)
                pq.write_table(pa.Table.from_pylist(recs), p)
            files = C.find_hotpot(d)
            self.assertEqual([p.name for p in files], ["train-00000-of-00001.parquet"])
            self.assertEqual(files[0].parent.name, "distractor")
            spec = C.DATASETS["hotpot"]
            st = C.Stats("hotpot")
            rows = self.check_rows(list(spec.convert(C.read_hotpot(files), st, CK, spec)))
        s = self.check_stats(st)
        self.assertGreaterEqual(len(rows), 2)
        self.assertEqual(s["n_in"], len(recs))
        self.assertEqual(sum(s["fit"]["by_trim_level"].values()), len(rows))
        for r in rows:
            self.assertEqual(r["family"], "hotpot_en_short")
            self.assertEqual(r["structural_template"], "acq_hotpot_en_short")
            self.assertTrue(r["target"].startswith("\n探す: "))
            self.assertTrue(r["target"].endswith(f"答え: {r['reference_final_answer']}。"))
            self.assertEqual(r["target"].count("根拠: 「"), len(r["quotes"]))
            self.assertTrue(r["prompt"].startswith("問題: 次の英語の段落を読んで"))

    def test_trimming_and_checks(self):
        rows, st = run("hotpot", [hp()])
        r = rows[0]
        self.assertEqual(r["trim_level"], 0)
        self.assertEqual(sorted(r["quotes"]), ["A is an old town in Spain.", "B is a new town in Spain."])
        self.assertLess(r["prompt"].index(r["quotes"][0]), r["prompt"].index(r["quotes"][1]))     # quotes follow the prompt's paragraph order
        self.assertEqual(r["answer_type"], "span_en")
        self.assertIn("段落", r["target"])
        self.assertEqual(len(re.findall(r"段落\d+（", r["prompt"])), 4)            # 2 supporting + 2 distractors
        self.assertEqual(len(re.findall(r"段落\d+（", r["target"])), 2)
        big = ["Sentence number %d is a longer padded sentence about the town." % i for i in range(40)]
        rows, st = run("hotpot", [hp(a_sents=big, supp=(("A", 0), ("B", 0)))])
        self.assertEqual(len(rows), 1)
        self.assertGreaterEqual(rows[0]["trim_level"], 0)
        self.assertLessEqual(len(rows[0]["full_text"]), CK.max_chars)
        hugeA = ["Sentence %d " % i + "x" * 80 + "." for i in range(30)]
        rows, st = run("hotpot", [hp(a_sents=hugeA, supp=(("A", 29), ("B", 0)))])
        self.assertEqual(rows[0]["trim_level"], 3)                                  # paragraph opening + the supporting sentence only
        self.assertIn(hugeA[29], rows[0]["quotes"])
        self.assertIn(hugeA[0], rows[0]["prompt"])
        self.assertNotIn(hugeA[5], rows[0]["prompt"])
        rows, st = run("hotpot", [hp(a_sents=["Z" * 2000 + "."], supp=(("A", 0), ("B", 0)))])
        self.assertEqual((len(rows), dict(st.rejects)), (0, {"row_too_long": 1}))
        rows, st = run("hotpot", [hp(a_sents=["A " + "y" * 900 + ".", " B " + "z" * 900 + "."], supp=(("A", 0), ("A", 1), ("B", 0)))])
        self.assertEqual(dict(st.rejects), {"row_too_long": 1})
        rows, st = run("hotpot", [hp(a_sents=hugeA, supp=(("A", 3), ("B", 0)))])
        self.assertEqual(rows[0]["trim_level"], 0)                                  # supporting paragraph cut after its last supporting sentence
        self.assertIn(hugeA[3], rows[0]["quotes"])
        self.assertIn(hugeA[2], rows[0]["prompt"])
        self.assertNotIn(hugeA[4], rows[0]["prompt"])

    def test_rejects_and_yes_no(self):
        yn = hp(rid="y", ans="no")
        notin = hp(rid="n", ans="Zanzibar")
        three = hp(rid="t", supp=(("A", 0), ("B", 0), ("C", 0)))
        one = hp(rid="o", supp=(("A", 0), ("A", 1)))
        badid = hp(rid="b", supp=(("A", 9), ("B", 0)))
        oov = hp(rid="v", extra={"B": ["B is a new town \U0001F600 in Spain.", " x."]})
        rows, st = run("hotpot", [yn, notin, three, one, badid, oov])
        s = self.check_stats(st)
        self.assertEqual([(r["id"], r["answer_type"]) for r in rows], [("v49acq_hotpot_y", "yes_no_en")])
        self.assertEqual(s["rejects"], {"answer_not_in_evidence": 1, "not_two_support_paragraphs": 2, "bad_supporting_fact": 1, "oov_in_quote_or_answer": 1})

    def test_group_is_supporting_title(self):
        rows, _ = run("hotpot", [hp(rid="a"), hp(rid="b")])
        self.assertEqual(rows[0]["group_key"], "hotpot:A")
        self.assertEqual(rows[0]["split"], rows[1]["split"])

    def test_questions_sharing_a_supporting_paragraph_share_a_group(self):
        if pa is None:
            self.skipTest("pyarrow missing")
        # q1 uses A+B, q2 uses B+C, q3 uses C+D (a chain: one component), q4 uses X+Y (separate); with the plain key min(title)
        # q1 -> A, q2 -> B, q3 -> C: three groups that can land in different splits.  The union-find root joins them.
        ex = {"X": ["X is a city."], "Y": ["Y is a lake."]}
        recs = [hp(rid="q1", supp=(("A", 0), ("B", 0))), hp(rid="q2", supp=(("B", 0), ("C", 0))), hp(rid="q3", supp=(("C", 0), ("D", 0))),
                hp(rid="q4", supp=(("X", 0), ("Y", 0)), extra=ex)]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "distractor" / "train-00000-of-00001.parquet"
            p.parent.mkdir()
            pq.write_table(pa.Table.from_pylist(recs), p)
            roots = C.build_hotpot_roots([p])
            self.assertEqual({t: roots[t] for t in "ABCDXY"}, {"A": "A", "B": "A", "C": "A", "D": "A", "X": "X", "Y": "X"})
            ck = C.Checks(vocab=CK.vocab, use_external=False, run_code=False)
            ck.hotpot_roots = roots
            rows, _ = run("hotpot", recs, ck=ck)
        self.assertEqual([r["group_key"] for r in rows], ["hotpot:A", "hotpot:A", "hotpot:A", "hotpot:X"])
        self.assertEqual(len({r["split"] for r in rows[:3]}), 1)


# ====================================================================================================== (e) ToolACE
FUNCS = [{"name": "Get Weather", "description": "Returns the weather for a city.", "parameters": {"type": "dict", "properties": {"city": {"description": "City", "type": "string"}, "days": {"description": "Days", "type": "int"}}, "required": ["city"]}, "required": None},
         {"name": "Get Time", "description": "Returns time.", "parameters": {"type": "dict", "properties": {}, "required": []}, "required": None}]


def system_of(funcs):
    return ("You are an expert in composing functions.\nHere is a list of functions in JSON format that you can invoke:\n"
            + json.dumps(funcs) + ". \nShould you decide to return the function call(s). \nPut it in the format of [func1(params_name=params_value)]\n\nNO other text MUST be included. \n")


def tc(conv, funcs=FUNCS):
    return {"system": system_of(funcs), "conversations": [{"from": a, "value": b} for a, b in conv]}


class TestToolace(RowAsserts):
    def test_fixture(self):
        recs = fixture("toolace_fixture.json")
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "data.json"
            p.write_text(json.dumps(recs, ensure_ascii=False), encoding="utf-8")
            files = C.find_toolace(d)
            spec = C.DATASETS["toolace"]
            st = C.Stats("toolace")
            rows = self.check_rows(list(spec.convert(C.read_toolace(files), st, CK, spec)))
        s = self.check_stats(st)
        self.assertGreaterEqual(len(rows), 1)
        self.assertEqual(s["language"]["parsed_en"] + s["language"]["parsed_zh"] + s["language"]["parsed_ja"] + s["language"]["parsed_other"], len(recs))
        for r in rows:
            self.assertEqual(r["adoption"], "hold")
            self.assertIn("generator", r["adoption_note"])
            if r["family"] == "toolace_call":
                self.assertEqual(len(r["loss_mask_spans"]), r["n_calls"])
                for (a, b), step in zip(r["loss_mask_spans"], r["tool_steps"]):
                    self.assertTrue(r["full_text"][a:b].startswith("\n結果: "))
                    self.assertIn(step["result"], r["full_text"][a:b])
                    self.assertNotIn(step["result"], r["target"][:a - len(r["prompt"])])    # the result is only inside the masked span

    def test_shapes(self):
        call = tc([("user", "Weather in Paris?"), ("assistant", "[Get Weather(city=\"Paris\")]"), ("tool", "[{\"name\": \"Get Weather\", \"results\": {\"t\": 20}}]"),
                   ("assistant", "It is 20 degrees."), ("user", "thanks"), ("assistant", "welcome")])
        two = tc([("user", "Weather and time?"), ("assistant", "[Get Weather(city=\"Paris\")]"), ("tool", "[{\"t\": 20}]"),
                  ("assistant", "[Get Time()]"), ("tool", "[{\"time\": \"10:00\"}]"), ("assistant", "20 degrees at 10:00.")])
        only = tc([("user", "Weather?"), ("assistant", "[Get Weather(city=\"Paris\")]")])
        nocall = tc([("user", "Hello"), ("assistant", "Hi, which city do you mean?")])
        empty = tc([("user", "Hello"), ("assistant", "")])
        no_final = tc([("user", "Weather?"), ("assistant", "[Get Weather(city=\"Paris\")]"), ("tool", "[{}]")])
        nofn = {"system": "no function list", "conversations": [{"from": "user", "value": "x"}, {"from": "assistant", "value": "y"}]}
        zh = tc([("user", "请告诉我巴黎的天气怎么样？"), ("assistant", "[Get Weather(city=\"巴黎\")]"), ("tool", "[{\"温度\": 20}]"), ("assistant", "巴黎今天二十度。")])
        rows, st = run("toolace", [call, two, only, nocall, empty, no_final, nofn, zh])
        s = self.check_stats(st)
        self.check_rows(rows)
        fam = {r["id"]: r["family"] for r in rows}
        self.assertEqual(fam, {"v49acq_toolace_000000": "toolace_call", "v49acq_toolace_000001": "toolace_call", "v49acq_toolace_000002": "toolace_callonly",
                               "v49acq_toolace_000003": "toolace_nocall"})
        r1 = next(r for r in rows if r["id"].endswith("000001"))
        self.assertEqual(r1["n_calls"], 2)
        self.assertEqual(len(r1["loss_mask_spans"]), 2)
        self.assertEqual([r1["full_text"][a:b] for a, b in r1["loss_mask_spans"]], ["\n結果: [{\"t\": 20}]\n", "\n結果: [{\"time\": \"10:00\"}]\n"])
        self.assertTrue(r1["target"].startswith("\n呼び出し: [Get Weather(city=\"Paris\")]\n結果: "))
        self.assertTrue(r1["target"].endswith("呼び出し: [Get Time()]\n結果: [{\"time\": \"10:00\"}]\n答え: 20 degrees at 10:00."))
        ro = next(r for r in rows if r["id"].endswith("000002"))
        self.assertEqual((ro["loss_mask_spans"], ro["tool_steps"], ro["answer_type"]), ([], [], "tool_call"))
        self.assertEqual(ro["target"], "\n呼び出し: [Get Weather(city=\"Paris\")]")
        self.assertIn("- Get Weather(city: string, days?: int): Returns the weather for a city.", rows[0]["prompt"])
        self.assertEqual(s["rejects"], {"empty_assistant": 1, "no_final_answer_after_tool": 1, "tool_list_not_json": 1, "invalid_row:masked_row_not_clean": 1})
        self.assertEqual((s["language"]["parsed_zh"], s["language"]["parsed_en"]), (1, 7))            # counted before any filter, incl. the row without a JSON tool list
        self.assertEqual((s["language"]["kept_zh"], s["language"]["kept_en"]), (0, 4))

    def test_parse_helpers(self):
        self.assertEqual([f["name"] for f in C.parse_tool_functions(system_of(FUNCS))], ["Get Weather", "Get Time"])
        self.assertIsNone(C.parse_tool_functions("x"))
        self.assertTrue(C.is_tool_call("[Get Chat Code (Code to Chat)(a=1.0)]"))
        self.assertFalse(C.is_tool_call("Sure, here you go."))
        self.assertFalse(C.is_tool_call("[1] first point"))


# ====================================================================================================== (f) wizard
class TestWizard(RowAsserts):
    def test_fixture_through_parquet(self):
        if pa is None:
            self.skipTest("pyarrow missing")
        recs = fixture("wizard_fixture.json")
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "data" / "train-00000-of-00002.parquet"
            p.parent.mkdir()
            pq.write_table(pa.Table.from_pylist(recs), p)
            spec = C.DATASETS["wizard"]
            st = C.Stats("wizard")
            rows = self.check_rows(list(spec.convert(C.read_messages_parquet(C.find_wizard(d)), st, CK, spec)))
        s = self.check_stats(st)
        self.assertGreaterEqual(len(rows), 1)
        for r in rows:
            self.assertEqual(r["adoption"], "hold")
            self.assertTrue(r["target"].startswith("\n答え: "))
            self.assertIn("avoid", r["adoption_note"])
        self.assertIn("arith_error_rate", s)

    def test_arithmetic_error_rate(self):
        ok = msg("計算してください。", "3 + 4 = 7 です。また 10 - 2 = 8 です。")
        wrong = msg("計算してください。", "3 + 4 = 8 です。")
        plain = msg("首都は?", "日本の首都は東京です。")
        too_long = msg("長い", "5 + 5 = 11。" + "あ" * 1900)
        multiturn = {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "6 × 7 = 42 です。"},
                                  {"role": "user", "content": "more"}, {"role": "assistant", "content": "ok"}]}
        rows, st = run("wizard", [ok, wrong, plain, too_long, multiturn])
        s = self.check_stats(st)
        self.check_rows(rows)
        self.assertEqual({r["id"]: r["family"] for r in rows}, {"v49acq_wizard_0000000": "wizard_math", "v49acq_wizard_0000002": "wizard_text",
                                                               "v49acq_wizard_0000004": "wizard_math"})
        self.assertEqual(s["rejects"], {"arith_statement_wrong": 1, "row_too_long": 1})
        a = s["arith_error_rate"]
        self.assertEqual((a["statements_checked"], a["statements_wrong"], a["rows_with_checked_statements"], a["rows_with_a_wrong_statement"]), (5, 2, 4, 2))
        self.assertAlmostEqual(a["statement_error_rate"], 0.4)
        self.assertAlmostEqual(a["row_error_rate"], 0.5)
        self.assertEqual((a["rows_fitting_cap_checked"], a["rows_fitting_cap_with_wrong"]), (3, 1))   # the too-long row is counted in the all-rows rate only
        self.assertEqual(s["extra"]["multiturn_first_pair_only"], 1)
        self.assertEqual(rows[0]["final_number_guess"], 8.0)

    def test_python_blocks_are_dropped(self):
        with_py = msg("合計を求めてください。", f"次のコードです。\n{FENCE}python\nprint(3 + 4)\n{FENCE}\n出力は7です。")
        other = msg("合計を求めてください。", "合計は次のとおりです。詳しい説明を以下に示します。この例では二つの数を足して、その結果を画面に表示します。実行すると結果が出ます。"
                    + f"\n{FENCE}javascript\nconsole.log(3 + 4)\n{FENCE}\n出力は7です。")
        rows, st = run("wizard", [with_py, other])
        self.assertEqual(dict(st.rejects), {"python_block_not_executed": 1})
        self.assertEqual([r["id"] for r in rows], ["v49acq_wizard_0000001"])


# ====================================================================================================== validation, output, CLI
class TestValidation(RowAsserts):
    def row(self, **kw):
        prompt, target = "問題: x\n思考:", "\n答え: 1。"
        r = C.build_row(id="v49acq_t_1", family="f", template="t", group="g", category="c", prompt=prompt, target=target, ref="1",
                        answer_type="a", difficulty=1, spec=C.DATASETS["jaquad"], split="train")
        r.update(kw)
        return r

    def test_validate_row(self):
        self.assertIsNone(C.validate_row(self.row(), CK))
        self.assertEqual(C.validate_row(self.row(full_text="x"), CK), "full_text_mismatch")
        r = self.row()
        r["prompt"], r["target"] = "p" * 1800, "t" * 200
        r["full_text"] = r["prompt"] + r["target"]
        self.assertEqual(C.validate_row(r, CK), "too_long")
        r = self.row()
        obs = "\n出力\n"
        r["target"] = "\n計画" + obs + "答え: 1。"
        r["full_text"] = r["prompt"] + r["target"]
        a = len(r["prompt"]) + len("\n計画")
        r["loss_mask_spans"] = [[a, a + len(obs)]]
        self.assertEqual(C.validate_row(r, CK), "mask_does_not_cover_observation")
        r["tool_steps"] = [{"index": 0, "observation": obs}]
        self.assertIsNone(C.validate_row(r, CK))
        r["loss_mask_spans"] = [[0, 3]]                                           # span inside the prompt
        self.assertEqual(C.validate_row(r, CK), "span_out_of_order_or_range")
        r2 = self.row(prompt="問題: ＡＢ\n思考:")
        r2["full_text"] = r2["prompt"] + r2["target"]
        self.assertIsNone(C.validate_row(r2, CK))                                 # unmasked rows may be normalised by the runtime
        r2["loss_mask_spans"], r2["tool_steps"] = [[len(r2["prompt"]) + 1, len(r2["prompt"]) + 3]], [{"observation": r2["full_text"][len(r2["prompt"]) + 1:len(r2["prompt"]) + 3]}]
        self.assertEqual(C.validate_row(r2, CK), "masked_row_not_clean")
        r3 = self.row(target="\n答え: " + "\U0001F600" * 3 + "ab" * 10 + "。")
        r3["full_text"] = r3["prompt"] + r3["target"]
        self.assertEqual(C.validate_row(r3, CK), "oov_frac")
        r4 = self.row(quotes=["not in the prompt"])
        self.assertEqual(C.validate_row(r4, CK), "quote_not_in_prompt")
        r5 = self.row(target="\n答え: \ud800。")
        r5["full_text"] = r5["prompt"] + r5["target"]
        self.assertEqual(C.validate_row(r5, CK), "surrogate")

    def test_all_converters_have_unique_family_and_template_names(self):
        fams, tmpls = {}, {}
        for key, recs in (("jaquad", C.read_jaquad([FIX / "jaquad_train_fixture.json"])),):
            rows, _ = run(key, [r for _, r in recs])
            for r in rows:
                fams.setdefault(r["family"], key)
                tmpls.setdefault(r["structural_template"], key)
        for key, name in (("coding", "coding_fixture.json"), ("hotpot", "hotpot_fixture.json"), ("toolace", "toolace_fixture.json"), ("wizard", "wizard_fixture.json")):
            rows, _ = run(key, fixture(name))
            for r in rows:
                self.assertEqual(fams.setdefault(r["family"], key), key, r["family"])
                self.assertEqual(tmpls.setdefault(r["structural_template"], key), key)
        recs = [json.loads(x) for x in (FIX / "tengentoppa_fixture.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        for r in run("tengentoppa", recs)[0]:
            self.assertEqual(fams.setdefault(r["family"], "tengentoppa"), "tengentoppa")
        known = {"human_jsquad", "human_jnli_label", "human_jnli_letter", "human_jcqa"}
        self.assertFalse(known & set(fams))
        self.assertTrue(all(t.startswith("acq_") for t in tmpls))


class TestExternalChecks(RowAsserts):
    """convert.py delegates python execution to checks.code_sandbox_check(profile='executor'): the rules of the model's own executor."""

    def setUp(self):
        self.ext = C._load_external_checks()
        if self.ext is None or not callable(getattr(self.ext, "code_sandbox_check", None)):
            self.skipTest("checks.py not available")

    def test_backend_and_equivalence(self):
        ck = C.Checks(vocab=CK.vocab, use_external=True)
        self.assertEqual(ck.backends["sandbox"], "checks.code_sandbox_check(profile=executor)")
        self.assertEqual(ck.backends["static_check"], "checks.static_check(profile=executor)")
        self.assertTrue(ck.backends["oov"].startswith("builtin per-row"))
        self.assertEqual(self.ext.RUNTIME_NORMALIZE, C.NORMALIZE)
        r = ck.run_python("print(sum(range(1, 11)))")
        self.assertEqual((r["ok"], r["stdout"]), (True, "55\n"))
        r = ck.run_python("print(1 / 0)")
        self.assertFalse(r["ok"])
        self.assertTrue(r["reason"].startswith("exception:ZeroDivisionError"))
        r = ck.run_python("import os\nprint(os.getcwd())")                      # refused by the external static rules as well
        self.assertFalse(r["ok"])
        for text in ("問題: 答え 1+1=2。\n", "abc\U0001F600", "ＡＢＣ"):
            f, n = self.ext.oov_fraction(text.translate(self.ext.RUNTIME_NORMALIZE), CK.vocab)
            self.assertEqual(n, CK.oov_count(text))
            self.assertAlmostEqual(f, CK.oov_stats(text)["frac"])

    def test_python_row_through_external_sandbox(self):
        ck = C.Checks(vocab=CK.vocab, use_external=True)
        ok = msg("1から10までの合計を表示するPythonコードを書いて。", f"{PY_PLAN}\n{FENCE}python\nprint(sum(range(1, 11)))\n{FENCE}\n出力:\n{FENCE}\n55\n{FENCE}")
        cls = msg("class", f"{FENCE}python\nclass A:\n    def __init__(self):\n        self.x = 3\nprint(A().x)\n{FENCE}\n出力:\n{FENCE}\n3\n{FENCE}")
        seeded = msg("random", f"{FENCE}python\nimport random\nrandom.seed(1)\nprint(random.randint(1, 3))\n{FENCE}\n出力:\n{FENCE}\n2\n{FENCE}")
        rows, st = run("coding", [ok, cls, seeded], ck=ck)
        self.check_rows(rows, ck)
        # code the model's own executor refuses (a class, seeded random) is not turned into a row with a success=True observation
        self.assertEqual([r["reference_final_answer"] for r in rows], ["55"])
        self.assertEqual(dict(st.rejects), {"static:class_def:class definitions are disabled": 1, "static:import:random": 1})
        self.assertEqual(rows[0]["sandbox_backend"], "checks.code_sandbox_check(profile=executor)")
        self.assertEqual(rows[0]["verification"], "stored_output_equals_sandbox")

    def test_vocab_fallback_goes_through_checks(self):
        v, src = C.load_vocab(None, drive_root="/nonexistent")                    # found next to src/v44fix in this checkout
        self.assertGreater(len(v), 3000)


class TestEndToEnd(RowAsserts):
    def test_all_six_through_the_cli_in_the_acquire_layout(self):
        if pa is None:
            self.skipTest("pyarrow missing")
        with tempfile.TemporaryDirectory() as raw, tempfile.TemporaryDirectory() as out:
            R = Path(raw)

            def put(repo, rel):
                p = R / repo / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                return p
            jq = fixture("jaquad_train_fixture.json")
            jq["data"].append({"title": "題", "paragraphs": [{"context": JQ_OK["context"], "qas": [{k: JQ_OK[k] for k in ("id", "question", "question_type", "answers")}]}]})
            put("SkelterLabsInc__JaQuAD", "data/train/jaquad_train_0000.json").write_text(json.dumps(jq, ensure_ascii=False), encoding="utf-8")
            put("SkelterLabsInc__JaQuAD", "data/dev/jaquad_dev_0000.json").write_text('{"data": []}', encoding="utf-8")
            pq.write_table(pa.Table.from_pylist(fixture("coding_fixture.json") + [PY_OK, RUST_OK]), put("llm-jp__Synthetic-JP-EN-Coding-Dataset", "data/train-00000-of-00001.parquet"))
            put("DeL-TaiseiOzaki__Tengentoppa-sft-qwen2.5-32b-reasoning-100k", "completed_sets.jsonl").write_bytes((FIX / "tengentoppa_fixture.jsonl").read_bytes())
            hp_rows = fixture("hotpot_fixture.json")
            pq.write_table(pa.Table.from_pylist(hp_rows), put("hotpotqa__hotpot_qa", "distractor/train-00000-of-00002.parquet"))
            pq.write_table(pa.Table.from_pylist(hp_rows[:1]), put("hotpotqa__hotpot_qa", "distractor/validation-00000-of-00001.parquet"))
            put("Team-ACE__ToolACE", "data.json").write_text(json.dumps(fixture("toolace_fixture.json"), ensure_ascii=False), encoding="utf-8")
            pq.write_table(pa.Table.from_pylist(fixture("wizard_fixture.json")), put("llm-jp__wizardlm8x22b-logical-math-coding-sft-ja", "data/train-00000-of-00002.parquet"))
            code = C.main(["--dataset", "all", "--raw-dir", raw, "--out-dir", out, "--sample-dump", "5"])          # the real sandbox, as on Colab
            self.assertEqual(code, 0)
            summ = json.loads((Path(out) / "convert_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(sorted(summ), sorted(C.DATASETS))
            for key in C.DATASETS:
                m = json.loads((Path(out) / key / "manifest.json").read_text(encoding="utf-8"))
                self.assertTrue(m["balanced"], key)
                self.assertGreater(m["n_out"], 0, key)
                self.assertEqual(m["n_in"], m["n_out"] + m["reject_total"])
                self.assertTrue(all("status" not in v for v in m["files"].values()))
                self.assertLessEqual((Path(out) / key / "sample_dump.txt").read_text(encoding="utf-8").count("=" * 100), 5)
                for sp in ("train", "dev", "heldout"):
                    for line in (Path(out) / key / f"{sp}.jsonl").read_text(encoding="utf-8").splitlines():
                        row = json.loads(line)
                        self.assertEqual(C.validate_row(row, CK), None)
                        self.assertEqual(row["split"], sp)
            hot = json.loads((Path(out) / "hotpot" / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(hot["n_in"], len(hp_rows))                      # the validation file was not read
            self.assertEqual([f["name"] for f in hot["raw_files"]], ["distractor/train-00000-of-00002.parquet"])
            jq = json.loads((Path(out) / "jaquad" / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual([f["name"] for f in jq["raw_files"]], ["data/train/jaquad_train_0000.json"])
            cd = json.loads((Path(out) / "coding" / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(cd["backends"]["sandbox"], "checks.code_sandbox_check(profile=executor)")
            self.assertEqual(cd["verification"], {"stored_output_equals_sandbox": 1, "unverified_no_execution": 1})
            for k in ("jaquad", "hotpot", "coding", "wizard"):
                m = json.loads((Path(out) / k / "manifest.json").read_text(encoding="utf-8"))
                self.assertIn("split_unit", m)
                self.assertEqual(m["group_independent_dev_heldout"], k in ("jaquad", "hotpot"))
            # a sandbox that cannot start stops the run instead of converting with something weaker
            self.assertEqual(C.main(["--dataset", "coding", "--raw-dir", str(R / "llm-jp__Synthetic-JP-EN-Coding-Dataset"), "--out-dir", out + "_x",
                                     "--no-external-checks"]), 4)


class TestOutputAndCli(RowAsserts):
    def raw_jaquad(self, d):
        # the five real fixture rows (none is a clean single-sentence row any more) + ten clean hand-made articles, so that every split file has a chance
        jq = fixture("jaquad_train_fixture.json")
        for i in range(10):
            jq["data"].append({"title": f"題{i}", "paragraphs": [{"context": JQ_OK["context"], "qas": [
                {"id": f"tr-ok-{i:02d}-000", "question": JQ_OK["question"], "question_type": "Syntactic variation", "answers": JQ_OK["answers"]}]}]})
        dst = Path(d) / "data" / "train" / "jaquad_train_0000.json"
        dst.parent.mkdir(parents=True)
        dst.write_text(json.dumps(jq, ensure_ascii=False), encoding="utf-8")
        dev = Path(d) / "data" / "dev" / "jaquad_dev_0000.json"
        dev.parent.mkdir(parents=True)
        dev.write_text(json.dumps({"data": [{"title": "DEV", "paragraphs": [{"context": "甲は乙である。", "qas": [
            {"id": "de-001-00-000", "question": "q", "question_type": "x", "answers": [{"text": "乙", "answer_start": 2, "answer_type": "O"}]}]}]}]}), encoding="utf-8")

    def test_run_dataset_writes_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as raw, tempfile.TemporaryDirectory() as out:
            self.raw_jaquad(raw)
            s = C.run_dataset("jaquad", raw, out, CK, sample_dump=3)
            self.assertFalse(s["refused_overwrite"])
            self.assertEqual([x["name"] for x in s["raw_files"]], ["data/train/jaquad_train_0000.json"])       # dev file never read
            d = Path(out) / "jaquad"
            self.assertEqual(sorted(p.name for p in d.iterdir() if not p.name.endswith(".part")), ["dev.jsonl", "heldout.jsonl", "manifest.json", "rejects_sample.jsonl", "sample_dump.txt", "train.jsonl"])
            n = sum(len((d / f"{sp}.jsonl").read_text(encoding="utf-8").splitlines()) for sp in ("train", "dev", "heldout"))
            self.assertEqual(n, s["n_out"])
            for sp in ("train", "dev", "heldout"):
                for line in (d / f"{sp}.jsonl").read_text(encoding="utf-8").splitlines():
                    row = json.loads(line)
                    self.assertEqual(row["split"], sp)
                    self.assertNotEqual(row["id"], "v49acq_jaquad_de-001-00-000")
            self.assertEqual(len((d / "sample_dump.txt").read_text(encoding="utf-8").split("=" * 100)) - 1, min(3, s["n_out"]))
            self.assertTrue(json.loads((d / "manifest.json").read_text(encoding="utf-8"))["balanced"])
            # second identical run: nothing changes, nothing is reported as refused
            s2 = C.run_dataset("jaquad", raw, out, CK, sample_dump=3)
            self.assertFalse(s2["refused_overwrite"])
            self.assertTrue(all(v["status"] == "identical_existing" for v in s2["files"].values()))
            # a pre-existing different file is never touched
            before = (d / "train.jsonl").read_bytes()
            (d / "train.jsonl").write_text("different\n", encoding="utf-8")
            s3 = C.run_dataset("jaquad", raw, out, CK, sample_dump=3)
            self.assertTrue(s3["refused_overwrite"])
            self.assertEqual((d / "train.jsonl").read_text(encoding="utf-8"), "different\n")
            self.assertTrue(any(p.name.startswith("train.jsonl.new-") and p.read_bytes() == before for p in d.iterdir()))

    def test_provenance_row_defaults_and_train_only(self):
        with tempfile.TemporaryDirectory() as raw, tempfile.TemporaryDirectory() as out:
            self.raw_jaquad(raw)
            extra = Path(raw) / "data" / "train" / "jaquad_train_0001.json"
            extra.write_bytes((FIX / "jaquad_train_fixture.json").read_bytes())
            prov = {"files": [{"dest_rel": "SkelterLabsInc__JaQuAD/data/train/jaquad_train_0000.json", "role": "train", "may_train": True},
                              {"dest_rel": "SkelterLabsInc__JaQuAD/data/train/jaquad_train_0001.json", "role": "train", "may_train": False}],
                    "row_defaults": {"source": "SRC-X", "license": "LIC-X"}, "attribution": {"lines": ["Title: T", "License: L"]}, "commit_sha": "abc"}
            (Path(raw) / "provenance.json").write_text(json.dumps(prov), encoding="utf-8")
            s = C.run_dataset("jaquad", raw, out, CK, sample_dump=0)
            self.assertEqual([x["name"] for x in s["raw_files"]], ["data/train/jaquad_train_0000.json"])
            self.assertEqual(s["provenance"], {"used": True, "commit_sha": "abc", "skipped_not_train_files": ["data/train/jaquad_train_0001.json"]})
            row = json.loads(next(l for l in (Path(out) / "jaquad" / "train.jsonl").read_text(encoding="utf-8").splitlines() if l))
            self.assertEqual((row["source"], row["license"], row["attribution"]), ("SRC-X", "LIC-X", "Title: T | License: L"))

    def test_cli(self):
        with tempfile.TemporaryDirectory() as raw, tempfile.TemporaryDirectory() as out:
            self.raw_jaquad(raw)
            quiet = ["--no-sandbox", "--no-external-checks"]
            self.assertEqual(C.main(["--dataset", "jaquad", "--raw-dir", raw, "--out-dir", out, "--limit", "4", "--sample-dump", "0"] + quiet), 0)
            m = json.loads((Path(out) / "jaquad" / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(m["n_in"], 4)
            self.assertEqual(m["limit"], 4)
            self.assertEqual(C.main(["--dataset", "jaquad", "--raw-dir", raw, "--out-dir", out, "--limit", "4", "--sample-dump", "0"] + quiet), 0)   # identical rerun
            self.assertEqual(C.main(["--dataset", "jaquad", "--raw-dir", raw, "--out-dir", out, "--limit", "5", "--sample-dump", "0"] + quiet), 3)   # differs -> refused
            self.assertEqual(C.main(["--dataset", "nope", "--raw-dir", raw, "--out-dir", out] + quiet), 2)
            self.assertEqual(C.main(["--dataset", "wizard", "--raw-dir", raw, "--out-dir", out] + quiet), 2)      # no raw files
        with tempfile.TemporaryDirectory() as raw, tempfile.TemporaryDirectory() as out:
            sub = Path(raw) / "SkelterLabsInc__JaQuAD"
            sub.mkdir()
            self.raw_jaquad(sub)
            self.assertEqual(C.main(["--dataset", "all", "--raw-dir", raw, "--out-dir", out, "--limit", "3", "--sample-dump", "0", "--no-sandbox", "--no-external-checks"]), 2)   # others missing
            self.assertTrue((Path(out) / "jaquad" / "train.jsonl").exists())

    def test_summarize_is_json_and_complete(self):
        rows, st = run("jaquad", [r for _, r in C.read_jaquad([FIX / "jaquad_train_fixture.json"])])
        s = C.summarize(st, CK)
        for k in ("dataset", "n_in", "n_out", "keep_rate", "rejects", "families", "splits", "family_splits", "balanced", "backends", "adoption", "license",
                  "split_unit", "group_independent_dev_heldout", "supervised_chars", "english", "english_rows_total"):
            self.assertIn(k, s)
        self.assertEqual(s["backends"]["oov"], "builtin")
        self.assertEqual((s["group_independent_dev_heldout"], s["split_unit"][:5]), (True, "group"))
        json.dumps(s, ensure_ascii=False)

    def test_row_level_datasets_say_their_dev_heldout_are_not_group_independent(self):
        for key in ("coding", "tengentoppa", "toolace", "wizard"):
            self.assertIn("NOT group-independent", C.SPLIT_UNIT[key], key)
            s = C.summarize(C.Stats(key), CK)
            self.assertFalse(s["group_independent_dev_heldout"], key)
        for key in ("jaquad", "hotpot"):
            self.assertNotIn("NOT", C.SPLIT_UNIT[key])


class TestShards(RowAsserts):
    """A dataset converted in shards (start <= index < stop) and merged equals the same dataset converted in one run."""

    def test_shards_merge_to_the_single_run(self):
        recs = [msg(f"{i}を表示するPythonコードを書いて。", f"{PY_PLAN}\n{FENCE}python\nprint({i} * 3)\n{FENCE}\n出力:\n{FENCE}\n{i * 3}\n{FENCE}") for i in range(1, 9)]
        recs.insert(3, msg("bad", "ただの説明です。"))                                             # one reject, inside a shard
        recs.insert(6, RUST_OK)
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            raw = td / "raw" / "data"
            raw.mkdir(parents=True)
            pq.write_table(pa.Table.from_pylist(recs), raw / "train-00000-of-00001.parquet")
            whole = C.run_dataset("coding", td / "raw", td / "whole", CKX, sample_dump=4)
            cuts = [(0, 3), (3, 7), (7, None)]
            for i, (a, b) in enumerate(cuts):
                s = C.run_dataset("coding", td / "raw", td / f"shard{i}", CKX, limit=b, start=a, sample_dump=0)
                self.assertEqual(s["range"], {"start": a, "stop": b})
                self.assertEqual(s["n_in"], (b if b else len(recs)) - a)
            merged = C.merge_shards("coding", [td / "shard2", td / "shard0", td / "shard1"], td / "merged", sample_dump=4)       # any order
            self.assertFalse(merged["refused_overwrite"])
            for k in ("n_in", "n_out", "rejects", "families", "splits", "family_splits", "extra", "balanced", "supervised_chars", "english"):
                self.assertEqual(merged[k], whole[k], k)
            for name in ("train.jsonl", "dev.jsonl", "heldout.jsonl", "sample_dump.txt"):
                self.assertEqual((td / "merged" / "coding" / name).read_bytes(), (td / "whole" / "coding" / name).read_bytes(), name)
            m = json.loads((td / "merged" / "coding" / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(m["range"]["shards"], [[0, 3], [3, 7], [7, None]])
            self.assertEqual(m["files"], {n: {k: v for k, v in e.items() if k in ("rows", "bytes", "sha256")} for n, e in whole["files"].items()})
            # a gap, an overlap or a missing tail is refused
            with self.assertRaises(ValueError):
                C.merge_shards("coding", [td / "shard0", td / "shard2"], td / "bad1")
            with self.assertRaises(ValueError):
                C.merge_shards("coding", [td / "shard0", td / "shard1"], td / "bad2")
            with self.assertRaises(ValueError):
                C.merge_shards("coding", [td / "shard1", td / "shard2"], td / "bad3")
            # merging again is idempotent, and never overwrites
            again = C.merge_shards("coding", [td / "shard0", td / "shard1", td / "shard2"], td / "merged", sample_dump=4)
            self.assertFalse(again["refused_overwrite"])
            self.assertTrue(all(v["status"] == "identical_existing" for v in again["files"].values()))


class TestRound3Findings(RowAsserts):
    """Review round 3: every case below was kept (or crashed a resume) before the fix."""

    def test_a_stored_output_of_another_block_cannot_validate_a_python_block(self):
        # block A prints 1 and its own stored output says 999; block B's stored output is 1: A was kept as 'stored_output_equals_sandbox'
        two = msg("two blocks", f"A:\n{FENCE}python\nprint(1)\n{FENCE}\n出力:\n{FENCE}\n999\n{FENCE}\nB:\n{FENCE}python\nprint(2)\n{FENCE}\n出力:\n{FENCE}\n1\n{FENCE}")
        a_no_output = msg("two blocks", f"A:\n{FENCE}python\nprint(1)\n{FENCE}\nB:\n{FENCE}python\nprint(2)\n{FENCE}\n出力:\n{FENCE}\n1\n{FENCE}")
        own = msg("two blocks", f"A:\n{FENCE}python\nprint(1)\n{FENCE}\n出力:\n{FENCE}\n1\n{FENCE}\nB:\n{FENCE}python\nprint(2)\n{FENCE}\n出力:\n{FENCE}\n2\n{FENCE}")
        rows, st = run("coding", [two, a_no_output, own], ck=CKX)
        self.assertEqual(dict(st.rejects), {"stored_output_mismatch": 1, "no_stored_output_to_corroborate": 1})
        self.assertEqual([(r["id"], r["verification"]) for r in rows], [("v49acq_coding_0000002", "stored_output_equals_sandbox")])
        # the helper itself: only the blocks between the python block and the next python block count
        blocks = C.parse_fences(own["messages"][1]["content"])
        py = [b for b in blocks if b["lang"] == "python"]
        self.assertEqual(C._stored_outputs(own["messages"][1]["content"], blocks, py[0]), ["1\n"[:-1]])

    def test_python_fences_with_a_stated_output_are_dropped_wherever_they_sit(self):
        steps = "1. **まず** 条件を整理します。\n\n2. **次に** 理由を述べます。"
        base = {"instruction": "合計を求めるコードを書いてください。", "reasoning": steps, "initial_answer": "x"}
        good = dict(base, refined_answer="合計は次のとおりです。")
        inline = dict(base, refined_answer=f"次のコードを実行します。{FENCE}python print(1+1) {FENCE} 結果は3になります。")
        untagged = dict(base, refined_answer=f"次のコードを実行します。\n{FENCE}\nprint(1 + 1)\n{FENCE}\n結果は3になります。")
        untagged_inline = dict(base, refined_answer=f"次のコードを実行します。{FENCE} print(1 + 1) {FENCE} 結果は3になります。")
        tilde = dict(base, refined_answer="次のコードです。\n~~~py\nprint(1)\n~~~\n結果は1です。")
        json_block = dict(base, refined_answer=f"設定は次のとおりです。\n{FENCE}\n{{\"a\": 1}}\n{FENCE}\n以上です。")
        text_block = dict(base, refined_answer=f"例は次のとおりです。\n{FENCE}\nこれは説明の文章です\n{FENCE}\n以上です。")
        shell_block = dict(base, refined_answer=f"次のコマンドです。\n{FENCE}\n$ ls -l\n{FENCE}\n以上です。")
        pyramid = dict(base, refined_answer=f"ピラミッドの例について、日本語で詳しく説明します。\n{FENCE}pyramid\nabc\n{FENCE}\n以上です。")
        rows, st = run("tengentoppa", [good, inline, untagged, untagged_inline, tilde, json_block, text_block, shell_block, pyramid])
        self.assertEqual(dict(st.rejects), {"python_block_not_executed": 4})
        self.assertEqual(sorted(int(r["id"][-7:]) for r in rows), [0, 5, 6, 7, 8])
        # the same rule for wizardlm rows (and the helper itself)
        w_inline = msg("計算してください。", f"次のコードを実行します。{FENCE}python print(1+1) {FENCE} 結果は3になります。")
        w_untagged = msg("計算してください。", f"次のコードを実行します。\n{FENCE}\nprint(1 + 1)\n{FENCE}\n結果は3になります。")
        w_ok = msg("計算してください。", "二つの数を足すと和になります。詳しい説明は省きます。")
        rows, st = run("wizard", [w_inline, w_untagged, w_ok])
        self.assertEqual((dict(st.rejects), [r["id"] for r in rows]), ({"python_block_not_executed": 2}, ["v49acq_wizard_0000002"]))
        for text, want in ((f"{FENCE}python x=1 {FENCE}", True), (f"{FENCE}\nprint(2)\n{FENCE}", True), (f"{FENCE}\nx = 1\n{FENCE}", False),
                           (f"{FENCE}json\n{{\"a\": 1}}\n{FENCE}", False), ("no fence at all print(1)", False)):
            self.assertEqual(C.has_python_block(text), want, text)

    def test_a_character_outside_the_vocabulary_in_the_target_is_rejected_in_the_prompt_it_is_tolerated(self):
        # the loader silently DELETES every out-of-vocabulary character: 'const msg = ‘hello’;' is trained as 'const msg = hello’;'
        self.assertGreater(CK.oov_count("\u2018"), 0)
        js = msg("JavaScriptで挨拶を表示してください。", f"次のとおりです。\n{FENCE}javascript\nconst msg = \u2018hello\u2019;\nconsole.log(msg);\n{FENCE}\nこれで表示されます。")
        ok = msg("JavaScriptで挨拶を表示してください。", f"次のとおりです。\n{FENCE}javascript\nconst msg = 'hello';\nconsole.log(msg);\n{FENCE}\nこれで表示されます。")
        rows, st = run("coding", [js, ok])
        self.assertEqual(dict(st.rejects), {"invalid_row:oov_in_target": 1})
        self.assertEqual([r["id"] for r in rows], ["v49acq_coding_0000001"])
        steps = "1. **まず** 条件を整理します。\n\n2. **次に** 理由を述べます。"
        long_q = "日本の首都について、歴史的な背景も含めて詳しく説明してください。" * 4 + "\U0001F600"          # one emoji in a long prompt: under 1 %, tolerated in the context
        good = {"instruction": long_q, "reasoning": steps, "initial_answer": "x", "refined_answer": "日本の首都は東京です。"}
        rows, st = run("tengentoppa", [good])
        self.assertEqual((len(rows), dict(st.rejects)), (1, {}))
        self.assertEqual(rows[0]["n_oov"], 1)
        self.assertEqual(C.validate_row(dict(rows[0], target=rows[0]["target"] + "\U0001F600", full_text=rows[0]["full_text"] + "\U0001F600"), CK), "oov_in_target")

    def test_a_hotpot_answer_must_be_a_whole_word_of_the_evidence(self):
        part_of_a_word = hp(rid="p", ans="Spa")                                                 # 'Spa' is inside 'Spain'
        word = hp(rid="q", ans="Spain")
        phrase = hp(rid="r", ans="old town")
        rows, st = run("hotpot", [part_of_a_word, word, phrase])
        self.assertEqual(dict(st.rejects), {"answer_not_in_evidence": 1})
        self.assertEqual([r["id"] for r in rows], ["v49acq_hotpot_q", "v49acq_hotpot_r"])

    def test_the_manifest_is_written_last_and_atomically(self):
        # a kill between the manifest and the other files left a folder that the pipeline took for finished (manifest.json = done)
        order = []
        real = C._write_new

        def spy(path, text):
            order.append(Path(path).name)
            return real(path, text)
        from unittest import mock
        with tempfile.TemporaryDirectory() as raw, tempfile.TemporaryDirectory() as out:
            TestOutputAndCli.raw_jaquad(self, raw)
            with mock.patch.object(C, "_write_new", spy):
                C.run_dataset("jaquad", raw, out, CK, sample_dump=3)
            self.assertEqual(order[-1], "manifest.json")
            self.assertEqual(sorted(order), ["manifest.json", "rejects_sample.jsonl", "sample_dump.txt"])
            # the text never appears under its final name before it is complete: the rename is the only step that creates it
            target = Path(out) / "atomic.txt"
            with mock.patch("os.replace", side_effect=OSError("killed before the rename")):
                with self.assertRaises(OSError):
                    C._write_new(target, "x" * 1000)
            self.assertFalse(target.exists())
            self.assertEqual(C._write_new(target, "x" * 1000), "written")
            self.assertEqual(target.read_text(encoding="utf-8"), "x" * 1000)
            self.assertEqual(C._write_new(target, "x" * 1000), "identical_existing")


class TestSandboxFailsLoudly(RowAsserts):
    """Review round 4 (major): a sandbox failure in the MIDDLE of a long run (no space in /tmp for a child, the unprivileged user vanished, a child that
    dies without a result) was turned into a per-row reject 'sandbox_failed:crash:...' / 'sandbox_failed:checks_error:...', so the manifest stayed
    balanced and every verified python row silently vanished into the reject counts.  It must stop the converter; the CODE under test failing,
    timing out or being refused stays a per-row reject."""

    GOOD = msg("1から10までの合計を表示するPythonコードを書いて。", f"{PY_PLAN}\n{FENCE}python\nprint(sum(range(1, 11)))\n{FENCE}\n出力:\n{FENCE}\n55\n{FENCE}\n結果は55です。")

    def setUp(self):
        self.ext = C._load_external_checks()
        if self.ext is None:
            self.skipTest("checks.py not available")

    def patched(self, fn):
        from unittest import mock
        return mock.patch.object(self.ext, "code_sandbox_check", fn)

    def test_the_infrastructure_categories(self):
        for reason in ("crash:spawn:OSError: [Errno 28] No space left on device", "crash:returncode=-11:", "crash:returncode=1:Traceback", "crash"):
            self.assertTrue(C.sandbox_reason_is_infrastructure(reason), reason)
        for reason in ("timeout:5.0s", "output_limit:captured output exceeded 8192 characters", "exception:ZeroDivisionError: division by zero",
                       "exception:MemoryError: ", "static:import:os", "static:match_class:keyword attributes in class patterns are disabled: format", ""):
            self.assertFalse(C.sandbox_reason_is_infrastructure(reason), reason)

    def test_run_python_raises_for_a_broken_sandbox_and_keeps_code_failures_per_row(self):
        ck = C.Checks(vocab=CK.vocab)
        for reason in ("crash:spawn:OSError: [Errno 28] No space left on device", "crash:returncode=-9:"):
            with self.patched(lambda code, timeout, **kw: (False, "", reason)):
                with self.assertRaises(C.SandboxUnavailable) as cm:
                    ck.run_python("print(1)")
            self.assertIn("in the middle of the run", str(cm.exception))
        for exc in (self.ext.SandboxError("user 'nobody' does not exist on this machine"), OSError(28, "No space left on device"), RuntimeError("boom")):
            def raiser(code, timeout, _exc=exc, **kw):
                raise _exc
            with self.patched(raiser):
                with self.assertRaises(C.SandboxUnavailable) as cm:
                    ck.run_python("print(1)")
            self.assertIn(type(exc).__name__, str(cm.exception))
        for reason in ("timeout:5.0s", "output_limit:captured output exceeded 8192 characters", "exception:ZeroDivisionError: division by zero",
                       "exception:MemoryError: ", "static:import:os"):
            with self.patched(lambda code, timeout, **kw: (False, "", reason)):
                self.assertEqual(ck.run_python("print(1)"), {"ok": False, "stdout": "", "reason": reason})
        with self.patched(lambda code, timeout, **kw: (True, "55\n", "")):
            self.assertEqual(ck.run_python("print(1)"), {"ok": True, "stdout": "55\n", "reason": ""})

    def test_a_failure_in_the_middle_of_a_run_stops_the_conversion_and_writes_no_manifest(self):
        ck = C.Checks(vocab=CK.vocab)
        real = self.ext.code_sandbox_check
        calls = []

        def flaky(code, timeout, **kw):
            calls.append(code)
            if len(calls) >= 2:
                return False, "", "crash:spawn:OSError: [Errno 28] No space left on device"
            return real(code, timeout, **kw)
        with self.patched(flaky):
            spec = C.DATASETS["coding"]
            st = C.Stats("coding")
            with self.assertRaises(C.SandboxUnavailable):
                list(spec.convert(list(enumerate([self.GOOD, self.GOOD, self.GOOD])), st, ck, spec))
        self.assertEqual(len(calls), 2)                                                      # nothing ran after the failure
        self.assertNotIn("sandbox_failed:crash:spawn:OSError: [Errno 28] No space left on device", st.rejects)

    def test_the_cli_exits_with_4_and_leaves_no_manifest(self):
        if pa is None:
            self.skipTest("pyarrow missing")
        import io
        from contextlib import redirect_stderr
        flaky_calls = []
        real = self.ext.code_sandbox_check

        def flaky(code, timeout, **kw):
            flaky_calls.append(1)
            if len(flaky_calls) >= 2:
                raise self.ext.SandboxError("user 'nobody' does not exist on this machine")
            return real(code, timeout, **kw)
        with tempfile.TemporaryDirectory() as raw, tempfile.TemporaryDirectory() as out:
            p = Path(raw) / "data" / "train-00000-of-00001.parquet"
            p.parent.mkdir()
            pq.write_table(pa.Table.from_pylist([self.GOOD, self.GOOD, self.GOOD]), p)
            err = io.StringIO()
            with self.patched(flaky), redirect_stderr(err):
                code = C.main(["--dataset", "coding", "--raw-dir", raw, "--out-dir", out, "--sample-dump", "0"])
            self.assertEqual(code, 4)
            self.assertIn("SANDBOX FAILED DURING THE RUN", err.getvalue())
            self.assertIn("SandboxError", err.getvalue())
            self.assertFalse((Path(out) / "coding" / "manifest.json").exists())              # 'finished' is marked by the manifest, written last
            self.assertFalse((Path(out) / "coding" / "train.jsonl").exists())

    def test_ordinary_code_failures_are_still_per_row_rejects(self):
        ck = C.Checks(vocab=CK.vocab)
        zero = msg("divide", f"{FENCE}python\nprint(1 / 0)\n{FENCE}")
        rows, st = run("coding", [self.GOOD, zero], ck=ck)
        self.assertEqual(len(rows), 1)
        self.assertEqual(dict(st.rejects), {"sandbox_failed:exception:ZeroDivisionError: division by zero": 1})
        slow = C.Checks(vocab=CK.vocab, sandbox_timeout=1.0)
        loop = msg("loop", f"{FENCE}python\nwhile True:\n    pass\nprint(1)\n{FENCE}")
        rows, st = run("coding", [loop], ck=slow)
        self.assertEqual((len(rows), dict(st.rejects)), (0, {"sandbox_failed:timeout:1.0s": 1}))

    def test_termination_signals_are_forwarded_to_the_sandbox_children(self):
        # POSIX-only behaviour, which cannot run on this Windows PC: the handler is exercised with a fake `signal.signal` and a fake extension
        from unittest import mock
        killed = []

        class Ext:
            @staticmethod
            def kill_live_sandbox_children():
                killed.append(1)
                return 1
        handlers = {}
        previous = {C.signal.SIGTERM: "old-term", C.signal.SIGINT: "old-int"}

        def fake_signal(sig, handler):
            old = handlers.get(sig, previous[sig])
            handlers[sig] = handler
            return old
        with mock.patch.object(C.signal, "signal", fake_signal):
            with C.forward_termination_to_sandbox(Ext, posix=True) as installed:
                self.assertTrue(installed)
                self.assertEqual(set(handlers), {C.signal.SIGTERM, C.signal.SIGINT})
                with self.assertRaises(SystemExit) as cm:
                    handlers[C.signal.SIGTERM](C.signal.SIGTERM, None)                    # what `kill -TERM <converter>` (pipeline._terminate) does
                self.assertEqual(cm.exception.code, 128 + int(C.signal.SIGTERM))
                self.assertEqual(killed, [1])                                              # the sandbox child was killed before the converter exits
            self.assertEqual(handlers[C.signal.SIGTERM], "old-term")                       # restored afterwards
            self.assertEqual(handlers[C.signal.SIGINT], "old-int")
        n = len(killed)
        with C.forward_termination_to_sandbox(Ext, posix=False) as installed:             # not POSIX: nothing is installed
            self.assertFalse(installed)
        with C.forward_termination_to_sandbox(None, posix=True) as installed:             # no sandbox module: nothing to forward to
            self.assertFalse(installed)
        self.assertEqual(len(killed), n)


def py_row(label, code, stored):
    return msg(f"{label}を表示するPythonコードを書いて。", f"{PY_PLAN}\n{FENCE}python\n{code}\n{FENCE}\n出力:\n{FENCE}\n{stored}\n{FENCE}\n結果は {stored} です。")


class TestCodeCrashCanary(RowAsserts):
    """Round 4 (F1): a child that ends without a result line ('crash:returncode=N') used to be treated as a broken sandbox for EVERY row, so one
    statically accepted row whose code overflows the C stack (a 300,000-deep map() chain) stopped the converter for good (exit 4 on every re-run).
    Now a canary (print(1) in a fresh child, same profile / account / limits / timeout) tells the two apart: canary ok -> that ROW killed the child,
    one counted reject 'sandbox_failed:code_crash:returncode=N'; canary fails too (or 'crash:spawn', or an exception out of the checks module) -> stop.
    Too many such rows in one run -> stop as well."""

    CRASH = "crash:returncode=-11:Segmentation fault"
    MARK = "print('CRASHME')"
    CANARY = "print(1)"                                                                          # the trivial known-good snippet (the task says: print(1))

    def setUp(self):
        self.ext = C._load_external_checks()
        if self.ext is None:
            self.skipTest("checks.py not available")
        self.real = self.ext.code_sandbox_check

    def fake(self, calls, row_result=None, canary=None, default=None):
        """code_sandbox_check that logs every call; the marked row returns row_result, the canary snippet returns canary(), every other snippet returns
        `default` (or runs for real when there is none)."""
        def f(code, timeout, **kw):
            calls.append((code, timeout, dict(kw)))
            if code.strip() == self.CANARY:
                if canary is not None:
                    return canary()
            elif self.MARK in code and row_result is not None:
                return row_result() if callable(row_result) else row_result
            if default is not None:
                return default
            return self.real(code, timeout, **kw)
        return f

    def patched(self, fn):
        from unittest import mock
        return mock.patch.object(self.ext, "code_sandbox_check", fn)

    def test_a_row_that_kills_the_child_is_one_counted_reject_when_the_canary_passes(self):
        ck = C.Checks(vocab=CK.vocab)
        calls = []
        recs = [py_row("a", "print('one')", "one"), py_row("b", self.MARK, "CRASHME"), py_row("c", "print('three')", "three")]
        with self.patched(self.fake(calls, row_result=(False, "", self.CRASH))):
            rows, st = run("coding", recs, ck=ck)
        self.assertEqual(len(rows), 2)                                                        # the run went on after the crash
        self.assertEqual(dict(st.rejects), {"sandbox_failed:code_crash:returncode=-11": 1})   # one row, greppable, no stderr tail in the key
        self.assertEqual((ck.code_runs, ck.code_crashes), (3, 1))
        s = C.summarize(st, ck)
        self.assertTrue(s["balanced"])
        self.assertEqual(s["python"]["rows_rejected_code_crash"], 1)                          # counted in the manifest next to the other python numbers
        # the canary ran right after the crashed row, in a fresh child with the same timeout and the same profile / run_as / limits (same kwargs) as the rows
        self.assertEqual(len(calls), 4)
        self.assertEqual([c[0].strip() == self.CANARY for c in calls], [False, False, True, False])
        self.assertEqual(calls[1][1:], calls[2][1:])
        self.assertEqual(calls[2][2], {"profile": "executor"})
        self.assertEqual([len(v) for v in st.reject_examples.values()], [1])                  # the crashed row is kept as the example of its reject reason
        self.assertEqual(list(st.reject_examples), ["sandbox_failed:code_crash:returncode=-11"])

    def test_the_canary_snippet_really_runs_in_the_real_sandbox(self):
        ck = C.Checks(vocab=CK.vocab)
        self.assertEqual(C.CANARY_CODE.strip(), self.CANARY)
        self.assertIsNone(ck._canary_failure())

    def test_a_canary_that_fails_too_stops_the_run(self):
        recs = [py_row("a", "print('one')", "one"), py_row("b", self.MARK, "CRASHME"), py_row("c", "print('three')", "three")]
        for label, canary in (("crashes", lambda: (False, "", "crash:returncode=-11:")), ("times out", lambda: (False, "", "timeout:5.0s")),
                              ("prints something else", lambda: (True, "2\n", "")), ("raises", lambda: (_ for _ in ()).throw(OSError(28, "No space left on device")))):
            with self.subTest(label):
                ck = C.Checks(vocab=CK.vocab)
                calls = []
                with self.patched(self.fake(calls, row_result=(False, "", self.CRASH), canary=canary)):
                    with self.assertRaises(C.SandboxUnavailable) as cm:
                        run("coding", recs, ck=ck)
                self.assertIn("canary", str(cm.exception))
                self.assertIn("in the middle of the run", str(cm.exception))
                self.assertEqual(len(calls), 3)                                               # row 1, the crashed row, the canary: nothing ran afterwards
                self.assertEqual(ck.code_crashes, 0)                                          # it was never counted as the code's doing

    def test_spawn_failures_and_exceptions_stop_without_a_canary(self):
        for label, result in (("crash:spawn", (False, "", "crash:spawn:OSError: [Errno 28] No space left on device")), ("bare crash", (False, "", "crash")),
                              ("raises", lambda: (_ for _ in ()).throw(self.ext.SandboxError("user 'nobody' does not exist on this machine")))):
            with self.subTest(label):
                ck = C.Checks(vocab=CK.vocab)
                calls = []
                with self.patched(self.fake(calls, row_result=result)):
                    with self.assertRaises(C.SandboxUnavailable):
                        run("coding", [py_row("b", self.MARK, "CRASHME")], ck=ck)
                self.assertEqual(len(calls), 1)                                               # no canary: these are never a verdict about the code
                self.assertFalse(any(c[0].strip() == self.CANARY for c in calls))

    def test_ordinary_code_outcomes_never_start_a_canary(self):
        ck = C.Checks(vocab=CK.vocab)
        for reason in ("timeout:5.0s", "exception:ZeroDivisionError: division by zero", "exception:MemoryError: ", "output_limit:captured output exceeded 8192 characters",
                       "static:import:os"):
            calls = []
            with self.patched(self.fake(calls, row_result=(False, "", reason))):
                self.assertEqual(ck.run_python(self.MARK), {"ok": False, "stdout": "", "reason": reason})
            self.assertEqual(len(calls), 1, reason)
        self.assertEqual(ck.code_crashes, 0)
        zero = py_row("divide", "print(1 / 0)", "x")
        rows, st = run("coding", [zero], ck=ck)
        self.assertEqual(dict(st.rejects), {"sandbox_failed:exception:ZeroDivisionError: division by zero": 1})

    def test_more_than_50_code_crash_rows_in_one_run_stop_the_run(self):
        ck = C.Checks(vocab=CK.vocab)
        calls = []
        with self.patched(self.fake(calls, row_result=(False, "", self.CRASH), canary=lambda: (True, "1\n", ""), default=(True, "1\n", ""))):
            for i in range(1, 51):                                                            # 50 crashes, one in every 60 rows: 1.7 % each time, so only the count matters
                for _ in range(59):
                    self.assertTrue(ck.run_python("print(1)")["ok"])
                self.assertEqual(ck.run_python(self.MARK)["reason"], "code_crash:returncode=-11")
            self.assertEqual(ck.code_crashes, 50)                                             # 50 is still allowed ...
            for _ in range(59):
                ck.run_python("print(1)")
            with self.assertRaises(C.SandboxUnavailable) as cm:                              # ... the 51st stops
                ck.run_python(self.MARK)
        msg_ = str(cm.exception)
        self.assertIn("51 of the 3060 rows", msg_)                                            # the message names the counts
        self.assertIn("more than 50 rows", msg_)

    def test_a_share_above_2_percent_stops_the_run_but_only_once_50_rows_have_executed(self):
        def fresh():
            return C.Checks(vocab=CK.vocab)
        with self.patched(self.fake([], row_result=(False, "", self.CRASH), canary=lambda: (True, "1\n", ""), default=(True, "1\n", ""))):
            ck = fresh()                                                                      # 3 crashes in the first 10 rows: a share of 10 rows means nothing yet
            for _ in range(7):
                ck.run_python("print(1)")
            for _ in range(3):
                ck.run_python(self.MARK)
            self.assertEqual((ck.code_runs, ck.code_crashes), (10, 3))
            ck = fresh()                                                                      # 1 crash in 50 rows is exactly 2 %: allowed ("exceeds")
            for _ in range(49):
                ck.run_python("print(1)")
            ck.run_python(self.MARK)
            self.assertEqual((ck.code_runs, ck.code_crashes), (50, 1))
            for _ in range(9):
                ck.run_python("print(1)")
            with self.assertRaises(C.SandboxUnavailable) as cm:                              # a 2nd crash in row 60: 3.3 % of 60 rows
                ck.run_python(self.MARK)
            self.assertIn("2 of the 60 rows", str(cm.exception))
            self.assertIn("2%", str(cm.exception))
            ck = fresh()
            for _ in range(60):
                ck.run_python("print(1)")
            for _ in range(2):                                                                # 2 crashes in 62 rows = 3.2 %: stop at the 2nd
                try:
                    ck.run_python(self.MARK)
                except C.SandboxUnavailable:
                    break
            else:
                self.fail("a share above 2 % did not stop the run")
            ck = fresh()                                                                      # 1 crash in 200 rows (0.5 %) is fine
            for _ in range(199):
                ck.run_python("print(1)")
            self.assertEqual(ck.run_python(self.MARK)["reason"], "code_crash:returncode=-11")

    def test_the_real_nested_map_chain_is_a_per_row_reject_not_a_stop(self):
        # the confirmer's repro, through the real sandbox: statically accepted, kills the child with a stack overflow (3221225725 on Windows, -11 on Linux)
        code = "it = iter([1])\nfor _ in range(300000):\n    it = map(abs, it)\nprint(next(it))"
        ck = C.Checks(vocab=CK.vocab)
        good = py_row("sum", "print(sum(range(1, 11)))", "55")
        rows, st = run("coding", [py_row("deep", code, "1"), good], ck=ck)
        keys = list(st.rejects)
        if not (keys and keys[0].startswith("sandbox_failed:code_crash:")):
            self.skipTest(f"this Python does not die on the deep chain here (reject: {keys})")
        self.assertRegex(keys[0], r"^sandbox_failed:code_crash:returncode=-?\d+$")
        self.assertEqual(len(rows), 1)                                                        # the next row was converted: the run did not stop
        self.assertEqual(rows[0]["reference_final_answer"], "55")
        self.assertEqual((ck.code_runs, ck.code_crashes), (2, 1))

    def test_the_cli_finishes_with_the_code_crash_counted_in_the_manifest(self):
        if pa is None:
            self.skipTest("pyarrow missing")
        import io
        from contextlib import redirect_stderr
        recs = [py_row("a", "print('one')", "one"), py_row("b", self.MARK, "CRASHME"), py_row("c", "print('three')", "three")]
        with tempfile.TemporaryDirectory() as raw, tempfile.TemporaryDirectory() as out:
            p = Path(raw) / "data" / "train-00000-of-00001.parquet"
            p.parent.mkdir()
            pq.write_table(pa.Table.from_pylist(recs), p)
            with self.patched(self.fake([], row_result=(False, "", self.CRASH))), redirect_stderr(io.StringIO()):
                code = C.main(["--dataset", "coding", "--raw-dir", raw, "--out-dir", out, "--sample-dump", "0"])
            self.assertEqual(code, 0)
            man = json.loads((Path(out) / "coding" / "manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(man["balanced"])
            self.assertEqual(man["rejects"], {"sandbox_failed:code_crash:returncode=-11": 1})
            self.assertEqual(man["python"]["rows_rejected_code_crash"], 1)
            self.assertEqual(man["n_out"], 2)
            self.assertEqual(list((Path(out) / "coding").glob("*.part-*")), [])


class TestNoPartFilesAfterAFailedRun(RowAsserts):
    """Round 4 (F2): after the fatal exit (rc 4) the converter left train / dev / heldout .jsonl.part-<pid> in the local output folder, one more set per
    retry, and pipeline._run_window's heartbeat sums every *.part-* file.  A failed run removes its OWN .part-<pid> files and nothing else."""

    OTHER = ("train.jsonl.part-99999999", "dev.jsonl.part-99999999", "heldout.jsonl.part-99999999")

    def setUp(self):
        self.ext = C._load_external_checks()
        if self.ext is None or pa is None:
            self.skipTest("checks.py or pyarrow not available")

    def parts(self, out):
        return sorted(p.name for p in (Path(out) / "coding").glob("*.part-*"))

    def raw_with(self, raw, recs):
        p = Path(raw) / "data" / "train-00000-of-00001.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(recs), p)

    def test_the_cli_removes_its_own_part_files_when_the_sandbox_stops_the_run(self):
        import io
        from contextlib import redirect_stderr
        from unittest import mock
        real = self.ext.code_sandbox_check
        n = []

        def flaky(code, timeout, **kw):
            n.append(1)
            if len(n) >= 2:
                raise self.ext.SandboxError("user 'nobody' does not exist on this machine")
            return real(code, timeout, **kw)
        recs = [py_row("a", "print('one')", "one"), py_row("b", "print('two')", "two"), py_row("c", "print('three')", "three")]
        with tempfile.TemporaryDirectory() as raw, tempfile.TemporaryDirectory() as out:
            self.raw_with(raw, recs)
            (Path(out) / "coding").mkdir()
            for name in self.OTHER:                                                           # leftovers of ANOTHER process (or of an older run): never touched
                (Path(out) / "coding" / name).write_text("x", encoding="utf-8")
            (Path(out) / "coding" / "train.jsonl").write_text("finished rows\n", encoding="utf-8")       # a finished file of an earlier run: never touched
            for attempt in range(2):                                                          # the retry must not add a second set
                err = io.StringIO()
                with mock.patch.object(self.ext, "code_sandbox_check", flaky), redirect_stderr(err):
                    n.clear()
                    code = C.main(["--dataset", "coding", "--raw-dir", raw, "--out-dir", out, "--sample-dump", "0"])
                self.assertEqual(code, 4)
                self.assertIn("SANDBOX FAILED DURING THE RUN", err.getvalue())
                self.assertEqual(self.parts(out), sorted(self.OTHER), attempt)               # only the foreign leftovers remain: this run's own files are gone
                self.assertEqual((Path(out) / "coding" / "train.jsonl").read_text(encoding="utf-8"), "finished rows\n")
                self.assertFalse((Path(out) / "coding" / "manifest.json").exists())
            for name in self.OTHER:
                self.assertEqual((Path(out) / "coding" / name).read_text(encoding="utf-8"), "x")

    def test_the_canary_failure_exit_also_leaves_nothing_behind(self):
        import io
        from contextlib import redirect_stderr
        from unittest import mock
        real = self.ext.code_sandbox_check

        def broken(code, timeout, **kw):
            if "CRASHME" in code or code.strip() == "print(1)":
                return False, "", "crash:returncode=-9:"
            return real(code, timeout, **kw)
        recs = [py_row("a", "print('one')", "one"), py_row("b", "print('CRASHME')", "CRASHME")]
        with tempfile.TemporaryDirectory() as raw, tempfile.TemporaryDirectory() as out:
            self.raw_with(raw, recs)
            err = io.StringIO()
            with mock.patch.object(self.ext, "code_sandbox_check", broken), redirect_stderr(err):
                code = C.main(["--dataset", "coding", "--raw-dir", raw, "--out-dir", out, "--sample-dump", "0"])
            self.assertEqual(code, 4)
            self.assertIn("canary", err.getvalue())
            self.assertEqual(self.parts(out), [])
            self.assertEqual(sorted(p.name for p in (Path(out) / "coding").iterdir()), [])

    def test_any_other_failure_of_run_dataset_removes_the_part_files_too(self):
        from unittest import mock
        recs = [py_row("a", "print('one')", "one"), py_row("b", "print('two')", "two")]
        with tempfile.TemporaryDirectory() as raw, tempfile.TemporaryDirectory() as out:
            self.raw_with(raw, recs)
            seen = []
            real_validate = C.validate_row

            def boom(row, ck):
                if seen:
                    raise KeyboardInterrupt()                                                  # a BaseException: Interrupt execution / SIGINT
                seen.append(1)
                return real_validate(row, ck)
            with mock.patch.object(C, "validate_row", boom):
                with self.assertRaises(KeyboardInterrupt):
                    C.run_dataset("coding", raw, out, CKX, sample_dump=0)
            self.assertEqual(seen, [1])                                                       # one row had been written to the .part file before it failed
            self.assertEqual(self.parts(out), [])
            with mock.patch.object(C.SplitWriter, "close", side_effect=OSError("disk full")):          # the failure comes while the files are being finished
                with self.assertRaises(OSError):
                    C.run_dataset("coding", raw, out, CKX, sample_dump=0)
            self.assertEqual(self.parts(out), [])

    def test_a_finished_run_leaves_no_part_files_and_a_failed_write_removes_its_own(self):
        from unittest import mock
        recs = [py_row("a", "print('one')", "one")]
        with tempfile.TemporaryDirectory() as raw, tempfile.TemporaryDirectory() as out:
            self.raw_with(raw, recs)
            s = C.run_dataset("coding", raw, out, CKX, sample_dump=2)
            self.assertEqual(s["n_out"], 1)
            self.assertEqual(self.parts(out), [])
            target = Path(out) / "x.txt"
            with mock.patch("os.replace", side_effect=OSError("killed before the rename")):
                with self.assertRaises(OSError):
                    C._write_new(target, "x" * 100)
            self.assertEqual(sorted(p.name for p in Path(out).glob("x.txt*")), [])            # the .part-<pid> of the failed write is gone

    def test_splitwriter_abort_removes_only_its_own_files(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "k"
            d.mkdir()
            (d / "train.jsonl").write_text("done\n", encoding="utf-8")
            (d / "train.jsonl.part-1").write_text("someone else\n", encoding="utf-8")
            w = C.SplitWriter(d)
            w.write_line("train", '{"a": 1}')
            self.assertEqual(len(list(d.glob("*.part-*"))), 4)
            w.abort()
            self.assertEqual(sorted(p.name for p in d.iterdir()), ["train.jsonl", "train.jsonl.part-1"])
            w.abort()                                                                         # idempotent

    def test_a_merge_that_fails_in_the_middle_removes_its_part_files(self):
        recs = [py_row(str(i), f"print({i} * 3)", str(i * 3)) for i in range(1, 5)]
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            self.raw_with(td / "raw", recs)
            for i, (a, b) in enumerate([(0, 2), (2, None)]):
                C.run_dataset("coding", td / "raw", td / f"shard{i}", CKX, limit=b, start=a, sample_dump=0)
            (td / "shard1" / "coding" / "heldout.jsonl").unlink()                              # the 2nd shard lost a file: the merge fails after it started writing
            with self.assertRaises(FileNotFoundError):
                C.merge_shards("coding", [td / "shard0", td / "shard1"], td / "merged", sample_dump=0)
            self.assertEqual(list((td / "merged" / "coding").glob("*.part-*")), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
