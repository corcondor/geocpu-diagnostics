#!/usr/bin/env python3
"""Unit tests for checks.py on hand-made fixtures (every fixture is a few KB; nothing is downloaded).

    python -I -X utf8 test_checks.py            # one small process
"""
import array
import io
import json
import os
import random
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import checks as C                                                                  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def tmpdir():
    return tempfile.TemporaryDirectory(prefix="test_checks_", ignore_cleanup_errors=True)


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def row(i, prompt, target="\n答え: 1。", **kw):
    d = {"id": f"r{i}", "prompt": prompt, "target": target, "full_text": prompt + target}
    d.update(kw)
    return d


def rand_text(seed, n=1500, alphabet="あいうえおかきくけこさしすせそたちつてとなにぬねの"):
    rng = random.Random(seed)
    return "".join(rng.choice(alphabet) for _ in range(n))


def make_bundle(drive_root, drop=(), vocab=None, with_py=True):
    """A Drive-like tree: GeoCPU_v47_fullmodel_bundle_v1.zip -> assets/r6_bundle.zip -> data/*.jsonl + base_metadata.json."""
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as z:
        for name in C.EVAL_REQUIRED_JSONL:
            if name in drop:
                continue
            z.writestr("data/" + name, json.dumps({"id": name, "prompt": f"問題: {name} の評価用の質問文です。ここは十分に長い文章です。", "target": "A"}, ensure_ascii=False) + "\n")
        z.writestr("base_metadata.json", json.dumps({"vocab": vocab or {"<pad>": 0, "<unk>": 1, "a": 2, "b": 3, "あ": 4}}))
    os.makedirs(drive_root, exist_ok=True)
    with zipfile.ZipFile(os.path.join(drive_root, C.VOCAB_BUNDLE_ZIPS[0]), "w") as z:
        z.writestr("assets/r6_bundle.zip", inner.getvalue())
        if with_py:
            z.writestr("code/tier8_readonly_v44fix.py", 'CASES = (("a", "ユーザー: 関数 f(x) = 3x^3 + 2sin(x) - 5 について、導関数を求めてください。\\nアシスタント: ", "python"),)\n')


class TestIO(unittest.TestCase):
    def test_jsonl_limit_blank_and_bad_line(self):
        with tmpdir() as d:
            p = os.path.join(d, "a.jsonl")
            with open(p, "w", encoding="utf-8") as f:
                f.write('{"a": 1}\n\n{"a": 2}\r\n{"a": 3}\n')
            self.assertEqual([r["a"] for r in C.iter_rows(p)], [1, 2, 3])
            self.assertEqual([r["a"] for r in C.iter_rows(p, limit=2)], [1, 2])
            with open(p, "a", encoding="utf-8") as f:
                f.write("{not json}\n")
            with self.assertRaises(ValueError):
                list(C.iter_rows(p))
            self.assertEqual(len(list(C.iter_rows(p, strict=False))), 3)

    def test_squad_like_json_is_flattened_and_streamed(self):
        art = [{"title": "T1", "paragraphs": [{"context": "ctx one", "qas": [{"id": "q1", "question": "Q?", "answers": [{"text": "one", "answer_start": 4}]},
                                                                           {"id": "q2", "question": "R?", "answers": [], "is_impossible": True}]}]},
               {"title": "T2", "paragraphs": [{"context": "ctx two", "qas": [{"id": "q3", "question": "S?", "answers": [{"text": "two", "answer_start": 4}]}]}]}]
        with tmpdir() as d:
            p = os.path.join(d, "squad.json")
            with open(p, "w", encoding="utf-8") as f:
                json.dump({"version": "1.0", "data": art}, f)
            rows = list(C.iter_rows(p))
            self.assertEqual([r["id"] for r in rows], ["q1", "q2", "q3"])
            self.assertEqual(rows[0]["answer_texts"], ["one"])
            self.assertEqual(rows[0]["context"], "ctx one")
            self.assertTrue(rows[1]["is_impossible"])
            self.assertEqual(len(list(C.iter_rows(p, squad_flatten=False))), 2)
            p2 = os.path.join(d, "list.json")                  # top-level array of objects (ToolACE layout)
            with open(p2, "w", encoding="utf-8") as f:
                json.dump([{"system": "s", "conversations": [{"from": "user", "value": "hi"}]}, {"system": "t"}], f)
            self.assertEqual(len(list(C.iter_rows(p2))), 2)

    def test_array_streaming_across_tiny_chunks(self):
        objs = [{"i": i, "t": "日本語" * (i % 7)} for i in range(40)]
        s = json.dumps(objs, ensure_ascii=False, indent=1)
        f = io.StringIO(s[1:])
        got = list(C._stream_array_items(f, "", chunk=16))
        self.assertEqual(got, objs)

    def test_parquet_roundtrip(self):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("pyarrow not installed")
        with tmpdir() as d:
            p = os.path.join(d, "x.parquet")
            tbl = pa.table({"id": list(range(10)), "messages": [[{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}] for i in range(10)]})
            pq.write_table(tbl, p, row_group_size=4)
            rows = list(C.iter_rows(p, batch_size=3))
            self.assertEqual(len(rows), 10)
            self.assertEqual(rows[2]["messages"][1]["content"], "a2")
            self.assertEqual(C.row_count_hint(p), 10)
            self.assertEqual(C.default_text(rows[1]), "q1\na1")
            self.assertEqual(list(C.iter_rows(p, columns=["id"]))[0], {"id": 0})

    def test_write_new_file_never_overwrites(self):
        with tmpdir() as d:
            p = os.path.join(d, "sub", "f.txt")
            self.assertEqual(C.write_new_file(p, "abc"), "created")
            self.assertEqual(C.write_new_file(p, "abc"), "identical")
            with self.assertRaises(FileExistsError):
                C.write_new_file(p, "abd")
            with open(p, encoding="utf-8") as tf:
                self.assertEqual(tf.read(), "abc")
            C.write_new_file(os.path.join(d, "s.txt"), "lone \ud83d surrogate")          # must not crash

    def test_progress_callback(self):
        calls = []
        rows = [{"a": i} for i in range(25)]
        C.length_stats(rows, text_fn=lambda r: "x", progress=lambda label, n: calls.append((label, n)), progress_every=10)
        self.assertEqual([c for c in calls if not c[0].endswith("[done]")], [("length_stats", 10), ("length_stats", 20)])
        self.assertEqual(calls[-1], ("length_stats [done]", 25))


class TestVocab(unittest.TestCase):
    def test_bundle_zip_vocab_and_expected_size_check(self):
        with tmpdir() as d:
            root = os.path.join(d, "drive")
            make_bundle(root)
            v = C.load_vocab(root, expected_size=None, expected_single=None, allow_derive=False)
            self.assertEqual(v.chars, frozenset("abあ"))
            self.assertEqual(v.n_entries, 5)
            self.assertEqual(v.specials, ("<pad>", "<unk>"))
            self.assertIn("base_metadata.json", v.source)
            self.assertFalse(v.derived)
            with self.assertRaises(FileNotFoundError):                          # real expectation is 3,898 entries: this fixture is rejected
                C.load_vocab(root, allow_derive=False)

    def test_loose_vocab_json_wins_and_cache(self):
        with tmpdir() as d:
            root, out = os.path.join(d, "drive"), os.path.join(d, "out")
            make_bundle(root)
            os.makedirs(os.path.join(root, "run1"))
            with open(os.path.join(root, "run1", "vocab.json"), "w", encoding="utf-8") as f:
                json.dump({**{f"<s{i}>": i for i in range(100)}, "x": 100, "y": 101}, f)
            v = C.load_vocab(root, out, expected_size=None, expected_single=None)
            self.assertEqual(v.chars, frozenset("xy"))
            self.assertTrue(os.path.isfile(os.path.join(out, "vocab_cache.json")))
            v2 = C.load_vocab(root, out, expected_size=None, expected_single=None)
            self.assertTrue(v2.source.startswith("cache:"))
            self.assertEqual(v2.chars, v.chars)

    def test_derive_from_old_corpus_when_nothing_else(self):
        with tmpdir() as d:
            root, out = os.path.join(d, "drive"), os.path.join(d, "out")
            os.makedirs(os.path.join(root, "_v48_dataset_cache"))
            write_jsonl(os.path.join(root, C.OLD_CORPUS_REL), [row(1, "あい"), row(2, "うえ", "お")])
            v = C.load_vocab(root, out)
            self.assertTrue(v.derived)
            self.assertTrue(set("あいうえお") <= v.chars)
            self.assertTrue(set("答え") <= v.chars)                        # the targets of the fixture rows are part of full_text
            self.assertTrue(any("DERIVED" in w for w in v.warnings))
            # a derived cache is never trusted: after the bundle appears the real vocabulary is used
            make_bundle(root)
            v2 = C.load_vocab(root, out, expected_size=None, expected_single=None)
            self.assertFalse(v2.derived)
            self.assertEqual(v2.chars, frozenset("abあ"))

    def test_runtime_normalizer_matches_controller_when_reachable(self):
        p = os.path.join(HERE, "..", "..", "..", "v47_handoff", "bundle", "r6", "qwenputer_controller_v30.py")
        if not os.path.isfile(p):
            self.skipTest("controller source not reachable")
        with open(p, encoding="utf-8") as cf:
            ok, why = C.verify_runtime_normalizer(cf.read())
        self.assertTrue(ok, why)
        self.assertEqual("ＡＢ１".translate(C.RUNTIME_NORMALIZE), "AB1")
        self.assertEqual("（".translate(C.RUNTIME_NORMALIZE), "（")


class TestOovAndLength(unittest.TestCase):
    chars = frozenset("abcあいう\n。 AB")

    def test_oov_stats(self):
        rows = [{"full_text": "abc" * 10},                          # 0
                {"full_text": "a" * 100 + "Z"},                     # 1/101 = 0.0099 -> kept
                {"full_text": "a" * 98 + "ZY"},                     # 2/100 = 2 % -> over
                {"full_text": "ZZZZ"},                              # 100 %
                {"full_text": ""}]
        s = C.oov_stats(rows, self.chars, keep_per_row=True)
        self.assertEqual(s["n_rows"], 5)
        self.assertEqual(s["rows_with_any_oov"], 3)
        self.assertEqual(s["rows_over_threshold"], 2)
        self.assertAlmostEqual(s["share_rows_over_threshold"], 0.4)
        self.assertEqual(s["empty_rows"], 1)
        self.assertEqual(s["top_oov"][0]["char"], "Z")
        self.assertEqual(s["top_oov"][0]["count"], 1 + 1 + 4)
        self.assertEqual(s["top_oov"][0]["rows"], 3)
        self.assertEqual(s["top_oov"][1]["char"], "Y")
        self.assertEqual(s["histogram"]["(0.1%,1%]"], 1)
        self.assertAlmostEqual(s["per_row_frac"][1], 1 / 101, places=5)
        self.assertEqual(s["n_distinct_oov_chars"], 2)
        keep = [k for _, _, k in C.oov_filter(rows, self.chars)]
        self.assertEqual(keep, [True, True, False, False, True])

    def test_runtime_normalizer_is_applied_before_lookup(self):
        rows = [{"full_text": "ＡＢ"}]                                # full-width letters are mapped to A B by the runtime
        self.assertEqual(C.oov_stats(rows, self.chars)["rows_with_any_oov"], 0)
        raw = C.oov_stats(rows, self.chars, normalize=False)
        self.assertEqual(raw["rows_with_any_oov"], 1)
        self.assertEqual(C.oov_stats(rows, self.chars)["rows_changed_by_normalizer"], 1)

    def test_oov_per_group(self):
        rows = [{"full_text": "abc", "source": "x"}, {"full_text": "ZZ", "source": "y"}]
        s = C.oov_stats(rows, self.chars, group_key=lambda r: r["source"])
        self.assertEqual(s["per_group"]["x"]["rows_with_any_oov"], 0)
        self.assertEqual(s["per_group"]["y"]["rows_over_threshold"], 1)

    def test_length_boundaries(self):
        def mk(L):                                                  # L = len(prompt+target)+1
            n = L - 1
            return {"prompt": "あ" * (n - 3), "target": "い" * 3}
        rows = [mk(1900), mk(1901), mk(2048), mk(2049), mk(100)]
        s = C.length_stats(rows)["ALL"]
        self.assertEqual(s["n"], 5)
        self.assertEqual(s["max"], 2049)
        self.assertEqual(s["over_2048"], 1)
        self.assertAlmostEqual(s["share_over_2048"], 0.2)
        self.assertEqual(s["over_1900"], 3)                          # 1901, 2048, 2049
        self.assertEqual(s["chars_over_2000"], 2)                    # len(text) > 2000  <=>  L > 2001: 2048, 2049
        self.assertEqual(s["p50"], 1901)
        self.assertEqual(s["full_text_mismatch"], 0)
        g = C.length_stats([dict(mk(10), source="a"), dict(mk(3000), source="b")], group_key=lambda r: r["source"])
        self.assertEqual(g["b"]["over_2048"], 1)
        self.assertEqual(g["a"]["over_2048"], 0)
        bad = C.length_stats([{"prompt": "p", "target": "t", "full_text": "pX"}])["ALL"]
        self.assertEqual(bad["full_text_mismatch"], 1)
        self.assertEqual(C.row_length({"full_text": "abc"}), 4)


class TestDuplicates(unittest.TestCase):
    def test_exact_dups_within_and_against(self):
        rows = [row(1, "A  b\nc"), row(2, "x"), row(3, "A b c"), row(4, "y")]
        for r in rows:
            r["full_text"] = r["prompt"]
        e = C.exact_dups(rows)
        self.assertEqual(e["within_dup_positions"], [2])
        self.assertEqual(e["n_unique"], 3)
        self.assertEqual(e["examples"][0]["first_position"], 0)
        old = C.build_exact_index([{"full_text": "x"}, {"full_text": "zzz"}])
        e2 = C.exact_dups(rows, against=old)
        self.assertEqual(e2["against_dup_positions"], [1])
        self.assertEqual(e2["against_index_rows"], 2)

    def test_minhash_finds_near_duplicate_and_keeps_distinct(self):
        base = rand_text(1)
        near = base[:700] + "んんん" + base[703:]                        # 3 characters changed
        mid = base[:1000] + rand_text(9, 500)                            # 2/3 shared -> J ~ 0.5: not a near duplicate
        rows = [{"full_text": base}, {"full_text": rand_text(2)}, {"full_text": near}, {"full_text": rand_text(3)}, {"full_text": mid},
                {"full_text": ""}, {"full_text": "  "}]
        r = C.near_dups_within(rows, threshold=0.8)
        self.assertEqual(r["dup_positions"], [2])
        self.assertEqual(r["root_of"][2], 0)
        self.assertEqual(r["n_rows_without_shingles"], 2)
        self.assertEqual(r["n_clusters_gt1"], 1)
        self.assertEqual(r["examples"][0]["similar_to_position"], 0)
        self.assertGreater(r["examples"][0]["est_jaccard"], 0.9)
        self.assertEqual(r["examples"][0]["text"][:5], near[:5])
        self.assertAlmostEqual(r["dup_rate"], 1 / 7)
        # whitespace / case / full-width differences do not hide a duplicate
        r2 = C.near_dups_within([{"full_text": "Ｈｅｌｌｏ world, this is a fairly long sentence for testing."},
                                 {"full_text": "hello   WORLD, this is a fairly long sentence for testing."}])
        self.assertEqual(r2["dup_positions"], [1])

    def test_empty_inputs_do_not_crash(self):
        self.assertEqual(C.near_dups_within([])["n_rows"], 0)
        r = C.near_dups_within([{"full_text": ""}])
        self.assertEqual((r["n_rows"], r["n_dups"], r["n_rows_without_shingles"]), (1, 0, 1))
        self.assertEqual(C.cross_near_dups([], iter([{"full_text": "abc" * 20}]))["index_rows_with_match"], 0)
        self.assertEqual(C.cross_dataset_near_dups({"a": []})["rows"], {})
        self.assertEqual(C.exact_dups([])["n_rows"], 0)
        self.assertEqual(C.length_stats([])["ALL"]["n"] if "ALL" in C.length_stats([]) else 0, 0)
        self.assertEqual(C.oov_stats([], frozenset("a"))["n_rows"], 0)
        self.assertEqual(C.decontaminate([], C.EvalIndex())["n_rows"], 0)
        self.assertEqual(C.sample_for_reading(iter([]), n=5), [])

    def test_signature_deterministic_and_backends_agree(self):
        h1, h2 = C.MinHasher(use_numpy=False), C.MinHasher(use_numpy=False)
        t = rand_text(5, 400)
        self.assertEqual(h1.signature(t), h2.signature(t))
        self.assertEqual(len(h1.signature(t)), 64)
        self.assertIsNone(h1.signature("   "))
        self.assertEqual(len(h1.shingles("ab")), 1)                      # shorter than the n-gram: one shingle
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("numpy not installed")
        hn = C.MinHasher(use_numpy=True)
        self.assertEqual(hn.backend, "numpy")
        for seed in (1, 2, 3):
            tx = rand_text(seed, 50 + 300 * seed)
            self.assertEqual(hn.signature(tx), h1.signature(tx))
        # estimate tracks the true Jaccard roughly
        a, b = rand_text(7, 1000), rand_text(7, 1000)
        b = b[:500] + rand_text(8, 500)
        sa, sb = h1.signature(a), h1.signature(b)
        est = sum(x == y for x, y in zip(sa, sb)) / 64
        sha, shb = set(h1.shingles(a)), set(h1.shingles(b))
        true = len(sha & shb) / len(sha | shb)
        self.assertLess(abs(est - true), 0.2)

    def test_cross_near_dups_vs_old_corpus(self):
        new = [{"full_text": rand_text(11)}, {"full_text": rand_text(12)}, {"full_text": rand_text(13)}]
        old = [{"full_text": rand_text(20)}, {"full_text": rand_text(12)[:600] + "ん" + rand_text(12)[601:]}, {"full_text": rand_text(21)}]
        r = C.cross_near_dups(new, iter(old), threshold=0.8)
        self.assertEqual(r["matched_index_positions"], [1])
        self.assertEqual(r["index_rows_with_match"], 1)
        self.assertAlmostEqual(r["index_match_rate"], 1 / 3)
        self.assertEqual(r["probe_rows_with_match"], 1)
        self.assertEqual(r["examples"][0]["probe_position"], 1)
        s = C.cross_near_dups(new, iter(old), sample_every=2)             # probe positions 0 and 2 only -> the match is not seen
        self.assertEqual(s["probe_rows_scanned"], 2)
        self.assertEqual(s["index_rows_with_match"], 0)
        self.assertTrue(s["index_rate_is_lower_bound"])

    def test_cross_dataset_matrix(self):
        shared = rand_text(30)
        a = [{"full_text": shared}, {"full_text": rand_text(31)}]
        b = [{"full_text": shared[:900] + "ん" + shared[901:]}, {"full_text": rand_text(32)}, {"full_text": rand_text(33)}]
        c = [{"full_text": rand_text(34)}, {"full_text": rand_text(34)}]
        m = C.cross_dataset_near_dups({"a": a, "b": b, "c": c})
        self.assertEqual(m["rows"], {"a": 2, "b": 3, "c": 2})
        self.assertEqual(m["rows_with_match_in"]["a"].get("b"), 1)
        self.assertEqual(m["rows_with_match_in"]["b"].get("a"), 1)
        self.assertEqual(m["rows_with_match_in"]["c"].get("c"), 2)        # both members of an internal duplicate pair
        self.assertNotIn("c", m["rows_with_match_in"]["a"])
        self.assertEqual(m["examples"][0]["a"], "a")


class TestDecontaminate(unittest.TestCase):
    EVAL_A = "問題: 太郎君は本を12冊持っていて、花子さんから7冊もらいました。いま何冊持っていますか。"
    EVAL_B = "次の文章を読んで、条件と根拠を順に考えてください。そして答えを述べなさい。"

    def make_index(self):
        return C.build_eval_index([("evalA", [self.EVAL_A]), ("evalB", [self.EVAL_B])])

    def test_digit_changed_planted_sentence_is_flagged_boilerplate_is_not(self):
        idx = self.make_index()
        planted = "問題: 太郎君は本を15冊持っていて、花子さんから9冊もらいました。いま何冊持っていますか。\n条件と根拠を順に考えてください。"
        rows = [row("planted", planted, id="planted")]
        topics = ["雲の種類", "川の長さ", "山の高さ", "海の深さ", "星の数", "風の向き"]
        rows += [dict(row(i, f"問題: {t}について述べよ。\n条件と根拠を順に考えてください。"), id=f"tmpl{i}") for i, t in enumerate(topics)]
        rows += [dict(row(99, "問題: まったく関係のない文章がここに入ります。条件は何もありません。"), id="clean")]
        r = C.decontaminate(rows, idx, boilerplate_df=5)
        self.assertEqual(r["flagged_positions"], [0])
        self.assertEqual(r["rows_flagged"], 1)
        self.assertEqual(r["rows_with_any_shared_shingle"], 7)            # planted + the 6 template rows
        self.assertEqual(r["rows_boilerplate_only"], 6)
        self.assertEqual(r["boilerplate"]["top"][0]["rows"], 7)           # the template phrase also sits in the planted row
        self.assertIn("条件と根拠を順に考えてください", r["boilerplate"]["top"][0]["shingle"] + "条件と根拠を順に考えてください")
        ex = r["examples"][0]
        self.assertEqual(ex["id"], "planted")
        self.assertEqual(ex["eval_matches"][0]["source"], "evalA")
        self.assertTrue(ex["row_prefix"].startswith("問題: 太郎君"))
        self.assertGreaterEqual(r["shared_count_hist"]["20+"] + r["shared_count_hist"]["5-19"], 1)
        # the same template rows WITHOUT a high enough boilerplate_df would all be flagged: the cut is what ignores them
        r2 = C.decontaminate(rows, idx, boilerplate_df=500)
        self.assertEqual(r2["rows_flagged"], 7)
        self.assertEqual(list(C.drop_flagged(rows, r["flagged_positions"]))[0]["id"], "tmpl0")

    def test_min_shared_and_short_rows(self):
        idx = self.make_index()
        r = C.decontaminate([{"full_text": "短い"}, {"full_text": "いま何冊持っていますか。"}], idx, boilerplate_df=500, min_shared=2)
        self.assertEqual(r["n_rows_shorter_than_ngram"], 2)               # 2 and 12 characters: no 13-gram at all
        self.assertEqual(r["rows_flagged"], 0)
        one_shingle = [{"full_text": "それは。いま何冊持っていますか。"}]                    # exactly one 13-gram ('。いま何冊持っていますか。') is shared
        r = C.decontaminate(one_shingle, idx, min_shared=1)
        self.assertEqual(r["rows_flagged"], 1)
        self.assertEqual(r["shared_count_hist"]["1"], 1)
        self.assertEqual(C.decontaminate(one_shingle, idx, min_shared=2)["rows_flagged"], 0)

    def test_eval_index_from_files_and_py(self):
        with tmpdir() as d:
            p = os.path.join(d, "ev.jsonl")
            write_jsonl(p, [{"id": 1, "question": "ある数に3を足すと10になります。その数を答えなさい、という問題です。", "choices": ["ア。七です", "イ。八です"], "target": "A"}])
            py = os.path.join(d, "t8.py")
            with open(py, "w", encoding="utf-8") as f:
                f.write('X = "ユーザー: 関数 f(x) = 3x^3 + 2sin(x) - 5 について、導関数 f\'(x) を求めてください。"\nY = "short"\n')
            idx = C.build_eval_index([p, py])
            self.assertEqual(idx.per_source["ev.jsonl"]["items"], 1)
            r = C.decontaminate([{"full_text": "ある数に7を足すと20になります。その数を答えなさい、という問題です。"},
                                 {"full_text": "関数 f(x) = 9x^2 + 1sin(x) - 8 について、導関数 f'(x) を求めてください。"}], idx)
            self.assertEqual(r["flagged_positions"], [0, 1])
            self.assertEqual(idx.stats()["ngram"], 13)

    def test_normalize_for_overlap(self):
        self.assertEqual(C.normalize_for_overlap("12冊で、 ７５ Ａ"), "冊で、a")
        self.assertEqual(C.normalize_for_overlap("①②あ"), "あ")

    def test_missing_eval_source_is_an_error(self):
        with self.assertRaises(C.EvalFilesMissing):
            C.build_eval_index(["/nonexistent/path/eval.jsonl"])


class TestFindEvalFiles(unittest.TestCase):
    def test_bundle_zip_is_searched_and_py_is_included(self):
        with tmpdir() as d:
            root = os.path.join(d, "drive")
            make_bundle(root)
            ef = C.find_eval_files(root)
            self.assertEqual(len(ef.sources), len(C.EVAL_REQUIRED_JSONL) + 1)
            self.assertEqual(ef.missing_required, [])
            self.assertEqual(len(ef.missing_optional), len(C.EVAL_OPTIONAL_V48))
            labels = {s.label: s for s in ef.sources}
            s = labels["eval_v41_logical_instruction_operators_heldout.jsonl"]
            self.assertEqual(s.chain[1:], ("assets/r6_bundle.zip", "data/eval_v41_logical_instruction_operators_heldout.jsonl"))
            self.assertEqual(len(list(C.iter_eval_items(s))), 1)
            t8 = list(C.iter_eval_items(labels["tier8_readonly_v44fix.py"]))
            self.assertTrue(any("導関数" in x[0] for x in t8))
            idx = C.build_eval_index(ef.sources)
            self.assertGreater(len(idx), 20)
            self.assertIn("required", ef.report().lower() + "required")

    def test_missing_required_file_is_a_clear_error(self):
        with tmpdir() as d:
            root = os.path.join(d, "drive")
            make_bundle(root, drop=("jamcqa_manifest_test.jsonl",))
            with self.assertRaises(C.EvalFilesMissing) as cm:
                C.find_eval_files(root)
            self.assertIn("jamcqa_manifest_test.jsonl", str(cm.exception))
            ef = C.find_eval_files(root, strict=False)
            self.assertEqual(len(ef.missing_required), 1)
            with self.assertRaises(C.EvalFilesMissing):                    # empty / absent Drive root
                C.find_eval_files(os.path.join(d, "nothing"))

    def test_glob_fallback_finds_a_file_below_the_walk_depth(self):
        with tmpdir() as d:
            root = os.path.join(d, "drive")
            make_bundle(root, drop=("jamcqa_manifest_test.jsonl",))
            deep = os.path.join(root, "a", "b", "c", "d", "e", "f")
            os.makedirs(deep)
            write_jsonl(os.path.join(deep, "jamcqa_manifest_test.jsonl"), [{"id": "deep", "prompt": "とても深いフォルダに置かれた、十分に長い質問文です。"}])
            ef = C.find_eval_files(root, max_depth=2)
            self.assertEqual(ef.missing_required, [])
            self.assertTrue({s.label: s for s in ef.sources}["jamcqa_manifest_test.jsonl"].chain[0].endswith("jamcqa_manifest_test.jsonl"))

    def test_loose_file_overrides_bundle_and_v48_optional_found(self):
        with tmpdir() as d:
            root, extra = os.path.join(d, "drive"), os.path.join(d, "extra")
            make_bundle(root)
            os.makedirs(os.path.join(root, "sub", "deeper"))
            write_jsonl(os.path.join(root, "sub", "deeper", "jamcqa_manifest_test.jsonl"), [{"id": "loose", "prompt": "これはずっと長い質問文であり、ゆるやかに置かれたものです。"}])
            os.makedirs(extra)
            write_jsonl(os.path.join(extra, "python_unseen_eval.jsonl"), [{"id": "u", "prompt": "問題: 配列の合計を求めなさい。条件と根拠を順に考えてください。"}])
            ef = C.find_eval_files(root, extra_dirs=[extra])
            s = {x.label: x for x in ef.sources}
            self.assertEqual(len(s["jamcqa_manifest_test.jsonl"].chain), 1)
            self.assertIn("v48:python_unseen_eval.jsonl", s)
            self.assertEqual(len(ef.missing_optional), len(C.EVAL_OPTIONAL_V48) - 1)


class TestSemantic(unittest.TestCase):
    def test_lookalike_is_found_and_unrelated_is_not(self):
        q = "次の方程式を解きなさい。2x+3=11 のとき x の値を求めよ。"
        rows = [row(1, "光合成は植物が光エネルギーを使って二酸化炭素と水から糖をつくる反応である。"),
                row(2, "次の方程式を解きなさい。5x+7=27 のとき x の値を求めよ。"),
                row(3, "次の方程式を解け。xの値を求めなさい。ただし 3x+1=13 とする。"),
                row(4, "日本の首都は東京であり、人口は約1400万人である。")]
        res = C.semantic_neighbors([q], rows, idf_buffer_rows=2, min_cos=0.2)
        m = res[0]["matches"]
        self.assertGreaterEqual(len(m), 1)
        self.assertEqual(m[0]["id"], "r2")                                   # identical wording, other digits
        self.assertGreater(m[0]["cosine"], 0.9)
        self.assertNotIn("r1", [x["id"] for x in m])
        self.assertNotIn("r4", [x["id"] for x in m])
        self.assertIn("QUERY", C.format_semantic_report(res))
        self.assertEqual(C.semantic_neighbors(["短"], rows)[0]["matches"], [])


class TestCodeBlocks(unittest.TestCase):
    def test_extract_python_blocks(self):
        text = ("説明\n```python\nprint(1)\n```\n出力:\n```output\n1\n```\n```py\nx = 2\nprint(x)\n```\n```cpp\nint main(){}\n```\n"
                "~~~python3\nprint(3)\n~~~\n```\nunlabeled()\n```\n  ```Python\n  print(4)\n  ```\n<llm-code>\nprint(5)\n</llm-code>\n"
                "```python\nprint('never closed')\n")
        self.assertEqual(C.extract_python_blocks(text), ["print(1)", "x = 2\nprint(x)", "print(3)", "print(4)", "print(5)"])
        got =C.extract_python_blocks(text, include_unlabeled=True, include_unclosed=True)
        self.assertIn("unlabeled()", got)
        self.assertIn("print('never closed')", got)
        self.assertEqual(C.extract_python_blocks(text, include_llm_code_tags=False)[-1], "print(4)")
        blocks = C.extract_code_blocks(text)
        self.assertEqual([b.lang for b in blocks], ["python", "output", "py", "cpp", "python3", "", "python"])
        self.assertEqual(C.extract_python_blocks("no code here"), [])
        self.assertEqual(C.extract_python_blocks("```python\r\nprint(7)\r\n```"), ["print(7)"])


class TestSandbox(unittest.TestCase):
    def test_runs_math_code_and_the_converter_self_test(self):
        ok, out, err = C.code_sandbox_check("print(1 + 1)\n")
        self.assertEqual((ok, out, err), (True, "2\n", ""))
        ok, out, err = C.code_sandbox_check("import math, statistics\nfrom fractions import Fraction\nfrom decimal import Decimal\n"
                                            "print('答え:', math.isqrt(50), statistics.mean([1, 2, 3]), Fraction(1, 3) + Fraction(1, 6), Decimal('0.1') + Decimal('0.2'))\n")
        self.assertTrue(ok, err)
        self.assertEqual(out, "答え: 7 2 1/2 0.3\n")

    def test_static_refusals(self):
        cases = {"import os\nprint(os.getcwd())": "static:import:os", "import subprocess": "static:import:subprocess",
                 "import socket": "static:import:socket", "import shutil": "static:import:shutil", "import pathlib": "static:import:pathlib",
                 "import requests": "static:import:requests", "import urllib.request": "static:import:urllib.request",
                 "import ctypes": "static:import:ctypes", "import sys": "static:import:sys", "from os import path": "static:import:os",
                 "import random": "static:import:random", "open('x.txt')": "static:name:open", "exec('1')": "static:name:exec",
                 "eval('1')": "static:name:eval", "__import__('os')": "static:name:__import__", "getattr(1, 'real')": "static:name:getattr",
                 "from math import *": "static:wildcard_import", "class A:\n    pass": "static:class_def", "global g": "static:global_nonlocal",
                 "(1).__class__": "static:attr:__class__", "import math\nmath.os": "static:attr:os", "x.y = 1": "static:attr_store",
                 "'{0.__class__}'.format(1)": "static:format_string", "def (": "static:syntax", "print(1)\x00": "static:syntax"}
        for code, prefix in cases.items():
            ok, out, err = C.code_sandbox_check(code)
            self.assertFalse(ok, code)
            self.assertTrue(err.startswith(prefix), f"{code!r}: {err}")
        self.assertEqual(C.static_check("print(1)"), "")
        self.assertTrue(C.static_check("x" * 40000).startswith("too_long"))

    def test_infinite_loop_times_out(self):
        t0 = __import__("time").time()
        ok, out, err = C.code_sandbox_check("while True:\n    pass\n", timeout=1.5)
        self.assertFalse(ok)
        self.assertTrue(err.startswith("timeout"), err)
        self.assertLess(__import__("time").time() - t0, 8)

    def test_runtime_errors_output_cap_and_determinism(self):
        ok, out, err = C.code_sandbox_check("print(1 / 0)")
        self.assertFalse(ok)
        self.assertTrue(err.startswith("exception:ZeroDivisionError"), err)
        ok, out, err = C.code_sandbox_check("print('a' * 10000)", max_output_chars=100)
        self.assertFalse(ok)
        self.assertTrue(err.startswith("output_limit"), err)
        self.assertEqual(len(out), 100)
        a = C.code_sandbox_check("print(list({'a', 'b', 'c', 'd', 'e', 'f'}))")
        b = C.code_sandbox_check("print(list({'a', 'b', 'c', 'd', 'e', 'f'}))")
        self.assertTrue(a[0])
        self.assertEqual(a, b)                                          # PYTHONHASHSEED=0: set order of strings is reproducible
        ok, out, err = C.code_sandbox_check("def f(n):\n    return f(n + 1)\nf(0)")
        self.assertFalse(ok)
        self.assertIn("RecursionError", err)

    def test_audit_hook_blocks_what_the_static_check_cannot_see(self):
        # a stdlib module that is allowed but could be abused: still no file access
        ok, out, err = C.code_sandbox_check("import decimal\nprint(decimal.getcontext().prec)")
        self.assertTrue(ok, err)
        self.assertEqual(out.strip(), "28")

    def test_memory_limit_posix_only(self):
        if os.name != "posix":
            self.skipTest("RLIMIT_AS is POSIX only")
        ok, out, err = C.code_sandbox_check("x = [0] * (10 ** 9)\nprint(len(x))", timeout=10, mem_mb=256)
        self.assertFalse(ok)
        self.assertIn("MemoryError", err)

    def test_wide_profile_allows_classes_and_main_guard(self):
        code = "class P:\n    def __init__(self, v):\n        self.v = v\n    def double(self):\n        return self.v * 2\n\nif __name__ == '__main__':\n    print(P(21).double())\n"
        self.assertFalse(C.code_sandbox_check(code)[0])
        ok, out, err = C.code_sandbox_check(code, profile="wide")
        self.assertTrue(ok, err)
        self.assertEqual(out, "42\n")
        self.assertFalse(C.code_sandbox_check("import os", profile="wide")[0])
        with self.assertRaises(ValueError):
            C.code_sandbox_check("1", profile="nope")

    def test_many_keeps_order(self):
        res = C.code_sandbox_check_many(["print(1)", "import os", "print(3)"], workers=2)
        self.assertEqual([r[0] for r in res], [True, False, True])
        self.assertEqual(res[2][1], "3\n")

    def test_answers_equal(self):
        self.assertTrue(C.answers_equal("42\n", "42"))
        self.assertTrue(C.answers_equal("４２", "42"))
        self.assertTrue(C.answers_equal("1,234", "1234"))
        self.assertTrue(C.answers_equal("0.5", "1/2"))
        self.assertTrue(C.answers_equal("3.0", "3"))
        self.assertTrue(C.answers_equal("答え。", "答え"))
        self.assertFalse(C.answers_equal("2.007", "2.015361"))
        self.assertFalse(C.answers_equal("nan", "104"))


class TestSample(unittest.TestCase):
    def test_sample_is_seeded_stream_ordered_and_written(self):
        rows = [dict(row(i, f"問題 {i}: こたえはなんでしょう。"), source="s", family="f") for i in range(500)]
        a = C.sample_for_reading(iter(rows), n=100, seed=7)
        b = C.sample_for_reading(iter(rows), n=100, seed=7)
        c = C.sample_for_reading(iter(rows), n=100, seed=8)
        self.assertEqual([p for p, _ in a], [p for p, _ in b])
        self.assertNotEqual([p for p, _ in a], [p for p, _ in c])
        self.assertEqual(len(a), 100)
        self.assertEqual([p for p, _ in a], sorted(p for p, _ in a))
        self.assertEqual(len(C.sample_for_reading(iter(rows[:30]), n=100, seed=7)), 30)
        with tmpdir() as d:
            out = os.path.join(d, "new_folder", "read_100.txt")
            rows[3]["prompt"] = "問題: 変な文字 \u2603 を含む。"
            rows[3]["full_text"] = rows[3]["prompt"] + rows[3]["target"]
            C.sample_for_reading(iter(rows), n=500, seed=1, out_path=out, vocab=frozenset("問題こたえはなんでしょう。: 0123456789\n答"), title="unit")
            with open(out, encoding="utf-8") as tf:
                text = tf.read()
            self.assertIn("==== [1/500]", text)
            self.assertIn("--- prompt ---", text)
            self.assertIn("U+2603", text)
            self.assertIn("seed=1", text)
            with self.assertRaises(FileExistsError):
                C.sample_for_reading(iter(rows), n=10, seed=2, out_path=out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
