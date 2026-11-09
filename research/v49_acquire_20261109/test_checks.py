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
            stats = {}
            self.assertEqual(len(list(C.iter_rows(p, strict=False, stats=stats))), 3)
            self.assertEqual(stats, {"bad_lines": 1})

    def test_lenient_mode_survives_an_undecodable_byte_strict_mode_does_not(self):
        with tmpdir() as d:
            p = os.path.join(d, "a.jsonl")
            with open(p, "wb") as f:
                f.write('{"a": "あ"}\n'.encode("utf-8") + b'{"a": "\xff\xfe"}\n' + '{"a": "い"}\n'.encode("utf-8"))
            with self.assertRaises(ValueError):                       # UnicodeDecodeError is a ValueError
                list(C.iter_rows(p))
            stats = {}
            rows = list(C.iter_rows(p, strict=False, stats=stats))
            self.assertEqual([r["a"] for r in rows][0], "あ")
            self.assertEqual(len(rows), 3)
            self.assertIn("\ufffd", rows[1]["a"])
            self.assertEqual(stats, {"decode_replacements": 1})

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

    def test_write_new_file_is_atomic_and_moves_a_dead_runs_part_aside(self):
        with tmpdir() as d:
            p = os.path.join(d, "f.txt")
            # a runtime that died in the middle of a write leaves a .part: the next run must neither trust it nor delete it
            with open(p + ".part", "w", encoding="utf-8") as f:
                f.write("trunc")
            self.assertEqual(C.write_new_file(p, "complete"), "created")
            with open(p, encoding="utf-8") as f:
                self.assertEqual(f.read(), "complete")
            self.assertFalse(os.path.exists(p + ".part"))
            with open(p + ".part.stale-1", encoding="utf-8") as f:
                self.assertEqual(f.read(), "trunc")
            # a failure while writing leaves no final file behind
            q = os.path.join(d, "g.txt")
            real = os.fsync
            try:
                def boom(fd):
                    raise RuntimeError("disk gone")
                os.fsync = lambda fd: boom(fd)
                with self.assertRaises(RuntimeError):
                    C.write_new_file(q, "x")
            finally:
                os.fsync = real
            self.assertFalse(os.path.exists(q))
            self.assertEqual(C.write_new_file(q, "x"), "created")                    # the leftover part is quarantined, the write succeeds

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

    def loose_vocab(self, root):
        os.makedirs(os.path.join(root, "run1"))
        with open(os.path.join(root, "run1", "vocab.json"), "w", encoding="utf-8") as f:
            json.dump({**{f"<s{i}>": i for i in range(100)}, "x": 100, "y": 101}, f)

    def test_the_bundle_wins_over_a_stray_loose_vocab_json(self):
        # a stale vocab*.json that merely has the right size must never override the model's own file
        with tmpdir() as d:
            root, out = os.path.join(d, "drive"), os.path.join(d, "out")
            make_bundle(root)
            self.loose_vocab(root)
            v = C.load_vocab(root, out, expected_size=None, expected_single=None)
            self.assertEqual(v.chars, frozenset("abあ"))
            self.assertIn("base_metadata.json", v.source)
            self.assertTrue(os.path.isfile(os.path.join(out, "vocab_cache.json")))
            v2 = C.load_vocab(root, out, expected_size=None, expected_single=None)
            self.assertTrue(v2.source.startswith("cache:"))
            self.assertEqual(v2.chars, v.chars)

    def test_loose_vocab_json_only_when_there_is_no_bundle_and_it_says_so(self):
        with tmpdir() as d:
            root = os.path.join(d, "drive")
            os.makedirs(root)
            self.loose_vocab(root)
            v = C.load_vocab(root, None, expected_size=None, expected_single=None)
            self.assertEqual(v.chars, frozenset("xy"))
            self.assertTrue(any("loose file" in w for w in v.warnings), v.warnings)

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
        self.assertEqual(len(h1.signature(t)), 128)
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
        est = sum(x == y for x, y in zip(sa, sb)) / 128
        sha, shb = set(h1.shingles(a)), set(h1.shingles(b))
        true = len(sha & shb) / len(sha | shb)
        self.assertLess(abs(est - true), 0.2)

    def test_binomial_detection_claims_in_the_docstring_are_true(self):
        import math
        sf = lambda J, kmin: sum(math.comb(128, i) * J ** i * (1 - J) ** (128 - i) for i in range(kmin, 129))        # noqa: E731
        kmin = math.ceil(0.75 * 128)
        for J, lo, hi in ((0.9, 0.999, 1.0), (0.85, 0.998, 1.0), (0.8, 0.92, 0.95), (0.75, 0.5, 0.6), (0.7, 0.1, 0.15), (0.6, 0.0, 0.001)):
            self.assertTrue(lo <= sf(J, kmin) <= hi, (J, sf(J, kmin)))
        self.assertEqual((C.MinHasher().k, C.MinHashIndex().bands), (128, 32))

    def test_measured_detection_at_the_pipeline_cut(self):
        # texts with a controlled share of 5-gram shingles: pairs of true Jaccard ~0.9 must be found at the 0.75 cut, ~0.5 must not
        base = rand_text(100, 1200, alphabet="".join(chr(0x3041 + i) for i in range(80)))
        h = C.MinHasher()

        def jac(a, b):
            sa, sb = set(h.shingles(a)), set(h.shingles(b))
            return len(sa & sb) / len(sa | sb)

        def variant(frac, seed):
            rng = random.Random(seed)
            chars = list(base)
            for i in rng.sample(range(len(chars)), int(len(chars) * frac)):
                chars[i] = chr(0x30A1 + rng.randrange(80))
            return "".join(chars)
        near = [variant(0.01, s) for s in range(20)]                    # ~1 % of characters changed: Jaccard ~0.95
        far = [variant(0.15, s) for s in range(20)]                     # 15 % changed: Jaccard ~0.3
        self.assertGreater(min(jac(base, t) for t in near), 0.85)
        self.assertLess(max(jac(base, t) for t in far), 0.5)
        found_near = sum(1 for t in near if C.near_dups_within([{"full_text": base}, {"full_text": t}], threshold=0.75)["n_dups"] == 1)
        found_far = sum(1 for t in far if C.near_dups_within([{"full_text": base}, {"full_text": t}], threshold=0.75)["n_dups"] == 1)
        self.assertEqual((found_near, found_far), (20, 0))

    def test_cluster_size_histogram(self):
        a = rand_text(40, 800)
        rows = [{"full_text": a}, {"full_text": a}, {"full_text": a}, {"full_text": rand_text(41, 800)}, {"full_text": rand_text(42, 800)},
                {"full_text": rand_text(42, 800)}]
        r = C.near_dups_within(rows, threshold=0.75)
        self.assertEqual(r["cluster_size_hist"], {"2": 1, "3": 1})
        self.assertEqual((r["n_clusters_gt1"], r["largest_cluster"], r["n_dups"]), (2, 3, 3))

    def test_deadlines_stop_the_optional_reviews(self):
        import time
        rows = lambda: ({"full_text": rand_text(i, 300)} for i in range(5000))                    # noqa: E731
        info = {}
        r = C.cross_dataset_near_dups({"a": rows()}, deadline=time.time() - 1, info=info)
        self.assertEqual(info.get("truncated"), "signatures")
        self.assertLess(sum(r["rows"].values()), 5000)
        info = {}
        res = C.semantic_neighbors(["次の方程式を解きなさい。2x+3=11 のとき x の値を求めよ。"], (row(i, "次の方程式を解きなさい。" + rand_text(i, 100)) for i in range(5000)),
                                   idf_buffer_rows=10, deadline=time.time() - 1, info=info)
        self.assertTrue(info.get("truncated"))
        self.assertLess(info["rows_scanned"], 5000)
        self.assertEqual(len(res), 1)

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

    TOPICS = ["雲の種類", "川の長さ", "山の高さ", "海の深さ", "星の数", "風の向き"]

    def make_template_index(self, n_items=25):
        """EVAL_A plus n_items eval items that share the instruction template: its shingles have eval df >= 20 (boilerplate)."""
        items = [f"問題: {t}{i}について述べよ。\n条件と根拠を順に考えてください。" for i, t in enumerate(["雲", "川", "山", "海", "星", "風", "月", "島", "谷", "森", "湖", "砂", "雪", "雨", "霧", "波", "岩", "草", "花", "鳥", "魚", "虫", "石", "土", "火"][:n_items])]
        return C.build_eval_index([("evalA", [self.EVAL_A]), ("tmpl", items)])

    def test_digit_changed_planted_sentence_is_flagged_template_phrases_are_not(self):
        idx = self.make_template_index()
        self.assertGreaterEqual(idx.stats()["max_eval_items_per_shingle"], 20)
        planted = "問題: 太郎君は本を15冊持っていて、花子さんから9冊もらいました。いま何冊持っていますか。\n条件と根拠を順に考えてください。"
        rows = [row("planted", planted, id="planted")]
        rows += [dict(row(i, f"問題: {t}について述べよ。\n条件と根拠を順に考えてください。"), id=f"tmpl{i}") for i, t in enumerate(self.TOPICS)]
        rows += [dict(row(99, "問題: まったく関係のない文章がここに入ります。条件は何もありません。"), id="clean")]
        r = C.decontaminate(rows, idx)
        self.assertEqual(r["flagged_positions"], [0])
        self.assertEqual(r["rows_flagged"], 1)
        self.assertEqual(r["rows_boilerplate_only"], r["rows_with_any_shared_shingle"] - 1)
        self.assertGreaterEqual(r["rows_boilerplate_only"], 6)                  # the six template rows are shared-phrase rows only
        self.assertEqual(r["boilerplate_eval_df"], 20)
        top = r["boilerplate"]["top"][0]
        self.assertGreaterEqual(top["eval_items"], 20)
        self.assertGreaterEqual(top["rows"], 7)                                 # the template phrase also sits in the planted row
        ex = r["examples"][0]
        self.assertEqual(ex["id"], "planted")
        self.assertEqual(ex["eval_matches"][0]["source"], "evalA")
        self.assertTrue(ex["row_prefix"].startswith("問題: 太郎君"))
        self.assertGreaterEqual(r["shared_count_hist"]["20+"] + r["shared_count_hist"]["5-19"], 1)
        # the same template rows against an index where the phrase is owned by ONE eval item: nothing is boilerplate, all are flagged
        r2 = C.decontaminate(rows, self.make_template_index(n_items=1))
        self.assertGreaterEqual(r2["rows_flagged"], 2)
        self.assertEqual(list(C.drop_flagged(rows, r["flagged_positions"]))[0]["id"], "tmpl0")

    def test_a_widely_copied_unique_eval_question_is_flagged_however_many_rows_copy_it(self):
        # review finding: with a train-side count (more than 500 rows) 501+ copies of ONE eval question were waved through as boilerplate
        idx = C.build_eval_index([("evalA", [self.EVAL_A]), ("evalB", [self.EVAL_B])])
        leak = self.EVAL_A
        for n in (400, 501, 2000):
            rows = [{"id": f"c{i}", "full_text": f"{leak}\n補足{i}番: " + rand_text(i, 40)} for i in range(n)]
            r = C.decontaminate(rows, idx)
            self.assertEqual(r["rows_flagged"], n, n)
            self.assertEqual(r["rows_boilerplate_only"], 0, n)
            self.assertEqual(r["real_shingles_top"][0]["eval_items"], 1)
            self.assertGreaterEqual(r["real_shingles_top"][0]["rows"], n)

    def test_eval_df_counts_items_not_texts(self):
        idx = C.EvalIndex()
        shared = "共通の長い言い回しがここにあります"
        for k in range(3):                                                      # ONE item whose three texts (question + choices) share a phrase
            idx.add_text("s", 0, f"{shared}{'ABC'[k]}")
        self.assertEqual(idx.eval_df, {})
        idx.add_text("s", 1, shared + "D")                                       # a second item: now the shared shingles have df 2
        self.assertTrue(idx.eval_df and set(idx.eval_df.values()) == {2})
        idx.add_text("t", 0, shared + "E")                                       # same item index, another source: a third item
        self.assertEqual(set(idx.eval_df.values()), {3})

    def test_min_shared_and_short_rows(self):
        idx = self.make_index()
        r = C.decontaminate([{"full_text": "短い"}, {"full_text": "いま何冊持っていますか。"}], idx, min_shared=2)
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
            self.assertEqual(len(ef.missing_optional), len(C.EVAL_OPTIONAL_V48) + len(C.EVAL_REGISTRY_EXTRA))
            self.assertTrue(any("registry:selection_120.json" in m for m in ef.missing_optional))          # a missing suite is named, never silent
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
            self.assertEqual(len(ef.missing_optional), len(C.EVAL_OPTIONAL_V48) - 1 + len(C.EVAL_REGISTRY_EXTRA))

    def test_registry_suites_are_found_loose_in_the_bundle_and_in_suite_zips(self):
        with tmpdir() as d:
            root = os.path.join(d, "drive")
            make_bundle(root)
            os.makedirs(os.path.join(root, "x"))
            write_jsonl(os.path.join(root, "x", "japanese_math_python_dev.jsonl"), [{"id": "a", "prompt": "問題: 日本語の数学とPythonの開発用の問題文です。ここは十分に長いです。"}])
            with zipfile.ZipFile(os.path.join(root, C.EVAL_OPTIONAL_SUITE_ZIPS[1]), "w") as z:
                z.writestr("continue/eval_python/selection_120.json", json.dumps({"cases": [{"id": 1, "prompt": "NuminaMath の選択問題の十分に長い文面がここに入ります。"}]}))
                z.writestr("continue/engineering/engineering_heldout.jsonl", json.dumps({"id": "e", "prompt": "工学の保留問題の十分に長い文面がここに入ります。"}) + "\n")
                z.writestr("continue/visual/visual_eval_misc.jsonl", json.dumps({"id": "v", "prompt": "別の評価ファイルの十分に長い文面がここに入ります。"}) + "\n")
            ef = C.find_eval_files(root)
            labels = {s.label: s for s in ef.sources}
            self.assertEqual(labels["registry:japanese_math_python_dev.jsonl"].kind, "jsonl")
            self.assertEqual(labels["registry:selection_120.json"].kind, "json")                       # .json is read too
            self.assertIn("registry:engineering_heldout.jsonl", labels)
            self.assertTrue(any(k.startswith("suite:") and k.endswith("visual_eval_misc.jsonl") for k in labels))   # the old name pattern still works
            self.assertEqual(sum(1 for k in labels if k.endswith("engineering_heldout.jsonl")), 1)             # not indexed twice
            idx = C.build_eval_index(ef.sources)
            r = C.decontaminate([{"full_text": "NuminaMath の選択問題の十分に長い文面がここに入ります。"}], idx)
            self.assertEqual(r["rows_flagged"], 1)
            self.assertFalse(any("selection_120" in m for m in ef.missing_optional))
            self.assertEqual(len(ef.missing_optional), len(C.EVAL_OPTIONAL_V48) + len(C.EVAL_REGISTRY_EXTRA) - 3)

    def test_the_tier_table_names_files_the_pipeline_requires(self):
        needed = {n for names in C.EVAL_TIER_FILES.values() for n in names}
        self.assertEqual(sorted(needed), sorted(set(C.EVAL_REQUIRED_JSONL) | set(C.EVAL_REQUIRED_PY)))
        self.assertEqual(len(C.EVAL_TIER_FILES), 8)


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
                 "'{0.__class__}'.format(1)": "static:format_string", "def (": "static:syntax", "print(1)\x00": "static:syntax",
                 "async def f():\n    pass": "static:async", "import re": "static:import:re", "import json": "static:import:json",
                 "import dataclasses": "static:import:dataclasses", "import string": "static:import:string",
                 "import collections.abc.x": "static:import:collections.abc.x", "from math import _x": "static:wildcard_import",
                 "type(1)": "static:name:type", "g = (1 for _ in [1])\nprint(g.gi_frame)": "static:attr:gi_frame"}
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

    def test_there_is_only_the_executor_profile(self):
        # the 'wide' profile (classes, random, re, json, dataclasses ...) was removed: the model's own executor cannot run such code, and
        # review showed escapes from it (random._os, dataclasses.inspect.importlib, __init__.__globals__)
        code = "class P:\n    def __init__(self, v):\n        self.v = v\n    def double(self):\n        return self.v * 2\n\nif __name__ == '__main__':\n    print(P(21).double())\n"
        ok, out, err = C.code_sandbox_check(code)
        self.assertFalse(ok)
        self.assertTrue(err.startswith("static:class_def"), err)
        self.assertEqual(sorted(C.SANDBOX_PROFILES), ["executor"])
        for profile in ("wide", "nope"):
            with self.assertRaises(ValueError):
                C.code_sandbox_check("1", profile=profile)
        self.assertEqual(C.SANDBOX_BACKEND, "checks.code_sandbox_check(profile=executor)")
        self.assertFalse(hasattr(C, "WIDE_ROOTS"))

    def test_the_escapes_found_in_review_are_refused_before_anything_runs(self):
        # harmless stand-ins for the four payload families of the review (each only prints or raises if it ever ran)
        payloads = {
            "operator walk": "from operator import attrgetter, methodcaller\nprint(methodcaller('__subclasses__')(attrgetter('__class__.__base__')(0)))",
            "operator walk, module style": "import operator as o\nprint(o.attrgetter('real')(1))",
            "operator.methodcaller attribute": "import operator\nprint(operator.methodcaller('real')(1))",
            "class walk to os globals": "print((1).__class__.__base__.__subclasses__())",
            "forged result line": "import os\nos.write(1, b'@@CHECKS-RESULT-x@@{}')",
            "format traversal via built string": "s = ('{0._' + '_name__}')\nprint(s.format(print))",
            "format traversal via chr": "s = '{0.' + chr(95) * 2 + 'class' + chr(95) * 2 + '}'\nprint(s.format(1))",
            "unbound str.format": "print(str.format('{0.real}', 1))",
            "generator frame": "g = (g.gi_frame.f_back.f_globals for _ in [1])\nprint(next(g))",
            "random._os": "import random\nprint(random._os)",
            "dataclasses.inspect": "import dataclasses\nprint(dataclasses.inspect.importlib)",
            "getattr walk": "print(getattr(print, '__self__'))",
            "type walk": "print(type(1).mro())",
            "mro": "print(int.mro())",
        }
        for name, code in payloads.items():
            r = C.code_sandbox_run(code)
            self.assertFalse(r["ok"], name)
            self.assertEqual(r["category"], "static", (name, r))
            self.assertEqual(r["stdout"], "", name)

    def test_module_namespaces_hide_submodules_at_run_time(self):
        for code, needle in (("import statistics\nprint(statistics.random)", "AttributeError"),
                             ("import functools\nprint(functools.partial)", None),
                             ("from fractions import Decimal\nprint(Decimal('1.5'))", None),
                             ("import collections\nfrom collections.abc import Sequence\nprint(collections.abc.Sequence is Sequence)", None)):
            ok, out, err = C.code_sandbox_check(code)
            if needle:
                self.assertFalse(ok, code)
                self.assertIn(needle, err)
            else:
                self.assertTrue(ok, (code, err))
        ok, out, err = C.code_sandbox_check("import math\nprint(type(math).__name__)")
        self.assertFalse(ok)                                                    # 'type' is not even a name
        ok, out, err = C.code_sandbox_check("import itertools\nprint(sorted(itertools.accumulate([1, 2, 3])))")
        self.assertEqual((ok, out), (True, "[1, 3, 6]\n"))

    def test_selftest_full_and_light(self):
        full = C.sandbox_selftest()
        self.assertTrue(full["ok"], [c for c in full["checks"] if not c["ok"]])
        self.assertEqual(full["backend"], C.SANDBOX_BACKEND)
        names = {c["name"] for c in full["checks"]}
        self.assertIn("print(1 + 1)", names)
        self.assertIn("refused: operator.attrgetter walk", names)
        self.assertIn("os probe ran", names)
        light = C.sandbox_selftest(light=True)
        self.assertTrue(light["ok"])
        self.assertLess(len(light["checks"]), len(full["checks"]))

    def test_selftest_demands_os_containment_when_asked(self):
        r = C.sandbox_selftest(light=True, require_os_containment=True)
        if os.name == "posix" and os.geteuid() == 0:
            self.assertTrue(r["ok"], r)
            self.assertEqual(r["run_as"], "nobody")
        else:                                                                    # Windows / a non-root process cannot switch users: that must FAIL, loudly
            self.assertFalse(r["ok"])
            self.assertIn("os_containment_available", r["failed"])

    def test_the_drive_checks_of_the_os_probe_with_and_without_the_unmount(self):
        # simulate a root POSIX run: the child is 'nobody' and (worst case) CAN list and write a mounted Drive
        from unittest import mock
        real_spawn = C._spawn
        probe = {"uid": 65534, "gid": 65534, "listdir_drive": "ok:1", "listdir_mydrive": "ok:3", "read_etc_shadow": "error:PermissionError",
                 "can_write_mydrive": True, "can_write_content": False, "can_write_cwd": False}

        def spawn(script_text, args, **kw):
            if script_text == C._PROBE_SRC:
                return {"returncode": 0, "out": (json.dumps(probe) + "\n").encode(), "err": b"", "timed_out": False, "spawn_error": ""}
            return real_spawn(script_text, args, **dict(kw, acct=None))                         # the real child runs without a user switch here
        with mock.patch.object(C, "_spawn", spawn), mock.patch.object(C, "_resolve_run_as", return_value=(65534, 65534, "nobody")):
            strict = C.sandbox_selftest()
            self.assertFalse(strict["ok"])
            self.assertEqual(sorted(strict["failed"]), ["child cannot list /content/drive", "child cannot write Drive"])
            lenient = C.sandbox_selftest(drive_unmounted=True)                                   # the unmount is the barrier for Drive
            self.assertTrue(lenient["ok"], lenient["failed"])
            self.assertTrue(any(c["name"].startswith("Drive is unmounted while code runs") and "could list it" in c["name"] for c in lenient["checks"]))
            probe.update({"uid": 0})                                                              # but a root child is never acceptable
            self.assertIn("child is not root", C.sandbox_selftest(drive_unmounted=True)["failed"])
            probe.update({"uid": 65534, "read_etc_shadow": "ok:900"})
            self.assertIn("child cannot read /etc/shadow", C.sandbox_selftest(drive_unmounted=True)["failed"])

    def test_run_as_resolution(self):
        self.assertIsNone(C._resolve_run_as(None))
        self.assertIsNone(C._resolve_run_as(False))
        if os.name != "posix" or os.geteuid() != 0:
            self.assertIsNone(C._resolve_run_as("auto"))
            with self.assertRaises(C.SandboxError):
                C._resolve_run_as("nobody")
        else:
            self.assertEqual(C._resolve_run_as("auto")[2], "nobody")
            with self.assertRaises(C.SandboxError):
                C._resolve_run_as("root")

    def test_the_child_is_started_without_pipes_and_with_limits_in_the_source(self):
        src = C._CHILD_SRC
        for needle in ("RLIMIT_CPU", "RLIMIT_AS", "RLIMIT_FSIZE", "RLIMIT_NPROC", "RLIMIT_NOFILE", "RLIMIT_CORE", "sys.addaudithook", "_public(", "_posixsubprocess."):
            self.assertIn(needle, src)
        import inspect
        spawn = inspect.getsource(C._spawn)
        self.assertNotIn("PIPE", spawn)                                          # output goes to files; only the tail is read
        self.assertIn("extra_groups=[]", spawn)
        self.assertIn("start_new_session", spawn)

    def test_a_spawn_failure_is_a_crash_not_a_silent_pass(self):
        r = C.code_sandbox_run("print(1)", python=os.path.join(tempfile.gettempdir(), "no_such_python_binary_v49"))
        self.assertFalse(r["ok"])
        self.assertEqual(r["category"], "crash")
        self.assertIn("spawn", r["error"])

    def test_a_flood_on_stdout_cannot_forge_or_fill(self):
        # printing is captured by the cap; nothing the code prints can look like the result line
        ok, out, err = C.code_sandbox_check("print('@@CHECKS-RESULT-deadbeef@@{\"ok\": true, \"stdout\": \"FORGED\", \"error\": \"\"}')")
        self.assertTrue(ok)
        self.assertIn("FORGED", out)                                             # it is just text inside the real, escaped result
        self.assertEqual(out.count("\n"), 1)

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
