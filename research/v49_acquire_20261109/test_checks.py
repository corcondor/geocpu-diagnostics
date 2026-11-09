#!/usr/bin/env python3
"""Unit tests for checks.py on hand-made fixtures (every fixture is a few KB; nothing is downloaded).

    python -I -X utf8 test_checks.py            # one small process
"""
import array
import io
import json
import os
import random
import subprocess
import sys
import tempfile
import time
import unittest
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import checks as C                                                                  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def tmpdir():
    return tempfile.TemporaryDirectory(prefix="test_checks_", ignore_cleanup_errors=True)


def run_runtime_only(code, timeout=30.0):
    """Run `code` in the REAL child with only its own validate() call replaced by a plain parse: the static rules are OFF, everything else (restricted
    builtins and __import__, the module namespaces built from the frozen lists, the audit hook) is the real child.  Returns the child's result dict
    {ok, stdout, error}.  Built here from checks._CHILD_SRC on purpose (not from a helper of checks.py), so the test also runs against older checks.py."""
    src = C._CHILD_SRC.replace("tree = validate(code, cfg)", 'tree = ast.parse(code, filename="<row-code>", mode="exec")')
    assert src != C._CHILD_SRC, "the validate() anchor of the child source moved"
    marker = "@@CHECKS-RESULT-test@@"
    cfg = C._profile_cfg("executor", None, False)
    cfg.update({"cpu": int(timeout) + 1, "cap": 1 << 16, "marker": marker, "mem_mb": 1024})
    r = C._spawn(src, [json.dumps(cfg), "code.py"], timeout=timeout, flags=["-s", "-B", "-X", "utf8", "-S"], python=None, acct=None, files={"code.py": code})
    for line in reversed(r["out"].decode("utf-8", "replace").splitlines()):
        if line.startswith(marker):
            return json.loads(line[len(marker):])
    raise AssertionError(f"the child gave no result line: returncode={r['returncode']} timed_out={r['timed_out']} err={r['err'][-300:]!r}")


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

    def test_an_fsync_error_that_is_not_just_unsupported_is_not_swallowed(self):
        # review round 3: every fsync error was swallowed.  On Drive (FUSE) the quota-full / I/O error that buffered writes defer to fsync would
        # have been recorded as a successful write; only "this file system cannot fsync" may be ignored.
        import errno
        from unittest import mock
        with tmpdir() as d:
            p = os.path.join(d, "a.txt")
            for code in (errno.EIO, errno.ENOSPC):
                with mock.patch("os.fsync", side_effect=OSError(code, "deferred write error")):
                    with self.assertRaises(OSError):
                        C.write_new_file(p, "x")
                self.assertFalse(os.path.exists(p), code)                      # nothing was renamed into place
            with mock.patch("os.fsync", side_effect=OSError(errno.EINVAL, "fsync not supported")):
                self.assertEqual(C.write_new_file(p, "x"), "created")           # a file system without fsync still works
            with open(p, encoding="utf-8") as f:
                self.assertEqual(f.read(), "x")

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

    def test_a_chain_of_near_pairs_does_not_wipe_out_rows_that_are_far_from_the_survivor(self):
        # review round 3: connected components let a chain a~b~c~... drop rows with Jaccard ~0 to the survivor (31 chained rows: 13 of the
        # 16 removed ones were < 0.5 from the kept row).  Now a row is dropped only when it is near a row that is KEPT.
        base = rand_text(77, 1200, alphabet="".join(chr(0x3041 + i) for i in range(80)))
        noise = rand_text(78, 1400, alphabet="".join(chr(0x30A1 + i) for i in range(80)))

        def link(k):                                                    # the first 100*k characters replaced: neighbours differ by one 100-char block
            return noise[:100 * k] + base[100 * k:]
        rows = [{"full_text": link(k)} for k in range(12)]
        h = C.MinHasher()

        def jac(a, b):
            sa, sb = set(h.shingles(a)), set(h.shingles(b))
            return len(sa & sb) / len(sa | sb)
        self.assertGreater(min(jac(rows[k]["full_text"], rows[k + 1]["full_text"]) for k in range(11)), 0.8)        # a chain of near pairs ...
        self.assertLess(jac(rows[0]["full_text"], rows[11]["full_text"]), 0.05)                                     # ... whose ends have nothing in common
        r = C.near_dups_within(rows, threshold=0.75)
        kept = [i for i in range(12) if r["root_of"][i] == i]
        self.assertGreaterEqual(len(kept), 4)                           # connected components would keep exactly one row
        self.assertLessEqual(r["largest_cluster"], 3)
        for i in range(12):
            self.assertEqual(r["root_of"][r["root_of"][i]], r["root_of"][i])           # the survivor of a dropped row is itself kept
            if r["root_of"][i] != i:
                self.assertGreater(jac(rows[i]["full_text"], rows[r["root_of"][i]]["full_text"]), 0.7, i)
        for a in kept:                                                  # no two survivors are near each other
            for b in kept:
                if a < b:
                    self.assertLess(jac(rows[a]["full_text"], rows[b]["full_text"]), 0.8, (a, b))

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

    def test_normalize_for_overlap_applies_the_loaders_symbol_map(self):
        # review round 3: the loader turns U+2212 into '-', Greek capitals into Latin ones, ... ; text that becomes identical tokens at
        # training time must be identical here too (a short eval sentence whose every 13-gram window holds the symbol was missed)
        self.assertEqual(C.normalize_for_overlap("ここでは値から−をとって計算します"), C.normalize_for_overlap("ここでは値から-をとって計算します"))
        self.assertEqual(C.normalize_for_overlap("ΑΒΕ型の判定をここで行います"), C.normalize_for_overlap("ABE型の判定をここで行います"))
        self.assertEqual(C.normalize_for_overlap("a÷b⟨x⟩"), C.normalize_for_overlap("a/b(x)"))
        idx = C.build_eval_index([("e", ["ここでは値から−をとって計算します"])])
        r = C.decontaminate([{"full_text": "ここでは値から-をとって計算します"}], idx)
        self.assertEqual(r["flagged_positions"], [0])

    def test_a_source_that_repeats_another_sources_content_still_reports_its_own_shingles(self):
        # review round 3 (blocker): the per-source count only counted shingles NEW to the index, so eval_swe_step_control_heldout_v43labels.jsonl
        # (the same prompts as eval_swe_step_control_heldout.jsonl) reported 0 and step 6b refused the real required eval files
        texts = [self.EVAL_A, self.EVAL_B]
        idx = C.build_eval_index([("first", texts), ("second", texts), ("short", ["短い文だけ"])])
        ps = idx.per_source
        self.assertGreater(ps["second"]["shingles"], 0)
        self.assertEqual(ps["first"]["shingles"], ps["second"]["shingles"])
        self.assertEqual(ps["second"]["new_shingles"], 0)
        self.assertEqual(ps["first"]["new_shingles"], ps["first"]["shingles"])
        self.assertEqual((ps["second"]["items"], ps["second"]["indexable_texts"]), (2, 2))
        self.assertEqual((ps["short"]["shingles"], ps["short"]["indexable_texts"]), (0, 0))              # nothing indexable: 6b still refuses such a source

    def test_a_verbatim_copy_of_a_templated_eval_item_is_flagged_although_all_its_shingles_are_boilerplate(self):
        # review round 3 (major): items that differ only in digits share ALL their shingles, so every shingle had eval df >= 20 and a verbatim
        # copy of a whole item (T1 193/200, T6 100/200, v48 unseen_eval 96-128/128 in the real files) was never flagged
        items = [f"問題: 箱に{i + 2}個のりんごが入っています。そこから{i}個取り出すと、残りは何個になりますか。理由も書きなさい。" for i in range(30)]
        idx = C.build_eval_index([("T1", items)])
        self.assertGreaterEqual(idx.stats()["max_eval_items_per_shingle"], 20)
        rows = [{"id": "copy", "full_text": items[7] + "\n答え: 5個。"},
                {"id": "copy_other_digits", "full_text": "問題: 箱に91個のりんごが入っています。そこから38個取り出すと、残りは何個になりますか。理由も書きなさい。\n答え: 53個。"},
                {"id": "partial_template_only", "full_text": "メモ: そこから4個取り出すと、残りは何個になりますか。という言い回しだけを借りた別の文章です。"},
                {"id": "clean", "full_text": "まったく関係のない文章がここに入ります。条件は何もありません。十分に長い文字列です。"}]
        r = C.decontaminate(rows, idx)
        self.assertEqual(r["flagged_positions"], [0, 1])
        self.assertEqual(r["rows_flagged_whole_item"], 2)
        self.assertEqual(r["rows_boilerplate_only"], 1)                          # the shared phrase alone is still boilerplate
        self.assertEqual(r["examples"][0]["id"], "copy")
        self.assertEqual(r["examples"][0]["eval_matches"][0]["source"], "T1")
        # 2,000 verbatim copies of one item: all flagged
        many = [{"id": f"m{i}", "full_text": items[3] + f"\n補足{i}: " + rand_text(i, 30)} for i in range(2000)]
        self.assertEqual(C.decontaminate(many, idx)["rows_flagged"], 2000)
        # a short eval text (< 40 normalised characters) is a phrase, not an item: it never flags a row by itself
        idx2 = C.build_eval_index([("s", [f"次の文章を読んで、条件を順に考えなさい{i}。" for i in range(30)])])
        self.assertEqual(C.decontaminate([{"id": "x", "full_text": "次の文章を読んで、条件を順に考えなさい7。"}], idx2)["rows_flagged"], 0)

    def test_missing_eval_source_is_an_error(self):
        with self.assertRaises(C.EvalFilesMissing):
            C.build_eval_index(["/nonexistent/path/eval.jsonl"])


class Hotpot:
    """Synthetic HotpotQA-like corpus for TestAsciiRunDecontaminate: validation QUESTIONS (the eval texts) and training rows made of two ~1,500
    character Wikipedia-style paragraphs.  Questions and paragraphs use DIFFERENT sentence frames and share only word-level vocabulary, so they
    collide on 2-3 word phrases (13 squashed characters) and, apart from chance, never on a long verbatim stretch.  Deterministic."""
    ADJ = ("american british french german canadian australian english irish scottish italian spanish japanese russian national royal local old new "
           "northern southern eastern western central independent annual popular famous early modern classic regional public private professional "
           "military political cultural historic municipal rural urban coastal mountain").split()
    NOUN = ("magazine newspaper company university school college station theatre museum library orchestra festival league club band album film "
            "series novel opera hospital bridge railway airport castle church cathedral village district county council committee "
            "brigade regiment battalion division squadron corps foundation society institute academy gallery studio network channel").split()
    VERB = ("published founded established released opened directed produced written built formed organized operated announced elected appointed "
            "recorded performed broadcast introduced designed funded renamed merged acquired launched presented developed").split()
    PLACE = ("London Paris Berlin Chicago Boston Toronto Sydney Dublin Edinburgh Madrid Rome Tokyo Moscow Vienna Prague Lisbon Oslo Denver Dallas Seattle "
             "Phoenix Atlanta Detroit Cleveland Baltimore Manchester Liverpool Leeds Glasgow Cardiff Belfast Montreal Vancouver Melbourne Brisbane Perth "
             "Auckland Cairo Mumbai Delhi Seoul Beijing Shanghai Hamburg Munich Zurich Geneva Brussels Amsterdam Warsaw Budapest Athens").split()
    FIRST = ("James Mary John Patricia Robert Jennifer Michael Linda William Elizabeth David Barbara Richard Susan Joseph Jessica Thomas Sarah Charles Karen "
             "Daniel Nancy Matthew Lisa Anthony Betty Mark Margaret Donald Sandra Steven Ashley Paul Kimberly Andrew Emily Joshua Donna Kenneth Michelle "
             "Kevin Carol Brian Amanda George Melissa Edward Deborah Ronald Stephanie Timothy Rebecca Jason Laura Jeffrey Helen Ryan Sharon").split()
    LAST = ("Smith Johnson Williams Brown Jones Garcia Miller Davis Rodriguez Martinez Hernandez Lopez Gonzalez Wilson Anderson Thomas Taylor Moore Jackson "
            "Martin Lee Perez Thompson White Harris Sanchez Clark Ramirez Lewis Robinson Walker Young Allen King Wright Scott Torres Nguyen Hill Flores "
            "Green Adams Nelson Baker Hall Rivera Campbell Mitchell Carter Roberts Gomez Phillips Evans Turner Diaz Parker Cruz Edwards Collins Reyes").split()
    PROF = ("actor director footballer politician novelist composer painter singer architect journalist engineer physician economist playwright "
            "photographer conductor historian lawyer").split()
    NATION = "American British French German Canadian Australian English Irish Scottish Italian Spanish Japanese Russian Dutch Swedish".split()
    TIME = "monthly weekly annually quarterly daily".split()

    @classmethod
    def name(cls, r):
        return f"{r.choice(cls.FIRST)} {r.choice(cls.LAST)}"

    @classmethod
    def org(cls, r):
        return f"{r.choice(cls.PLACE)} {r.choice(cls.NOUN).capitalize()}"

    @classmethod
    def np(cls, r):
        return f"{r.choice(cls.ADJ)} {r.choice(cls.NOUN)}"

    @classmethod
    def question(cls, r):
        c, k = r.choice, r.randrange(8)
        if k == 0:
            return f"Which {c(cls.ADJ)} {c(cls.NOUN)} {c(cls.VERB)} by {cls.name(r)} came first, {cls.org(r)} or {cls.org(r)}?"
        if k == 1:
            return f"Who was born first, {cls.name(r)} or {cls.name(r)}?"
        if k == 2:
            return f"What nationality is the {c(cls.PROF)} who {c(cls.VERB)} the {cls.np(r)} in {c(cls.PLACE)}?"
        if k == 3:
            return f"The {cls.np(r)} {c(cls.VERB)} by {cls.name(r)} is located in which {c(cls.NATION)} city?"
        if k == 4:
            return f"Are {cls.name(r)} and {cls.name(r)} both {c(cls.NATION)} {c(cls.PROF)}s?"
        if k == 5:
            return f"What is the name of the {cls.np(r)} {c(cls.VERB)} in {c(cls.PLACE)} by the {c(cls.PROF)} {cls.name(r)}?"
        if k == 6:
            return f"Which {c(cls.NATION)} {c(cls.PROF)} was a member of the {cls.np(r)} {c(cls.VERB)} in {c(cls.PLACE)}?"
        return f"In what year was the {cls.np(r)} {c(cls.VERB)} by the {c(cls.NATION)} {c(cls.PROF)} {cls.name(r)} first {c(cls.VERB)}?"

    @classmethod
    def sentence(cls, r):
        c, k = r.choice, r.randrange(10)
        if k == 0:
            return f"{cls.name(r)} (born {r.randrange(1900, 1990)}) is an {c(cls.NATION)} {c(cls.PROF)} and a member of the {c(cls.ADJ)} {c(cls.NOUN)}."
        if k == 1:
            return f"{cls.org(r)} is a {c(cls.ADJ)} {c(cls.NOUN)} {c(cls.VERB)} {c(cls.TIME)} in {c(cls.PLACE)} since {r.randrange(1850, 2000)}."
        if k == 2:
            return f"The {c(cls.ADJ)} {c(cls.NOUN)} was {c(cls.VERB)} in {c(cls.PLACE)} and later {c(cls.VERB)} by {cls.name(r)}."
        if k == 3:
            return f"After the war, {cls.name(r)} moved to {c(cls.PLACE)}, where he worked as a {c(cls.PROF)} until {r.randrange(1950, 2015)}."
        if k == 4:
            return f"Its headquarters are in {c(cls.PLACE)}, and the {c(cls.ADJ)} {c(cls.NOUN)} {c(cls.VERB)} there has about {r.randrange(100, 9000)} members."
        if k == 5:
            return f"{cls.name(r)} starred alongside {cls.name(r)} in the {c(cls.NATION)} {c(cls.NOUN)} {c(cls.VERB)} by {cls.name(r)} in {r.randrange(1960, 2012)}."
        if k == 6:
            return f"Critics described the {c(cls.ADJ)} {c(cls.NOUN)} as a {c(cls.ADJ)} success, and it was {c(cls.VERB)} across {c(cls.PLACE)} and {c(cls.PLACE)}."
        if k == 7:
            return f"The {c(cls.ADJ)} {c(cls.NOUN)} {c(cls.VERB)} by {cls.org(r)} has been {c(cls.VERB)} several times since {r.randrange(1900, 2010)}."
        if k == 8:
            return f"{cls.name(r)}, a former {c(cls.PROF)}, served as chairman of the {c(cls.ADJ)} {c(cls.NOUN)} from {r.randrange(1930, 1990)} to {r.randrange(1990, 2015)}."
        return f"In {r.randrange(1800, 2010)} the {c(cls.NOUN)} was {c(cls.VERB)} by {cls.name(r)}, a {c(cls.NATION)} {c(cls.PROF)} who {c(cls.VERB)} it in {c(cls.PLACE)}."

    @classmethod
    def paragraph(cls, r, chars=1500, extra=()):
        sents, n = [], 0
        while n < chars - 120:
            s = cls.sentence(r)
            sents.append(s)
            n += len(s) + 1
        for e in extra:                                     # sentences that carry a known phrase collision, inserted at random places
            sents.insert(r.randrange(len(sents) + 1), e)
        return " ".join(sents)

    @classmethod
    def questions(cls, n, seed=1):
        r = random.Random(seed)
        return [cls.question(r) for _ in range(n)]

    @classmethod
    def rows(cls, n, seed=2, extra=()):
        r = random.Random(seed)
        return [cls.paragraph(r, 1500, extra) + "\n" + cls.paragraph(r, 1500) for _ in range(n)]


def brute_force_flags(rows, evals, n=13, win=40, min_shared=1, boilerplate_eval_df=10 ** 9):
    """Reference implementation of the ASCII rule with plain strings and no hashing (every eval text is one eval item): a row is flagged when
    it has >= min_shared distinct counted 13-grams - a gram with a non-ASCII character that is in an eval text, or an ASCII gram inside a
    window of `win` all-ASCII characters that sits verbatim in an eval text - whose eval document frequency is below the boilerplate limit,
    or when it contains a whole eval text of >= win normalised characters."""
    ev = [C.normalize_for_overlap(t) for t in evals]
    df, windows, whole = {}, set(), []
    for e in ev:
        for g in {e[i:i + n] for i in range(len(e) - n + 1)}:
            df[g] = df.get(g, 0) + 1
        windows.update(e[i:i + win] for i in range(len(e) - win + 1) if e[i:i + win].isascii())
        if len(e) >= win:
            whole.append(e)
    out = []
    for pos, t in enumerate(rows):
        r = C.normalize_for_overlap(t)
        counted = {r[i:i + n] for i in range(len(r) - n + 1) if r[i:i + n] in df and not r[i:i + n].isascii()}
        for s in range(len(r) - win + 1):
            if r[s:s + win] in windows:
                counted.update(r[j:j + n] for j in range(s, s + win - n + 1))
        if len([g for g in counted if df.get(g, 1) < boilerplate_eval_df]) >= min_shared or any(e in r for e in whole):
            out.append(pos)
    return out


class TestAsciiRunDecontaminate(unittest.TestCase):
    """Round 8: a 13-gram of pure ASCII flags a row only inside a run of >= 40 matching normalised characters of ONE eval text (28 consecutive
    grams); grams with a non-ASCII character keep the 13-character rule.  Measured on the real Colab run: 76,288 of 78,864 HotpotQA rows were
    dropped by 2-3 word English phrases ('smagazinepubl' = woman's magazine published, 'thmilitarypol' = th military police) that also occur in
    the validation QUESTIONS; no question was copied."""
    WOMAN_Q = "Which woman's magazine published an interview with the author of the novel Shadow Harbor?"
    POLICE_Q = "How many soldiers served in the 16th Military Police Brigade under the command of General Harlan Pike?"
    EXTRA = ["The magazine is a woman's magazine published monthly in London.", "The 8th Military Police Group was activated in Frankfurt after the war."]
    JA_EVAL = "吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。"
    JA_MIXED_EVAL = "次の文章を読んで、print(len(items)) の意味を説明しなさい。その後で、変数の意味を答えなさい。"
    FILLER = "Zebras graze quietly near the river at dawn while the herd moves slowly toward the northern hills."

    @classmethod
    def setUpClass(cls):
        cls.qs = Hotpot.questions(7000) + [cls.WOMAN_Q, cls.POLICE_Q]
        cls.sources = [("hotpot_validation", cls.qs), ("ja_eval", [cls.JA_EVAL, cls.JA_MIXED_EVAL])]
        cls.idx = C.build_eval_index(cls.sources)                                            # the rule since round 8
        cls.idx_old = C.build_eval_index(cls.sources, ascii_run_chars=13)                    # every ASCII gram flags by itself: the rule before
        cls.rows = Hotpot.rows(300, seed=7, extra=cls.EXTRA)
        dicts = [{"id": f"p{i}", "full_text": t} for i, t in enumerate(cls.rows)]
        t0 = time.time()
        cls.r_old = C.decontaminate(dicts, cls.idx_old)
        cls.t_old = time.time() - t0
        t0 = time.time()
        cls.r_new = C.decontaminate(dicts, cls.idx)
        cls.t_new = time.time() - t0

    def flagged(self, text, idx=None, **kw):
        r = C.decontaminate([{"id": "x", "full_text": text}], idx or self.idx, **kw)
        return r["rows_flagged"] == 1, r

    # ---- (1) paragraphs that share only common 2-3 word phrases with 7,000 questions are not flagged
    def test_english_paragraphs_sharing_only_common_phrases_are_not_flagged(self):
        qs_norm = {C.normalize_for_overlap(q) for q in self.qs}
        windows = {q[i:i + 40] for q in qs_norm for i in range(len(q) - 39)}
        for i, t in enumerate(self.rows):                                                    # the corpus really has no 40 verbatim characters in common with a question
            norm = C.normalize_for_overlap(t)
            self.assertFalse(any(norm[s:s + 40] in windows for s in range(len(norm) - 39)), i)
        row0 = C.normalize_for_overlap(self.rows[0])
        self.assertGreater(len(self.rows[0]), 2900)                                            # two paragraphs of ~1,500 characters
        for gram in ("smagazinepubl", "thmilitarypol"):                                      # the two collisions of the real run
            self.assertIn(gram, row0)
            self.assertEqual([m["source"] for m in self.idx.find_sources(gram)], ["hotpot_validation"])      # a validation QUESTION owns it
            self.assertIn(hash(gram), self.idx.hashes)
        self.assertEqual(self.r_old["rows_flagged"], len(self.rows))                         # before: every row is dropped by such phrases
        self.assertEqual(self.r_old["rows_flagged_whole_item"], 0)
        self.assertEqual(self.r_new["rows_flagged"], 0)                                      # after: none
        self.assertEqual(self.r_new["rows_flagged_ascii_run"], 0)
        self.assertEqual(self.r_new["rows_with_any_shared_shingle"], len(self.rows))         # they still collide on 13-grams: only the rule changed
        self.assertEqual(self.r_new["rows_not_flagged_ascii_short"], len(self.rows))
        self.assertEqual(self.r_new["rows_ascii_short_only"], len(self.rows))
        self.assertEqual(self.r_new["rows_flagged_legacy_13gram"], len(self.rows))
        self.assertEqual(self.r_new["rows_boilerplate_only"], 0)
        self.assertEqual(self.r_new["flagged_positions"], brute_force_flags(self.rows, self.qs + [self.JA_EVAL, self.JA_MIXED_EVAL]))
        self.assertTrue(self.r_new["ascii_short_top"] and self.r_new["ascii_short_examples"])
        self.assertEqual(self.r_new["ascii_short_examples"][0]["eval_matches"][0]["source"], "hotpot_validation")
        n = len(self.rows)
        print(f"\n[ascii-run] synthetic HotpotQA-like corpus ({n} rows of ~{sum(map(len, self.rows)) // n} characters, {len(self.qs):,} validation questions): "
              f"flagged before {self.r_old['rows_flagged']}/{n} ({self.r_old['rows_flagged'] / n:.1%}), after {self.r_new['rows_flagged']}/{n} "
              f"({self.r_new['rows_flagged'] / n:.1%}); decontaminate {self.t_old:.2f} s before, {self.t_new:.2f} s after", flush=True)

    # ---- (2) a verbatim English question of 45+ characters is flagged
    def test_a_verbatim_english_question_is_flagged_whatever_the_surroundings(self):
        qn = C.normalize_for_overlap(self.WOMAN_Q)
        self.assertGreaterEqual(len(qn), 45)
        para = Hotpot.paragraph(random.Random(5), 1500)
        mid = len(para) // 2
        for variant in (self.WOMAN_Q, self.WOMAN_Q.upper(), self.WOMAN_Q.replace(" ", "  ") + " 2024", self.WOMAN_Q.replace("Which", "Which 12")):
            ok, r = self.flagged(para[:mid] + "\n" + variant + "\n" + para[mid:])
            self.assertTrue(ok, variant)
            self.assertEqual(r["rows_flagged_whole_item"], 1)
            self.assertEqual(r["rows_flagged_ascii_run"], 1)
            self.assertEqual(r["examples"][0]["eval_matches"][0]["source"], "hotpot_validation")
        ok, r = self.flagged(para[:mid] + "\n" + self.POLICE_Q + "\n" + para[mid:])
        self.assertTrue(ok)
        # the old rule agreed on a verbatim copy: nothing that was a real copy is lost
        self.assertTrue(self.flagged(para[:mid] + self.WOMAN_Q + para[mid:], self.idx_old)[0])

    def test_the_run_must_reach_40_normalised_characters_and_must_come_from_one_eval_text(self):
        qn = C.normalize_for_overlap(self.WOMAN_Q)
        para = Hotpot.paragraph(random.Random(6), 1200)
        for length, want in ((41, True), (40, True), (39, False), (30, False)):
            piece = qn[8:8 + length]                                                         # part of the question, not the whole item
            ok, r = self.flagged(para + " ##### " + piece + " ##### " + para)
            self.assertEqual(ok, want, length)
            self.assertEqual(r["rows_flagged_whole_item"], 0)                                 # it is the run, not the whole-item rule
            self.assertEqual(r["rows_flagged_ascii_run"], int(want))
            if not want:
                self.assertTrue(self.flagged(para + " ##### " + piece + " ##### " + para, self.idx_old)[0], length)        # the old rule flagged it
        # two halves from two DIFFERENT eval items, 22 characters each: no single eval text holds 40 of them
        qa, qb = C.normalize_for_overlap(self.WOMAN_Q), C.normalize_for_overlap(self.POLICE_Q)
        ok, r = self.flagged(para + " ##### " + qa[8:30] + qb[4:26] + " ##### " + para)
        self.assertFalse(ok)
        ok, r = self.flagged("#####" + qa[8:30] + qb[4:26] + "#####" + qb[4:26] + qa[8:30] + "#####")
        self.assertFalse(ok)
        # a CHAIN: every 13-gram of the row is an eval gram and 60 of them follow each other, but they belong to five different eval texts of 24
        # characters (a, b, c and the two junction texts a|b and b|c): no eval text holds 40 of the row's characters, so there is no run
        a, b, c = C.normalize_for_overlap(self.WOMAN_Q)[10:34], C.normalize_for_overlap(self.POLICE_Q)[10:34], C.normalize_for_overlap(self.qs[5])[10:34]
        idx_chain = C.build_eval_index([("chain", [a, b, c, a[12:] + b[:12], b[12:] + c[:12]])])
        ok, r = self.flagged("##### " + a + b + c + " #####", idx_chain)
        self.assertFalse(ok)
        self.assertEqual(r["rows_with_any_shared_shingle"], 1)
        self.assertEqual(r["rows_not_flagged_ascii_short"], 1)
        self.assertTrue(self.flagged("##### " + a + b + c + " #####", C.build_eval_index([("chain", [a, b, c, a[12:] + b[:12], b[12:] + c[:12]])], ascii_run_chars=13))[0])
        # the same two pieces back to back, as they stand in ONE eval text, make a run
        self.assertTrue(self.flagged(para + " ##### " + qa[8:30] + qa[30:50] + " ##### " + para)[0])

    # ---- (3) a short English eval sentence copied verbatim is NOT flagged by grams (documented limit)
    def test_a_short_english_eval_sentence_copied_verbatim_is_not_flagged(self):
        sentence = "The old library closed early today."
        n = len(C.normalize_for_overlap(sentence))
        self.assertTrue(30 <= n < 40, n)                                                     # 13+ characters (it has 13-grams) but below the 40-character run
        idx_short = C.build_eval_index([("short", [sentence])])
        idx_short_old = C.build_eval_index([("short", [sentence])], ascii_run_chars=13)
        row = self.FILLER + " " + sentence + " " + self.FILLER
        ok, r = self.flagged(row, idx_short)
        self.assertFalse(ok)                                                                  # KNOWN LIMIT (README): below the 40-character run, and the whole-item rule needs 40 too
        self.assertEqual(r["rows_flagged_whole_item"], 0)
        self.assertEqual(r["rows_with_any_shared_shingle"], 1)
        self.assertTrue(self.flagged(row, idx_short_old)[0])                                 # the old rule did flag it
        # at the boundary: an eval text of 39 normalised characters is not flagged, one of 40 is (whole item and run)
        qn = C.normalize_for_overlap(self.WOMAN_Q)
        for length, want in ((39, False), (40, True)):
            idx_b = C.build_eval_index([("b", [qn[:length]])])
            ok, r = self.flagged(self.FILLER + " " + qn[:length] + " " + self.FILLER, idx_b)
            self.assertEqual(ok, want, length)
            self.assertEqual(r["rows_flagged_whole_item"], int(want))
        self.assertEqual(idx_b.stats()["ascii_run_chars"], 40)

    # ---- (4) Japanese is unchanged; mixed rows are judged gram by gram
    def test_a_japanese_13_character_copy_is_flagged_and_a_japanese_word_with_english_inside_too(self):
        jn = C.normalize_for_overlap(self.JA_EVAL)
        for length, want in ((13, True), (12, False)):
            piece = jn[6:6 + length]
            ok, r = self.flagged("全然関係のない文章です。ここに引用します：「" + piece + "」以上です。ありがとうございました。")
            self.assertEqual(ok, want, length)
            self.assertEqual(r["rows_flagged_ascii_run"], 0)
            self.assertEqual(ok, self.flagged("全然関係のない文章です。ここに引用します：「" + piece + "」以上です。ありがとうございました。", self.idx_old)[0], length)
        mn = C.normalize_for_overlap(self.JA_MIXED_EVAL)
        p = mn.index("print")
        piece = mn[p - 6:p + 7]                                                              # '章を読んで、print(l': Japanese around an English word
        self.assertEqual(len(piece), 13)
        self.assertFalse(piece.isascii())
        ok, r = self.flagged("無関係な文章です。「" + piece + "」と書いてあります。以上です。")
        self.assertTrue(ok)
        self.assertEqual(r["rows_flagged_ascii_run"], 0)
        english_only = mn[p:p + 17]                                                          # 'print(len(items))': 17 ASCII characters of the same eval text are not enough
        self.assertEqual(english_only, "print(len(items))")
        ok, r = self.flagged("Notes: " + english_only + " and nothing else follows here.")
        self.assertFalse(ok)
        self.assertEqual(r["rows_not_flagged_ascii_short"], 1)
        self.assertTrue(self.flagged("Notes: " + english_only + " and nothing else follows here.", self.idx_old)[0])

    def test_mixed_rows_japanese_scaffolding_plus_english_passages(self):
        para1, para2 = Hotpot.paragraph(random.Random(21), 1500, self.EXTRA), Hotpot.paragraph(random.Random(22), 1500)
        scaffold = "次の英文を読んで、質問に日本語で答えなさい。\n"
        row = scaffold + para1 + "\n" + para2
        ok, r = self.flagged(row)
        self.assertFalse(ok)                                                                  # the English passages collide on phrases and nothing else
        self.assertEqual(r["rows_flagged_ascii_run"], 0)
        self.assertEqual(r["rows_not_flagged_ascii_short"], 1)
        self.assertTrue(self.flagged(row, self.idx_old)[0])
        jn = C.normalize_for_overlap(self.JA_EVAL)
        ok, r = self.flagged(scaffold + para1 + "\n「" + jn[10:23] + "」\n" + para2)         # plus ONE copied Japanese 13-gram
        self.assertTrue(ok)
        self.assertEqual(r["rows_flagged_ascii_run"], 0)
        ok, r = self.flagged(scaffold + para1 + "\n" + self.WOMAN_Q + "\n" + para2)         # plus ONE copied English question
        self.assertTrue(ok)
        self.assertEqual(r["rows_flagged_ascii_run"], 1)

    def test_japanese_text_is_judged_exactly_as_before(self):
        items = [f"{head}{'ぁあぃいぅうぇえぉおかがきぎくぐけげこごさざしじすず'[i:i + 14]}という問題文です。条件は特にありません。最後まで読んでください。" for i, head in enumerate(["問題：", "設問：", "質問：", "課題："] * 3)]
        idx_new, idx_old = C.build_eval_index([("ja", items)]), C.build_eval_index([("ja", items)], ascii_run_chars=13)
        variants = []
        for t in items:
            variants += [t, t.replace("、", "，"), "前置きの文です。\n" + t + "\n後書きです。", t[len(t) // 2:], t[:12], t[:13], " ".join(t), t.replace("。", "。\n")]
        rows = [{"id": str(i), "full_text": v} for i, v in enumerate(variants)]
        r_new, r_old = C.decontaminate(rows, idx_new), C.decontaminate(rows, idx_old)
        self.assertEqual(r_new["flagged_positions"], r_old["flagged_positions"])
        self.assertGreater(r_new["rows_flagged"], len(items))
        self.assertEqual(r_new["rows_not_flagged_ascii_short"], 0)
        self.assertEqual(r_new["rows_flagged_ascii_run"], 0)
        self.assertEqual({k: r_new[k] for k in ("rows_boilerplate_only", "shared_count_hist", "rows_flagged_whole_item", "real_shingles_top", "boilerplate")},
                         {k: r_old[k] for k in ("rows_boilerplate_only", "shared_count_hist", "rows_flagged_whole_item", "real_shingles_top", "boilerplate")})

    # ---- (5) digits: the normaliser deletes them, so only letters count
    def test_digits_are_not_indexed_so_a_13_digit_run_never_flags_and_a_digit_only_text_is_invisible(self):
        self.assertEqual(C.normalize_for_overlap("4567890123456"), "")
        self.assertEqual(C.normalize_for_overlap("1234567890" * 6), "")
        ev_digits = "1234567890" * 6                                                         # 60 digits: NOT flaggable, before or after this rule: the normaliser (unchanged) deletes digits
        ev_text = "Invoice number 4567890123456 was paid in full when the goods arrived at the warehouse."
        idx = C.build_eval_index([("digits_only", [ev_digits]), ("invoice", [ev_text])])
        self.assertEqual(idx.per_source["digits_only"]["indexable_texts"], 0)
        row = "The customer quoted 4567890123456 twice and also 1234567890" * 3 + " " + ev_digits + " end."
        ok, r = self.flagged(row, idx)
        self.assertFalse(ok)
        self.assertEqual(r["rows_flagged"], 0)
        # letters between digits still count: 'ab12cd34...' is the letters 'abcd...' once the digits are gone, and 40+ of them are a run
        serial = " ".join(f"{a}{i % 10}{b}{(i * 3) % 10}" for i, (a, b) in enumerate(zip("abcdefghijklmnopqrstuvwx", "yzabcdefghijklmnopqrstuv")))
        self.assertGreaterEqual(len(C.normalize_for_overlap(serial)), 45)
        idx2 = C.build_eval_index([("serial", ["Reference code: " + serial + " (keep it safe)"])])
        self.assertTrue(self.flagged("Quote: " + serial.replace("0", "7") + " done.", idx2)[0])
        self.assertFalse(self.flagged("Quote: " + serial[:20] + " done.", idx2)[0])           # about 20 letters: below 40

    # ---- (6) the mechanics: brute-force oracle, determinism, speed
    def test_against_a_brute_force_reference_on_random_text(self):
        total = flagged = 0
        for seed in range(160):
            rng = random.Random(seed)
            alphabet = ("abcd", "あいう", "abcdabcdabcdあいう")[seed % 3]
            texts = lambda lo, hi: "".join(rng.choice(alphabet) for _ in range(rng.randint(lo, hi)))
            evals = [texts(10, 90) for _ in range(rng.randint(1, 6))]
            rows = []
            for _ in range(25):
                base = texts(5, 120)
                if rng.random() < 0.6:                                                       # splice a piece of an eval text in
                    e = rng.choice(evals)
                    a = rng.randint(0, len(e) - 1)
                    cut = rng.randint(0, len(base))
                    base = base[:cut] + e[a:rng.randint(a, len(e))] + base[cut:]
                rows.append(base)
            idx = C.build_eval_index([("s", evals)])
            for min_shared, limit in ((1, 10 ** 9), (2, 10 ** 9), (1, 2), (3, 3)):
                got = C.decontaminate(rows, idx, min_shared=min_shared, boilerplate_eval_df=limit)["flagged_positions"]
                want = brute_force_flags(rows, evals, min_shared=min_shared, boilerplate_eval_df=limit)
                self.assertEqual(got, want, (seed, min_shared, limit))
                total += len(rows)
                flagged += len(want)
        self.assertGreater(flagged, 1500)                                                    # the random cases do contain runs, copies and near misses
        self.assertLess(flagged, total * 0.5)

    def test_stats_are_consistent_and_the_run_length_is_a_parameter(self):
        r = self.r_new
        self.assertEqual(r["ascii_run_chars"], 40)
        self.assertEqual(self.idx.stats()["ascii_run_chars"], 40)
        self.assertGreater(self.idx.stats()["ascii_windows"], 100000)
        self.assertEqual(self.idx_old.stats()["ascii_windows"], 0)
        self.assertEqual(r["rows_with_any_shared_shingle"], r["rows_with_real_overlap"] + r["rows_ascii_short_only"] + r["rows_boilerplate_only"])
        self.assertEqual(r["rows_not_flagged_ascii_short"], len(r["ascii_short_positions"]))
        self.assertEqual(r["rows_flagged_ascii_run"], len(r["ascii_run_positions"]))
        self.assertLessEqual(r["rows_flagged_ascii_run"], r["rows_flagged"])
        self.assertTrue(set(r["ascii_run_positions"]) <= set(r["flagged_positions"]))
        self.assertTrue(set(r["ascii_short_positions"]).isdisjoint(r["flagged_positions"]))
        self.assertEqual(self.r_old["rows_flagged"], self.r_old["rows_flagged_legacy_13gram"])      # the legacy switch reproduces the old rule
        self.assertEqual(self.r_old["rows_not_flagged_ascii_short"], 0)
        # a shorter run flags a shorter copy: ascii_run_chars is a real parameter (26 characters = 14 consecutive grams)
        qn = C.normalize_for_overlap(self.WOMAN_Q)
        idx26 = C.build_eval_index([("q", self.qs)], ascii_run_chars=26)
        para = Hotpot.paragraph(random.Random(8), 600)
        self.assertTrue(self.flagged(para + " ##### " + qn[8:34] + " ##### " + para, idx26)[0])
        self.assertFalse(self.flagged(para + " ##### " + qn[8:33] + " ##### " + para, idx26)[0])
        self.assertEqual(C.EvalIndex(ascii_run_chars=5).ascii_run_chars, 13)                 # never shorter than one gram

    def test_the_result_does_not_depend_on_the_hash_seed(self):
        code = ("import sys, json, random, hashlib\n"
                "sys.path.insert(0, %r)\n"
                "import checks as C\n"
                "import test_checks as T\n"
                "qs = T.Hotpot.questions(800)\n"
                "ql = [q for q in qs if len(C.normalize_for_overlap(q)) >= 45]\n"
                "rows = T.Hotpot.rows(25, seed=3, extra=T.TestAsciiRunDecontaminate.EXTRA) + [q + ' ' + q for q in ql[:15]] + ['次の文章を読んで、条件を順に考えなさい。' + q for q in ql[15:20]]\n"
                "idx = C.build_eval_index([('v', qs + ['次の文章を読んで、条件を順に考えなさい。'])])\n"
                "r = C.decontaminate([{'id': str(i), 'full_text': t} for i, t in enumerate(rows)], idx)\n"
                "r.pop('eval_index')\n"
                "print(hashlib.sha256(json.dumps(r, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest(), r['rows_flagged'], r['rows_flagged_ascii_run'])\n") % HERE
        outs = []
        for seed in ("1", "2", "12345"):
            env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
            env["PYTHONHASHSEED"] = seed
            p = subprocess.run([sys.executable, "-B", "-X", "utf8", "-c", code], capture_output=True, text=True, timeout=120, env=env, cwd=HERE)
            self.assertEqual(p.returncode, 0, p.stderr[-500:])
            outs.append(p.stdout.strip())
        self.assertEqual(len(set(outs)), 1, outs)
        self.assertGreaterEqual(int(outs[0].split()[1]), 20)                                  # the 15 doubled questions and the 5 Japanese-scaffold rows are flagged
        dicts = [{"id": str(i), "full_text": t} for i, t in enumerate(self.rows[:40])]
        a, b = C.decontaminate(dicts, self.idx), C.decontaminate(dicts, self.idx)
        self.assertEqual(a, b)

    def test_speed_200k_short_rows_and_no_slowdown_against_the_old_rule(self):
        rng = random.Random(9)
        pool = [Hotpot.sentence(rng) for _ in range(4000)]

        def rows(n):                                                                         # ~120 characters each, generated lazily
            return ({"full_text": pool[i % 4000] + " " + pool[(i * 7 + 1) % 4000][:30]} for i in range(n))
        t0 = time.time()
        r = C.decontaminate(rows(200000), self.idx)
        dt = time.time() - t0
        self.assertEqual(r["n_rows"], 200000)
        self.assertLess(r["rows_flagged"], 200)                                              # (chance 40-character overlaps only; the old rule flags ~96%)
        self.assertGreater(r["rows_with_any_shared_shingle"], 150000)
        t0 = time.time()
        r_old = C.decontaminate(rows(50000), self.idx_old)
        dt_old = time.time() - t0
        t0 = time.time()
        r_new = C.decontaminate(rows(50000), self.idx)
        dt_new = time.time() - t0
        self.assertGreater(r_old["rows_flagged"], 45000)
        print(f"\n[ascii-run] speed: 200,000 synthetic rows of ~120 characters in {dt:.1f} s with the new rule (flagged {r['rows_flagged']}); on 50,000 of them "
              f"{dt_old:.1f} s with the old rule (flagged {r_old['rows_flagged']}), {dt_new:.1f} s with the new one (flagged {r_new['rows_flagged']}); "
              f"{len(self.rows)} HotpotQA-like rows of ~3,000 characters: {self.t_old:.2f} s old, {self.t_new:.2f} s new", flush=True)
        self.assertLess(dt, 120.0)
        self.assertLess(dt_new, 3.0 * dt_old + 1.0)                                          # the task allowed 3x of the old time
        self.assertLess(self.t_new, 3.0 * self.t_old + 0.5)


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


# ---- AST tripwire tables (round 5) --------------------------------------------------------------------------------------------------------------------
# (class, field) pairs of the ast module that carry a NAME or a STRING.  Each is dealt with by validate() (checks._VALIDATE_SRC); the second column says how.
AST_TEXT_FIELDS = {
    ("Name", "id"): "name rules; an imported module's alias may only be the base of an attribute",
    ("Attribute", "attr"): "attribute rules (underscore, DENIED_ATTRS, store, str.format) and the frozen list of the module",
    ("FunctionDef", "name"): "a bound name: the name rules",
    ("AsyncFunctionDef", "name"): "refused (async)",
    ("ClassDef", "name"): "refused (class definitions)",
    ("arg", "arg"): "a bound name: the name rules",
    ("keyword", "arg"): "a call keyword binds to the callee's PARAMETER name (it is not only a dict key): refused when it starts with an underscore "
                        "(Fraction(4, 8, _normalize=False) works on 3.10 / 3.11); any other name is harmless (dict(format=1), sorted(key=f))",
    ("alias", "name"): "import: the root must be allowed, the name of `from m import n` must be on the frozen list of m",
    ("alias", "asname"): "a bound name: the name rules",
    ("ImportFrom", "module"): "the module must be an allowed root (or collections.abc)",
    ("ExceptHandler", "name"): "a bound name: the name rules",
    ("MatchAs", "name"): "a bound name: the name rules",
    ("MatchStar", "name"): "a bound name: the name rules",
    ("MatchMapping", "rest"): "a bound name: the name rules",
    ("MatchClass", "kwd_attrs"): "refused: keyword attributes of class patterns are looked up with getattr",
    ("Global", "names"): "refused (global / nonlocal)",
    ("Nonlocal", "names"): "refused (global / nonlocal)",
    ("TypeVar", "name"): "a bound name (3.12 type parameter): the name rules",
    ("ParamSpec", "name"): "a bound name (3.12 type parameter): the name rules",
    ("TypeVarTuple", "name"): "a bound name (3.12 type parameter): the name rules",
    ("Constant", "value"): "str constants: format-string rules; inside an annotation: refused (string_annotation)",
    ("Constant", "kind"): "only 'u' (u-prefixed string literal)",
    ("TypeIgnore", "tag"): "only produced with type_comments=True; validate() parses without it",
}
AST_TEXT_FIELDS.update({(c, "type_comment"): "None: validate() parses without type_comments=True"
                        for c in ("Assign", "For", "AsyncFor", "With", "AsyncWith", "FunctionDef", "AsyncFunctionDef", "arg")})
# 3.8-era shims that ast.parse() never produces (a Constant is built instead); tolerated by name, asserted never to appear in a parse
AST_LEGACY = ("Num", "Str", "Bytes", "NameConstant", "Ellipsis", "Index", "ExtSlice", "Suite", "AugLoad", "AugStore", "Param")
# every other field: nodes, lists of nodes, ints, None / True / False.  FROZEN from the union of Python 3.10 and 3.12 (3.11 is a subset of 3.12),
# typed with the ast module's own validator.  A field that is missing here on a newer Python is reported by the tripwire, not guessed.
AST_NODE_FIELDS = {
    "AnnAssign": "target annotation value simple",
    "Assert": "test msg",
    "Assign": "targets value",
    "AsyncFor": "target iter body orelse",
    "AsyncFunctionDef": "args body decorator_list returns type_params",
    "AsyncWith": "items body",
    "Attribute": "value ctx",
    "AugAssign": "target op value",
    "Await": "value",
    "BinOp": "left op right",
    "BoolOp": "op values",
    "Call": "func args keywords",
    "ClassDef": "bases keywords body decorator_list type_params",
    "Compare": "left ops comparators",
    "Delete": "targets",
    "Dict": "keys values",
    "DictComp": "key value generators",
    "ExceptHandler": "type body",
    "Expr": "value",
    "Expression": "body",
    "For": "target iter body orelse",
    "FormattedValue": "value conversion format_spec",
    "FunctionDef": "args body decorator_list returns type_params",
    "FunctionType": "argtypes returns",
    "GeneratorExp": "elt generators",
    "If": "test body orelse",
    "IfExp": "test body orelse",
    "Import": "names",
    "ImportFrom": "names level",
    "Interactive": "body",
    "JoinedStr": "values",
    "Lambda": "args body",
    "List": "elts ctx",
    "ListComp": "elt generators",
    "Match": "subject cases",
    "MatchAs": "pattern",
    "MatchClass": "cls patterns kwd_patterns",
    "MatchMapping": "keys patterns",
    "MatchOr": "patterns",
    "MatchSequence": "patterns",
    "MatchSingleton": "value",
    "MatchValue": "value",
    "Module": "body type_ignores",
    "Name": "ctx",
    "NamedExpr": "target value",
    "Raise": "exc cause",
    "Return": "value",
    "Set": "elts",
    "SetComp": "elt generators",
    "Slice": "lower upper step",
    "Starred": "value ctx",
    "Subscript": "value slice ctx",
    "Try": "body handlers orelse finalbody",
    "TryStar": "body handlers orelse finalbody",
    "Tuple": "elts ctx",
    "TypeAlias": "name type_params value",
    "TypeIgnore": "lineno",
    "TypeVar": "bound",
    "UnaryOp": "op operand",
    "While": "test body orelse",
    "With": "items body",
    "Yield": "value",
    "YieldFrom": "value",
    "arg": "annotation",
    "arguments": "posonlyargs args vararg kwonlyargs kw_defaults kwarg defaults",
    "comprehension": "target iter ifs is_async",
    "keyword": "value",
    "match_case": "pattern guard body",
    "withitem": "context_expr optional_vars",
}


def ast_classes():
    """Every class below ast.AST (recursively) that belongs to the ast module, abstract bases and legacy shims included."""
    import ast
    seen, stack = [], [ast.AST]
    while stack:
        for sub in stack.pop().__subclasses__():
            if sub not in seen:
                seen.append(sub)
                stack.append(sub)
    return [c for c in seen if c.__module__ in ("ast", "_ast")]


def unclassified_ast_pairs(classes):
    """The (class name, field) pairs of `classes` that neither AST_TEXT_FIELDS nor AST_NODE_FIELDS knows (legacy shims are skipped by name)."""
    out = set()
    for c in classes:
        if c.__name__ in AST_LEGACY:
            continue
        for f in getattr(c, "_fields", ()):
            if (c.__name__, f) not in AST_TEXT_FIELDS and f not in AST_NODE_FIELDS.get(c.__name__, "").split():
                out.add((c.__name__, f))
    return sorted(out)


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

    def test_match_statement_patterns_cannot_read_attributes_the_static_rules_forbid(self):
        # review round 3 (blocker): ast.MatchClass carries attribute names as plain STRINGS (kwd_attrs) - `case int(__class__=c)` read
        # __class__, a second pattern __base__, a third __subclasses__ - and the validator only looked at ast.Attribute.  Every payload
        # below is a harmless stand-in (it only prints); static_check must refuse it before anything runs.
        if sys.version_info < (3, 10):
            self.skipTest("match statements need Python 3.10")
        payloads = {
            "match keyword pattern reads __class__": "match 0:\n    case int(__class__=c):\n        print(c)\n",
            "match chain reaches the subclass graph": ("match 0:\n    case int(__class__=c):\n        match c:\n            case c(__base__=b):\n"
                                                       "                match b:\n                    case b(__subclasses__=f):\n                        print(len(f()) > 10)\n"),
            "match keyword pattern reads a frame attribute": "g = (1 for _ in [1])\nmatch g:\n    case g(gi_frame=f):\n        print(f)\n",
            "match keyword pattern reads a read_ method": "match 0:\n    case int(read_csv=f):\n        print(f)\n",
            "match capture named like a dunder": "match 0:\n    case int() as __builtins__:\n        print(1)\n",
            "match star capture named like a dunder": "match [1]:\n    case [*__x]:\n        print(1)\n",
            "match mapping rest named like a dunder": "match {}:\n    case {**__r}:\n        print(1)\n",
            "except as a dunder name": "try:\n    pass\nexcept Exception as __e:\n    pass\n",
            "import alias that is a dunder name": "import math as __m\nprint(__m)\n",
        }
        for name, code in payloads.items():
            why = C.static_check(code)
            self.assertNotEqual(why, "", name)
            r = C.code_sandbox_run(code)
            self.assertFalse(r["ok"], name)
            self.assertEqual((r["category"], r["stdout"]), ("static", ""), (name, r))
        self.assertTrue(C.static_check(payloads["match keyword pattern reads __class__"]).startswith("match_class:"))
        self.assertTrue(any(n.startswith("match") for n, _ in C.SANDBOX_MUST_REJECT))           # the 3d self-test checks it on the Colab machine too
        # ordinary match statements still run
        ok, out, err = C.code_sandbox_check("x = (3, 4)\nmatch x:\n    case (a, b) if a < b:\n        print(a + b)\n    case _:\n        print(0)\n")
        self.assertEqual((ok, out), (True, "7\n"), err)
        # review round 4: a class pattern with keyword attributes is refused WHATEVER the name (this case used to run and print 3)
        ok, out, err = C.code_sandbox_check("match 3:\n    case int(real=r):\n        print(r)\n")
        self.assertEqual((ok, out), (False, ""), err)
        self.assertTrue(err.startswith("static:match_class:"), err)

    def test_class_patterns_are_deny_by_default_the_format_hole(self):
        # review round 4 (major): `case str(format=f)` is getattr(subject, 'format'); the str.format rule only existed in the ast.Attribute branch, so on a
        # string built at run time it handed out str.format and '{0._' + '_globals__}' walked to the function's __globals__ (including the child's SAFE
        # builtins).  Patching names one by one failed twice (__class__, then format): keyword attributes are now refused by default.
        if sys.version_info < (3, 10):
            self.skipTest("match statements need Python 3.10")
        probe = {
            "p6 kwd format on a built string": "s = '{0._' + '_class__}'\nmatch s:\n    case str(format=f):\n        print(f(1))\n",
            "p6 kwd format_map": "s = '{a._' + '_class__}'\nmatch s:\n    case str(format_map=f):\n        print(f({'a': 1}))\n",
            "p6 kwd format walk to __globals__": "def g():\n    pass\ns = '{0._' + '_globals__}'\nmatch s:\n    case str(format=f):\n        print(f(g)[:60])\n",
            "kwd format_map walk to __globals__": "def g():\n    pass\ns = '{a._' + '_globals__}'\nmatch s:\n    case str(format_map=f):\n        print(f({'a': g})[:60])\n",
            "benign-looking keyword (int.real)": "match 3:\n    case int(real=r):\n        print(r)\n",
            "benign-looking keyword on complex": "match 3j:\n    case complex(imag=i):\n        print(i)\n",
            "keyword in a nested alternative": "match [1]:\n    case [int(x)] | [str(format=x)]:\n        print(x)\n",
            "keyword next to a positional pattern": "match 3:\n    case int(n, real=r):\n        print(n)\n",
        }
        for name, code in probe.items():
            self.assertTrue(C.static_check(code).startswith("match_class:keyword attributes in class patterns are disabled"), (name, C.static_check(code)))
            r = C.code_sandbox_run(code)
            self.assertEqual((r["ok"], r["category"], r["stdout"]), (False, "static", ""), (name, r))
        self.assertIn("format", C.static_check(probe["p6 kwd format on a built string"]))
        # the class name itself: only the builtin types of the short allow-list (a positional pattern on them reads no attribute at all)
        for t in ("int", "float", "str", "bool", "list", "tuple", "dict", "set", "frozenset", "complex", "bytes"):
            self.assertEqual(C.static_check(f"match 1:\n    case {t}():\n        pass\n"), "", t)
            self.assertEqual(C.static_check(f"match 1:\n    case {t}(x):\n        pass\n"), "", t)
        self.assertEqual(sorted(C._VALIDATE_NS["MATCH_CLASS_OK"]),
                         sorted(("int", "float", "str", "bool", "list", "tuple", "dict", "set", "frozenset", "complex", "bytes")))
        for code, needle in {
                "from collections import Counter\nmatch Counter('a'):\n    case Counter():\n        pass\n": "only allowed on these builtin types",
                "from collections import Counter\nP = Counter\nmatch P('x'):\n    case P(f):\n        pass\n": "only allowed on these builtin types",
                "import collections\nmatch 1:\n    case collections.OrderedDict():\n        pass\n": "only allowed on these builtin types",
                "match 1:\n    case bytearray():\n        pass\n": "only allowed on these builtin types",       # not on the list
                "match 1:\n    case Foo(x):\n        pass\n": "only allowed on these builtin types",
                "from collections import Counter\nstr = Counter\nmatch str(1):\n    case str(a):\n        pass\n": "is rebound",
                "def int(x):\n    return x\nmatch 1:\n    case int():\n        pass\n": "is rebound",
                "match 1:\n    case int() as int:\n        pass\n": "is rebound",
                "import math as str\nmatch 1:\n    case str():\n        pass\n": "is rebound"}.items():
            why = C.static_check(code)
            self.assertTrue(why.startswith("match_class:"), (code, why))
            self.assertIn(needle, why)
        # what keeps running (the 3d self-test checks the first two on the Colab machine as well)
        for code, want in (("match (3, 4):\n    case (a, b) if a < b:\n        print(a + b)\n    case _:\n        print(0)\n", "7\n"),
                           ("match 5:\n    case int(n):\n        print(n + 1)\n", "6\n"),
                           ("match 'abc':\n    case str() | bytes():\n        print('text')\n", "text\n"),
                           ("match [1, 2]:\n    case list([a, b]):\n        print(a + b)\n", "3\n"),
                           ("match 2.5:\n    case int():\n        print('int')\n    case float(x):\n        print(x * 2)\n", "5.0\n")):
            ok, out, err = C.code_sandbox_check(code)
            self.assertEqual((ok, out), (True, want), (code, err))
        names = {n for n, _ in C.SANDBOX_MUST_REJECT}
        for needle in ("str.format on a built string", "str.format_map", "format walks to __globals__", "format_map walks to __globals__", "benign-looking name"):
            self.assertTrue(any(needle in n for n in names), needle)
        reject = dict(C.SANDBOX_MUST_REJECT)
        for code in list(probe.values())[:3]:
            self.assertIn(code, reject.values())                                       # the three payloads of the verifier's probe, verbatim
        for n, code in C.SANDBOX_MUST_REJECT:
            self.assertNotEqual(C.static_check(code), "", n)                           # every payload of the 3d list is refused by the static rules

    def test_names_that_become_attribute_lookups_or_are_only_bound(self):
        # every place where a NAME is carried as a plain string in the tree.  (1) looked up as an attribute at run time: refused for the same names an
        # attribute access is; (2) only bound: never an attribute read, so only the name rules apply (and `format` as a variable name is harmless).
        if sys.version_info < (3, 10):
            self.skipTest("match statements need Python 3.10")
        lookup = {                                                                    # {n} is read with getattr at run time
            "Attribute.attr": "(1).{n}\n", "MatchClass.kwd_attrs": "match 0:\n    case int({n}=c):\n        pass\n", "ImportFrom alias.name": "from math import {n}\n"}
        for what, tmpl in lookup.items():
            for n in ("format", "format_map", "__class__", "__globals__", "gi_frame", "read_csv", "os"):
                self.assertNotEqual(C.static_check(tmpl.format(n=n)), "", (what, n))
        bound = {
            "Name store": "{n} = 1\n", "FunctionDef.name": "def {n}():\n    pass\n", "arg.arg": "def f({n}):\n    return {n}\n",
            "Lambda arg": "f = lambda {n}: {n}\n", "ExceptHandler.name": "try:\n    pass\nexcept ValueError as {n}:\n    pass\n",
            "MatchAs.name": "match 0:\n    case int() as {n}:\n        pass\n", "MatchStar.name": "match [1]:\n    case [*{n}]:\n        pass\n",
            "MatchMapping.rest": "match {{}}:\n    case {{**{n}}}:\n        pass\n", "alias.asname": "import math as {n}\n",
            "keyword.arg": "print(dict({n}=1))\n"}
        for what, tmpl in bound.items():
            self.assertEqual(C.static_check(tmpl.format(n="format")), "", what)                       # a variable called format reads no attribute
            self.assertEqual(C.static_check(tmpl.format(n="format_map")), "", what)
            self.assertTrue(C.static_check(tmpl.format(n="__class__")).startswith("name:"), what)     # every one, call keywords included (round 6)
        for code in ("global {n}\n", "def f():\n    nonlocal {n}\n"):
            self.assertTrue(C.static_check(code.format(n="format")).startswith("global_nonlocal"), code)       # refused for every name
        if sys.version_info >= (3, 12):
            for tmpl in ("def f[{n}](x):\n    pass\n", "type {n} = int\n"):
                self.assertEqual(C.static_check(tmpl.format(n="format")), "", tmpl)
                self.assertNotEqual(C.static_check(tmpl.format(n="__class__")), "", tmpl)
        ok, out, err = C.code_sandbox_check("def f(format):\n    return format\nprint(f(1), dict(format_map=2))\n")
        self.assertEqual((ok, out), (True, "1 {'format_map': 2}\n"), err)

    def test_no_unreviewed_node_field_carries_a_name_as_a_string(self):
        # tripwire: parse every statement form and list the AST fields that hold a plain string (or a list of strings) - each one has to be a field that
        # validate() deals with (see the two tests above).  A Python version that adds another one must fail here, not pass silently.
        reviewed = {("Name", "id"), ("Attribute", "attr"), ("FunctionDef", "name"), ("AsyncFunctionDef", "name"), ("ClassDef", "name"),
                    ("arg", "arg"), ("keyword", "arg"), ("alias", "name"), ("alias", "asname"), ("ImportFrom", "module"),
                    ("ExceptHandler", "name"), ("MatchAs", "name"), ("MatchStar", "name"), ("MatchMapping", "rest"), ("MatchClass", "kwd_attrs"),
                    ("Global", "names"), ("Nonlocal", "names"), ("TypeVar", "name"), ("ParamSpec", "name"), ("TypeVarTuple", "name")}
        import ast
        corpus = ["import a.b as c\nfrom m import n as o, p\n", "def f(x, /, y=1, *args, z, **kw) -> int:\n    global g\n    return lambda u, *v, **w: u\n",
                  "async def af(x):\n    async with x as y:\n        await y\n    async for i in x:\n        pass\n", "class K(B, metaclass=M):\n    attr = 1\n",
                  "x.attr = 2\ndel x.attr\n", "try:\n    pass\nexcept (E1, E2) as err:\n    pass\n", "try:\n    pass\nexcept* E3 as err2:\n    pass\n",
                  "with open() as fh, other() as oh:\n    pass\n", "f(1, k=2, **{'a': 1})\n[a for a in b if a]\n{k: v for k, v in b}\nf\"{x!r:>{w}} {y=}\"\n",
                  "match v:\n    case 1 | 2 | 's' | None:\n        pass\n    case [a, b, *rest]:\n        pass\n    case {'k': v1, **others}:\n        pass\n"
                  "    case Cls(1, 2, kx=3) as whole:\n        pass\n    case mod.Cls(x):\n        pass\n    case _:\n        pass\n",
                  "def gen[T: int, *Ts, **Ps](x: T): pass\ntype Alias[T] = list[T]\nclass G[T]: pass\n", "x: int = 1\ndef h():\n    nonlocal zz\n"]
        found = set()
        for src in corpus:
            try:
                tree = ast.parse(src)
            except SyntaxError:                                                       # syntax of a newer Python (type parameters, match, except*)
                continue
            for node in ast.walk(tree):
                for fld in node._fields:
                    v = getattr(node, fld, None)
                    if isinstance(node, ast.Constant) and fld in ("value", "kind"):
                        continue
                    if isinstance(v, str) or (isinstance(v, list) and v and all(isinstance(e, str) for e in v)):
                        found.add((type(node).__name__, fld))
        self.assertTrue(found <= reviewed, sorted(found - reviewed))
        self.assertIn(("MatchClass", "kwd_attrs"), found if sys.version_info >= (3, 10) else {("MatchClass", "kwd_attrs")})

    def test_every_ast_node_class_and_field_of_this_python_is_classified(self):
        # round 5 (G4): the corpus test above only sees the node classes that its 12 snippets instantiate, so a class that no snippet builds (say a new
        # ast.Interpolation in a future Python) was invisible.  This one walks EVERY ast.AST subclass (recursively) and EVERY entry of its _fields and
        # classifies the pair against two explicit tables: AST_TEXT_FIELDS (the fields that carry a name or a string; each one is dealt with by validate())
        # and AST_NODE_FIELDS (fields that only hold nodes, lists of nodes, ints or None / True / False; frozen from the union of Python 3.10 and 3.12,
        # typed with the ast module's own validator).  An unclassified pair FAILS, so a newer Python cannot add a carrier silently.  A class or field that
        # exists only on some versions is simply absent from the walk on the others: tolerated, never a failure.
        import ast
        classes = ast_classes()
        names = {c.__name__ for c in classes}
        self.assertTrue({"Name", "Attribute", "Call", "FunctionDef", "ImportFrom", "Constant"} <= names, sorted(names)[:10])
        self.assertEqual(unclassified_ast_pairs(classes), [], "a node field this test has not classified: decide whether it carries a name or a string "
                                                              "(then validate() must deal with it) and add it to AST_TEXT_FIELDS or AST_NODE_FIELDS")
        # the walk really reaches classes that the corpus test never instantiates (the weakness that was reported)
        never_built = {"BinOp", "Compare", "Raise", "While", "Assert", "AugAssign", "Slice", "Starred", "IfExp", "Yield", "Set", "Lambda", "Subscript"}
        self.assertTrue(never_built <= names, sorted(never_built - names))
        # every text carrier of the table is a field of its class on this Python, or the class / field is absent from it (version difference): tolerated
        table_pairs = set(AST_TEXT_FIELDS) | {(c, f) for c, fs in AST_NODE_FIELDS.items() for f in fs.split()}
        live = {(c.__name__, f) for c in classes for f in getattr(c, "_fields", ())}
        absent = sorted(table_pairs - live)
        print(f"[ast tripwire] python {sys.version.split()[0]}: {len(classes)} classes, {len(live)} (class, field) pairs, all classified; "
              f"{len(absent)} table pairs absent on this version: {absent}", file=sys.stderr)
        # the legacy shims are never produced by ast.parse (so a field of theirs can never reach validate())
        corpus = [code for _, code in C.SANDBOX_MUST_REJECT] + [C.SANDBOX_RUNS_PROGRAM, "x = [i for i in range(3) if i]\nf'{x!r:>{3}}'\nlambda a, *b, **c: a\n"]
        found = set()
        for src in corpus:
            try:
                tree = ast.parse(src)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                self.assertNotIn(type(node).__name__, AST_LEGACY)
                for fld in node._fields:
                    v = getattr(node, fld, None)
                    if isinstance(v, str) or (isinstance(v, list) and v and all(isinstance(e, str) for e in v)):
                        found.add((type(node).__name__, fld))
        self.assertTrue(found <= set(AST_TEXT_FIELDS), sorted(found - set(AST_TEXT_FIELDS)))

    def test_the_ast_tripwire_flags_a_class_and_a_field_it_has_never_seen(self):
        # simulate what a newer Python does: a brand-new node class with a string field (3.14 t-strings: ast.Interpolation.str), and a new field on an
        # old class.  Neither is in any table, so both must be reported; once they are gone the walk is clean again.
        import ast
        import gc
        self.assertEqual(unclassified_ast_pairs(ast_classes()), [])
        fakes = [type("Interpolation", (ast.expr,), {"_fields": ("value", "str", "conversion", "format_spec"), "__module__": "ast"}),
                 type("Call", (ast.expr,), {"_fields": ("func", "args", "keywords", "secret_name"), "__module__": "ast"})]   # a second 'Call' with one extra field
        try:
            self.assertEqual(unclassified_ast_pairs(ast_classes()),
                             [("Call", "secret_name"), ("Interpolation", "conversion"), ("Interpolation", "format_spec"), ("Interpolation", "str"),
                              ("Interpolation", "value")])
        finally:
            fakes.clear()
            gc.collect()
        self.assertEqual(unclassified_ast_pairs(ast_classes()), [])

    def test_module_namespaces_hide_submodules_at_run_time(self):
        # round 5: the static layer now refuses `statistics.random` / `from fractions import Decimal` itself (deny by default); the RUNTIME layer alone
        # (static rules off in the child) must hide them too: see TestFrozenNames for the full matrix
        for code, needle in (("import statistics\nprint(statistics.random)", "static:module_attr:statistics.random"),
                             ("import functools\nprint(functools.partial)", None),
                             ("from fractions import Decimal\nprint(Decimal('1.5'))", "static:import_name:Decimal"),
                             ("import collections\nfrom collections.abc import Sequence\nprint(collections.abc.Sequence is Sequence)", None)):
            ok, out, err = C.code_sandbox_check(code)
            if needle:
                self.assertFalse(ok, code)
                self.assertTrue(err.startswith(needle), (code, err))
            else:
                self.assertTrue(ok, (code, err))
        self.assertEqual(run_runtime_only("import statistics\nprint(statistics.random)\n")["error"][:25], "exception:AttributeError:")
        self.assertEqual(run_runtime_only("from fractions import Decimal\nprint(Decimal('1.5'))\n")["error"][:22], "exception:ImportError:")
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
        # round 5: the new payloads, the combined 'runs' program and the two checks of the runtime layer on its own are part of the 3d self-test
        self.assertTrue(any(n.startswith("runs: the names that stay") for n in names), sorted(names))
        for n in ("runtime layer alone (static rules off): excluded names are not in the module namespaces",
                  "runtime layer alone (static rules off): listed names are there", "refused: functools.wraps", "refused: collections.namedtuple (eval of a template)",
                  "refused: string annotation (typing evaluates it as an expression)", "refused: a module that fractions merely imports"):
            self.assertIn(n, names)
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
            # review round 4: 3d runs while Drive is STILL mounted, so a line named 'Drive is unmounted while code runs' proved nothing and was counted as a
            # pass; it is now one information line that says what the child could see, and it is not a passed check
            self.assertFalse(any("Drive is unmounted while code runs" in c["name"] for c in lenient["checks"]))
            info = [c for c in lenient["checks"] if c.get("info")]
            self.assertEqual(len(info), 1)
            self.assertTrue(info[0]["name"].startswith("info: what the child could see of Drive at 3d time"))
            self.assertIn("Drive is unmounted only during 7a/7b, enforced by the notebook _detach", info[0]["name"])
            self.assertIn("could list it", info[0]["detail"])
            self.assertIn("could write it", info[0]["detail"])
            self.assertEqual((lenient["n_info"], lenient["n_passed"]), (1, len(lenient["checks"]) - 1))
            self.assertEqual(strict["n_info"], 0)
            self.assertNotIn("info", strict["checks"][0])
            probe.update({"uid": 0})                                                              # but a root child is never acceptable
            self.assertIn("child is not root", C.sandbox_selftest(drive_unmounted=True)["failed"])
            probe.update({"uid": 65534, "read_etc_shadow": "ok:900"})
            self.assertIn("child cannot read /etc/shadow", C.sandbox_selftest(drive_unmounted=True)["failed"])

    def test_an_unusable_run_as_fails_the_selftest_instead_of_raising_out_of_it(self):
        # review round 4: the failed 'run_as' check was recorded and the next line called code_sandbox_run with the same bad run_as, which raised
        # SandboxError out of sandbox_selftest.  On Windows run_as='nobody' triggers it for real; the mock makes it the same on every platform.
        from unittest import mock
        r = C.sandbox_selftest(run_as="nobody") if (os.name != "posix" or os.geteuid() != 0) else None
        if r is not None:
            self.assertFalse(r["ok"])
            self.assertEqual(r["failed"], ["run_as"])
        with mock.patch.object(C, "_resolve_run_as", side_effect=C.SandboxError("user 'nobody' does not exist on this machine")), \
                mock.patch.object(C, "code_sandbox_run", side_effect=AssertionError("must not run code with an unusable run_as")):
            for light in (True, False):
                r = C.sandbox_selftest(light=light, require_os_containment=True)
                self.assertFalse(r["ok"])
                self.assertEqual(r["failed"], ["run_as"])
                self.assertEqual([c["name"] for c in r["checks"]], ["run_as"])
                self.assertIn("does not exist", r["checks"][0]["detail"])
                self.assertEqual((r["backend"], r["run_as"], r["n_passed"], r["n_info"]), (C.SANDBOX_BACKEND, None, 0, 0))

    def test_sandbox_children_are_registered_and_killed_by_process_group(self):
        # review round 4: the children are session leaders (start_new_session), so killing the CONVERTER's process group never reached them.
        # POSIX-only: this Windows PC runs it with a fake os.killpg.
        from unittest import mock
        calls = []
        with mock.patch.object(C.os, "killpg", lambda pid, sig: calls.append((pid, sig)), create=True), \
                mock.patch.object(C.signal, "SIGKILL", 9, create=True), mock.patch.object(C, "_POSIX", True):
            self.assertEqual(C.kill_live_sandbox_children(), 0)                              # nothing running
            C._track_child(4242, True)
            C._track_child(4243, True)
            try:
                self.assertEqual(C.kill_live_sandbox_children(), 2)
                self.assertEqual(sorted(calls), [(4242, 9), (4243, 9)])
                C._track_child(4242, False)
                calls.clear()
                self.assertEqual(C.kill_live_sandbox_children(), 1)
                self.assertEqual(calls, [(4243, 9)])
            finally:
                C._track_child(4242, False)
                C._track_child(4243, False)
            self.assertEqual(C._LIVE_CHILDREN, {})

            def gone(pid, sig):
                raise ProcessLookupError(pid)
            C._track_child(4244, True)
            try:
                with mock.patch.object(C.os, "killpg", gone, create=True):
                    self.assertEqual(C.kill_live_sandbox_children(), 0)                      # already gone: not an error
            finally:
                C._track_child(4244, False)
        with mock.patch.object(C.os, "killpg", lambda pid, sig: calls.append("BAD"), create=True), mock.patch.object(C, "_POSIX", False):
            C._track_child(4245, True)
            try:
                self.assertEqual(C.kill_live_sandbox_children(), 0)                          # not POSIX: nothing is signalled
            finally:
                C._track_child(4245, False)
        self.assertNotIn("BAD", calls)
        # the signal handler of convert.py runs in the main thread, possibly while _track_child holds the lock: a plain Lock would deadlock there
        import threading
        self.assertIsInstance(C._LIVE_LOCK, type(threading.RLock()))
        # _spawn registers the child while it runs and always unregisters it
        seen = []
        with mock.patch.object(C, "_POSIX", True), mock.patch.object(C, "_track_child", lambda pid, running: seen.append((pid, running))):
            r = C._spawn("print('x')\n", [], timeout=20, flags=["-B", "-X", "utf8"], python=None, acct=None)
        self.assertEqual(r["out"].strip(), b"x")
        self.assertEqual([running for _, running in seen], [True, False])
        self.assertEqual(seen[0][0], seen[1][0])
        seen.clear()
        with mock.patch.object(C, "_POSIX", True), mock.patch.object(C, "_track_child", lambda pid, running: seen.append((pid, running))):
            r = C._spawn("while True:\n    pass\n", [], timeout=1.0, flags=["-B", "-X", "utf8"], python=None, acct=None)   # killed on timeout: still unregistered
        self.assertTrue(r["timed_out"])
        self.assertEqual([running for _, running in seen], [True, False])

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


VERSION_ADDED = {("math", "cbrt"): (3, 11), ("math", "exp2"): (3, 11), ("math", "sumprod"): (3, 12), ("itertools", "batched"): (3, 12),
                 ("collections.abc", "Buffer"): (3, 12)}                 # listed names that older Pythons lack (the child's namespace simply does not have them)
VERSION_REMOVED = {("collections.abc", "ByteString"): (3, 14)}           # listed names that newer Pythons dropped
OLD_HIDE = {"attrgetter", "methodcaller", "call"}                        # what the round-4 child hid by name
BY_STRING_HELPERS = {"attrgetter", "methodcaller", "call", "update_wrapper", "wraps", "getattr", "setattr", "delattr", "hasattr", "vars", "dir", "eval",
                     "exec", "compile", "namedtuple", "singledispatch", "singledispatchmethod", "import_module", "__import__"}
EXPLICITLY_EXCLUDED = {                                                  # the names the review of round 4 and the owner's brief name
    "functools": ("update_wrapper", "wraps", "singledispatch", "singledispatchmethod", "partialmethod", "cached_property", "total_ordering", "namedtuple"),
    "collections": ("namedtuple", "UserDict", "UserList", "UserString"),
    "operator": ("attrgetter", "methodcaller", "call"),
    "fractions": ("Decimal", "re", "sys", "numbers", "math", "operator", "functools"),
    "statistics": ("namedtuple", "random", "Counter", "Decimal", "Fraction", "groupby", "itemgetter", "sqrt", "LinearRegression"),
    "decimal": ("DecimalTuple",),
}
ROUND4_REPROS = {                        # the programs of the round-4 coverage review (harmless: static_check inputs, never run before the fix)
    "rA1 update_wrapper transplants __code__": ("import functools\nimport statistics\n\n\ndef payload():\n    return sys.version_info[0]\n\n\n"
                                                "functools.update_wrapper(statistics.mean, payload, assigned=('__code__',), updated=())\nprint(statistics.mean())\n",
                                                "module_attr:functools.update_wrapper", "exception:AttributeError"),
    "rA2 wraps and an aliased update_wrapper": ("import functools\nimport statistics\nfrom functools import update_wrapper as uw\n\n\ndef payload():\n"
                                                "    return sys.version_info[0]\n\n\nfunctools.wraps(payload, assigned=('__code__',), updated=())(statistics.mean)\n"
                                                "uw(statistics.fmean, payload, assigned=('__code__',), updated=())\nprint(statistics.mean(), statistics.fmean())\n",
                                                "import_name:update_wrapper", "exception:ImportError"),
    "rA3 attribute store by string": ("import functools\n\n\ndef f():\n    return 1\n\n\ndef g():\n    return 2\n\n\n"
                                      "functools.update_wrapper(f, g, assigned=('__code__', '__name__', '__defaults__'), updated=())\nprint(f())\n",
                                      "module_attr:functools.update_wrapper", "exception:AttributeError"),
    # (rB1 as the reviewer wrote it ends in print(sorted(c.__name__ ...)), which the underscore rule refused already; the last line is made name-free here so
    # that the program is accepted by a checker without the string-annotation rule, i.e. it really tests that rule)
    "rB1 singledispatch + string annotation": ("import functools\n\n\n@functools.singledispatch\ndef describe(x):\n    return 'other'\n\n\n"
                                               "def impl(x: \"(1).__class__\"):\n    return 'int'\n\n\ndescribe.register(impl)\n"
                                               "print(len(describe.registry))\n",
                                               "string_annotation:", "exception:AttributeError"),
    "rB3 singledispatchmethod + return annotation": ("import functools\n\n\ndef impl(x: \"(1).__class__\"):\n    pass\n\n\n"
                                                     "d = functools.singledispatchmethod(impl)\nd.register(impl)\n",
                                                     "string_annotation:", "exception:AttributeError"),
    "rC1 from-import of a module name": ("from fractions import re\nfrom statistics import random\nfrom fractions import numbers\nimport fractions\n"
                                         "print(fractions.re.compile('a'))\n", "import_name:re", "exception:ImportError"),
    "rC2 from-import of a name that does not exist": ("from math import this_name_does_not_exist\n", "import_name:this_name_does_not_exist", "exception:ImportError"),
}


def real_module(name):
    import importlib
    return importlib.import_module(name)


def old_public(name):
    """What the round-4 child handed out for a module: the public attributes that are not modules (attrgetter / methodcaller / call were hidden by name)."""
    import types
    return {k: v for k, v in vars(real_module(name)).items() if not k.startswith("_") and not isinstance(v, types.ModuleType) and k not in OLD_HIDE}


def readme_gaps(module_name, row, dropped, namespace):
    """(missing, leaked) of the names that the README table row `row` does not list.  `dropped` are the names this Python dropped for the module,
    `namespace` is vars(module).  `leaked` = a function / class that the module merely IMPORTED from another module (statistics.mul is operator.mul on
    Python 3.11 only): not API, and a different minor version leaks different ones, so a README written for 3.12 cannot be complete for every
    version; these are tolerated (and reported).  `missing` = everything else (a name the module defines itself, a constant such as tau or
    HAVE_THREADS, a class such as Counter that is not recognisable as an import): a failure."""
    import types
    missing, leaked = [], []
    for n in sorted(dropped):
        if "`" + n + "`" in row:
            continue
        v = namespace.get(n)
        home = getattr(v, "__module__", None)
        if isinstance(v, (types.FunctionType, types.BuiltinFunctionType, type)) and isinstance(home, str) and home.split(".")[0].lstrip("_") != module_name.split(".")[0]:
            leaked.append(n)
        else:
            missing.append(n)
    return missing, leaked


def import_probe(module, names):
    """A program for the child with the static rules OFF: lists every name of `names` that is reachable as module.name or by `from module import name`."""
    lines = ["found = []", f"import {module} as zz"]
    for n in names:
        lines.append(f"try:\n    zz.{n}\n    found.append('zz.{n}')\nexcept Exception:\n    pass")
        lines.append(f"try:\n    from {module} import {n}\n    found.append('from import {n}')\nexcept Exception:\n    pass")
    lines.append("print(found)")
    return "\n".join(lines) + "\n"


class TestFrozenNames(unittest.TestCase):
    """Round 5: deny by default.  Which names of the allowed modules are reachable is decided by ONE frozen list per module (checks.MODULE_NAMES),
    enforced by the static rules AND by the namespaces the child builds."""

    def test_the_lists_are_plain_frozen_tuples_and_both_layers_run_the_same_ones(self):
        self.assertEqual(sorted(C.MODULE_NAMES), sorted(list(C.EXECUTOR_ROOTS) + ["collections.abc"]))
        self.assertEqual(C.SUBMODULE_ATTRS, {"collections": ("abc",)})
        for m, names in C.MODULE_NAMES.items():
            self.assertIsInstance(names, tuple, m)
            self.assertEqual(len(names), len(set(names)), f"{m}: duplicate names")
            for n in names:
                self.assertTrue(n.isidentifier() and not n.startswith("_"), (m, n))
        self.assertEqual(C._VALIDATE_NS["MODULE_NAMES"], C.MODULE_NAMES)               # the validator ...
        self.assertEqual(C._VALIDATE_NS["SUBMODULE_ATTRS"], C.SUBMODULE_ATTRS)
        self.assertIn("MODULE_NAMES = {", C._CHILD_SRC)                                  # ... and the child run the same source text, tuples included
        self.assertIn(repr(C.FUNCTOOLS_NAMES), C._CHILD_SRC)
        self.assertEqual(C.FUNCTOOLS_NAMES, ("cache", "cmp_to_key", "lru_cache", "partial", "reduce"))
        self.assertEqual(C.COLLECTIONS_NAMES, ("ChainMap", "Counter", "OrderedDict", "defaultdict", "deque"))
        self.assertEqual(C.FRACTIONS_NAMES, ("Fraction",))
        for forbidden in ("attrgetter", "methodcaller", "call"):
            self.assertNotIn(forbidden, C.OPERATOR_NAMES)
        for n in ("Decimal", "Context", "getcontext", "localcontext", "InvalidOperation", "DivisionByZero", "Overflow"):
            self.assertIn(n, C.DECIMAL_NAMES)
        self.assertEqual(sum(1 for n in C.DECIMAL_NAMES if n.startswith("ROUND_")), 8)
        # a module without a frozen list is not importable, however it is asked for
        for kw in ({"allowed_modules": ["math", "random"]}, {"allow_third_party": True}):
            with self.assertRaises(ValueError):
                C.static_check("import math\n", **kw)
            with self.assertRaises(ValueError):
                C.code_sandbox_run("print(1)", run_as=None, **kw)

    def test_every_listed_name_exists_on_this_python_or_is_version_optional(self):
        missing = []
        for m, names in C.MODULE_NAMES.items():
            have = vars(real_module(m))
            missing += [(m, n) for n in names if n not in have]
        for key in missing:
            self.assertTrue(key in VERSION_ADDED and sys.version_info[:2] < VERSION_ADDED[key] or key in VERSION_REMOVED and sys.version_info[:2] >= VERSION_REMOVED[key],
                            f"{key} is listed but this Python ({sys.version.split()[0]}) does not have it and it is not a known version-optional name")
        for (m, n), ver in VERSION_ADDED.items():                                         # the table is honest about itself
            self.assertIn(n, C.MODULE_NAMES[m])
            if (m, n) not in missing:
                self.assertGreaterEqual(sys.version_info[:2], ver, (m, n))
        print(f"[frozen names] python {sys.version.split()[0]}: {sum(len(v) for v in C.MODULE_NAMES.values())} listed names, "
              f"{len(missing)} absent here (version-optional): {sorted(missing)}", file=sys.stderr)

    def test_no_listed_name_is_underscored_a_module_a_frame_a_helper_that_takes_names_or_from_outside_the_allowed_modules(self):
        import types
        home_ok = set(C.MODULE_NAMES) | {"_functools", "_operator", "_decimal", "_collections", "_collections_abc", "_statistics", "builtins"}
        self_modules = set()
        for m, names in C.MODULE_NAMES.items():
            have = vars(real_module(m))
            for n in names:
                self.assertFalse(n.startswith("_"), (m, n))
                self.assertNotIn(n, BY_STRING_HELPERS, (m, n))
                if n not in have:
                    continue                                                              # version-optional name this Python lacks
                v = have[n]
                self.assertNotIsInstance(v, (types.ModuleType, types.FrameType, types.CodeType, types.TracebackType, types.CellType, types.GeneratorType), (m, n))
                if isinstance(v, types.BuiltinFunctionType):
                    s = v.__self__
                    if isinstance(s, types.ModuleType):                                   # every C function of math has its module as __self__ (unavoidable; the
                        self_modules.add(s.__name__)                                      # name `__self__` is refused statically); it must be one of OUR modules
                        self.assertIn(s.__name__, home_ok | {"itertools", "cmath", "math"}, (m, n))
                    else:
                        self.assertIsNone(s, (m, n))
                if isinstance(v, (types.FunctionType, type, types.BuiltinFunctionType)):
                    self.assertIn(getattr(v, "__module__", None), home_ok, f"{m}.{n} comes from {getattr(v, '__module__', None)!r}")
        self.assertTrue(self_modules <= {"math", "cmath", "decimal", "itertools", "_functools", "_operator"}, self_modules)

    def test_nothing_is_exposed_that_the_old_child_did_not_expose(self):
        for m, names in C.MODULE_NAMES.items():
            old = old_public(m)
            for n in names:
                if n in vars(real_module(m)):
                    self.assertIn(n, old, f"{m}.{n} was not reachable before: the new list may only remove")

    def test_the_readme_table_lists_every_name_that_was_dropped(self):
        path = os.path.join(HERE, "README_JA.md")
        if not os.path.isfile(path):
            self.skipTest("README_JA.md is not next to the tests")
        rows = {}
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()
        for line in lines:
            if line.startswith("| ") and line.count("|") >= 4:
                cells = [c.strip() for c in line.strip().strip("|").split("|")]
                rows[cells[0]] = cells[1]
        for m in C.MODULE_NAMES:
            self.assertIn(m, rows, f"README_JA.md has no table row for {m}")
            dropped = set(old_public(m)) - set(C.MODULE_NAMES[m])                           # the yield loss the owner is shown must be complete
            missing, leaked = readme_gaps(m, rows[m], dropped, vars(real_module(m)))
            self.assertEqual(missing, [], f"{m}: dropped but README_JA.md does not say so")
            if leaked:                                                                      # only ever a leaked import of THIS minor version (round 6)
                print(f"[readme table] python {sys.version.split()[0]}: {m} leaks {leaked} (imports of another module; not in the table, tolerated)", file=sys.stderr)
        # the 3.11 leak that the round-5 review found is now in the table itself (so the tolerance above is not what makes 3.11 pass)
        self.assertIn("`mul`", rows["statistics"])

    def test_the_readme_gap_helper_tolerates_leaked_imports_only(self):
        import operator
        import statistics
        row = "`Counter` `tau`"
        # a function that statistics imported from operator, missing from the row: tolerated; a constant or a class of its own: a failure
        self.assertEqual(readme_gaps("statistics", row, {"Counter", "mul"}, {"mul": operator.mul, "Counter": statistics.NormalDist}), ([], ["mul"]))
        self.assertEqual(readme_gaps("statistics", row, {"tau", "newconst"}, {"tau": 6.28, "newconst": 1.0}), (["newconst"], []))
        self.assertEqual(readme_gaps("statistics", row, {"helper"}, {"helper": statistics.mean}), (["helper"], []))            # defined by statistics itself
        self.assertEqual(readme_gaps("statistics", "", {"Foo"}, {"Foo": statistics.NormalDist}), (["Foo"], []))
        self.assertEqual(readme_gaps("collections.abc", "", {"x"}, {"x": operator.add}), ([], ["x"]))                           # _operator is not collections
        self.assertEqual(readme_gaps("operator", "", {"y"}, {"y": operator.add}), (["y"], []))                                  # _operator is operator's own C part

    def test_every_excluded_name_is_refused_statically_and_absent_in_the_child_namespace(self):
        excluded = {m: sorted(set(old_public(m)) - set(C.MODULE_NAMES[m])) for m in C.MODULE_NAMES}
        for m, names in EXPLICITLY_EXCLUDED.items():
            for n in names:
                if n in old_public(m):                                                    # (call is not there on 3.10; re / sys etc. are modules: not public here)
                    self.assertIn(n, excluded[m], (m, n))
                self.assertTrue(C.static_check(f"from {m} import {n}\n").startswith("import_name:" + n), (m, n, C.static_check(f"from {m} import {n}\n")))
                self.assertNotEqual(C.static_check(f"import {m} as zz\nprint(zz.{n})\n"), "", (m, n))
        total = 0
        for m, names in excluded.items():
            for n in names:
                total += 1
                a = C.static_check(f"from {m} import {n}\n")
                self.assertTrue(a.startswith("import_name:" + n), (m, n, a))
                b = C.static_check(f"import {m} as zz\nprint(zz.{n})\n")
                self.assertTrue(b.startswith(("module_attr:" + m + "." + n, "attr:")), (m, n, b))
                if total % 25 == 0:                                                       # a sample through the full entry point as well
                    c = C.code_sandbox_run(f"import {m} as zz\nprint(zz.{n})\n", run_as=None)
                    self.assertEqual((c["ok"], c["category"], c["stdout"]), (False, "static", ""), (m, n, c))
            if names:                                                                     # the child namespace itself (static rules off): none of them is there
                res = run_runtime_only(import_probe(m, names))
                self.assertTrue(res["ok"], (m, res))
                self.assertEqual(res["stdout"].strip(), "[]", f"{m}: reachable in the child although excluded")
        self.assertGreaterEqual(total, 40)                                                # the matrix is not empty (44 names on 3.12, 40 on 3.10)
        # the harmless probe the owner asked for, in the form that works in this sandbox (hasattr is not one of its builtins, so try / except Exception)
        res = run_runtime_only("import functools\ntry:\n    functools.update_wrapper\n    print(True)\nexcept Exception:\n    print(False)\n")
        self.assertEqual((res["ok"], res["stdout"].strip()), (True, "False"))
        res = C._runtime_layer_run("import functools\ntry:\n    functools.update_wrapper\n    print(True)\nexcept Exception:\n    print(False)\n", run_as=None)
        self.assertEqual((res["ok"], res["stdout"].strip()), (True, "False"))
        res = C.code_sandbox_run("import functools\nprint(hasattr(functools, 'update_wrapper'))\n", run_as=None)
        self.assertTrue(res["error"].startswith("static:module_name:"), res)             # the module may not even be passed to a function
        res = run_runtime_only("print(hasattr(1, 'real'))\n")
        self.assertTrue(res["error"].startswith("exception:NameError"), res)             # and hasattr is not a name of the sandbox at all

    def test_every_listed_name_is_accepted_statically_and_reachable_at_run_time(self):
        for m, names in C.MODULE_NAMES.items():
            present = [n for n in names if n in vars(real_module(m))]
            for n in names:
                self.assertEqual(C.static_check(f"from {m} import {n}\n"), "", (m, n))
                self.assertEqual(C.static_check(f"import {m} as zz\nprint(zz.{n})\n"), "", (m, n))
            code = (f"from {m} import (" + ", ".join(present) + ")\n" + f"import {m} as zz\n" + "x = [" + ", ".join("zz." + n for n in present) + "]\n"
                    + "print(len(x))\n")
            ok, out, err = C.code_sandbox_check(code, timeout=20, run_as=None)
            self.assertEqual((ok, out.strip()), (True, str(len(present))), (m, err))
        # a name a newer Python adds and this one lacks: listed, so allowed statically, then an ordinary per-row ImportError / AttributeError here
        if sys.version_info < (3, 12):
            ok, out, err = C.code_sandbox_check("from math import sumprod\nprint(sumprod([1, 2], [3, 4]))\n", run_as=None)
            self.assertFalse(ok)
            self.assertTrue(err.startswith("exception:ImportError"), err)

    def test_ordinary_exercise_programs_keep_running(self):
        programs = (
            (C.SANDBOX_RUNS_PROGRAM, C.SANDBOX_RUNS_EXPECT + "\n"),
            ("import math\nprint(math.comb(10, 3), math.isqrt(99), round(math.log2(8)), math.prod([1, 2, 3]), math.gcd(12, 18), math.lcm(4, 6))\n", "120 9 3 6 6 12\n"),
            ("import cmath\nprint(cmath.sqrt(-4), round(abs(cmath.exp(1j * cmath.pi) + 1), 9))\n", "2j 0.0\n"),
            ("from decimal import Decimal, ROUND_HALF_UP, Context, getcontext, localcontext\nprint(Decimal('2.675').quantize(Decimal('0.01'), rounding=ROUND_HALF_UP), "
             "Context(prec=5).divide(Decimal(1), Decimal(7)))\n", "2.68 0.14286\n"),
            ("from fractions import Fraction\nprint(sum(Fraction(1, n * (n + 1)) for n in range(1, 10)), Fraction(3, 6).limit_denominator(10))\n", "9/10 1/2\n"),
            ("from statistics import NormalDist, median, stdev, StatisticsError\nprint(NormalDist(0, 1).cdf(0), median([3, 1, 2]), round(stdev([1, 2, 3, 4]), 4))\n", "0.5 2 1.291\n"),
            ("import statistics\nprint(statistics.mean([1, 2, 3, 4]), statistics.multimode('aabbc'), statistics.fmean([1, 2]), statistics.quantiles([1, 2, 3, 4], n=4))\n",
             "2.5 ['a', 'b'] 1.5 [1.25, 2.5, 3.75]\n"),
            ("import operator\nprint(sorted([(1, 'b'), (0, 'a')], key=operator.itemgetter(0)), operator.truediv(1, 4), operator.countOf([1, 1, 2], 1))\n",
             "[(0, 'a'), (1, 'b')] 0.25 2\n"),
            ("import itertools\nprint(list(itertools.accumulate([1, 2, 3])), len(list(itertools.permutations(range(4)))), [(k, len(list(g))) for k, g in itertools.groupby('aabbb')])\n",
             "[1, 3, 6] 24 [('a', 2), ('b', 3)]\n"),
            ("from collections import Counter, OrderedDict, ChainMap, deque\nd = deque(range(5))\nd.rotate(2)\nprint(list(d), Counter('mississippi').most_common(2), "
             "list(OrderedDict(a=1, b=2)), ChainMap({'a': 1}, {'a': 2, 'b': 3})['b'])\n", "[3, 4, 0, 1, 2] [('i', 4), ('s', 4)] ['a', 'b'] 3\n"),
            ("from collections.abc import Sequence, Mapping\nimport collections\nprint(isinstance([1], Sequence), isinstance({}, Mapping), collections.abc.Sequence is Sequence)\n",
             "True True True\n"),
            ("from functools import cmp_to_key, cache, reduce\n\n\n@cache\ndef fib(n):\n    return n if n < 2 else fib(n - 1) + fib(n - 2)\n\n\n"
             "print(fib(30), sorted([3, 1, 2], key=cmp_to_key(lambda a, b: b - a)), reduce(lambda a, b: a * b, range(1, 6)))\n", "832040 [3, 2, 1] 120\n"),
            ("import math as m, collections.abc as abc_\nprint(m.floor(2.5), isinstance([], abc_.Iterable))\n", "2 True\n"),
        )
        for code, want in programs:
            ok, out, err = C.code_sandbox_check(code, timeout=20, run_as=None)
            self.assertEqual((ok, out), (True, want), (code[:80], err))

    def test_the_round_4_repro_programs_are_stopped_by_each_layer_on_its_own(self):
        for name, (code, static_prefix, runtime_error) in ROUND4_REPROS.items():
            why = C.static_check(code)
            self.assertTrue(why.startswith(static_prefix), (name, why))                  # layer (b): static
            r = C.code_sandbox_run(code, run_as=None)
            self.assertEqual((r["ok"], r["category"], r["stdout"]), (False, "static", ""), (name, r))
            res = run_runtime_only(code)                                                  # layer (a): the namespace has no such name (static rules off)
            self.assertFalse(res["ok"], (name, res))
            self.assertTrue(res["error"].startswith(runtime_error), (name, res["error"]))
            self.assertEqual(res["stdout"], "", name)

    def test_string_annotations_are_refused_everywhere_an_annotation_can_be(self):
        bad = ('def f(x: "(1).real"):\n    pass\n', 'def f(x) -> "(1).real":\n    pass\n', 'z: "(1).real" = 3\n', 'def f(*a: "int"):\n    pass\n',
               'def f(**k: "int"):\n    pass\n', 'def f(*, x: "int"):\n    pass\n', 'def f(x, /, y: "int"):\n    pass\n', 'def f(x: list["int"]):\n    pass\n',
               'def f(x: f"{1}"):\n    pass\n', 'def f(x: int = 1, y: "str" = ""):\n    pass\n', 'def f(x):\n    def g(y: "int"):\n        pass\n')
        for code in bad:
            self.assertTrue(C.static_check(code).startswith("string_annotation:"), (code, C.static_check(code)))
            self.assertEqual(C.code_sandbox_run(code, run_as=None)["category"], "static", code)
        good = "def f(x: int, y: float = 1.0, *a: int, **k: float) -> int:\n    return x\n\nz: int = 3\nw: list = [1]\nprint(f(1), z, w, 'a' + 'b')\n"
        self.assertEqual(C.static_check(good), "")
        ok, out, err = C.code_sandbox_check(good, run_as=None)
        self.assertEqual((ok, out), (True, "1 3 [1] ab\n"), err)

    def test_module_alias_rules(self):
        ok_codes = ("import math as m\nprint(m.sqrt(4))\n", "import collections.abc\nprint(collections.abc.Sequence)\n",
                    "import collections\nprint(collections.abc.Sequence)\n", "import collections.abc as ca\nprint(ca.Mapping)\n",
                    "import math\nimport math\nprint(math.pi)\n", "import math\n\n\ndef f(x):\n    return math.sqrt(x)\n\n\nprint(f(4))\n",
                    "from math import sqrt as s\nprint(s(4))\n", "import functools\nprint(functools.partial)\n",
                    "import statistics\nprint(statistics.NormalDist.from_samples)\n", "import math\nprint(f'{math.pi:.3f}')\n",
                    "import math\nmatch 3.14:\n    case math.pi:\n        print(1)\n    case _:\n        print(0)\n")
        for code in ok_codes:
            self.assertEqual(C.static_check(code), "", code)
        rebound = ("import math\nmath = 3\n", "import math\ndef f(math):\n    return math\n", "import math as m\nimport cmath as m\n", "import math\nfor math in [1]:\n    pass\n",
                   "import math\ndef math():\n    pass\n", "from math import pi as math\nimport math\n", "import math\nprint([math for math in [1]])\n",
                   "import math\ntry:\n    pass\nexcept Exception as math:\n    pass\n", "import math\nmatch 1:\n    case int() as math:\n        pass\n")
        for code in rebound:
            self.assertTrue(C.static_check(code).startswith("module_alias:"), (code, C.static_check(code)))
        escaped = ("import math\nx = math\nprint(x.pi)\n", "import math\nprint(math)\n", "import collections\nx = collections.abc\nprint(x.Sequence)\n",
                   "import math\nprint((lambda m: m.pi)(math))\n", "import collections.abc as ca\nprint([ca])\n", "import functools\nx = functools\nprint(x.update_wrapper)\n")
        for code in escaped:
            self.assertTrue(C.static_check(code).startswith("module_name:"), (code, C.static_check(code)))
        for code, want in (("import collections\nprint(collections.namedtuple)\n", "module_attr:collections.namedtuple"),
                           ("import collections.abc as ca\nprint(ca.abc)\n", "module_attr:collections.abc.abc"),
                           ("import collections.abc\nprint(collections.abc.Foo)\n", "module_attr:collections.abc.Foo"),
                           ("import functools as ft\nprint(ft.wraps)\n", "module_attr:functools.wraps"),
                           ("import math\nprint(math.nope)\n", "module_attr:math.nope"),
                           # round 6: `from collections import abc` is the submodule (the child serves it as an attribute of collections) and is held to
                           # exactly the rules of `import collections.abc as abc`
                           ("from collections import abc\nprint(abc.update_wrapper)\n", "module_attr:collections.abc.update_wrapper"),
                           ("from collections import abc\nprint(abc.abc)\n", "module_attr:collections.abc.abc"),
                           ("from collections import abc as c\nprint(c.Foo)\n", "module_attr:collections.abc.Foo"),
                           ("from collections import abc\nx = abc\nprint(x.Sequence)\n", "module_name:"),
                           ("from collections import abc\nprint(abc)\n", "module_name:"),
                           ("from collections import abc\nabc = 1\n", "module_alias:"),
                           ("from collections import abc\ndef f(abc):\n    return abc\n", "module_alias:"),
                           ("import collections\nfrom collections import abc as collections\n", "module_alias:"),
                           ("from collections import abc\nprint(abc.__class__)\n", "attr:__class__"),
                           ("from collections import abc, namedtuple\n", "import_name:namedtuple"),
                           ("from collections import abc as __x\n", "name:__x"),
                           ("import operator\nprint(operator.attrgetter)\n", "attr:attrgetter")):
            self.assertTrue(C.static_check(code).startswith(want), (code, C.static_check(code)))

    def test_the_3d_payload_lists_cover_the_review_gaps(self):
        must = [code for _, code in C.SANDBOX_MUST_REJECT]
        for needle in ("from functools import update_wrapper", "from functools import wraps", "functools.update_wrapper", "from functools import singledispatch",
                       "from collections import namedtuple", 'def f(x: "(1).real"):', "from fractions import re"):
            self.assertTrue(any(needle in code for code in must), needle)
        for name, code in C.SANDBOX_MUST_REJECT:
            self.assertNotEqual(C.static_check(code), "", name)                           # every payload is refused STATICALLY on every Python
        for word in ("reduce", "partial", "lru_cache", "Counter", "deque", "defaultdict", "Fraction", "Decimal", "statistics.mean", "itertools.combinations",
                     "operator.add", "math.gcd", "cmath.sqrt"):
            self.assertIn(word.split(".")[-1], C.SANDBOX_RUNS_PROGRAM, word)             # the 'runs' program of the 3d self-test uses every one of them
        self.assertEqual(C.static_check(C.SANDBOX_RUNS_PROGRAM), "")
        ok, out, err = C.code_sandbox_check(C.SANDBOX_RUNS_PROGRAM, run_as=None)
        self.assertEqual((ok, out.strip()), (True, C.SANDBOX_RUNS_EXPECT), err)
        for name, probe in C.SANDBOX_RUNTIME_HIDDEN.items():                              # every runtime probe of the self-test names something the real module has
            self.assertTrue(set(probe) <= set(vars(real_module(name))), (name, set(probe) - set(vars(real_module(name)))))
            self.assertFalse(set(probe) & set(C.MODULE_NAMES[name]), name)                # ... and that is NOT on the list


def _attr_chain(n):
    return "import math\nx = math" + ".pi" * n + "\n"                       # n attribute links on a module name (a 1,100-link one is 3,321 bytes)


def _format_nest(n):
    return "print('" + "{:" * n + "{}" + "}" * n + "'.format(1))\n"         # n levels of nested format spec


def _nested_shapes():
    """Pathological programs of every shape that nests; each is a few KB.  A 1,000 deep nest of brackets is refused by the tokenizer itself (200 levels)."""
    return {
        "parentheses": "x = " + "(" * 1000 + "1" + ")" * 1000 + "\n",
        "lists": "x = " + "[" * 1000 + "]" * 1000 + "\n",
        "maps": "x = " + "{1: " * 1000 + "1" + "}" * 1000 + "\n",
        "calls inside calls": "x = " + "abs(" * 1000 + "1" + ")" * 1000 + "\n",
        "lambdas": "f = " + "lambda: " * 1000 + "1\n",
        "unary minus": "x = " + "-" * 1000 + "1\n",
        "not": "x = " + "not " * 1000 + "1\n",
        "binary operator chain": "x = " + "+".join(["1"] * 1000) + "\n",
        "subscript chain": "x = [0]" + "[0]" * 1000 + "\n",
        "call chain": "f = abs\nx = f" + "()" * 1000 + "\n",
        "conditional chain": "x = " + "1 if 1 else " * 1000 + "1\n",
        "attribute chain 1,100": _attr_chain(1100),
        "attribute chain 990 (3 KB)": _attr_chain(990),
        "nested format spec 1,100": _format_nest(1100),
        "elif ladder of 600": "x = 5\nif x == 0:\n    y = 0\n" + "".join(f"elif x == {i}:\n    y = {i}\n" for i in range(1, 600)),
    }


class TestCheckerResourceLimits(unittest.TestCase):
    """Round 6 (R1): ONE crafted row must never make the static layer raise or run for seconds.  convert.Checks._sandbox turns any exception out of the
    checker into SandboxUnavailable, which stops the whole converter run; a RecursionError from a long attribute chain did exactly that when the converter
    was called from a deep stack (test_convert.TestCoding.test_a_pathological_python_row_is_one_reject_and_not_the_end_of_the_run)."""

    def setUp(self):
        import time
        self.clock = time.perf_counter

    def unlimited_validate(self):
        """The validator text with the tree-depth limit switched off: what the later passes (module_of, the walk) can take when the limit does not protect them."""
        src = C._VALIDATE_SRC
        patched = src.replace("MAX_AST_DEPTH = 200", "MAX_AST_DEPTH = 10 ** 9")
        self.assertNotEqual(patched, src, "the depth-limit anchor moved")
        ns = {}
        exec(compile(patched, "<validate-without-the-depth-limit>", "exec"), ns)
        return ns

    def test_the_limits_are_what_the_readme_says(self):
        self.assertEqual((C._VALIDATE_NS["MAX_AST_DEPTH"], C._VALIDATE_NS["MAX_FORMAT_NEST"]), (200, 8))

    def test_attribute_chains_of_100_500_990_and_1100_links(self):
        for n, want in ((100, ""), (190, ""), (500, "too_deep:"), (990, "too_deep:"), (1100, "too_deep:")):
            t0 = self.clock()
            r = C.static_check(_attr_chain(n))                                  # must RETURN (the old code raised RecursionError at 1,100)
            dt = self.clock() - t0
            self.assertEqual(r == "", want == "", (n, r))
            self.assertTrue(r.startswith(want), (n, r))
            self.assertLess(dt, 1.0, (n, dt))
        self.assertLess(len(_attr_chain(1100)), 4096)

    def test_a_3_kb_row_of_a_long_chain_is_fast(self):
        code = _attr_chain(990)
        self.assertLess(len(code), 3100)
        t0 = self.clock()
        for _ in range(20):
            C.static_check(code)
        self.assertLess((self.clock() - t0) / 20, 0.25)                         # was about 0.4 s per call (quadratic), seconds on a loaded machine

    def test_nested_format_specs(self):
        for n, ok in ((1, True), (2, True), (5, True), (8, True), (9, False), (10, False), (50, False), (1100, False)):
            r = C.static_check(_format_nest(n))
            if ok:
                self.assertEqual(r, "", n)
            else:
                self.assertTrue(r.startswith("format_string:nested too deep"), (n, r))
        # a spec that is accepted statically but too deep for str.format itself (2 levels) is an ordinary row failure at run time, not a hole
        c = C.code_sandbox_run(_format_nest(5), run_as=None)
        self.assertEqual((c["ok"], c["category"]), (False, "exception"), c)                 # (IndexError here: .format(1) has one argument; ValueError with more)
        # the check recurses at most MAX_FORMAT_NEST + 1 levels, whatever the input
        with self.assertRaises(C._VALIDATE_NS["StaticReject"]):
            C._VALIDATE_NS["_check_format"]("{:" * 100000 + "{}" + "}" * 100000)

    def test_every_shape_of_nesting_is_a_reject_with_a_category_and_never_an_exception(self):
        for name, code in _nested_shapes().items():
            t0 = self.clock()
            r = C.static_check(code)
            self.assertLess(self.clock() - t0, 1.0, name)
            self.assertTrue(r.startswith(("too_deep:", "syntax:", "format_string:nested too deep")), (name, r))
            res = C.code_sandbox_run(code, run_as=None)                          # the public entry point: a dict, category 'static', no child is started
            self.assertEqual((res["ok"], res["category"]), (False, "static"), (name, res))
            self.assertTrue(res["error"].startswith("static:"), (name, res["error"]))
            self.assertEqual(C.code_sandbox_check(code, run_as=None)[0], False, name)
        # the two named in the review get the categories that name them
        self.assertTrue(C.static_check(_attr_chain(1100)).startswith("too_deep:"))
        self.assertTrue(C.static_check(_nested_shapes()["lambdas"]).startswith(("too_deep:", "syntax:")))

    def test_the_depth_limit_does_not_touch_ordinary_programs(self):
        ladder = "x = 5\nif x == 0:\n    y = 0\n" + "".join(f"elif x == {i}:\n    y = {i}\n" for i in range(1, 100))
        for code in (ladder, "print(" + " + ".join(["1"] * 150) + ")\n", "".join(" " * i + "if 1:\n" for i in range(90)) + " " * 90 + "pass\n",
                     C.SANDBOX_RUNS_PROGRAM, "x = " + "(" * 150 + "1" + ")" * 150 + "\n"):
            self.assertEqual(C.static_check(code), "", code[:60])
        c = C.code_sandbox_run(ladder + "print(y)\n", run_as=None)
        self.assertEqual((c["ok"], c["stdout"]), (True, "5\n"), c)
        for code in ("x = 5\nif x == 0:\n    y = 0\n" + "".join(f"elif x == {i}:\n    y = {i}\n" for i in range(1, 400)), "print(" + " + ".join(["1"] * 400) + ")\n"):
            self.assertTrue(C.static_check(code).startswith("too_deep:"), code[:60])

    def test_module_of_is_iterative_and_linear_even_without_the_depth_limit(self):
        ns = self.unlimited_validate()
        cfg = C._profile_cfg("executor", None, False)
        # a chain on a module alias, 1,100 links: the old recursive _module_of raised RecursionError here; and a 990-link one took 0.4 s (quadratic)
        for n in (990, 1100):
            t0 = self.clock()
            ns["validate"](_attr_chain(n), cfg)                                  # accepted: only the first link is a module attribute
            self.assertLess(self.clock() - t0, 0.5, n)
        # through the submodule path (collections -> collections.abc -> ...), where the chain really is resolved link by link
        ns["validate"]("import collections\nx = collections.abc.Sequence" + ".x" * 1100 + "\n", cfg)
        with self.assertRaises(ns["StaticReject"]) as cm:
            ns["validate"]("import collections\nx = collections" + ".abc" * 1100 + "\n", cfg)
        self.assertTrue(str(cm.exception).startswith("module_attr|collections.abc.abc"), str(cm.exception))
        # the answer did not change: same verdicts as the limited validator on shallow input
        for code in ("import math\nprint(math.pi)\n", "import math\nprint(math)\n", "import collections\nprint(collections.abc.Sequence)\n",
                     "import collections.abc as ca\nprint(ca.Foo)\n", "from collections import abc\nprint(abc.Mapping)\n"):
            try:
                ns["validate"](code, cfg)
                a = ""
            except ns["StaticReject"] as e:
                a = str(e).replace("|", ":", 1)
            self.assertEqual(a, C.static_check(code), code)

    def test_the_format_check_and_the_tree_measure_do_not_recurse_per_level(self):
        ns = self.unlimited_validate()
        deep = ns["ast"].parse(_attr_chain(1100))
        self.assertTrue(ns["_ast_too_deep"](deep, 200))
        self.assertFalse(ns["_ast_too_deep"](deep, 5000))                         # measured iteratively: a tree deeper than the recursion limit is no problem
        self.assertFalse(ns["_ast_too_deep"](ns["ast"].parse("x = 1\n"), 200))
        self.assertTrue(ns["_ast_too_deep"](ns["ast"].parse("x = 1\n"), 3))        # Module > Assign > Name > Store is 4 levels

    def test_anything_that_runs_out_of_stack_or_memory_is_a_reject(self):
        import ast
        from unittest import mock
        for exc in (RecursionError("maximum recursion depth exceeded"), MemoryError()):
            with mock.patch.object(ast, "parse", side_effect=exc):                        # (e) the parser itself
                self.assertTrue(C.static_check("print(1)\n").startswith("too_deep:"), exc)
            with mock.patch.dict(C._VALIDATE_NS, {"_ast_too_deep": mock.Mock(side_effect=exc)}):   # (c) inside the checker, after the parse
                self.assertTrue(C.static_check("print(1)\n").startswith("too_deep:"), exc)
            with mock.patch.dict(C._VALIDATE_NS, {"validate": mock.Mock(side_effect=exc)}):         # the second net in static_check itself
                self.assertTrue(C.static_check("print(1)\n").startswith("too_deep:"), exc)
        for exc, want in ((ValueError("source code string cannot contain null bytes"), "syntax:ValueError"), (OverflowError("x"), "syntax:OverflowError")):
            with mock.patch.object(ast, "parse", side_effect=exc):
                self.assertEqual(C.static_check("print(1)\n"), want)
        # a lone surrogate cannot be encoded: ast.parse raises UnicodeEncodeError (a ValueError); still a reject, and the entry point still returns a dict
        lone = "x = '" + chr(0xD800) + "'\n"
        self.assertNotEqual(C.static_check(lone), "")
        self.assertEqual(C.code_sandbox_run(lone, run_as=None)["category"], "static")

    def test_a_real_stack_overflow_inside_the_checker_is_a_reject(self):
        # no mock: use up the stack and call the checker with a few frames left.  Whichever frame the overflow lands in (the checker's own, validate(),
        # the parse, the walk), static_check must RETURN; some depth must land inside it, and the answer is either 'passed' or 'too_deep'.
        code = "import math\nx = [abs(i) for i in range(3)]\nprint(math.pi)\n"
        frame, depth_now = sys._getframe(), 0
        while frame is not None:
            depth_now, frame = depth_now + 1, frame.f_back
        limit = sys.getrecursionlimit()

        def deeper(n):
            return C.static_check(code) if n <= 0 else deeper(n - 1)
        seen = set()
        for headroom in range(1, 90):
            try:
                seen.add(deeper(limit - depth_now - headroom)[:9])
            except RecursionError:
                pass                                                              # the overflow landed in this test's own frames: not the checker's business
        self.assertTrue(seen <= {"", "too_deep:"}, seen)
        self.assertIn("too_deep:", seen)
        self.assertIn("", seen)

    def test_the_real_child_gives_the_same_reject_as_the_parent(self):
        import time
        cfg = C._profile_cfg("executor", None, False)
        for code, want in ((_attr_chain(1100), "static:too_deep:"), (_format_nest(1100), "static:format_string:nested too deep")):
            r = C._run_in_child(code, dict(cfg), 20.0, child_src=C._CHILD_SRC, max_output_chars=1000, mem_mb=None, python=None, acct=None,
                                allow_third_party=False, t0=time.time())              # bypasses the parent's static_check: the CHILD's validate() answers
            self.assertEqual((r["ok"], r["category"]), (False, "static"), r)
            self.assertTrue(r["error"].startswith(want), r)

    def test_the_child_text_carries_the_same_validator_and_the_runtime_only_variant_still_finds_its_anchor(self):
        self.assertTrue(C._CHILD_SRC.startswith(C._VALIDATE_SRC))
        self.assertIn("tree = validate(code, cfg)", C._CHILD_SRC)
        self.assertNotIn("tree = validate(code, cfg)", C._CHILD_SRC_STATIC_OFF)


class TestCallKeywordNames(unittest.TestCase):
    """Round 6 (R2): a call keyword binds to the callee's parameter name, private ones included."""

    def test_a_keyword_that_starts_with_an_underscore_is_refused(self):
        for code in ("from fractions import Fraction\nprint(Fraction(4, 8, _normalize=False))\n", "print(dict(_x=1))\n", "print(dict(__x=1))\n", "print(dict(_=1))\n",
                     "print(dict(__class__=1))\n", "f = lambda **k: k\nprint(f(_a=1))\n", "import math\nprint(math.isclose(1, 1, _rel=1))\n",
                     "print(sorted([2, 1], _key=abs))\n"):
            r = C.static_check(code)
            self.assertTrue(r.startswith("name:_"), (code, r))
        c = C.code_sandbox_run("from fractions import Fraction\nprint(Fraction(4, 8, _normalize=False))\n", run_as=None)
        self.assertEqual((c["ok"], c["category"], c["error"], c["stdout"]), (False, "static", "static:name:_normalize", ""), c)

    def test_the_private_keyword_really_was_reachable_on_the_old_pythons(self):
        # (information, not a guard) why the rule exists: Fraction.__new__ has a keyword-only _normalize on 3.10 / 3.11
        from fractions import Fraction
        try:
            a = Fraction(4, 8, _normalize=False)
        except TypeError:
            self.skipTest("this Python's Fraction has no _normalize parameter (3.12+)")
        self.assertEqual((a.numerator, a.denominator), (4, 8))

    def test_ordinary_keywords_still_work(self):
        program = (
            "import math, statistics, itertools\n"
            "from decimal import Decimal, Context, ROUND_HALF_UP\n"
            "from fractions import Fraction\n"
            "from functools import lru_cache, partial\n"
            "from collections import Counter, deque\n"
            "print(sorted([3, 1, 2], key=lambda v: -v), sorted([3, 1, 2], reverse=True))\n"
            "print(1, 2, sep='-', end='!')\n"
            "print()\n"
            "print(round(2.567, ndigits=1), max([1, 5, 3], key=lambda v: -v), min([], default=0), sum([1, 2], start=10), int('11', base=2))\n"
            "print(Context(prec=3, rounding=ROUND_HALF_UP).create_decimal('1.2355'), Fraction(numerator=1, denominator=3))\n"
            "print(statistics.quantiles([1, 2, 3, 4, 5, 6, 7, 8], n=4), math.isclose(0.1 + 0.2, 0.3, rel_tol=1e-9))\n"
            "print(list(itertools.product(range(2), repeat=2)), Counter(a=2, b=1).most_common(1), list(deque(range(5), maxlen=2)), dict(a=1, b=2))\n"
            "@lru_cache(maxsize=None)\n"
            "def sq(n):\n"
            "    return n * n\n"
            "print(sq(4), partial(int, base=2)('101'), dict(format=1, type=2))\n")
        self.assertEqual(C.static_check(program), "")
        c = C.code_sandbox_run(program, run_as=None)
        self.assertTrue(c["ok"], c)
        self.assertEqual(c["stdout"].splitlines(), ["[3, 2, 1] [3, 2, 1]", "1-2!", "2.6 1 0 13 3", "1.24 1/3", "[2.25, 4.5, 6.75] True",
                                                    "[(0, 0), (0, 1), (1, 0), (1, 1)] [('a', 2)] [3, 4] {'a': 1, 'b': 2}", "16 5 {'format': 1, 'type': 2}"], c["stdout"])

    def test_the_known_limit_a_mapping_unpacked_into_a_call_is_not_seen(self):
        # NOT a guard that the hole is open: it pins what README_JA.md says.  The names inside **{...} are dict keys built at run time; the static layer
        # cannot read them, so `Fraction(4, 8, **{'_normalize': False})` is not refused by it (the runtime namespaces and the OS are what hold).  If the
        # rule is ever extended (for instance by refusing `**` in calls) this test must change together with the README.
        self.assertEqual(C.static_check("from fractions import Fraction\nprint(Fraction(4, 8, **{'_normalize': False}))\n"), "")


class TestFromCollectionsImportAbc(unittest.TestCase):
    """Round 6 (R3): `from collections import abc` is `import collections.abc as abc`: the child's `collections` namespace already carries `.abc`."""

    def test_it_is_allowed_and_works(self):
        for code in ("from collections import abc\nprint(isinstance([1], abc.Iterable), isinstance({}, abc.Mapping), issubclass(list, abc.Sequence))\n",
                     "from collections import abc as c\nprint(isinstance([1], c.Iterable), isinstance({}, c.Mapping), issubclass(list, c.Sequence))\n",
                     "import collections\nfrom collections import abc\nprint(isinstance([1], abc.Iterable), isinstance({}, collections.abc.Mapping), issubclass(list, abc.Sequence))\n"):
            self.assertEqual(C.static_check(code), "", code)
            c = C.code_sandbox_run(code, run_as=None)
            self.assertEqual((c["ok"], c["stdout"]), (True, "True True True\n"), c)
        r = run_runtime_only("from collections import abc\nprint(abc.Iterable is not None)\n")           # the child alone (static rules off) serves it
        self.assertEqual((r["ok"], r["stdout"]), (True, "True\n"), r)

    def test_the_other_names_of_collections_stay_refused_and_the_module_rules_still_apply(self):
        for n in ("namedtuple", "UserDict", "UserList", "UserString", "abcd", "ChainMapp"):
            self.assertTrue(C.static_check(f"from collections import {n}\n").startswith("import_name:" + n), n)
        self.assertTrue(C.static_check("from collections.abc import abc\n").startswith("import_name:abc"))
        self.assertTrue(C.static_check("from collections import abc\n", allowed_modules=["math"]).startswith("import:collections"))
        self.assertEqual(C.static_check("from collections import abc\n"), "")
        for code, want in (("from collections import abc\nprint(abc.update_wrapper)\n", "module_attr:collections.abc.update_wrapper"),
                           ("from collections import abc\nprint(abc.__subclasshook__)\n", "attr:__subclasshook__"),
                           ("from collections import abc\nprint(abc.Foo)\n", "module_attr:collections.abc.Foo"),
                           ("from collections import abc\nx = abc\n", "module_name:"),
                           ("from collections import abc\nprint([abc])\n", "module_name:"),
                           ("from collections import abc\nabc = 1\n", "module_alias:"),
                           ("from collections import abc\nfor abc in [1]:\n    pass\n", "module_alias:"),
                           ("from collections import abc\nimport math as abc\n", "module_alias:")):
            self.assertTrue(C.static_check(code).startswith(want), (code, C.static_check(code)))


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
