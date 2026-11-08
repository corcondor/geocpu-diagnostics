"""CPU-only tests for the A100 fast preflight. No checkpoint, GPU or Drive access.

- the one-pass simulator equals the pinned v47 preview_plan (real code from the repo)
  on a synthetic corpus, including out-of-vocabulary rows, and stops at the accepted budget;
- the SHA-bound dataset cache never hands out a file whose hash differs;
- the A100 runtime keeps the pinned BF16 path and only relaxes A100->A100 resume;
- the GPU stage refuses approvals bound to other code/profile or an unconfirmed budget.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
V47 = REPO / "research/v47_practical_20261107"
sys.path.insert(0, str(HERE))

import multibudget_preflight as MB  # noqa: E402
import train_from_40500 as T  # noqa: E402

BASE_SAMPLER = '''
class DynamicTokenBucketBatchSampler:
    def __init__(self, dataset, max_batch_size=64, max_tokens=36864, megabucket_mult=32):
        self.dataset = dataset
        self.max_batch_size = max_batch_size
        self.max_tokens = max_tokens
'''
RESUMABLE = '''
class ResumableTokenBucketSampler(DynamicTokenBucketBatchSampler):
    def __init__(self, dataset, max_batch_size=64, max_tokens=36864, megabucket_mult=32, seed=0):
        super().__init__(dataset, max_batch_size, max_tokens, megabucket_mult)
        self.seed = seed
        self.epoch = 0
    def __iter__(self):
        order = list(range(len(self.dataset.samples)))
        random.Random(self.seed + self.epoch).shuffle(order)
        self.epoch += 1
        batch, tokens = [], 0
        for i in order:
            n = len(self.dataset.samples[i][0])
            if batch and (len(batch) >= self.max_batch_size or tokens + n > 600):
                yield batch
                batch, tokens = [], 0
            batch.append(i)
            tokens += n
        if batch:
            yield batch
'''


def load_prep():
    spec = importlib.util.spec_from_file_location("prepare_training_pinned", V47 / "prepare_training.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class OnePassParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        d = Path(cls.tmp.name)
        cls.prep = load_prep()
        cls.prep.NEW_GROUP = T.V48_GROUP
        (d / "base.py").write_text(BASE_SAMPLER, encoding="utf-8")
        (d / "runtime.py").write_text(RESUMABLE, encoding="utf-8")
        (d / "norm.py").write_text('_BASE_NORM_DICT = {"−": "-"}\n', encoding="utf-8")
        vocab = {c: i for i, c in enumerate("abcdefghijklmnopqrstuvwxyz 0123456789:答え問題思考。-")}
        (d / "meta.json").write_text(json.dumps({"vocab": vocab}, ensure_ascii=False), encoding="utf-8")
        rows = []
        for i in range(160):
            group = T.V48_GROUP if i % 5 == 0 else "legacy"
            src = ["old_a", "old_b", "rare"][i % 3] if group == "legacy" else "v48_python"
            prompt = f"問題{i}: what is {i}?"
            target = " 思考: " + "x" * (5 + i % 23) + (" ★" if i % 17 == 0 else "") + f" 答え: {i}"
            rows.append({"id": f"r{i}", "prompt": prompt, "target": target, "_v46_group": group,
                         "_practical_source": src, "loss_mask_spans": []})
        cls.data = d / "train.jsonl"
        cls.data.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        cls.common = dict(meta=d / "meta.json", normalizer_source=d / "norm.py",
                          base_sampler_source=d / "base.py", runtime_source=d / "runtime.py")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_every_budget_equals_pinned_preview_plan(self):
        snaps = MB.preview_budgets(self.prep, self.data, budgets=[40, 55, 70], repeat=1, min_new=0, **self.common)
        for steps in (40, 55, 70):
            frozen = self.prep.preview_plan(self.data, steps=steps, repeat=1, min_new=0, **self.common)
            self.assertEqual(snaps[steps], frozen, steps)
        self.assertTrue(any(s > 0 for s in snaps[70]["source_supervised_tokens"].values()))

    def test_stops_at_first_accepted_budget(self):
        seen = []
        def accept(snap):
            seen.append(snap["steps"])
            return snap["steps"] >= 55
        snaps = MB.preview_budgets(self.prep, self.data, budgets=[40, 55, 70, 85], repeat=1, min_new=0,
                                   accept=accept, **self.common)
        self.assertEqual(sorted(snaps), [40, 55])
        self.assertEqual(seen, [40, 55])

    def test_fast_count_matches_python_loop(self):
        vocab = set("abc答")
        count = MB._in_vocab_counter(vocab)
        for s in ("abc", "ab★c", "", "答えab"):
            self.assertEqual(count(s), sum(c in vocab for c in s))


class PlanChoice(unittest.TestCase):
    """Replay weights are sized at the 20k horizon, so the plan uses the fewest GPU updates;
    the no-replay alternative is recorded for the report."""

    def plan(self, floor, maximum=60000):
        from unittest import mock
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            data = d / "train.jsonl"
            data.write_text(json.dumps({"id": "r1", "prompt": "q", "target": "a", "_v46_group": "old",
                                        "_practical_source": "rare", "loss_mask_spans": []}) + "\n", encoding="utf-8")
            def fake_preview(prep, path, *, budgets, accept=None, **kw):
                rare = sum(json.loads(x).get("_practical_source") == "rare"
                           for x in Path(path).read_text(encoding="utf-8").splitlines())
                out = {}
                for steps in budgets:
                    out[steps] = {"steps": steps, "passed": True, "supervised_tokens": 10**6,
                                  "source_supervised_tokens": {"rare": rare * (steps // 1000), "v48_python": 300,
                                                               "v48_logic": 300, "v48_evidence": 300,
                                                               "v48_japanese": 300}}
                    if accept and accept(out[steps]):
                        break
                return out
            alt = {}
            with (mock.patch("multibudget_preflight.preview_budgets", side_effect=fake_preview),
                  mock.patch("multibudget_preflight.assert_preview_parity", return_value=True)):
                res = T.cpu_plan_with_replay_floor_repair(data, d, object(), d, d,
                                                          {"source_supervised_tokens": {"rare": floor}},
                                                          10000, maximum, alternatives=alt)
            return res[1], res[4]["replay_replicas_added"], alt

    def test_fewer_updates_with_explicit_replay_than_without(self):
        steps, replicas, alt = self.plan(25)       # without replay: first pass at 25,000 (beyond 20k)
        self.assertEqual(alt["first_passing_updates_without_replay"], 25000)
        self.assertEqual(replicas, 1)              # weight 2 on the one 'rare' row
        self.assertEqual(steps, 13000)             # 2 * 13 >= 25

    def test_no_replay_when_floors_hold_within_horizon(self):
        steps, replicas, alt = self.plan(15)
        self.assertEqual((steps, replicas), (15000, 0))
        self.assertEqual(alt["first_passing_updates_without_replay"], 15000)


class DatasetCache(unittest.TestCase):
    def test_roundtrip_and_rejects_changed_bytes(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            src = d / "parent.jsonl"
            src.write_bytes(b'{"prompt":"a"}\n' * 100)
            sha = hashlib.sha256(src.read_bytes()).hexdigest()
            cache = d / "cache"
            T.cache_store(cache, "parent.jsonl", sha, src)
            got = T.cache_fetch(cache, "parent.jsonl", sha, d / "copy.jsonl")
            self.assertEqual(got.read_bytes(), src.read_bytes())
            (cache / "parent.jsonl").write_bytes(b"tampered\n")
            self.assertIsNone(T.cache_fetch(cache, "parent.jsonl", sha, d / "copy2.jsonl"))
            self.assertFalse((d / "copy2.jsonl").exists())
            self.assertFalse(any(p.name.endswith(".part") for p in d.iterdir()))
            self.assertIsNone(T.cache_fetch(None, "parent.jsonl", sha, d / "copy3.jsonl"))

    def test_prefix_digest_proves_inherited_rows(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "c.jsonl"
            parent = b"row1\nrow2\n"
            p.write_bytes(parent + b"new\n")
            self.assertEqual(T.digest_prefix(p, len(parent)), hashlib.sha256(parent).hexdigest())
            with self.assertRaises(ValueError):
                T.digest_prefix(p, 10_000)


class A100Runtime(unittest.TestCase):
    def setUp(self):
        raw = (V47 / "runtime_practical_v1.py").read_bytes()
        if hashlib.sha256(raw).hexdigest() != T.PINNED_RUNTIME:
            self.skipTest("pinned runtime checked out with different line endings")

    def test_a100_keeps_bf16_and_only_relaxes_a100_variants(self):
        text = T.runtime_text(V47, "a100")
        self.assertIn('if not (args.new_dataset_sampler and "A100" in state["gpu_name"] and "A100" in _v48_gpu):', text)
        self.assertNotIn('raise RuntimeError("GPU model changed on exact resume")\n            if args.new_dataset_sampler', text)
        self.assertIn("dtype=torch.bfloat16", text)
        self.assertNotIn("GradScaler", text)
        self.assertNotIn("amp_dtype", text)
        self.assertIn(T.V48_GROUP, text)

    def test_t4_profile_still_builds(self):
        self.assertIn("amp_dtype", T.runtime_text(V47, "t4"))

    def test_commands(self):
        p = Path("/x")
        a = T.command_for(p, p, p, p, p, p, 52500, "a100")
        t = T.command_for(p, p, p, p, p, p, 52500, "t4")
        self.assertNotIn("--chunk_tokens", a)
        self.assertEqual(t[t.index("--chunk_tokens") + 1], "16384")
        for cmd in (a, t):
            self.assertEqual(cmd[cmd.index("--step_tokens") + 1], "36864")
            self.assertEqual(cmd[cmd.index("--batch_size") + 1], "64")
            self.assertEqual(cmd[cmd.index("--phase_lr") + 1], "3e-6")
            self.assertEqual(cmd[cmd.index("--stop_at_step") + 1], "52500")


class Approval(unittest.TestCase):
    def setUp(self):
        self.meta = {"repeat_factor_by_domain": {"python": 1}, "reference_target_tokens_by_domain": {"python": 9}}
        self.ok = {"event": "v48_preflight", "preflight_revision": "cpu_fast_v2", "start_step": 40500,
                   "parent_checkpoint_sha256": T.PINNED_PARENT_MODEL, "parent_train_sha256": T.PINNED_PARENT_TRAIN,
                   "runtime_sha256": "r" * 64, "gpu_profile": "a100", "steps": 12000, "estimated_gpu_hours": 2.62,
                   "train": {"new_unique_examples": 4096, "inherited_rows": 423348, **self.meta}}

    def check(self, approval, **kw):
        args = dict(runtime_sha="r" * 64, profile="a100", metadata=self.meta, minimum=10000, maximum=60000,
                    confirm=12000)
        args.update(kw)
        return T.validate_approval(approval, **args)

    def test_accepts_exact_match(self):
        self.assertEqual(self.check(self.ok)[1], 12000)

    def test_rejects_other_profile_runtime_or_unconfirmed_budget(self):
        with self.assertRaises(ValueError):
            self.check(self.ok, profile="t4")
        with self.assertRaises(ValueError):
            self.check(self.ok, runtime_sha="s" * 64)
        with self.assertRaisesRegex(ValueError, "confirm"):
            self.check(self.ok, confirm=None)
        with self.assertRaisesRegex(ValueError, "confirm"):
            self.check(self.ok, confirm=11000)
        stale = dict(self.ok, preflight_revision="cpu_fast_v1")
        with self.assertRaises(ValueError):
            self.check(stale)
        shrunk = json.loads(json.dumps(self.ok))
        shrunk["train"]["inherited_rows"] = 400000
        with self.assertRaises(ValueError):
            self.check(shrunk)

    def test_cost_estimate_uses_measured_a100_rate(self):
        self.assertAlmostEqual(T.estimated_gpu_hours("a100", 10000), round(10000 / 1.27 / 3600, 2))
        self.assertIsNone(T.estimated_gpu_hours("t4", 10000))


if __name__ == "__main__":
    unittest.main()
