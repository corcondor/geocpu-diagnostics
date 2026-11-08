"""CPU-only tests. No model checkpoint or GPU is required."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_floor_repair import next_weights, write_weighted_candidate
from train_from_40500 import cpu_plan_with_replay_floor_repair


class ReplayFloorRepairTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.original = self.root / "train.jsonl"
        self.old_rows = [
            {"id":"r1","prompt":"question","target":"answer",
             "_v46_group":"old","_practical_source":"rare","loss_mask_spans":[]},
            {"id":"c1","prompt":"q2","target":"a2",
             "_v46_group":"old","_practical_source":"common","loss_mask_spans":[]},
        ]
        self.raw = "".join(json.dumps(r) + "\n" for r in self.old_rows)
        self.original.write_text(self.raw, encoding="utf-8")

    def test_weighted_copy_preserves_original_bytes_and_group(self):
        candidate = self.root / "weighted.jsonl"
        info = write_weighted_candidate(self.original, candidate, {"rare":3})
        self.assertEqual(self.original.read_text(encoding="utf-8"), self.raw)
        body = candidate.read_text(encoding="utf-8")
        self.assertTrue(body.startswith(self.raw))
        rows = [json.loads(line) for line in body.splitlines()]
        self.assertEqual(len(rows), 4)
        self.assertEqual(info["replay_replicas_added"], 2)
        for clone in rows[2:]:
            self.assertTrue(clone["_v48_replay_replica"])
            self.assertEqual(clone["_practical_source"], "rare")
            self.assertEqual(clone["target"], "answer")
            self.assertEqual(clone["loss_mask_spans"], [])
        self.assertNotEqual(rows[2]["id"], rows[3]["id"])

    def test_absent_source_fails_closed_and_unlinks_partial(self):
        dest = self.root / "bad.jsonl"
        with self.assertRaisesRegex(RuntimeError, "absent"):
            write_weighted_candidate(self.original, dest, {"missing":2})
        self.assertFalse(dest.exists())

    def test_existing_path_refused(self):
        with self.assertRaises(FileExistsError):
            write_weighted_candidate(self.original, self.original, {"rare":2})

    def test_repair_weights_only_underrepresented_source(self):
        first = next_weights({"rare":{"baseline":100,"candidate":20}}, {})
        self.assertEqual(first, {"rare":6})
        self.assertEqual(next_weights({"rare":{"baseline":100,"candidate":80}}, first)["rare"], 12)
        with self.assertRaises(ValueError):
            next_weights({"rare":{"baseline":100,"candidate":100}}, {})

    def test_cannot_silently_relax_floor(self):
        with self.assertRaises(RuntimeError):
            next_weights({"rare":{"baseline":100,"candidate":1}}, {"rare":32})

    def test_cpu_plan_selects_lowest_passing_budget(self):
        class FakePrep:
            def __init__(self): self.calls = []
            def preview_plan(self, train, *, steps, **kwargs):
                self.calls.append(steps)
                rows = [json.loads(s) for s in Path(train).read_text(encoding="utf-8").splitlines()]
                rare = sum(row.get("_practical_source") == "rare" for row in rows)
                # Deterministic toy sampler: one rare-source unit per 1000 steps.
                supervision = rare * (steps // 1000)
                return {"source_supervised_tokens":{
                            "rare": supervision,
                            "v48_python":300,"v48_logic":300,
                            "v48_evidence":300,"v48_japanese":300},
                        "supervised_tokens":supervision+1200}
        src = self.root / "source"
        src.mkdir()
        original = self.root / "previous"
        original.mkdir()
        prep = FakePrep()
        resulting, steps, preview, review, stats = cpu_plan_with_replay_floor_repair(
            self.original,self.root,prep,src,original,
            {"source_supervised_tokens":{"rare":100}},10000,20000)
        self.assertTrue(review["passed"])
        self.assertEqual(steps, 17000)
        self.assertEqual(stats["replay_replicas_added"], 5)
        self.assertEqual(prep.calls[0], 20000)
        self.assertEqual(prep.calls[1], 20000)
        self.assertTrue(resulting.exists())
        self.assertEqual(self.original.read_text(encoding="utf-8"), self.raw)


if __name__ == "__main__":
    unittest.main()
