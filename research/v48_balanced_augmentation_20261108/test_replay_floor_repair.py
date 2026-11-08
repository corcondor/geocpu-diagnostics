"""CPU-only tests. No model checkpoint or GPU is required."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

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
        calls = []
        def fake_preview(prep,data,*,budgets,**kwargs):
            calls.append(list(budgets))
            rows = [json.loads(x) for x in Path(data).read_text(encoding="utf-8").splitlines()]
            rare = sum(row.get("_practical_source")=="rare" for row in rows)
            return {
                steps:{"steps":steps,"passed":True,
                       "source_supervised_tokens":{
                           "rare":rare*(steps//1000),
                           "v48_python":300,"v48_logic":300,
                           "v48_evidence":300,"v48_japanese":300},
                       "supervised_tokens":rare*(steps//1000)+1200}
                for steps in budgets
            }
        with (mock.patch("multibudget_preflight.preview_budgets", side_effect=fake_preview),
              mock.patch("multibudget_preflight.assert_preview_parity", return_value=True)):
            resulting,steps,preview,review,stats=cpu_plan_with_replay_floor_repair(
                self.original,self.root,object(),self.root,self.root,
                {"source_supervised_tokens":{"rare":100}},10000,20000)
        self.assertTrue(review["passed"])
        self.assertEqual(steps,17000)
        self.assertEqual(stats["replay_replicas_added"],5)
        self.assertEqual(len(calls),2)  # Not 11 scans per attempt.
        self.assertEqual(calls[0],list(range(10000,20001,1000)))
        self.assertTrue(resulting.exists())
        self.assertEqual(self.original.read_text(encoding="utf-8"),self.raw)


if __name__ == "__main__":
    unittest.main()
