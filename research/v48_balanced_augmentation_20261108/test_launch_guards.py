"""CPU-only tests for the launch guards found by the first real A100 attempt (2026-11-08).

The attempt failed before any training update with
    ValueError: additional steps must be 1..10000
because v47's checkpoint_steps()/IntermediateCheckpointBackup cap the update count at 10,000 while the CPU
preflight had approved 21,000, and the failure left training_active.json behind, which blocks every later launch.
These tests run the REAL v47 train_candidate module (no GPU, no Drive), so the same code path that failed is exercised.
"""
import ast
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest

HERE = Path(__file__).resolve().parent
V47 = HERE.parents[1] / "research/v47_practical_20261107"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(V47))

import train_from_40500 as T  # noqa: E402
import train_candidate as OLD  # noqa: E402


class FakeOld:
    """what refuse_active_work_v48 needs from the v47 module"""
    def __init__(self, active=None):
        self._active = active or []

    def active_work(self):
        return self._active


class CheckpointStepsCap(unittest.TestCase):
    def setUp(self):
        self.saved = (OLD.PARENT_STEP, OLD.CHECKPOINT_INTERVAL, OLD.checkpoint_steps)
        self.addCleanup(self.restore)
        OLD.PARENT_STEP, OLD.CHECKPOINT_INTERVAL = 40500, 5000

    def restore(self):
        OLD.PARENT_STEP, OLD.CHECKPOINT_INTERVAL, OLD.checkpoint_steps = self.saved

    def backup(self, steps):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            return OLD.IntermediateCheckpointBackup(local=Path(a), output=Path(b), steps=steps, binding={},
                                                    state_loader=lambda p: {})

    def test_unpatched_v47_reproduces_the_colab_failure(self):
        with self.assertRaisesRegex(ValueError, r"additional steps must be 1\.\.10000"):
            self.backup(21000)

    def test_patched_backup_accepts_the_approved_update_counts(self):
        T.install_uncapped_checkpoint_steps(OLD)
        b = self.backup(21000)
        self.assertEqual(b.steps, [45500, 50500, 55500, 60500])
        self.assertEqual(b.final_step, 61500)
        self.assertEqual(OLD.checkpoint_steps(21000)[-1], 61500)
        self.assertEqual(self.backup(12000).steps, [45500, 50500])   # stop 52,500: 55,500 is beyond it

    def test_patched_function_equals_v47_wherever_v47_works(self):
        original = OLD.checkpoint_steps
        T.install_uncapped_checkpoint_steps(OLD)
        for steps in (1, 2500, 4999, 5000, 5001, 9999, 10000):
            self.assertEqual(OLD.checkpoint_steps(steps), original(steps), steps)

    def test_still_bounded(self):
        T.install_uncapped_checkpoint_steps(OLD)
        for bad in (0, -1, 100001):
            with self.assertRaises(ValueError):
                OLD.checkpoint_steps(bad)


class StaleLocks(unittest.TestCase):
    def make(self, root, name, files=(), dirs=(), age=3600, lock=True):
        folder = Path(root) / name
        folder.mkdir()
        for f in files:
            (folder / f).write_text("{}", encoding="utf-8")
        for d in dirs:
            (folder / d).mkdir()
        if lock:
            p = folder / "training_active.json"
            p.write_text('{"pid": 3129}', encoding="utf-8")
            t = time.time() - age
            os.utime(p, (t, t))
        return folder

    def test_failed_launch_without_training_is_stale(self):
        with tempfile.TemporaryDirectory() as td:
            self.make(td, "v48_failed", files=["fork_from_40500.json", "input_manifest.json", "preflight.json",
                                               "runtime_v48_40500.py"])
            self.assertEqual([p.parent.name for p in T.unstarted_stale_locks(td)], ["v48_failed"])
            T.refuse_active_work_v48(FakeOld(), Path(td))     # no error

    def test_real_or_fresh_runs_still_block(self):
        for name, kw in (("trained", dict(dirs=["console"])), ("saved", dict(dirs=["checkpoints"])),
                         ("telemetry", dict(files=["training_events.jsonl"])), ("fresh", dict(age=30))):
            with self.subTest(name), tempfile.TemporaryDirectory() as td:
                self.make(td, name, **kw)
                self.assertEqual(T.unstarted_stale_locks(td), [])
                with self.assertRaisesRegex(RuntimeError, "retains its lock"):
                    T.refuse_active_work_v48(FakeOld(), Path(td))

    def test_a_live_process_always_blocks(self):
        with tempfile.TemporaryDirectory() as td:
            self.make(td, "v48_failed")
            with self.assertRaisesRegex(RuntimeError, "active"):
                T.refuse_active_work_v48(FakeOld([{"pid": 1, "cmdline": "runtime_v48_40500.py"}]), Path(td))

    def test_other_lock_kinds_are_not_ignored(self):
        with tempfile.TemporaryDirectory() as td:
            folder = self.make(td, "eval_run", lock=False)
            (folder / "evaluation_active.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "retains its lock"):
                T.refuse_active_work_v48(FakeOld(), Path(td))

    def test_stale_lock_is_ignored_not_deleted(self):
        with tempfile.TemporaryDirectory() as td:
            folder = self.make(td, "v48_failed")
            T.refuse_active_work_v48(FakeOld(), Path(td))
            self.assertTrue((folder / "training_active.json").exists())


class FailureLeavesNoLock(unittest.TestCase):
    def test_backup_is_built_inside_the_try_that_removes_the_lock(self):
        tree = ast.parse((HERE / "train_from_40500.py").read_text(encoding="utf-8"))
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        tries = [n for n in ast.walk(main) if isinstance(n, ast.Try)]
        self.assertTrue(tries)
        guarded = [t for t in tries if any(
            isinstance(s, ast.Assign) and "IntermediateCheckpointBackup" in ast.dump(s.value) for s in t.body)]
        self.assertEqual(len(guarded), 1)
        self.assertTrue(any("training_active.json" in ast.dump(s) for s in guarded[0].finalbody))
        self.assertTrue(any(h.type is not None and "BaseException" in ast.dump(h.type) for h in guarded[0].handlers))
        # nothing that can fail is left between the lock file being written and the try
        body = main.body
        src = (HERE / "train_from_40500.py").read_text(encoding="utf-8")
        before = src[:src.index("backup=None")]
        self.assertGreater(before.rindex('new_json(output/"training_active.json"'), 0)
        between = before[before.rindex('shutil.copyfile(runtime,output/"runtime_v48_40500.py")'):]
        self.assertNotIn("IntermediateCheckpointBackup", between)
        self.assertTrue(body)


if __name__ == "__main__":
    unittest.main()
