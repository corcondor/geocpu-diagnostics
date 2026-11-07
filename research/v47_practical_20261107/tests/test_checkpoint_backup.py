"""CPU-only checkpoint persistence protocol tests; no data bundle or torch needed."""
from __future__ import annotations

import io
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_candidate import (IntermediateCheckpointBackup, checkpoint_steps,
                             checkpoint_storage_requirements, copy_checkpoint_artifact,
                             require_checkpoint_storage)
from prepare_training import sha


class CheckpointBackupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.local = self.root / "scratch/checkpoints"
        self.output = self.root / "drive/new_branch"
        self.local.mkdir(parents=True)
        self.output.mkdir(parents=True)
        self.stream = io.StringIO()
        self.binding = {"variant": "multistep", "new_curriculum_repeat": 16,
                        "training_data_sha256": "a"*64, "parent_sha256": "b"*64,
                        "runtime_sha256": "c"*64, "source_receipt_sha256": "d"*64}

    def tearDown(self):
        self.temp.cleanup()

    def files(self, step):
        return {kind: self.local / f"{'train_state' if kind == 'state' else kind}_step_{step}.pt"
                for kind in ("checkpoint", "optimizer", "state")}

    def make_triplet(self, step):
        paths = self.files(step)
        paths["checkpoint"].write_bytes(f"CPU fixture model {step}; not a checkpoint".encode())
        paths["optimizer"].write_bytes(f"CPU fixture Adam {step}; not optimizer tensors".encode())
        paths["state"].write_text(json.dumps({"step": step, "practical_variant": "multistep",
            "new_curriculum_repeat": 16, "new_dataset_sampler_started": True}), encoding="utf-8")
        return paths

    def monitor(self, **overrides):
        args = dict(local=self.local, output=self.output, steps=10000, binding=self.binding,
                    state_loader=lambda p: json.loads(p.read_text()), stream=self.stream, poll_seconds=.01)
        args.update(overrides)
        return IntermediateCheckpointBackup(**args)

    def test_periodic_steps_exclude_final_from_intermediate_work(self):
        self.assertEqual(checkpoint_steps(10000), [33000, 35500, 38000, 40500])
        self.assertEqual(checkpoint_steps(1000), [31500])
        self.assertEqual(checkpoint_steps(2500), [33000])
        self.assertEqual(checkpoint_steps(2501), [33000, 33001])
        self.assertEqual(self.monitor().steps, [33000, 35500, 38000])

    def test_two_files_and_temporary_state_do_not_trigger_any_copy(self):
        paths = self.make_triplet(33000)
        temporary = paths["state"].with_suffix(".pt.tmp")
        paths["state"].rename(temporary)
        calls = []
        monitor = self.monitor(copy_file=lambda *args: calls.append(args))
        monitor.poll_once()
        self.assertEqual(calls, [])
        self.assertEqual(monitor.attempted, set())
        self.assertFalse((self.output / "checkpoints").exists())
        self.assertTrue(temporary.exists())

    def test_receipt_is_last_verified_and_published_once_without_source_changes(self):
        paths = self.make_triplet(33000)
        before = {kind: path.read_bytes() for kind, path in paths.items()}
        receipt_path = self.output / "checkpoints/receipt_step_33000.json"
        calls = []
        def copier(source, target):
            self.assertFalse(receipt_path.exists())
            calls.append(source.name)
            return copy_checkpoint_artifact(source, target)
        monitor = self.monitor(copy_file=copier)
        monitor.poll_once()
        receipt = json.loads(receipt_path.read_text())
        self.assertEqual(len(calls), 3)
        self.assertEqual(receipt["state"], "intermediate_backup_complete")
        self.assertFalse(receipt["automatic_resume_supported"])
        self.assertFalse(receipt["final_training_complete"])
        self.assertEqual(receipt["evaluation"], "not_run")
        self.assertFalse((self.output / "training_saved.json").exists())
        for kind, source in paths.items():
            self.assertEqual(source.read_bytes(), before[kind])
            self.assertEqual(receipt[kind + "_sha256"], sha(source))
            self.assertEqual(sha(self.output / "checkpoints" / source.name), sha(source))
        original_receipt = receipt_path.read_bytes()
        monitor.poll_once()
        self.assertEqual(len(calls), 3)
        self.assertEqual(receipt_path.read_bytes(), original_receipt)

    def test_copy_error_is_visible_and_does_not_publish_receipt_or_retry(self):
        self.make_triplet(33000)
        calls = []
        def broken(source, target):
            calls.append(source.name)
            if source.name.startswith("optimizer"):
                raise OSError("CPU fixture simulated Drive copy failure")
            return copy_checkpoint_artifact(source, target)
        monitor = self.monitor(copy_file=broken)
        monitor.poll_once()
        count = len(calls)
        monitor.poll_once()
        self.assertEqual(len(calls), count)
        self.assertFalse((self.output / "checkpoints/receipt_step_33000.json").exists())
        self.assertNotIn(33000, monitor.saved)
        for folder in (self.output, self.local.parent):
            status = json.loads((folder / "backup_status.json").read_text())
            self.assertEqual(status["errors"][0]["step"], 33000)
            self.assertIn("simulated Drive copy failure", status["errors"][0]["error"])
            self.assertIn("intermediate_copy_failed", (folder / "backup_events.jsonl").read_text())
        self.assertIn("GECPU_CHECKPOINT_BACKUP", self.stream.getvalue())
        self.assertIn("intermediate_copy_failed", self.stream.getvalue())

    def test_state_binding_error_prevents_copying(self):
        paths = self.make_triplet(33000)
        state = json.loads(paths["state"].read_text())
        state["step"] = 32999
        paths["state"].write_text(json.dumps(state))
        monitor = self.monitor()
        monitor.poll_once()
        self.assertFalse((self.output / "checkpoints").exists())
        self.assertIn("binding differs", monitor.errors[0]["error"])

    def test_backup_never_overwrites_an_existing_destination(self):
        paths = self.make_triplet(33000)
        target = self.output / "checkpoints" / paths["checkpoint"].name
        target.parent.mkdir()
        target.write_bytes(b"existing unrelated data must be retained")
        with self.assertRaises(FileExistsError):
            copy_checkpoint_artifact(paths["checkpoint"], target)
        monitor = self.monitor()
        monitor.poll_once()
        self.assertEqual(target.read_bytes(), b"existing unrelated data must be retained")
        self.assertFalse((target.parent / "receipt_step_33000.json").exists())

    def test_daemon_saves_intermediates_only_and_stops_without_touching_final(self):
        for step in checkpoint_steps(10000):
            self.make_triplet(step)
        finished_copy = threading.Event()
        def copier(source, target):
            digest = copy_checkpoint_artifact(source, target)
            if source.name == "train_state_step_38000.pt":
                finished_copy.set()
            return digest
        monitor = self.monitor(copy_file=copier)
        monitor.start()
        self.assertTrue(monitor.thread.daemon)
        self.assertTrue(finished_copy.wait(2), "CPU fixture backups did not finish")
        report = monitor.close(timeout=2)
        self.assertEqual(report["state"], "complete")
        self.assertEqual(report["saved_steps"], [33000, 35500, 38000])
        self.assertFalse(monitor.thread.is_alive())
        self.assertFalse((self.output / "checkpoints/checkpoint_step_40500.pt").exists())
        self.assertFalse((self.output / "checkpoints/receipt_step_40500.json").exists())
        self.assertTrue(self.files(40500)["checkpoint"].exists())

    def test_failed_backup_does_not_prevent_later_backup_or_separate_final_save(self):
        self.make_triplet(33000)
        self.make_triplet(35500)
        final_paths = self.make_triplet(40500)
        def copier(source, target):
            if "33000" in source.name:
                raise OSError("CPU fixture first backup fails")
            return copy_checkpoint_artifact(source, target)
        monitor = self.monitor(copy_file=copier)
        monitor.poll_once()  # records the failure; it must not propagate
        self.assertTrue((self.output / "checkpoints/receipt_step_35500.json").is_file())
        self.assertFalse((self.output / "checkpoints/receipt_step_33000.json").exists())
        # Main final saving uses separate names and remains independently able
        # to publish them; these are CPU fixture files, never GPU results.
        for source in final_paths.values():
            copy_checkpoint_artifact(source, self.output / "checkpoints" / source.name)
        self.assertTrue((self.output / "checkpoints/checkpoint_step_40500.pt").is_file())

    def test_storage_reserves_all_four_triplets_parent_and_atomic_work(self):
        sizes = {"checkpoint": 1000, "optimizer": 2000, "state": 30}
        plan = checkpoint_storage_requirements(sizes, 10000)
        snapshot = 1000 + 2000 + (64 << 20)
        largest = 64 << 20
        self.assertEqual(plan["drive_required_free_bytes"], 4 * snapshot + 2 * largest + 1_500_000_000)
        self.assertEqual(plan["scratch_required_free_bytes"], sum(sizes.values()) + 4 * snapshot +
                         largest + sizes["optimizer"] + 1_500_000_000)

    def test_storage_checks_each_volume_and_combines_shared_filesystem(self):
        parent = self.root / "parent"
        (parent / "checkpoints").mkdir(parents=True)
        for kind, size in (("checkpoint", 1000), ("optimizer", 2000), ("train_state", 30)):
            (parent / "checkpoints" / f"{kind}_step_30500.pt").write_bytes(b"x" * size)
        plan = checkpoint_storage_requirements({"checkpoint": 1000, "optimizer": 2000, "state": 30}, 10000)
        shared_free = max(plan["drive_required_free_bytes"], plan["scratch_required_free_bytes"])
        usage = lambda path: SimpleNamespace(free=shared_free)
        with self.assertRaisesRegex(RuntimeError, "insufficient checkpoint storage"):
            require_checkpoint_storage(parent, self.local.parent, self.output, 10000,
                                       disk_usage=usage, device_id=lambda path: 1)
        distinct = require_checkpoint_storage(parent, self.local.parent, self.output, 10000,
            disk_usage=usage, device_id=lambda path: 1 if path == self.output.parent else 2)
        self.assertFalse(distinct["shared_filesystem"])
        with self.assertRaisesRegex(RuntimeError, "insufficient checkpoint storage"):
            require_checkpoint_storage(parent, self.local.parent, self.output, 10000,
                disk_usage=lambda path: SimpleNamespace(free=1), device_id=lambda path: 1)


if __name__ == "__main__":
    unittest.main()
