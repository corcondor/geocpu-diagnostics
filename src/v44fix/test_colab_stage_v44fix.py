"""Static/temporary-file checks for the isolated Colab admission path."""
import ast
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock
import torch

import colab_v44fix
from resume_v44fix import PARENT_SHA256, sha256_file


ROOT = Path(__file__).resolve().parent


class ColabStageTests(unittest.TestCase):
    @staticmethod
    def public_fixture(public, work, *, verified):
        (work / "gpt").mkdir(parents=True)
        (work / "data").mkdir()
        for name in ("prepare_public_v44.py", "tir_engine_v44r1.py", "executor_worker_v44r1.py",
                     "public_source_lock_v44.json"):
            shutil.copy(ROOT / "gpt" / name, work / "gpt" / name)
        shutil.copy(ROOT / "base_metadata.json", work / "base_metadata.json")
        anchor = work / "data" / "train_v43_knowledge_unified.jsonl"
        anchor.write_text("{}\n", encoding="utf-8")
        for name in colab_v44fix.REQUIRED_PUBLIC_EVAL_FILES:
            (work / "data" / name).write_text("{}\n", encoding="utf-8")
        public.mkdir()
        for name in ("train.jsonl", "dev.jsonl", "test.jsonl"):
            (public / name).write_text("{}\n", encoding="utf-8")
        lock_path = work / "gpt" / "public_source_lock_v44.json"
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        source_pins = [{"source": name, "split": split, "bytes": entry["bytes"], "sha256": entry["sha256"]}
                       for name, spec in lock["repos"].items() for split, entry in spec["files"].items()]
        eval_pins = [{"path": str(work / "data" / name), "bytes": (work / "data" / name).stat().st_size,
                      "sha256": sha256_file(work / "data" / name)}
                     for name in sorted(colab_v44fix.REQUIRED_PUBLIC_EVAL_FILES)]
        official = lock["repos"]["numina"]["files"]["test"]
        eval_pins.append({"path": official["path"], "source": "Numina official test: never training",
                          "bytes": official["bytes"], "sha256": official["sha256"], "rows": 1})
        record = {"training_eligible": True, "public_input_pins_match": True,
                  "max_total_tokens": 8192,
                  "tool_verification": {"mode": "all" if verified else "fixed_first_eligible_problem_sample",
                                        "all_observations_locally_executed": verified},
                  "statistics": {"tir:accepted_unique_rows": 1, "tir:sample_execution_passed": 1},
                  "counts": {"unique_training_pool_by_bucket": {"tir": 1}},
                  "source_lock_sha256": sha256_file(lock_path),
                  "public_source_inputs": source_pins,
                  "vocabulary_sha256": sha256_file(work / "base_metadata.json"),
                  "anchor_input": {"bytes": anchor.stat().st_size, "sha256": sha256_file(anchor)},
                  "evaluation_inputs": eval_pins,
                  "outputs": {name: {"sha256": sha256_file(public / name),
                                     "bytes": (public / name).stat().st_size}
                              for name in ("train.jsonl", "dev.jsonl", "test.jsonl")}}
        (public / "dataset_manifest.json").write_text(json.dumps(record), encoding="utf-8")
        receipt = {"schema": "geocpu_public_tir_offline_verification_v1",
                   "public_manifest_sha256": sha256_file(public / "dataset_manifest.json"),
                   "isolation": {"kind": "credential_free_disposable_vm", "network": "disabled",
                                 "drive_mounted": False},
                   "source_code_sha256": {name: sha256_file(work / "gpt" / name) for name in
                                          ("prepare_public_v44.py", "tir_engine_v44r1.py",
                                           "executor_worker_v44r1.py")}}
        receipt_path = public / "offline_verification_receipt.json"
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        return sha256_file(receipt_path)

    def test_isolated_run_paths_and_pinned_parent(self):
        self.assertIn("GeoCPU_v44fix_TIR_consistency_20261103", str(colab_v44fix.OUT))
        self.assertNotEqual(str(colab_v44fix.OUT),
                            "/content/drive/MyDrive/GeoCPU_v44fix_20261103/run_1")
        self.assertEqual(PARENT_SHA256,
                         "a4b95d7d0f486146196271b3b1fbc6a1b6b3342d8fadd590320fc3842c1e80e5")

    def test_cached_gate_requires_same_checkpoint_and_case_count(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "parent.pt"
            checkpoint.write_bytes(b"first")
            output = Path(directory) / "gate.json"
            record = {"fingerprint": "profile", "checkpoint_sha256": sha256_file(checkpoint),
                      "n_tir_requested": 20, "baseline_sha256": None}
            output.write_text(json.dumps(record), encoding="utf-8")
            with mock.patch.object(colab_v44fix, "V43_CKPT", checkpoint), \
                 mock.patch.object(colab_v44fix, "fingerprint", return_value="profile"), \
                 mock.patch.object(colab_v44fix, "run") as execute:
                self.assertEqual(colab_v44fix.gate_eval(checkpoint, output, "baseline", n_tir=20), record)
                execute.assert_not_called()
                with self.assertRaises(RuntimeError):
                    colab_v44fix.gate_eval(checkpoint, output, "baseline", n_tir=100)
                checkpoint.write_bytes(b"second")
                with self.assertRaises(RuntimeError):
                    colab_v44fix.gate_eval(checkpoint, output, "baseline", n_tir=20)

    def test_public_reuse_requires_all_execution_and_matching_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            public, out, work = (Path(directory) / name for name in ("public", "out", "work"))
            out.mkdir()
            pin = self.public_fixture(public, work, verified=False)
            with mock.patch.object(colab_v44fix, "PUBLIC", public), mock.patch.object(colab_v44fix, "OUT", out), \
                 mock.patch.object(colab_v44fix, "WORK", work), \
                 mock.patch.object(colab_v44fix, "OFFLINE_VERIFICATION_RECEIPT_SHA256", pin), \
                 mock.patch.object(colab_v44fix, "run") as execute:
                with self.assertRaisesRegex(SystemExit, "does not establish complete execution"):
                    colab_v44fix.stage_public()
                execute.assert_not_called()
            record = json.loads((public / "dataset_manifest.json").read_text(encoding="utf-8"))
            record["tool_verification"] = {"mode": "all", "all_observations_locally_executed": True}
            (public / "dataset_manifest.json").write_text(json.dumps(record), encoding="utf-8")
            receipt_path = public / "offline_verification_receipt.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["public_manifest_sha256"] = sha256_file(public / "dataset_manifest.json")
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            with mock.patch.object(colab_v44fix, "PUBLIC", public), mock.patch.object(colab_v44fix, "OUT", out), \
                 mock.patch.object(colab_v44fix, "WORK", work), \
                 mock.patch.object(colab_v44fix, "OFFLINE_VERIFICATION_RECEIPT_SHA256", sha256_file(receipt_path)), \
                 mock.patch.object(colab_v44fix, "run") as execute:
                colab_v44fix.stage_public()
                execute.assert_not_called()
                (public / "train.jsonl").write_text("changed\n", encoding="utf-8")
                with self.assertRaisesRegex(SystemExit, "output hash"):
                    colab_v44fix.stage_public()

    def test_missing_public_curriculum_stops_before_untrusted_code_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            public, out = Path(directory) / "public", Path(directory) / "out"
            out.mkdir()
            with mock.patch.object(colab_v44fix, "PUBLIC", public), mock.patch.object(colab_v44fix, "OUT", out), \
                 mock.patch.object(colab_v44fix, "run") as execute:
                with self.assertRaisesRegex(SystemExit, "training_ready=false"):
                    colab_v44fix.stage_public()
                execute.assert_not_called()
            self.assertFalse((out / "public_dataset_manifest.json").exists())

    def test_main_stops_before_preflight_or_python_worker_when_receipt_unpinned(self):
        with mock.patch.object(colab_v44fix, "OFFLINE_VERIFICATION_RECEIPT_SHA256", None), \
             mock.patch.object(colab_v44fix, "ensure_inputs") as preflight, \
             mock.patch.object(colab_v44fix, "stage_public") as stage, \
             mock.patch.object(colab_v44fix, "run") as execute:
            with self.assertRaisesRegex(SystemExit, "training_ready=false"):
                colab_v44fix.main()
            preflight.assert_not_called()
            stage.assert_not_called()
            execute.assert_not_called()

    def test_full_run_uses_separate_parent_until_its_own_verified_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with mock.patch.object(colab_v44fix, "run") as execute:
                colab_v44fix.train(output, "full", resume=True)
                command = execute.call_args.args[0]
                self.assertEqual(command[command.index("--warmstart_path") + 1], colab_v44fix.V43_CKPT)
                self.assertNotIn("--require_exact_resume", command)
                checkpoint = output / "checkpoint_step_17500.pt"
                checkpoint.write_bytes(b"fixture")
                with mock.patch.object(colab_v44fix, "ckpt_done", return_value=True):
                    colab_v44fix.train(output, "full", resume=True)
                command = execute.call_args.args[0]
                self.assertEqual(command[command.index("--warmstart_path") + 1], checkpoint)
                self.assertIn("--require_exact_resume", command)

    def test_low_drive_space_stops_before_any_model_or_data_work(self):
        with tempfile.TemporaryDirectory() as directory:
            drive = Path(directory)
            (drive / "GeoCPU_Frontier_CoTraining_v43").mkdir()
            with mock.patch.object(colab_v44fix, "DRIVE", drive), \
                 mock.patch.object(torch.cuda, "is_available", return_value=True), \
                 mock.patch.object(torch.cuda, "device_count", return_value=1), \
                 mock.patch.object(torch.cuda, "get_device_name", return_value="NVIDIA A100-SXM4-40GB"), \
                 mock.patch.object(colab_v44fix.shutil, "disk_usage",
                                   return_value=mock.Mock(free=17_000_000_000)):
                with self.assertRaisesRegex(RuntimeError, "at least 18000000000"):
                    colab_v44fix.ensure_inputs()

    def test_gpu_preflight_and_gate_counts_before_training(self):
        with tempfile.TemporaryDirectory() as directory:
            drive = Path(directory)
            (drive / "GeoCPU_Frontier_CoTraining_v43").mkdir()
            with mock.patch.object(colab_v44fix, "DRIVE", drive), \
                 mock.patch.object(torch.cuda, "is_available", return_value=False):
                with self.assertRaisesRegex(RuntimeError, "A100 GPU"):
                    colab_v44fix.ensure_inputs()
        valid = {name: {"n": count} for name, count in
                 {"swe": 200, "state": 200, "search": 200, "alu": 8, "tir": 20}.items()}
        valid["loops"] = {"samples": [{} for _ in range(10)]}
        colab_v44fix.require_gate_counts(valid, n_tir=20)
        valid["search"]["n"] = 0
        with self.assertRaisesRegex(RuntimeError, "search"):
            colab_v44fix.require_gate_counts(valid, n_tir=20)

    def test_runtime_fails_closed_for_missing_parent_oom_and_nonfinite(self):
        tree = ast.parse((ROOT / "runtime_v44fix.py").read_text(encoding="utf-8"))
        missing = [node for node in ast.walk(tree) if isinstance(node, ast.Raise)
                   and isinstance(node.exc, ast.Call) and isinstance(node.exc.func, ast.Name)
                   and node.exc.func.id == "FileNotFoundError"]
        self.assertTrue(missing)
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                     and node.name == "run_microbatch_safe"]
        self.assertEqual(len(functions), 1)
        oom_rejections = [node for node in ast.walk(functions[0]) if isinstance(node, ast.Raise)
                          and isinstance(node.exc, ast.Call) and isinstance(node.exc.func, ast.Name)
                          and node.exc.func.id == "RuntimeError"
                          and node.exc.args and isinstance(node.exc.args[0], ast.Constant)
                          and "entire update rejected" in str(node.exc.args[0].value)]
        self.assertEqual(len(oom_rejections), 1)
        self.assertTrue(any(isinstance(node, ast.Raise)
                            and isinstance(node.exc, ast.Call)
                            and isinstance(node.exc.func, ast.Name)
                            and node.exc.func.id == "FloatingPointError"
                            for node in ast.walk(tree)))


if __name__ == "__main__":
    unittest.main()
