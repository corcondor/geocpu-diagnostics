"""CPU-only rejection and static checks for the private paired evaluation package."""
from __future__ import annotations

import ast
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import prepare
import compare_fixed44
from source_pools import PINNED_V43_ZIP_SHA, PYTHON_MODE_CUE, file_sha


class PackageTests(unittest.TestCase):
    def test_pinned_sources_and_unchanged_parent_evaluator(self):
        for name, expected in prepare.SOURCE_SHA.items():
            self.assertEqual(file_sha(prepare.STUDY / name), expected, name)
        for name, expected in prepare.SOURCE_OVERLAY_SHA.items():
            self.assertEqual(file_sha(prepare.OVERLAY / name), expected, name)
        self.assertEqual(prepare.SOURCE_SHA["eval_fixed_diagnostic.py"],
                         compare_fixed44.EVALUATOR_SHA)
        self.assertEqual(prepare.SOURCE_SHA["fixed_interim_diagnostic_44_v2.jsonl"],
                         compare_fixed44.CASES_SHA)
        self.assertEqual(prepare.PAIRED_KERNEL_ID,
                         "geocpuresearch/geocpu-v43py-stage1-m4096-20261104")
        self.assertLessEqual(len(prepare.KERNEL_ID.split("/", 1)[1]), 50)

    def test_bootstrap_is_syntactic_and_uses_both_gpus_and_bounded_evaluator(self):
        template = (prepare.HERE / "bootstrap_template.py").read_text(encoding="utf-8")
        self.assertEqual(template.count("__EMBEDDED__"), 1)
        self.assertEqual(template.count("__EXPECTED__"), 1)
        ast.parse(template.replace("__EMBEDDED__", "{}").replace("__EXPECTED__", "{}"))
        self.assertIn('env["CUDA_VISIBLE_DEVICES"] = str(index)', template)
        self.assertIn('str(code / "launch_fixed_diagnostic_bounded.py")', template)
        self.assertIn('"--checkpoint-kind", "hybrid_250"', template)
        self.assertIn('"--hard-stop-seconds", "1800"', template)
        self.assertIn('"--save-reserve-seconds", "180"', template)
        self.assertIn('"stage_promotion": False', template)
        self.assertIn('find_paired_mount(Path("/kaggle/input")', template)
        self.assertIn('"kernel_sources": [PAIRED_KERNEL_ID]',
                      (prepare.HERE / "prepare.py").read_text(encoding="utf-8"))
        ast.parse((prepare.HERE / "prepare.py").read_text(encoding="utf-8"))
        ast.parse((prepare.HERE / "compare_fixed44.py").read_text(encoding="utf-8"))

    def test_direct_kernel_source_mount_requires_one_matching_status_hash(self):
        template = (prepare.HERE / "bootstrap_template.py").read_text(encoding="utf-8")
        code = template.replace("__EMBEDDED__", "{}").replace("__EXPECTED__", "{}")
        namespace = {"__name__": "bootstrap_test"}
        exec(compile(code, "bootstrap_template.py", "exec"), namespace)
        with tempfile.TemporaryDirectory() as directory:
            input_root = Path(directory)
            candidate = input_root / "provider-chosen-slug" / prepare.PAIRED_ROOT_REL
            candidate.mkdir(parents=True)
            status = candidate / "pair_status.json"
            status.write_bytes(b"sealed status\n")
            expected = {"pair_status.json": {"sha256": file_sha(status)}}
            self.assertEqual(namespace["find_paired_mount"](input_root, expected), candidate)
            second = input_root / "second-provider-slug" / prepare.PAIRED_ROOT_REL
            second.mkdir(parents=True)
            (second / "pair_status.json").write_bytes(status.read_bytes())
            with self.assertRaises(FileNotFoundError):
                namespace["find_paired_mount"](input_root, expected)
            (second / "pair_status.json").write_bytes(b"different status\n")
            status.write_bytes(b"tampered status\n")
            with self.assertRaises(FileNotFoundError):
                namespace["find_paired_mount"](input_root, expected)

    def test_staged_metadata_attaches_v3_notebook_without_checkpoint_upload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            v43 = root / "original.zip"
            parent = root / "parent.pt"
            v43.write_bytes(b"synthetic original")
            parent.write_bytes(b"synthetic parent")
            swe = root / "swe"
            fixed44 = root / "fixed44"
            swe.mkdir()
            fixed44.mkdir()
            for name in prepare.SWE_PROOF_FILES:
                (swe / name).write_bytes(b"synthetic SWE proof\n")
            for name in prepare.FIXED44_PROOF_FILES:
                (fixed44 / name).write_bytes(b"synthetic fixed44 proof\n")
            claims = {}
            for arm in ("control", "python"):
                path = root / arm / "prelaunch_contract_once.json"
                path.parent.mkdir()
                path.write_bytes(arm.encode())
                claims[arm] = path
            paired_files = {"pair_status.json": {"sha256": "e" * 64}}
            for arm in ("control", "python"):
                paired_files[f"{arm}/hybrid_stage1_{arm}_update_0250.pt"] = {
                    "sha256": ("c" if arm == "control" else "d") * 64, "bytes": 10}
            true_file_sha = prepare.file_sha
            def local_sha(path):
                if Path(path) == v43:
                    return prepare.PINNED_V43_ZIP_SHA
                if Path(path) == parent:
                    return prepare.PARENT_CHECKPOINT_SHA
                return true_file_sha(path)
            def claim(_, arm, __):
                return ({"claim_id": arm + "_0123456789abcdef"},
                        {"prelaunch_contract_once.json": claims[arm]})
            staged = root / "new-eval-stage"
            with (patch.object(prepare, "file_sha", side_effect=local_sha),
                  patch.object(prepare, "PARENT_CHECKPOINT_BYTES", parent.stat().st_size),
                  patch.object(prepare, "_verify_original_swe_baseline", return_value="a" * 64),
                  patch.object(prepare, "_verify_pretrain_raw_diagnostic", return_value="b" * 64),
                  patch.object(prepare, "_pair_checkpoints", return_value=paired_files),
                  patch.object(prepare, "_claim", side_effect=claim),
                  patch.object(prepare, "_pair_claims")):
                prepare.package(
                    paired_dir=root / "paired-small-proofs", v43_zip=v43,
                    parent_checkpoint=parent, swe_proof_dir=swe,
                    fixed44_proof_dir=fixed44,
                    control_claim=claims["control"], python_claim=claims["python"],
                    staged=staged)
            metadata = json.loads((staged / "kernel-metadata.json").read_text())
            manifest = json.loads((staged / "package-manifest.json").read_text())
            self.assertEqual(metadata["kernel_sources"], [prepare.PAIRED_KERNEL_ID])
            self.assertEqual(metadata["dataset_sources"], [prepare.ORIGINAL_DATASET])
            self.assertEqual(metadata["title"], prepare.KERNEL_ID.split("/", 1)[1])
            self.assertTrue(metadata["is_private"])
            self.assertTrue(metadata["enable_gpu"])
            self.assertEqual(manifest["stage1_files"], paired_files)
            ast.parse((staged / "bootstrap.py").read_text(encoding="utf-8"))

    def test_pair_status_receipt_and_plan_are_all_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            swe = "a" * 64
            fixed44 = "b" * 64
            pair = {"schema": "geocpu_stage1_paired_t4x2_result_v1",
                    "status": "BLOCK1_NEEDS_PAIRED_EVALUATION",
                    "stage_promotion": False, "absolute_stop_reached": False,
                    "process_exit_codes": {"control": 0, "python": 0},
                    "arm_results": {}}
            for arm in ("control", "python"):
                arm_dir = root / arm
                arm_dir.mkdir()
                checkpoint = arm_dir / f"hybrid_stage1_{arm}_update_0250.pt"
                checkpoint.write_bytes(arm.encode())
                digest = file_sha(checkpoint)
                binding = {
                    "schema": "geocpu_v43_python_gradual_stage_v1",
                    "curriculum_version": 2, "stage": 1, "arm": arm,
                    "parent_sha256": prepare.PARENT_CHECKPOINT_SHA,
                    "v43_zip_sha256": PINNED_V43_ZIP_SHA,
                    "stage_source_sha256": prepare.TRAIN_V2_SHA,
                    "source_pools_sha256": prepare.SOURCE_SHA["source_pools.py"],
                    "source_pools_v2_sha256": prepare.SOURCE_SHA["source_pools_v2.py"],
                    "stratified_schedule_sha256": prepare.SOURCE_SHA["stratified_exposure_v2.py"],
                    "hybrid_data_sha256": prepare.SOURCE_SHA["hybrid_data.py"],
                    "prelaunch_verifier_sha256": prepare.SOURCE_SHA["prelaunch_contract.py"],
                    "overlay_sha256": prepare.SOURCE_OVERLAY_SHA,
                    "precision_profile": "train_fp16_gradscaler_v1",
                    "python_actual_target_cap": 0.0 if arm == "control" else 0.01,
                    "original_swe200_baseline_complete_sha256": swe,
                    "original_swe200_baseline_precision_profile": "eval_bf16_native_historical_only",
                    "parent_fixed44_raw_diagnostic_complete_sha256": fixed44,
                    "common_boundary_plan_sha256": "e" * 64,
                    "seed": 20261103, "lr": 0.00001, "batch_size": 64,
                    "max_micro_tokens": 4096,
                    "ce_normalization": "sum_all_active_labels_per_update",
                    "compression_loss": "0.2_times_sum_per_example_over_64",
                    "microbatch_boundary": "paired_old_or_python_max_tokens",
                    "converted_sha256": prepare.PINNED_CONVERTED_SHA,
                    "swe_converted_sha256": prepare.SWE_SHA,
                    "swe_manifest_sha256": prepare.SWE_MANIFEST_SHA,
                    "swe_independent_audit_sha256": prepare.SWE_AUDIT_SHA,
                    "replay_complete_sha256": "4" * 64,
                    "gpu_names": ["NVIDIA Tesla T4"],
                }
                (arm_dir / "admitted_actual_target_plan.json").write_text(
                    json.dumps({"binding": binding}), encoding="utf-8")
                receipt = {
                    "schema": "geocpu_v43_python_gradual_checkpoint_receipt_v1",
                    "status": "BLOCK1_NEEDS_PAIRED_EVALUATION",
                    "accepted_updates": 250, "bytes": checkpoint.stat().st_size,
                    "sha256": digest,
                    "binding_sha256": sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest(),
                }
                (arm_dir / (checkpoint.name + ".receipt.json")).write_text(
                    json.dumps(receipt), encoding="utf-8")
                pair["arm_results"][arm] = {
                    "accepted_updates": 250, "checkpoint_sha256": digest,
                    "first_evaluation_barrier_reached": True, "stage_complete": False,
                }
            (root / "pair_status.json").write_text(json.dumps(pair), encoding="utf-8")
            files = prepare._pair_checkpoints(root, swe_sha=swe, fixed44_sha=fixed44)
            self.assertEqual(len(files), 7)
            for arm in ("control", "python"):
                (root / arm / f"hybrid_stage1_{arm}_update_0250.pt").unlink()
            # The local proof bundle can omit the multi-GB weights. The
            # Kaggle mount will hash both actual checkpoints at runtime.
            self.assertEqual(prepare._pair_checkpoints(root, swe_sha=swe,
                                                       fixed44_sha=fixed44), files)
            receipt_path = root / "python" / "hybrid_stage1_python_update_0250.pt.receipt.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["accepted_updates"] = 249
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checkpoint, receipt"):
                prepare._pair_checkpoints(root, swe_sha=swe, fixed44_sha=fixed44)

    def test_claim_pair_requires_two_arm_reservation_and_parent_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ids = ["control_0123456789abcdef", "python_0123456789abcdef"]
            for arm in ("control", "python"):
                (root / arm).mkdir()
                (root / arm / "quota.json").write_text(json.dumps({
                    "existing_unreturned_hold_device_seconds": 0,
                    "available_after_existing_holds_device_seconds": 3960,
                    "new_paired_reservation_device_seconds": 3960,
                    "total_allowance_counted": False,
                    "pay_to_scale_allowance_counted": False,
                }), encoding="utf-8")
                (root / arm / "jobs.json").write_text(json.dumps({
                    "jobs": [{"kernel_id": job, "status": "COMPLETE"}
                             for job in prepare.PRIOR_TERMINAL_JOBS]}), encoding="utf-8")
                (root / arm / "ledger.json").write_text(json.dumps({
                    "paired_kernel_id": prepare.KERNEL_ID,
                    "paired_claim_ids": sorted(ids),
                    "paired_reserved_device_seconds": 3960,
                }), encoding="utf-8")
            def claim(arm, claim_id):
                return {
                    "claim_id": claim_id, "run_id": arm + "_run_0123456789abcdef",
                    "gpu_names": ["Tesla T4"], "hard_stop_at_utc": "2026-11-04T10:00:00+00:00",
                    "official_quota_observed_at_utc": "2026-11-04T09:00:00+00:00",
                    "active_jobs_observed_at_utc": "2026-11-04T09:00:00+00:00",
                    "official_minimum_free_device_seconds": 3960,
                    "root_reserved_device_seconds": 1980,
                    "parent_swe200_baseline_complete_sha256": "a" * 64,
                    "parent_fixed44_raw_diagnostic_complete_sha256": "b" * 64,
                    "evidence": {
                        "official_quota_evidence": {"file": "quota.json", "sha256": "c" * 64},
                        "active_jobs_snapshot": {"file": "jobs.json", "sha256": "d" * 64},
                        "root_hold_ledger": {"file": "ledger.json"},
                    },
                }
            control = claim("control", ids[0])
            python = claim("python", ids[1])
            args = (root / "control" / "claim.json", root / "python" / "claim.json")
            prepare._pair_claims(control, python, *args, swe_sha="a" * 64,
                                 fixed44_sha="b" * 64)
            python["parent_fixed44_raw_diagnostic_complete_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "parent proofs"):
                prepare._pair_claims(control, python, *args, swe_sha="a" * 64,
                                     fixed44_sha="b" * 64)
            python["parent_fixed44_raw_diagnostic_complete_sha256"] = "b" * 64
            control["official_minimum_free_device_seconds"] = 3959
            python["official_minimum_free_device_seconds"] = 3959
            with self.assertRaisesRegex(ValueError, "two simultaneous"):
                prepare._pair_claims(control, python, *args, swe_sha="a" * 64,
                                     fixed44_sha="b" * 64)

    def test_exact_84_case_profile_comparison_rejects_prompt_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases_path = prepare.STUDY / "fixed_interim_diagnostic_44_v2.jsonl"
            runs = compare_fixed44._cases(cases_path)
            overlay_sha = sha256(json.dumps(prepare.SOURCE_OVERLAY_SHA,
                                            sort_keys=True).encode()).hexdigest()
            checkpoints = {"parent": prepare.PARENT_CHECKPOINT_SHA,
                           "control": "c" * 64, "python": "d" * 64}
            empty_ids_sha = sha256(b"[]").hexdigest()
            for arm in checkpoints:
                output = root / arm
                output.mkdir()
                with (output / "raw_answers_44_profiles.jsonl").open("x", encoding="utf-8") as sink:
                    for case, profile in runs:
                        prompt = PYTHON_MODE_CUE + case["prompt"] if profile == "python_cued" else case["prompt"]
                        row = {
                            "case_id": case["id"], "group": case["group"],
                            "profile": profile, "prompt": prompt,
                            "expected_answer": case["target"], "arm": arm,
                            "checkpoint_sha256": checkpoints[arm],
                            "cases_sha256": prepare.CASES_SHA,
                            "source_sha256": prepare.EVALUATOR_SHA,
                            "overlay_sha256": overlay_sha,
                            "precision_profile": prepare.PROFILE,
                            "v43_zip_sha256": PINNED_V43_ZIP_SHA,
                            "raw_answer": arm, "raw_answer_sha256": sha256(arm.encode()).hexdigest(),
                            "model_generated_text": arm,
                            "model_generated_text_sha256": sha256(arm.encode()).hexdigest(),
                            "generated_tokens": 0, "generated_token_ids": [],
                            "generated_token_ids_sha256": empty_ids_sha,
                            "generation_iteration_limit": 2048,
                            "ended_with_eos": True,
                        }
                        if case["group"] == "swe20":
                            row["old_original_evaluator_correct"] = (arm == "python")
                        sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                complete = {
                    "schema": "geocpu_v43_python_gradual_raw_diagnostic_v1",
                    "status": "COMPLETE", "arm": arm,
                    "checkpoint_kind": "parent" if arm == "parent" else "hybrid_250",
                    "checkpoint_sha256": checkpoints[arm],
                    "v43_zip_sha256": PINNED_V43_ZIP_SHA,
                    "cases_sha256": prepare.CASES_SHA,
                    "source_sha256": prepare.EVALUATOR_SHA,
                    "overlay_sha256": overlay_sha,
                    "precision_profile": prepare.PROFILE,
                    "profile_counts": compare_fixed44.COUNTS,
                    "max_generated_tokens": 2048, "training_updates": 0,
                    "raw_answers_sha256": file_sha(output / "raw_answers_44_profiles.jsonl"),
                }
                (output / "COMPLETE.json").write_text(json.dumps(complete), encoding="utf-8")
            result = compare_fixed44.compare(
                parent_dir=root / "parent", control_dir=root / "control",
                python_dir=root / "python", checkpoint_sha={k: checkpoints[k]
                                                          for k in ("control", "python")},
                cases_path=cases_path, output_dir=root)
            self.assertEqual(result["matched_case_profiles"], 84)
            self.assertEqual(result["metrics"]["swe20|native|old_original_evaluator_correct"]["cases"], 20)
            (root / "matched_rows_84.jsonl").unlink()
            (root / "comparison.json").unlink()
            raw = root / "control" / "raw_answers_44_profiles.jsonl"
            rows = [json.loads(line) for line in raw.read_text(encoding="utf-8").splitlines()]
            rows[0]["prompt"] += " changed"
            raw.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                           encoding="utf-8")
            complete_path = root / "control" / "COMPLETE.json"
            complete = json.loads(complete_path.read_text(encoding="utf-8"))
            complete["raw_answers_sha256"] = file_sha(raw)
            complete_path.write_text(json.dumps(complete), encoding="utf-8")
            with self.assertRaises(ValueError):
                compare_fixed44.compare(
                    parent_dir=root / "parent", control_dir=root / "control",
                    python_dir=root / "python", checkpoint_sha={k: checkpoints[k]
                                                              for k in ("control", "python")},
                    cases_path=cases_path, output_dir=root)


if __name__ == "__main__":
    unittest.main()
