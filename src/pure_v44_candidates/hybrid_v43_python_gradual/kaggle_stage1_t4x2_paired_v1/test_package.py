"""Static and fail-closed checks for the private paired Kaggle package."""
from __future__ import annotations

import ast
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest

import prepare


class PackageTests(unittest.TestCase):
    def test_all_bundled_source_is_pinned(self):
        for name, expected in prepare.SOURCE_SHA.items():
            self.assertEqual(prepare.file_sha(prepare.STUDY / name), expected, name)
        for name, expected in prepare.SOURCE_OVERLAY_SHA.items():
            self.assertEqual(prepare.file_sha(prepare.OVERLAY / name), expected, name)

    def test_bootstrap_template_is_syntactically_valid_and_uses_two_inputs(self):
        template = (prepare.HERE / "bootstrap_template.py").read_text(encoding="utf-8")
        self.assertEqual(template.count("__EMBEDDED__"), 1)
        ast.parse(template.replace("__EMBEDDED__", "{}"))
        self.assertEqual(prepare.ORIGINAL_DATASET,
                         "geocpuresearch/geocpu-v43-v44fix-paired-eval-20261103")
        self.assertEqual(prepare.BRIDGE_DATASET,
                         "geocpuresearch/geocpu-v43-python-bridge-verified-20261104")
        self.assertIn('env["CUDA_VISIBLE_DEVICES"] = str(index)', template)
        self.assertIn('"--stage", "1", "--arm", arm', template)
        self.assertIn('"accepted_updates") != 250', template)

    def test_missing_parent_proof_cannot_stage_a_kernel(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staged = root / "staged"
            with self.assertRaises(FileNotFoundError):
                prepare.package(
                    swe_proof_dir=root / "missing_swe",
                    fixed44_proof_dir=root / "missing_fixed44",
                    control_claim=root / "missing_control.json",
                    python_claim=root / "missing_python.json",
                    bridge_manifest=root / "missing_bridge.json",
                    staged=staged,
                    now=datetime(2026, 11, 4, tzinfo=timezone.utc),
                )
            self.assertFalse(staged.exists())

    def test_pair_claim_requires_full_two_arm_quota_and_distinct_claims(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ids = ["control_0123456789abcdef", "python_0123456789abcdef"]
            for arm, claim_id in zip(("control", "python"), ids):
                (root / arm).mkdir()
                ledger = {
                    "paired_kernel_id": prepare.KERNEL_ID,
                    "paired_claim_ids": sorted(ids),
                    "paired_reserved_device_seconds": 3960,
                }
                (root / arm / "ledger.json").write_text(json.dumps(ledger), encoding="utf-8")
                (root / arm / "jobs.json").write_text(json.dumps({
                    "jobs": [{"kernel_id": job, "status": "COMPLETE"}
                             for job in prepare.PRIOR_TERMINAL_JOBS]}), encoding="utf-8")
            def claim(arm, claim_id, minimum=3960):
                return {
                    "claim_id": claim_id, "run_id": arm + "_run_0123456789abcdef",
                    "gpu_names": ["Tesla T4"],
                    "hard_stop_at_utc": "2026-11-04T01:00:00+00:00",
                    "official_quota_observed_at_utc": "2026-11-04T00:30:00+00:00",
                    "active_jobs_observed_at_utc": "2026-11-04T00:30:00+00:00",
                    "official_minimum_free_device_seconds": minimum,
                    "root_reserved_device_seconds": 1980,
                    "evidence": {
                        "official_quota_evidence": {"sha256": "a" * 64},
                        "active_jobs_snapshot": {"sha256": "b" * 64, "file": "jobs.json"},
                        "root_hold_ledger": {"file": "ledger.json"},
                    },
                }
            control = claim("control", ids[0])
            python = claim("python", ids[1])
            prepare._pair_claims(control, python, root / "control" / "claim.json",
                                 root / "python" / "claim.json")
            python["claim_id"] = control["claim_id"]
            with self.assertRaises(ValueError):
                prepare._pair_claims(control, python, root / "control" / "claim.json",
                                     root / "python" / "claim.json")
            python = claim("python", ids[1], minimum=3959)
            control["official_minimum_free_device_seconds"] = 3959
            with self.assertRaises(ValueError):
                prepare._pair_claims(control, python, root / "control" / "claim.json",
                                     root / "python" / "claim.json")


if __name__ == "__main__":
    unittest.main()
