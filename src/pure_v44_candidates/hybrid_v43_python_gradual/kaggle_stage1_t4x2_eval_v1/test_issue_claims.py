"""Offline read-only Kaggle claim issuance checks with an injected mock API."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import issue_claims
from prelaunch_contract import verify_prelaunch_contract

NOW = datetime(2026, 11, 4, 1, 0, tzinfo=timezone.utc)
SWE_SHA = "a" * 64
FIXED_SHA = "b" * 64


class MockKaggleApi:
    def __init__(self, *, minimum=25_000, v3_status="COMPLETE", recent_status="COMPLETE",
                 include_eval=False):
        self.quota_calls = 0
        self.list_calls = []
        self.status_calls = []
        self.quota = SimpleNamespace(
            minimum_time_allowed=timedelta(seconds=minimum),
            time_used=timedelta(seconds=2_000),
            time_reserved=timedelta(seconds=500),
            total_time_allowed=timedelta(seconds=1_000_000),
            pay_to_scale_time_allowed=timedelta(seconds=1_000_000),
        )
        self.statuses = {ref: "COMPLETE" for ref in issue_claims.PRIOR_TERMINAL_JOBS}
        self.statuses[issue_claims.PAIRED_KERNEL_ID] = v3_status
        self.statuses["geocpuresearch/recent-other"] = recent_status
        if include_eval:
            self.statuses[issue_claims.KERNEL_ID] = "COMPLETE"

    def quota_view(self):
        self.quota_calls += 1
        return SimpleNamespace(gpu_quota=self.quota, quota_refresh_time=None)

    def kernels_list_with_response(self, **kwargs):
        self.list_calls.append(kwargs)
        kernels = [SimpleNamespace(ref=ref, last_run_time=NOW - timedelta(hours=1))
                   for ref in issue_claims.PRIOR_TERMINAL_JOBS]
        kernels.append(SimpleNamespace(ref="geocpuresearch/recent-other",
                                       last_run_time=NOW - timedelta(minutes=3)))
        kernels.append(SimpleNamespace(ref="geocpuresearch/historical-provider-404",
                                       last_run_time=NOW - timedelta(days=3)))
        kernels.append(SimpleNamespace(ref=None, to_dict=lambda: {
            "title": "[Private Notebook]", "lastRunTime": "2010-04-01T00:00:00.000Z"}))
        if issue_claims.KERNEL_ID in self.statuses:
            kernels.append(SimpleNamespace(ref=issue_claims.KERNEL_ID,
                                           last_run_time=NOW - timedelta(minutes=1)))
        return SimpleNamespace(kernels=kernels if kwargs["page"] == 1 else [])

    def kernels_status(self, ref):
        self.status_calls.append(ref)
        if ref == "geocpuresearch/historical-provider-404":
            raise RuntimeError("provider 404")
        return SimpleNamespace(status=self.statuses[ref])


class IssueClaimsTests(unittest.TestCase):
    def call(self, root: Path, api: MockKaggleApi, *, existing=15_840,
             hard_stop=None, reconciliation=None):
        with (patch.object(issue_claims, "_verify_original_swe_baseline", return_value=SWE_SHA),
              patch.object(issue_claims, "_verify_pretrain_raw_diagnostic", return_value=FIXED_SHA)):
            return issue_claims.issue(
                api=api, output_dir=root / "new-eval-claims",
                hard_stop_at_utc=hard_stop or NOW + timedelta(hours=2),
                existing_holds_device_seconds=existing,
                reconciliation_evidence=reconciliation,
                swe_proof_dir=root / "swe", fixed44_proof_dir=root / "fixed44",
                clock=lambda: NOW)

    def test_distinct_claims_bind_verified_parent_and_v3_terminal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            api = MockKaggleApi()
            result = self.call(root, api)
            self.assertEqual(api.quota_calls, 1)
            self.assertEqual(len(api.list_calls), 1)
            self.assertNotIn("geocpuresearch/historical-provider-404", api.status_calls)
            self.assertEqual(set(api.status_calls), set(api.statuses))
            control_path = Path(result["control_claim"])
            python_path = Path(result["python_claim"])
            control = verify_prelaunch_contract(
                control_path, now=NOW + timedelta(seconds=1), expected_arm="control",
                required_run_seconds=1800, required_save_reserve_seconds=180)
            python = verify_prelaunch_contract(
                python_path, now=NOW + timedelta(seconds=1), expected_arm="python",
                required_run_seconds=1800, required_save_reserve_seconds=180)
            self.assertNotEqual(control["claim_id"], python["claim_id"])
            self.assertEqual(control["gpu_names"], ["Tesla T4"])
            self.assertEqual(control["precision_profile"], "eval_fp16_native_v1")
            self.assertEqual(control["parent_swe200_baseline_complete_sha256"], SWE_SHA)
            self.assertEqual(python["parent_fixed44_raw_diagnostic_complete_sha256"], FIXED_SHA)
            self.assertEqual(control["evidence"]["official_quota_evidence"]["sha256"],
                             python["evidence"]["official_quota_evidence"]["sha256"])
            ledger = json.loads((control_path.parent / "root_hold_ledger.json").read_text())
            self.assertEqual(ledger["paired_kernel_id"], issue_claims.KERNEL_ID)
            self.assertEqual(ledger["paired_reserved_device_seconds"], 3960)
            self.assertEqual(ledger["all_unreturned_holds_after_issue_device_seconds"], 19800)
            jobs = json.loads((control_path.parent / "active_jobs_snapshot.json").read_text())
            self.assertEqual(jobs["historical_over_24h_not_status_queried"], 1)
            self.assertEqual(jobs["ignored_provider_placeholders"], 1)

    def test_running_v3_and_recent_job_fail_without_claims(self):
        for kwargs in ({"v3_status": "RUNNING"}, {"recent_status": "RUNNING"}):
            with self.subTest(kwargs=kwargs), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with self.assertRaises(RuntimeError):
                    self.call(root, MockKaggleApi(**kwargs))
                self.assertFalse((root / "new-eval-claims").exists())

    def test_insufficient_minimum_quota_and_reused_eval_id_fail(self):
        for api in (MockKaggleApi(minimum=6_000), MockKaggleApi(include_eval=True)):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with self.assertRaises(RuntimeError):
                    self.call(root, api)
                self.assertFalse((root / "new-eval-claims").exists())

    def test_one_use_directory_and_full_hard_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "new-eval-claims").mkdir()
            api = MockKaggleApi()
            with self.assertRaises(FileExistsError):
                self.call(root, api)
            self.assertEqual(api.quota_calls, 0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "Hard stop"):
                self.call(root, MockKaggleApi(), hard_stop=NOW + timedelta(minutes=30))
            self.assertFalse((root / "new-eval-claims").exists())

    def test_reducing_prior_hold_requires_fresh_reconciliation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            api = MockKaggleApi()
            with self.assertRaisesRegex(ValueError, "fresh reconciliation"):
                self.call(root, api, existing=0)
            self.assertEqual(api.quota_calls, 0)
            reconciliation = root / "reconciliation.json"
            reconciliation.write_text(json.dumps({
                "schema": "geocpu_kaggle_hold_reconciliation_v1",
                "checked_utc": NOW.isoformat(),
                "unreturned_device_seconds": 0,
                "zero_hold_evidence": True,
            }), encoding="utf-8")
            result = self.call(root, api, existing=0, reconciliation=reconciliation)
            control = json.loads(Path(result["control_claim"]).read_text())
            self.assertIn("existing_holds_reconciliation", control["evidence"])


if __name__ == "__main__":
    unittest.main()
