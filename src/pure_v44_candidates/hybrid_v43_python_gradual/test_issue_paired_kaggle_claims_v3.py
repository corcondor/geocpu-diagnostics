"""Offline checks for issuing v3 paired claims; no Kaggle network calls."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from issue_paired_kaggle_claims_v3 import (
    CONSERVATIVE_PRIOR_HOLDS_DEVICE_SECONDS, KERNEL_ID, PAIRED_DEVICE_SECONDS,
    PRIOR_FAILED_JOB, PRIOR_TERMINAL_JOBS, REQUIRED_PRIOR_STATUSES,
    V2_REJECTED_KERNEL_ID, issue,
)
from prelaunch_contract import verify_prelaunch_contract


NOW = datetime(2026, 11, 4, 1, 0, tzinfo=timezone.utc)


class MockKaggleApi:
    def __init__(self, *, minimum=20_000, used=2_000, reserved=500,
                 extra_status="COMPLETE", prior_status="COMPLETE", failed_status="ERROR",
                 v2_present=False, v2_on_second_page=False):
        self.quota_calls = 0
        self.list_calls = []
        self.status_calls = []
        self.quota = SimpleNamespace(
            minimum_time_allowed=timedelta(seconds=minimum),
            time_used=timedelta(seconds=used),
            time_reserved=timedelta(seconds=reserved),
            total_time_allowed=timedelta(seconds=1_000_000),
            pay_to_scale_time_allowed=timedelta(seconds=1_000_000),
        )
        self.statuses = {ref: prior_status for ref in PRIOR_TERMINAL_JOBS}
        self.statuses[PRIOR_FAILED_JOB] = failed_status
        self.statuses["geocpuresearch/unrelated-old-job"] = extra_status
        self.v2_present = v2_present
        self.v2_on_second_page = v2_on_second_page
        if v2_present or v2_on_second_page:
            self.statuses[V2_REJECTED_KERNEL_ID] = "ERROR"

    def quota_view(self):
        self.quota_calls += 1
        return SimpleNamespace(gpu_quota=self.quota, quota_refresh_time=None)

    def kernels_list_with_response(self, **kwargs):
        self.list_calls.append(kwargs)
        if kwargs["page"] == 1:
            refs = [PRIOR_TERMINAL_JOBS[0], PRIOR_FAILED_JOB,
                    "geocpuresearch/unrelated-old-job"]
            if self.v2_present:
                refs.append(V2_REJECTED_KERNEL_ID)
            return SimpleNamespace(kernels=[SimpleNamespace(ref=ref) for ref in refs] +
                [SimpleNamespace(ref=None, to_dict=lambda: {
                    "title": "[Private Notebook]", "lastRunTime": "2010-04-01T00:00:00.000Z"})],
                next_page_token="page-2" if self.v2_on_second_page else "")
        if self.v2_on_second_page:
            return SimpleNamespace(kernels=[SimpleNamespace(ref=V2_REJECTED_KERNEL_ID)],
                                   next_page_token="")
        return SimpleNamespace(kernels=[], next_page_token="")

    def kernels_status(self, ref):
        self.status_calls.append(ref)
        return SimpleNamespace(status=self.statuses[ref])


class IssuePairedClaimsTests(unittest.TestCase):
    def call(self, root, api, *, existing=CONSERVATIVE_PRIOR_HOLDS_DEVICE_SECONDS,
             hard_stop=None, reconciliation=None):
        return issue(api=api, output_dir=root / "new-paired-claims",
                     hard_stop_at_utc=hard_stop or NOW + timedelta(hours=2),
                     existing_holds_device_seconds=existing,
                     reconciliation_evidence=reconciliation,
                     clock=lambda: NOW)

    def test_issues_distinct_valid_claims_with_shared_hashed_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            api = MockKaggleApi()
            result = self.call(root, api)
            self.assertEqual(api.quota_calls, 1)
            self.assertEqual(len(api.list_calls), 1)
            self.assertTrue(all(call["mine"] and call["sort_by"] == "dateRun"
                                for call in api.list_calls))
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
            self.assertEqual(control["evidence"]["official_quota_evidence"]["sha256"],
                             python["evidence"]["official_quota_evidence"]["sha256"])
            self.assertEqual(control["evidence"]["active_jobs_snapshot"]["sha256"],
                             python["evidence"]["active_jobs_snapshot"]["sha256"])
            quota = json.loads((control_path.parent / "quota_evidence.json").read_text())
            self.assertEqual(quota["minimum_free_device_seconds"], 17500)
            self.assertEqual(quota["prior_hold_basis"], "conservative_unreturned_11880")
            self.assertFalse(quota["pay_to_scale_allowance_counted"])
            self.assertFalse(quota["total_allowance_counted"])
            ledger = json.loads((control_path.parent / "root_hold_ledger.json").read_text())
            self.assertEqual(ledger["paired_kernel_id"], KERNEL_ID)
            self.assertEqual(ledger["paired_reserved_device_seconds"], PAIRED_DEVICE_SECONDS)
            self.assertEqual(ledger["all_unreturned_holds_after_issue_device_seconds"], 15840)
            jobs = json.loads((control_path.parent / "active_jobs_snapshot.json").read_text())
            statuses = {job["kernel_id"]: job["status"] for job in jobs["jobs"]}
            self.assertEqual({ref: statuses[ref] for ref in REQUIRED_PRIOR_STATUSES},
                             REQUIRED_PRIOR_STATUSES)
            self.assertTrue(jobs["all_pages_read"])
            self.assertEqual(jobs["confirmed_absent_kernel_ids"], [V2_REJECTED_KERNEL_ID])
            self.assertNotIn(V2_REJECTED_KERNEL_ID, statuses)

    def test_insufficient_guaranteed_quota_fails_without_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            api = MockKaggleApi(minimum=10_000, used=2_000, reserved=500)
            with self.assertRaisesRegex(RuntimeError, "cannot cover"):
                self.call(root, api)
            self.assertFalse((root / "new-paired-claims").exists())
            self.assertEqual(api.list_calls, [])

    def test_nonterminal_job_and_incomplete_prior_fail_closed(self):
        for kwargs in ({"extra_status": "RUNNING"}, {"prior_status": "ERROR"},
                       {"failed_status": "COMPLETE"}):
            with self.subTest(kwargs=kwargs), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with self.assertRaises(RuntimeError):
                    self.call(root, MockKaggleApi(**kwargs))
                self.assertFalse((root / "new-paired-claims").exists())

    def test_rejected_v2_slug_must_be_absent(self):
        for kwargs in ({"v2_present": True}, {"v2_on_second_page": True}):
            with self.subTest(kwargs=kwargs), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with self.assertRaisesRegex(RuntimeError, "v2 kernel unexpectedly exists"):
                    self.call(root, MockKaggleApi(**kwargs))
                self.assertFalse((root / "new-paired-claims").exists())

    def test_existing_output_and_short_hard_stop_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "new-paired-claims").mkdir()
            api = MockKaggleApi()
            with self.assertRaises(FileExistsError):
                self.call(root, api)
            self.assertEqual(api.quota_calls, 0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "hard stop"):
                self.call(root, MockKaggleApi(), hard_stop=NOW + timedelta(minutes=30))
            self.assertFalse((root / "new-paired-claims").exists())

    def test_reducing_conservative_prior_hold_requires_fresh_reconciliation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "fresh reconciliation"):
                self.call(root, MockKaggleApi(), existing=0)
            evidence = root / "reconciled.json"
            evidence.write_text(json.dumps({
                "schema": "geocpu_kaggle_hold_reconciliation_v1",
                "checked_utc": NOW.isoformat(),
                "unreturned_device_seconds": 0,
                "zero_hold_evidence": True,
            }), encoding="utf-8")
            result = self.call(root, MockKaggleApi(), existing=0, reconciliation=evidence)
            claim_path = Path(result["control_claim"])
            claim = json.loads(claim_path.read_text())
            self.assertIn("existing_holds_reconciliation", claim["evidence"])
            quota = json.loads((claim_path.parent / "quota_evidence.json").read_text())
            self.assertEqual(quota["prior_hold_basis"], "fresh_reconciliation")


if __name__ == "__main__":
    unittest.main()
