import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest

from prelaunch_contract import SCHEMA, verify_prelaunch_contract
from source_pools import file_sha


class PrelaunchContractTests(unittest.TestCase):
    def make(self, root, now, *, reserve=3960, active=False):
        observed = (now - timedelta(minutes=1)).isoformat()
        files = {
            "official_quota_evidence": {"source":"official_kaggle_free_quota",
                "observed_at_utc":observed,"minimum_free_device_seconds":5000},
            "root_hold_ledger":{"claim_id":"claim_0123456789abcdef",
                "reserved_device_seconds":reserve,"status":"reserved_once_unspent"},
            "active_jobs_snapshot":{"observed_at_utc":observed,
                "jobs":[{"status":"RUNNING"}] if active else []},
        }
        evidence = {}
        for name, item in files.items():
            path = root / (name+".json")
            path.write_text(json.dumps(item), encoding="utf-8")
            evidence[name] = {"file":path.name,"sha256":file_sha(path)}
        claim = {"schema":SCHEMA,"root_authorized_once":True,"arm":"python",
                 "run_id":"run_0123456789abcdef","claim_id":"claim_0123456789abcdef",
                 "official_quota_observed_at_utc":observed,
                 "active_jobs_observed_at_utc":observed,
                 "root_claim_created_at_utc":observed,
                 "hard_stop_at_utc":(now+timedelta(minutes=30)).isoformat(),
                 "official_minimum_free_device_seconds":5000,
                 "root_reserved_device_seconds":reserve,
                 "bounded_run_seconds":1800,"save_reserve_seconds":180,
                 "gpu_count":2,"evidence":evidence}
        contract = root / "prelaunch.json"
        contract.write_text(json.dumps(claim),encoding="utf-8")
        return contract

    def test_fresh_capacity_and_job_status(self):
        now = datetime(2026,11,3,15,0,tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            valid=self.make(root,now)
            self.assertEqual(verify_prelaunch_contract(valid,now=now,
                expected_arm="python",required_run_seconds=1800,
                required_save_reserve_seconds=180)["root_reserved_device_seconds"],3960)
            with self.assertRaises(ValueError):
                verify_prelaunch_contract(valid,now=now+timedelta(minutes=11),expected_arm="python")
            self.make(root,now,reserve=1800)
            with self.assertRaises(ValueError):
                verify_prelaunch_contract(valid,now=now,expected_arm="python")
            self.make(root,now,active=True)
            with self.assertRaises(ValueError):
                verify_prelaunch_contract(valid,now=now,expected_arm="python")
            jobs_path = root / "active_jobs_snapshot.json"
            jobs = json.loads(jobs_path.read_text(encoding="utf-8"))
            jobs["jobs"] = [{"status":"UNKNOWN"}]
            jobs_path.write_text(json.dumps(jobs), encoding="utf-8")
            claim = json.loads(valid.read_text(encoding="utf-8"))
            claim["evidence"]["active_jobs_snapshot"]["sha256"] = file_sha(jobs_path)
            valid.write_text(json.dumps(claim), encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_prelaunch_contract(valid,now=now,expected_arm="python")


if __name__ == "__main__":
    unittest.main()
