"""Small CPU-only checks for source identity, immutable resume and output closure."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import package_sources
import prelaunch_gate
import runner
from verify_outputs import ARMS, verify_complete, verify_records


def cases() -> list[dict]:
    return [{"id": f"case-{index:02d}", "prompt": f"question {index}",
             "target": "A", "bucket": "japanese32" if index < 32 else "logic16",
             "tier": 1 if index < 32 else 2} for index in range(64)]


def answer(index: int, case: dict, case_sha: str) -> dict:
    return {
        "case_index": index, "case_id": case["id"], "case_sha256": case_sha,
        "bucket": case["bucket"], "prompt": case["prompt"],
        "reference": case["target"], "raw_answer": "A",
        "generated_tokens": [3], "sampled_token_count": 1,
        "ended_with_eos": True, "stop_reason": "eos",
        "tool_invocations": 0, "error": None,
        "first_token_probe_error": None,
        "first_logit_probe_matches_generated_first_token": True,
        "origin": {"kind": "generated_this_job"},
    }


class PortableTests(unittest.TestCase):
    def test_package_pins_original_paths_sources_and_no_weights(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path, digest = package_sources.build(Path(temp) / "sources.zip")
            self.assertEqual(len(digest), 64)
            with zipfile.ZipFile(path) as archive:
                names = archive.namelist()
                self.assertFalse(any(name.endswith(".pt") for name in names))
                manifest = json.loads(archive.read("source_manifest.json"))
                self.assertEqual(
                    manifest["files_sha256"]["matched_pure_v44_readonly.py"],
                    "055d1ee8c6aecb48a323fc8e54762afce69a12443317bfb7ef398d0d9cfe2236")
                self.assertEqual(
                    manifest["files_sha256"]["partial_head_restore_readonly.py"],
                    "0c364ca0ae69c44cfc5a12e4077aa5cde569fd53e2b0065942add512480dfede")
                self.assertEqual(
                    manifest["asset_paths"]["parent"],
                    "GeoCPU_Frontier_CoTraining_v43/checkpoints/checkpoint_step_15000.pt")
                self.assertEqual(
                    manifest["asset_pins"]["child"],
                    [2_971_632_900, "28308d0445829b8bf137f801f98b7573f80e699fb00e97d56a15523158ee984a"])
                self.assertEqual(manifest["suite"], "mixed64")
                self.assertEqual(manifest["case_count"], 64)
                self.assertEqual(manifest["generation_limit_per_arm"], 2048)
                archive.extractall(Path(temp) / "unpacked")
            runner.verify_source_bundle(Path(temp) / "unpacked")
            draft = (Path(temp) / package_sources.ENTRYPOINT_NAME).read_text(encoding="utf-8")
            self.assertIn(digest, draft)
            self.assertIn("BUDGET_RECEIPT = None", draft)

    def test_official_quota_budget_and_existing_hold_fail_closed(self) -> None:
        terminal = {slug: "complete" for slug in prelaunch_gate.EXISTING_JOBS}
        outer = prelaunch_gate.select_budget(
            minimum_device_seconds=100_000, used_device_seconds=40_000,
            reserved_device_seconds=0, unreturned_hold_device_seconds=0,
            existing_job_statuses=terminal)
        self.assertEqual(outer, 28_200)
        self.assertEqual(prelaunch_gate.select_budget(
            minimum_device_seconds=108_000, used_device_seconds=40_000,
            reserved_device_seconds=10_000, unreturned_hold_device_seconds=10_000,
            existing_job_statuses=terminal), 22_200)
        held = dict(terminal)
        held[prelaunch_gate.EXISTING_JOBS[0]] = "running"
        with self.assertRaisesRegex(RuntimeError, "still running"):
            prelaunch_gate.select_budget(
                minimum_device_seconds=100_000, used_device_seconds=40_000,
                reserved_device_seconds=0, unreturned_hold_device_seconds=0,
                existing_job_statuses=held)
        with self.assertRaisesRegex(RuntimeError, "too small"):
            prelaunch_gate.select_budget(
                minimum_device_seconds=10_000, used_device_seconds=9_000,
                reserved_device_seconds=0, unreturned_hold_device_seconds=0,
                existing_job_statuses=terminal)
        with tempfile.TemporaryDirectory() as temp:
            submission = Path(temp) / "submission"
            with patch.object(prelaunch_gate, "fresh_official_receipt",
                              side_effect=RuntimeError("official endpoint unavailable")):
                with self.assertRaisesRegex(RuntimeError, "official endpoint unavailable"):
                    prelaunch_gate.prepare(
                        submission, kernel_id="geocpuresearch/readonly-probe",
                        source_dataset="geocpuresearch/pinned-source",
                        asset_dataset="geocpuresearch/pinned-assets",
                        hold_ledger=Path(temp) / "ledger.json")
            self.assertFalse(submission.exists())

    def test_root_hold_ledger_requires_fresh_hashed_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            evidence = root / "hold_evidence.json"
            evidence.write_text('{"job":"saved"}', encoding="utf-8")
            reconciliation = root / "reconciliation.json"
            reconciliation.write_text(json.dumps({
                "schema": "geocpu_kaggle_hold_reconciliation_v1",
                "checked_utc": datetime.now(timezone.utc).isoformat(),
                "unreturned_device_seconds": 7_200,
                "zero_hold_evidence": False,
            }), encoding="utf-8")
            ledger = root / "holds.json"
            record = {
                "schema": "geocpu_kaggle_unreturned_holds_v1",
                "reconciled_utc": datetime.now(timezone.utc).isoformat(),
                "reconciliation_evidence_path": reconciliation.name,
                "reconciliation_evidence_sha256": hashlib.sha256(
                    reconciliation.read_bytes()).hexdigest(),
                "holds": [{"id": "one", "unreturned_device_seconds": 7_200,
                           "evidence_path": evidence.name,
                           "evidence_sha256": hashlib.sha256(evidence.read_bytes()).hexdigest()}],
            }
            ledger.write_text(json.dumps(record), encoding="utf-8")
            self.assertEqual(prelaunch_gate.load_hold_ledger(ledger)[0], 7_200)
            evidence.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "evidence SHA differs"):
                prelaunch_gate.load_hold_ledger(ledger)
            record["holds"] = []
            ledger.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "totals differ"):
                prelaunch_gate.load_hold_ledger(ledger)
            reconciliation.write_text(json.dumps({
                "schema": "geocpu_kaggle_hold_reconciliation_v1",
                "checked_utc": datetime.now(timezone.utc).isoformat(),
                "unreturned_device_seconds": 0,
                "zero_hold_evidence": False,
            }), encoding="utf-8")
            record["reconciliation_evidence_sha256"] = hashlib.sha256(
                reconciliation.read_bytes()).hexdigest()
            ledger.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "no explicit reconciliation evidence"):
                prelaunch_gate.load_hold_ledger(ledger)

    def test_resume_keeps_existing_arm_and_rejects_profile_or_tier_change(self) -> None:
        rows = cases()
        contract = runner.make_contract(
            source_manifest_sha256="a" * 64,
            source_pins={"parent": "b" * 64},
            asset_paths={"parent": "original/path.pt"}, cases=rows)
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "result"
            self.assertTrue(runner.prepare_output(output, contract, {}))
            self.assertTrue((output / "PARTIAL.json").is_file())
            calls = []
            path = output / "arms" / "case_00_parent.json"
            def generate():
                calls.append(1)
                return answer(0, rows[0], contract["case_sha256"][0])
            first = runner.saved_or_generate(
                path, index=0, case=rows[0], case_sha256=contract["case_sha256"][0],
                generate=generate)
            original_sha = runner.sha256(path)
            second = runner.saved_or_generate(
                path, index=0, case=rows[0], case_sha256=contract["case_sha256"][0],
                generate=generate)
            self.assertEqual(first, second)
            self.assertEqual(calls, [1])
            self.assertEqual(runner.sha256(path), original_sha)
            for change in ("path", "sha", "profile", "tier"):
                changed = json.loads(json.dumps(contract))
                if change == "path":
                    changed["asset_paths"]["parent"] = "other/path.pt"
                elif change == "sha":
                    changed["source_pins"]["parent"] = "c" * 64
                elif change == "profile":
                    changed["decode"]["max_new_tokens"] = 1024
                else:
                    changed_rows = cases()
                    changed_rows[0]["tier"] = 9
                    changed["case_sha256"] = [runner.canonical_sha256(case) for case in changed_rows]
                with self.assertRaisesRegex(RuntimeError, "Resume contract differs"):
                    runner.prepare_output(output, changed, {})

    def test_complete_requires_all_64_three_arm_pairs_and_hashes(self) -> None:
        rows = cases()
        contract = runner.make_contract(
            source_manifest_sha256="a" * 64, source_pins={}, asset_paths={},
            cases=rows)
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "result"
            runner.prepare_output(output, contract, {})
            for index, case in enumerate(rows):
                pair = {"case_index": index, "case_id": case["id"],
                        "case_sha256": contract["case_sha256"][index],
                        "arms": {}}
                for arm in ARMS:
                    value = answer(index, case, contract["case_sha256"][index])
                    path = output / "arms" / f"case_{index:02d}_{arm}.json"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(value), encoding="utf-8")
                    pair["arms"][arm] = value
                path = output / "pairs" / f"case_{index:02d}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(pair), encoding="utf-8")
            with self.assertRaises(FileNotFoundError):
                verify_records(output)
            runner.atomic_json_new(output / "summary.json", {
                "case_count": 64, "arms": list(ARMS), "training_updates": 0,
                "errors": 0, "natural_eos": {arm: 64 for arm in ARMS},
                "stop_reasons": {arm: {"eos": 64} for arm in ARMS}})
            hashes = verify_records(output)
            runner.atomic_json_new(output / "result_manifest.json", {"files": hashes})
            runner.atomic_json_new(output / "COMPLETE.json", {
                "status": "complete", "case_count": 64, "arms": list(ARMS),
                "training_updates": 0,
                "result_manifest_sha256": runner.sha256(output / "result_manifest.json")})
            verify_complete(output)
            missing = output / "arms" / "case_63_child_parent_head.json"
            missing.unlink()
            with self.assertRaises(ValueError):
                verify_complete(output)


if __name__ == "__main__":
    unittest.main()
