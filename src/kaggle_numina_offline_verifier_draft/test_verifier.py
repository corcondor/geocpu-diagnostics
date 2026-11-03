"""Synthetic fixtures only. These tests never execute source-row Python."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import kaggle_entrypoint
import probe
import verifier as v


GOOD = "Start\n```python\nprint(1)\n```\n```output\n1\n```\nAfter\n```python\nprint(2)\n```\n```output\n2\n```\nDone"
MISMATCH = "Start\n```python\nprint(2)\n```\n```output\n3\n```\nDone"


class FakeWorker:
    instances = []

    def __init__(self, _path):
        self.rows = 0
        self.closed = False
        self.codes = []
        self.instances.append(self)

    def reset(self):
        self.codes = []
        return {"reset": True}

    def execute(self, code):
        self.codes.append(code)
        return {"success": True, "stdout": "1\n" if code == "print(1)\n" else "2\n",
                "stderr": "", "timeout": False, "truncated": False, "exception_type": None}

    def cpu_seconds(self):
        return 0.1

    def close(self):
        self.closed = True


def row(problem, solution):
    return {"problem": problem, "solution": solution,
            "messages": [{"role": "user", "content": problem}, {"role": "assistant", "content": solution}]}


def test_multiple_blocks_exact_receipt_and_mismatch():
    vocabulary = set("ユーザー: アシスタント\n" + "P1P2" + GOOD + MISMATCH)
    worker = FakeWorker(None)
    receipt, record = v.transform_verified(row("P1", GOOD), 0, worker, vocab=vocabulary,
                                            eval_questions=set(), eval_ids=set(), repo_id="public/repo", revision="commit")
    assert receipt["status"] == "accepted"
    assert len(receipt["tool_blocks"]) == 2
    assert all(item["recorded_stdout_sha256"] == item["observed_stdout_sha256"] for item in receipt["tool_blocks"])
    assert [span["train"] for span in record["target_segments"]] == [True, False, True, False, True]
    assert worker.codes == ["print(1)\n", "print(2)\n"]
    failed, rejected_record = v.transform_verified(row("P2", MISMATCH), 1, worker, vocab=vocabulary,
                                                    eval_questions=set(), eval_ids=set(), repo_id="public/repo", revision="commit")
    assert failed["reason"] == "executed_stdout_differs_from_public_record"
    assert failed["tool_blocks"][0]["status"] == "stdout_mismatch"
    assert rejected_record is None
    assert "stdout" not in failed
    assert "stdout" not in failed["tool_blocks"][0]


def synthetic_input(tmp_path, monkeypatch):
    root = tmp_path / "public_input"
    root.mkdir()
    script_dir = tmp_path / "scripts"
    script_dir.mkdir()
    (script_dir / "verifier.py").write_text("synthetic verifier marker", encoding="utf-8")
    (script_dir / "executor_worker_v44r1.py").write_text("synthetic worker marker", encoding="utf-8")
    monkeypatch.setattr(v, "HERE", script_dir)
    records = [row("P1", GOOD), row("P2", MISMATCH), row("P3", GOOD)]
    pq.write_table(pa.Table.from_pylist(records), root / "train.parquet")
    pq.write_table(pa.Table.from_pylist([row("Official test", GOOD)]), root / "test.parquet")
    source_lock = {"repos": {"numina": {"repo_id": "public/repo", "revision": "synthetic-commit",
                                               "files": {"train": v._pin(root / "train.parquet"),
                                                         "test": v._pin(root / "test.parquet")}}}}
    (script_dir / "public_source_lock_v44.json").write_text(json.dumps(source_lock), encoding="utf-8")
    monkeypatch.setattr(v, "LOCK_SHA256", v.sha_file(script_dir / "public_source_lock_v44.json"))
    characters = set("ユーザー: アシスタント\n" + "P1P2P3Official test" + GOOD + MISMATCH)
    (root / "vocab.json").write_text(json.dumps({"vocab": {char: i + 5 for i, char in enumerate(sorted(characters))}}, ensure_ascii=False), encoding="utf-8")
    eval_pins = [{"name": "synthetic_eval.jsonl", "bytes": 123, "sha256": hashlib.sha256(b"synthetic eval").hexdigest()}]
    exclusions = {"schema": "geocpu_evaluation_hash_index_v1", "evaluation_inputs": eval_pins,
                  "question_sha256": [], "id_sha256": []}
    (root / "exclusions.json").write_text(json.dumps(exclusions), encoding="utf-8")
    binding = {"schema": "geocpu_numina_hash_binding_v1", "source_lock_sha256": v.LOCK_SHA256,
               "vocabulary": v._pin(root / "vocab.json"),
               "exclusion_index": v._pin(root / "exclusions.json"),
               "anchor": {"bytes": 42, "sha256": hashlib.sha256(b"synthetic anchor hash only").hexdigest()},
               "evaluation_inputs": eval_pins}
    (root / "binding.json").write_text(json.dumps(binding), encoding="utf-8")
    names = {"train": "train.parquet", "test": "test.parquet", "vocab": "vocab.json",
             "exclusions": "exclusions.json", "binding": "binding.json"}
    return root, names, len(records)


def test_shard_resume_merge_and_tamper(tmp_path, monkeypatch):
    root, names, total = synthetic_input(tmp_path, monkeypatch)
    original_clock = v.time.monotonic
    clock = itertools.count()
    monkeypatch.setattr(v.time, "monotonic", lambda: float(next(clock)))
    first = v.run_shard(root, names, tmp_path / "session_1", start=0, end=total,
                        wall_budget_seconds=1.5, production=False, executor_factory=FakeWorker)
    monkeypatch.setattr(v.time, "monotonic", original_clock)
    assert first["complete"] is False
    assert first["next_index"] == 1
    second = v.run_shard(root, names, tmp_path / "session_2", start=0, end=total,
                         wall_budget_seconds=100, prior=tmp_path / "session_1",
                         production=False, executor_factory=FakeWorker)
    assert second["complete"] is True
    merged = v.merge_shards([tmp_path / "session_2"], tmp_path / "merged")
    assert merged["source_rows"] == total
    assert merged["counts"]["accepted"] == 2
    assert merged["counts"]["rejected"] == 1
    assert merged["stage_a_admissible"] is False
    receipt = json.loads((tmp_path / "merged" / "offline_verification_receipt.json").read_text(encoding="utf-8"))
    assert receipt["schema"] != "geocpu_public_tir_offline_verification_v1"
    first_chunk = second["chunks"][0]["receipts"]["name"]
    with (tmp_path / "session_2" / first_chunk).open("a", encoding="utf-8") as stream:
        stream.write("{}\n")
    with pytest.raises(ValueError, match="Pinned input differs"):
        v.merge_shards([tmp_path / "session_2"], tmp_path / "rejected_merge")


def test_probe_reports_booleans_without_secret_values(tmp_path):
    metadata = Path(probe.__file__).with_name("probe-kernel-metadata.json")
    report = probe.probe(metadata, connect=lambda: False, environ={"HF_TOKEN": "private-test-value"}, home=tmp_path)
    assert report["metadata"]["private_requested"] is True
    assert report["metadata"]["internet_disabled_requested"] is True
    assert report["metadata"]["gpu_disabled_requested"] is True
    assert report["observed"]["credential_named_environment_present"] is True
    assert report["observed"]["external_tcp_connect_succeeded"] is False
    assert "private-test-value" not in json.dumps(report)
    assert report["limits"]["stage_a_admissible"] is False


def test_production_entries_stop_before_inputs_or_worker(tmp_path, monkeypatch):
    calls = []
    def forbidden(*_args, **_kwargs):
        calls.append("forbidden")
        raise AssertionError("Must stop before reading inputs or launching a worker")
    monkeypatch.setattr(kaggle_entrypoint, "_checked_file", forbidden, raising=False)
    monkeypatch.setattr(kaggle_entrypoint, "run_shard", forbidden, raising=False)
    with pytest.raises(SystemExit, match="blocked in this draft"):
        kaggle_entrypoint.main()
    monkeypatch.setattr(v, "_input_context", forbidden)
    with pytest.raises(RuntimeError, match="blocked in this draft"):
        v.run_shard(tmp_path, {}, tmp_path / "never_created", start=0, end=1,
                    wall_budget_seconds=100, production=True, executor_factory=forbidden)
    assert calls == []
    assert not (tmp_path / "never_created").exists()


def test_merge_rejects_receipt_record_mismatch_with_updated_file_pin(tmp_path, monkeypatch):
    root, names, total = synthetic_input(tmp_path, monkeypatch)
    directory = tmp_path / "session"
    result = v.run_shard(root, names, directory, start=0, end=total,
                         wall_budget_seconds=100, production=False, executor_factory=FakeWorker)
    record_path = directory / result["chunks"][0]["records"]["name"]
    records = [json.loads(line) for line in record_path.read_text(encoding="utf-8").splitlines()]
    records[0]["provenance"]["source_target_sha256"] = "0" * 64
    record_path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
    result["chunks"][0]["records"]["pin"] = v._pin(record_path)
    v._write_json_atomic(directory / "run_manifest.json", result)
    with pytest.raises(ValueError, match="Accepted receipt and curriculum provenance differ"):
        v.merge_shards([directory], tmp_path / "merged_rejected")
