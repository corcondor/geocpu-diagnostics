"""GeoCPU evaluation logs that keep progress, scores and missing measurements apart.

This module only records results produced by a caller or reads existing files.
It neither loads checkpoints nor trains/evaluates a model by itself. The CLI
``run`` starts only the explicit command supplied by its caller; ``watch`` is
strictly read-only. Python's standard library is sufficient.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Iterable, Mapping, TextIO
import uuid


SCHEMA_VERSION = 1
RESULT_STATUSES = {"scored", "invalid", "unmeasured", "error"}
IDENTITY_FIELDS = ("suite", "mode", "checkpoint_id", "checkpoint_sha256",
                   "dataset_sha256", "scorer_sha256", "runtime_sha256",
                   "budget", "seed", "metric")
COMPARISON_FIELDS = ("suite", "mode", "dataset_sha256", "scorer_sha256",
                     "runtime_sha256", "budget", "seed", "metric")
PYTHON_METRICS = ("explicit_final_numeric_exact", "terminal_explicit_exact",
                  "python_execution_success", "tool_and_terminal_exact",
                  "python_only_tool_and_terminal_exact", "ended_with_eos")


def utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    """Replace one logger-owned status file; never write a source evaluation file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp",
                                     dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(canonical(value) + b"\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _count(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _check_identity(identity: Mapping[str, Any]) -> dict:
    missing = [field for field in IDENTITY_FIELDS if field not in identity]
    if missing:
        raise ValueError(f"identity fields missing: {missing}")
    copied = json.loads(canonical(dict(identity)))
    for field in ("suite", "mode", "checkpoint_id", "metric"):
        if not isinstance(copied[field], str) or not copied[field].strip():
            raise ValueError(f"identity.{field} must be nonempty")
    for field in ("checkpoint_sha256", "dataset_sha256", "scorer_sha256", "runtime_sha256"):
        if not isinstance(copied[field], str) or not re.fullmatch(r"[0-9a-f]{64}", copied[field]):
            raise ValueError(f"identity.{field} requires the actual lowercase SHA-256")
    if not isinstance(copied["budget"], dict) or not copied["budget"]:
        raise ValueError("identity.budget must record the actual inference settings")
    return copied


class RunLogger:
    """One immutable evaluation binding, one append-only event stream.

    ``invalid`` means a model-produced answer rejected by the original scoring
    rule. It counts in the accuracy denominator, separately from valid wrong
    answers. Missing measurements and evaluator errors are excluded from that
    denominator and are always shown. They prevent an unqualified 'complete'.
    A scored result must be the original scorer's boolean; this logger never
    extracts answers, interprets a tool's stdout, or changes a scoring rule.
    """

    def __init__(self, out_dir: str | Path, identity: Mapping[str, Any], *, total: int,
                 resume: bool = False, interval_s: float = 1.0,
                 heartbeat_s: float = 5.0, stream: TextIO | None = None):
        self.out_dir = Path(out_dir)
        self.identity = _check_identity(identity)
        self.total = _count(total, "total")
        if not 0 < heartbeat_s <= 10:
            raise ValueError("heartbeat_s must be in (0, 10]")
        if interval_s < 1:
            raise ValueError("display interval must be at least 1 second")
        self.interval_s, self.heartbeat_s = interval_s, heartbeat_s
        self.stream = sys.stdout if stream is None else stream
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None
        self.last_render = -math.inf
        self.begin_mono = time.monotonic()
        self.begin_wall = time.time()
        self.closed = False
        self.results: dict[str, dict] = {}
        self.active: dict[str, float] = {}
        self.state = "running"
        self.progress_hint: dict | None = None
        self.writer_error: BaseException | None = None
        self.sequence = 0
        self.binding = {"schema_version": SCHEMA_VERSION, "identity": self.identity,
                        "total": total}
        self.config_sha256 = digest(self.binding)
        manifest_path = self.out_dir / "run.json"
        events_path = self.out_dir / "events.jsonl"
        if manifest_path.exists():
            if not resume:
                raise FileExistsError(f"result directory already exists; use resume: {self.out_dir}")
            saved = json.loads(manifest_path.read_text(encoding="utf-8"))
            if saved.get("binding") != self.binding or saved.get("config_sha256") != self.config_sha256:
                raise ValueError("cannot resume: checkpoint, dataset, scorer or inference settings changed")
            self.run_id = saved["run_id"]
            self.begin_wall = float(saved["started_unix"])
            self._load(events_path)
        else:
            if resume:
                raise FileNotFoundError("resume requires an existing run.json")
            if self.out_dir.exists() and any(self.out_dir.iterdir()):
                raise FileExistsError("logger output must be a new/empty directory")
            self.out_dir.mkdir(parents=True, exist_ok=True)
            self.run_id = uuid.uuid4().hex
            saved = {"binding": self.binding, "config_sha256": self.config_sha256,
                     "run_id": self.run_id, "started_utc": utc(),
                     "started_unix": self.begin_wall}
            # Exclusive create prevents an accidental second writer at startup.
            with manifest_path.open("x", encoding="utf-8") as f:
                json.dump(saved, f, ensure_ascii=False, sort_keys=True)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
        # flock is not reliable on every Drive FUSE mount. The lock is scoped to
        # this logger's outputs only and is never used to kill another process.
        self.lock_path = self.out_dir / "logger_active.json"
        try:
            with self.lock_path.open("x", encoding="utf-8") as f:
                json.dump({"pid": os.getpid(), "run_id": self.run_id,
                           "started_utc": utc()}, f)
        except FileExistsError as error:
            raise RuntimeError("logger output has an active/stale writer lock; use a new directory "
                               "or verify the old logger has exited before removing only its lock") from error
        self.events = events_path.open("ab", buffering=0)
        if resume:
            # A started case without a durable result is pending, never wrong.
            interrupted = sorted(self.active)
            self.active.clear()
            self.state = "running"
            self._emit("run_resumed", {"pending_interrupted_cases": interrupted})
        else:
            self._emit("run_started", {})

    def _load(self, path: Path) -> None:
        if not path.is_file():
            raise ValueError("run manifest exists but events are missing")
        with path.open("rb") as f:
            for number, line in enumerate(f, 1):
                if not line.endswith(b"\n"):
                    raise ValueError(f"incomplete event tail at line {number}; preserve it and use a new output")
                event = json.loads(line)
                if event.get("config_sha256") != self.config_sha256 or event.get("run_id") != self.run_id:
                    raise ValueError("events contain a different run binding")
                if event.get("sequence") != self.sequence + 1:
                    raise ValueError("event sequence is incomplete or duplicated")
                self.sequence += 1
                kind = event.get("event")
                if kind == "case_started":
                    self.active[event["case_id"]] = float(event["timestamp_unix"])
                elif kind == "case_result":
                    item = event["result"]
                    key = item["case_id"]
                    if key in self.results:
                        raise ValueError("duplicate scored case in durable log")
                    self.results[key] = item
                    self.active.pop(key, None)
                elif kind == "progress":
                    self.progress_hint = event["progress"]
                elif kind == "run_finished":
                    self.state = event["state"]

    def _ensure_open(self) -> None:
        if self.closed:
            raise RuntimeError("logger is closed")
        if self.writer_error is not None:
            raise RuntimeError("background logger failed") from self.writer_error

    def _emit(self, event: str, payload: dict) -> None:
        with self.lock:
            self._ensure_open()
            self.sequence += 1
            item = {"schema_version": SCHEMA_VERSION, "sequence": self.sequence,
                    "event": event, "run_id": self.run_id,
                    "config_sha256": self.config_sha256, "timestamp_utc": utc(),
                    "timestamp_unix": time.time(), **payload}
            self.events.write(canonical(item) + b"\n")
            os.fsync(self.events.fileno())
            atomic_json(self.out_dir / "status.json", self.snapshot())
            self.render()

    def snapshot(self) -> dict:
        with self.lock:
            counts = Counter(row["status"] for row in self.results.values())
            correct = sum(row["status"] == "scored" and row["correct"]
                          for row in self.results.values())
            valid_incorrect = counts["scored"] - correct
            answered = counts["scored"] + counts["invalid"]
            processed = len(self.results)
            elapsed = max(0.0, time.time() - self.begin_wall)
            eta = elapsed * (self.total - processed) / processed if processed else None
            metric_totals: dict[str, float | int] = {}
            metric_observations: dict[str, int] = {}
            for row in self.results.values():
                for name, value in row["metrics"].items():
                    if isinstance(value, (int, float)) and math.isfinite(value):
                        metric_totals[name] = metric_totals.get(name, 0) + value
                        metric_observations[name] = metric_observations.get(name, 0) + 1
            return {"schema_version": SCHEMA_VERSION, "run_id": self.run_id,
                    "identity": self.identity, "config_sha256": self.config_sha256,
                    "state": self.state, "updated_utc": utc(), "total": self.total,
                    "processed": processed, "answered": answered,
                    "correct": correct, "incorrect": valid_incorrect,
                    "invalid": counts["invalid"], "unmeasured": counts["unmeasured"],
                    "errors": counts["error"],
                    "accuracy": correct / answered if answered else None,
                    "coverage": answered / self.total if self.total else None,
                    "pending": self.total - processed,
                    "active_cases": sorted(self.active), "progress_hint": self.progress_hint,
                    "metric_totals": metric_totals,
                    "metric_observations": metric_observations,
                    "elapsed_seconds": round(elapsed, 3),
                    "eta_seconds": round(eta, 3) if eta is not None else None,
                    "score_note": "accuracy=correct/(scored+invalid); unmeasured/error excluded and shown; "
                                  "progress hints never supply scores"}

    def render(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last_render < self.interval_s:
            return
        self.last_render = now
        s = self.snapshot()
        rate = "--" if s["accuracy"] is None else f"{100 * s['accuracy']:.2f}%"
        eta = "--" if s["eta_seconds"] is None else f"{s['eta_seconds'] / 60:.1f}m"
        case = s["active_cases"][0] if s["active_cases"] else "--"
        extra = ""
        for key, label in (("python_execution_success", "Python実行成功"),
                           ("explicit_final_numeric_exact", "最終数値一致"),
                           ("tool_and_terminal_exact", "ツール＋最終一致")):
            if key in s["metric_totals"]:
                denominator = s["metric_observations"][key]
                extra += f" | {label} {s['metric_totals'][key]}/{denominator}"
        print(f"[{utc()}] {self.identity['suite']} | {self.identity['checkpoint_id']} | "
              f"{self.identity['mode']} | {self.identity['metric']}: "
              f"{s['correct']}/{s['answered']} ({rate}) | "
              f"processed={s['processed']}/{self.total} | "
              f"invalid={s['invalid']} unmeasured={s['unmeasured']} errors={s['errors']} | "
              f"elapsed={s['elapsed_seconds'] / 60:.1f}m ETA={eta} | "
              f"case={case} | {s['state']}{extra}", file=self.stream, flush=True)

    def start_case(self, case_id: str) -> bool:
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("case_id must be a nonempty stable ID")
        with self.lock:
            self._ensure_open()
            if case_id in self.results or case_id in self.active:
                return False
            if len(self.results) + len(self.active) >= self.total:
                raise ValueError("case starts exceed declared total")
            self.active[case_id] = time.time()
            self._emit("case_started", {"case_id": case_id})
            return True

    def result(self, case_id: str, *, correct: bool | None = None,
               status: str = "scored", metrics: Mapping[str, Any] | None = None,
               raw_ref: str | None = None, elapsed_s: float | None = None,
               detail: str | None = None) -> bool:
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("case_id must be a nonempty stable ID")
        if status not in RESULT_STATUSES:
            raise ValueError(f"unknown result status: {status}")
        if status == "scored" and type(correct) is not bool:
            raise ValueError("scored result requires the original scorer's boolean")
        if status != "scored" and correct is not None:
            raise ValueError("invalid/unmeasured/error results must not invent correctness")
        if elapsed_s is not None and (not math.isfinite(elapsed_s) or elapsed_s < 0):
            raise ValueError("elapsed_s must be nonnegative and finite")
        item = {"case_id": case_id, "status": status, "correct": correct,
                "metrics": dict(metrics or {}), "raw_ref": raw_ref, "detail": detail}
        canonical(item)  # Check serializability, NaN and infinity before changing state.
        with self.lock:
            self._ensure_open()
            if case_id in self.results:
                old = dict(self.results[case_id])
                old.pop("elapsed_seconds", None)
                if old != item:
                    raise ValueError(f"contradictory duplicate result for {case_id}")
                return False
            if len(self.results) >= self.total:
                raise ValueError("result count exceeds declared total")
            if elapsed_s is None and case_id in self.active:
                elapsed_s = max(0.0, time.time() - self.active[case_id])
            item["elapsed_seconds"] = elapsed_s
            self.results[case_id] = item
            self.active.pop(case_id, None)
            self._emit("case_result", {"result": item})
            return True

    def progress(self, processed: int, total: int, *, source: str = "caller") -> None:
        """Record a displayed counter without assuming that it is a score."""
        _count(processed, "processed")
        _count(total, "total")
        if processed > total:
            raise ValueError("progress exceeds its total")
        with self.lock:
            self.progress_hint = {"processed": processed, "total": total, "source": source}
            self._emit("progress", {"progress": self.progress_hint})

    def heartbeat(self) -> None:
        self._emit("heartbeat", {})

    def _heartbeat_loop(self) -> None:
        while not self.stop.wait(self.heartbeat_s):
            try:
                self.heartbeat()
            except BaseException as error:
                self.writer_error = error
                self.stop.set()
                print(f"LOGGER_ERROR {type(error).__name__}: {error}", file=self.stream, flush=True)
                return

    def start_heartbeat(self) -> None:
        if self.thread is None:
            self.thread = threading.Thread(target=self._heartbeat_loop,
                                           name="geocpu-eval-heartbeat", daemon=True)
            self.thread.start()

    def finish(self, *, error: str | None = None, interrupted: bool = False) -> dict:
        self.stop.set()
        if self.thread is not None and threading.current_thread() is not self.thread:
            self.thread.join(timeout=self.heartbeat_s + 1)
        with self.lock:
            self._ensure_open()
            s = self.snapshot()
            if interrupted:
                self.state = "interrupted"
            elif error or s["errors"]:
                self.state = "failed"
            elif s["processed"] != self.total or self.active:
                self.state = "incomplete"
            elif s["unmeasured"]:
                self.state = "complete_with_unmeasured"
            else:
                self.state = "complete"
            self._emit("run_finished", {"state": self.state, "error": error})
            self.render(force=True)
            result = self.snapshot()
            self._close()
            return result

    def _close(self) -> None:
        if not self.closed:
            self.events.close()
            self.closed = True
            if self.lock_path.exists():
                lock = json.loads(self.lock_path.read_text(encoding="utf-8"))
                if lock.get("run_id") == self.run_id and lock.get("pid") == os.getpid():
                    self.lock_path.unlink()

    def __enter__(self) -> "RunLogger":
        self.start_heartbeat()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if not self.closed:
            try:
                self.finish(error=str(exc) if exc is not None else None,
                            interrupted=bool(exc_type and issubclass(exc_type, KeyboardInterrupt)))
            finally:
                self._close()
        return False


class TextProgressParser:
    """Recognize observed progress syntax, never infer correct from n/N alone."""

    def feed(self, line: str) -> dict | None:
        tier = re.search(r"\[TIER ([1-8])/(?:7|8)\]", line)
        if tier:
            return {"kind": "stage", "suite": f"T{tier.group(1)}", "raw": line.rstrip()}
        match = re.search(r"PYTHON_TIR_STEP_(\d+)_PROGRESS\s+(\d+)/(\d+)", line)
        if match:
            return {"kind": "progress", "suite": "Python120", "checkpoint_id": f"step_{match[1]}",
                    "processed": int(match[2]), "total": int(match[3]), "accuracy": None}
        match = re.search(r"\bEXTENDED_CASE\s+(\d+)/(\d+)\s+(\S+)\s+(\S+)", line)
        if match:
            return {"kind": "progress", "suite": "extended", "processed": int(match[1]),
                    "total": int(match[2]), "case_id": match[4], "accuracy": None}
        match = re.search(r"\bProgress:\s*(\d+)\s*/\s*(\d+)", line)
        if match:
            result = {"kind": "progress", "processed": int(match[1]), "total": int(match[2]),
                      "accuracy": None, "raw": line.rstrip()}
            # Explicit scorer counts can be displayed, but are not per-case records.
            score = re.search(r"\bAcc:\s*[\d.]+%\s*\((\d+)\s*/\s*(\d+)\)", line)
            exact = re.search(r"\bCorrect:\s*(\d+)", line)
            if score and int(score[2]) == result["processed"] and int(score[1]) <= int(score[2]):
                result.update(correct=int(score[1]), answered=int(score[2]),
                              accuracy=int(score[1]) / int(score[2]) if int(score[2]) else None)
            elif exact and int(exact[1]) <= result["processed"]:
                result.update(correct=int(exact[1]), answered=result["processed"],
                              accuracy=int(exact[1]) / result["processed"] if result["processed"] else None)
            return result
        return None


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


def _tail(path: Path, max_bytes: int = 12000) -> str:
    with path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - max_bytes))
        data = f.read(max_bytes)
    return data.decode("utf-8", errors="replace")


class ExistingSuiteMonitor:
    """Read-only v47 suite monitor. It does not create files under ``root``.

    The current Python evaluator saves each result atomically, then only prints
    progress every ten cases. Reading those per-case partials makes its existing
    scores visible without editing or restarting the evaluator.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)
        self.parser = TextProgressParser()
        self.cache: dict[str, tuple[tuple[int, int], dict]] = {}

    def _json(self, path: Path) -> dict:
        stat = path.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
        hit = self.cache.get(str(path))
        if hit and hit[0] == signature:
            return hit[1]
        data = _read_json(path)
        self.cache[str(path)] = (signature, data)
        return data

    def snapshot(self) -> dict:
        result: dict = {"root": str(self.root), "updated_utc": utc(), "source_writes": 0,
                        "python": [], "extended": [], "tiers": [], "read_errors": []}
        for path in (self.root / "suite_status.json", self.root / "status.json"):
            if path.is_file():
                try:
                    result[path.name] = self._json(path)
                except (OSError, ValueError) as error:
                    result["read_errors"].append(f"{path.name}: {type(error).__name__}")
        roots = [self.root]
        if (self.root / "python_120").is_dir():
            roots.append(self.root / "python_120")
        for parent in roots:
            expected_total = None
            selection_sha = None
            expected_ids: set[str] | None = None
            selection = parent / "selection_120.json"
            if selection.is_file():
                try:
                    selected = self._json(selection)
                    expected_ids = {row["id"] for row in selected["cases"]}
                    expected_total = len(selected["cases"])
                    if len(expected_ids) != expected_total:
                        raise ValueError("duplicate selection IDs")
                    selection_sha = selected["provenance"]["selection_sha256"]
                except (OSError, ValueError, KeyError, TypeError) as error:
                    result["read_errors"].append(f"{selection.name}: {type(error).__name__}")
            for folder in sorted(parent.glob("partials_step_*")):
                if not folder.is_dir():
                    continue
                groups: dict[str, dict] = {}
                seen: dict[tuple[str, str], dict] = {}
                for path in sorted(folder.glob("*.json")):
                    try:
                        saved = self._json(path)
                        binding = {key: saved[key] for key in
                                   ("checkpoint_sha256", "selection_sha256", "max_tokens")}
                        if selection_sha is not None and binding["selection_sha256"] != selection_sha:
                            raise ValueError("partial dataset binding differs from selection")
                        key = digest(binding)
                        row = saved["result"]
                        if expected_ids is not None and row["id"] not in expected_ids:
                            raise ValueError("partial case is absent from selection")
                        if any(type(row.get(metric)) is not bool for metric in PYTHON_METRICS):
                            raise ValueError("Python partial lacks explicit scorer booleans")
                        pair = (key, row["id"])
                        if pair in seen:
                            if seen[pair] != row:
                                raise ValueError("contradictory duplicate partial")
                            continue
                        seen[pair] = row
                        group = groups.setdefault(key, {"step": folder.name.removeprefix("partials_step_"),
                            "binding": binding, "processed": 0, "total": expected_total,
                            "metric_counts": dict.fromkeys(PYTHON_METRICS, 0),
                            "python_invocations": 0, "successful_python_invocations": 0,
                            "source": "saved_per_case_scores"})
                        group["processed"] += 1
                        for metric in PYTHON_METRICS:
                            group["metric_counts"][metric] += int(row[metric])
                        for metric in ("python_invocations", "successful_python_invocations"):
                            group[metric] += _count(row.get(metric, 0), metric)
                    except (OSError, ValueError, KeyError, TypeError) as error:
                        result["read_errors"].append(f"{path.name}: {type(error).__name__}: {error}")
                if len(groups) > 1:
                    result["read_errors"].append(f"{folder.name}: multiple checkpoint/dataset/budget bindings; kept separate")
                result["python"].extend(groups.values())
        ext = self.root / "extended" / "cases"
        if ext.is_dir():
            groups: dict[str, dict] = {}
            for path in sorted(ext.glob("*.json")):
                try:
                    item = self._json(path)
                    score = item["score"]
                    key = digest({"binding": item["binding"], "group": item["group"]})
                    summary = groups.setdefault(key, {"binding": item["binding"], "group": item["group"],
                        "processed": 0, "answered": 0, "correct": 0, "unmeasured": 0, "errors": 0})
                    summary["processed"] += 1
                    if score.get("status") == "unmeasured":
                        summary["unmeasured"] += 1
                    elif type(score.get("correct")) is bool:
                        summary["answered"] += 1
                        summary["correct"] += int(score["correct"])
                    elif score.get("status") in ("passed", "failed"):
                        summary["answered"] += 1
                        summary["correct"] += int(score["status"] == "passed")
                    else:
                        summary["errors"] += 1
                except (OSError, ValueError, KeyError, TypeError) as error:
                    result["read_errors"].append(f"{path.name}: {type(error).__name__}")
            result["extended"] = list(groups.values())
        tiers = self.root / "original_tiers"
        if not tiers.is_dir() and any(self.root.glob("tier_*.log")):
            tiers = self.root
        if tiers.is_dir():
            for number in range(1, 8):
                path = tiers / f"tier_{number}.log"
                if path.is_file():
                    try:
                        lines = _tail(path).splitlines()
                        parsed = [self.parser.feed(line) for line in lines]
                        progress = next((p for p in reversed(parsed) if p and p["kind"] == "progress"), None)
                        result["tiers"].append({"suite": f"T{number}", "last_progress": progress,
                            "last_lines": lines[-3:], "log_mtime_unix": path.stat().st_mtime})
                    except OSError as error:
                        result["read_errors"].append(f"{path.name}: {type(error).__name__}")
            tier8 = tiers / "tier8_raw.json"
            if tier8.is_file():
                try:
                    raw = self._json(tier8)
                    result["tiers"].append({"suite": "T8", "profiles": {k: {
                        key: v[key] for key in ("n", "tir_code_rate", "narrative_rate", "python_invocation_count")
                        if key in v} for k, v in raw.items() if isinstance(v, dict)},
                        "score_note": "T8 reports format/tool-use rates, not numerical answer accuracy"})
                except (OSError, ValueError, TypeError) as error:
                    result["read_errors"].append(f"tier8_raw.json: {type(error).__name__}")
        logs = list(self.root.glob("suite_*.log"))
        if logs:
            latest = max(logs, key=lambda p: p.stat().st_mtime)
            try:
                tail = _tail(latest)
                result["latest_log"] = str(latest)
                result["seconds_since_log_update"] = round(max(0, time.time() - latest.stat().st_mtime), 1)
                result["last_lines"] = tail.splitlines()[-3:]
                parsed = [self.parser.feed(line) for line in tail.splitlines()]
                result["last_progress_hint"] = next((p for p in reversed(parsed) if p), None)
            except OSError as error:
                result["read_errors"].append(f"{latest.name}: {type(error).__name__}")
        return result


def render_existing(snapshot: Mapping[str, Any], *, stream: TextIO | None = None) -> None:
    output = sys.stdout if stream is None else stream
    lines = [f"[{snapshot['updated_utc']}] 読取専用の進捗表示（元の評価は継続）"]
    status = snapshot.get("suite_status.json", snapshot.get("status.json", {}))
    if status:
        lines.append(f"状態: {status.get('state', 'unknown')} | stages={status.get('stages', {})}")
    for item in snapshot["python"]:
        n = item["processed"]
        counts = item["metric_counts"]
        accuracy = 100 * counts["explicit_final_numeric_exact"] / n if n else None
        percent = "--" if accuracy is None else f"{accuracy:.2f}%"
        lines.append(f"Python step {item['step']} | 処理 {n}/{item['total'] or '?'} | "
                     f"最終数値一致 {counts['explicit_final_numeric_exact']}/{n} ({percent}) | "
                     f"EOS付き最終一致 {counts['terminal_explicit_exact']}/{n} | "
                     f"Python実行成功 {counts['python_execution_success']}/{n} | "
                     f"ツール結果+EOS付き最終一致 {counts['tool_and_terminal_exact']}/{n}")
    for item in snapshot["extended"]:
        lines.append(f"追加評価 {item['group']} | 処理 {item['processed']} | "
                     f"正解 {item['correct']}/{item['answered']} | "
                     f"未測定 {item['unmeasured']} エラー {item['errors']}")
    for item in snapshot["tiers"]:
        if "profiles" in item:
            lines.append(f"T8 形式・ツール利用指標: {json.dumps(item['profiles'], ensure_ascii=False)}")
            continue
        progress = item.get("last_progress")
        if progress:
            scores = (f"正解 {progress['correct']}/{progress['answered']}"
                      if "correct" in progress else "正答数は元ログの確定値待ち")
            lines.append(f"{item['suite']} | 処理 {progress['processed']}/{progress['total']} | {scores}")
        else:
            lines.append(f"{item['suite']} | ログあり（件数・正答数は未抽出）")
    if snapshot.get("last_progress_hint") and not snapshot["python"]:
        hint = snapshot["last_progress_hint"]
        if hint.get("kind") == "progress":
            lines.append(f"処理進捗 {hint['processed']}/{hint['total']}（正解数ではありません）")
    if "seconds_since_log_update" in snapshot:
        lines.append(f"元ログの最終更新から {snapshot['seconds_since_log_update']:.1f}秒")
    if snapshot["read_errors"]:
        lines.append(f"読取・整合性エラー {len(snapshot['read_errors'])}: " + "; ".join(snapshot["read_errors"][:3]))
    lines.extend(snapshot.get("last_lines", []))
    print("\n".join(lines), file=output, flush=True)


def watch_existing(root: str | Path, *, poll_s: float = 5.0,
                   duration_s: float | None = None, once: bool = False,
                   stream: TextIO | None = None, notebook: bool = False) -> None:
    if not 1 <= poll_s <= 10:
        raise ValueError("poll_s must be between 1 and 10 seconds")
    monitor = ExistingSuiteMonitor(root)
    started = time.monotonic()
    while True:
        snapshot = monitor.snapshot()
        if notebook:
            try:
                from IPython.display import clear_output
                clear_output(wait=True)
            except ImportError:
                pass
        render_existing(snapshot, stream=stream)
        if once or (duration_s is not None and time.monotonic() - started >= duration_s):
            return
        time.sleep(poll_s)


def run_with_logging(command: list[str], out_dir: str | Path, *,
                     watch_root: str | Path | None = None, heartbeat_s: float = 5.0,
                     stream: TextIO | None = None, cwd: str | Path | None = None) -> int:
    """Unbuffered child tee plus heartbeat; never infer accuracy from progress.

    Source files and existing processes are untouched. Ctrl-C terminates only
    this wrapper's explicit child and records interruption; read-only ``watch``
    never starts, signals or stops any evaluator.
    """
    if not command:
        raise ValueError("an explicit command is required")
    if not 0 < heartbeat_s <= 10:
        raise ValueError("heartbeat_s must be in (0,10]")
    output = sys.stdout if stream is None else stream
    folder = Path(out_dir).resolve()
    if watch_root is not None and folder.is_relative_to(Path(watch_root).resolve()):
        raise ValueError("wrapper output must be outside the read-only watched tree")
    folder.mkdir(parents=True, exist_ok=False)
    cmd = list(map(str, command))
    if Path(cmd[0]).name.startswith("python") and "-u" not in cmd[1:]:
        cmd.insert(1, "-u")
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    monitor = ExistingSuiteMonitor(watch_root) if watch_root and Path(watch_root).is_dir() else None
    messages: queue.Queue[tuple[str, str | None]] = queue.Queue()
    started = time.monotonic()
    parser = TextProgressParser()
    progress = None
    interrupted = False
    atomic_json(folder / "command.json", {"command": cmd, "cwd": str(cwd) if cwd else None,
                "started_utc": utc(), "watch_root": str(watch_root) if watch_root else None})
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", errors="replace", bufsize=1, env=env, cwd=cwd)

    def relay(pipe, name):
        try:
            for line in iter(pipe.readline, ""):
                messages.put((name, line))
        finally:
            pipe.close()
            messages.put((name, None))

    threads = [threading.Thread(target=relay, args=(process.stdout, "stdout"), daemon=True),
               threading.Thread(target=relay, args=(process.stderr, "stderr"), daemon=True)]
    for thread in threads:
        thread.start()
    last_heartbeat = -math.inf
    ends = 0
    try:
        with (folder / "console.log").open("x", encoding="utf-8") as log:
            while ends < 2 or process.poll() is None:
                try:
                    source, line = messages.get(timeout=min(0.25, heartbeat_s))
                    if line is None:
                        ends += 1
                    else:
                        entry = f"[{utc()}] [{source}] {line.rstrip(chr(10) + chr(13))}"
                        print(entry, file=output, flush=True)
                        log.write(entry + "\n")
                        log.flush()
                        parsed = parser.feed(line)
                        if parsed:
                            progress = parsed
                except queue.Empty:
                    pass
                now = time.monotonic()
                if now - last_heartbeat >= heartbeat_s:
                    heartbeat = {"state": "running", "pid": process.pid,
                                 "updated_utc": utc(), "elapsed_seconds": round(now - started, 3),
                                 "last_progress_hint": progress}
                    if monitor is None and watch_root is not None:
                        if Path(watch_root).is_dir():
                            monitor = ExistingSuiteMonitor(watch_root)
                        else:
                            heartbeat["watched_directory_pending"] = True
                    if monitor is not None:
                        heartbeat["observed_scores"] = monitor.snapshot()
                        render_existing(heartbeat["observed_scores"], stream=output)
                    atomic_json(folder / "process_status.json", heartbeat)
                    msg = f"[{utc()}] HEARTBEAT elapsed={now - started:.1f}s child_pid={process.pid}"
                    print(msg, file=output, flush=True)
                    log.write(msg + "\n")
                    log.flush()
                    last_heartbeat = now
            code = process.wait()
    except BaseException:
        interrupted = True
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        raise
    finally:
        for thread in threads:
            thread.join(timeout=1)
        atomic_json(folder / "process_status.json", {"state": "interrupted" if interrupted else
                    ("complete" if process.returncode == 0 else "failed"),
                    "returncode": process.returncode, "pid": process.pid,
                    "updated_utc": utc(), "elapsed_seconds": round(time.monotonic() - started, 3),
                    "last_progress_hint": progress,
                    "score_note": "process success is not model correctness"})
    return code


def read_logged_run(folder: str | Path) -> dict:
    """Replay durable results; a status summary alone is never score evidence.

    In-progress snapshots can lag a newly appended event, so they retain their
    incomplete state while counts are derived from events. A completed run must
    agree exactly with its terminal event and durable counts before comparison.
    """
    folder = Path(folder)
    manifest = _read_json(folder / "run.json")
    binding = manifest["binding"]
    if manifest["config_sha256"] != digest(binding):
        raise ValueError(f"changed run binding: {folder}")
    identity = _check_identity(binding["identity"])
    total = _count(binding["total"], "bound total")
    status = _read_json(folder / "status.json")
    if (status.get("run_id") != manifest["run_id"] or
            status.get("config_sha256") != manifest["config_sha256"] or
            status.get("identity") != identity or status.get("total") != total):
        raise ValueError(f"mismatched status binding: {folder}")
    result_rows, last_event = {}, None
    with (folder / "events.jsonl").open(encoding="utf-8") as f:
        for sequence, line in enumerate(f, 1):
            if not line.endswith("\n"):
                raise ValueError("incomplete durable event tail")
            item = json.loads(line)
            if (item.get("sequence") != sequence or item.get("config_sha256") != manifest["config_sha256"]
                    or item.get("run_id") != manifest["run_id"]):
                raise ValueError("changed event binding or sequence")
            last_event = item
            if item["event"] == "case_result":
                row = item["result"]
                case_id = row.get("case_id")
                if not isinstance(case_id, str) or not case_id or case_id in result_rows:
                    raise ValueError("missing or duplicate comparison case ID")
                if (row.get("status") not in RESULT_STATUSES or
                        (row["status"] == "scored" and type(row.get("correct")) is not bool) or
                        (row["status"] != "scored" and row.get("correct") is not None) or
                        not isinstance(row.get("metrics"), dict)):
                    raise ValueError("invalid durable case result")
                canonical(row)
                result_rows[case_id] = row
    if len(result_rows) > total:
        raise ValueError("durable results exceed the bound case count")
    counts = Counter(row["status"] for row in result_rows.values())
    correct = sum(row["status"] == "scored" and row["correct"] for row in result_rows.values())
    answered = counts["scored"] + counts["invalid"]
    metric_totals, metric_observations = {}, {}
    for row in result_rows.values():
        for key, value in row["metrics"].items():
            if isinstance(value, (int, float)) and math.isfinite(value):
                metric_totals[key] = metric_totals.get(key, 0) + value
                metric_observations[key] = metric_observations.get(key, 0) + 1
    replayed = {"correct": correct, "answered": answered, "processed": len(result_rows),
                "incorrect": counts["scored"] - correct, "invalid": counts["invalid"],
                "unmeasured": counts["unmeasured"], "errors": counts["error"],
                "accuracy": correct / answered if answered else None,
                "coverage": answered / total if total else None,
                "pending": total - len(result_rows), "metric_totals": metric_totals,
                "metric_observations": metric_observations}
    if status.get("state") == "complete":
        if ((folder / "logger_active.json").exists() or not last_event or
                last_event.get("event") != "run_finished" or last_event.get("state") != "complete" or
                len(result_rows) != total or counts["unmeasured"] or counts["error"]):
            raise ValueError("complete status lacks complete durable terminal evidence")
        for key, value in replayed.items():
            # Older schema-1 loggers did not yet emit metric_observations. It is
            # reconstructed from events, never guessed from the primary score.
            if key == "metric_observations" and key not in status:
                continue
            if status.get(key) != value:
                raise ValueError(f"status summary differs from durable events: {key}")
    verified_status = {**status, **replayed}
    return {"folder": folder, "identity": identity, "binding": binding,
            "status": verified_status, "rows": result_rows,
            "signature": {key: identity[key] for key in COMPARISON_FIELDS},
            "correct": correct, "answered": answered, "counts": counts}


def comparison_rows(run_dirs: Iterable[str | Path]) -> list[dict]:
    """Read completed manifests/events; reject unmatched comparison conditions."""
    records = [read_logged_run(folder) for folder in run_dirs]
    if not records:
        raise ValueError("at least one run is required")
    first = records[0]
    for other in records[1:]:
        if other["signature"] != first["signature"] or other["binding"]["total"] != first["binding"]["total"]:
            raise ValueError("comparison rejected: suite/dataset/scorer/runtime/mode/budget/seed/metric differ")
    # Missing cases cannot create a flattering comparison on an easier subset.
    for item in records:
        n = item["binding"]["total"]
        if (item["status"]["state"] != "complete" or len(item["rows"]) != n
                or item["counts"]["unmeasured"] or item["counts"]["error"]):
            raise ValueError(f"comparison requires complete measured runs: {item['folder']}")
        if set(item["rows"]) != set(first["rows"]):
            raise ValueError("comparison rejected: per-case IDs differ")
    output = []
    for item in records:
        improved = regressed = both_correct = 0
        for case_id, row in item["rows"].items():
            before = first["rows"][case_id]
            a = before["status"] == "scored" and before["correct"]
            b = row["status"] == "scored" and row["correct"]
            improved += int(not a and b)
            regressed += int(a and not b)
            both_correct += int(a and b)
        ident = item["identity"]
        rate = item["correct"] / item["answered"] if item["answered"] else None
        baseline = first["correct"] / first["answered"] if first["answered"] else None
        output.append({"checkpoint_id": ident["checkpoint_id"],
                       "checkpoint_sha256": ident["checkpoint_sha256"],
                       "suite": ident["suite"], "mode": ident["mode"], "metric": ident["metric"],
                       "total": item["binding"]["total"], "correct": item["correct"],
                       "answered": item["answered"], "accuracy": rate,
                       "delta_percentage_points": (rate - baseline) * 100 if rate is not None else None,
                       "invalid": item["counts"]["invalid"], "unmeasured": 0, "errors": 0,
                       "paired_improved": improved, "paired_regressed": regressed,
                       "paired_both_correct": both_correct,
                       "dataset_sha256": ident["dataset_sha256"], "scorer_sha256": ident["scorer_sha256"],
                       "runtime_sha256": ident["runtime_sha256"],
                       "budget_json": canonical(ident["budget"]).decode("utf-8"),
                       "seed": ident["seed"]})
    return output


def export_comparison(run_dirs: Iterable[str | Path], destination: str | Path) -> list[dict]:
    rows = comparison_rows(run_dirs)
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    watch = sub.add_parser("watch", help="read an existing v47 evaluation; make no changes")
    watch.add_argument("root", type=Path)
    watch.add_argument("--poll-seconds", type=float, default=5)
    watch.add_argument("--duration-seconds", type=float)
    watch.add_argument("--once", action="store_true")
    watch.add_argument("--notebook", action="store_true")
    run = sub.add_parser("run", help="tee one explicitly supplied command and show a heartbeat")
    run.add_argument("--out-dir", type=Path, required=True)
    run.add_argument("--watch-root", type=Path)
    run.add_argument("--heartbeat-seconds", type=float, default=5)
    run.add_argument("--cwd", type=Path)
    run.add_argument("child_command", nargs=argparse.REMAINDER)
    compare = sub.add_parser("compare", help="compare only matched, complete evaluation runs")
    compare.add_argument("run_dirs", nargs="+", type=Path)
    compare.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "watch":
        try:
            watch_existing(args.root, poll_s=args.poll_seconds, duration_s=args.duration_seconds,
                           once=args.once, notebook=args.notebook)
        except KeyboardInterrupt:
            print("監視を終了しました。元の学習・評価プロセスは操作していません。", flush=True)
    elif args.command == "run":
        cmd = args.child_command
        if cmd and cmd[0] == "--":
            cmd = cmd[1:]
        sys.exit(run_with_logging(cmd, args.out_dir, watch_root=args.watch_root,
                                 heartbeat_s=args.heartbeat_seconds, cwd=args.cwd))
    else:
        rows = export_comparison(args.run_dirs, args.out)
        print(json.dumps({"csv": str(args.out), "runs": len(rows)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
