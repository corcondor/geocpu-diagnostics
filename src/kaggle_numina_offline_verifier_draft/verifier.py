"""Offline, shardable verifier for the pinned public NuminaMath-TIR Parquet.

The only executable source text is sent to a child on Linux. Importing this
module, parsing rows, and the synthetic tests never execute source code.
The child restrictions are defense in depth, not a security proof.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unicodedata


HERE = Path(__file__).resolve().parent
LOCK_SHA256 = "f72eb60ad2a93ede9cd80633aa7531dbac069f5ef40dbd8d881cffdfee17db2c"
REAL_LOCK_SHA256 = LOCK_SHA256
FENCE = re.compile(r"(?m)^```(?P<language>[^\n`]*)\n(?P<body>.*?)^```[ \t]*(?P<end>\n|$)", re.S)
LEGACY = re.compile(r"<(?:SET|ADD|SUB|MUL|DIV|MOD|MOV|LOAD|STORE|DEC|INC|AND|OR|XOR|NOT|SHL|SHR|CMP|JMP|JZ|JNZ|HALT)\b|\[R\d+\s*=|\bExactRAM\w*\b|\bKogge.Stone\b", re.I)
MAX_CODE_CHARS = 32768
MAX_OUTPUT_CHARS = 8192
MAX_TOTAL_TOKENS = 8192
PER_BLOCK_SECONDS = 5.0
WORKER_CPU_HARD_SECONDS = 90
WORKER_CPU_RECYCLE_SECONDS = 20.0
WORKER_MAX_ROWS = 100
MAX_BLOCKS_PER_ROW = 8
MAX_PROTOCOL_BYTES = 65536
CHUNK_ROWS = 64


class RowRejected(Exception):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


class WorkerUnavailable(Exception):
    pass


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_question(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).strip()
    text = re.sub(r"^(?:ユーザー|user)\s*[:：]\s*", "", text, flags=re.I)
    text = re.sub(r"\n\s*(?:アシスタント|assistant)\s*[:：]\s*$", "", text, flags=re.I)
    text = re.sub(r"^\s*【分野:[^\n]*】\s*", "", text)
    text = re.sub(r"(?m)^\s*分野\s*[:：][^\n]*\n?", "", text)
    text = re.sub(r"\s*(?:正解|回答|答え)\s*[:：]\s*$", "", text)
    return re.sub(r"\s+", "", text)


def split_question(key: str) -> str:
    value = int(sha_text(key)[:12], 16) % 100
    return "train" if value < 90 else "dev" if value < 95 else "test"


def observation_text(stdout: str) -> str:
    max_run = max([len(match.group()) for match in re.finditer(r"`+", stdout)] + [0])
    fence = "`" * max(3, max_run + 1)
    content = stdout if stdout else "[no stdout]"
    return "\n" + fence + "output\n" + content + ("" if content.endswith("\n") else "\n") + fence + "\n"


def segments(length: int, tool_spans: list[tuple[int, int]]) -> list[dict]:
    result: list[dict] = []
    position = 0
    for begin, end in tool_spans:
        if begin > position:
            result.append({"start": position, "end": begin, "role": "assistant", "train": True})
        result.append({"start": begin, "end": end, "role": "tool", "train": False})
        position = end
    if position < length:
        result.append({"start": position, "end": length, "role": "assistant", "train": True})
    return result or [{"start": 0, "end": length, "role": "assistant", "train": True}]


def parse_pairs(solution: str) -> tuple[str, list[tuple[re.Match, re.Match]]]:
    if not isinstance(solution, str) or not solution.strip():
        raise RowRejected("empty_tir_solution")
    source = solution.replace("\r\n", "\n")
    fences = list(FENCE.finditer(source))
    if not fences or source.count("```") != 2 * len(fences):
        raise RowRejected("unclosed_or_nested_fence")
    pairs = []
    for index in range(0, len(fences), 2):
        python = fences[index]
        if python.group("language").strip().lower() not in ("python", "py"):
            raise RowRejected("non_python_or_unpaired_output_in_tir")
        if index + 1 >= len(fences):
            raise RowRejected("python_without_recorded_output")
        output = fences[index + 1]
        if output.group("language").strip().lower() != "output" or source[python.end():output.start()].strip():
            raise RowRejected("python_output_not_adjacent")
        if not python.group("body").strip() or "```" in output.group("body"):
            raise RowRejected("empty_python_or_nested_output")
        if re.search(r"(?m)^(?:Traceback|Cell In\[|\w*(?:Error|Exception):)", output.group("body")):
            raise RowRejected("public_error_trace_not_identical_to_runtime_error_protocol")
        pairs.append((python, output))
    return source, pairs


def check_row(raw: dict, *, vocab: set[str], eval_questions: set[str], eval_ids: set[str]) -> tuple[str, str, str, list[tuple[re.Match, re.Match]]]:
    if not isinstance(raw, dict):
        raise RowRejected("row_not_object")
    problem, solution = raw.get("problem"), raw.get("solution")
    if not isinstance(problem, str) or not isinstance(solution, str):
        raise RowRejected("numina_schema")
    messages = raw.get("messages")
    if (not isinstance(messages, list) or len(messages) != 2 or
            any(not isinstance(message, dict) for message in messages) or
            messages[0].get("role") != "user" or messages[1].get("role") != "assistant"):
        raise RowRejected("numina_message_roles")
    if messages[0].get("content") != problem or messages[1].get("content") != solution:
        raise RowRejected("numina_message_text_mismatch")
    prompt = "ユーザー: " + problem + "\nアシスタント: "
    if not prompt.strip() or not solution.strip():
        raise RowRejected("empty_prompt_or_target")
    group = canonical_question(problem)
    if sha_text(group) in eval_questions or sha_text(canonical_question(prompt)) in eval_questions:
        raise RowRejected("evaluation_prompt_exact_or_wrapper_overlap")
    if any(sha_text(str(raw[field])) in eval_ids for field in ("id", "question_id", "instance_id") if raw.get(field) is not None):
        raise RowRejected("evaluation_id_overlap")
    if LEGACY.search(prompt + solution):
        raise RowRejected("legacy_alu_protocol_excluded")
    if set(prompt + solution) - vocab:
        raise RowRejected("unknown_vocabulary_character_whole_row_rejected")
    if len(prompt) + len(solution) + 1 > MAX_TOTAL_TOKENS:
        raise RowRejected("full_answer_and_true_eos_exceed_context")
    source, pairs = parse_pairs(solution)
    if len(pairs) > MAX_BLOCKS_PER_ROW:
        raise RowRejected("tool_call_limit")
    if any(len(python.group("body")) > MAX_CODE_CHARS for python, _ in pairs):
        raise RowRejected("code_length_limit")
    return prompt, group, source, pairs


def transform_verified(raw: dict, row_index: int, executor, *, vocab: set[str], eval_questions: set[str], eval_ids: set[str], repo_id: str, revision: str) -> tuple[dict, dict | None]:
    """Execute a problem's blocks in one fresh namespace; return hash-only receipt."""
    solution = raw.get("solution") if isinstance(raw, dict) else None
    receipt = {"row_index": row_index, "source_target_sha256": sha_text(solution) if isinstance(solution, str) else None,
               "question_group_sha256": None, "split": None, "status": "rejected", "reason": None,
               "execution_attempted": False, "tool_blocks": []}
    try:
        prompt, group, source, pairs = check_row(raw, vocab=vocab, eval_questions=eval_questions, eval_ids=eval_ids)
        receipt["question_group_sha256"] = sha_text(group)
        receipt["split"] = split_question(group)
        if executor.reset().get("reset") is not True:
            raise RowRejected("worker_reset_failure")
        parts, spans, position, length = [], [], 0, 0
        for python, output in pairs:
            close_end = python.end() - len(python.group("end"))
            prefix = source[position:close_end]
            parts.append(prefix)
            length += len(prefix)
            code, expected = python.group("body"), output.group("body")
            receipt["execution_attempted"] = True
            result = executor.execute(code)
            item = {"block_index": len(receipt["tool_blocks"]), "code_sha256": sha_text(code),
                    "recorded_stdout_sha256": sha_text(expected),
                    "observed_stdout_sha256": sha_text(result.get("stdout", "")) if isinstance(result.get("stdout"), str) else None,
                    "status": "passed"}
            receipt["tool_blocks"].append(item)
            if result.get("timeout"):
                item["status"] = "timeout"
                raise RowRejected("executed_python_timeout")
            if result.get("truncated"):
                item["status"] = "truncated"
                raise RowRejected("executed_python_truncated")
            if result.get("exception_type") in ("ModuleNotFoundError", "ImportError"):
                item["status"] = "dependency_unavailable"
                raise RowRejected("executed_python_dependency_unavailable")
            if not result.get("success") or result.get("stderr"):
                item["status"] = "error"
                raise RowRejected("executed_python_error_or_stderr")
            stdout = result.get("stdout", "")
            if not isinstance(stdout, str) or stdout.replace("\r\n", "\n") != expected:
                item["status"] = "stdout_mismatch"
                raise RowRejected("executed_stdout_differs_from_public_record")
            observation = observation_text(stdout)
            item["observation_sha256"] = sha_text(observation)
            item["verification"] = "executed_all_blocks_in_this_problem_stdout_exact"
            begin = length
            parts.append(observation)
            length += len(observation)
            spans.append((begin, length))
            position = output.end()
        parts.append(source[position:])
        target = "".join(parts)
        if set(prompt + target) - vocab:
            raise RowRejected("normalized_tool_protocol_unknown_vocabulary")
        if len(prompt) + len(target) + 1 > MAX_TOTAL_TOKENS:
            raise RowRejected("normalized_full_answer_and_true_eos_exceed_context")
        record = {"prompt": prompt, "target": target, "full_text": prompt + target,
                  "source": "tir_public_numina", "bucket": "tir", "language": "en",
                  "target_segments": segments(len(target), spans),
                  "provenance": {"public_repo": repo_id, "revision": revision, "source_row_index": row_index,
                                 "original_source": None, "source_full_text_sha256": None,
                                 "anchor_boundary_alignment": None, "anchor_boundary_separator": None,
                                 "question_group_sha256": sha_text(group),
                                 "source_target_sha256": sha_text(solution), "target_sha256": sha_text(target),
                                 "split": receipt["split"], "verification": "executed_all_blocks_in_this_problem",
                                 "tool_blocks": [{"code_sha256": item["code_sha256"],
                                                  "recorded_stdout_sha256": item["recorded_stdout_sha256"],
                                                  "observation_sha256": item["observation_sha256"],
                                                  "verification": item["verification"]}
                                                 for item in receipt["tool_blocks"]],
                                 "new_model_authored_example": False, "protocol_normalized": True,
                                 "full_answer_and_true_eos": True}}
        receipt["status"] = "accepted"
        receipt["target_sha256"] = sha_text(target)
        return receipt, record
    except RowRejected as error:
        receipt["reason"] = error.reason
        return receipt, None


def _cpu_seconds(pid: int) -> float:
    # /proc stat fields 14 and 15 are cumulative user and system CPU ticks.
    data = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    tail = data[data.rfind(")") + 2:].split()
    return (int(tail[11]) + int(tail[12])) / os.sysconf("SC_CLK_TCK")


class LinuxWorker:
    """One persistent namespace, reset between rows and recycled before CPU cap."""

    def __init__(self, worker_path: Path):
        if sys.platform != "linux":
            raise WorkerUnavailable("linux_required")
        self.worker_path = worker_path
        self.temp = tempfile.TemporaryDirectory(prefix="numina_worker_")
        self.process = None
        self.buffer = b""
        self.rows = 0
        self.closed = False
        try:
            import pwd
            user = pwd.getpwnam("nobody") if os.geteuid() == 0 else None
            uid = user.pw_uid if user else os.geteuid()
            gid = user.pw_gid if user else os.getegid()
            if uid == 0:
                raise WorkerUnavailable("unprivileged_uid_unavailable")
            os.chmod(self.temp.name, 0o700)
            if user:
                os.chown(self.temp.name, uid, gid)

            def confine():
                import resource
                os.setsid()
                resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
                resource.setrlimit(resource.RLIMIT_CPU, (WORKER_CPU_HARD_SECONDS, WORKER_CPU_HARD_SECONDS))
                resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
                resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
                resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
                if user:
                    os.setgroups([])
                    os.setgid(gid)
                    os.setuid(uid)

            self.process = subprocess.Popen(
                [sys.executable, "-I", "-B", "-u", str(worker_path)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                cwd=self.temp.name, env={"PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1",
                                         "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
                close_fds=True, preexec_fn=confine, bufsize=0)
            ready = self._read_response(30.0)
            if ready.get("ready") is not True or ready.get("security_sandbox") is not False:
                raise WorkerUnavailable("worker_startup_protocol")
        except BaseException:
            self.close()
            raise

    def _read_response(self, timeout: float) -> dict:
        deadline = time.monotonic() + timeout
        while True:
            if b"\n" in self.buffer:
                line, self.buffer = self.buffer.split(b"\n", 1)
                try:
                    result = json.loads(line.decode("utf-8"))
                except (UnicodeError, json.JSONDecodeError) as error:
                    raise WorkerUnavailable("worker_protocol_json") from error
                if not isinstance(result, dict):
                    raise WorkerUnavailable("worker_protocol_object")
                return result
            if len(self.buffer) > MAX_PROTOCOL_BYTES:
                raise WorkerUnavailable("worker_protocol_size")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("worker_wall_timeout")
            if self.process is None or self.process.poll() is not None:
                raise WorkerUnavailable("worker_exited")
            readable, _, _ = select.select([self.process.stdout], [], [], remaining)
            if not readable:
                continue
            part = os.read(self.process.stdout.fileno(), min(4096, MAX_PROTOCOL_BYTES + 1 - len(self.buffer)))
            if not part:
                raise WorkerUnavailable("worker_eof")
            self.buffer += part

    def _rpc(self, request: dict, timeout: float) -> dict:
        if self.closed or self.process is None or self.process.poll() is not None:
            raise WorkerUnavailable("worker_exited")
        try:
            self.process.stdin.write((json.dumps(request, ensure_ascii=False) + "\n").encode("utf-8"))
            self.process.stdin.flush()
            return self._read_response(timeout)
        except (BrokenPipeError, OSError) as error:
            raise WorkerUnavailable("worker_pipe") from error

    def reset(self) -> dict:
        return self._rpc({"op": "reset"}, 5.0)

    def execute(self, code: str) -> dict:
        try:
            result = self._rpc({"op": "execute", "code": code,
                                "max_output_chars": MAX_OUTPUT_CHARS, "max_code_chars": MAX_CODE_CHARS,
                                "search_enabled": False}, PER_BLOCK_SECONDS)
            if not isinstance(result.get("success"), bool) or not isinstance(result.get("stdout"), str):
                raise WorkerUnavailable("worker_result_protocol")
            return result
        except TimeoutError:
            self.close()
            return {"success": False, "stdout": "", "timeout": True, "truncated": False,
                    "exception_type": "TimeoutError"}

    def cpu_seconds(self) -> float:
        if self.process is None or self.process.poll() is not None:
            raise WorkerUnavailable("worker_exited")
        try:
            return _cpu_seconds(self.process.pid)
        except (OSError, ValueError) as error:
            raise WorkerUnavailable("worker_cpu_accounting") from error

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.process is not None and self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if self.process is not None:
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            for stream in (self.process.stdin, self.process.stdout):
                if stream:
                    stream.close()
        self.temp.cleanup()


class WorkerPool:
    def __init__(self, worker_path: Path, factory=LinuxWorker):
        self.worker_path, self.factory = worker_path, factory
        self.worker = None
        self.recycles = Counter()

    def _fresh(self):
        if self.worker is not None:
            self.worker.close()
        self.worker = self.factory(self.worker_path)

    def reset(self) -> dict:
        if self.worker is None:
            self._fresh()
        try:
            return self.worker.reset()
        except (WorkerUnavailable, TimeoutError):
            self.recycles["reset_failure"] += 1
            self._fresh()
            return self.worker.reset()

    def execute(self, code: str) -> dict:
        if self.worker is None:
            raise WorkerUnavailable("worker_not_started")
        result = self.worker.execute(code)
        if result.get("timeout"):
            self.recycles["timeout"] += 1
            self.worker = None
        return result

    def end_row(self) -> None:
        if self.worker is None:
            return
        self.worker.rows += 1
        if self.worker.rows >= WORKER_MAX_ROWS:
            self.recycles["row_cap"] += 1
            self.worker.close()
            self.worker = None
            return
        try:
            cpu = self.worker.cpu_seconds()
        except WorkerUnavailable:
            self.recycles["cpu_accounting_failed"] += 1
            self.worker.close()
            self.worker = None
            return
        if cpu >= WORKER_CPU_RECYCLE_SECONDS:
            self.recycles["cpu_threshold"] += 1
            self.worker.close()
            self.worker = None

    def close(self) -> None:
        if self.worker is not None:
            self.worker.close()
            self.worker = None


def _hex_sha(value) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _checked_file(root: Path, name: str) -> Path:
    if not isinstance(name, str) or not name or Path(name).is_absolute() or ".." in Path(name).parts:
        raise ValueError("Input filename must be relative to the single public input root")
    path = root / name
    if path.is_symlink() or any(part.is_symlink() for part in path.parents if part != root.parent):
        raise ValueError("Symlink in input path")
    if not path.is_file() or root.resolve() not in path.resolve().parents:
        raise ValueError("Input file missing or outside public input root")
    return path


def _load_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Expected JSON object")
    return data


def _pin(path: Path) -> dict:
    return {"bytes": path.stat().st_size, "sha256": sha_file(path)}


def _check_pin(path: Path, expected: dict) -> dict:
    actual = _pin(path)
    if not isinstance(expected, dict) or actual != {"bytes": expected.get("bytes"), "sha256": expected.get("sha256")}:
        raise ValueError("Pinned input differs")
    return actual


def _iter_parquet_rows(path: Path, start: int, end: int):
    import pyarrow.parquet as pq
    index = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
        batch_end = index + batch.num_rows
        if batch_end <= start:
            index = batch_end
            continue
        if index >= end:
            break
        for offset, row in enumerate(batch.to_pylist()):
            row_index = index + offset
            if start <= row_index < end:
                yield row_index, row
        index = batch_end


def _official_test_hashes(path: Path) -> tuple[set[str], int]:
    import pyarrow.parquet as pq
    hashes, count = set(), 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
        for row in batch.to_pylist():
            if not isinstance(row, dict) or not isinstance(row.get("problem"), str):
                raise ValueError("Official Numina test schema differs")
            hashes.add(sha_text(canonical_question(row["problem"])))
            count += 1
    if count <= 0:
        raise ValueError("Official Numina test is empty")
    return hashes, count


def _write_json_atomic(path: Path, value: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def _write_jsonl_exclusive(path: Path, rows: list[dict]) -> dict:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return _pin(path)


def _load_chunks(directory: Path, manifest: dict) -> list[dict]:
    chunks = manifest.get("chunks")
    if not isinstance(chunks, list):
        raise ValueError("Prior chunk manifest missing")
    next_index = manifest["range"]["start"]
    for chunk in chunks:
        if chunk.get("start") != next_index or not isinstance(chunk.get("end"), int) or chunk["end"] <= next_index:
            raise ValueError("Prior chunk range has a gap or overlap")
        for kind in ("receipts", "records"):
            path = _checked_file(directory, chunk[kind]["name"])
            _check_pin(path, chunk[kind]["pin"])
        with (directory / chunk["receipts"]["name"]).open(encoding="utf-8") as stream:
            receipts = [json.loads(line) for line in stream]
        with (directory / chunk["records"]["name"]).open(encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream]
        if [receipt.get("row_index") for receipt in receipts] != list(range(chunk["start"], chunk["end"])):
            raise ValueError("Prior row receipts are not complete and ordered")
        accepted = [receipt for receipt in receipts if receipt.get("status") == "accepted"]
        if len(accepted) != len(records):
            raise ValueError("Accepted receipts and records differ in count")
        if any(receipt.get("status") not in ("accepted", "rejected") for receipt in receipts):
            raise ValueError("Unknown row receipt status")
        for receipt, record in zip(accepted, records):
            _validate_accepted_receipt_record(receipt, record)
        next_index = chunk["end"]
    if next_index != manifest.get("next_index"):
        raise ValueError("Prior next index differs from chunks")
    return chunks


def _validate_accepted_receipt_record(receipt: dict, record: dict) -> None:
    if not isinstance(record, dict):
        raise ValueError("Accepted curriculum record is not an object")
    provenance = record.get("provenance")
    prompt, target = record.get("prompt"), record.get("target")
    if not isinstance(provenance, dict) or not isinstance(prompt, str) or not isinstance(target, str):
        raise ValueError("Accepted curriculum record schema differs")
    pairs = ((receipt.get("row_index"), provenance.get("source_row_index")),
             (receipt.get("source_target_sha256"), provenance.get("source_target_sha256")),
             (receipt.get("question_group_sha256"), provenance.get("question_group_sha256")),
             (receipt.get("target_sha256"), provenance.get("target_sha256")),
             (receipt.get("split"), provenance.get("split")))
    if (any(left != right for left, right in pairs)
            or receipt.get("reason") is not None or receipt.get("execution_attempted") is not True
            or record.get("bucket") != "tir" or record.get("source") != "tir_public_numina"
            or provenance.get("verification") != "executed_all_blocks_in_this_problem"
            or record.get("full_text") != prompt + target
            or sha_text(target) != receipt.get("target_sha256")
            or receipt.get("split") not in ("train", "dev", "test")):
        raise ValueError("Accepted receipt and curriculum provenance differ")
    receipt_blocks = receipt.get("tool_blocks")
    record_blocks = provenance.get("tool_blocks")
    spans = record.get("target_segments")
    if not isinstance(receipt_blocks, list) or not isinstance(record_blocks, list) or not isinstance(spans, list):
        raise ValueError("Accepted tool receipt or target segments missing")
    tool_spans = [span for span in spans if isinstance(span, dict) and span.get("role") == "tool"]
    if not receipt_blocks or len(receipt_blocks) != len(record_blocks) or len(receipt_blocks) != len(tool_spans):
        raise ValueError("Accepted tool block counts differ")
    cursor = 0
    for index, (receipt_block, record_block, span) in enumerate(zip(receipt_blocks, record_blocks, tool_spans)):
        if (not isinstance(receipt_block, dict) or not isinstance(record_block, dict)
                or receipt_block.get("block_index") != index or receipt_block.get("status") != "passed"
                or receipt_block.get("verification") != "executed_all_blocks_in_this_problem_stdout_exact"
                or receipt_block.get("recorded_stdout_sha256") != receipt_block.get("observed_stdout_sha256")
                or {key: receipt_block.get(key) for key in ("code_sha256", "recorded_stdout_sha256", "observation_sha256", "verification")} != record_block
                or span.get("train") is not False
                or not isinstance(span.get("start"), int) or not isinstance(span.get("end"), int)
                or span["start"] < cursor or not span["start"] < span["end"] <= len(target)
                or sha_text(target[span["start"]:span["end"]]) != receipt_block.get("observation_sha256")):
            raise ValueError("Accepted tool receipt and transformed observation differ")
        cursor = span["end"]


def _input_context(root: Path, names: dict, *, production: bool) -> dict:
    if production:
        if sys.platform != "linux" or root.parent.resolve() != Path("/kaggle/input"):
            raise ValueError("Production verifier requires one /kaggle/input dataset on Linux")
        mounted = [path.name for path in root.parent.iterdir() if path.is_dir()]
        if mounted != [root.name]:
            raise ValueError("Unexpected Kaggle input dataset mount")
        if Path("/content/drive").exists() or Path("/kaggle/working/MyDrive").exists():
            raise ValueError("Drive-like mount found")
    required = {"train", "test", "vocab", "exclusions", "binding"}
    if not isinstance(names, dict) or set(names) != required:
        raise ValueError("Input file inventory differs")
    files = {key: _checked_file(root, value) for key, value in names.items()}
    lock_path = HERE / "public_source_lock_v44.json"
    if sha_file(lock_path) != LOCK_SHA256:
        raise ValueError("Bundled public source lock differs")
    lock = _load_json(lock_path)
    numina = lock["repos"]["numina"]
    train_pin = _check_pin(files["train"], numina["files"]["train"])
    test_pin = _check_pin(files["test"], numina["files"]["test"])
    binding = _load_json(files["binding"])
    if binding.get("schema") != "geocpu_numina_hash_binding_v1" or binding.get("source_lock_sha256") != LOCK_SHA256:
        raise ValueError("Hash binding schema or source lock differs")
    vocab_pin = _check_pin(files["vocab"], binding.get("vocabulary", {}))
    exclusions_pin = _check_pin(files["exclusions"], binding.get("exclusion_index", {}))
    anchor = binding.get("anchor")
    if not isinstance(anchor, dict) or not isinstance(anchor.get("bytes"), int) or not _hex_sha(anchor.get("sha256")):
        raise ValueError("Anchor hash binding missing")
    external_eval = binding.get("evaluation_inputs")
    if not isinstance(external_eval, list) or not external_eval:
        raise ValueError("Evaluation file hash bindings missing")
    for pin in external_eval:
        if not isinstance(pin, dict) or not isinstance(pin.get("name"), str) or not isinstance(pin.get("bytes"), int) or not _hex_sha(pin.get("sha256")):
            raise ValueError("Malformed evaluation file hash binding")
    exclusions = _load_json(files["exclusions"])
    if exclusions.get("schema") != "geocpu_evaluation_hash_index_v1" or exclusions.get("evaluation_inputs") != external_eval:
        raise ValueError("Evaluation hash index binding differs")
    question_hashes = exclusions.get("question_sha256")
    id_hashes = exclusions.get("id_sha256")
    if not isinstance(question_hashes, list) or not isinstance(id_hashes, list) or not all(map(_hex_sha, question_hashes + id_hashes)):
        raise ValueError("Evaluation exclusion hashes malformed")
    vocab_data = _load_json(files["vocab"])
    vocab_map = vocab_data.get("vocab")
    if not isinstance(vocab_map, dict):
        raise ValueError("Character vocabulary absent")
    vocab = {key for key, value in vocab_map.items() if isinstance(key, str) and len(key) == 1 and isinstance(value, int) and value >= 5}
    if not vocab:
        raise ValueError("Character vocabulary empty")
    official_hashes, official_rows = _official_test_hashes(files["test"])
    eval_questions = set(question_hashes) | official_hashes
    eval_ids = set(id_hashes)
    import pyarrow.parquet as pq
    source_rows = pq.ParquetFile(files["train"]).metadata.num_rows
    fingerprint = {"source_lock_sha256": LOCK_SHA256,
                   "numina_train": train_pin, "numina_test": test_pin, "numina_test_rows": official_rows,
                   "vocabulary": vocab_pin, "anchor": anchor, "evaluation_inputs": external_eval,
                   "exclusion_index": exclusions_pin,
                   "binding_sha256": sha_file(files["binding"]),
                   "verifier_source_sha256": sha_file(HERE / "verifier.py"),
                   "worker_source_sha256": sha_file(HERE / "executor_worker_v44r1.py")}
    return {"files": files, "fingerprint": fingerprint, "vocab": vocab,
            "eval_questions": eval_questions, "eval_ids": eval_ids,
            "source_rows": source_rows, "repo_id": numina["repo_id"], "revision": numina["revision"]}


def _materialize(directory: Path, chunks: list[dict]) -> dict:
    # Keep first source row for each exact question/target pair, like v44fix.
    seen: set[tuple[str, str]] = set()
    counts = Counter()
    outputs = {}
    handles = {}
    try:
        for split in ("train", "dev", "test"):
            handles[split] = (directory / (split + ".jsonl.tmp")).open("x", encoding="utf-8", newline="\n")
        for chunk in chunks:
            with (directory / chunk["records"]["name"]).open(encoding="utf-8") as stream:
                for line in stream:
                    record = json.loads(line)
                    provenance = record["provenance"]
                    key = (provenance["question_group_sha256"], provenance["target_sha256"])
                    if key in seen:
                        counts["duplicate_exact"] += 1
                        continue
                    seen.add(key)
                    split = provenance["split"]
                    handles[split].write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                    counts[split] += 1
    finally:
        for handle in handles.values():
            handle.close()
    for split in ("train", "dev", "test"):
        path = directory / (split + ".jsonl")
        os.replace(directory / (split + ".jsonl.tmp"), path)
        outputs[path.name] = _pin(path)
    return {"counts": dict(counts), "outputs": outputs}


def run_shard(root: Path, names: dict, output: Path, *, start: int, end: int,
              wall_budget_seconds: float, prior: Path | None = None,
              production: bool = True, executor_factory=None) -> dict:
    """Verify one contiguous range; only committed chunks survive a restart."""
    if production:
        raise RuntimeError("Public-code verification is blocked in this draft; a separately reviewed distribution is required")
    if executor_factory is None or LOCK_SHA256 == REAL_LOCK_SHA256:
        raise RuntimeError("Synthetic mode requires a fake executor and a synthetic source lock")
    root, output = Path(root), Path(output)
    context = _input_context(root, names, production=production)
    if not (isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= context["source_rows"]):
        raise ValueError("Invalid source row range")
    if not (60 <= wall_budget_seconds <= 11 * 3600 + 30 * 60) and production:
        raise ValueError("Production wall budget must leave time before Kaggle's 12-hour limit")
    if not isinstance(wall_budget_seconds, (int, float)) or wall_budget_seconds <= 0:
        raise ValueError("Wall budget must be positive")
    if output.exists():
        raise FileExistsError("Output directory already exists")
    output.mkdir(parents=True)
    (output / "chunks").mkdir()
    manifest = {"schema": "geocpu_numina_offline_shard_v1", "fingerprint": context["fingerprint"],
                "range": {"start": start, "end": end}, "source_rows": context["source_rows"],
                "next_index": start, "complete": False, "chunks": [], "worker_recycles": {},
                "isolation_claim": "kaggle_offline_cpu_worker_limits_unprivileged_if_available_not_security_proof",
                "evaluation_exclusion_provenance": "external_hash_index_not_recomputed_from_eval_files"}
    if prior is not None:
        previous = _load_json(Path(prior) / "run_manifest.json")
        if (previous.get("schema") != manifest["schema"] or previous.get("fingerprint") != manifest["fingerprint"]
                or previous.get("range") != manifest["range"] or previous.get("source_rows") != manifest["source_rows"]):
            raise ValueError("Prior session binding or row range differs")
        chunks = _load_chunks(Path(prior), previous)
        for chunk in chunks:
            for kind in ("receipts", "records"):
                relative = chunk[kind]["name"]
                destination = output / relative
                shutil.copyfile(Path(prior) / relative, destination)
                _check_pin(destination, chunk[kind]["pin"])
        manifest["chunks"] = chunks
        manifest["next_index"] = previous["next_index"]
    _write_json_atomic(output / "run_manifest.json", manifest)
    pool = WorkerPool(HERE / "executor_worker_v44r1.py", executor_factory or LinuxWorker)
    started = time.monotonic()
    current_receipts: list[dict] = []
    current_records: list[dict] = []
    chunk_start = manifest["next_index"]

    def commit(next_index: int):
        nonlocal current_receipts, current_records, chunk_start
        if not current_receipts:
            return
        prefix = f"chunks/{chunk_start:08d}-{next_index:08d}"
        receipt_name, record_name = prefix + ".receipts.jsonl", prefix + ".records.jsonl"
        receipt_pin = _write_jsonl_exclusive(output / receipt_name, current_receipts)
        record_pin = _write_jsonl_exclusive(output / record_name, current_records)
        manifest["chunks"].append({"start": chunk_start, "end": next_index,
                                   "receipts": {"name": receipt_name, "pin": receipt_pin},
                                   "records": {"name": record_name, "pin": record_pin}})
        manifest["next_index"] = next_index
        manifest["worker_recycles"] = dict(pool.recycles)
        _write_json_atomic(output / "run_manifest.json", manifest)
        chunk_start = next_index
        current_receipts, current_records = [], []

    try:
        for index, raw in _iter_parquet_rows(context["files"]["train"], manifest["next_index"], end):
            if time.monotonic() - started >= wall_budget_seconds:
                break
            receipt, record = transform_verified(raw, index, pool,
                                                 vocab=context["vocab"], eval_questions=context["eval_questions"],
                                                 eval_ids=context["eval_ids"], repo_id=context["repo_id"], revision=context["revision"])
            current_receipts.append(receipt)
            if record is not None:
                current_records.append(record)
            pool.end_row()
            if len(current_receipts) >= CHUNK_ROWS:
                commit(index + 1)
        commit(chunk_start + len(current_receipts))
    finally:
        pool.close()
    manifest["complete"] = manifest["next_index"] == end
    materialized = _materialize(output, manifest["chunks"])
    manifest["materialized"] = materialized
    manifest["training_eligible"] = False
    manifest["stage_a_admissible"] = False
    manifest["stage_a_blockers"] = ["only_Numina_TIR_bucket", "anchor_content_not_attached",
                                    "external_evaluation_hash_index_not_independently_proven",
                                    "credential_free_disposable_vm_not_established",
                                    "v44fix_source_code_hash_contract_differs"]
    _write_json_atomic(output / "run_manifest.json", manifest)
    return manifest


def merge_shards(shard_dirs: list[Path], output: Path) -> dict:
    """Join only complete, contiguous, hash-matched Numina source row ranges."""
    if not shard_dirs:
        raise ValueError("No shards supplied")
    sessions = []
    for directory in shard_dirs:
        directory = Path(directory)
        manifest = _load_json(directory / "run_manifest.json")
        if manifest.get("schema") != "geocpu_numina_offline_shard_v1" or manifest.get("complete") is not True:
            raise ValueError("Incomplete or unknown shard")
        chunks = _load_chunks(directory, manifest)
        sessions.append((manifest["range"]["start"], directory, manifest, chunks))
    sessions.sort(key=lambda item: item[0])
    common = sessions[0][2]["fingerprint"]
    row_count = sessions[0][2]["source_rows"]
    cursor = 0
    for start, _, manifest, _ in sessions:
        if manifest["fingerprint"] != common or manifest["source_rows"] != row_count or start != cursor:
            raise ValueError("Shards have different bindings or a row gap/overlap")
        cursor = manifest["range"]["end"]
    if cursor != row_count:
        raise ValueError("Full pinned Numina train Parquet is not covered")
    output = Path(output)
    if output.exists():
        raise FileExistsError("Merged output already exists")
    output.mkdir(parents=True)
    (output / "chunks").mkdir()
    merged_chunks = []
    for _, directory, _, chunks in sessions:
        for chunk in chunks:
            for kind in ("receipts", "records"):
                relative = chunk[kind]["name"]
                destination = output / relative
                shutil.copyfile(directory / relative, destination)
                _check_pin(destination, chunk[kind]["pin"])
            merged_chunks.append(chunk)
    materialized = _materialize(output, merged_chunks)
    with (output / "row_receipts.jsonl").open("x", encoding="utf-8", newline="\n") as destination:
        for chunk in merged_chunks:
            with (output / chunk["receipts"]["name"]).open(encoding="utf-8") as source:
                shutil.copyfileobj(source, destination)
    row_receipts_pin = _pin(output / "row_receipts.jsonl")
    status = Counter()
    with (output / "row_receipts.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            receipt = json.loads(line)
            status[receipt["status"]] += 1
            if receipt["status"] == "rejected":
                status["reason:" + receipt["reason"]] += 1
    if status["accepted"] + status["rejected"] != row_count:
        raise ValueError("Merged receipt count differs from source row count")
    manifest = {"schema": "geocpu_numina_verified_curriculum_v1", "source_rows": row_count,
                "fingerprint": common, "shard_count": len(sessions), "counts": dict(status),
                "unique_curriculum_counts": materialized["counts"],
                "outputs": materialized["outputs"], "row_receipts": row_receipts_pin,
                "tool_verification": {"numina_source_rows_checked_in_synthetic_fixture": True,
                                      "accepted_rows_all_blocks_stdout_exact": False,
                                      "public_execution_performed": False,
                                      "mismatching_or_failed_rows_excluded_in_synthetic_fixture": True},
                "training_eligible": False, "stage_a_admissible": False,
                "stage_a_blockers": ["only_Numina_TIR_bucket", "anchor_content_not_attached",
                                     "external_evaluation_hash_index_not_independently_proven",
                                     "credential_free_disposable_vm_not_established",
                                     "v44fix_source_code_hash_contract_differs"],
                "isolation": {"kind": "kaggle_private_cpu_offline_config_requested",
                              "network_enforcement": "unverified", "credentials_absent": "unverified",
                              "drive_mounted": False, "security_proof": False}}
    _write_json_atomic(output / "dataset_manifest.json", manifest)
    receipt = {"schema": "geocpu_numina_offline_verification_provisional_v1",
               "dataset_manifest_sha256": sha_file(output / "dataset_manifest.json"),
               "source_lock_sha256": LOCK_SHA256, "verifier_source_sha256": common["verifier_source_sha256"],
               "worker_source_sha256": common["worker_source_sha256"],
               "row_receipts": row_receipts_pin, "stage_a_admissible": False,
               "isolation": manifest["isolation"]}
    _write_json_atomic(output / "offline_verification_receipt.json", receipt)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    shard = commands.add_parser("run-shard")
    shard.add_argument("--input-root", required=True)
    shard.add_argument("--config", required=True, help="relative JSON filename inside input root")
    shard.add_argument("--output", required=True)
    shard.add_argument("--prior", help="previous session's read-only output directory")
    merge = commands.add_parser("merge")
    merge.add_argument("--shard", action="append", required=True)
    merge.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "run-shard":
        root = Path(args.input_root)
        config = _load_json(_checked_file(root, args.config))
        if config.get("schema") != "geocpu_numina_shard_config_v1":
            raise SystemExit("Unexpected shard configuration schema")
        result = run_shard(root, config.get("input_files"), Path(args.output),
                           start=config["shard_start"], end=config["shard_end"],
                           wall_budget_seconds=config["wall_budget_seconds"],
                           prior=Path(args.prior) if args.prior else None)
        print(json.dumps({"complete": result["complete"], "next_index": result["next_index"],
                          "stage_a_admissible": False}, sort_keys=True))
    else:
        result = merge_shards([Path(path) for path in args.shard], Path(args.output))
        print(json.dumps({"source_rows": result["source_rows"], "counts": result["counts"],
                          "stage_a_admissible": False}, sort_keys=True))


if __name__ == "__main__":
    main()
