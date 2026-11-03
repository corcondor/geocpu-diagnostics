"""Pinned public TIR, code and Japanese instruction curriculum for v44.

This module never loads neural weights. Colab downloads the pinned public files;
all rows receive format, vocabulary, length, evaluation and question-split gates.
The default executes a fixed sample, NOT every public Numina observation. Each
row and the manifest distinguish author-recorded observations from local runs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import unicodedata

TARGET_COUNTS = {"tir": 105000, "code": 40000, "japanese": 40000, "anchor": 80000}
FENCE = re.compile(r"(?m)^```(?P<language>[^\n`]*)\n(?P<body>.*?)^```[ \t]*(?P<end>\n|$)", re.S)
LEGACY = re.compile(r"<(?:SET|ADD|SUB|MUL|DIV|MOD|MOV|LOAD|STORE|DEC|INC|AND|OR|XOR|NOT|SHL|SHR|CMP|JMP|JZ|JNZ|HALT)\b|\[R\d+\s*=|\bExactRAM\w*\b|\bKogge.Stone\b", re.I)
EVAL_PATTERNS = ("benchmark_*.jsonl", "*heldout*.jsonl", "eval_*.jsonl", "*manifest_test*.jsonl", "*manifest_dev*.jsonl")


class RowRejected(ValueError):
    pass


def sha_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def new_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def canonical_question(text):
    text = unicodedata.normalize("NFKC", str(text)).strip()
    text = re.sub(r"^(?:ユーザー|user)\s*[:：]\s*", "", text, flags=re.I)
    text = re.sub(r"\n\s*(?:アシスタント|assistant)\s*[:：]\s*$", "", text, flags=re.I)
    text = re.sub(r"^\s*【分野:[^\n]*】\s*", "", text)
    text = re.sub(r"(?m)^\s*分野\s*[:：][^\n]*\n?", "", text)
    text = re.sub(r"\s*(?:正解|回答|答え)\s*[:：]\s*$", "", text)
    return re.sub(r"\s+", "", text)


def split_question(key):
    # One global split rule, shared by every public source and old language anchor.
    value = int(sha_text(key)[:12], 16) % 100
    return "train" if value < 90 else "dev" if value < 95 else "test"


def json_records(path):
    path = Path(path)
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
            yield from batch.to_pylist()
    elif path.suffix == ".json":
        with path.open(encoding="utf-8-sig") as stream:
            records = json.load(stream)
        if not isinstance(records, list):
            raise ValueError("Public JSON dataset must contain a row list")
        yield from records
    else:
        with path.open(encoding="utf-8-sig") as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)


def load_vocab(path):
    metadata = json.loads(Path(path).read_text(encoding="utf-8"))
    vocab = metadata["vocab"]
    if not isinstance(vocab, dict):
        raise TypeError("Expected the original character-to-ID vocabulary")
    chars = {key for key, value in vocab.items() if len(key) == 1 and isinstance(value, int) and value >= 5}
    if not chars:
        raise ValueError("No legal vocabulary characters")
    return chars


def download_sources(source_lock, cache_dir):
    """Public, commit-pinned downloads; no token, API inference or new purchase."""
    from huggingface_hub import hf_hub_download
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    files, pins = {}, []
    for name, spec in source_lock["repos"].items():
        for split, record in spec["files"].items():
            path = Path(hf_hub_download(repo_id=spec["repo_id"], repo_type="dataset", revision=spec["revision"],
                                       filename=record["path"], cache_dir=str(cache_dir), token=False))
            actual = digest(path)
            if path.stat().st_size != record["bytes"] or actual != record["sha256"]:
                raise RuntimeError("Pinned public source differs: " + spec["repo_id"] + "/" + record["path"])
            files[(name, split)] = path
            pins.append({"source": name, "split": split, "repo_id": spec["repo_id"], "revision": spec["revision"],
                         "path": str(path), "bytes": path.stat().st_size, "sha256": actual, "license": spec["license"]})
    return files, pins


def evaluation_index(evaluation_dir, official_test):
    directory = Path(evaluation_dir)
    paths = sorted({p.resolve() for pattern in EVAL_PATTERNS for p in directory.glob(pattern) if p.is_file()})
    if not paths:
        raise RuntimeError("Tier evaluation manifests missing; preparation cannot exclude their questions")
    keys, ids, pins = set(), set(), []
    for path in paths:
        pins.append({"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size})
        for row in json_records(path):
            if not isinstance(row, dict):
                raise TypeError("Evaluation row is not an object")
            for field in ("prompt", "question", "input", "instruction", "problem"):
                if isinstance(row.get(field), str) and row[field].strip():
                    keys.add(canonical_question(row[field]))
            for field in ("id", "question_id", "instance_id"):
                if row.get(field) is not None:
                    ids.add(str(row[field]))
    official_count = 0
    for row in json_records(official_test):
        if not isinstance(row.get("problem"), str):
            raise TypeError("Numina official test problem field missing")
        keys.add(canonical_question(row["problem"]))
        official_count += 1
    if not keys or not official_count:
        raise RuntimeError("Evaluation questions or Numina official test split could not be indexed")
    pins.append({"path": str(official_test), "source": "Numina official test: never training", "rows": official_count,
                 "sha256": digest(official_test), "bytes": Path(official_test).stat().st_size})
    return keys, ids, pins


def _segments(length, tools):
    result, position = [], 0
    for begin, end in tools:
        if begin > position:
            result.append({"start": position, "end": begin, "role": "assistant", "train": True})
        result.append({"start": begin, "end": end, "role": "tool", "train": False})
        position = end
    if position < length:
        result.append({"start": position, "end": length, "role": "assistant", "train": True})
    return result or [{"start": 0, "end": length, "role": "assistant", "train": True}]


def add_v43_public_boundary(target, segments):
    """Restore the first supervised space in v43's prompt + ' ' + answer."""
    return " " + target, ([{"start": 0, "end": 1, "role": "assistant", "train": True}]
                         + [{**part, "start": part["start"] + 1, "end": part["end"] + 1}
                            for part in segments])


def normalize_tir(solution, *, executor=None):
    """Keep source prose/code; normalize only the tool observation protocol.

    If executor is None, observations are the published author's records and are
    explicitly unexecuted here. If supplied, every block in this problem runs in
    one persistent namespace and must match its published stdout exactly.
    """
    from tir_engine_v44r1 import observation_text
    if not isinstance(solution, str) or not solution.strip():
        raise RowRejected("empty_tir_solution")
    source = solution.replace("\r\n", "\n")
    fences = list(FENCE.finditer(source))
    if not fences or source.count("```") != 2 * len(fences):
        raise RowRejected("unclosed_or_nested_fence")
    pairs = []
    index = 0
    while index < len(fences):
        python = fences[index]
        language = python.group("language").strip().lower()
        if language not in ("python", "py"):
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
        index += 2
    if executor is not None:
        reset = executor.reset()
        if not isinstance(reset, dict) or reset.get("reset") is not True:
            raise RuntimeError("Python executor namespace reset was not acknowledged")
    parts, tools, receipts, position, length = [], [], [], 0, 0
    for python, output in pairs:
        close_end = python.end() - len(python.group("end"))
        prefix = source[position:close_end]
        parts.append(prefix)
        length += len(prefix)
        expected = output.group("body")
        if executor is None:
            result = {"success": True, "stdout": expected, "stderr": "", "error": None}
            verification = "public_author_recorded_not_executed_here"
        else:
            result = executor.execute(python.group("body"))
            if result.get("exception_type") in ("ModuleNotFoundError", "ImportError"):
                raise RuntimeError("Public TIR dependency unavailable: " + str(result.get("error")))
            if result.get("timeout") or result.get("truncated") or not result.get("success") or result.get("stderr"):
                raise RowRejected("executed_python_error_timeout_or_truncated")
            if str(result.get("stdout", "")).replace("\r\n", "\n") != expected:
                raise RowRejected("executed_stdout_differs_from_public_record")
            verification = "executed_all_blocks_in_this_problem_stdout_exact"
        observation = observation_text(result)
        begin = length
        parts.append(observation)
        length += len(observation)
        tools.append((begin, length))
        receipts.append({"code_sha256": sha_text(python.group("body")), "recorded_stdout_sha256": sha_text(expected),
                         "observation_sha256": sha_text(observation), "verification": verification})
        position = output.end()
    parts.append(source[position:])
    target = "".join(parts)
    return target, _segments(len(target), tools), receipts


def public_candidate(bucket, raw):
    if bucket == "tir":
        problem, target = raw.get("problem"), raw.get("solution")
        if not isinstance(problem, str) or not isinstance(target, str):
            raise RowRejected("numina_schema")
        messages = raw.get("messages")
        if not isinstance(messages, list) or len(messages) != 2 or messages[0].get("role") != "user" or messages[1].get("role") != "assistant":
            raise RowRejected("numina_message_roles")
        if messages[0].get("content") != problem or messages[1].get("content") != target:
            raise RowRejected("numina_message_text_mismatch")
        return {"question": problem, "prompt": "問題: " + problem + "\n思考:", "target": target, "language": "en"}
    if bucket == "code":
        problem, target = raw.get("problem"), raw.get("solution")
        if not isinstance(problem, str) or not isinstance(target, str):
            raise RowRejected("magicoder_schema")
        fences = list(FENCE.finditer(target.replace("\r\n", "\n")))
        languages = {f.group("language").strip().lower() for f in fences}
        # Python fences are executable actions in this model. Unpaired Python code
        # supervision would train an inconsistent action/observation protocol.
        if not fences or target.count("```") != 2 * len(fences) or not languages or languages & {"python", "py", "output", ""}:
            raise RowRejected("code_bucket_requires_non_python_language_fences")
        return {"question": problem, "prompt": "指示: " + problem + "\n回答:", "target": target,
                "language": ",".join(sorted(languages))}
    if bucket == "japanese":
        question, context, target = raw.get("instruction"), raw.get("input", ""), raw.get("output")
        if not isinstance(question, str) or not isinstance(context, str) or not isinstance(target, str):
            raise RowRejected("dolly_schema")
        content = ("本文:\n" + context + "\n質問:\n" if context else "") + question
        prompt = "指示: " + question + ("\n文脈: " + context if context else "") + "\n回答:"
        return {"question": content, "leak_question": question, "prompt": prompt,
                "target": target, "language": "ja", "category": raw.get("category")}
    raise ValueError("Unknown public bucket")


def anchor_candidate(raw):
    source = str(raw.get("source", ""))
    if source.startswith(("tir_", "jmmlu_knowledge_")) or source == "multilingual_code":
        raise RowRejected("old_v44_authored_or_benchmark_derived_not_anchor")
    prompt, target = raw.get("prompt"), raw.get("target")
    if not isinstance(prompt, str) or not isinstance(target, str):
        raise RowRejected("anchor_schema")
    if LEGACY.search(prompt + target):
        raise RowRejected("legacy_alu_protocol_excluded")
    if "```python" in target.lower() or "```output" in target.lower() or re.search(r"<SEARCH(?:\s|>)", prompt + target, re.I):
        raise RowRejected("legacy_anchor_tool_protocol_not_python_tir")
    original_target = target
    full = raw.get("full_text")
    if full is None:
        alignment = "explicit_prompt_target_without_full_text"
        separator = ""
    elif not isinstance(full, str):
        raise RowRejected("anchor_full_text_mismatch")
    elif full == prompt + target:
        alignment = "full_text_equals_prompt_plus_target"
        separator = ""
    elif full == prompt + " " + target:
        # Original v43 writers used f"{prompt} {target}" for these rows.
        # Keep the original prompt and extract its exact continuation. The
        # boundary space is source text, not a newly authored answer label.
        alignment = "full_text_equals_prompt_plus_space_plus_target"
        separator = " "
        target = full[len(prompt):]
    else:
        raise RowRejected("anchor_full_text_mismatch")
    if full is not None and prompt + target != full:
        raise RuntimeError("Lossless anchor continuation extraction failed")
    return {"question": prompt, "prompt": prompt, "target": target, "language": "legacy_language_anchor",
            "original_source": source, "source_target_sha256": sha_text(original_target),
            "source_full_text_sha256": sha_text(full) if full is not None else None,
            "anchor_boundary_alignment": alignment, "anchor_boundary_separator": separator}


def mixture_diversity_guard(target_counts, unique_questions, maximum_factor=16):
    """Operational guard against collapsing a large mixture into a few drills.

    This is not a claim that sixteen exposures establish model quality. The
    measured ratio uses distinct training question groups, not target variants.
    """
    if not isinstance(maximum_factor, (int, float)) or not math.isfinite(maximum_factor) or maximum_factor < 1:
        raise ValueError("Maximum exposure-to-unique-question factor must be finite and at least one")
    details = {}
    for bucket, exposures in target_counts.items():
        available = int(unique_questions.get(bucket, 0))
        required = math.ceil(exposures / maximum_factor)
        details[bucket] = {"target_exposures": exposures, "unique_training_questions": available,
                           "minimum_unique_questions_for_declared_exposures": required,
                           "exposure_to_unique_question_ratio": exposures / available if available else None,
                           "passed": available >= required}
    return {"passed": all(record["passed"] for record in details.values()), "maximum_factor": maximum_factor,
            "definition": "target exposure count divided by distinct admitted training question groups",
            "purpose": "Prevent excessive repeated exposure from silently replacing curriculum diversity; not a performance guarantee",
            "buckets": details}


def prepare_public_curriculum(output_dir, vocab_path, evaluation_dir, anchor_path, *, source_files=None,
                              source_lock_path=None, target_counts=None, executor_factory=None,
                              verify_sample_count=16, verify_all=False, max_total_tokens=8192,
                              max_repeat_factor_per_bucket=16):
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError("Public curriculum output already claimed; no overwrite or automatic retry")
    output.mkdir(parents=True, exist_ok=False)
    new_json(output / "preparation_claim.json", {"single_writer": True, "local_model_loads": 0, "model_training": False})
    lock_path = Path(source_lock_path) if source_lock_path else Path(__file__).with_name("public_source_lock_v44.json")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    new_json(output / "source_lock.json", lock)
    counts = dict(TARGET_COUNTS if target_counts is None else target_counts)
    if set(counts) != set(TARGET_COUNTS) or any(not isinstance(n, int) or n <= 0 for n in counts.values()):
        raise ValueError("All four exposure buckets must have positive integer targets")
    if not isinstance(verify_sample_count, int) or verify_sample_count < 0:
        raise ValueError("Execution sample count must be a nonnegative integer")
    stats, rejects, rejection_examples = Counter(), Counter(), []
    anchor_source_coverage = defaultdict(Counter)
    source_pins, eval_pins = [], []
    try:
        if source_files is None:
            source_files, source_pins = download_sources(lock, output / "source_download_cache")
        else:
            # This injection supports small deterministic format fixtures only.
            # Production files must also match the public lock before train admission.
            source_files = {(str(k[0]), str(k[1])): Path(p) for k, p in source_files.items()}
            for (name, split), path in source_files.items():
                pin = {"source": name, "split": split, "path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}
                expected = lock["repos"][name]["files"][split]
                pin["matches_public_lock"] = pin["bytes"] == expected["bytes"] and pin["sha256"] == expected["sha256"]
                source_pins.append(pin)
        required = {("numina", "train"), ("numina", "test"), ("code", "train"), ("japanese", "train")}
        if not required.issubset(source_files):
            raise RuntimeError("Pinned public source split missing")
        chars = load_vocab(vocab_path)
        eval_keys, eval_ids, eval_pins = evaluation_index(evaluation_dir, source_files[("numina", "test")])
        anchor_path = Path(anchor_path)
        anchor_pin = {"path": str(anchor_path), "bytes": anchor_path.stat().st_size, "sha256": digest(anchor_path),
                      "role": "Only prior-v43 language anchor rows; old v44 authored templates never used"}
        db = sqlite3.connect(output / "curriculum_state.sqlite")
        db.execute("CREATE TABLE rows(group_key TEXT, target_sha TEXT, bucket TEXT, split TEXT, payload TEXT, PRIMARY KEY(group_key,target_sha))")
        db.execute("CREATE INDEX row_split_bucket ON rows(split,bucket)")
        if executor_factory is None and (verify_all or verify_sample_count):
            from tir_engine_v44r1 import PythonExecutor
            executor_factory = lambda: PythonExecutor(timeout_seconds=5, max_output_chars=8192)
        context = executor_factory() if executor_factory else nullcontext(None)

        def reject(bucket, index, reason):
            rejects[bucket + ":" + reason] += 1
            if bucket == "anchor" and isinstance(raw, dict):
                anchor_source_coverage[str(raw.get("source", ""))]["rejected:" + reason] += 1
            if len(rejection_examples) < 50:
                rejection_examples.append({"bucket": bucket, "row_index": index, "reason": reason})

        with context as executor:
            for bucket, source_name, path in (("tir", "numina", source_files[("numina", "train")]),
                                               ("code", "code", source_files[("code", "train")]),
                                               ("japanese", "japanese", source_files[("japanese", "train")]),
                                               ("anchor", "anchor", anchor_path)):
                for index, raw in enumerate(json_records(path)):
                    stats[bucket + ":read"] += 1
                    if bucket == "anchor" and isinstance(raw, dict):
                        anchor_source_coverage[str(raw.get("source", ""))]["read_rows"] += 1
                    if stats[bucket + ":read"] % 5000 == 0:
                        print(f"Public preparation: {bucket} rows={stats[bucket + ':read']}, accepted_unique={stats[bucket + ':accepted_unique_rows']}", flush=True)
                    try:
                        if not isinstance(raw, dict):
                            raise RowRejected("row_not_object")
                        candidate = anchor_candidate(raw) if bucket == "anchor" else public_candidate(bucket, raw)
                        if bucket == "anchor":
                            stats["anchor:boundary:" + candidate["anchor_boundary_alignment"]] += 1
                        prompt, target = candidate["prompt"], candidate["target"]
                        if not prompt.strip() or not target.strip():
                            raise RowRejected("empty_prompt_or_target")
                        leak_values = (candidate["question"], prompt, candidate.get("leak_question", ""))
                        if any(canonical_question(text) in eval_keys for text in leak_values if text):
                            raise RowRejected("evaluation_prompt_exact_or_wrapper_overlap")
                        if any(str(raw.get(field)) in eval_ids for field in ("id", "question_id", "instance_id") if raw.get(field) is not None):
                            raise RowRejected("evaluation_id_overlap")
                        if LEGACY.search(prompt + target):
                            raise RowRejected("legacy_alu_protocol_excluded")
                        unknown = set(prompt + target) - chars
                        if unknown:
                            stats[bucket + ":unknown_character_occurrences"] += sum((prompt + target).count(c) for c in unknown)
                            raise RowRejected("unknown_vocabulary_character_whole_row_rejected")
                        boundary = "" if bucket == "anchor" else " "
                        if len(prompt) + len(boundary) + len(target) + 1 > max_total_tokens:
                            raise RowRejected("full_answer_and_true_eos_exceed_context")
                        source_target_sha = candidate.get("source_target_sha256", sha_text(target))
                        if bucket == "tir":
                            run = verify_all or stats["tir:sample_execution_attempted"] < verify_sample_count
                            if run:
                                stats["tir:sample_execution_attempted"] += 1
                            target, segments, receipts = normalize_tir(target, executor=executor if run else None)
                            verification = "executed_all_blocks_in_this_problem" if run else "public_author_recorded_not_executed_here"
                            if run:
                                stats["tir:sample_execution_passed"] += 1
                        else:
                            segments = _segments(len(target), [])
                            receipts, verification = [], "public_response_no_tool_observation" if bucket != "anchor" else "previous_language_anchor_no_alu"
                        if boundary:
                            # v43 full_text used prompt + " " + answer. Normalize TIR
                            # before adding this separator so a leading code fence is
                            # still recognized and tool spans retain their meaning.
                            target, segments = add_v43_public_boundary(target, segments)
                        if set(prompt + target) - chars:
                            raise RowRejected("normalized_tool_protocol_unknown_vocabulary")
                        if len(prompt) + len(target) + 1 > max_total_tokens:
                            raise RowRejected("normalized_full_answer_and_true_eos_exceed_context")
                        group = canonical_question(candidate["question"])
                        split = split_question(group)
                        record = {"prompt": prompt, "target": target, "full_text": prompt + target,
                                  "source": "tir_public_numina" if bucket == "tir" else "public_" + source_name,
                                  "bucket": bucket, "language": candidate["language"], "target_segments": segments,
                                  "provenance": {"public_repo": lock["repos"][source_name]["repo_id"] if source_name in lock["repos"] else None,
                                                 "revision": lock["repos"][source_name]["revision"] if source_name in lock["repos"] else None,
                                                 "source_row_index": index, "original_source": candidate.get("original_source"),
                                                 "source_full_text_sha256": candidate.get("source_full_text_sha256"),
                                                 "anchor_boundary_alignment": candidate.get("anchor_boundary_alignment"),
                                                 "anchor_boundary_separator": candidate.get("anchor_boundary_separator"),
                                                 "question_group_sha256": sha_text(group), "source_target_sha256": source_target_sha,
                                                 "target_sha256": sha_text(target), "split": split, "verification": verification,
                                                 "tool_blocks": receipts, "new_model_authored_example": False,
                                                 "protocol_normalized": bucket == "tir", "full_answer_and_true_eos": True}}
                        cursor = db.execute("INSERT OR IGNORE INTO rows VALUES(?,?,?,?,?)", (group, sha_text(target), bucket, split, json.dumps(record, ensure_ascii=False)))
                        if cursor.rowcount:
                            stats[bucket + ":accepted_unique_rows"] += 1
                            stats[bucket + ":language:" + candidate["language"]] += 1
                            if bucket == "anchor":
                                anchor_source_coverage[candidate["original_source"]]["accepted_unique_rows"] += 1
                                anchor_source_coverage[candidate["original_source"]]["accepted_unique_rows:" + split] += 1
                        else:
                            stats[bucket + ":exact_duplicate_rows"] += 1
                    except RowRejected as error:
                        reject(bucket, index, str(error))
                db.commit()
        pools = {}
        for bucket in counts:
            pools[bucket] = [row[0] for row in db.execute("SELECT rowid FROM rows WHERE split='train' AND bucket=? ORDER BY group_key,target_sha", (bucket,))]
        missing = [bucket for bucket, ids in pools.items() if not ids]
        split_counts = dict(db.execute("SELECT split,COUNT(*) FROM rows GROUP BY split"))
        input_pins_match = all(pin.get("matches_public_lock", True) for pin in source_pins)
        execution_sample_passed = not (verify_all or verify_sample_count) or stats["tir:sample_execution_passed"] > 0
        unique_train_questions = {bucket: db.execute("SELECT COUNT(DISTINCT group_key) FROM rows WHERE split='train' AND bucket=?", (bucket,)).fetchone()[0] for bucket in counts}
        diversity = mixture_diversity_guard(counts, unique_train_questions, max_repeat_factor_per_bucket)
        anchor_structure_understood = rejects["anchor:anchor_full_text_mismatch"] == 0
        blockers = []
        if missing:
            blockers.append("missing_training_buckets")
        if not all(split_counts.get(s, 0) for s in ("train", "dev", "test")):
            blockers.append("empty_question_split")
        if not input_pins_match:
            blockers.append("public_source_pin_mismatch")
        if not execution_sample_passed:
            blockers.append("no_executed_tir_sample_passed")
        if not anchor_structure_understood:
            blockers.append("unrecognized_original_anchor_structure")
        if not diversity["passed"]:
            blockers.append("excessive_exposure_to_unique_question_ratio")
        eligible = not blockers
        exposure = Counter()
        print("Writing the original-size public training mixture; repeats stay inside the training split." if eligible else
              "Curriculum admission blocked; no repeated training mixture will be materialized: " + ", ".join(blockers), flush=True)
        with (output / "train.jsonl").open("x", encoding="utf-8") as train:
            # Interleave all four exact exposure budgets without making a giant
            # in-memory target copy. Cycling is recorded, never called new data.
            for round_index in range(max(counts.values()) if eligible else 0):
                for bucket, target_count in counts.items():
                    if round_index >= target_count or not pools[bucket]:
                        continue
                    rowid = pools[bucket][round_index % len(pools[bucket])]
                    record = json.loads(db.execute("SELECT payload FROM rows WHERE rowid=?", (rowid,)).fetchone()[0])
                    record["provenance"]["exposure_index_in_bucket"] = round_index
                    record["provenance"]["repeat_cycle"] = round_index // len(pools[bucket])
                    train.write(json.dumps(record, ensure_ascii=False) + "\n")
                    exposure[bucket] += 1
        for split in ("dev", "test"):
            with (output / (split + ".jsonl")).open("x", encoding="utf-8") as stream:
                for (payload,) in db.execute("SELECT payload FROM rows WHERE split=? ORDER BY group_key,target_sha,bucket", (split,)):
                    stream.write(payload + "\n")
        groups = dict(db.execute("SELECT split,COUNT(DISTINCT group_key) FROM rows GROUP BY split"))
        db.close()
        manifest = {"schema": "geocpu_public_curriculum_v44_hotfix_v1", "training_eligible": bool(eligible),
                    "admission_blockers": blockers, "mixture_diversity_guard": diversity,
                    "original_anchor_structure_understood": anchor_structure_understood,
                    "anchor_source_coverage": {source: dict(values) for source, values in anchor_source_coverage.items()},
                    "counts": {"target_exposures": counts, "actual_train_exposures": dict(exposure),
                               "unique_rows_by_split": split_counts, "unique_question_groups_by_split": groups,
                               "unique_training_pool_by_bucket": {bucket: len(ids) for bucket, ids in pools.items()}},
                    "statistics": dict(stats), "rejections": dict(rejects), "rejection_examples": rejection_examples,
                    "missing_training_buckets": missing, "public_input_pins_match": input_pins_match,
                    "execution_sample_has_passed_problem": bool(execution_sample_passed),
                    "public_source_inputs": source_pins, "anchor_input": anchor_pin, "evaluation_inputs": eval_pins,
                    "source_lock_sha256": digest(lock_path), "vocabulary_sha256": digest(vocab_path),
                    "split_rule": "Global canonical question SHA256 mod100: train<90,dev<95,test>=95; repetitions remain within train only",
                    "tool_verification": {"mode": "all" if verify_all else "fixed_first_eligible_problem_sample",
                                          "sample_problem_limit": verify_sample_count, "unexecuted_rows_are_author_recorded": True,
                                          "all_observations_locally_executed": bool(verify_all),
                                          "mismatching_executed_rows_excluded_not_rewritten": True},
                    "scope": {"new_improvised_examples": 0, "old_v44_authored_templates_in_training": False,
                              "benchmark_derived_academic_rows_in_training": False, "legacy_alu_actions_in_training": False,
                              "previous_checkpoint_benchmark_exposure_erased": False,
                              "whole_answer_and_true_eos": True, "unknown_characters_silently_deleted": False,
                              "long_examples_truncated": False, "tir_language": "English", "japanese_language_source": "Japanese Dolly translated instruction responses",
                              "japanese_source_is_verified_72_field_cot": False, "all_code_solutions_correct_or_executed": False,
                              "semantic_paraphrase_leakage_certified_absent": False},
                    "outputs": {name: {"bytes": (output / name).stat().st_size, "sha256": digest(output / name)}
                                for name in ("train.jsonl", "dev.jsonl", "test.jsonl")}}
        new_json(output / "dataset_manifest.json", manifest)
        return manifest
    except BaseException as error:
        new_json(output / "preparation_error.json", {"type": type(error).__name__, "message": str(error), "automatic_retry": False})
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--vocab-path", required=True)
    parser.add_argument("--evaluation-dir", required=True)
    parser.add_argument("--anchor-path", required=True)
    parser.add_argument("--source-lock", default=str(Path(__file__).with_name("public_source_lock_v44.json")))
    parser.add_argument("--verify-sample-count", type=int, default=16)
    parser.add_argument("--verify-all", action="store_true")
    parser.add_argument("--max-repeat-factor-per-bucket", type=float, default=16)
    args = parser.parse_args()
    manifest = prepare_public_curriculum(args.output_dir, args.vocab_path, args.evaluation_dir, args.anchor_path,
                                         source_lock_path=args.source_lock, verify_sample_count=args.verify_sample_count, verify_all=args.verify_all,
                                         max_repeat_factor_per_bucket=args.max_repeat_factor_per_bucket)
    print(json.dumps({"training_eligible": manifest["training_eligible"], "counts": manifest["counts"], "rejections": manifest["rejections"]}, ensure_ascii=False), flush=True)
    if not manifest["training_eligible"]:
        raise SystemExit("Public curriculum incomplete or input pin mismatch; no training admission")


if __name__ == "__main__":
    main()
