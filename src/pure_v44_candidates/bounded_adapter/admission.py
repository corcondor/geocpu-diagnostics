"""Fail-closed admission for a future v43-parent Python-TIR training stage."""

from __future__ import annotations

import json
import hashlib
import re
import unicodedata
from collections import Counter
from pathlib import Path

from adapter import file_sha256


SCHEMA = "geocpu_bounded_python_tir_dataset_v1"
PINNED_PURE_SOURCE_MANIFEST_SHA256 = "7c072451921710d7a43ee3172957def3c279e40a21ae03ec6c416d07a84baebf"
MIN_UNIQUE_TRAIN = 50_000
MIN_PYTHON_CANDIDATES = 10_000
MIN_PUBLIC_TIR = 5_000
MIN_CONVERTED_CPU = 5_000
MIN_PUBLIC_MULTILINGUAL_CODE = 10_000
MIN_JAPANESE = 10_000
MIN_V43_TOOL_FREE_ANCHORS = 10_000
MIN_EVAL_PER_SPLIT = 64
MIN_UNSEEN_JAPANESE = 16
MIN_OLD_CPU_CASES = 12
VERIFIER_SCHEMA = "geocpu_independent_full_row_replay_v1"
VERIFIER_SOURCE_SHA256 = "325dbb2ba5eb0e73e5b01a7d6646e2724f0844e47b04bfbd5f27a96f844a4170"


def canonical_prompt(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip()


def source_ref_fields(ref: dict) -> tuple[str, str, int, str]:
    """Normalize original v43 provenance without rewriting its saved rows."""
    if not isinstance(ref, dict):
        raise ValueError("Missing original source reference")
    if set(ref) == {"zip_member", "line_number_1based", "raw_line_sha256"}:
        kind, name, line, sha = ("v43_zip_member", ref["zip_member"],
                                 ref["line_number_1based"], ref["raw_line_sha256"])
    else:
        kind, name, line, sha = (ref.get("kind"), ref.get("file"),
                                 ref.get("line"), ref.get("raw_sha256"))
    minimum = 1 if kind == "v43_zip_member" else 0
    if (kind not in ("v43_zip_member", "public_parquet", "public_json", "public_jsonl")
            or not isinstance(name, str) or not name or type(line) is not int or line < minimum
            or not re.fullmatch(r"[0-9a-f]{64}", str(sha))):
        raise ValueError("Invalid original source reference")
    return kind, name, line, sha


def _lines(path: Path):
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{number}")
            yield number, row


def verify_source_manifest(source_dir: str | Path, expected_sha256: str) -> dict:
    if expected_sha256 != PINNED_PURE_SOURCE_MANIFEST_SHA256:
        raise ValueError("Only the previously reviewed pure-v44 Hotfix source is admitted")
    source_dir = Path(source_dir)
    manifest_path = source_dir / "hotfix_manifest.json"
    if file_sha256(manifest_path) != expected_sha256:
        raise ValueError("Pinned pure-v44 source manifest SHA differs")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = ("GeoCPU_CED_v1.py", "GeoCPU_CED_v2.py", "neural_model_v30_python_base.py",
                "integrated_model_v44_python.py", "tir_engine_v44r1.py", "training_data_v44.py",
                "checkpoint_recovery_v44r1.py", "executor_worker_v44r1.py",
                "prepare_public_v44.py", "public_source_lock_v44.json")
    for name in required:
        record = manifest.get("files", {}).get(name)
        if not isinstance(record, dict) or file_sha256(source_dir / name) != record.get("sha256"):
            raise ValueError(f"Pinned pure-v44 source file differs: {name}")
    return {"manifest_sha256": expected_sha256, "checked_source_files": list(required)}


def verify_dataset(data_dir: str | Path, expected_manifest_sha256: str) -> dict:
    data_dir = Path(data_dir)
    manifest_path = data_dir / "dataset_manifest.json"
    if file_sha256(manifest_path) != expected_manifest_sha256:
        raise ValueError("Training dataset manifest SHA differs")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA:
        raise ValueError("Uncertified dataset schema")
    files = manifest.get("files", {})
    required = ("train.jsonl", "old_skill_eval.jsonl", "unseen_eval.jsonl",
                "v43_original_answers.jsonl")
    for name in required:
        record = files.get(name)
        if not isinstance(record, dict) or file_sha256(data_dir / name) != record.get("sha256"):
            raise ValueError(f"Dataset file SHA differs: {name}")
    audit = manifest.get("quality_audit", {})
    # Dataset-authored quality flags never substitute for independent replay.
    if audit.get("source_family_separation_checked") is not True:
        raise ValueError("Train and evaluation family separation was not certified")

    train_prompts: set[str] = set()
    train_families: set[str] = set()
    eval_prompts: set[str] = set()
    eval_families: set[str] = set()
    train_ids: set[str] = set()
    eval_ids: set[str] = set()
    counts = Counter()
    originals = {}
    for number, row in _lines(data_dir / "v43_original_answers.jsonl"):
        if (row.get("id") in originals or row.get("profile") != "v43_cpu_native"
                or row.get("parent_sha256") != "a4b95d7d0f486146196271b3b1fbc6a1b6b3342d8fadd590320fc3842c1e80e5"
                or row.get("correct") is not True
                or not isinstance(row.get("raw_output"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", str(row.get("prompt_sha256", "")))
                or hashlib.sha256(row["raw_output"].encode("utf-8")).hexdigest() != row.get("raw_output_sha256")):
            raise ValueError(f"Original v43 answer proof incomplete at v43_original_answers.jsonl:{number}")
        originals[row["id"]] = row
    for number, row in _lines(data_dir / "train.jsonl"):
        prompt, target = row.get("prompt"), row.get("target")
        family, row_id, bucket = row.get("family_id"), row.get("id"), row.get("bucket")
        if not all(isinstance(x, str) and x for x in (prompt, target, family, row_id, bucket)):
            raise ValueError(f"Incomplete training lineage at train.jsonl:{number}")
        if bucket not in ("public_python_tir", "converted_legacy_python", "japanese_prose",
                          "v43_tool_free_anchor", "public_multilingual_code"):
            raise ValueError(f"Unknown training bucket at train.jsonl:{number}")
        try:
            source_ref_fields(row.get("source_ref"))
        except ValueError:
            raise ValueError(f"Training row lacks an original source reference at train.jsonl:{number}")
        if row_id in train_ids:
            raise ValueError(f"Duplicate training instance ID at train.jsonl:{number}")
        train_ids.add(row_id)
        train_prompts.add(canonical_prompt(prompt))
        train_families.add(family)
        counts["train_rows"] += 1
        counts["train_" + bucket] += 1
        if bucket in ("public_python_tir", "converted_legacy_python"):
            counts["python_candidate_rows"] += 1
    if len(train_prompts) < MIN_UNIQUE_TRAIN or counts["python_candidate_rows"] < MIN_PYTHON_CANDIDATES:
        raise ValueError("Too few unique prompts or Python candidates for a full stage")
    if counts["train_japanese_prose"] < MIN_JAPANESE:
        raise ValueError("Too few Japanese prose rows for a general-language stage")
    if counts["train_v43_tool_free_anchor"] < MIN_V43_TOOL_FREE_ANCHORS:
        raise ValueError("Too few original v43 tool-free language/knowledge anchors")
    if counts["train_public_python_tir"] < MIN_PUBLIC_TIR:
        raise ValueError("Too few original-public, executed Python TIR rows")
    if counts["train_converted_legacy_python"] < MIN_CONVERTED_CPU:
        raise ValueError("Too few strictly converted legacy CPU rows")
    if counts["train_public_multilingual_code"] < MIN_PUBLIC_MULTILINGUAL_CODE:
        raise ValueError("Too few original-public non-Python code rows")

    for split in ("old_skill_eval", "unseen_eval"):
        split_count = 0
        for number, row in _lines(data_dir / (split + ".jsonl")):
            prompt, family, row_id, expected, bucket = (row.get("prompt"), row.get("family_id"),
                                                       row.get("id"), row.get("expected_answer"),
                                                       row.get("bucket"))
            if not all(isinstance(x, str) and x for x in (prompt, family, row_id, expected, bucket)):
                raise ValueError(f"Incomplete evaluation row at {split}.jsonl:{number}")
            if split == "old_skill_eval":
                if type(row.get("v43_cpu_task")) is not bool:
                    raise ValueError(f"Old-skill row lacks CPU-task classification at {split}.jsonl:{number}")
                if row["v43_cpu_task"]:
                    counts["old_skill_eval_original_cpu_tasks"] += 1
                if row.get("v43_original_correct") is not True or not re.fullmatch(
                    r"[0-9a-f]{64}", str(row.get("v43_original_raw_sha256", ""))
                ):
                    raise ValueError(f"Old-skill row lacks a correct, pinned v43 original answer at {split}.jsonl:{number}")
                original = originals.get(row_id)
                if (original is None or original.get("prompt_sha256") != hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                        or original.get("raw_output_sha256") != row["v43_original_raw_sha256"]
                        or original.get("expected_answer") != expected):
                    raise ValueError(f"Old-skill row differs from pinned original v43 raw answer at {split}.jsonl:{number}")
            normalized = canonical_prompt(prompt)
            if normalized in train_prompts or normalized in eval_prompts:
                raise ValueError(f"Evaluation prompt overlap at {split}.jsonl:{number}")
            if family in train_families or family in eval_families or row_id in train_ids or row_id in eval_ids:
                raise ValueError(f"Evaluation lineage overlap at {split}.jsonl:{number}")
            eval_prompts.add(normalized)
            eval_families.add(family)
            eval_ids.add(row_id)
            split_count += 1
            counts[split + "_" + bucket] += 1
        if split_count < MIN_EVAL_PER_SPLIT:
            raise ValueError(f"Too few independent evaluation rows: {split}")
        counts[split + "_rows"] = split_count
    if len(originals) != counts["old_skill_eval_rows"]:
        raise ValueError("Original v43 answer proof must cover exactly the old-skill evaluation rows")
    counts["v43_original_answers_rows"] = len(originals)
    if counts["unseen_eval_japanese_prose"] < MIN_UNSEEN_JAPANESE:
        raise ValueError("Too few separate Japanese prose evaluation problems")
    if counts["old_skill_eval_original_cpu_tasks"] < MIN_OLD_CPU_CASES:
        raise ValueError("Too few originally solved v43 CPU cases to assess Python replacement")

    declared = manifest.get("counts", {})
    if any(declared.get(key) != value for key, value in counts.items()):
        raise ValueError("Dataset manifest counts differ from the actual files")
    return {"manifest_sha256": expected_manifest_sha256, "counts": dict(counts),
            "unique_train_prompts": len(train_prompts), "train_family_count": len(train_families),
            "eval_family_count": len(eval_families), "files_sha256": {name: files[name]["sha256"] for name in required},
            "execution_verification": "UNVERIFIED_SELF_ATTESTATION_ONLY"}


def require_independent_execution_verification(data: dict, *, data_dir: str | Path,
                                               proof_dir: str | Path | None,
                                               expected_proof_sha256: str | None,
                                               expected_meta_sha256: str,
                                               expected_source_manifest_sha256: str) -> dict:
    """Require a separate replay proof for every exact training row.

    The public and converted rows are actually executed by independent_verify,
    which checks source hashes and original CPU observations. This admission
    checks its source version, result closure and exact row coverage. Neither
    author-supplied booleans nor 64-hex receipt-looking strings suffice.
    """
    if proof_dir is None or not expected_proof_sha256:
        raise RuntimeError("Independent full-row replay proof is missing; GPU work is blocked")
    if file_sha256(Path(__file__).with_name("independent_verify.py")) != VERIFIER_SOURCE_SHA256:
        raise ValueError("Pinned independent verifier source has changed")
    proof_dir = Path(proof_dir)
    proof_path = proof_dir / "proof.json"
    receipts_path = proof_dir / "receipts.jsonl"
    if file_sha256(proof_path) != expected_proof_sha256:
        raise ValueError("Independent replay proof SHA differs")
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    if (proof.get("schema") != VERIFIER_SCHEMA or proof.get("status") != "ALL_ROWS_REPLAYED"
            or proof.get("verifier_source_sha256") != VERIFIER_SOURCE_SHA256
            or proof.get("data_manifest_sha256") != data["manifest_sha256"]
            or proof.get("train_sha256") != data["files_sha256"]["train.jsonl"]
            or proof.get("meta_sha256") != expected_meta_sha256
            or proof.get("pure_source_manifest_sha256") != expected_source_manifest_sha256
            or proof.get("train_rows") != data["counts"]["train_rows"]
            or not re.fullmatch(r"[0-9a-f]{64}", str(proof.get("converter_source_sha256", "")))
            or not re.fullmatch(r"[0-9a-f]{64}", str(proof.get("v43_zip_sha256", "")))
            or file_sha256(receipts_path) != proof.get("receipts_sha256")):
        raise ValueError("Independent replay proof is incomplete or has wrong lineage")
    manifest = json.loads((Path(data_dir) / "dataset_manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("source_files", {}).get("v43_zip_sha256") != proof["v43_zip_sha256"]
            or manifest.get("verification", {}).get("converter_source_sha256") != proof["converter_source_sha256"]
            or manifest.get("verification", {}).get("verifier_source_sha256") != VERIFIER_SOURCE_SHA256):
        raise ValueError("Replay verifier, converter or original archive is not pinned by the dataset")
    tool_calls = 0
    with (Path(data_dir) / "train.jsonl").open("rb") as source, receipts_path.open("rb") as receipts:
        train_lines = (line for line in source if line.strip())
        receipt_lines = (line for line in receipts if line.strip())
        seen = 0
        for row_line, receipt_line in zip(train_lines, receipt_lines, strict=True):
            row = json.loads(row_line)
            receipt = json.loads(receipt_line)
            _, _, _, source_sha = source_ref_fields(row.get("source_ref"))
            if (receipt.get("id") != row.get("id")
                    or receipt.get("train_row_sha256") != hashlib.sha256(row_line).hexdigest()
                    or receipt.get("source_row_sha256") != source_sha
                    or receipt.get("bucket") != row.get("bucket")
                    or type(receipt.get("tool_calls_replayed")) is not int
                    or not re.fullmatch(r"[0-9a-f]{64}", str(receipt.get("observations_sha256", "")))):
                raise ValueError("Independent replay receipt does not bind to an exact training row")
            if row["bucket"] in ("public_python_tir", "converted_legacy_python") and receipt["tool_calls_replayed"] < 1:
                raise ValueError("Tool training row has no independently replayed call")
            if row["bucket"] in ("japanese_prose", "v43_tool_free_anchor", "public_multilingual_code") and receipt["tool_calls_replayed"] != 0:
                raise ValueError("Tool-free row unexpectedly contains a replayed call")
            tool_calls += receipt["tool_calls_replayed"]
            seen += 1
    if seen != proof["train_rows"] or tool_calls != proof.get("replayed_tool_calls"):
        raise ValueError("Independent replay proof row count or call count differs")
    return {"proof_sha256": expected_proof_sha256, "verified_rows": seen,
            "replayed_tool_calls": tool_calls, "verifier_source_sha256": VERIFIER_SOURCE_SHA256}


def require_v43_native_baseline(data_dir: str | Path, *, proof_dir: str | Path | None,
                                expected_capture_sha256: str | None,
                                expected_grade_sha256: str | None) -> dict:
    """Bind old-skill cases to actual full v43 CPU-native outputs and grading.

    The current workspace has no such complete record. Missing proof always
    blocks admission; a self-declared correct flag in dataset JSON is never
    sufficient to prove retention of originally solved questions.
    """
    if proof_dir is None or not expected_capture_sha256 or not expected_grade_sha256:
        raise RuntimeError("Complete pinned original-v43 CPU-native answers and external grades are missing")
    proof_dir = Path(proof_dir)
    captured = proof_dir / "v43_native_full_answers.jsonl"
    graded = proof_dir / "v43_native_external_grades.jsonl"
    if file_sha256(captured) != expected_capture_sha256 or file_sha256(graded) != expected_grade_sha256:
        raise ValueError("Pinned original-v43 native capture or external grade SHA differs")
    cases = {row["id"]: row for _, row in _lines(Path(data_dir) / "old_skill_eval.jsonl")}
    originals = {row["id"]: row for _, row in _lines(Path(data_dir) / "v43_original_answers.jsonl")}
    answer_rows = [row for _, row in _lines(captured)]
    grade_rows = [row for _, row in _lines(graded)]
    answers = {row["id"]: row for row in answer_rows}
    grades = {row["id"]: row for row in grade_rows}
    if (len(answer_rows) != len(cases) or len(grade_rows) != len(cases)
            or set(cases) != set(originals) or set(cases) != set(answers) or set(cases) != set(grades)):
        raise ValueError("Original-v43 native proof does not cover every fixed old-skill case")
    for ident, case in cases.items():
        answer, grade = answers[ident], grades[ident]
        text = answer.get("raw_output")
        if (answer.get("profile") != "v43_cpu_native"
                or answer.get("parent_sha256") != "a4b95d7d0f486146196271b3b1fbc6a1b6b3342d8fadd590320fc3842c1e80e5"
                or answer.get("full_answer_captured") is not True
                or not isinstance(text, str)
                or answer.get("prompt") != case["prompt"]
                or hashlib.sha256(text.encode("utf-8")).hexdigest() != case["v43_original_raw_sha256"]
                or originals[ident].get("raw_output") != text
                or grade.get("prompt_sha256") != hashlib.sha256(case["prompt"].encode("utf-8")).hexdigest()
                or grade.get("raw_output_sha256") != case["v43_original_raw_sha256"]
                or grade.get("expected_answer") != case["expected_answer"]
                or grade.get("correct") is not True
                or not isinstance(grade.get("reason"), str) or not grade["reason"]):
            raise ValueError("Original-v43 native capture/grade differs from fixed old-skill row: " + ident)
    return {"old_skill_originally_correct": len(cases),
            "capture_sha256": expected_capture_sha256, "external_grade_sha256": expected_grade_sha256}
