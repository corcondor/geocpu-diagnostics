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
PINNED_V43_DATA_ZIP_SHA256 = "0102dc9b01b4b795c3123b18cefc7a8734e47a06dd85292af464d3232704d198"
PINNED_V43_PARENT_SHA256 = "a4b95d7d0f486146196271b3b1fbc6a1b6b3342d8fadd590320fc3842c1e80e5"
PINNED_V43_SWE_SOURCE_SHA256 = "0bf62099ea2a41c8eb35c1ebdc2584f2c164ae3c82f741ab3eca44113f70f8cd"
PINNED_V43_SWE_EVALUATOR_SHA256 = "c3589b97f531b6460a95907e56284df5f20be990eff1a4d2e56a921b4c705899"
PINNED_V43_META_SHA256 = "7efa8dea9a0100e2250f72805c89cca7a281efec30b4479795743814756cdcd5"
PINNED_V43_CAPTURE_SOURCE_SHA256 = "6bcf453c32cacff89d3e6b3062162c3ae43f6027fe8f6b3336c8c2c8740fc12d"
MIN_UNIQUE_TRAIN = 50_000
MIN_PYTHON_CANDIDATES = 10_000
MIN_PUBLIC_TIR = 5_000
MIN_CONVERTED_CPU = 10_000
MIN_PUBLIC_MULTILINGUAL_CODE = 10_000
MIN_JAPANESE = 10_000
MIN_V43_TOOL_FREE_ANCHORS = 10_000
MIN_EVAL_PER_SPLIT = 64
MIN_UNSEEN_JAPANESE = 16
MIN_OLD_CPU_CASES = 12
VERIFIER_SCHEMA = "geocpu_independent_full_row_replay_v1"
VERIFIER_SOURCE_SHA256 = "1c3106f32b557674d2a035f6093870260a2f98197866fa655ba3e85ca8c627a3"
REQUIRED_V43_EVAL_SHA256 = {
    "eval_v42_search_retrieval_aligned.jsonl": "c07d962d4fa59196d74b8223353a5f52d60cabe3d58bfe604c9d83b708e45ab6",
    "eval_v42_search_retrieval_heldout.jsonl": "805d4b91ef9128095191cdeb1180e95b0759196f3d471204d2e47c6f77e6ef53",
    "eval_swe_step_control_heldout.jsonl": "0bf62099ea2a41c8eb35c1ebdc2584f2c164ae3c82f741ab3eca44113f70f8cd",
    "eval_v38_decomposed_state_bench200.jsonl": "e96db14ba2b6ec620d42bcaf598199da814d93cef971c334889bd88147567cd1",
    "eval_v33_state_tracking_heldout.jsonl": "6406c92c106512d2046368d6acba2dfb2ff24181301a14e975d76700ab7ab7b5",
    "jamcqa_manifest_test.jsonl": "f7590adbce3d8f444b089d102a7f79be091c942c7c132d7090fd4e3d0710543e",
    "benchmark_8100_manifest.jsonl": "33723d78aaa1595b1ed7dbc3382ecb165ce788246f6f70a63470f5fa107de78e",
    "eval_v41_logical_instruction_operators_heldout.jsonl": "b887aade00323a7241088a4e84dfcb166430f554209a09668ad204ea564e9989",
}


def canonical_prompt(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip()


def canonical_question(value: str) -> str:
    """Mirror the pinned public builder's family key for every split."""
    text = unicodedata.normalize("NFKC", value).strip()
    text = re.sub(r"^(?:ユーザー|user)\s*[:：]\s*", "", text, flags=re.I)
    text = re.sub(r"\n\s*(?:アシスタント|assistant)\s*[:：]\s*$", "", text, flags=re.I)
    text = re.sub(r"^\s*【分野:[^\n]*】\s*", "", text)
    text = re.sub(r"(?m)^\s*分野\s*[:：][^\n]*\n?", "", text)
    text = re.sub(r"\s*(?:正解|回答|答え)\s*[:：]\s*$", "", text)
    return re.sub(r"\s+", "", text)


def family_id_for_prompt(value: str) -> str:
    return hashlib.sha256(("family:" + canonical_question(value)).encode("utf-8")).hexdigest()


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
    if manifest.get("source_files", {}).get("v43_zip_sha256") != PINNED_V43_DATA_ZIP_SHA256:
        raise ValueError("Original v43 training ZIP is not the pinned archive")
    evaluation_pins = manifest.get("evaluation_input_pins")
    if not isinstance(evaluation_pins, list):
        raise ValueError("Fixed v43 evaluation manifests have not been pinned")
    pinned_eval = {Path(item["path"]).name: item["sha256"] for item in evaluation_pins
                   if isinstance(item, dict) and isinstance(item.get("path"), str)
                   and isinstance(item.get("sha256"), str)}
    if any(pinned_eval.get(name) != expected for name, expected in REQUIRED_V43_EVAL_SHA256.items()):
        raise ValueError("A required original v43 evaluation manifest is missing or changed")
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
    train_questions: set[str] = set()
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
        if family != family_id_for_prompt(prompt):
            raise ValueError(f"Training family ID differs from canonical question at train.jsonl:{number}")
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
        if canonical_question(prompt) in train_questions:
            raise ValueError(f"Duplicate canonical training question at train.jsonl:{number}")
        train_questions.add(canonical_question(prompt))
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
            if family != family_id_for_prompt(prompt):
                raise ValueError(f"Evaluation family ID differs from canonical question at {split}.jsonl:{number}")
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
            question = canonical_question(prompt)
            if normalized in train_prompts or normalized in eval_prompts or question in train_questions:
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
    if expected_meta_sha256 != PINNED_V43_META_SHA256:
        raise ValueError("Tokenizer metadata must be the fixed original v43 metadata")
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
            or proof.get("v43_zip_sha256") != PINNED_V43_DATA_ZIP_SHA256
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
    all_captured = proof_dir / "swe200_full_answers.jsonl"
    all_grades = proof_dir / "swe200_original_grades.jsonl"
    original_audit = proof_dir / "original_swe_audit.jsonl"
    claim_path = proof_dir / "claim.json"
    if file_sha256(Path(__file__).with_name("v43_native_swe_capture_readonly.py")) != PINNED_V43_CAPTURE_SOURCE_SHA256:
        raise ValueError("Original v43 capture source SHA differs")
    claim = json.loads(claim_path.read_text(encoding="utf-8"))
    if (claim.get("source_zip_sha256") != PINNED_V43_DATA_ZIP_SHA256
            or claim.get("parent_sha256") != PINNED_V43_PARENT_SHA256
            or claim.get("meta_sha256") != PINNED_V43_META_SHA256
            or claim.get("eval_member_sha256") != PINNED_V43_SWE_SOURCE_SHA256
            or claim.get("original_evaluator_sha256") != PINNED_V43_SWE_EVALUATOR_SHA256
            or claim.get("script_sha256") != PINNED_V43_CAPTURE_SOURCE_SHA256
            or claim.get("suite") != "heldout_swe200"
            or claim.get("generation_limit") != 2048
            or claim.get("training_updates") != 0
            or claim.get("parent_checkpoint_modified") is not False):
        raise ValueError("Original v43 capture claim differs from pinned source/profile")
    if file_sha256(captured) != expected_capture_sha256 or file_sha256(graded) != expected_grade_sha256:
        raise ValueError("Pinned original-v43 native capture or external grade SHA differs")
    for name in ("old_skill_eval.jsonl", "v43_original_answers.jsonl"):
        if file_sha256(Path(data_dir) / name) != file_sha256(proof_dir / name):
            raise ValueError("Dataset old-skill source differs from native v43 capture: " + name)
    full_rows = [row for _, row in _lines(all_captured)]
    audit_rows = [row for _, row in _lines(original_audit)]
    full_grade_rows = [row for _, row in _lines(all_grades)]
    if not (len(full_rows) == len(audit_rows) == len(full_grade_rows) == 200):
        raise ValueError("Original v43 evaluator did not cover exactly SWE200")
    accepted_ids = set()
    all_ids = set()
    for index, (answer, audit, grade) in enumerate(zip(full_rows, audit_rows, full_grade_rows, strict=True)):
        raw = answer.get("raw_output")
        trace = answer.get("cpu_trace")
        if not isinstance(trace, list) or any(not isinstance(step, dict) for step in trace):
            raise ValueError(f"Original v43 CPU trace malformed at SWE row {index}")
        r0_writes = [step["val"] for step in trace if step.get("dest") == "R0"]
        cpu_final_val = str(r0_writes[-1]) if r0_writes else None
        cleaned = re.sub(r"<[^>]+>(?:\s*=>\s*\[[^\]]+\])?", "", raw.strip()).strip() if isinstance(raw, str) else ""
        text_emitted_ans = cleaned.split()[-1] if cleaned.split() else ""
        reconstructed_final = (cpu_final_val if answer.get("cpu_invocations", 0) > 0
                               and cpu_final_val is not None else text_emitted_ans)
        reconstructed_correct = reconstructed_final == answer.get("expected_answer")
        if (answer.get("id") != audit.get("id") or answer.get("id") != grade.get("id")
                or answer.get("source_index") != index
                or answer.get("source_member_sha256") != PINNED_V43_SWE_SOURCE_SHA256
                or answer.get("parent_sha256") != PINNED_V43_PARENT_SHA256
                or answer.get("profile") != "v43_cpu_native"
                or answer.get("generation_limit") != 2048
                or answer.get("full_answer_captured") is not True
                or answer.get("reference_given_to_generation") is not False
                or not isinstance(raw, str)
                or answer.get("raw_output_sha256") != hashlib.sha256(raw.encode("utf-8")).hexdigest()
                or audit.get("raw_output") != raw.strip()
                or audit.get("target") != answer.get("expected_answer")
                or audit.get("cpu_invocations") != answer.get("cpu_invocations")
                or audit.get("ended_with_eos") != answer.get("ended_with_eos")
                or grade.get("scorer_sha256") != PINNED_V43_SWE_EVALUATOR_SHA256
                or grade.get("prompt_sha256") != answer.get("prompt_sha256")
                or grade.get("raw_output_sha256") != answer.get("raw_output_sha256")
                or grade.get("expected_answer") != answer.get("expected_answer")
                or grade.get("correct") != audit.get("is_correct")
                or audit.get("is_correct") != reconstructed_correct
                or audit.get("final_answer") != reconstructed_final
                or grade.get("final_answer") != audit.get("final_answer")
                or grade.get("expected_loops") != audit.get("expected_loops")
                or grade.get("executed_loops") != audit.get("executed_loops")
                or grade.get("step_category") != audit.get("step_category")):
            raise ValueError(f"Original v43 full answer/evaluator audit mismatch at SWE row {index}")
        if answer["id"] in all_ids:
            raise ValueError("Original v43 SWE IDs are not unique")
        all_ids.add(answer["id"])
        if grade["correct"] is True and answer["ended_with_eos"] is True:
            accepted_ids.add(answer["id"])
    cases = {row["id"]: row for _, row in _lines(Path(data_dir) / "old_skill_eval.jsonl")}
    originals = {row["id"]: row for _, row in _lines(Path(data_dir) / "v43_original_answers.jsonl")}
    answer_rows = [row for _, row in _lines(captured)]
    grade_rows = [row for _, row in _lines(graded)]
    answers = {row["id"]: row for row in answer_rows}
    grades = {row["id"]: row for row in grade_rows}
    if (len(answer_rows) != len(cases) or len(grade_rows) != len(cases)
            or set(cases) != set(originals) or set(cases) != set(answers) or set(cases) != set(grades)):
        raise ValueError("Original-v43 native proof does not cover every fixed old-skill case")
    if set(cases) != accepted_ids:
        raise ValueError("Fixed old-skill cases differ from original v43 SWE correct subset")
    for ident, case in cases.items():
        answer, grade = answers[ident], grades[ident]
        text = answer.get("raw_output")
        if (answer.get("profile") != "v43_cpu_native"
                or answer.get("parent_sha256") != PINNED_V43_PARENT_SHA256
                or answer.get("full_answer_captured") is not True
                or not isinstance(text, str)
                or answer.get("prompt") != case["prompt"]
                or hashlib.sha256(text.encode("utf-8")).hexdigest() != case["v43_original_raw_sha256"]
                or originals[ident].get("raw_output") != text
                or grade.get("prompt_sha256") != hashlib.sha256(case["prompt"].encode("utf-8")).hexdigest()
                or grade.get("raw_output_sha256") != case["v43_original_raw_sha256"]
                or grade.get("expected_answer") != case["expected_answer"]
                or grade.get("correct") is not True
                or grade.get("scorer_sha256") != PINNED_V43_SWE_EVALUATOR_SHA256
                or not isinstance(grade.get("reason"), str) or not grade["reason"]):
            raise ValueError("Original-v43 native capture/grade differs from fixed old-skill row: " + ident)
        full = full_rows[answer["source_index"]]
        if answer != full or grade != full_grade_rows[answer["source_index"]]:
            raise ValueError("Accepted native subset differs from the original SWE200 capture: " + ident)
    return {"old_skill_originally_correct": len(cases),
            "capture_sha256": expected_capture_sha256, "external_grade_sha256": expected_grade_sha256,
            "original_evaluator_sha256": PINNED_V43_SWE_EVALUATOR_SHA256,
            "capture_source_sha256": PINNED_V43_CAPTURE_SOURCE_SHA256,
            "all_swe_cases": len(full_rows)}
