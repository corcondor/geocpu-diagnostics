"""Read-only full raw answers for the same 44 interim questions, before/after.

This is a collapse detector between 250-update blocks.  It does not replace
the original Tier 1–8 suite or independent Japanese reasoning grades.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sys
import time
import zipfile

from source_pools import PINNED_V43_ZIP_SHA, PINNED_CONVERTED_SHA, PYTHON_MODE_CUE, file_sha
from prelaunch_contract import verify_prelaunch_contract
from train_hybrid_stage import (_load_stage_checkpoint, _prepare_source,
                                _verify_extracted_source, SOURCE_OVERLAY_SHA,
                                PARENT_CHECKPOINT_BYTES, PARENT_CHECKPOINT_SHA)

CASES_SHA256 = "4032a2c17de2ce56a7758cd35c16db2c2b357ce76b394522411e86c0aa682b45"
CASES_NAME = "fixed_interim_diagnostic_44_v2.jsonl"
FP16_DERIVED_SOURCE_SHA = {
    "integrated_model_v42.py": "b0399051354edc8348b6dfe61b46ce1874ae2e59fa01a11ae852eb076f72ba30",
    "integrated_model_v44fix.py": "41ce599ffbe15455c6a3c8167d1357fb5d50c007e2bb04195620992f2c2b22d5",
}
_BF16_LITERAL = b"dtype=torch.bfloat16"
_FP16_LITERAL = b"dtype=torch.float16"


def _verify_hybrid_precision_binding(binding: dict, contract: dict) -> None:
    corresponding = {"train_bf16_native": "eval_bf16_native",
                     "train_fp16_gradscaler_v1": "eval_fp16_native_v1"}
    expected_eval = corresponding.get(binding.get("precision_profile"))
    if expected_eval is None or contract.get("precision_profile") != expected_eval:
        raise ValueError("Training and diagnostic precision profiles differ")
    for bound_name, claimed_name in (
        ("original_swe200_baseline_complete_sha256", "parent_swe200_baseline_complete_sha256"),
        ("parent_fixed44_raw_diagnostic_complete_sha256", "parent_fixed44_raw_diagnostic_complete_sha256"),
    ):
        digest = binding.get(bound_name)
        if (not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                or contract.get(claimed_name) != digest):
            raise ValueError("Same-precision parent baseline is not bound to this evaluation claim")


def _derived_fp16_source(original: bytes, name: str) -> bytes:
    if original.count(_BF16_LITERAL) != 1:
        raise ValueError("Pinned generator autocast expression changed: " + name)
    changed = original.replace(_BF16_LITERAL, _FP16_LITERAL)
    if sha256(changed).hexdigest() != FP16_DERIVED_SOURCE_SHA[name]:
        raise ValueError("Derived fp16 generator source differs from the fixed profile")
    return changed


def _prepare_eval_source(v43_zip: Path, overlay_source: Path, target: Path,
                         precision_profile: str) -> Path:
    source = _prepare_source(v43_zip, overlay_source, target)
    if precision_profile == "eval_fp16_native_v1":
        for name in FP16_DERIVED_SOURCE_SHA:
            path = source / name
            path.write_bytes(_derived_fp16_source(path.read_bytes(), name))
    _verify_eval_source(v43_zip, overlay_source, source, precision_profile)
    return source


def _verify_eval_source(v43_zip: Path, overlay_source: Path, source: Path,
                        precision_profile: str) -> None:
    if source.is_symlink():
        raise ValueError("Isolated evaluation source root is a symbolic link")
    if precision_profile == "eval_bf16_native":
        _verify_extracted_source(v43_zip, overlay_source, source)
        return
    if precision_profile != "eval_fp16_native_v1" or file_sha(v43_zip) != PINNED_V43_ZIP_SHA:
        raise ValueError("Unknown or changed evaluation precision/source")
    with zipfile.ZipFile(v43_zip) as archive:
        expected_files = {member.filename for member in archive.infolist() if not member.is_dir()}
        expected_files.update(SOURCE_OVERLAY_SHA)
        for member in archive.infolist():
            if member.is_dir() or member.filename in SOURCE_OVERLAY_SHA:
                continue
            path = source / member.filename
            if not path.is_file():
                raise ValueError("Evaluation source/data is missing: " + member.filename)
            if member.filename == "integrated_model_v42.py":
                expected = _derived_fp16_source(archive.read(member), member.filename)
                digest = sha256(expected).hexdigest()
            else:
                h = sha256()
                with archive.open(member) as stream:
                    for block in iter(lambda: stream.read(8 << 20), b""):
                        h.update(block)
                digest = h.hexdigest()
            if file_sha(path) != digest:
                raise ValueError("Evaluation source/data changed: " + member.filename)
    for name, digest in SOURCE_OVERLAY_SHA.items():
        original = overlay_source / name
        if file_sha(original) != digest:
            raise ValueError("Original additive overlay changed: " + name)
        expected = original.read_bytes()
        if name == "integrated_model_v44fix.py":
            expected = _derived_fp16_source(expected, name)
        if sha256((source / name).read_bytes()).digest() != sha256(expected).digest():
            raise ValueError("Evaluation overlay changed: " + name)
    actual_files = set()
    for candidate in source.rglob("*"):
        if candidate.is_symlink():
            raise ValueError("Isolated evaluation source contains a symbolic link")
        if candidate.is_file():
            actual_files.add(candidate.relative_to(source).as_posix())
    if actual_files != expected_files:
        raise ValueError("Isolated evaluation source contains a missing or unpinned file")


def _expected_runs(cases):
    return [(case, profile) for case in cases for profile in case["profiles"]]


def _validate_saved_rows(existing, expected_runs, *, arm, checkpoint_sha,
                         evaluator_sha, overlay_sha, precision_profile):
    if len(existing) > len(expected_runs):
        raise ValueError("Saved raw-answer cursor exceeds the fixed diagnostic")
    for row, (case, profile) in zip(existing, expected_runs):
        expected_prompt = PYTHON_MODE_CUE + case["prompt"] if profile == "python_cued" else case["prompt"]
        if (row.get("case_id") != case["id"] or row.get("profile") != profile
                or row.get("prompt") != expected_prompt or row.get("arm") != arm
                or row.get("checkpoint_sha256") != checkpoint_sha
                or row.get("cases_sha256") != CASES_SHA256
                or row.get("source_sha256") != evaluator_sha
                or row.get("overlay_sha256") != overlay_sha
                or row.get("precision_profile") != precision_profile
                or row.get("v43_zip_sha256") != PINNED_V43_ZIP_SHA
                or row.get("expected_answer") != case["target"]
                or row.get("group") != case["group"]
                or type(row.get("generated_tokens")) is not int
                or row["generated_tokens"] < 0
                or row.get("generation_iteration_limit") != 2048
                or not isinstance(row.get("generated_token_ids"), list)
                or len(row["generated_token_ids"]) != row["generated_tokens"]
                or any(type(token) is not int or token < 0 for token in row["generated_token_ids"])
                or row.get("generated_token_ids_sha256") != sha256(json.dumps(
                    row["generated_token_ids"], separators=(",", ":")).encode()).hexdigest()
                or row.get("raw_answer_sha256") != sha256(row["raw_answer"].encode()).hexdigest()
                or row.get("model_generated_text_sha256") != sha256(row["model_generated_text"].encode()).hexdigest()):
            raise ValueError("Saved raw answer differs from the fixed source/checkpoint/cases")


def _partial_record(*, arm, checkpoint_sha, evaluator_sha, overlay_sha,
                    raw_sha, completed, required, precision_profile):
    return {"schema":"geocpu_v43_python_gradual_raw_diagnostic_partial_v1",
            "arm":arm, "checkpoint_sha256":checkpoint_sha,
            "cases_sha256":CASES_SHA256, "completed_generations":completed,
            "required_generations":required, "raw_answers_sha256":raw_sha,
            "source_sha256":evaluator_sha, "v43_zip_sha256":PINNED_V43_ZIP_SHA,
            "overlay_sha256":overlay_sha, "max_generated_tokens":2048,
            "precision_profile":precision_profile,
            "training_updates":0}


def seal_interrupted_partial(*, output_dir: Path, checkpoint: Path, arm: str,
                             overlay_source: Path, v43_zip: Path,
                             precision_profile: str) -> dict:
    """Watchdog seals only whole flushed rows after the child has exited."""
    raw_path = output_dir / "raw_answers_44_profiles.jsonl"
    if (output_dir / "COMPLETE.json").exists() or (output_dir / "PARTIAL.json").exists():
        raise ValueError("A complete or already sealed diagnostic cannot be sealed again")
    if not raw_path.is_file():
        raise FileNotFoundError("Interrupted diagnostic has no flushed raw answers")
    raw = raw_path.read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError("Interrupted diagnostic has a torn final row; preserve for manual inspection")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    cases_path = Path(__file__).with_name(CASES_NAME)
    if file_sha(cases_path) != CASES_SHA256 or file_sha(v43_zip) != PINNED_V43_ZIP_SHA:
        raise ValueError("Diagnostic cases or original source changed")
    _verify_eval_source(v43_zip, overlay_source, output_dir / "isolated_source", precision_profile)
    cases = [json.loads(line) for line in cases_path.read_text(encoding="utf-8").splitlines()]
    runs = _expected_runs(cases)
    if not rows or len(rows) >= len(runs):
        raise ValueError("Interrupted cursor is empty or already at the end")
    evaluator_sha = file_sha(__file__)
    checkpoint_sha = file_sha(checkpoint)
    overlay_sha = sha256(json.dumps(SOURCE_OVERLAY_SHA, sort_keys=True).encode()).hexdigest()
    _validate_saved_rows(rows, runs, arm=arm, checkpoint_sha=checkpoint_sha,
                         evaluator_sha=evaluator_sha, overlay_sha=overlay_sha,
                         precision_profile=precision_profile)
    partial = _partial_record(arm=arm, checkpoint_sha=checkpoint_sha,
                              evaluator_sha=evaluator_sha, overlay_sha=overlay_sha,
                              raw_sha=sha256(raw).hexdigest(),
                              completed=len(rows), required=len(runs),
                              precision_profile=precision_profile)
    target = output_dir / "PARTIAL.json"
    temporary = output_dir / "PARTIAL.json.tmp"
    with temporary.open("x", encoding="utf-8") as sink:
        json.dump(partial, sink, sort_keys=True, indent=2)
        sink.write("\n")
    os.replace(temporary, target)
    return partial


def _score_swe(result, target: str, profile: str, expected_loops: int) -> dict:
    raw = result["output_text"]
    trace = result.get("cpu_trace", [])
    r0 = [str(step["val"]) for step in trace if step.get("dest") == "R0"]
    cleaned = re.sub(r'<[^>]+>(?:\s*=>\s*\[[^\]]+\])?', '', raw.strip()).strip()
    words = cleaned.split()
    text_last = words[-1] if words else ""
    original_final = r0[-1] if result.get("cpu_invocations", 0) and r0 else text_last
    cpu_trace_final = r0[-1] if result.get("cpu_invocations", 0) and r0 else None
    native_cpu_correct = cpu_trace_final == target.strip()
    dec_count = sum(step.get("op") == "DEC" for step in trace)
    compute_count = sum(step.get("op") in ("ADD", "SUB", "MUL", "MOD", "BXOR") for step in trace)
    executed_loops = dec_count if dec_count else compute_count
    step_class = "exact_step" if executed_loops == expected_loops else (
        "overshoot" if executed_loops > expected_loops else "undershoot")
    py_trace = result.get("python_trace", [])
    py_stdout = "".join(step.get("stdout", "") for step in py_trace)
    py_r0 = re.findall(r"\[R0=(\d+)\]", py_stdout)
    python_final = py_r0[-1] if py_r0 else None
    model_text = result.get("model_generated_text", raw)
    # The model's final response begins only after the final closed code fence.
    # Tool stdout and numbers in code must never count as a prose answer.
    closed_fences = list(re.finditer(r"```[^`]*```", model_text, flags=re.S))
    final_prose = model_text[closed_fences[-1].end():].strip() if closed_fences else ""
    explanation_numbers = re.findall(r"(?<!\d)\d+(?!\d)", final_prose)
    explanation_final = explanation_numbers[-1] if explanation_numbers else None
    japanese_explanation_present = bool(re.search(r"[ぁ-んァ-ン一-龯]", final_prose))
    python_execution_correct = bool(py_trace and py_r0 and all(step.get("success") for step in py_trace)
                                    and python_final == target.strip())
    prose_correct = explanation_final == target.strip()
    natural_eos = bool(result.get("ended_with_eos", False))
    return {"old_original_evaluator_correct": original_final == target.strip() if profile == "native" else None,
            "old_cpu_native_rule_correct": native_cpu_correct if profile == "native" else None,
            "old_cpu_native_rule_and_natural_eos_correct":
                (native_cpu_correct and natural_eos) if profile == "native" else None,
            "python_executed_and_numeric_correct": python_execution_correct if profile != "native" else None,
            "python_final_answer_numeric_correct": prose_correct if profile != "native" else None,
            "python_japanese_explanation_present": japanese_explanation_present if profile != "native" else None,
            "python_grounded_explanation_grade": None,
            "python_execution_and_final_answer_correct": (python_execution_correct and prose_correct)
                if profile != "native" else None,
            "python_natural_eos": natural_eos if profile != "native" else None,
            "python_execution_final_and_eos_correct":
                (python_execution_correct and prose_correct and natural_eos) if profile != "native" else None,
            "cpu_final": cpu_trace_final, "original_evaluator_final": original_final,
            "python_final": python_final,
            "explanation_final_numeric": explanation_final,
            "final_model_prose": final_prose if profile != "native" else None,
            "native_expected_loops": expected_loops if profile == "native" else None,
            "native_executed_loops": executed_loops if profile == "native" else None,
            "native_step_class": step_class if profile == "native" else None,
            "native_natural_eos": natural_eos if profile == "native" else None}


def evaluate(*, v43_zip: Path, overlay_source: Path, checkpoint: Path,
             checkpoint_kind: str, output_dir: Path, prelaunch_contract: Path,
             resume: bool = False, hard_stop_seconds: int = 1800,
             save_reserve_seconds: int = 120):
    if os.environ.get("GEOCPU_EVAL_WATCHDOG_CHILD") != "1":
        raise RuntimeError("Launch this diagnostic through the bounded watchdog entry")
    if not Path("/kaggle").exists() or Path("/content/drive/MyDrive").exists():
        raise RuntimeError("Only a disposable private Kaggle GPU may run fixed raw-answer diagnostics")
    if output_dir.exists() and not resume:
        raise FileExistsError("Use a new raw-answer output directory")
    if not 900 <= hard_stop_seconds <= 3600 or save_reserve_seconds < 120:
        raise ValueError("A bounded diagnostic and 120 seconds for result saving are required")
    contract = verify_prelaunch_contract(prelaunch_contract,
        required_run_seconds=hard_stop_seconds,
        required_save_reserve_seconds=save_reserve_seconds)
    absolute_stop = datetime.fromisoformat(contract["hard_stop_at_utc"].replace("Z", "+00:00")).astimezone(timezone.utc)
    if (absolute_stop - datetime.now(timezone.utc)).total_seconds() < save_reserve_seconds + 600:
        raise ValueError("Diagnostic hard stop leaves too little time to load and save")
    started = time.monotonic()
    if file_sha(v43_zip) != PINNED_V43_ZIP_SHA:
        raise ValueError("Original v43 source ZIP differs")
    cases_path = Path(__file__).with_name(CASES_NAME)
    if file_sha(cases_path) != CASES_SHA256:
        raise ValueError("Fixed original evaluation questions differ")
    cases = [json.loads(line) for line in cases_path.read_text(encoding="utf-8").splitlines()]
    if len(cases) != 44:
        raise ValueError("Fixed diagnostics are incomplete")
    if checkpoint_kind == "parent" and (checkpoint.stat().st_size != PARENT_CHECKPOINT_BYTES
                                        or file_sha(checkpoint) != PARENT_CHECKPOINT_SHA):
        raise ValueError("Original v43 parent checkpoint differs")
    precision_profile = contract["precision_profile"]
    checkpoint_sha = file_sha(checkpoint)
    evaluator_sha = file_sha(__file__)
    overlay_binding_sha = sha256(json.dumps(SOURCE_OVERLAY_SHA, sort_keys=True).encode()).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=resume)
    source = output_dir / "isolated_source"
    if not resume:
        source = _prepare_eval_source(v43_zip, overlay_source, source, precision_profile)
    elif not source.is_dir():
        raise FileNotFoundError("Partial diagnostic has no isolated original source")
    else:
        _verify_eval_source(v43_zip, overlay_source, source, precision_profile)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source))
    import torch
    from GeoCPU_CED_v1 import CEDConfig
    from GeoCPU_CED_v2 import GeoCPU_CED_v2
    from integrated_model_v44fix import IntegratedModelV44Fix
    from local_search_engine_v42 import LocalSearchEngineV42
    from tir_engine_v44r1 import PythonExecutor
    with zipfile.ZipFile(v43_zip) as archive:
        metadata = json.loads(archive.read("base_metadata.json"))
    cfg = dict(metadata["config"])
    cfg["scan_backend"] = "jit"
    model = IntegratedModelV44Fix(base=GeoCPU_CED_v2(CEDConfig(**cfg)),
        search_engine=LocalSearchEngineV42(corpus_path=source / "data" / "knowledge_corpus_v43.jsonl"))
    if checkpoint_kind == "parent":
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if payload.get("step") != 15_000 or payload.get("meta") != metadata or payload.get("config") != cfg:
            raise ValueError("Original v43 parent tokenizer/config differs")
        state = payload["model_state_dict"]
        arm = "parent"
    elif checkpoint_kind == "hybrid_250":
        payload = _load_stage_checkpoint(checkpoint, torch=torch)
        binding = payload["binding"]
        if (payload["accepted_updates"] != 250 or binding.get("parent_sha256") != PARENT_CHECKPOINT_SHA
                or binding.get("v43_zip_sha256") != PINNED_V43_ZIP_SHA
                or binding.get("converted_sha256") != PINNED_CONVERTED_SHA
                or binding.get("stage_source_sha256") != file_sha(Path(__file__).with_name("train_hybrid_stage.py"))
                or binding.get("source_pools_sha256") != file_sha(Path(__file__).with_name("source_pools.py"))
                or binding.get("hybrid_data_sha256") != file_sha(Path(__file__).with_name("hybrid_data.py"))
                or binding.get("overlay_sha256") != SOURCE_OVERLAY_SHA
                or binding.get("arm") not in ("control", "python")):
            raise ValueError("Expected an admitted 250-update hybrid checkpoint")
        _verify_hybrid_precision_binding(binding, contract)
        state = payload["model_state_dict"]
        arm = payload["binding"]["arm"]
    else:
        raise ValueError("Checkpoint kind must be parent or hybrid_250")
    if (contract["arm"] != arm or contract["gpu_count"] != torch.cuda.device_count()
            or precision_profile not in ("eval_bf16_native", "eval_fp16_native_v1")
            or contract["gpu_names"] != [torch.cuda.get_device_name(i)
                                          for i in range(torch.cuda.device_count())]):
        raise ValueError("Fresh Root claim arm or GPU count differs from the diagnostic")
    if precision_profile == "eval_bf16_native":
        for device_index in range(torch.cuda.device_count()):
            with torch.cuda.device(device_index):
                if (torch.cuda.get_device_capability(device_index)[0] < 8
                        or not torch.cuda.is_bf16_supported()):
                    raise ValueError("Native bf16 evaluation requires support on every allocated GPU")
    claim_copy = output_dir / ("prelaunch_claim_" + contract["claim_id"] + ".json")
    if claim_copy.exists():
        raise ValueError("This one-use diagnostic claim was already consumed")
    with claim_copy.open("x", encoding="utf-8") as sink:
        json.dump({"schema":"geocpu_hybrid_eval_claim_used_v1",
                   "arm":arm, "claim_sha256":file_sha(prelaunch_contract)},
                  sink, sort_keys=True, indent=2)
        sink.write("\n")
    if checkpoint.stat().st_size == 0 or file_sha(checkpoint) != checkpoint_sha:
        raise ValueError("Checkpoint changed during diagnostic model loading")
    model.load_state_dict(state, strict=True)
    model.cuda().eval()
    vocab = metadata["vocab"]
    inverse = {int(index): char for char, index in vocab.items()}
    output_file = output_dir / "raw_answers_44_profiles.jsonl"
    expected_runs = _expected_runs(cases)
    existing = []
    if resume:
        if (output_dir / "COMPLETE.json").exists() or not output_file.is_file():
            raise ValueError("Only an incomplete raw-answer file can be resumed")
        partial = json.loads((output_dir / "PARTIAL.json").read_text(encoding="utf-8"))
        if (partial.get("schema") != "geocpu_v43_python_gradual_raw_diagnostic_partial_v1"
                or partial.get("arm") != arm
                or partial.get("checkpoint_sha256") != checkpoint_sha
                or partial.get("v43_zip_sha256") != PINNED_V43_ZIP_SHA
                or partial.get("cases_sha256") != CASES_SHA256
                or partial.get("source_sha256") != evaluator_sha
                or partial.get("overlay_sha256") != overlay_binding_sha
                or partial.get("precision_profile") != precision_profile
                or partial.get("raw_answers_sha256") != file_sha(output_file)
                or partial.get("max_generated_tokens") != 2048):
            raise ValueError("Partial run source, checkpoint, or raw answers differ")
        existing = [json.loads(line) for line in output_file.read_text(encoding="utf-8").splitlines()]
        if (len(existing) >= len(expected_runs)
                or partial.get("completed_generations") != len(existing)
                or partial.get("required_generations") != len(expected_runs)):
            raise ValueError("Partial diagnostic cursor is invalid")
        _validate_saved_rows(existing, expected_runs, arm=arm,
                             checkpoint_sha=checkpoint_sha,
                             evaluator_sha=evaluator_sha, overlay_sha=overlay_binding_sha,
                             precision_profile=precision_profile)
    profile_counts = {name: sum(row["profile"] == name for row in existing)
                      for name in ("native", "python_cued", "python_uncued")}
    with output_file.open("a" if resume else "x", encoding="utf-8", newline="\n") as sink, PythonExecutor() as executor, torch.inference_mode():
        for case, profile in expected_runs[len(existing):]:
            if (time.monotonic() - started >= hard_stop_seconds - save_reserve_seconds
                    or (absolute_stop - datetime.now(timezone.utc)).total_seconds() <= save_reserve_seconds):
                _verify_eval_source(v43_zip, overlay_source, source, precision_profile)
                if file_sha(checkpoint) != checkpoint_sha:
                    raise ValueError("Checkpoint changed during diagnostic")
                partial = _partial_record(arm=arm, checkpoint_sha=checkpoint_sha,
                    evaluator_sha=evaluator_sha, overlay_sha=overlay_binding_sha,
                    raw_sha=file_sha(output_file), completed=sum(profile_counts.values()),
                    required=len(expected_runs), precision_profile=precision_profile)
                (output_dir / "PARTIAL.json").write_text(json.dumps(partial, sort_keys=True, indent=2)+"\n", encoding="utf-8")
                return partial
            prompt = PYTHON_MODE_CUE + case["prompt"] if profile == "python_cued" else case["prompt"]
            if profile == "native":
                result = model.generate_with_qwenputer_v42(prompt, vocab, inverse,
                        max_tokens=2048, arm="learned", repetition_penalty=1.0, enable_search=True)
            else:
                executor.reset()
                result = model.generate_v44fix(prompt, vocab, inverse,
                        max_tokens=2048, arm="learned", repetition_penalty=1.0,
                        enable_search=True, enable_python=True, loop_guard=False,
                        executor=executor)
            row = {"case_id": case["id"], "group": case["group"],
                       "source_family": case["source_family"], "profile": profile,
                       "arm": arm, "prompt": prompt, "expected_answer": case["target"],
                       "checkpoint_sha256":checkpoint_sha,
                       "cases_sha256":CASES_SHA256,
                       "source_sha256":evaluator_sha,
                       "overlay_sha256":overlay_binding_sha,
                       "precision_profile":precision_profile,
                       "v43_zip_sha256":PINNED_V43_ZIP_SHA,
                       "raw_answer": result["output_text"],
                       "raw_answer_sha256": sha256(result["output_text"].encode()).hexdigest(),
                       "model_generated_text": result.get("model_generated_text", result["output_text"]),
                       "model_generated_text_sha256": sha256(result.get("model_generated_text", result["output_text"]).encode()).hexdigest(),
                       "generated_tokens": len(result["generated_tokens"]),
                       "generation_iteration_limit": 2048,
                       "generated_token_ids": result["generated_tokens"],
                       "generated_token_ids_sha256": sha256(json.dumps(
                           result["generated_tokens"], separators=(",", ":")).encode()).hexdigest(),
                       "ended_with_eos": bool(result["ended_with_eos"]),
                       "stop_reason": result.get("stop_reason", "eos" if result["ended_with_eos"] else "limit_or_other"),
                       "cpu_invocations": result.get("cpu_invocations", 0),
                       "cpu_trace": result.get("cpu_trace", []),
                       "search_invocations": result.get("search_invocations", 0),
                       "search_trace": result.get("search_trace", []),
                       "python_invocations": result.get("python_invocations", 0),
                       "python_trace": result.get("python_trace", [])}
            if case["group"] == "swe20":
                row.update(_score_swe(result, case["target"], profile, case["expected_loops"]))
            sink.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            sink.flush()
            profile_counts[profile] = profile_counts.get(profile, 0) + 1
    if profile_counts != {"native": 44, "python_cued": 20, "python_uncued": 20}:
        raise ValueError("A fixed diagnostic profile is missing")
    _verify_eval_source(v43_zip, overlay_source, source, precision_profile)
    if file_sha(checkpoint) != checkpoint_sha:
        raise ValueError("Checkpoint changed during diagnostic")
    manifest = {"schema": "geocpu_v43_python_gradual_raw_diagnostic_v1",
                "status": "COMPLETE", "arm": arm,
                "checkpoint_kind": checkpoint_kind,
                "checkpoint_sha256": checkpoint_sha,
                "v43_zip_sha256": PINNED_V43_ZIP_SHA,
                "cases_sha256": CASES_SHA256,
                "source_sha256": evaluator_sha,
                "overlay_sha256": overlay_binding_sha,
                "precision_profile":precision_profile,
                "raw_answers_sha256": file_sha(output_file),
                "profile_counts": profile_counts,
                "max_generated_tokens": 2048,
                "training_updates": 0}
    with (output_dir / "COMPLETE.json").open("x", encoding="utf-8") as sink:
        json.dump(manifest, sink, ensure_ascii=False, sort_keys=True, indent=2)
        sink.write("\n")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    for name in ("v43_zip", "overlay_source", "checkpoint", "output_dir", "prelaunch_contract"):
        parser.add_argument("--" + name.replace("_", "-"), required=True, type=Path)
    parser.add_argument("--checkpoint-kind", choices=("parent", "hybrid_250"), required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--hard-stop-seconds", type=int, default=1800)
    parser.add_argument("--save-reserve-seconds", type=int, default=120)
    args = parser.parse_args()
    print(json.dumps(evaluate(**vars(args)), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
