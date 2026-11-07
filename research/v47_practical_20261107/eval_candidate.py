"""Frozen legacy tests plus resumable, per-question logged auxiliary evaluations.

`plan` is CPU-only. `run` explicitly loads a saved model on a free Colab GPU.
Original T1--T8 use the byte-preserved original stage, including old prompts,
selection and scorers. Additional generated knowledge modes have separate IDs.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
import uuid
import zipfile

import frozen_eval as frozen
from live_logging import RunLogger, atomic_json, canonical, file_sha, run_with_logging

ROOT = Path(__file__).resolve().parent
STUDY = Path("/content/drive/MyDrive/GeoCPU_v47_fullmodel_20261106")
PARENT_RUN = "v47_from28000_to30500_curriculum_v1"
BASE_BUNDLE = "GeoCPU_v47_fullmodel_bundle_v1_READY.zip"
BASE_SHA = "a25d312429d5b2ee43ec4b812e7a1fb948d0625fbfdfbe9cdf7fac54c075fb70"
AUX = ["python120", "engineering64", "visual10", "search_decision200", "search_final198", "public_dev80"]
FENCE = re.compile(r"(?ms)^```[^\n]*\n.*?^```[ \t]*(?:\n|$)")
SUPPLEMENTAL = {"practical_dev", "practical_known_families", "practical_unseen_families",
                "knowledge_diagnostic", "benchmark_tool_jmmlu", "benchmark_tool_jamcqa"}


class EvaluationContractError(ValueError):
    """A runner/input binding error, which must stop before more generation."""

def hash_object(value):
    return hashlib.sha256(canonical(value)).hexdigest()

def read_rows(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]

def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", value)

def import_path(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def prepare_source(study, destination):
    bundle = Path(study) / BASE_BUNDLE
    if file_sha(bundle) != BASE_SHA:
        raise ValueError("Original source bundle differs from the pinned v47 bundle")
    source = Path(destination) / "model_source"
    source.mkdir()
    with zipfile.ZipFile(bundle) as outer:
        with zipfile.ZipFile(io.BytesIO(outer.read("assets/r6_bundle.zip"))) as r6:
            for member in r6.infolist():
                if not (source / member.filename).resolve().is_relative_to(source.resolve()):
                    raise ValueError("Unsafe archive member")
            r6.extractall(source)
        for name in ("integrated_model_v44fix.py", "runtime_full_v46.py"):
            (source / name).write_bytes(outer.read("code/" + name))
    return source

def validate_checkpoint(study, run, step, expected=None):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run) or step < 1:
        raise ValueError("Invalid saved run/step")
    branch = Path(study) / run
    receipt = json.loads((branch / f"checkpoints/receipt_step_{step}.json").read_text())
    saved = json.loads((branch / "training_saved.json").read_text())
    manifest = json.loads((branch / "input_manifest.json").read_text())
    sha = receipt.get("checkpoint_sha256")
    training_sha = receipt.get("training_data_sha256")
    if (receipt.get("step") != step or saved.get("step") != step or saved.get("full_model") is not True
            or not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha)
            or (expected and sha != expected.lower())
            or not isinstance(training_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", training_sha)
            or saved.get("training_data_sha256") != training_sha
            or training_sha not in (manifest.get("output_sha256"), manifest.get("train_sha256"))
            or (saved.get("checkpoint_sha256") is not None and saved["checkpoint_sha256"] != sha)):
        raise ValueError("Saved checkpoint receipt and training state disagree")
    path = branch / f"checkpoints/checkpoint_step_{step}.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path, receipt

def load_model(source, checkpoint, checkpoint_sha):
    if file_sha(checkpoint) != checkpoint_sha:
        raise ValueError("Checkpoint bytes differ from its receipt")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Model evaluation needs a separate free Colab GPU; use plan/watch on CPU")
    sys.path.insert(0, str(source))
    from GeoCPU_CED_v1 import CEDConfig
    from GeoCPU_CED_v2 import GeoCPU_CED_v2
    from integrated_model_v44fix import IntegratedModelV44Fix
    from local_search_engine_v42 import LocalSearchEngineV42
    meta = json.loads((source / "base_metadata.json").read_text(encoding="utf-8"))
    cfg = dict(meta["config"]); cfg["scan_backend"] = "jit"
    index = frozen.source_path("public/retrieval/combined_dev_wikidata_first.jsonl")
    model = IntegratedModelV44Fix(GeoCPU_CED_v2(CEDConfig(**cfg)),
                search_engine=LocalSearchEngineV42(index, geosearch_url=""))
    ck = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    state = ck["model_state_dict"] if "model_state_dict" in ck else ck
    incompatible = model.load_state_dict(state, strict=False)
    bad = [k for k in incompatible.missing_keys if not k.startswith(("alu.", "ram."))]
    if bad or incompatible.unexpected_keys:
        raise ValueError(f"Checkpoint architecture mismatch: {bad[:5]} {incompatible.unexpected_keys[:5]}")
    model.to("cuda").eval()
    del ck, state
    return model, meta["vocab"], {int(v): k for k, v in meta["vocab"].items()}


def _original_source_binding():
    inventory = frozen.source_inventory()
    paths = ["eval_swe_step_control_heldout.jsonl", "eval_v38_decomposed_state_bench200.jsonl",
             "jamcqa_manifest_test.jsonl", "benchmark_8100_manifest.jsonl",
             "eval_v41_logical_instruction_operators_heldout.jsonl", "eval_v42_search_retrieval_aligned.jsonl"]
    return {"bundle_sha256": BASE_SHA,
            "r6_sha256": frozen.load_original_scorer("python120").R6_SHA,
            "tier_manifest_counts": {"data/"+name: len(read_rows(frozen.source_path("r6/data/"+name))) for name in paths},
            "retrieval_corpus_sha256": inventory["public/retrieval/combined_train_wikidata_first.jsonl"]["sha256"],
            "tier8_main_sha256": inventory["continue/tier_stage/eval_tier8_v46.py"]["sha256"],
            "tier8_helper_sha256": inventory["continue/tier_stage/tier8_readonly_v44fix.py"]["sha256"]}


def verify_original_stage(directory, *, checkpoint_sha, run, step, training_sha=None, receipt_sha=None):
    """Read-only validation of an already completed, original T1--T8 run.

    Native marker command hashes include the original output path and are not
    rewritten or used to pretend that the copied directory was natively run.
    This verifier checks the original input binding and every durable result
    hash itself, without invoking the stage or loading a model.
    """
    directory = Path(directory).resolve(strict=True)
    if (directory / "tier_stage_active.json").exists():
        raise EvaluationContractError("Original Tier output has an active/stale stage lock; it is not reusable")
    binding = json.loads((directory / "input_binding.json").read_text(encoding="utf-8"))
    expected = {**_original_source_binding(), "checkpoint_sha256": checkpoint_sha,
                "run": run, "step": step, "stage_order": "original Tier1-7, then original Tier8"}
    if training_sha is not None:
        expected["training_data_sha256"] = training_sha
    if receipt_sha is not None:
        expected["receipt_sha256"] = receipt_sha
    if any(binding.get(key) != value for key, value in expected.items()):
        raise EvaluationContractError("Original Tier checkpoint/source/input binding differs")
    for key in ("training_data_sha256", "receipt_sha256"):
        if not isinstance(binding.get(key), str) or re.fullmatch(r"[0-9a-f]{64}", binding[key]) is None:
            raise EvaluationContractError(f"Original Tier binding lacks {key}")
    complete = json.loads((directory / "complete.json").read_text(encoding="utf-8"))
    if complete.get("checkpoint_sha256") != checkpoint_sha:
        raise EvaluationContractError("Original Tier completion checkpoint differs")
    completion_files = {"tier4_raw_sha256": "tier4_raw.jsonl", "tier8_raw_sha256": "tier8_raw.json",
                        "tier1_7_log_sha256": "original_tier1_7.log", "tier8_log_sha256": "original_tier8.log"}
    for key, filename in completion_files.items():
        if complete.get(key) != file_sha(directory / filename):
            raise EvaluationContractError(f"Original Tier output hash differs: {filename}")
    names = ["input_binding.json", "complete.json", *completion_files.values()]
    for label in ("original_tier1_7", "original_tier8"):
        marker_name = label + ".complete.json"
        marker = json.loads((directory / marker_name).read_text(encoding="utf-8"))
        if (marker.get("returncode") != 0 or marker.get("log_sha256") != file_sha(directory / (label+".log"))
                or re.fullmatch(r"[0-9a-f]{64}", str(marker.get("command_sha256", ""))) is None):
            raise EvaluationContractError(f"Original stage completion is invalid: {label}")
        if label == "original_tier1_7" and marker.get("observed_tiers") != list(range(1, 8)):
            raise EvaluationContractError("Original stage did not observe all T1--T7")
        names.append(marker_name)
    # Guard the full public benchmark and both Tier8 profiles against partial
    # raw files even when a hand-created completion marker is present.
    cases4 = list(frozen.iter_cases("T4"))
    ordered4 = [case for case in cases4 if case.get("task_type") == "multiple_choice"]
    ordered4 += [case for case in cases4 if case.get("task_type") != "multiple_choice"]
    raw4 = read_rows(directory / "tier4_raw.jsonl")
    if ([row.get("id") for row in raw4] != [case["id"] for case in ordered4]
            or [row.get("target") for row in raw4] != [case["target"].strip() for case in ordered4]):
        raise EvaluationContractError("Original Tier4 raw rows do not match all frozen 8100 cases")
    raw8 = json.loads((directory / "tier8_raw.json").read_text(encoding="utf-8"))
    if set(raw8) != {"original_v44", "aligned_v43"}:
        raise EvaluationContractError("Original Tier8 profiles differ")
    for profile in ("original_v44", "aligned_v43"):
        expected_ids = [case["id"] for case in frozen.iter_cases("T8", profile)]
        if (raw8[profile].get("n") != 4 or
                [row.get("case_id") for row in raw8[profile].get("details", [])] != expected_ids):
            raise EvaluationContractError(f"Original Tier8 raw profile is incomplete: {profile}")
    for tier in range(1, 8):
        filename = f"tier_{tier}.log"
        if not (directory / filename).is_file():
            raise EvaluationContractError(f"Original per-tier log missing: {filename}")
        names.append(filename)
    return {"binding": binding, "file_sha256": {name: file_sha(directory / name) for name in names}}


def verify_reused_original(directory, **binding):
    """Verify our imported copy; never invoke native path-bound completion."""
    directory = Path(directory)
    receipt = json.loads((directory / "reuse_receipt.json").read_text(encoding="utf-8"))
    verified = verify_original_stage(directory, **binding)
    if (receipt.get("state") != "verified_original_copy"
            or receipt.get("input_binding") != verified["binding"]
            or receipt.get("copied_files_sha256") != verified["file_sha256"]):
        raise EvaluationContractError("Reused original Tier evidence differs from its copy receipt")
    return receipt


def reuse_original_stage(source, destination, **binding):
    """Copy only validated completed evidence into a fresh result directory.

    The source is strictly read-only. Locks, checkpoints and temporary files
    are never copied. The import is atomic at the destination directory level.
    """
    source = Path(source).resolve(strict=True)
    destination = Path(destination).resolve()
    if source == destination or source.is_relative_to(destination) or destination.is_relative_to(source):
        raise EvaluationContractError("Original source and reuse destination must be separate directories")
    verified = verify_original_stage(source, **binding)
    if destination.exists():
        receipt = verify_reused_original(destination, **binding)
        if (receipt.get("source_directory") != str(source)
                or receipt["copied_files_sha256"] != verified["file_sha256"]):
            raise EvaluationContractError("Existing import belongs to different original evidence")
        return receipt
    destination.parent.mkdir(parents=True, exist_ok=True)
    receipt = {"state": "verified_original_copy", "source_directory": str(source),
               "input_binding": verified["binding"], "copied_files_sha256": verified["file_sha256"],
               "native_stage_executed": False, "model_evaluation_executed": False,
               "native_command_hashes_rebased": False,
               "note": "Original path-bound stage markers are preserved. Use this import verifier, never native stage rerun on the copy."}
    with tempfile.TemporaryDirectory(prefix=".original_tiers_import_", dir=destination.parent) as temporary:
        staged = Path(temporary) / "evidence"
        staged.mkdir()
        for name, expected_sha in verified["file_sha256"].items():
            shutil.copyfile(source / name, staged / name)
            if file_sha(staged / name) != expected_sha:
                raise EvaluationContractError(f"Original evidence changed while copying: {name}")
        atomic_json(staged / "reuse_receipt.json", receipt)
        os.rename(staged, destination)
    return receipt

def final_text(model_text):
    """Supplemental only: score the last answer line, never code or observations."""
    closed = list(FENCE.finditer(model_text))
    prose = model_text[closed[-1].end():] if closed else model_text
    if re.search(r"(?m)^```", prose):
        return None  # An unterminated fence cannot turn its payload into an answer.
    lines = [x.strip() for x in prose.splitlines() if x.strip()]
    if not lines:
        return None
    return re.sub(r"^(?:最終回答|答え|回答|正解)\s*[:：]\s*", "", lines[-1]).strip()

def knowledge_score(case, generation):
    got = final_text(generation["model_generated_text"])
    expected = case["reference_final_answer"]
    answer_format = case["answer_format"]
    if answer_format == "free_text":
        format_valid = isinstance(got, str) and bool(got)
    elif answer_format == "letter":
        format_valid = got in list("ABCD")
    elif answer_format == "text":
        format_valid = got in case["options"]
    else:
        raise EvaluationContractError(f"Unsupported knowledge answer_format: {answer_format}")
    return {"exact_final": got == expected, "terminal_exact": got == expected and bool(generation.get("ended_with_eos")),
            "format_valid": format_valid,
            "final_answer": got, "expected_answer": expected,
            "search_invocations": int(generation.get("search_invocations", 0)),
            "python_invocations": int(generation.get("python_invocations", 0)),
            "ended_with_eos": bool(generation.get("ended_with_eos"))}

def public_score(case, generation, work):
    scorer = frozen.load_original_scorer("public_dev80")
    a, b = work / "one_reference.jsonl", work / "one_generation.jsonl"
    a.write_bytes(canonical(case)+b"\n")
    b.write_bytes(canonical({"id": case["id"], **generation})+b"\n")
    return scorer.score(a, b)["cases"][0]

def score_case(suite, case, generation, work):
    if suite == "python120" or suite.startswith("practical_"):
        reference = case
        if suite.startswith("practical_"):
            reference = {"id": case["id"], "group": "practical", "family": case["family"],
                         "expected_numeric": str(case["reference"]["value"]),
                         "prompt_sha256": hashlib.sha256(case["prompt"].encode()).hexdigest()}
        result = frozen.load_original_scorer("python120").score_generation(reference, generation)
        return result, "terminal_explicit_exact", "scored"
    if suite == "engineering64":
        result = frozen.load_original_scorer(suite).score_case(case, generation)
        return result, "correct", "scored"
    if suite == "visual10":
        result = frozen.load_original_scorer(suite).evaluate(case, generation["model_generated_text"], work)
        result["correct"] = result.get("status") == "pass"
        return result, "correct", "unmeasured" if result.get("status") == "unmeasured" else "scored"
    if suite == "search_decision200":
        return frozen.score_search_decision(case, generation), "exact_query_match", "scored"
    if suite == "search_final198":
        result = frozen.score_search_final(case, generation)
        return result, "terminal_exact", "scored"
    if suite == "public_dev80":
        return public_score(case, generation, work), "explicit_final_answer_exact", "scored"
    if suite.startswith("knowledge_") or suite.startswith("benchmark_tool_"):
        return knowledge_score(case, generation), "terminal_exact", "scored"
    raise ValueError(suite)

def metric_for(suite):
    if suite == "python120" or suite.startswith("practical_"):
        return "terminal_explicit_exact"
    if suite in ("engineering64", "visual10"):
        return "correct"
    if suite == "search_decision200":
        return "exact_query_match"
    if suite == "public_dev80":
        return "explicit_final_answer_exact"
    if suite == "search_final198" or suite in SUPPLEMENTAL:
        return "terminal_exact"
    raise EvaluationContractError(f"Unknown evaluation suite: {suite}")


def serializable_generation(generation):
    """Keep the original textual/tool evidence; exclude only runtime CPUState.

    CPUState contains tensors and is not an answer or a scorer input.  Its
    JSON-compatible cpu_trace/cpu_invocations remain in the original evidence.
    Unexpected non-JSON fields fail explicitly instead of being stringified.
    """
    if not isinstance(generation, dict):
        raise EvaluationContractError("Generator must return the native generation dictionary")
    for key in ("model_generated_text", "output_text"):
        if not isinstance(generation.get(key), str):
            raise EvaluationContractError(f"Native generation lacks {key}")
    if type(generation.get("ended_with_eos")) is not bool:
        raise EvaluationContractError("Native generation lacks boolean ended_with_eos")
    for tool in ("python", "search"):
        count, trace = generation.get(tool+"_invocations"), generation.get(tool+"_trace")
        if type(count) is not int or count < 0 or not isinstance(trace, list) or len(trace) != count:
            raise EvaluationContractError(f"Native {tool} count/trace mismatch")
    result = {key: value for key, value in generation.items() if key != "cpu_state"}
    canonical(result)
    return result, (["cpu_state"] if "cpu_state" in generation else [])


def _raw_path(output, case_id):
    return Path(output) / "raw" / (hashlib.sha256(case_id.encode()).hexdigest()[:24] + ".json")


def _read_bound_raw(path, case, identity, mode):
    record = json.loads(path.read_text(encoding="utf-8"))
    if (record.get("identity_sha256") != hash_object(identity) or record.get("id") != case["id"]
            or record.get("case_sha256") != hash_object(case) or record.get("prompt") != case["prompt"]
            or record.get("mode") != mode):
        raise EvaluationContractError("Raw case binding differs; existing evidence retained")
    return record


def _raw_detail(path, case):
    return json.dumps({"raw_sha256": file_sha(path), "case_sha256": hash_object(case)}, sort_keys=True)


def _verify_completed_raw(saved, path, case, identity, mode):
    record = _read_bound_raw(path, case, identity, mode)
    try:
        detail = json.loads(saved.get("detail") or "null")
    except (ValueError, TypeError) as error:
        raise EvaluationContractError("Completed result lacks a raw evidence hash") from error
    if (detail != json.loads(_raw_detail(path, case)) or saved.get("raw_ref") != str(path)
            or saved.get("status") != record.get("status")):
        raise EvaluationContractError("Completed raw evidence differs from the durable result")
    if saved["status"] == "scored" and saved.get("correct") is not record.get("score", {}).get(identity["metric"]):
        raise EvaluationContractError("Completed correctness differs from original raw score")


def run_questions(suite, mode, cases, generator, output, identity, *, resume=False,
                  max_consecutive_errors=3):
    """Model-independent loop with CPU-testable resume and error accounting.

    Infrastructure/generation/schema failures stop immediately.  An isolated
    scorer exception is recorded after raw generation and can be inspected;
    repeated scorer failures stop before the entire suite wastes GPU time.
    """
    cases = list(cases)
    if suite not in AUX and suite not in SUPPLEMENTAL:
        raise EvaluationContractError(f"Unsupported question-loop suite: {suite}")
    if not cases or not callable(generator) or max_consecutive_errors < 1:
        raise EvaluationContractError("Evaluation requires nonempty cases, a generator, and a positive error limit")
    ids = [case.get("id") for case in cases]
    if (any(not isinstance(cid, str) or not cid for cid in ids) or len(set(ids)) != len(ids)
            or any(not isinstance(case.get("prompt"), str) or not case["prompt"] for case in cases)):
        raise EvaluationContractError("Cases require unique nonempty IDs and original prompts")
    if (identity.get("suite") != suite or identity.get("mode") != mode
            or identity.get("metric") != metric_for(suite)
            or identity.get("dataset_sha256") != hash_object(cases)):
        raise EvaluationContractError("Suite, mode, metric or actual case data differs from its binding")
    out = Path(output)
    effective_resume = resume and (out / "run.json").is_file()
    with RunLogger(out, identity, total=len(cases), resume=effective_resume) as log:
        raw = out / "raw"; raw.mkdir(exist_ok=True)
        if set(log.results) - set(ids):
            raise EvaluationContractError("Saved result IDs are not in the bound suite")
        consecutive_errors = 0
        for case in cases:
            cid = case["id"]
            path = _raw_path(out, cid)
            if cid in log.results:
                _verify_completed_raw(log.results[cid], path, case, identity, mode)
                continue
            # Binding errors are not recoverable case errors. Never overwrite a
            # raw file that belongs to another checkpoint, case, or prompt.
            record = _read_bound_raw(path, case, identity, mode) if path.exists() else None
            log.start_case(cid)
            if record is None:
                record = {"id": cid, "identity_sha256": hash_object(identity),
                          "case_sha256": hash_object(case), "mode": mode, "prompt": case["prompt"]}
                began = time.monotonic()
                try:
                    generation, omitted = serializable_generation(generator(case))
                    record.update(generation=generation, generation_omitted_fields=omitted,
                                  seconds=time.monotonic()-began)
                    atomic_json(path, record)
                except (KeyboardInterrupt, SystemExit):
                    raise
                except Exception as exc:
                    record.update(status="error", evaluation_error=f"{type(exc).__name__}: {exc}",
                                  error_phase="generation_or_serialization", seconds=time.monotonic()-began)
                    atomic_json(path, record)
                    log.result(cid, status="error", raw_ref=str(path), detail=_raw_detail(path, case),
                               metrics={"error": record["evaluation_error"], "error_phase": record["error_phase"]})
                    raise RuntimeError("Generation/interface failed; stopped before further cases") from exc
            if "generation" not in record:
                log.result(cid, status="error", raw_ref=str(path), detail=_raw_detail(path, case),
                           metrics={"error": record.get("evaluation_error", "raw generation missing")})
                raise EvaluationContractError("Saved case has no generation; inspect its error before a new run")
            try:
                work = out / "artifacts" / path.stem; work.mkdir(parents=True, exist_ok=True)
                result, metric, status = score_case(suite, case, record["generation"], work)
                if (metric != identity["metric"] or status not in ("scored", "unmeasured")
                        or not isinstance(result, dict)
                        or (status == "scored" and type(result.get(metric)) is not bool)):
                    raise EvaluationContractError("Original scorer metric/type/status binding mismatch")
                canonical(result)  # Record a scorer/serialization error while the raw generation is durable.
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as exc:
                # The native engineering scorer lacks FACTORS['min']. Preserve
                # that source and expose this known unsupported unit separately.
                unsupported_unit = (suite == "engineering64" and isinstance(exc, KeyError)
                                    and exc.args == ("min",) and case.get("reference", {}).get("unit") == "min")
                status = "unmeasured" if unsupported_unit else "error"
                record.update(status=status, evaluation_error=f"{type(exc).__name__}: {exc}", error_phase="scoring")
                atomic_json(path, record)
                log.result(cid, status=status, metrics={"error": record["evaluation_error"]},
                           raw_ref=str(path), detail=_raw_detail(path, case))
                print(f"EVALUATOR_ERROR {suite} {cid} {type(exc).__name__}: {exc}", flush=True)
                consecutive_errors = 0 if unsupported_unit else consecutive_errors + 1
                if isinstance(exc, (EvaluationContractError, OSError)) or consecutive_errors >= max_consecutive_errors:
                    raise RuntimeError("Evaluator contract/infrastructure or repeated scoring errors; stopped") from exc
                continue
            # Persist evidence before marking a case complete. Filesystem or
            # logger write failures propagate instead of generating more cases.
            record["score"] = result; record["status"] = status
            atomic_json(path, record)
            boolean_metrics = {k: v for k, v in result.items()
                               if isinstance(v, (bool, int, float)) and math.isfinite(v)}
            log.result(cid, correct=result[metric] if status == "scored" else None,
                       status=status, metrics=boolean_metrics, raw_ref=str(path), detail=_raw_detail(path, case))
            consecutive_errors = 0
        return log.finish()

def suite_cases(suite, mode="formal"):
    if suite.startswith("practical_"):
        name = "practical_dev.jsonl" if suite == "practical_dev" else "practical_expanded_eval.jsonl"
        rows = read_rows(ROOT / "data" / name)
        if suite == "practical_unseen_families":
            rows = [r for r in rows if r["family_partition"] == "unseen_family"]
        elif suite == "practical_known_families":
            rows = [r for r in rows if r["family_partition"] == "known_family"]
        return rows
    if suite.startswith("knowledge_"):
        rows = read_rows(ROOT / "data/knowledge_free_dev.jsonl")
        if mode == "oracle":
            rows = [dict(row, prompt="参考資料:\n"+"\n".join(row["reference_observations"])+"\n"+row["prompt"]) for row in rows]
        return rows
    if suite.startswith("benchmark_tool_"):
        source = "T3" if suite.endswith("jamcqa") else "T4"
        rows = list(frozen.iter_cases(source))
        if source == "T4":
            rows = [r for r in rows if r.get("task_type") == "multiple_choice"]
        return [dict(r, answer_format="letter", options=list("ABCD"), reference_final_answer=r["target"].strip()) for r in rows]
    return list(frozen.iter_cases(suite))

def generation_settings(suite, mode):
    if suite in AUX:
        return frozen.generation_kwargs(suite)
    return {"max_tokens": 2048, "enable_search": mode == "tool",
            "enable_python": not suite.startswith("knowledge_"),
            "loop_guard": False, "repetition_penalty": 1.0}

def make_plan(include_full_knowledge=False):
    frozen.verify_frozen()
    specs = [(suite, "formal") for suite in AUX]
    specs += [("practical_known_families", "python"), ("practical_unseen_families", "python")]
    specs += [("knowledge_diagnostic", mode) for mode in ("closed", "tool", "oracle")]
    if include_full_knowledge:
        specs += [(suite, mode) for suite in ("benchmark_tool_jmmlu", "benchmark_tool_jamcqa") for mode in ("closed", "tool")]
    plan = [{"suite": "T1-T8", "mode": "original", "implementation": "unchanged_original_stage"}]
    for suite, mode in specs:
        rows = suite_cases(suite, mode)
        plan.append({"suite": suite, "mode": mode, "rows": len(rows),
                     "unique_prompts": len({r["prompt"] for r in rows}),
                     "dataset_sha256": hash_object(rows), "generation": generation_settings(suite, mode),
                     "metric": metric_for(suite)})
    plan.append({"suite": "new_public_9_groups", "mode": "original_raw_and_nll",
                 "implementation": "original_loss_for_rows_and_generate_rows", "score_is_accuracy": False})
    return {"model_evaluation_executed": False, "plan": plan,
            "notes": ["Legacy tests and supplemental generated modes are distinct result series.",
                      "All original T3/JMMLU rows remain in T1-T8; optional full tool modes add substantial inference cost.",
                      "Default knowledge_diagnostic uses the separate 64-chain free-text dev set in closed/tool/oracle modes. The small MC dev file is preserved but is not the default diagnostic.",
                      "Missing gate/sentinel inputs are listed in frozen/registry.json; no full-retention certificate is issued."]}

def _public_row(folder, kind, index, case, binding, compute, *, resume=False):
    """Keep each native raw/NLL row intact and bind its bytes for resumption."""
    target = Path(folder) / f"{kind}_{index:04d}.json"
    receipt_path = target.with_suffix(".receipt.json")
    expected = {"binding_sha256": hash_object(binding), "case_sha256": hash_object(case),
                "id": case["id"], "kind": kind, "index": index}
    if target.exists() or receipt_path.exists():
        if not resume or not target.is_file() or not receipt_path.is_file():
            raise EvaluationContractError("Existing public row is incomplete/unverified; inspect it before reuse")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt != {**expected, "raw_sha256": file_sha(target)}:
            raise EvaluationContractError("Public row binding or raw hash differs")
        detail = json.loads(target.read_text(encoding="utf-8"))
    else:
        detail = compute()
        if not isinstance(detail, dict) or detail.get("id") != case["id"]:
            raise EvaluationContractError("Native public row ID differs")
        atomic_json(target, detail)
        atomic_json(receipt_path, {**expected, "raw_sha256": file_sha(target)})
    if detail.get("id") != case["id"]:
        raise EvaluationContractError("Native public row ID differs")
    return detail


def run_new_public(model, vocab, inverse, output, cksha, *, resume=False, runtime_sha=None):
    """Same original per-row CE and generation routines; adds durable row-level logs."""
    import torch
    original = frozen.load_original_scorer("new_public:math_python")
    for suite in frozen.registry()["suites"]:
        sid = suite["id"]
        if not sid.startswith("new_public:"):
            continue
        group = sid.split(":", 1)[1]
        folder = Path(output) / ("new_public_"+group)
        folder.mkdir(exist_ok=True, parents=True)
        rows = list(frozen.iter_cases(sid, "nll"))
        raw_cases = list(frozen.iter_cases(sid, "raw"))
        binding = {"checkpoint_sha256": cksha, "dataset_sha256": hash_object(rows),
                   "raw_dataset_sha256": hash_object(raw_cases),
                   "scorer_sha256": file_sha(frozen.source_path("base/code/eval_new_public_v46.py")),
                   "runtime_sha256": runtime_sha,
                   "retrieval_corpus_sha256": file_sha(frozen.source_path("public/retrieval/combined_dev_wikidata_first.jsonl")),
                   "generation": frozen.generation_kwargs(sid), "nll_amp_bfloat16": True}
        binding_path = folder / "binding.json"
        if binding_path.exists():
            if not resume or json.loads(binding_path.read_text()) != binding:
                raise ValueError("New-public output binding mismatch")
        else:
            atomic_json(binding_path, binding)
        nll_rows, raw_rows = [], []
        for i, row in enumerate(rows):
            detail = _public_row(folder, "nll", i, row, binding,
                lambda: original.loss_for_rows(model, [row], vocab, torch.device("cuda"), True)["per_row"][0],
                resume=resume)
            nll_rows.append(detail)
            mean = sum(x["ce"]*x["positions"] for x in nll_rows)/sum(x["positions"] for x in nll_rows)
            print(f"REFERENCE_LOSS {group} processed={i+1}/{len(rows)} CE={mean:.6f} (not accuracy)", flush=True)
            atomic_json(folder / "nll_status.json", {"processed": i+1, "total": len(rows), "conditional_ce": mean, "state": "running"})
        for i, row in enumerate(raw_cases):
            detail = _public_row(folder, "raw", i, row, binding,
                lambda: original.generate_rows(model, group, [row], vocab, inverse, 2048)[0], resume=resume)
            raw_rows.append(detail)
            print(f"PUBLIC_RAW {group} processed={i+1}/{min(12,len(rows))} eos={sum(x['ended_with_eos'] for x in raw_rows)} (not answer accuracy)", flush=True)
        nll = {"rows": len(nll_rows), "positions": sum(x["positions"] for x in nll_rows),
               "conditional_ce": mean, "per_row": nll_rows}
        atomic_json(folder / "nll_status.json", {"processed": len(nll_rows), "total": len(rows),
                    "conditional_ce": mean, "state": "complete"})
        atomic_json(folder / "summary.json", {"binding": binding, "summary": original.group_summary(group, nll, raw_rows), "state": "complete"})

def run(args):
    output = args.output.resolve()
    if not args.resume and output.exists():
        raise FileExistsError("Use a new evaluation directory")
    # An existing model branch is never an evaluation output directory.
    branch = (args.study / args.run).resolve()
    if output == branch or output.is_relative_to(branch):
        raise ValueError("Evaluation output must be outside saved model branches")
    checkpoint, receipt = validate_checkpoint(args.study, args.run, args.step, args.checkpoint_sha)
    plan = make_plan(args.full_knowledge)
    output.mkdir(parents=True, exist_ok=True)
    plan_binding = {"plan": plan, "checkpoint_sha256": receipt["checkpoint_sha256"], "run": args.run, "step": args.step}
    plan_path = output / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan_binding:
        raise ValueError("Evaluation plan or checkpoint changed on resume")
    atomic_json(plan_path, plan_binding)
    original_dir = output / "original_tiers"
    original_binding = {"checkpoint_sha": receipt["checkpoint_sha256"], "run": args.run,
                        "step": args.step, "training_sha": receipt["training_data_sha256"],
                        "receipt_sha": file_sha(args.study / args.run / f"checkpoints/receipt_step_{args.step}.json")}
    reuse_from = getattr(args, "reuse_original_from", None)
    if reuse_from is not None:
        reused = reuse_original_stage(reuse_from, original_dir, **original_binding)
        print("ORIGINAL_TIER_EVIDENCE_REUSED", reused["source_directory"], flush=True)
    elif (original_dir / "reuse_receipt.json").is_file():
        verify_reused_original(original_dir, **original_binding)
        print("ORIGINAL_TIER_IMPORT_VERIFIED: native stage is not rerun on copied path-bound markers", flush=True)
    elif not args.aux_only:
        command = frozen.build_official_tier_command(study=args.study, run=args.run, step=args.step,
                    checkpoint_sha=receipt["checkpoint_sha256"], output=output / "original_tiers")
        code = run_with_logging(command, output / ("console_original_"+uuid.uuid4().hex[:8]),
                                watch_root=output / "original_tiers")
        if code:
            print("ORIGINAL_STAGE_INCOMPLETE: retained as incomplete; auxiliary runs have separate status", flush=True)
    with tempfile.TemporaryDirectory(prefix="geocpu_compare_") as temp:
        source = prepare_source(args.study, temp)
        model, vocab, inverse = load_model(source, checkpoint, receipt["checkpoint_sha256"])
        from local_search_engine_v42 import LocalSearchEngineV42
        dev_index = frozen.source_path("public/retrieval/combined_dev_wikidata_first.jsonl")
        train_index = frozen.source_path("public/retrieval/combined_train_wikidata_first.jsonl")
        runtime_sha = hash_object({p.name: file_sha(p) for p in sorted(source.glob("*.py"))})
        for spec in plan["plan"]:
            suite, mode = spec["suite"], spec["mode"]
            if suite in ("T1-T8", "new_public_9_groups"):
                continue
            corpus = train_index if suite.startswith("benchmark_tool_") else dev_index
            model.search_engine = LocalSearchEngineV42(corpus, geosearch_url="")
            settings = spec["generation"]
            identity = {"suite": suite, "mode": mode, "checkpoint_id": args.run+":"+str(args.step),
                        "checkpoint_sha256": receipt["checkpoint_sha256"], "dataset_sha256": spec["dataset_sha256"],
                        "scorer_sha256": hash_object({"wrapper": file_sha(Path(__file__)), "frozen": frozen.SOURCE_FILES_SHA256}),
                        "runtime_sha256": runtime_sha, "budget": dict(settings, retrieval_corpus_sha256=file_sha(corpus)),
                        "seed": 0, "metric": spec["metric"]}
            def generate(case):
                generation = model.generate_v44fix(case["prompt"], vocab, inverse, **settings)
                generation["prompt_oov_characters"] = sorted(set(case["prompt"])-set(vocab))
                return generation
            run_questions(suite, mode, suite_cases(suite, mode), generate,
                          output / (safe_name(suite)+"__"+mode), identity, resume=args.resume)
        model.search_engine = LocalSearchEngineV42(dev_index, geosearch_url="")
        run_new_public(model, vocab, inverse, output, receipt["checkpoint_sha256"], resume=args.resume,
                       runtime_sha=runtime_sha)
    atomic_json(output / "runner_finished.json", {"state": "finished_inspect_per_suite_status", "no_overall_score": True})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pp = sub.add_parser("plan"); pp.add_argument("--full-knowledge", action="store_true")
    rp = sub.add_parser("run")
    rp.add_argument("--study", type=Path, default=STUDY)
    rp.add_argument("--run", default=PARENT_RUN); rp.add_argument("--step", type=int, default=30500)
    rp.add_argument("--checkpoint-sha")
    rp.add_argument("--output", type=Path, required=True)
    rp.add_argument("--resume", action="store_true"); rp.add_argument("--aux-only", action="store_true")
    rp.add_argument("--reuse-original-from", type=Path,
                    help="read-only verify and import a completed native T1-T8 directory instead of re-generating it")
    rp.add_argument("--full-knowledge", action="store_true")
    args = parser.parse_args()
    if args.command == "plan":
        print(json.dumps(make_plan(args.full_knowledge), ensure_ascii=False, indent=2))
    else:
        run(args)
