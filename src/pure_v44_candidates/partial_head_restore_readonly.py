"""Predeclared, read-only parent-head intervention for the pure-v44 diagnosis.

The only candidate is the v43 parent's ``base.head`` projection used inside an
otherwise unchanged step-25000 pure-v44 model. A forward hook substitutes its
output during inference; no parameter, checkpoint, or training file is written.
The five practice prompts come from pinned v43 training rows. Eight held-out
prompts come from the pinned mixed64 suite. Neither a dev score nor a held-out
answer chooses the candidate, its strength, or its cases.

Run ``--preflight`` on a CPU to validate the fixed cases without loading models.
An actual comparison requires an already authorized Colab GPU and mounted Drive.
The zero-tool-call generator diagnoses neural decoding, not full Python tool use.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import pathlib
import shutil
import sys
import tempfile
import uuid
import zipfile
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from types import SimpleNamespace

import matched_pure_v44_readonly as matched


PRACTICE_ROWS = (
    (1, "search_augmented_1hop_fact",
     "ced056d7cea4aa1ce0a021aae9242f969508a71cc0d63255ae9a55cb4470f2c1",
     "56f19d536fd0a171d83bbd794e358cf604d95e4a22017094c59c6cf0b6b9eea9"),
    (2, "Algebraic_Composite_Aligned",
     "61142786c5a44719c885660b855410778e4201d3ed3f816df061e43dc5fddd27",
     "261e318865f5871e71a71f0935097cc312662891dd68f8806ab60d9383c86410"),
    (10, "swe_alu_accumulate_sum",
     "8bcc198982e6b340190f133429cf600af4591da52f5be4842761d8514b7a9d42",
     "500b8afb19301a17f1bf7ab1c3afe78aca54621d4c396a78d265db00289693a9"),
    (16, "SoftBank_JSQuAD_Full",
     "f84fbbfbbb6dbb65fe9be363133de209d50811739f8e538fb7aff49f9f99789f",
     "d11a2b672237f8d38d46910a1f5c1695764ea5c9e565a216eca858f086db7f4f"),
    (575, "JNLI_Logical_Operator_positive_entailment",
     "8e735715d46da49c5c2ef952d7c393a60bd13ec67f0c16f7ae468a4d3791c13c",
     "9a9f2ba7935d2d964c435a5fdb318c2132c1d9523f2794902599862f4ee159cd"),
)

HELDOUT8_IDS = (
    "jmmlu:762cbf192c9e:human_sexuality:0087",
    "jmmlu:762cbf192c9e:japanese_idiom:0032",
    "arc_agi_0197", "arc_agi_0115",
    "swe_step_control_heldout_0001", "swe_step_control_heldout_0013",
    "logic_op_eval_2776_positive_entailment",
    "logic_op_eval_2782_negation_contradiction",
)


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def practice_cases(vocab: dict[str, int]) -> list[dict]:
    """Read only five pinned v43 train rows; legacy targets are not v44 labels."""
    wanted = {row[0]: row for row in PRACTICE_ROWS}
    if len(wanted) != len(PRACTICE_ROWS):
        raise RuntimeError("Repeated practice row number")
    found = {}
    with zipfile.ZipFile(matched.V43_SOURCE) as source:
        with source.open("data/train_v43_knowledge_unified.jsonl") as stream:
            for number, line in enumerate(stream, 1):
                if number in wanted:
                    row = json.loads(line)
                    _, expected_source, prompt_sha, target_sha = wanted[number]
                    prompt, target = row.get("prompt"), row.get("target")
                    if (row.get("source") != expected_source
                            or not isinstance(prompt, str) or not prompt
                            or not isinstance(target, str) or not target
                            or text_sha256(prompt) != prompt_sha
                            or text_sha256(target) != target_sha):
                        raise RuntimeError(f"Pinned v43 practice row {number} differs")
                    full_text = row.get("full_text") or prompt + target
                    if full_text not in (prompt + target, prompt + " " + target):
                        raise RuntimeError(f"Practice row {number} has a different train continuation")
                    continuation = full_text[len(prompt):]
                    if (len(prompt) >= 8192 or set(prompt) - set(vocab)
                            or continuation[0] not in vocab):
                        raise RuntimeError(f"Practice row {number} is not a valid probe")
                    found[number] = {
                        "id": f"v43_train_row_{number:06d}",
                        "bucket": expected_source,
                        "prompt": prompt,
                        "target": continuation,
                        "v43_raw_target_sha256": target_sha,
                        "reference_protocol": "exact_v43_train_continuation_not_v44_cot_grading",
                    }
                if number >= max(wanted):
                    break
    if set(found) != set(wanted):
        raise RuntimeError("Pinned v43 practice row missing")
    return [found[number] for number, *_ in PRACTICE_ROWS]


def static_preflight(heldout_size: str) -> tuple[list[dict], list[dict], list[dict], dict, dict]:
    """No GPU, checkpoint load, or file writes."""
    broad, meta, evidence = matched.static_preflight("mixed64")
    seen = practice_cases(meta["vocab"])
    if len({row["id"] for row in broad}) != 64:
        raise RuntimeError("Pinned mixed64 IDs are not unique")
    broad_by_id = {row["id"]: row for row in broad}
    if len(HELDOUT8_IDS) != 8 or len(set(HELDOUT8_IDS)) != 8:
        raise RuntimeError("Held-out eight-case selection is not fixed")
    if any(case_id not in broad_by_id for case_id in HELDOUT8_IDS):
        raise RuntimeError("Predeclared held-out case missing")
    heldout = ([broad_by_id[case_id] for case_id in HELDOUT8_IDS]
               if heldout_size == "mini8" else broad)
    if heldout_size not in ("mini8", "mixed64"):
        raise ValueError("Unknown held-out size")
    if {row["id"] for row in seen} & {row["id"] for row in heldout}:
        raise RuntimeError("Practice and held-out IDs overlap")
    if {row["prompt"] for row in seen} & {row["prompt"] for row in heldout}:
        raise RuntimeError("Practice and held-out prompts overlap exactly")
    return seen, heldout, broad, meta, {
        "pure_v44": evidence,
        "practice_v43_train_rows": [row[0] for row in PRACTICE_ROWS],
        "heldout_ids": [row["id"] for row in heldout],
        "candidate": "parent base.head.weight in otherwise child model",
        "candidate_selection": "fixed from saved weight-drift audit, before these generations",
        "dev_used_for_selection": False,
    }


def verified_matched_cache(cache_dir: pathlib.Path, broad: list[dict], evidence: dict) -> tuple[dict, dict]:
    """Verify every matched64 answer before reusing any parent/child arm."""
    cache_dir = cache_dir.resolve(strict=True)
    def read_hashed_json(path: pathlib.Path) -> tuple[object, str]:
        content = path.read_bytes()
        return json.loads(content), hashlib.sha256(content).hexdigest()

    claim_path = cache_dir / "claim.json"
    complete_path = cache_dir / "COMPLETE.json"
    summary_path = cache_dir / "observed_stop_summary.json"
    claim, claim_sha = read_hashed_json(claim_path)
    complete, complete_sha = read_hashed_json(complete_path)
    summary, summary_sha = read_hashed_json(summary_path)
    decode = claim.get("decode", {})
    expected_decode = {"max_new_tokens": 2048, "arm": "learned", "search": False,
                       "repetition_penalty": 1.0, "autocast": False,
                       "python_tool_calls": 0, "per_case_wall_seconds": 180}
    def same_selection(saved: object) -> bool:
        if not isinstance(saved, dict):
            return False
        stripped = dict(saved)
        scanned = stripped.pop("actual_train_rows_directly_scanned", None)
        actual_sha = stripped.pop("actual_train_file_sha256", None)
        overlap = stripped.pop("exact_prompt_overlap", None)
        expected = dict(evidence["pure_v44"])
        expected.pop("actual_train_rows_directly_scanned")
        if stripped != expected:
            return False
        if scanned is False:
            return actual_sha is None and overlap is None
        if scanned is True:
            train_sha = json.loads(matched.DATASET_MANIFEST.read_text(encoding="utf-8"))["outputs"]["train.jsonl"]["sha256"]
            return actual_sha == train_sha and type(overlap) is int and overlap == 0
        return False

    if (claim.get("kind") != "read_only_pure_v44_same_code_parent_vs_step25000"
            or claim.get("cases") != 64 or claim.get("training_updates") != 0
            or any(type(decode.get(key)) is not type(value) or decode[key] != value
                   for key, value in expected_decode.items())
            or not same_selection(claim.get("separation_evidence"))):
        raise RuntimeError("Matched cache claim or generation profile differs")
    pins = claim.get("source_pins", {})
    essential = ("v43_source", "pure_source", "parent", "child", "dataset_manifest")
    if (any(pins.get(name) != matched.PINS[name][2] for name in essential)
            or any(name in pins and pins[name] != spec[2] for name, spec in matched.PINS.items())):
        raise RuntimeError("Matched cache source/checkpoint SHA pins differ")
    if (complete.get("suite") != "mixed64" or complete.get("parent_cases") != 64
            or complete.get("child_cases") != 64 or complete.get("training_updates") != 0
            or complete.get("same_source_zip_sha256") != matched.PINS["pure_source"][2]
            or complete.get("separation_evidence") != claim.get("separation_evidence")
             or complete.get("observed_stop_summary") != "observed_stop_summary.json"
             or complete.get("meaning_quality_scored") is not False
            or summary.get("cases_per_arm") != 64
            or summary.get("meaning_quality_scored") is not False
            or summary.get("python_tool_use_scored") is not False):
        raise RuntimeError("Matched cache completion record differs")
    records = {}
    stop_counts = {name: Counter() for name in ("parent", "child")}
    eos_counts = {name: 0 for name in stop_counts}
    for index, case in enumerate(broad):
        pair_path = cache_dir / "pairs" / f"case_{index:02d}.json"
        pair, pair_sha = read_hashed_json(pair_path)
        if (pair.get("case_index") != index or pair.get("case_id") != case["id"]
                or set(pair.get("arms", {})) != {"parent", "child"}):
            raise RuntimeError(f"Matched cache pair {index} has different identity")
        origins = {}
        for name in ("parent", "child"):
            arm_path = cache_dir / "arms" / f"{name}_case_{index:02d}.json"
            arm, arm_sha = read_hashed_json(arm_path)
            if arm != pair["arms"][name]:
                raise RuntimeError(f"Matched cache pair/arm {index} {name} differs")
            if (arm.get("case_index") != index or arm.get("case_id") != case["id"]
                    or arm.get("bucket") != case["bucket"]
                    or arm.get("prompt") != case["prompt"]
                    or arm.get("reference") != case["target"]
                     or arm.get("error") is not None or arm.get("tool_invocations") != 0
                     or arm.get("validation_errors") != []
                    or arm.get("first_logit_probe_matches_generated_first_token") is not True
                    or type(arm.get("sampled_token_count")) is not int
                    or not 0 < arm["sampled_token_count"] <= 2048
                    or not isinstance(arm.get("raw_answer"), str)
                    or not isinstance(arm.get("generated_tokens"), list)
                    or len(arm["generated_tokens"]) != arm["sampled_token_count"]
                    or not isinstance(arm.get("first_token"), dict)
                    or type(arm["first_token"].get("argmax_id")) is not int
                    or arm["first_token"]["argmax_id"] != arm["generated_tokens"][0]
                    or arm["first_token"].get("gold_first_character") != case["target"][0]
                    or arm.get("tool_trace") != []
                    or type(arm.get("ended_with_eos")) is not bool
                    or not isinstance(arm.get("stop_reason"), str)
                    or (arm["ended_with_eos"] != (arm["stop_reason"] == "eos"))
                    or (arm["ended_with_eos"] and arm["generated_tokens"][-1] != 3)):
                raise RuntimeError(f"Matched cache arm {index} {name} is incomplete")
            stop_counts[name][arm["stop_reason"]] += 1
            eos_counts[name] += int(arm["ended_with_eos"])
            origins[name] = {
                "kind": "verified_matched64_cache", "source_case_index": index,
                "source_pair_sha256": pair_sha,
                "source_arm_sha256": arm_sha,
            }
        records[case["id"]] = {"arms": pair["arms"], "origins": origins}
    if (summary.get("natural_eos") != eos_counts
            or summary.get("stop_reasons") != {name: dict(counts) for name, counts in stop_counts.items()}):
        raise RuntimeError("Matched cache stop summary differs from its 64 pairs")
    provenance = {"path": str(cache_dir), "claim_sha256": claim_sha,
                  "complete_sha256": complete_sha, "stop_summary_sha256": summary_sha,
                  "verified_pairs": 64, "reused_arm_labels": ["parent", "child"],
                  "same_gpu_runtime_as_intervention_verified": False}
    return records, provenance


@contextmanager
def temporary_parent_head(parent, child, *, require_cuda: bool = True):
    """Substitute only the child head's output, then remove the hook."""
    import torch
    from torch import nn
    from torch.nn import functional as F

    if parent.training or child.training:
        raise RuntimeError("Head substitution requires eval-mode models")
    old_head, child_head = parent.base.head, child.base.head
    if not isinstance(old_head, nn.Linear) or not isinstance(child_head, nn.Linear):
        raise TypeError("Expected the pinned linear output head")
    if old_head.bias is not None or child_head.bias is not None:
        raise RuntimeError("The pinned output head must be bias-free")
    if (old_head.weight.shape != child_head.weight.shape
            or old_head.weight.dtype != child_head.weight.dtype
            or old_head.weight.dtype != torch.float32
            or old_head.weight.device != child_head.weight.device
            or (require_cuda and child_head.weight.device.type != "cuda")):
        raise RuntimeError("Parent and child output heads do not share FP32 geometry")
    if old_head.weight.data_ptr() == child_head.weight.data_ptr():
        raise RuntimeError("Parent and child output heads share storage")
    original = child_head.weight.detach().clone()

    def substitute(_module, inputs, result):
        if torch.is_grad_enabled() or len(inputs) != 1 or inputs[0].shape[-1] != original.shape[1]:
            raise RuntimeError("Unexpected output-head call during read-only evaluation")
        replacement = F.linear(inputs[0], old_head.weight.detach())
        if replacement.shape != result.shape:
            raise RuntimeError("Restored head changed output geometry")
        return replacement

    handle = child_head.register_forward_hook(substitute)
    try:
        yield
    finally:
        handle.remove()
        if not torch.equal(child_head.weight.detach(), original):
            raise RuntimeError("Child head was changed despite the read-only hook")


def load_models(workspace: pathlib.Path, meta: dict):
    """Reuse the matched evaluator's pinned copies and strict migration path."""
    import torch

    matched.extract_members(matched.PURE_SOURCE, matched.SOURCE_FILES | {"hotfix_manifest.json"},
                            workspace / "pure")
    matched.extract_members(matched.V43_SOURCE,
                            {"base_metadata.json", "data/knowledge_corpus_v43.jsonl"},
                            workspace / "v43_data")
    loaded = sorted(pathlib.Path(name).stem for name in matched.SOURCE_FILES
                    if pathlib.Path(name).stem in sys.modules)
    if loaded:
        raise RuntimeError("Run in a fresh Python process; pinned modules are loaded: " + repr(loaded))
    sys.path.insert(0, str(workspace / "pure"))
    from GeoCPU_CED_v1 import CEDConfig
    from GeoCPU_CED_v2 import GeoCPU_CED_v2
    from integrated_model_v44_python import IntegratedModelV44Python, load_v43_neural_weights
    from local_search_engine_v42 import LocalSearchEngineV42

    config = dict(meta["config"])
    config["scan_backend"] = "jit"
    vocab = meta["vocab"]
    corpus = workspace / "v43_data" / "data" / "knowledge_corpus_v43.jsonl"
    data_manifest = json.loads(matched.DATASET_MANIFEST.read_text(encoding="utf-8"))
    models = {}
    for name, step in (("parent", 15_000), ("child", 25_000)):
        base = GeoCPU_CED_v2(CEDConfig(**config))
        model = IntegratedModelV44Python(base, search_engine=LocalSearchEngineV42(corpus_path=corpus))
        local_checkpoint = workspace / (name + ".pt")
        matched.copy_checked(name, local_checkpoint)
        payload = torch.load(local_checkpoint, map_location="cpu", weights_only=True, mmap=True)
        if type(payload.get("step")) is not int or payload["step"] != step:
            raise RuntimeError(name + " checkpoint step mismatch")
        if payload.get("meta", {}).get("vocab") != vocab:
            raise RuntimeError(name + " checkpoint vocabulary differs")
        if name == "parent":
            prior_config = dict(payload["meta"]["config"])
            if prior_config.get("max_positions") != 2048:
                raise RuntimeError("Parent context length differs")
            prior_config["max_positions"] = 8192
            if prior_config != config:
                raise RuntimeError("Parent config differs beyond context length")
            receipt = load_v43_neural_weights(
                model, SimpleNamespace(model_state_dict=payload["model_state_dict"],
                                       sha256=matched.PINS[name][2], step=step))
            if receipt["status"] != "strict_neural_migration_complete":
                raise RuntimeError("Strict parent migration failed")
        else:
            if (payload.get("meta") != meta
                    or payload.get("parent_sha256") != matched.PINS["parent"][2]
                    or payload.get("public_dataset_manifest") != data_manifest):
                raise RuntimeError("Child metadata, parent, or dataset binding differs")
            model.load_state_dict(payload["model_state_dict"], strict=True)
        model.to("cuda")
        model.eval()
        models[name] = model
        del base, payload
        gc.collect()
    return models


def evaluate_case(model, label: str, index: int, case: dict, vocab: dict, torch) -> dict:
    """Use the matched evaluator's decoding settings; retain partial answers."""
    from tir_engine_v44r1 import TIRLimits

    answer = model.generate_with_python_tir(
        case["prompt"], vocab, max_new_tokens=2048, arm="learned",
        enable_search=False, repetition_penalty=1.0,
        limits=TIRLimits(max_new_tokens=8192, max_context_tokens=8192,
                         max_tool_calls=0),
    )
    result = {
        "case_index": index, "case_id": case["id"], "bucket": case["bucket"],
        "prompt": case["prompt"], "reference": case["target"],
        "raw_answer": answer["output_text"],
        "generated_tokens": answer["generated_tokens"],
        "ended_with_eos": answer["ended_with_eos"],
        "stop_reason": answer["stop_reason"],
        "sampled_token_count": answer["sampled_token_count"],
        "tool_invocations": answer["tool_invocations"],
        "tool_trace": answer["tool_trace"], "error": answer["error"],
        "first_token": None, "first_logit_probe_matches_generated_first_token": False,
        "first_token_probe_error": None,
    }
    try:
        first = matched.probe_first_token(model, case["prompt"], case["target"], vocab, torch)
        result["first_token"] = first
        result["first_logit_probe_matches_generated_first_token"] = (
            bool(answer["generated_tokens"])
            and first["argmax_id"] == answer["generated_tokens"][0])
    except Exception as error:
        result["first_token_probe_error"] = repr(error)
    return result


def run(output: pathlib.Path, cohort: str, heldout_size: str,
        matched_cache: pathlib.Path | None) -> None:
    import torch
    from weight_direction.weight_direction import repeated_suffix

    if not torch.cuda.is_available() or not pathlib.Path("/content").is_dir():
        raise RuntimeError("This comparison requires an already authorized Colab GPU")
    if not matched.DRIVE.is_dir():
        raise RuntimeError("Drive must already be mounted")
    if shutil.disk_usage("/content").free < 8_000_000_000:
        raise RuntimeError("At least 8 GB free local disk is required")
    seen, heldout, broad, meta, evidence = static_preflight(heldout_size)
    if matched_cache is not None and cohort == "practice":
        raise ValueError("A matched64 cache is useful only when evaluating held-out cases")
    cached, cache_provenance = ({}, None) if matched_cache is None else verified_matched_cache(
        matched_cache, broad, evidence)
    groups = (("seen_v43_train", seen), ("heldout", heldout))
    if cohort != "both":
        groups = tuple((name, rows) for name, rows in groups
                       if name == ("seen_v43_train" if cohort == "practice" else "heldout"))
    output.mkdir(parents=True, exist_ok=False)
    matched.write_json_new(output / "claim.json", {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": matched.sha256_file(pathlib.Path(__file__)),
        "matched_evaluator_sha256": matched.sha256_file(pathlib.Path(matched.__file__)),
        "checkpoint_sha256": {name: matched.PINS[name][2] for name in ("parent", "child")},
        "candidate": "replace child base.head output with parent base.head projection",
        "weights_modified": False, "training_updates": 0, "dev_used_for_selection": False,
        "reference_use": "practice references are legacy v43 traces; no semantic grading",
        "matched_cache": cache_provenance,
        "same_gpu_runtime_across_arms": cache_provenance is None,
        "decode": {"max_new_tokens": 2048, "arm": "learned", "search": False,
                   "repetition_penalty": 1.0, "python_tool_calls": 0, "autocast": False},
        "selection_evidence": evidence,
    })
    with tempfile.TemporaryDirectory(prefix="geocpu_head_restore_", dir="/content") as temp:
        models = load_models(pathlib.Path(temp), meta=meta)
        vocab = meta["vocab"]
        arms = ("parent", "child", "child_parent_head")
        stop_counts = {group_name: {name: Counter() for name in arms}
                       for group_name, _cases in groups}
        eos_counts = {group_name: {name: 0 for name in arms}
                      for group_name, _cases in groups}
        origin_counts = {group_name: {name: Counter() for name in arms}
                         for group_name, _cases in groups}
        for group_name, cases in groups:
            for index, case in enumerate(cases):
                pair = {"cohort": group_name, "case_id": case["id"],
                        "case_index": index, "source": case["bucket"],
                        "reference_protocol": case.get("reference_protocol", "heldout_target"),
                        "arms": {}, "arm_origins": {}}
                for label in arms:
                    try:
                        if group_name == "heldout" and label in ("parent", "child") and cached:
                            entry = cached[case["id"]]
                            result = dict(entry["arms"][label])
                            result["case_index"] = index
                            origin = dict(entry["origins"][label])
                        elif label == "child_parent_head":
                            with temporary_parent_head(models["parent"], models["child"]):
                                result = evaluate_case(models["child"], label, index, case, vocab, torch)
                            origin = {"kind": "generated_this_run"}
                        else:
                            result = evaluate_case(models[label], label, index, case, vocab, torch)
                            origin = {"kind": "generated_this_run"}
                    except Exception as error:
                        matched.write_json_new(output / "errors" / f"{group_name}_{index:02d}_{label}.json",
                                               {"cohort": group_name, "case_id": case["id"],
                                                "arm": label, "error": repr(error)})
                        raise
                    result["repetition"] = repeated_suffix(result["raw_answer"])
                    result["origin"] = origin
                    pair["arms"][label] = result
                    pair["arm_origins"][label] = origin
                    origin_counts[group_name][label][origin["kind"]] += 1
                    matched.write_json_new(output / "arms" / group_name / f"{index:02d}_{label}.json", result)
                    if (result["error"] is not None or result["tool_invocations"] != 0
                            or result.get("first_token_probe_error") is not None
                            or not result["first_logit_probe_matches_generated_first_token"]
                            or type(result.get("sampled_token_count")) is not int
                            or not 0 < result["sampled_token_count"] <= 2048
                            or len(result["generated_tokens"]) != result["sampled_token_count"]
                            or type(result.get("ended_with_eos")) is not bool
                            or (result["ended_with_eos"] != (result["stop_reason"] == "eos"))):
                        raise RuntimeError(f"{group_name} case {index} {label} failed the read-only generation checks")
                    stop_counts[group_name][label][result["stop_reason"]] += 1
                    eos_counts[group_name][label] += int(result["ended_with_eos"])
                matched.write_json_new(output / "pairs" / group_name / f"{index:02d}.json", pair)
        matched.write_json_new(output / "COMPLETE.json", {
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "cohorts": {name: len(rows) for name, rows in groups},
            "stop_reasons": {group_name: {name: dict(counts) for name, counts in by_arm.items()}
                             for group_name, by_arm in stop_counts.items()},
            "natural_eos": eos_counts, "semantic_correctness_scored": False,
            "python_tool_use_scored": False, "training_updates": 0,
            "matched_cache": cache_provenance,
            "same_gpu_runtime_across_arms": cache_provenance is None,
            "arm_origin_counts": {
                group_name: {name: dict(counts) for name, counts in by_arm.items()}
                for group_name, by_arm in origin_counts.items()},
        })
        del models
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--cohort", choices=("practice", "heldout", "both"), default="practice")
    parser.add_argument("--heldout-size", choices=("mini8", "mixed64"), default="mini8")
    parser.add_argument("--matched-cache", type=pathlib.Path,
                        help="Completed matched mixed64 output to verify and reuse for held-out parent/child arms")
    parser.add_argument("--output-dir", type=pathlib.Path)
    args = parser.parse_args()
    if args.preflight:
        seen, heldout, broad, _meta, evidence = static_preflight(args.heldout_size)
        if args.matched_cache is not None:
            _cached, provenance = verified_matched_cache(args.matched_cache, broad, evidence)
            print("MATCHED_CACHE_OK", provenance["verified_pairs"], provenance["claim_sha256"])
        print("STATIC_PREFLIGHT_OK", len(seen), len(heldout), evidence["dev_used_for_selection"])
        return
    output = args.output_dir or (matched.HOTFIX / ("head_restore_probe_" + uuid.uuid4().hex))
    run(output, args.cohort, args.heldout_size, args.matched_cache)
    print("READ_ONLY_HEAD_RESTORE_EVAL_COMPLETE", output)


if __name__ == "__main__":
    main()
