"""Read-only, same-runtime comparison of the pure v44 parent and trained weights.

Run on an already authorized Colab GPU with Drive mounted. This does not train,
save checkpoints, or change the original v43/v44 files. The two arms use the
same pure-v44 Hotfix code, prompts, decoding settings, and CUDA precision.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import pathlib
import shutil
import sys
import tempfile
import uuid
import zipfile
from collections import Counter
from datetime import datetime, timezone
from types import SimpleNamespace


DRIVE = pathlib.Path(os.environ.get("GEOCPU_DRIVE_ROOT", "/content/drive/MyDrive"))
V43 = DRIVE / "GeoCPU_Frontier_CoTraining_v43"
HOTFIX = DRIVE / "GeoCPU_Frontier_CoTraining_v44_PythonTIR_Hotfix_20261103"
RUN = HOTFIX / "run_1" / "training"

V43_SOURCE = V43 / "Grand_cotraining_v43.zip"
PURE_SOURCE = HOTFIX / "GeoCPU_v44_PythonTIR_hotfix.zip"
PARENT = V43 / "checkpoints" / "checkpoint_step_15000.pt"
CHILD = RUN / (
    "checkpoint_step_25000_aa6c44ab47a14297b5c6481a5fb1374d_"
    "27867b30ef624c8985a22b81ae7789fa.pt"
)
CASES = RUN / "fixed_public_cases.json"
SEPARATION = RUN / "recovery_data_validation.json"
DATASET_MANIFEST = HOTFIX / "run_1" / "dataset_manifest.json"

# Fixed before either checkpoint is loaded. This is the previously pinned
# Japanese32 selection with one predeclared replacement: idiom:0016 contains
# characters outside the model vocabulary, so idiom:0018 takes its place.
JAPANESE32_IDS = (
    "jmmlu:762cbf192c9e:human_sexuality:0087",
    "jmmlu:762cbf192c9e:japanese_idiom:0032",
    "jmmlu:762cbf192c9e:japanese_idiom:0018",
    "jmmlu:762cbf192c9e:japanese_geography:0120",
    "jmmlu:762cbf192c9e:high_school_mathematics:0027",
    "swe_code_0126", "arc_agi_0197",
    "jmmlu:762cbf192c9e:high_school_european_history:0140",
    "jmmlu:762cbf192c9e:prehistory:0060",
    "jmmlu:762cbf192c9e:logical_fallacies:0148",
    "jmmlu:762cbf192c9e:elementary_mathematics:0052",
    "jmmlu:762cbf192c9e:high_school_psychology:0104",
    "jmmlu:762cbf192c9e:logical_fallacies:0102",
    "jmmlu:762cbf192c9e:college_biology:0029",
    "jmmlu:762cbf192c9e:nutrition:0084", "arc_agi_0115",
    "jmmlu:762cbf192c9e:miscellaneous:0073",
    "jmmlu:762cbf192c9e:machine_learning:0064",
    "jmmlu:762cbf192c9e:college_mathematics:0058",
    "jmmlu:762cbf192c9e:elementary_mathematics:0077",
    "jmmlu:762cbf192c9e:econometrics:0011",
    "jmmlu:762cbf192c9e:japanese_geography:0095",
    "jmmlu:762cbf192c9e:high_school_psychology:0062",
    "jmmlu:762cbf192c9e:security_studies:0041",
    "jmmlu:762cbf192c9e:logical_fallacies:0136",
    "jmmlu:762cbf192c9e:high_school_microeconomics:0128",
    "jmmlu:762cbf192c9e:japanese_geography:0063", "swe_code_0102",
    "jmmlu:762cbf192c9e:astronomy:0030", "arc_agi_0240",
    "jmmlu:762cbf192c9e:high_school_computer_science:0069",
    "jmmlu:762cbf192c9e:sociology:0030",
)
EVAL_MEMBERS = {
    "benchmark": ("data/benchmark_8100_manifest.jsonl", 4_877_203,
                  "33723d78aaa1595b1ed7dbc3382ecb165ce788246f6f70a63470f5fa107de78e"),
    "swe": ("data/eval_swe_step_control_heldout.jsonl", 82_008,
            "0bf62099ea2a41c8eb35c1ebdc2584f2c164ae3c82f741ab3eca44113f70f8cd"),
    "logic": ("data/eval_v41_logical_instruction_operators_heldout.jsonl", 433_800,
              "b887aade00323a7241088a4e84dfcb166430f554209a09668ad204ea564e9989"),
}
JAPANESE32_SHA256 = "9b6d266d531f15d5acbda7c0606462f508924a5c6031c2f013291cedc332e2ea"

PINS = {
    "v43_source": (V43_SOURCE, 67_976_709, "0102dc9b01b4b795c3123b18cefc7a8734e47a06dd85292af464d3232704d198"),
    "pure_source": (PURE_SOURCE, 129_010, "3f58351ef4576e4e25ecee9a1a2037ba1c3ef2d397eb605e3a5df01eb6f462a9"),
    "parent": (PARENT, 990_545_719, "a4b95d7d0f486146196271b3b1fbc6a1b6b3342d8fadd590320fc3842c1e80e5"),
    "child": (CHILD, 2_971_632_900, "28308d0445829b8bf137f801f98b7573f80e699fb00e97d56a15523158ee984a"),
    "cases": (CASES, 15_355, "672c12f7fee9faf28e7733117dfe30a85689aec6c060b6bc2a53253e67ce1365"),
    "separation": (SEPARATION, 1,
                   "2b812ca0cc1f7f49ee3bd5e7511bec4185d6f58c88a451592930e8b4123bfade"),
    "dataset_manifest": (DATASET_MANIFEST, 40_139,
                         "9e5eece13b71fe33f26a68bdf22dfb7cc9367864be80d95ebdcb19cca33a4678"),
}
ORIGINAL_META_SHA256 = "7efa8dea9a0100e2250f72805c89cca7a281efec30b4479795743814756cdcd5"
PURE_META_SHA256 = "ca53934facb94bd17cb7bb881bdf223edbc653eb90f62fdda3b7663e64f3e677"
CORPUS_SHA256 = "3118998dc32687868757227d7939d29e0180cad69892256a2747cf89659d826a"
SOURCE_FILES = {
    "GeoCPU_CED_v1.py",
    "GeoCPU_CED_v2.py",
    "fast_encoder_scan.py",
    "official_dc_reference_v22.py",
    "local_search_engine_v42.py",
    "neural_model_v30_python_base.py",
    "integrated_model_v44_python.py",
    "tir_engine_v44r1.py",
    "executor_worker_v44r1.py",
    "prepare_public_v44.py",
}


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_pin(name: str, *, full_hash: bool = True) -> None:
    path, minimum_bytes, expected = PINS[name]
    if not path.is_file() or path.stat().st_size < minimum_bytes:
        raise RuntimeError(f"Pinned {name} is missing or smaller than expected")
    if full_hash and sha256_file(path) != expected:
        raise RuntimeError(f"Pinned {name} SHA-256 mismatch")


def copy_checked(name: str, target: pathlib.Path) -> None:
    source, _, expected = PINS[name]
    if target.exists():
        raise FileExistsError(target)
    digest = hashlib.sha256()
    with source.open("rb") as reader, target.open("xb") as writer:
        for block in iter(lambda: reader.read(8 << 20), b""):
            writer.write(block)
            digest.update(block)
        writer.flush()
        os.fsync(writer.fileno())
    if digest.hexdigest() != expected:
        raise RuntimeError(f"Copied {name} SHA-256 mismatch")


def extract_members(archive: pathlib.Path, names: set[str], target: pathlib.Path) -> None:
    target.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(archive) as bundle:
        available = set(bundle.namelist())
        if not names <= available:
            raise RuntimeError("Required source members missing: " + repr(sorted(names - available)))
        for name in sorted(names):
            if pathlib.PurePosixPath(name).is_absolute() or ".." in pathlib.PurePosixPath(name).parts:
                raise RuntimeError("Unsafe archive member: " + name)
            output = target / name
            output.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(name) as reader, output.open("xb") as writer:
                shutil.copyfileobj(reader, writer, length=8 << 20)


def write_json_new(path: pathlib.Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())


def validate_cases(cases: list[dict], separation: dict) -> None:
    if len(cases) != 4 or [row.get("bucket") for row in cases] != ["tir", "tir", "japanese", "japanese"]:
        raise RuntimeError("Pinned four-case selection differs")
    if separation.get("train_dev_fixed_case_overlap") != 0:
        raise RuntimeError("Fixed evaluation cases overlap training")
    occurrences = separation.get("fixed_case_test_occurrences")
    expected_groups = {row.get("provenance", {}).get("question_group_sha256") for row in cases}
    if (not isinstance(occurrences, dict) or len(expected_groups) != 4
            or None in expected_groups or set(occurrences) != expected_groups
            or any(type(count) is not int or count != 1 for count in occurrences.values())):
        raise RuntimeError("Pinned fixed cases are not unique in the test split")
    for row in cases:
        if not isinstance(row.get("prompt"), str) or not isinstance(row.get("target"), str):
            raise RuntimeError("Invalid fixed evaluation case")


def load_broad_cases(source: zipfile.ZipFile, suite: str) -> tuple[list[dict], dict]:
    members = {}
    for label, (name, size, expected_sha) in EVAL_MEMBERS.items():
        content = source.read(name)
        if len(content) != size or hashlib.sha256(content).hexdigest() != expected_sha:
            raise RuntimeError("Pinned held-out member differs: " + label)
        members[label] = content
    benchmark_lines = members["benchmark"].splitlines()
    by_id = {json.loads(line)["id"]: line for line in benchmark_lines}
    if len(by_id) != 8_100 or len(set(JAPANESE32_IDS)) != 32:
        raise RuntimeError("Benchmark IDs are duplicated or missing")
    if any(case_id not in by_id for case_id in JAPANESE32_IDS):
        raise RuntimeError("Pinned Japanese32 case ID is missing")
    fixed_bytes = bytes([13, 10]).join(by_id[case_id] for case_id in JAPANESE32_IDS) + bytes([13, 10])
    if hashlib.sha256(fixed_bytes).hexdigest() != JAPANESE32_SHA256:
        raise RuntimeError("Reconstructed Japanese32 differs from prior pinned selection")
    rows = [dict(json.loads(by_id[case_id]), bucket="japanese32") for case_id in JAPANESE32_IDS]
    if suite == "mixed64":
        swe = [json.loads(line) for line in members["swe"].splitlines()]
        logic = [json.loads(line) for line in members["logic"].splitlines()]
        if len(swe) != 200 or len(logic) != 400:
            raise RuntimeError("Pinned held-out suite length differs")
        rows.extend(dict(row, bucket="swe16") for row in swe[::12][:16])
        rows.extend(dict(row, bucket="logic16") for row in logic[::25][:16])
    if len(rows) != (64 if suite == "mixed64" else 32):
        raise RuntimeError("Broad suite length differs")
    prompts = [row.get("prompt") for row in rows]
    if any(not isinstance(prompt, str) or not prompt for prompt in prompts) or len(set(prompts)) != len(prompts):
        raise RuntimeError("Broad suite prompts are missing or duplicated")
    if any(not isinstance(row.get("target"), str) or not row["target"] for row in rows):
        raise RuntimeError("Broad suite reference is missing")
    return rows, {
        "suite": suite, "selection": "fixed Japanese32 IDs; mixed64 adds every 12th SWE and every 25th logic row",
        "source_member_sha256": {key: spec[2] for key, spec in EVAL_MEMBERS.items()},
        "japanese32_sha256": JAPANESE32_SHA256,
    }


def check_training_overlap(cases: list[dict], train_jsonl: pathlib.Path, manifest: dict) -> dict:
    train_info = manifest["outputs"]["train.jsonl"]
    if train_jsonl.stat().st_size != train_info["bytes"] or sha256_file(train_jsonl) != train_info["sha256"]:
        raise RuntimeError("Training file differs from the pinned Hotfix training output")
    evaluation_prompts = {row["prompt"] for row in cases}
    overlap = set()
    with train_jsonl.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("prompt") in evaluation_prompts:
                overlap.add(row["prompt"])
    if overlap:
        raise RuntimeError(f"Exact training-prompt overlap: {len(overlap)}")
    return {"actual_train_file_sha256": train_info["sha256"], "exact_prompt_overlap": 0,
            "actual_train_rows_directly_scanned": True}


def static_preflight(suite: str = "mixed64", train_jsonl: pathlib.Path | None = None) -> tuple[list[dict], dict, dict]:
    required_pins = ("v43_source", "pure_source", "dataset_manifest")
    if suite == "smoke4":
        required_pins += ("cases", "separation")
    for name in required_pins:
        check_pin(name)
    with zipfile.ZipFile(V43_SOURCE) as source:
        original_meta = source.read("base_metadata.json")
        corpus = source.read("data/knowledge_corpus_v43.jsonl")
        broad_cases, broad_evidence = load_broad_cases(source, suite) if suite != "smoke4" else (None, None)
    if hashlib.sha256(original_meta).hexdigest() != ORIGINAL_META_SHA256:
        raise RuntimeError("V43 source metadata does not match the pinned original")
    old = b'"max_positions": 2048'
    if original_meta.count(old) != 1:
        raise RuntimeError("Expected one original context-length field")
    pure_meta = original_meta.replace(old, b'"max_positions": 8192')
    if hashlib.sha256(pure_meta).hexdigest() != PURE_META_SHA256:
        raise RuntimeError("The only v44 metadata change is not the pinned context extension")
    if hashlib.sha256(corpus).hexdigest() != CORPUS_SHA256:
        raise RuntimeError("The offline corpus differs from the pinned v44 source")
    with zipfile.ZipFile(PURE_SOURCE) as source:
        manifest = json.loads(source.read("hotfix_manifest.json"))
        for name in SOURCE_FILES:
            if hashlib.sha256(source.read(name)).hexdigest() != manifest["files"][name]["sha256"]:
                raise RuntimeError("Source member differs from pinned Hotfix manifest: " + name)
    if suite == "smoke4":
        cases = json.loads(CASES.read_text(encoding="utf-8"))["cases"]
        validate_cases(cases, json.loads(SEPARATION.read_text(encoding="utf-8")))
    else:
        cases = broad_cases
    data_manifest = json.loads(DATASET_MANIFEST.read_text(encoding="utf-8"))
    eval_pins = {(item.get("sha256"), item.get("bytes")) for item in data_manifest["evaluation_inputs"]}
    if any((spec[2], spec[1]) not in eval_pins for spec in EVAL_MEMBERS.values()):
        raise RuntimeError("Hotfix builder manifest does not bind all broad evaluation members")
    vocab = json.loads(pure_meta)["vocab"]
    for index, case in enumerate(cases):
        if len(case["prompt"]) > 8192 or set(case["prompt"]) - set(vocab):
            raise RuntimeError(f"Case {index} exceeds context or has unknown prompt characters")
        if case["target"][0] not in vocab:
            raise RuntimeError(f"Case {index} has unknown first reference character")
    evidence = {"suite": suite, "fixed_four_case_overlap_receipt": "checked" if suite == "smoke4" else "not used",
                "training_manifest_sha256": PINS["dataset_manifest"][2],
                "source_level_eval_exclusion": "Pinned prepare_public_v44.py indexes these three eval files and rejects canonical question keys and IDs",
                "actual_train_rows_directly_scanned": False}
    if broad_evidence:
        evidence.update(broad_evidence)
    if train_jsonl is not None:
        evidence.update(check_training_overlap(cases, train_jsonl, data_manifest))
    return cases, json.loads(pure_meta), evidence


def probe_first_token(model, prompt: str, target: str, vocab: dict, torch) -> dict:
    if not prompt or not target or set(prompt) - set(vocab):
        raise RuntimeError("Empty or out-of-vocabulary evaluation input")
    input_ids = torch.tensor([[vocab[char] for char in prompt]], dtype=torch.long, device="cuda")
    with torch.inference_mode():
        state = model.prefill(input_ids, arm="learned")
        logits = state.last_logits.reshape(-1, state.last_logits.shape[-1])[-1].float().cpu()
        probabilities = torch.softmax(logits, dim=-1)
    chosen = int(logits.argmax())
    expected = int(vocab[target[0]])
    rank = int((logits > logits[expected]).sum()) + 1
    return {
        "argmax_id": chosen,
        "argmax_character": next((char for char, token in vocab.items() if token == chosen), None),
        "gold_first_character": target[0],
        "gold_first_rank": rank,
        "gold_first_probability": float(probabilities[expected]),
        "eos_probability": float(probabilities[3]),
        "top5": [[int(token), float(logits[token])] for token in torch.topk(logits, 5).indices.tolist()],
    }


def evaluate_case(model, label: str, index: int, case: dict, vocab: dict, torch) -> dict:
    from tir_engine_v44r1 import TIRLimits

    model.eval()
    answer = model.generate_with_python_tir(
        case["prompt"], vocab, max_new_tokens=2048, arm="learned",
        enable_search=False, repetition_penalty=1.0,
        # The original executor is not a security sandbox. This diagnostic
        # compares weights while Drive is mounted, so generated code must
        # never be executed. The first fenced block ends the run instead.
        limits=TIRLimits(max_new_tokens=8192, max_context_tokens=8192,
                         max_tool_calls=0),
    )
    # Probe after generation so an extra prefill cannot affect the answer.
    first = probe_first_token(model, case["prompt"], case["target"], vocab, torch)
    first_matches_sample = bool(answer["generated_tokens"]) and first["argmax_id"] == answer["generated_tokens"][0]
    result = {
        "case_index": index,
        "case_id": case.get("id"),
        "bucket": case["bucket"],
        "prompt": case["prompt"],
        "reference": case["target"],
        "first_token": first,
        "first_logit_probe_matches_generated_first_token": first_matches_sample,
        "raw_answer": answer["output_text"],
        "generated_tokens": answer["generated_tokens"],
        "ended_with_eos": answer["ended_with_eos"],
        "stop_reason": answer["stop_reason"],
        "sampled_token_count": answer["sampled_token_count"],
        "tool_invocations": answer["tool_invocations"],
        "tool_trace": answer["tool_trace"],
        "error": answer["error"],
    }
    result["validation_errors"] = []
    if answer["error"] is not None:
        result["validation_errors"].append("generation_error")
    if answer["tool_invocations"] != 0:
        result["validation_errors"].append("unexpected_python_tool_call")
    if not first_matches_sample:
        result["validation_errors"].append("first_logit_replay_mismatch")
    return result


def run(output: pathlib.Path, suite: str, train_jsonl: pathlib.Path | None) -> None:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("A GPU is required for the 247M-model read-only comparison")
    if pathlib.Path("/content/drive/MyDrive").is_dir() is False:
        raise RuntimeError("Drive must already be mounted; no new authorization is requested")
    if shutil.disk_usage("/content").free < 8_000_000_000:
        raise RuntimeError("At least 8 GB free local disk is required for verified checkpoint copies")
    cases, meta, evidence = static_preflight(suite, train_jsonl)
    output.mkdir(parents=True, exist_ok=False)
    write_json_new(output / "claim.json", {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "kind": "read_only_pure_v44_same_code_parent_vs_step25000",
        "source_pins": {key: value[2] for key, value in PINS.items()},
        "cases": len(cases), "training_updates": 0,
        "separation_evidence": evidence,
        "decode": {"max_new_tokens": 2048, "arm": "learned", "search": False,
                   "repetition_penalty": 1.0, "autocast": False,
                   "python_tool_calls": 0,
                   "per_case_wall_seconds": 180,
                   "scope": "same-code neural-weight diagnosis; not full Python tool-use evaluation"},
    })
    with tempfile.TemporaryDirectory(prefix="geocpu_matched_pure_", dir="/content") as temp:
        workspace = pathlib.Path(temp)
        extract_members(PURE_SOURCE, SOURCE_FILES | {"hotfix_manifest.json"}, workspace / "pure")
        extract_members(V43_SOURCE, {"base_metadata.json", "data/knowledge_corpus_v43.jsonl"}, workspace / "v43_data")
        corpus = workspace / "v43_data" / "data" / "knowledge_corpus_v43.jsonl"
        vocab = meta["vocab"]
        already_loaded = sorted(pathlib.Path(name).stem for name in SOURCE_FILES
                                if pathlib.Path(name).stem in sys.modules)
        if already_loaded:
            raise RuntimeError("Run this evaluator in a fresh Python process; pinned modules are already loaded: "
                               + repr(already_loaded))
        sys.path.insert(0, str(workspace / "pure"))
        from GeoCPU_CED_v1 import CEDConfig
        from GeoCPU_CED_v2 import GeoCPU_CED_v2
        from integrated_model_v44_python import IntegratedModelV44Python, load_v43_neural_weights
        from local_search_engine_v42 import LocalSearchEngineV42

        config = dict(meta["config"])
        config["scan_backend"] = "jit"
        models = {}
        data_manifest = json.loads(DATASET_MANIFEST.read_text(encoding="utf-8"))
        for name, step in (("parent", 15_000), ("child", 25_000)):
            # Independent module and caches for each arm; only the checkpoint
            # weights differ. Keep both models loaded to save matched pairs
            # after each question instead of losing all pairs on interruption.
            base = GeoCPU_CED_v2(CEDConfig(**config))
            model = IntegratedModelV44Python(base, search_engine=LocalSearchEngineV42(corpus_path=corpus))
            local_checkpoint = workspace / (name + ".pt")
            copy_checked(name, local_checkpoint)
            payload = torch.load(local_checkpoint, map_location="cpu", weights_only=True, mmap=True)
            if type(payload.get("step")) is not int or payload["step"] != step:
                raise RuntimeError(name + " checkpoint step mismatch")
            if payload.get("meta", {}).get("vocab") != vocab:
                raise RuntimeError(name + " checkpoint vocabulary IDs differ")
            if name == "child" and payload["meta"] != meta:
                raise RuntimeError("Child checkpoint metadata differs from the pinned v44 metadata")
            if name == "parent":
                parent_config = dict(payload["meta"]["config"])
                if parent_config.get("max_positions") != 2048:
                    raise RuntimeError("Parent checkpoint context length is not 2048")
                parent_config["max_positions"] = 8192
                if parent_config != config:
                    raise RuntimeError("Parent and child configs differ beyond context length")
            if name == "parent":
                receipt = load_v43_neural_weights(
                    model, SimpleNamespace(model_state_dict=payload["model_state_dict"],
                                           sha256=PINS[name][2], step=step),
                )
                if receipt["status"] != "strict_neural_migration_complete":
                    raise RuntimeError("Strict parent migration failed")
            else:
                if payload.get("parent_sha256") != PINS["parent"][2]:
                    raise RuntimeError("Child does not bind to pinned parent")
                if payload.get("public_dataset_manifest") != data_manifest:
                    raise RuntimeError("Child does not bind to pinned training manifest")
                model.load_state_dict(payload["model_state_dict"], strict=True)
            model.to("cuda")
            model.eval()
            models[name] = model
            del base, payload
            gc.collect()
        stop_counts = {"parent": Counter(), "child": Counter()}
        eos_counts = {"parent": 0, "child": 0}
        for index, case in enumerate(cases):
            pair = {"case_index": index, "case_id": case.get("id"), "arms": {}}
            for name in ("parent", "child"):
                try:
                    result = evaluate_case(models[name], name, index, case, vocab, torch)
                except Exception as exc:
                    write_json_new(output / "arms" / f"{name}_case_{index:02d}_failure.json", {
                        "case_index": index, "case_id": case.get("id"),
                        "arm": name, "error": repr(exc),
                    })
                    raise
                pair["arms"][name] = result
                write_json_new(output / "arms" / f"{name}_case_{index:02d}.json", result)
                if result["validation_errors"]:
                    raise RuntimeError(f"{name} case {index} failed validation: {result['validation_errors']}")
                stop_counts[name][result["stop_reason"]] += 1
                eos_counts[name] += int(result["ended_with_eos"])
            write_json_new(output / "pairs" / f"case_{index:02d}.json", pair)
        write_json_new(output / "observed_stop_summary.json", {
            "cases_per_arm": len(cases),
            "natural_eos": eos_counts,
            "stop_reasons": {name: dict(counts) for name, counts in stop_counts.items()},
            "meaning_quality_scored": False,
            "python_tool_use_scored": False,
        })
        del models
        gc.collect()
        torch.cuda.empty_cache()
        write_json_new(output / "COMPLETE.json", {
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "parent_cases": len(cases), "child_cases": len(cases),
            "same_source_zip_sha256": PINS["pure_source"][2],
            "suite": suite, "separation_evidence": evidence,
            "training_updates": 0,
            "observed_stop_summary": "observed_stop_summary.json",
            "meaning_quality_scored": False,
        })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=pathlib.Path)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--suite", choices=("smoke4", "japanese32", "mixed64"), required=True)
    parser.add_argument("--train-jsonl", type=pathlib.Path,
                        help="Optional pinned Hotfix train.jsonl for independent exact prompt overlap scan")
    args = parser.parse_args()
    if args.preflight:
        cases, meta, evidence = static_preflight(args.suite, args.train_jsonl)
        print("STATIC_PREFLIGHT_OK", len(cases), len(meta["vocab"]), PURE_META_SHA256,
              evidence["actual_train_rows_directly_scanned"])
        return
    output = args.output_dir or (HOTFIX / ("matched_pure_eval_" + uuid.uuid4().hex))
    run(output, args.suite, args.train_jsonl)
    print("READ_ONLY_MATCHED_EVAL_COMPLETE", output)


if __name__ == "__main__":
    main()
