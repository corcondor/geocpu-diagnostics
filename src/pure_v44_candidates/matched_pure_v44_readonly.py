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
from datetime import datetime, timezone
from types import SimpleNamespace


DRIVE = pathlib.Path("/content/drive/MyDrive")
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

PINS = {
    "v43_source": (V43_SOURCE, 67_976_709, "0102dc9b01b4b795c3123b18cefc7a8734e47a06dd85292af464d3232704d198"),
    "pure_source": (PURE_SOURCE, 129_010, "3f58351ef4576e4e25ecee9a1a2037ba1c3ef2d397eb605e3a5df01eb6f462a9"),
    "parent": (PARENT, 990_545_719, "a4b95d7d0f486146196271b3b1fbc6a1b6b3342d8fadd590320fc3842c1e80e5"),
    "child": (CHILD, 2_971_632_900, "28308d0445829b8bf137f801f98b7573f80e699fb00e97d56a15523158ee984a"),
    "cases": (CASES, 15_355, "672c12f7fee9faf28e7733117dfe30a85689aec6c060b6bc2a53253e67ce1365"),
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
    if not isinstance(occurrences, dict) or sorted(occurrences.values()) != [1, 1, 1, 1]:
        raise RuntimeError("Pinned fixed cases are not unique in the test split")
    for row in cases:
        if not isinstance(row.get("prompt"), str) or not isinstance(row.get("target"), str):
            raise RuntimeError("Invalid fixed evaluation case")


def static_preflight() -> tuple[list[dict], dict]:
    for name in ("v43_source", "pure_source", "cases"):
        check_pin(name)
    if not SEPARATION.is_file():
        raise FileNotFoundError(SEPARATION)
    with zipfile.ZipFile(V43_SOURCE) as source:
        original_meta = source.read("base_metadata.json")
        corpus = source.read("data/knowledge_corpus_v43.jsonl")
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
    cases = json.loads(CASES.read_text(encoding="utf-8"))["cases"]
    validate_cases(cases, json.loads(SEPARATION.read_text(encoding="utf-8")))
    return cases, json.loads(pure_meta)


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


def evaluate_arm(model, label: str, cases: list[dict], vocab: dict, output: pathlib.Path, torch) -> None:
    model.eval()
    results = []
    for index, case in enumerate(cases):
        answer = model.generate_with_python_tir(
            case["prompt"], vocab, max_new_tokens=2048, arm="learned",
            enable_search=False, repetition_penalty=1.0,
        )
        # Probe after generation so an extra prefill cannot affect the answer.
        first = probe_first_token(model, case["prompt"], case["target"], vocab, torch)
        first_matches_sample = bool(answer["generated_tokens"]) and first["argmax_id"] == answer["generated_tokens"][0]
        results.append({
            "case_index": index,
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
        })
        write_json_new(output / f"{label}_case_{index:02d}.json", results[-1])
        if answer["error"] is not None:
            raise RuntimeError(f"{label} case {index} generation error: {answer['error']}")
        if not first_matches_sample:
            raise RuntimeError(f"{label} case {index} first-logit replay differs from generated first token")
    write_json_new(output / f"{label}_complete.json", {"label": label, "cases": len(results)})


def run(output: pathlib.Path) -> None:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("A GPU is required for the 247M-model read-only comparison")
    if pathlib.Path("/content/drive/MyDrive").is_dir() is False:
        raise RuntimeError("Drive must already be mounted; no new authorization is requested")
    if shutil.disk_usage("/content").free < 8_000_000_000:
        raise RuntimeError("At least 8 GB free local disk is required for verified checkpoint copies")
    cases, meta = static_preflight()
    output.mkdir(parents=True, exist_ok=False)
    write_json_new(output / "claim.json", {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "kind": "read_only_pure_v44_same_code_parent_vs_step25000",
        "source_pins": {key: value[2] for key, value in PINS.items()},
        "cases": len(cases), "training_updates": 0,
        "decode": {"max_new_tokens": 2048, "arm": "learned", "search": False,
                   "repetition_penalty": 1.0, "autocast": False},
    })
    with tempfile.TemporaryDirectory(prefix="geocpu_matched_pure_", dir="/content") as temp:
        workspace = pathlib.Path(temp)
        extract_members(PURE_SOURCE, SOURCE_FILES | {"hotfix_manifest.json"}, workspace / "pure")
        extract_members(V43_SOURCE, {"base_metadata.json", "data/knowledge_corpus_v43.jsonl"}, workspace / "v43_data")
        corpus = workspace / "v43_data" / "data" / "knowledge_corpus_v43.jsonl"
        vocab = meta["vocab"]
        sys.path.insert(0, str(workspace / "pure"))
        from GeoCPU_CED_v1 import CEDConfig
        from GeoCPU_CED_v2 import GeoCPU_CED_v2
        from integrated_model_v44_python import IntegratedModelV44Python, load_v43_neural_weights
        from local_search_engine_v42 import LocalSearchEngineV42

        config = dict(meta["config"])
        config["scan_backend"] = "jit"
        for name, step in (("parent", 15_000), ("child", 25_000)):
            # Fresh module and caches for each arm; only the checkpoint weights differ.
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
                model.load_state_dict(payload["model_state_dict"], strict=True)
            model.to("cuda")
            model.eval()
            evaluate_arm(model, name, cases, vocab, output, torch)
            del model, base, payload
            gc.collect()
            torch.cuda.empty_cache()
        write_json_new(output / "COMPLETE.json", {
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "parent_cases": 4, "child_cases": 4,
            "same_source_zip_sha256": PINS["pure_source"][2],
            "same_prompt_file_sha256": PINS["cases"][2],
            "training_updates": 0,
        })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=pathlib.Path)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        cases, meta = static_preflight()
        print("STATIC_PREFLIGHT_OK", len(cases), len(meta["vocab"]), PURE_META_SHA256)
        return
    output = args.output_dir or (HOTFIX / ("matched_pure_eval_" + uuid.uuid4().hex))
    run(output)
    print("READ_ONLY_MATCHED_EVAL_COMPLETE", output)


if __name__ == "__main__":
    main()
