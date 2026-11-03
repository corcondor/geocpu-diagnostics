"""Capture complete v43 CPU-native SWE200 answers with the original evaluator.

This is an entrypoint for a separate, already authorized Kaggle GPU runtime.
It never launches a job, updates a model, or writes to its source/checkpoint.
The original v43 evaluator decides correctness; this wrapper retains the
unstripped output, CPU trace, exact source row, and file lineage it omits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import unicodedata
import zipfile
from pathlib import Path


SOURCE_SHA256 = "0102dc9b01b4b795c3123b18cefc7a8734e47a06dd85292af464d3232704d198"
PARENT_SHA256 = "a4b95d7d0f486146196271b3b1fbc6a1b6b3342d8fadd590320fc3842c1e80e5"
PARENT_BYTES = 990_545_719
META_SHA256 = "7efa8dea9a0100e2250f72805c89cca7a281efec30b4479795743814756cdcd5"
SWE_MEMBER = "data/eval_swe_step_control_heldout.jsonl"
SWE_SHA256 = "0bf62099ea2a41c8eb35c1ebdc2584f2c164ae3c82f741ab3eca44113f70f8cd"
CORPUS_MEMBER = "data/knowledge_corpus_v43.jsonl"
CORPUS_SHA256 = "3118998dc32687868757227d7939d29e0180cad69892256a2747cf89659d826a"
EVALUATOR_SHA256 = "c3589b97f531b6460a95907e56284df5f20be990eff1a4d2e56a921b4c705899"
GENERATION_LIMIT = 2048


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def evaluation_family_id(prompt: str) -> str:
    """Use the pure-v44 builder's canonical-question family convention."""
    value = unicodedata.normalize("NFKC", prompt).strip()
    value = re.sub(r"^(?:ユーザー|user)\s*[:：]\s*", "", value, flags=re.I)
    value = re.sub(r"\n\s*(?:アシスタント|assistant)\s*[:：]\s*$", "", value, flags=re.I)
    value = re.sub(r"^\s*【分野:[^\n]*】\s*", "", value)
    value = re.sub(r"(?m)^\s*分野\s*[:：][^\n]*\n?", "", value)
    value = re.sub(r"\s*(?:正解|回答|答え)\s*[:：]\s*$", "", value)
    key = re.sub(r"\s+", "", value)
    return sha256_bytes(("family:" + key).encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_new_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def preflight(source_zip: Path, parent: Path) -> tuple[dict, list[dict], bytes, bytes]:
    source_zip, parent = source_zip.resolve(strict=True), parent.resolve(strict=True)
    if sha256_file(source_zip) != SOURCE_SHA256:
        raise ValueError("Original v43 source ZIP SHA differs")
    if parent.stat().st_size != PARENT_BYTES or sha256_file(parent) != PARENT_SHA256:
        raise ValueError("Original step15000 checkpoint size/SHA differs")
    with zipfile.ZipFile(source_zip) as archive:
        meta_bytes = archive.read("base_metadata.json")
        eval_bytes = archive.read(SWE_MEMBER)
        corpus_bytes = archive.read(CORPUS_MEMBER)
        evaluator_bytes = archive.read("eval_step_control_heldout.py")
    if (sha256_bytes(meta_bytes) != META_SHA256
            or sha256_bytes(eval_bytes) != SWE_SHA256
            or sha256_bytes(corpus_bytes) != CORPUS_SHA256
            or sha256_bytes(evaluator_bytes) != EVALUATOR_SHA256):
        raise ValueError("Original v43 metadata/evaluation/corpus/evaluator member differs")
    raw_lines = [line for line in eval_bytes.splitlines(keepends=True) if line.strip()]
    cases = []
    for index, line in enumerate(raw_lines):
        case = json.loads(line)
        if (not isinstance(case, dict) or not isinstance(case.get("id"), str)
                or not isinstance(case.get("prompt"), str)
                or not isinstance(case.get("target"), str)
                or not case["id"] or not case["prompt"] or not case["target"]):
            raise ValueError(f"SWE source row {index} is incomplete")
        cases.append({"source_index": index, "source_raw_line_sha256": sha256_bytes(line), **case})
    if (len(cases) != 200 or len({row["id"] for row in cases}) != 200
            or len({row["prompt"] for row in cases}) != 200):
        raise ValueError("Pinned SWE200 count, IDs, or prompts differ")
    return json.loads(meta_bytes), cases, eval_bytes, corpus_bytes


class CaptureProxy:
    """Keep complete native outputs while the pinned evaluator scores them."""

    def __init__(self, model, cases: list[dict], capture_path: Path):
        self.model = model
        self.cases = cases
        self.records: list[dict] = []
        self.stream = capture_path.open("x", encoding="utf-8", newline="\n")

    def eval(self):
        self.model.eval()
        return self

    def generate_with_qwenputer(self, prompt, vocab, inv_vocab, *, max_tokens=None,
                                 arm="learned", repetition_penalty=1.0):
        index = len(self.records)
        if index >= len(self.cases):
            raise RuntimeError("Original evaluator requested an extra case")
        case = self.cases[index]
        if (prompt != case["prompt"] or max_tokens is not None
                or arm != "learned" or repetition_penalty != 1.0):
            raise RuntimeError("Original evaluator changed the pinned prompt/generation profile")
        result = self.model.generate_with_qwenputer(
            prompt, vocab, inv_vocab, max_tokens=GENERATION_LIMIT,
            arm=arm, repetition_penalty=repetition_penalty)
        output = result["output_text"]
        tokens = result["tokens"]
        trace = result["cpu_trace"]
        if (not isinstance(output, str) or not isinstance(tokens, list)
                or not isinstance(trace, list) or type(result["ended_with_eos"]) is not bool
                or type(result["cpu_invocations"]) is not int
                or len(tokens) > GENERATION_LIMIT):
            raise RuntimeError(f"Native generation case {index} has invalid or over-cap output")
        row = {
            "id": case["id"], "source_index": index,
            "source_member": SWE_MEMBER, "source_member_sha256": SWE_SHA256,
            "source_raw_line_sha256": case["source_raw_line_sha256"],
            "parent_sha256": PARENT_SHA256, "profile": "v43_cpu_native",
            "prompt": prompt, "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
            "expected_answer": case["target"].strip(),
            "raw_output": output, "raw_output_sha256": sha256_bytes(output.encode("utf-8")),
            "generated_tokens": tokens, "generated_token_count": len(tokens),
            "cpu_trace": trace, "cpu_invocations": result["cpu_invocations"],
            "ended_with_eos": result["ended_with_eos"],
            "natural_eos": result["ended_with_eos"],
            "stop_reason": "eos" if result["ended_with_eos"] else "native_non_eos_unspecified",
            "generation_limit": GENERATION_LIMIT, "full_answer_captured": True,
            "reference_given_to_generation": False,
        }
        self.stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.records.append(row)
        return result

    def close(self):
        self.stream.close()


def run(source_zip: Path, parent: Path, output_dir: Path) -> dict:
    meta, cases, eval_bytes, corpus_bytes = preflight(source_zip, parent)
    if not Path("/kaggle").is_dir() or Path("/content/drive/MyDrive").exists():
        raise RuntimeError("Run this entrypoint only in a separate Kaggle runtime without mounted Drive")
    local_working = Path("/kaggle/working").resolve(strict=True)
    output_dir = output_dir.resolve(strict=False)
    if output_dir == local_working or not output_dir.is_relative_to(local_working):
        raise ValueError("New output directory must be below /kaggle/working")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("An existing Kaggle CUDA device is required")
    if any(name in sys.modules for name in ("GeoCPU_CED_v1", "GeoCPU_CED_v2",
            "integrated_model_v30", "integrated_model_v42", "eval_step_control_heldout",
            "local_search_engine_v42")):
        raise RuntimeError("Use a fresh Python process for original v43 source imports")
    output_dir.mkdir(parents=True, exist_ok=False)
    write_new_json(output_dir / "claim.json", {
        "source_zip_sha256": SOURCE_SHA256, "parent_sha256": PARENT_SHA256,
        "meta_sha256": META_SHA256, "eval_member_sha256": SWE_SHA256,
        "original_evaluator_sha256": EVALUATOR_SHA256,
        "script_sha256": sha256_file(Path(__file__)),
        "suite": "heldout_swe200", "generation_limit": GENERATION_LIMIT,
        "generation_method": "original IntegratedModelV42 inherited generate_with_qwenputer",
        "score_method": "original eval_step_control_heldout.run_heldout_evaluation",
        "training_updates": 0, "parent_checkpoint_modified": False,
    })
    try:
        with tempfile.TemporaryDirectory(prefix="v43_native_swe_", dir="/kaggle/working") as temp:
            temporary = Path(temp)
            manifest_path = temporary / "eval_swe_step_control_heldout.jsonl"
            corpus_path = temporary / "knowledge_corpus_v43.jsonl"
            manifest_path.write_bytes(eval_bytes)
            corpus_path.write_bytes(corpus_bytes)
            sys.path.insert(0, str(source_zip.resolve(strict=True)))
            from GeoCPU_CED_v1 import CEDConfig
            from GeoCPU_CED_v2 import GeoCPU_CED_v2
            from integrated_model_v42 import IntegratedModelV42
            from local_search_engine_v42 import LocalSearchEngineV42
            from eval_step_control_heldout import run_heldout_evaluation

            config = dict(meta["config"])
            config["scan_backend"] = "jit"
            if config.get("max_positions") != GENERATION_LIMIT:
                raise RuntimeError("Original v43 context/generation limit differs")
            model = IntegratedModelV42(
                base=GeoCPU_CED_v2(CEDConfig(**config)),
                search_engine=LocalSearchEngineV42(corpus_path=corpus_path))
            payload = torch.load(parent, map_location="cpu", weights_only=True, mmap=True)
            if (type(payload.get("step")) is not int or payload["step"] != 15_000
                    or payload.get("meta") != meta):
                raise RuntimeError("Original checkpoint step or metadata differs")
            model.load_state_dict(payload["model_state_dict"], strict=True)
            del payload
            model.to("cuda").eval()
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            inverse = {int(token): char for char, token in meta["vocab"].items()}
            capture = CaptureProxy(model, cases, output_dir / "swe200_full_answers.jsonl")
            try:
                legacy_stats = run_heldout_evaluation(
                    model=capture, vocab=meta["vocab"], inv_vocab=inverse,
                    manifest_path=str(manifest_path), arm="learned", device="cuda",
                    output_audit_path=str(output_dir / "original_swe_audit.jsonl"))
            finally:
                capture.close()
        audit = read_jsonl(output_dir / "original_swe_audit.jsonl")
        if len(audit) != 200 or len(capture.records) != 200:
            raise RuntimeError("Original evaluator did not complete SWE200")
        grades = []
        for case, answer, verdict in zip(cases, capture.records, audit, strict=True):
            if (answer["id"] != case["id"] or verdict["id"] != case["id"]
                    or verdict["raw_output"] != answer["raw_output"].strip()
                    or verdict["target"] != case["target"].strip()
                    or verdict["cpu_invocations"] != answer["cpu_invocations"]
                    or verdict["ended_with_eos"] != answer["ended_with_eos"]):
                raise RuntimeError("Original SWE verdict does not bind to captured full answer")
            grades.append({
                "id": case["id"], "prompt_sha256": answer["prompt_sha256"],
                "raw_output_sha256": answer["raw_output_sha256"],
                "expected_answer": answer["expected_answer"],
                "correct": verdict["is_correct"],
                "final_answer": verdict["final_answer"],
                "expected_loops": verdict["expected_loops"],
                "executed_loops": verdict["executed_loops"],
                "step_category": verdict["step_category"],
                "scorer_sha256": EVALUATOR_SHA256,
                "reason": "Original v43 SWE evaluator: final CPU R0 write when present, otherwise last emitted word; exact target match.",
            })
        correct_total = sum(row["correct"] for row in grades)
        strict_with_eos = sum(grade["correct"] and answer["ended_with_eos"]
                              for answer, grade in zip(capture.records, grades, strict=True))
        if (correct_total != legacy_stats["correct_val"]
                or strict_with_eos != legacy_stats["strict_with_eos"]):
            raise RuntimeError("Original SWE aggregate differs from per-case grades")
        write_new_jsonl(output_dir / "swe200_original_grades.jsonl", grades)
        accepted = [(case, answer, grade) for case, answer, grade in
                    zip(cases, capture.records, grades, strict=True)
                    if grade["correct"] and answer["ended_with_eos"]]
        write_new_jsonl(output_dir / "v43_native_full_answers.jsonl",
                        [answer for _, answer, _ in accepted])
        write_new_jsonl(output_dir / "v43_native_external_grades.jsonl",
                        [grade for _, _, grade in accepted])
        write_new_jsonl(output_dir / "v43_original_answers.jsonl", [
            {"id": answer["id"], "profile": "v43_cpu_native",
             "parent_sha256": PARENT_SHA256, "correct": True,
             "prompt_sha256": answer["prompt_sha256"],
             "raw_output": answer["raw_output"],
             "raw_output_sha256": answer["raw_output_sha256"],
             "expected_answer": answer["expected_answer"]}
            for _, answer, _ in accepted])
        old_skill_eval = [
            {"id": case["id"], "family_id": evaluation_family_id(answer["prompt"]),
             "bucket": "swe_step_control", "prompt": answer["prompt"],
             "prompt_sha256": answer["prompt_sha256"],
             "expected_answer": answer["expected_answer"],
             "v43_cpu_task": True, "v43_original_correct": True,
             "v43_original_raw_sha256": answer["raw_output_sha256"]}
            for case, answer, _ in accepted]
        if any(type(row["v43_cpu_task"]) is not bool for row in old_skill_eval):
            raise RuntimeError("Old-skill CPU-task classification is missing")
        write_new_jsonl(output_dir / "old_skill_eval.jsonl", old_skill_eval)
        # Exercise the downstream baseline schema against the actual output files.
        from admission import require_v43_native_baseline
        native_proof = require_v43_native_baseline(
            output_dir, proof_dir=output_dir,
            expected_capture_sha256=sha256_file(output_dir / "v43_native_full_answers.jsonl"),
            expected_grade_sha256=sha256_file(output_dir / "v43_native_external_grades.jsonl"))
        files = ("swe200_full_answers.jsonl", "swe200_original_grades.jsonl",
                 "original_swe_audit.jsonl", "v43_native_full_answers.jsonl",
                 "v43_native_external_grades.jsonl", "v43_original_answers.jsonl",
                 "old_skill_eval.jsonl")
        source_after = sha256_file(source_zip)
        parent_after = sha256_file(parent)
        with zipfile.ZipFile(source_zip) as archive:
            evaluator_after = sha256_bytes(archive.read("eval_step_control_heldout.py"))
        if (source_after != SOURCE_SHA256 or parent_after != PARENT_SHA256
                or evaluator_after != EVALUATOR_SHA256):
            raise RuntimeError("Original source ZIP, evaluator, or parent checkpoint changed during capture")
        result = {
            "status": "COMPLETE", "suite": "heldout_swe200",
            "cases": 200, "originally_correct": correct_total,
            "originally_correct_and_natural_eos": strict_with_eos,
            "eligible_old_skill_rows": len(accepted),
            "old_skill_minimum_64_met": len(accepted) >= 64,
            "source_zip_sha256": SOURCE_SHA256, "parent_sha256": PARENT_SHA256,
            "source_zip_sha256_after": source_after,
            "parent_sha256_after": parent_after,
            "source_and_parent_unchanged": True,
            "eval_member_sha256": SWE_SHA256,
            "original_evaluator_sha256": EVALUATOR_SHA256,
            "original_evaluator_sha256_after": evaluator_after,
            "generation_limit": GENERATION_LIMIT, "training_updates": 0,
            "downstream_native_baseline_schema_check": native_proof,
            "files_sha256": {name: sha256_file(output_dir / name) for name in files},
        }
        write_new_json(output_dir / "COMPLETE.json", result)
        return result
    except BaseException as error:
        write_new_json(output_dir / "FAILED.json", {"error": repr(error), "status": "INCOMPLETE"})
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-zip", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    meta, cases, _, _ = preflight(args.source_zip, args.parent)
    if args.preflight:
        print(json.dumps({"status": "STATIC_PREFLIGHT_OK", "cases": len(cases),
                          "max_positions": meta["config"]["max_positions"],
                          "source_zip_sha256": SOURCE_SHA256,
                          "parent_sha256": PARENT_SHA256}, ensure_ascii=False))
        return
    if args.output_dir is None:
        parser.error("--output-dir is required unless --preflight is selected")
    print(json.dumps(run(args.source_zip, args.parent, args.output_dir), ensure_ascii=False))


if __name__ == "__main__":
    main()
