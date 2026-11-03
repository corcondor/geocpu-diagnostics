"""Read-only matched raw answers for one bounded adapter stage.

Run in a fresh isolated Kaggle process with local assets. The evaluated
checkpoint is never updated; incomplete results remain explicitly partial.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from adapter import (AdapterSpec, adapter_parameters, file_sha256,
                     install_adapters, resume_adapter_checkpoint)
from admission import (PINNED_V43_META_SHA256, require_independent_execution_verification,
                       require_v43_native_baseline, verify_dataset,
                       verify_source_manifest)
from train_stage import (PARENT_BYTES, PARENT_SHA256, PARENT_STORAGE_COUNT,
                         _source_imports)


def _hash_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def _write_record(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    for name in ("source_dir", "source_manifest_sha256", "parent", "meta", "meta_sha256",
                 "data_dir", "data_manifest_sha256", "adapter", "adapter_sha256", "output_dir"):
        parser.add_argument("--" + name.replace("_", "-"), required=True)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--execute-python", action="store_true")
    parser.add_argument("--hard-stop-utc", required=True)
    parser.add_argument("--verification-proof-dir", required=True)
    parser.add_argument("--verification-proof-sha256", required=True)
    parser.add_argument("--v43-native-proof-dir", required=True)
    parser.add_argument("--v43-native-capture-sha256", required=True)
    parser.add_argument("--v43-native-grade-sha256", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        hard_stop = datetime.fromisoformat(args.hard_stop_utc.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Invalid evaluation hard-stop UTC") from exc
    if hard_stop.tzinfo is None or hard_stop.utcoffset() != timedelta(0):
        raise ValueError("Evaluation hard-stop must explicitly be UTC")
    source = verify_source_manifest(args.source_dir, args.source_manifest_sha256)
    data = verify_dataset(args.data_dir, args.data_manifest_sha256)
    verification = require_independent_execution_verification(
        data, data_dir=args.data_dir, proof_dir=args.verification_proof_dir,
        expected_proof_sha256=args.verification_proof_sha256,
        expected_meta_sha256=args.meta_sha256,
        expected_source_manifest_sha256=args.source_manifest_sha256)
    v43_native = require_v43_native_baseline(
        args.data_dir, proof_dir=args.v43_native_proof_dir,
        expected_capture_sha256=args.v43_native_capture_sha256,
        expected_grade_sha256=args.v43_native_grade_sha256)
    if (file_sha256(args.parent) != PARENT_SHA256
            or args.meta_sha256 != PINNED_V43_META_SHA256
            or file_sha256(args.meta) != args.meta_sha256):
        raise ValueError("Parent checkpoint or metadata differs")
    if file_sha256(args.adapter) != args.adapter_sha256:
        raise ValueError("Adapter checkpoint differs")
    if args.execute_python and (Path("/content/drive/MyDrive").exists() or not Path("/kaggle").exists()):
        raise ValueError("Python execution requires a separate Kaggle runtime without a mounted Drive")
    if hard_stop <= datetime.now(timezone.utc) + timedelta(minutes=6):
        raise ValueError("Evaluation deadline must allow at least six minutes")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Matched raw-answer comparison requires an existing GPU")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    raw = out / "paired_raw_answers.jsonl"
    _write_json(out / "evaluation_claim.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
                                                "source": source, "data": data,
                                                "parent_sha256": PARENT_SHA256,
                                                "adapter_sha256": args.adapter_sha256,
                                                "independent_verification": verification,
                                                "v43_native_baseline": v43_native,
                                                "max_new_tokens": 2048,
                                                "hard_stop_utc": hard_stop.isoformat(),
                                                "model_python_execution": args.execute_python})
    (CEDConfig, GeoCPU_CED_v2, IntegratedModelV44Python, load_v43_neural_weights,
     recover_checkpoint, _StrictDataset, _batches_for_lengths, _collate) = _source_imports(args.source_dir)
    from tir_engine_v44r1 import TIRLimits
    meta = json.loads(Path(args.meta).read_text(encoding="utf-8"))
    config = dict(meta["config"])
    config["scan_backend"] = "jit"
    model = IntegratedModelV44Python(GeoCPU_CED_v2(CEDConfig(**config)), search_engine=None)
    recovered = recover_checkpoint(args.parent, expected_sha256=PARENT_SHA256,
                                   staging_dir=str(out / "parent_staging"), expected_step=15_000,
                                   expected_bytes=PARENT_BYTES, expected_storage_count=PARENT_STORAGE_COUNT)
    load_v43_neural_weights(model, recovered)
    del recovered
    spec = AdapterSpec(rank=args.rank, alpha=args.alpha)
    names = install_adapters(model, spec)
    model.to("cuda")
    optimizer = torch.optim.AdamW(adapter_parameters(model, names), lr=1e-4,
                                  betas=(0.9, 0.95), weight_decay=0.01, fused=True)
    limits = TIRLimits(max_new_tokens=2048, max_context_tokens=min(8192, config["max_positions"]),
                       max_tool_calls=8 if args.execute_python else 0, max_wall_seconds=180.0)
    cases = []
    for split in ("old_skill_eval", "unseen_eval"):
        with (Path(args.data_dir) / (split + ".jsonl")).open(encoding="utf-8") as stream:
            cases.extend((split, json.loads(line)) for line in stream if line.strip())
    # Parent and candidate see exactly the same cases, source code and token cap.
    for arm in ("parent", "adapter"):
        if arm == "adapter":
            resume_adapter_checkpoint(args.adapter, expected_sha256=args.adapter_sha256,
                                      expected_parent_sha256=PARENT_SHA256,
                                      expected_train_manifest_sha256=args.data_manifest_sha256,
                                      model=model, names=names, spec=spec, optimizer=optimizer)
        model.eval()
        for split, case in cases:
            if datetime.now(timezone.utc) >= hard_stop - timedelta(seconds=360):
                partial = {"schema": "geocpu_bounded_adapter_paired_eval_v1",
                           "status": "PARTIAL_DEADLINE", "completed_records": sum(1 for _ in raw.open(encoding="utf-8")) if raw.exists() else 0,
                           "raw_answer_sha256": file_sha256(raw) if raw.exists() else None,
                           "adapter_sha256": args.adapter_sha256, "data_manifest_sha256": args.data_manifest_sha256,
                           "hard_stop_utc": hard_stop.isoformat()}
                _write_json(out / "partial_result.json", partial)
                return partial
            record = {"schema": "geocpu_bounded_adapter_raw_answer_v1", "arm": arm,
                      "split": split, "id": case["id"], "family_id": case["family_id"],
                      "prompt_sha256": _hash_text(case["prompt"]),
                      "expected_answer_sha256": _hash_text(case["expected_answer"]),
                      "expected_answer_given_to_generation": False}
            try:
                answer = model.generate_with_python_tir(case["prompt"], meta["vocab"],
                                                        max_new_tokens=2048, arm="learned",
                                                        repetition_penalty=1.0, enable_search=False,
                                                        limits=limits)
                record.update(status="observed", raw_output=answer["output_text"],
                              generated_text=answer["model_generated_text"],
                              output_sha256=_hash_text(answer["output_text"]),
                              stop_reason=answer["stop_reason"], ended_with_eos=answer["ended_with_eos"],
                              tool_invocations=answer["tool_invocations"],
                              successful_tool_invocations=answer["successful_tool_invocations"],
                              tool_trace=answer["tool_trace"], error=answer["error"])
            except BaseException as error:
                record.update(status="failed", error=repr(error), raw_output=None)
                _write_record(raw, record)
                _write_json(out / "FAILED.json", {"error": repr(error), "completed_records": sum(1 for _ in raw.open(encoding="utf-8"))})
                raise
            _write_record(raw, record)
    total = len(cases) * 2
    result = {"schema": "geocpu_bounded_adapter_paired_eval_v1", "status": "COMPLETE",
              "records": total, "cases_per_arm": len(cases), "max_new_tokens": 2048,
              "parent_sha256": PARENT_SHA256, "adapter_sha256": args.adapter_sha256,
              "data_manifest_sha256": args.data_manifest_sha256,
              "independent_verification": verification,
              "v43_native_baseline": v43_native,
              "raw_answer_sha256": file_sha256(raw), "model_python_execution": args.execute_python,
              "semantic_grading_status": "PENDING_INDEPENDENT_GRADE"}
    _write_json(out / "paired_result.json", result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return result


if __name__ == "__main__":
    main()
