"""Isolated, resumable v43 + small Python TIR stage on a disposable Kaggle GPU.

The v43 step-15000 file has model weights, config and meta only.  Stage 1 is a
strict weight warm start with a newly seeded optimizer.  From its first saved
hybrid checkpoint onward Adam, every RNG, the complete exposure schedule and
the accepted update cursor resume exactly.  No older job or parent is edited.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
import zipfile

from hybrid_data import (STAGE_PYTHON_CAPS, collate_exact, plan_exposure,
                         verify_realized_update)
from source_pools import (PINNED_V43_ZIP_SHA, PYTHON_MODE_CUE, file_sha, load_verified_pools)
from prelaunch_contract import verify_prelaunch_contract

PARENT_CHECKPOINT_SHA = "a4b95d7d0f486146196271b3b1fbc6a1b6b3342d8fadd590320fc3842c1e80e5"
PARENT_CHECKPOINT_BYTES = 990_545_719
PINNED_SWE_EVAL_SHA = "0bf62099ea2a41c8eb35c1ebdc2584f2c164ae3c82f741ab3eca44113f70f8cd"
PINNED_SWE_EVALUATOR_SHA = "c3589b97f531b6460a95907e56284df5f20be990eff1a4d2e56a921b4c705899"
PINNED_INTERIM_EVALUATOR_SHA = "bfce8f14f28e8e3714d30234da445e3d2c1e47a97e8666ca0a959e0dad630d8b"
PINNED_INTERIM_CASES_SHA = "4032a2c17de2ce56a7758cd35c16db2c2b357ce76b394522411e86c0aa682b45"
STEPS_PER_STAGE = 1_000
FIRST_EVALUATION_BARRIER = 250
BATCH_SIZE = 64
DEFAULT_MAX_MICRO_TOKENS = 24_576
CHECKPOINT_EVERY = 250
SOURCE_OVERLAY_SHA = {
    "integrated_model_v44fix.py": "2718cb61734590c7329aaacee1eccc872eb545d32a1aaa532c230a8206affc22",
    "tir_engine_v44r1.py": "375d0844b616f3913b942db11edd425e78bf6ed13b8fa803c7e3ea183739538e",
    "executor_worker_v44r1.py": "b3a8f35d8d56857cfa7eaa50f0dc3d31a42eea9eaf1692ad6c01ff938f02f44d",
    "resource_limited_worker_v44r1.py": "d8563d23ddc86277fa87aa17cdbc78decc5e2022215b5d20e7cf107e13fc4294",
}


def _binding(*, stage: int, arm: str, parent_sha: str, v43_zip: Path, converted: Path,
             replay_receipt: Path, plan_sha: str, boundary_plan_sha: str, source_root: Path,
             seed: int, lr: float, micro_tokens: int, resource_run_id: str,
             precision_profile: str, gpu_names: list[str]) -> dict:
    return {"schema": "geocpu_v43_python_gradual_stage_v1", "stage": stage,
            "arm": arm,
            "python_actual_target_cap": 0.0 if arm == "control" else STAGE_PYTHON_CAPS[stage-1],
            "parent_sha256": parent_sha, "v43_zip_sha256": file_sha(v43_zip),
            "converted_sha256": file_sha(converted),
            "replay_complete_sha256": file_sha(replay_receipt),
            "plan_sha256": plan_sha, "stage_source_sha256": file_sha(__file__),
            "source_pools_sha256": file_sha(Path(__file__).with_name("source_pools.py")),
            "hybrid_data_sha256": file_sha(Path(__file__).with_name("hybrid_data.py")),
            "common_boundary_plan_sha256": boundary_plan_sha,
            "prelaunch_verifier_sha256": file_sha(Path(__file__).with_name("prelaunch_contract.py")),
            "resource_run_id": resource_run_id,
            "precision_profile": precision_profile, "gpu_names": gpu_names,
            "overlay_sha256": {name: file_sha(source_root / name) for name in SOURCE_OVERLAY_SHA},
            "seed": seed, "lr": lr, "steps": STEPS_PER_STAGE,
            "batch_size": BATCH_SIZE, "max_micro_tokens": micro_tokens,
            "ce_normalization": "sum_all_active_labels_per_update",
            "compression_loss": "0.2_times_sum_per_example_over_64", 
            "microbatch_boundary": "paired_old_or_python_max_tokens"}


def _prepare_source(v43_zip: Path, overlay: Path, scratch: Path):
    if scratch.exists():
        raise FileExistsError("Scratch source directory already exists")
    if file_sha(v43_zip) != PINNED_V43_ZIP_SHA:
        raise ValueError("Original v43 source ZIP changed")
    for name, expected in SOURCE_OVERLAY_SHA.items():
        if file_sha(overlay / name) != expected:
            raise ValueError("Pinned additive v44fix source changed: " + name)
    scratch.mkdir(parents=True)
    with zipfile.ZipFile(v43_zip) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            relative = Path(member.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe original source ZIP member")
            destination = scratch / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, destination.open("xb") as sink:
                shutil.copyfileobj(source, sink)
    for name in SOURCE_OVERLAY_SHA:
        shutil.copyfile(overlay / name, scratch / name)
    _verify_extracted_source(v43_zip, overlay, scratch)
    return scratch


def _verify_extracted_source(v43_zip: Path, overlay: Path, scratch: Path):
    if scratch.is_symlink():
        raise ValueError("Isolated source root is a symbolic link")
    if file_sha(v43_zip) != PINNED_V43_ZIP_SHA:
        raise ValueError("Original source ZIP changed before or during resume")
    with zipfile.ZipFile(v43_zip) as archive:
        expected_files = {member.filename for member in archive.infolist() if not member.is_dir()}
        expected_files.update(SOURCE_OVERLAY_SHA)
        for member in archive.infolist():
            if member.is_dir() or member.filename in SOURCE_OVERLAY_SHA:
                continue
            relative = Path(member.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe original source ZIP member")
            existing = scratch / relative
            digest = sha256()
            with archive.open(member) as stream:
                for block in iter(lambda: stream.read(8 << 20), b""):
                    digest.update(block)
            if not existing.is_file() or file_sha(existing) != digest.hexdigest():
                raise ValueError("Extracted original source/data changed: " + member.filename)
    for name, expected in SOURCE_OVERLAY_SHA.items():
        if (file_sha(overlay / name) != expected
                or file_sha(scratch / name) != expected):
            raise ValueError("Additive Python runtime source changed: " + name)
    actual_files = set()
    for candidate in scratch.rglob("*"):
        if candidate.is_symlink():
            raise ValueError("Isolated source contains a symbolic link")
        if candidate.is_file():
            actual_files.add(candidate.relative_to(scratch).as_posix())
    if actual_files != expected_files:
        raise ValueError("Isolated source contains a missing or unpinned file")


def _rng_capture(torch, np):
    np_state = np.random.get_state()
    return {"torch_cpu": torch.get_rng_state().clone(),
            "torch_cuda": [state.clone() for state in torch.cuda.get_rng_state_all()],
            "python": random.getstate(),
            "numpy": (np_state[0], np_state[1].tolist(), int(np_state[2]),
                      int(np_state[3]), float(np_state[4]))}


def _rng_restore(state, torch, np):
    if set(state) != {"torch_cpu", "torch_cuda", "python", "numpy"}:
        raise ValueError("Hybrid checkpoint RNG is incomplete")
    if len(state["torch_cuda"]) != torch.cuda.device_count():
        raise ValueError("CUDA RNG device count differs")
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), int(n[2]), int(n[3]), float(n[4])))


def _save_checkpoint(path, *, model, optimizer, torch, np, binding, update,
                     old_active, python_active, status, scaler, fp16_skipped_attempts):
    if path.exists() or path.with_suffix(path.suffix + ".receipt.json").exists():
        raise FileExistsError("Hybrid checkpoint name already exists")
    temp = path.with_name(path.name + ".partial")
    if temp.exists():
        raise FileExistsError("Partial checkpoint requires manual inspection")
    payload = {"schema": binding["schema"], "binding": binding,
               "accepted_updates": update, "model_state_dict": model.state_dict(),
               "optimizer_state_dict": optimizer.state_dict(),
               "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
               "rng": _rng_capture(torch, np),
               "old_active_positions": old_active,
               "python_active_positions": python_active,
               "fp16_skipped_attempts": fp16_skipped_attempts,
               "status": status}
    torch.save(payload, temp)
    os.replace(temp, path)
    receipt = {"schema": "geocpu_v43_python_gradual_checkpoint_receipt_v1",
               "bytes": path.stat().st_size, "sha256": file_sha(path),
               "binding_sha256": sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest(),
               "accepted_updates": update,
               "fp16_skipped_attempts": fp16_skipped_attempts,
               "status": status}
    with path.with_suffix(path.suffix + ".receipt.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2)
        stream.write("\n")
    return receipt


def _load_stage_checkpoint(path, *, torch, expected_binding=None):
    receipt_path = path.with_suffix(path.suffix + ".receipt.json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (receipt.get("bytes") != path.stat().st_size
            or receipt.get("sha256") != file_sha(path)):
        raise ValueError("Hybrid checkpoint file and receipt differ")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    binding = payload.get("binding")
    if (payload.get("schema") != "geocpu_v43_python_gradual_stage_v1"
            or receipt.get("binding_sha256") != sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
            or (expected_binding is not None and binding != expected_binding)
            or receipt.get("accepted_updates") != payload.get("accepted_updates")
            or receipt.get("status") != payload.get("status")
            or type(payload.get("fp16_skipped_attempts")) is not int
            or payload["fp16_skipped_attempts"] < 0
            or receipt.get("fp16_skipped_attempts") != payload["fp16_skipped_attempts"]):
        raise ValueError("Hybrid checkpoint binding or cursor differs")
    is_fp16 = binding.get("precision_profile") == "train_fp16_gradscaler_v1"
    scaler_state = payload.get("scaler_state_dict")
    if (is_fp16 and (not isinstance(scaler_state, dict) or not scaler_state)
            or not is_fp16 and (scaler_state is not None or payload["fp16_skipped_attempts"] != 0)):
        raise ValueError("Saved AMP scaler state differs from the fixed precision profile")
    return payload


def _microbatches(rows, common_slot_lengths, budget):
    if len(rows) != len(common_slot_lengths):
        raise ValueError("The paired old-row schedule is incomplete")
    chunks = []
    chunk = []
    maximum = 0
    for row, slot_length in zip(rows, common_slot_lengths):
        if len(row.tokens) > slot_length:
            raise ValueError("The common microbatch envelope is smaller than an actual row")
        next_maximum = max(maximum, ((slot_length+31)//32)*32)
        if next_maximum > budget:
            raise ValueError("A source example exceeds the microbatch token budget")
        if chunk and next_maximum * (len(chunk)+1) > budget:
            chunks.append(chunk)
            chunk, maximum = [], 0
        chunk.append(row)
        maximum = max(maximum, ((slot_length+31)//32)*32)
    if chunk:
        chunks.append(chunk)
    return chunks


def _finish_fp16_attempt(torch, model, optimizer, scaler) -> bool:
    """Return False only when GradScaler skipped the optimizer step."""
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0,
                                             error_if_nonfinite=False)
    finite_norm = bool(torch.isfinite(norm))
    old_scale = float(scaler.get_scale())
    if not finite_norm:
        # GradScaler checks each gradient for inf, but a float32 aggregate
        # norm can overflow even when every individual gradient is finite.
        # In that case it may still perform an optimizer step. Never call it.
        if not math.isfinite(old_scale) or old_scale <= 0:
            raise FloatingPointError("Invalid fp16 gradient scale")
        new_scale = old_scale / 2.0 if old_scale > 2 ** -24 else old_scale
        scaler.update(new_scale=new_scale)
        if float(scaler.get_scale()) != new_scale:
            raise FloatingPointError("Nonfinite norm did not retain the fixed fp16 scale")
        return False
    scaler.step(optimizer)
    scaler.update()
    new_scale = float(scaler.get_scale())
    if new_scale < old_scale:
        return False
    return True


def _retry_same_slot(*, attempt, restore_after_skip, can_continue,
                     max_skips=3) -> tuple[bool, int]:
    """Keep one fixed curriculum slot pending until its optimizer step occurs."""
    skipped = 0
    while can_continue():
        if attempt():
            return True, skipped
        skipped += 1
        restore_after_skip()
        if skipped >= max_skips:
            break
    return False, skipped


def _run_one_update(*, rows, micro, total_active, model, optimizer, scaler,
                    autocast_dtype, torch, np, F, can_continue) -> tuple[bool, int]:
    """Retry the same 64-row slot after an fp16 overflow; count only real steps."""
    rng_before = _rng_capture(torch, np)
    buffer_before = {name: buffer.detach().clone() for name, buffer in model.named_buffers()}
    def attempt():
        optimizer.zero_grad(set_to_none=True)
        for selected in micro:
            batch = collate_exact(selected)
            input_ids = batch["input_ids"].cuda(non_blocking=True)
            targets = batch["targets"].cuda(non_blocking=True)
            valid_mask = batch["valid_mask"].cuda(non_blocking=True)
            mask = batch["loss_mask"].cuda(non_blocking=True) > 0.5
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                result = model(input_ids, valid_mask, arm="learned", skip_head=True)
                fine = result.fine_states.reshape(-1, result.fine_states.size(-1))[mask.reshape(-1)]
                labels = targets.reshape(-1)[mask.reshape(-1)]
                logits = model.base.head(model.base.final_norm(fine))
                ce = F.cross_entropy(logits, labels, reduction="sum") / total_active
                if result.compression_ratio.numel() != len(selected):
                    raise ValueError("Per-example compression ratios are missing")
                ratio = 0.2 * (result.compression_ratio - 1.0).pow(2).sum() / BATCH_SIZE
                loss = ce + ratio
            if scaler is None and not bool(torch.isfinite(loss.detach())):
                raise FloatingPointError("Nonfinite hybrid loss; update rejected")
            if scaler is None:
                loss.backward()
            else:
                scaler.scale(loss).backward()
        if scaler is None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0,
                                           error_if_nonfinite=True)
            optimizer.step()
            return True
        return _finish_fp16_attempt(torch, model, optimizer, scaler)

    def restore_after_skip():
        with torch.no_grad():
            for name, buffer in model.named_buffers():
                buffer.copy_(buffer_before[name])
        _rng_restore(rng_before, torch, np)

    return _retry_same_slot(attempt=attempt, restore_after_skip=restore_after_skip,
                            can_continue=can_continue)


def _verify_original_swe_baseline(proof_dir: Path, *, expected_precision_profile="eval_bf16_native"):
    if expected_precision_profile == "eval_fp16_native_v1":
        raise ValueError("A separately pinned v43 SWE200 fp16 parent capture is required before fp16 training")
    if expected_precision_profile != "eval_bf16_native":
        raise ValueError("Unknown original SWE baseline precision")
    complete = json.loads((proof_dir / "COMPLETE.json").read_text(encoding="utf-8"))
    if (complete.get("status") != "COMPLETE" or complete.get("suite") != "heldout_swe200"
            or complete.get("cases") != 200 or complete.get("generation_limit") != 2048
            or complete.get("training_updates") != 0
            or complete.get("source_zip_sha256") != PINNED_V43_ZIP_SHA
            or complete.get("parent_sha256") != PARENT_CHECKPOINT_SHA
            or complete.get("eval_member_sha256") != PINNED_SWE_EVAL_SHA
            or complete.get("original_evaluator_sha256") != PINNED_SWE_EVALUATOR_SHA
            or complete.get("source_and_parent_unchanged") is not True
            or complete.get("eligible_old_skill_rows", 0) < 64):
        raise ValueError("Original v43 CPU-native full-answer baseline is missing or incomplete")
    names = ("swe200_full_answers.jsonl", "swe200_original_grades.jsonl",
             "original_swe_audit.jsonl", "v43_native_full_answers.jsonl",
             "v43_native_external_grades.jsonl", "v43_original_answers.jsonl",
             "old_skill_eval.jsonl")
    for name in names:
        if complete.get("files_sha256", {}).get(name) != file_sha(proof_dir / name):
            raise ValueError("Original v43 raw answer or grade changed: " + name)
    return file_sha(proof_dir / "COMPLETE.json")


def _verify_pretrain_raw_diagnostic(proof_dir: Path, *, expected_precision_profile="eval_bf16_native"):
    evaluator_source = Path(__file__).with_name("eval_fixed_diagnostic.py")
    cases_path = Path(__file__).with_name("fixed_interim_diagnostic_44_v2.jsonl")
    if file_sha(evaluator_source) != PINNED_INTERIM_EVALUATOR_SHA or file_sha(cases_path) != PINNED_INTERIM_CASES_SHA:
        raise ValueError("Pinned interim evaluator source or cases changed")
    complete = json.loads((proof_dir / "COMPLETE.json").read_text(encoding="utf-8"))
    raw_path = proof_dir / "raw_answers_44_profiles.jsonl"
    if (complete.get("schema") != "geocpu_v43_python_gradual_raw_diagnostic_v1"
            or complete.get("status") != "COMPLETE" or complete.get("arm") != "parent"
            or complete.get("checkpoint_kind") != "parent"
            or complete.get("checkpoint_sha256") != PARENT_CHECKPOINT_SHA
            or complete.get("v43_zip_sha256") != PINNED_V43_ZIP_SHA
            or complete.get("cases_sha256") != PINNED_INTERIM_CASES_SHA
            or complete.get("source_sha256") != PINNED_INTERIM_EVALUATOR_SHA
            or complete.get("precision_profile") != expected_precision_profile
            or complete.get("overlay_sha256") != sha256(json.dumps(SOURCE_OVERLAY_SHA, sort_keys=True).encode()).hexdigest()
            or complete.get("raw_answers_sha256") != file_sha(raw_path)
            or complete.get("profile_counts") != {"native":44, "python_cued":20, "python_uncued":20}
            or complete.get("max_generated_tokens") != 2048
            or complete.get("training_updates") != 0):
        raise ValueError("Same-parent pre-training raw diagnostic is absent or incomplete")
    cases = [json.loads(line) for line in cases_path.read_text(encoding="utf-8").splitlines()]
    expected = {(case["id"], profile): case for case in cases for profile in case["profiles"]}
    observed = set()
    with raw_path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            key = (row.get("case_id"), row.get("profile"))
            case = expected.get(key)
            if key in observed or case is None:
                raise ValueError("Pre-training raw answer ID/profile differs")
            observed.add(key)
            required_prompt = (PYTHON_MODE_CUE if key[1] == "python_cued" else "") + case["prompt"]
            if (row.get("arm") != "parent" or row.get("prompt") != required_prompt
                    or row.get("expected_answer") != case["target"]
                    or row.get("group") != case["group"]
                    or row.get("checkpoint_sha256") != PARENT_CHECKPOINT_SHA
                    or row.get("cases_sha256") != PINNED_INTERIM_CASES_SHA
                    or row.get("source_sha256") != PINNED_INTERIM_EVALUATOR_SHA
                    or row.get("precision_profile") != expected_precision_profile
                    or row.get("overlay_sha256") != sha256(json.dumps(SOURCE_OVERLAY_SHA, sort_keys=True).encode()).hexdigest()
                    or row.get("v43_zip_sha256") != PINNED_V43_ZIP_SHA
                    or not isinstance(row.get("raw_answer"), str)
                    or row.get("raw_answer_sha256") != sha256(row["raw_answer"].encode()).hexdigest()
                    or not isinstance(row.get("model_generated_text"), str)
                    or row.get("model_generated_text_sha256") != sha256(row["model_generated_text"].encode()).hexdigest()
                    or type(row.get("ended_with_eos")) is not bool):
                raise ValueError("Pre-training raw answer text or provenance differs")
    if observed != set(expected):
        raise ValueError("Pre-training raw diagnostic lacks a fixed case")
    return file_sha(proof_dir / "COMPLETE.json")


def _stage_target_breakdown(plan, pools, *, limit=None):
    counts = Counter()
    for batch in plan.batches[:limit]:
        for kind, index in batch:
            row = pools.python[index] if kind == "python" else pools.old[index]
            counts[kind + ":rows"] += 1
            counts[kind + ":active_positions"] += row.active_positions
            counts[kind + ":family:" + row.source_family + ":rows"] += 1
            counts[kind + ":family:" + row.source_family + ":active_positions"] += row.active_positions
            if row.japanese_target:
                counts[kind + ":japanese_character_target_rows"] += 1
                counts[kind + ":japanese_character_target_active_positions"] += row.active_positions
    return dict(counts)


def _needs_partial_save(accepted: int, skipped: int,
                        saved_cursor: tuple[int, int] | None) -> bool:
    return (accepted < FIRST_EVALUATION_BARRIER
            and saved_cursor != (accepted, skipped))


def run(args):
    if not Path("/kaggle").exists() or Path("/content/drive/MyDrive").exists():
        raise RuntimeError("Only a disposable private Kaggle runtime without mounted Drive is admitted")
    if args.stage != 1:
        # Stage advancement must verify old SWE/state/search/Japanese raw-answer
        # retention and executed Python on the same fixed heldout cases.  The
        # scorer/grade closure is not yet complete; fail rather than trust JSON.
        raise ValueError("Stage 2–4 promotion is blocked until full paired evaluation gate is implemented")
    if args.arm not in ("control", "python"):
        raise ValueError("Only the 0% control and 1% Python intervention are admitted")
    if args.output_dir.exists() and args.resume is None:
        raise FileExistsError("Use a new isolated output directory")
    if not 900 <= args.hard_stop_seconds <= 3600 or args.save_reserve_seconds < 120:
        raise ValueError("A bounded run and at least 120 seconds for saving are required")
    contract = verify_prelaunch_contract(args.prelaunch_contract, expected_arm=args.arm,
        required_run_seconds=args.hard_stop_seconds,
        required_save_reserve_seconds=args.save_reserve_seconds)
    if contract["precision_profile"] not in {"train_bf16_native", "train_fp16_gradscaler_v1"}:
        raise ValueError("A training claim must declare a training precision profile")
    absolute_stop = datetime.fromisoformat(contract["hard_stop_at_utc"].replace("Z", "+00:00")).astimezone(timezone.utc)
    if (absolute_stop - datetime.now(timezone.utc)).total_seconds() < args.save_reserve_seconds + 600:
        raise ValueError("Predeclared absolute stop leaves too little time for this stage")
    expected_eval_profile = ("eval_fp16_native_v1" if contract["precision_profile"] == "train_fp16_gradscaler_v1"
                             else "eval_bf16_native")
    baseline_sha = _verify_original_swe_baseline(args.baseline_proof_dir,
                                                  expected_precision_profile=expected_eval_profile)
    baseline_raw_sha = _verify_pretrain_raw_diagnostic(args.baseline_diagnostic_dir,
                                                        expected_precision_profile=expected_eval_profile)
    started = time.monotonic()
    pools = load_verified_pools(v43_zip=args.v43_zip, converted_jsonl=args.converted,
                                replay_receipt=args.replay_receipt)
    cap = 0.0 if args.arm == "control" else STAGE_PYTHON_CAPS[0]
    plan = plan_exposure(pools.old, pools.python, cap=cap,
                         updates=STEPS_PER_STAGE, batch_size=BATCH_SIZE, seed=args.seed)
    intervention_plan = plan if args.arm == "python" else plan_exposure(
        pools.old, pools.python, cap=STAGE_PYTHON_CAPS[0],
        updates=STEPS_PER_STAGE, batch_size=BATCH_SIZE, seed=args.seed)
    plan_json = plan.as_json()
    plan_sha = sha256(json.dumps(plan_json, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    boundary_plan_sha = sha256(json.dumps(intervention_plan.as_json(), sort_keys=True,
                                          separators=(",", ":")).encode()).hexdigest()
    if args.resume is None and (args.parent_checkpoint.stat().st_size != PARENT_CHECKPOINT_BYTES
                                or file_sha(args.parent_checkpoint) != PARENT_CHECKPOINT_SHA):
        raise ValueError("Pinned original v43 step-15000 checkpoint differs")
    parent_sha = PARENT_CHECKPOINT_SHA
    binding = _binding(stage=1, arm=args.arm, parent_sha=parent_sha, v43_zip=args.v43_zip,
                       converted=args.converted, replay_receipt=args.replay_receipt,
                       plan_sha=plan_sha, boundary_plan_sha=boundary_plan_sha,
                       source_root=args.overlay_source,
                       seed=args.seed, lr=args.lr, micro_tokens=args.max_micro_tokens,
                       resource_run_id=contract["run_id"],
                       precision_profile=contract["precision_profile"],
                       gpu_names=contract["gpu_names"])
    binding["original_swe200_baseline_complete_sha256"] = baseline_sha
    binding["parent_fixed44_raw_diagnostic_complete_sha256"] = baseline_raw_sha
    args.output_dir.mkdir(parents=True, exist_ok=args.resume is not None)
    claim_copy = args.output_dir / ("prelaunch_claim_" + contract["claim_id"] + ".json")
    if claim_copy.exists():
        raise ValueError("The same prelaunch claim has already been used in this run")
    with claim_copy.open("x", encoding="utf-8") as sink:
        json.dump({"schema":"geocpu_hybrid_prelaunch_claim_used_v1",
                   "claim_sha256":file_sha(args.prelaunch_contract),
                   "run_id":contract["run_id"], "arm":args.arm}, sink, sort_keys=True, indent=2)
        sink.write("\n")
    plan_path = args.output_dir / "admitted_actual_target_plan.json"
    if not plan_path.exists():
        with plan_path.open("x", encoding="utf-8") as sink:
            json.dump({"binding": binding, "pool_counts": pools.counts,
                       "planned_active_target_breakdown": _stage_target_breakdown(plan, pools),
                       "first_250_planned_active_target_breakdown":
                           _stage_target_breakdown(plan, pools, limit=FIRST_EVALUATION_BARRIER),
                       "plan": plan_json},
                      sink, ensure_ascii=False, sort_keys=True, indent=2)
            sink.write("\n")
    elif json.loads(plan_path.read_text(encoding="utf-8"))["plan"] != json.loads(json.dumps(plan_json)):
        raise ValueError("Saved exposure plan differs")
    scratch = args.output_dir / "isolated_original_source_plus_python"
    if args.resume is None:
        _prepare_source(args.v43_zip, args.overlay_source, scratch)
    elif not scratch.is_dir():
        raise FileNotFoundError("Resume source extraction is missing")
    else:
        _verify_extracted_source(args.v43_zip, args.overlay_source, scratch)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(scratch))
    import numpy as np
    import torch
    import torch.nn.functional as F
    from GeoCPU_CED_v2 import GeoCPU_CED_v2
    from GeoCPU_CED_v1 import CEDConfig
    from integrated_model_v44fix import IntegratedModelV44Fix
    from local_search_engine_v42 import LocalSearchEngineV42
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("A free Kaggle GPU is required; local CPU training is disabled")
    if torch.cuda.device_count() != contract["gpu_count"]:
        raise ValueError("Actual allocated GPU count differs from Root's free-resource hold")
    precision_profile = contract["precision_profile"]
    if precision_profile not in {"train_bf16_native", "train_fp16_gradscaler_v1"}:
        raise ValueError("This training branch requires a predeclared AMP profile")
    for device_index, expected_name in enumerate(contract["gpu_names"]):
        if torch.cuda.get_device_name(device_index) != expected_name:
            raise ValueError("Allocated GPU model differs from Root's predeclared claim")
        if precision_profile == "train_bf16_native":
            with torch.cuda.device(device_index):
                if (torch.cuda.get_device_capability(device_index)[0] < 8
                        or not torch.cuda.is_bf16_supported()):
                    raise ValueError("Allocated GPU lacks native bfloat16; no automatic fp16 fallback")
    autocast_dtype = torch.bfloat16 if precision_profile == "train_bf16_native" else torch.float16
    torch.backends.cuda.matmul.allow_tf32 = True
    metadata = pools.metadata
    config = dict(metadata["config"])
    config["scan_backend"] = "jit"
    base = GeoCPU_CED_v2(CEDConfig(**config))
    corpus = scratch / "data" / "knowledge_corpus_v43.jsonl"
    model = IntegratedModelV44Fix(base=base, search_engine=LocalSearchEngineV42(corpus_path=corpus))
    if args.resume is None:
        parent = torch.load(args.parent_checkpoint, map_location="cpu", weights_only=True)
        expected_saved_config = dict(metadata["config"])
        expected_saved_config["scan_backend"] = "jit"
        if (parent.get("step") != 15_000 or parent.get("meta") != metadata
                or parent.get("config") != expected_saved_config):
            raise ValueError("Parent checkpoint metadata/config differs from pinned original v43")
        model.load_state_dict(parent["model_state_dict"], strict=True)
    model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                                  weight_decay=0.01, fused=True)
    scaler = torch.amp.GradScaler("cuda") if precision_profile == "train_fp16_gradscaler_v1" else None
    accepted = old_active = py_active = fp16_skipped_attempts = 0
    if args.resume is not None:
        saved = _load_stage_checkpoint(args.resume, torch=torch, expected_binding=binding)
        model.load_state_dict(saved["model_state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        if scaler is not None:
            scaler.load_state_dict(saved["scaler_state_dict"])
        accepted = int(saved["accepted_updates"])
        old_active = int(saved["old_active_positions"])
        py_active = int(saved["python_active_positions"])
        fp16_skipped_attempts = int(saved["fp16_skipped_attempts"])
        if not 0 <= accepted < STEPS_PER_STAGE:
            raise ValueError("Resume cursor is outside this stage")
        if accepted >= FIRST_EVALUATION_BARRIER:
            raise ValueError("At 250 updates both arms require fixed paired raw-answer evaluation before continuation")
        expected_old = expected_py = 0
        for batch in plan.batches[:accepted]:
            for kind, index in batch:
                if kind == "python":
                    expected_py += pools.python[index].active_positions
                else:
                    expected_old += pools.old[index].active_positions
        if (old_active, py_active) != (expected_old, expected_py):
            raise ValueError("Resume active target counts differ from the fixed schedule")
        _rng_restore(saved["rng"], torch, np)
    model.train()
    saved_cursor = ((accepted, fp16_skipped_attempts) if args.resume is not None
                    else None)
    last_receipt = (json.loads(args.resume.with_suffix(args.resume.suffix + ".receipt.json")
                               .read_text(encoding="utf-8")) if args.resume is not None
                    else None)
    for update in range(accepted, STEPS_PER_STAGE):
        if update >= FIRST_EVALUATION_BARRIER:
            break
        if (time.monotonic() - started >= args.hard_stop_seconds - args.save_reserve_seconds
                or (absolute_stop - datetime.now(timezone.utc)).total_seconds() <= args.save_reserve_seconds):
            break
        rows = [pools.python[index] if kind == "python" else pools.old[index]
                for kind, index in plan.batches[update]]
        envelope_lengths = [max(len(pools.old[old_index].tokens),
                                len((pools.python[index] if kind == "python" else pools.old[index]).tokens))
                            for old_index, (kind, index) in zip(plan.paired_old_batches[update],
                                                                 intervention_plan.batches[update])]
        micro = _microbatches(rows, envelope_lengths, args.max_micro_tokens)
        packed_all = collate_exact(rows)
        live_old, live_py = verify_realized_update(plan, update, rows, packed_all)
        total_active = live_old + live_py
        stepped, skipped = _run_one_update(rows=rows, micro=micro,
            total_active=total_active, model=model, optimizer=optimizer, scaler=scaler,
            autocast_dtype=autocast_dtype, torch=torch, np=np, F=F,
            can_continue=lambda: (
                time.monotonic() - started < args.hard_stop_seconds - args.save_reserve_seconds
                and (absolute_stop - datetime.now(timezone.utc)).total_seconds() > args.save_reserve_seconds))
        fp16_skipped_attempts += skipped
        if not stepped:
            break
        accepted = update + 1
        old_active += live_old
        py_active += live_py
        if py_active / (old_active + py_active) > cap + 1e-12:
            raise AssertionError("Live Python supervised target fraction exceeds its stage cap")
        if accepted % CHECKPOINT_EVERY == 0 or accepted == STEPS_PER_STAGE:
            name = f"hybrid_stage1_{args.arm}_update_{accepted:04d}.pt"
            last_receipt = _save_checkpoint(args.output_dir / name, model=model,
                optimizer=optimizer, torch=torch, np=np, binding=binding,
                update=accepted, old_active=old_active, python_active=py_active,
                status="BLOCK1_NEEDS_PAIRED_EVALUATION" if accepted == FIRST_EVALUATION_BARRIER else "PARTIAL",
                scaler=scaler, fp16_skipped_attempts=fp16_skipped_attempts)
            saved_cursor = (accepted, fp16_skipped_attempts)
            print(json.dumps({"accepted_updates": accepted,
                              "old_active_positions": old_active,
                              "python_active_positions": py_active,
                              "measured_python_fraction": py_active/(old_active+py_active),
                              "checkpoint_sha256": last_receipt["sha256"]}), flush=True)
    if _needs_partial_save(accepted, fp16_skipped_attempts, saved_cursor):
        # This also seals a scale-down after an earlier periodic checkpoint.
        # A fresh claim permits an exact checkpoint at zero accepted updates.
        name = (f"hybrid_stage1_{args.arm}_update_{accepted:04d}"
                f"_claim_{contract['claim_id']}.pt")
        last_receipt = _save_checkpoint(args.output_dir / name,
            model=model, optimizer=optimizer, torch=torch, np=np, binding=binding,
            update=accepted, old_active=old_active, python_active=py_active,
            status="PARTIAL", scaler=scaler,
            fp16_skipped_attempts=fp16_skipped_attempts)
    return {"arm": args.arm, "accepted_updates": accepted,
            "stage_complete": accepted == STEPS_PER_STAGE,
            "first_evaluation_barrier_reached": accepted == FIRST_EVALUATION_BARRIER,
            "checkpoint_receipt": last_receipt,
            "fp16_skipped_attempts": fp16_skipped_attempts,
            "measured_python_fraction": py_active/(old_active+py_active) if old_active+py_active else 0.0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=(1, 2, 3, 4), required=True)
    parser.add_argument("--arm", choices=("control", "python"), required=True)
    for name in ("v43_zip", "converted", "replay_receipt", "parent_checkpoint",
                 "overlay_source", "output_dir", "baseline_proof_dir", "baseline_diagnostic_dir",
                 "prelaunch_contract"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--lr", type=float, default=2.5e-5)
    parser.add_argument("--max-micro-tokens", type=int, default=DEFAULT_MAX_MICRO_TOKENS)
    parser.add_argument("--hard-stop-seconds", type=int, default=1800)
    parser.add_argument("--save-reserve-seconds", type=int, default=180)
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
