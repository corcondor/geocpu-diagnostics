"""One bounded GPU stage on verified data, starting from pinned v43 weights.

This script intentionally does not submit a job or run with the current
uncertified 101-row conversion sample. Every stage produces an unpromoted
adapter checkpoint. A later stage requires an independent retention gate.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from adapter import (AdapterSpec, adapter_parameters, file_sha256, install_adapters,
                     resume_adapter_checkpoint, save_adapter_checkpoint)
from admission import (PINNED_V43_META_SHA256, require_independent_execution_verification,
                       require_v43_native_baseline, verify_dataset,
                       verify_source_manifest)


PARENT_SHA256 = "a4b95d7d0f486146196271b3b1fbc6a1b6b3342d8fadd590320fc3842c1e80e5"
PARENT_BYTES = 990_545_719
PARENT_STORAGE_COUNT = 563
SOURCE_MODULES = ("GeoCPU_CED_v1", "GeoCPU_CED_v2", "neural_model_v30_python_base",
                  "integrated_model_v44_python", "tir_engine_v44r1", "training_data_v44",
                  "checkpoint_recovery_v44r1")
MAX_RELATIVE_ADAPTER_FROBENIUS = 0.01  # fixed before training; never relaxed using dev results


def _relative_adapter_norms(model, names, parent_norms):
    """Compute ||(alpha/r) B A||_F / ||W_parent||_F without forming B A."""
    import torch
    ratios = {}
    with torch.no_grad():
        for name in names:
            module = model.get_submodule(name)
            a = module.down.weight.float()
            b = module.up.weight.float()
            gram_a = a @ a.T
            gram_b = b.T @ b
            delta_sq = (gram_a * gram_b).sum().clamp_min(0)
            ratio = float((abs(module.scale) * torch.sqrt(delta_sq) / parent_norms[name]).item())
            ratios[name] = ratio
    return ratios


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    for name in ("source_dir", "source_manifest_sha256", "parent", "meta", "meta_sha256",
                 "data_dir", "data_manifest_sha256", "output_dir"):
        parser.add_argument("--" + name.replace("_", "-"), required=True)
    parser.add_argument("--admit-only", action="store_true")
    parser.add_argument("--updates", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--token-budget", type=int, default=36_864)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--resume-sha256")
    parser.add_argument("--previous-gate")
    parser.add_argument("--previous-gate-sha256")
    parser.add_argument("--previous-evidence-map")
    parser.add_argument("--previous-evidence-map-sha256")
    parser.add_argument("--verification-proof-dir")
    parser.add_argument("--verification-proof-sha256")
    parser.add_argument("--v43-native-proof-dir")
    parser.add_argument("--v43-native-capture-sha256")
    parser.add_argument("--v43-native-grade-sha256")
    parser.add_argument("--hard-stop-utc", required=True,
                        help="UTC training deadline; saving begins at least 180 s earlier")
    parser.add_argument("--save-every", type=int, default=25)
    return parser.parse_args(argv)


def _source_imports(source_dir):
    if any(name in sys.modules for name in SOURCE_MODULES):
        raise RuntimeError("Stale pure-v44 source modules; use a fresh Python process")
    sys.path.insert(0, str(Path(source_dir).resolve()))
    from GeoCPU_CED_v1 import CEDConfig
    from GeoCPU_CED_v2 import GeoCPU_CED_v2
    from integrated_model_v44_python import IntegratedModelV44Python, load_v43_neural_weights
    from checkpoint_recovery_v44r1 import recover_checkpoint
    from training_data_v44 import StrictDataset, batches_for_lengths, collate
    return CEDConfig, GeoCPU_CED_v2, IntegratedModelV44Python, load_v43_neural_weights, recover_checkpoint, StrictDataset, batches_for_lengths, collate


def _verify_previous_gate(path, expected_gate_sha256, checkpoint_sha256, data_manifest_sha256,
                          evidence_map_path, evidence_map_sha256):
    if path is None or not expected_gate_sha256 or file_sha256(path) != expected_gate_sha256:
        raise ValueError("A completed retention gate is required before resuming another stage")
    proof = json.loads(Path(path).read_text(encoding="utf-8"))
    if (proof.get("schema") != "geocpu_bounded_adapter_retention_gate_v1"
            or proof.get("status") != "PASS"
            or proof.get("scope") != "NEXT_BOUNDED_STAGE_ONLY"
            or proof.get("public_adoption_approved") is not False
            or proof.get("tier_1_to_8_full_evaluation_status") != "NOT_RUN"
            or proof.get("checkpoint_sha256") != checkpoint_sha256
            or proof.get("data_manifest_sha256") != data_manifest_sha256
            or proof.get("old_skill_regressions") != 0
            or proof.get("checks", {}).get("old_v43_natural_eos_retained_on_every_item") is not True
            or proof.get("checks", {}).get("old_v43_cpu_tasks_semantically_correct_with_actual_python") is not True
            or type(proof.get("unseen_evaluated")) is not int
            or proof["unseen_evaluated"] < 64
            or proof.get("raw_answer_sha256") is None):
        raise ValueError("Previous adapter stage has no complete passing retention gate")
    from retention_gate import build_gate
    gate_source = Path(__file__).with_name("retention_gate.py")
    if proof.get("retention_gate_source_sha256") != file_sha256(gate_source):
        raise ValueError("Previous retention gate code differs from the recorded evaluator")
    inputs = proof.get("evidence_inputs")
    if not isinstance(inputs, dict):
        raise ValueError("Previous retention gate lacks pinned underlying evidence")
    if (not evidence_map_path or not evidence_map_sha256
            or file_sha256(evidence_map_path) != evidence_map_sha256):
        raise ValueError("A pinned relocation map for the previous raw evidence is required")
    moved = json.loads(Path(evidence_map_path).read_text(encoding="utf-8"))
    path_keys = {"data_dir", "paired_dir", "packet_dir", "grades_path", "grade_pin_path",
                 "verification_proof_dir", "v43_native_proof_dir"}
    if (not isinstance(moved, dict) or set(moved) != path_keys
            or any(not isinstance(moved[key], str) or not Path(moved[key]).exists()
                   for key in path_keys)):
        raise ValueError("Previous evidence relocation map is incomplete")
    rebuilt = build_gate(**{**inputs, **moved})
    # Only the verified transport paths may differ between Kaggle runtimes.
    rebuilt["evidence_inputs"] = inputs
    if rebuilt != proof:
        raise ValueError("Previous retention gate cannot be reproduced from its raw evidence")


def _write_new_json(path: Path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()


def _actual_model_path_gate(model, names, dataset):
    """Run the real integrated model on both training and native prefill paths.

    All selected residual modules must execute. Perturbing the output-head
    residual must change the same next-token logits in both paths. This is
    an admission check, not a claim about answer quality or all 8192 positions.
    """
    import torch
    token_ids, _ = dataset[0]
    ids = torch.tensor([list(token_ids[:min(64, len(token_ids)-1)])],
                       dtype=torch.long, device="cuda")
    if ids.shape[1] < 2:
        raise ValueError("Actual-model path gate needs at least two tokens")
    valid = torch.ones_like(ids, dtype=torch.bool)
    hits = {name: 0 for name in names}
    handles = []
    for name in names:
        handles.append(model.get_submodule(name).register_forward_hook(
            lambda _module, _inputs, _output, key=name: hits.__setitem__(key, hits[key] + 1)))
    old_mode = model.training
    old_cpu_rng = torch.get_rng_state()
    old_cuda_rng = torch.cuda.get_rng_state_all()
    head = model.get_submodule("base.head")
    old_down = head.down.weight.detach().clone()
    old_up = head.up.weight.detach().clone()
    try:
        model.eval()
        with torch.no_grad():
            def both():
                forward = model(ids, valid, arm="learned", skip_head=True)
                hidden = model.base.final_norm(forward.fine_states[:, -1, :])
                train_logits = model.base.head(hidden)[0]
                native_logits = model.prefill(ids, arm="learned").last_logits.reshape(-1,
                    train_logits.numel())[-1]
                return train_logits.detach().clone(), native_logits.detach().clone(), hidden.detach().clone()
            before_train, before_native, hidden = both()
            if torch.max(torch.abs(before_train.float() - before_native.float())).item() > 5e-4:
                raise RuntimeError("Real training and native prefill next-token logits differ")
            head.down.weight.zero_()
            head.up.weight.zero_()
            head.down.weight[0].copy_(hidden[0].float().to(head.down.weight.dtype))
            head.up.weight[0, 0] = 0.1
            after_train, after_native, _ = both()
            if (torch.max(torch.abs(after_train.float() - before_train.float())).item() <= 1e-6
                    or torch.max(torch.abs(after_native.float() - before_native.float())).item() <= 1e-6
                    or torch.max(torch.abs(after_train.float() - after_native.float())).item() > 5e-4):
                raise RuntimeError("Head adapter is absent from a real training or generation path")
            if any(value < 2 for value in hits.values()):
                raise RuntimeError("One selected adapter was bypassed by the real model")
            return {"status": "PASS", "observed_adapter_modules": hits,
                    "baseline_train_native_max_abs": float(torch.max(torch.abs(before_train.float()-before_native.float()))),
                    "perturbed_train_delta_max_abs": float(torch.max(torch.abs(after_train.float()-before_train.float()))),
                    "perturbed_native_delta_max_abs": float(torch.max(torch.abs(after_native.float()-before_native.float())))}
    finally:
        with torch.no_grad():
            head.down.weight.copy_(old_down)
            head.up.weight.copy_(old_up)
        model.train(old_mode)
        torch.set_rng_state(old_cpu_rng)
        torch.cuda.set_rng_state_all(old_cuda_rng)
        for handle in handles:
            handle.remove()


def main(argv=None):
    args = parse_args(argv)
    try:
        hard_stop = datetime.fromisoformat(args.hard_stop_utc.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Invalid hard-stop UTC time") from exc
    if hard_stop.tzinfo is None or hard_stop.utcoffset() != timedelta(0):
        raise ValueError("Hard-stop must explicitly be in UTC")
    if not 1 <= args.save_every <= 25:
        raise ValueError("Save cadence must be 1..25 updates")
    source = verify_source_manifest(args.source_dir, args.source_manifest_sha256)
    data = verify_dataset(args.data_dir, args.data_manifest_sha256)
    if file_sha256(args.parent) != PARENT_SHA256:
        raise ValueError("Original v43 checkpoint SHA differs")
    if args.meta_sha256 != PINNED_V43_META_SHA256 or file_sha256(args.meta) != args.meta_sha256:
        raise ValueError("Metadata SHA differs")
    if args.resume_checkpoint is not None:
        if not args.resume_sha256 or file_sha256(args.resume_checkpoint) != args.resume_sha256:
            raise ValueError("Resume adapter SHA differs")
        _verify_previous_gate(args.previous_gate, args.previous_gate_sha256,
                              args.resume_sha256, args.data_manifest_sha256,
                              args.previous_evidence_map, args.previous_evidence_map_sha256)
    elif any((args.resume_sha256, args.previous_gate, args.previous_gate_sha256,
              args.previous_evidence_map, args.previous_evidence_map_sha256)):
        raise ValueError("Resume arguments must be supplied together")
    admission = {"status": "PASS", "source": source, "data": data,
                 "parent_sha256": PARENT_SHA256, "meta_sha256": args.meta_sha256,
                 "admit_only": args.admit_only}
    if args.admit_only:
        print(json.dumps(admission, ensure_ascii=False, sort_keys=True))
        return admission
    # The structural admission above is not enough to prove that every tool
    # observation and every CPU-to-Python conversion was actually executed.
    # Keep this before CUDA import, output creation or parent checkpoint load.
    verification = require_independent_execution_verification(
        data, data_dir=args.data_dir, proof_dir=args.verification_proof_dir,
        expected_proof_sha256=args.verification_proof_sha256,
        expected_meta_sha256=args.meta_sha256,
        expected_source_manifest_sha256=args.source_manifest_sha256)
    v43_native = require_v43_native_baseline(
        args.data_dir, proof_dir=args.v43_native_proof_dir,
        expected_capture_sha256=args.v43_native_capture_sha256,
        expected_grade_sha256=args.v43_native_grade_sha256)
    if hard_stop <= datetime.now(timezone.utc) + timedelta(minutes=5):
        raise ValueError("Hard-stop must allow at least five minutes from admission")
    if args.updates < 1 or args.updates > 250 or args.lr is None or not (0 < args.lr <= 1e-3):
        raise ValueError("One stage requires 1..250 updates and an explicit bounded learning rate")
    if not 1 <= args.batch_size <= 64 or not 2048 <= args.token_budget <= 36_864:
        raise ValueError("Batch size or token budget outside declared limits")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("GPU stage requires an existing CUDA device")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    _write_new_json(out / "stage_claim.json", {"status": "RUNNING", "created_utc": datetime.now(timezone.utc).isoformat(),
                                               "source": source, "data": data, "parent_sha256": PARENT_SHA256,
                                               "independent_verification": verification,
                                               "v43_native_baseline": v43_native,
                                               "updates_requested": args.updates, "lr": args.lr,
                                               "batch_size": args.batch_size, "token_budget": args.token_budget})
    (CEDConfig, GeoCPU_CED_v2, IntegratedModelV44Python, load_v43_neural_weights,
     recover_checkpoint, StrictDataset, batches_for_lengths, collate) = _source_imports(args.source_dir)
    # Seed before constructing the 247M parent and zero-B/random-A adapters.
    # A resumed stage later restores its saved RNG and adapter tensors exactly.
    random.seed(20261103)
    torch.manual_seed(20261103)
    torch.cuda.manual_seed_all(20261103)
    meta = json.loads(Path(args.meta).read_text(encoding="utf-8"))
    config = dict(meta["config"])
    config["scan_backend"] = "jit"
    base = GeoCPU_CED_v2(CEDConfig(**config))
    model = IntegratedModelV44Python(base, search_engine=None)
    recovered = recover_checkpoint(args.parent, expected_sha256=PARENT_SHA256,
                                   staging_dir=str(out / "parent_staging"), expected_step=15_000,
                                   expected_bytes=PARENT_BYTES, expected_storage_count=PARENT_STORAGE_COUNT)
    migration = load_v43_neural_weights(model, recovered)
    del recovered
    spec = AdapterSpec(rank=args.rank, alpha=args.alpha)
    names = install_adapters(model, spec)
    model.to("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    optimizer = torch.optim.AdamW(adapter_parameters(model, names), lr=args.lr,
                                  betas=(0.9, 0.95), weight_decay=0.01, fused=True)
    dataset = StrictDataset(Path(args.data_dir) / "train.jsonl", meta["vocab"], config["max_positions"])
    path_gate = _actual_model_path_gate(model, names, dataset)
    _write_new_json(out / "actual_model_path_gate.json", path_gate)
    epoch = 0
    cursor = 0
    step = 0
    if args.resume_checkpoint:
        restored = resume_adapter_checkpoint(args.resume_checkpoint, expected_sha256=args.resume_sha256,
                                             expected_parent_sha256=PARENT_SHA256,
                                             expected_train_manifest_sha256=args.data_manifest_sha256,
                                             model=model, names=names, spec=spec, optimizer=optimizer)
        state = restored["rng"]
        epoch, cursor, step = state["epoch"], state["batch_cursor"], restored["step"]
        random.setstate(state["python"])
        torch.set_rng_state(state["torch_cpu"])
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    parent_norms = {name: model.get_submodule(name).parent.weight.detach().float().norm().clamp_min(1e-12)
                    for name in names}
    drift = _relative_adapter_norms(model, names, parent_norms)
    if any(not math.isfinite(value) or value > MAX_RELATIVE_ADAPTER_FROBENIUS for value in drift.values()):
        raise RuntimeError("Loaded adapter exceeds the predeclared parent-relative drift cap")
    model.train()
    original_step = step
    batches = batches_for_lengths(dataset.lengths, args.batch_size, args.token_budget, seed=20261103 + epoch)
    saved = None
    save_reserve = timedelta(seconds=180)
    guard_rejected = False
    for _ in range(args.updates):
        if datetime.now(timezone.utc) >= hard_stop - save_reserve:
            break
        if cursor >= len(batches):
            epoch += 1
            cursor = 0
            batches = batches_for_lengths(dataset.lengths, args.batch_size, args.token_budget, seed=20261103 + epoch)
        batch = collate([dataset[i] for i in batches[cursor]])
        cursor += 1
        qa_rng = {"python": random.getstate(), "torch_cpu": torch.get_rng_state(),
                  "torch_cuda": torch.cuda.get_rng_state_all()}
        x = batch["input_ids"].to("cuda", non_blocking=True)
        y = batch["targets"].to("cuda", non_blocking=True)
        valid = batch["valid_mask"].to("cuda", non_blocking=True)
        mask = batch["loss_mask"].to("cuda", non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = model(x, valid, arm="learned", skip_head=True)
            logits = model.base.head(model.base.final_norm(prediction.fine_states[mask]))
            ce = torch.nn.functional.cross_entropy(logits, y[mask])
            ratio = (prediction.compression_ratio - 1.0).square().mean()
            loss = ce + 0.2 * ratio
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter_parameters(model, names), 1.0, error_if_nonfinite=True)
        params = adapter_parameters(model, names)
        before_params = [parameter.detach().clone() for parameter in params]
        before_optimizer = copy.deepcopy(optimizer.state_dict())
        optimizer.step()
        drift = _relative_adapter_norms(model, names, parent_norms)
        if any(not math.isfinite(value) or value > MAX_RELATIVE_ADAPTER_FROBENIUS for value in drift.values()):
            with torch.no_grad():
                for parameter, saved_parameter in zip(params, before_params, strict=True):
                    parameter.copy_(saved_parameter)
            optimizer.load_state_dict(before_optimizer)
            random.setstate(qa_rng["python"])
            torch.set_rng_state(qa_rng["torch_cpu"])
            torch.cuda.set_rng_state_all(qa_rng["torch_cuda"])
            cursor -= 1
            guard_rejected = True
            break
        step += 1
        if (step - original_step) % args.save_every == 0:
            torch.cuda.synchronize()
            rng = {"epoch": epoch, "batch_cursor": cursor, "python": random.getstate(),
                   "torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state_all()}
            path = out / f"adapter_step_{step}.pt"
            sha = save_adapter_checkpoint(path, model=model, names=names, spec=spec,
                                          parent_sha256=PARENT_SHA256,
                                          train_manifest_sha256=args.data_manifest_sha256,
                                          optimizer=optimizer, step=step, rng=rng)
            saved = (step, path, sha)
            _write_new_json(out / f"progress_step_{step}.json",
                            {"step": step, "checkpoint": str(path), "checkpoint_sha256": sha,
                             "hard_stop_utc": hard_stop.isoformat(), "status": "SAVED_UNPROMOTED"})
        if step % 25 == 0:
            print(json.dumps({"step": step, "teacher_forced_ce": float(ce.detach()),
                              "loss": float(loss.detach())}), flush=True)
    if saved is None or saved[0] != step:
        torch.cuda.synchronize()
        rng = {"epoch": epoch, "batch_cursor": cursor, "python": random.getstate(),
               "torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state_all()}
        path = out / f"adapter_step_{step}.pt"
        sha = save_adapter_checkpoint(path, model=model, names=names, spec=spec,
                                      parent_sha256=PARENT_SHA256,
                                      train_manifest_sha256=args.data_manifest_sha256,
                                      optimizer=optimizer, step=step, rng=rng)
        saved = (step, path, sha)
    _, path, sha = saved
    result = {"status": "STOPPED_DRIFT_GUARD_UNPROMOTED" if guard_rejected else
              "SAVED_UNPROMOTED" if step-original_step == args.updates else "STOPPED_DEADLINE_UNPROMOTED",
              "old_step": original_step, "step": step,
              "new_updates": step-original_step, "adapter_checkpoint": str(path),
              "adapter_sha256": sha, "parent_sha256_after": file_sha256(args.parent),
              "hard_stop_utc": hard_stop.isoformat(), "save_every_updates": args.save_every,
              "max_parent_relative_adapter_frobenius": MAX_RELATIVE_ADAPTER_FROBENIUS,
              "observed_parent_relative_adapter_frobenius": _relative_adapter_norms(model, names, parent_norms),
              "source_manifest_sha256": args.source_manifest_sha256,
              "data_manifest_sha256": args.data_manifest_sha256,
              "independent_verification": verification,
              "v43_native_baseline": v43_native,
              "migration": {key: migration[key] for key in ("status", "neural_tensor_count", "neural_parameter_count", "all_fp32_neural_bytes_identical")},
              "actual_model_path_gate": path_gate,
              "target_modules": names, "parent_checkpoint_modified": False,
              "evaluation_status": "NOT_STARTED", "promotion_status": "BLOCKED_PENDING_PAIRED_RAW_EVALUATION"}
    if result["parent_sha256_after"] != PARENT_SHA256:
        raise RuntimeError("Parent checkpoint changed unexpectedly")
    _write_new_json(out / "stage_result.json", result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return result


if __name__ == "__main__":
    main()
