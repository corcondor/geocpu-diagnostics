"""Prepare/train one new GeoCPU branch from the verified saved step 30,500.

Never modifies a source checkpoint, source training data or running process.
The default is a 1,000-update pilot. --preflight-only does CPU preparation and
deterministic sampling checks, with zero model updates. Training uses a frozen
copy of the parent as the replay teacher, then leaves evaluation pending.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

from prepare_training import (NEW_GROUP, ORIGINAL_RUNTIME_SHA, PARENT_ROWS,
                              PARENT_TRAIN_SHA, VARIANTS, manifest_train,
                              prepare_dataset, preview_plan, require_sha, sha,
                              write_runtime)
from live_logging import run_with_logging


HERE = Path(__file__).resolve().parent
STUDY = Path("/content/drive/MyDrive/GeoCPU_v47_fullmodel_20261106")
PARENT_RUN = "v47_from28000_to30500_curriculum_v1"
PARENT_STEP = 30500
CONTINUE_ZIP = "GeoCPU_v47_continue30500_train_eval_v1.zip"
CONTINUE_ZIP_SHA = "bba78c81b763608b70691395c39bc01996f7c483f0c4bcc7f1fa7b8269e079c5"
PARENT_ARTIFACTS = {
    "checkpoint": "cc38411e66aab0ed309d0fa6e0b268d04317da17ffa47f730383e6ad0e2d0391",
    "optimizer": "9a3b5dce54c8a9acdfe94fdd4de2d5c560911b1875f843ec55dae7964bf47a51",
    "state": "f000f933463d5b1ea6924d1a390e5d5c233e7fa4447152d4d0fee36bd804d7bf",
}


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def new_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load verified preparation source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def safe_unpack(path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            if not (destination / member.filename).resolve().is_relative_to(destination.resolve()):
                raise ValueError("unsafe ZIP path")
        archive.extractall(destination)


def checked_parent(study: Path, run: str, step: int) -> tuple[Path, dict, dict]:
    if step != PARENT_STEP or re.fullmatch(r"[A-Za-z0-9_-]+", run) is None:
        raise ValueError("this reviewed candidate supports the pinned saved step 30500 only")
    parent = study / run
    receipt_path = parent / "checkpoints/receipt_step_30500.json"
    receipt = read_json(receipt_path)
    saved = read_json(parent / "training_saved.json")
    manifest = read_json(parent / "input_manifest.json")
    fork = read_json(parent / "fork_from_28000.json")
    if (receipt.get("step") != 30500 or saved.get("step") != 30500 or not saved.get("full_model") or
            receipt.get("training_data_sha256") != PARENT_TRAIN_SHA or
            saved.get("training_data_sha256") != PARENT_TRAIN_SHA or
            manifest.get("output_sha256") != PARENT_TRAIN_SHA or manifest.get("output_rows") != PARENT_ROWS or
            receipt.get("runtime_sha256") != ORIGINAL_RUNTIME_SHA or receipt.get("branch_from") != 28000 or
            fork.get("branch_stop") != 30500 or fork.get("new_training_data_sha256") != PARENT_TRAIN_SHA or
            fork.get("source_receipt_sha256") != receipt.get("source_receipt_sha256") or
            fork.get("source_receipt", {}).get("checkpoint_sha256") != receipt.get("parent_sha256")):
        raise RuntimeError("saved parent metadata is not the reviewed 30,500 branch")
    for kind, expected in PARENT_ARTIFACTS.items():
        name = "train_state" if kind == "state" else kind
        path = parent / f"checkpoints/{name}_step_30500.pt"
        if receipt.get(kind + "_sha256") != expected or not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"missing or unbound parent artifact: {path}")
    return parent, receipt, {"receipt_sha256": sha(receipt_path),
                            "input_manifest_sha256": sha(parent / "input_manifest.json"),
                            "training_saved_sha256": sha(parent / "training_saved.json"),
                            "fork_sha256": sha(parent / "fork_from_28000.json")}


def active_work(proc_root: Path = Path("/proc")) -> list[dict]:
    """Observe other Python training/evaluation processes; never signal them."""
    markers = ("runtime_practical_v1.py", "runtime_30500.py", "runtime_full_v46",
               "runtime_v46_eval.py", "runtime_v44b.py", "train_candidate.py",
               "runtime_cotraining", "runtime_v44fix.py", "eval_candidate.py",
               "launch_python_science_next.py", "colab_v47_continue",
               "launch_30500.py", "launch_suite_30500.py", "launch_suite.py",
               "stage_official_tier_v47.py", "eval_python_tir_", "run_extended",
               "launch_colab_python_tir_", "colab_v47_fullmodel.py")
    found = []
    if not proc_root.is_dir():
        return found
    for entry in proc_root.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            parts = [part.decode("utf-8", errors="replace") for part in
                     (entry / "cmdline").read_bytes().split(b"\0") if part]
        except OSError:
            continue
        if not parts or "python" not in Path(parts[0]).name.lower():
            continue
        scripts = [Path(arg).name for arg in parts[1:] if arg.endswith(".py")]
        for script in scripts:
            if any(marker in script for marker in markers):
                found.append({"pid": int(entry.name), "script": script})
                break
    return found


def refuse_active_work(study: Path) -> None:
    active = active_work()
    # Drive lock files can represent work on another Colab host. Do not remove
    # them automatically, even if the process is not present in this /proc.
    locks = []
    for pattern in ("*/suite_active.json", "*/evaluation_active.json", "*/training_active.json"):
        locks.extend(str(path) for path in study.glob(pattern))
    if active or locks:
        raise RuntimeError("another training/evaluation is active or retains its lock; this launch has not "
                           "stopped it. Watch that run, then start a new branch after it finishes: " +
                           json.dumps({"processes": active, "locks": locks}, ensure_ascii=False))


def fixed_heldouts(package: Path) -> list[Path]:
    paths = list((package / "frozen/base/data").glob("*dev*.jsonl"))
    paths += list((package / "frozen/r6/data").glob("eval_*.jsonl"))
    paths += [package / "frozen/r6/data/jamcqa_manifest_test.jsonl",
              package / "frozen/r6/data/benchmark_8100_manifest.jsonl",
              package / "frozen/public/data/dev_overlay.jsonl",
              package / "frozen/continue/eval_python/selection_120.json",
              package / "frozen/continue/engineering/engineering_heldout.jsonl",
              package / "frozen/continue/visual/visual_heldout.jsonl",
              package / "frozen/searchprobe/dev_after_final_result_198.jsonl",
              package / "data/practical_dev.jsonl",
              package / "data/practical_expanded_eval.jsonl",
              package / "data/knowledge_dev.jsonl",
              package / "data/knowledge_free_dev.jsonl"]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"heldout inputs are missing: {missing}")
    return sorted(set(paths))


def support_sources(study: Path, scratch: Path):
    archive = study / CONTINUE_ZIP
    require_sha(archive, CONTINUE_ZIP_SHA)
    original = scratch / "original_30500"
    safe_unpack(archive, original)
    launch = load_module(original / "training/launch_30500.py", "verified_30500_preparer")
    old, support = launch.load_support(study, scratch)
    outer = old.prepare_original_bundle(scratch)
    runner = old.load_pinned_old_runner(scratch)
    return original, launch, old, support, outer, runner


def find_30500_cache(explicit: Path | None) -> Path | None:
    if explicit is not None:
        require_sha(explicit, PARENT_TRAIN_SHA)
        return explicit
    candidates = list(Path(tempfile.gettempdir()).glob("geocpu_30500_*/train_30500.jsonl"))
    candidates += list(Path(tempfile.gettempdir()).glob("geocpu_practical_pilot_*/train_30500.jsonl"))
    for path in sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True):
        if sha(path) == PARENT_TRAIN_SHA:
            return path
    return None


def make_parent_dataset(study: Path, scratch: Path, prepared_parent: Path | None):
    original, launch, old, support, outer, runner = support_sources(study, scratch)
    cached = find_30500_cache(prepared_parent)
    if cached is not None:
        source = old.prepare_private_source(scratch, outer)
        public_dev = launch.public_dev_from_outer(outer, scratch)
        corpus = scratch / "retrieval/combined_train_wikidata_first.jsonl"
        corpus.parent.mkdir()
        with zipfile.ZipFile(outer / "assets/r6_bundle.zip") as r6:
            with zipfile.ZipFile(r6.open("GeoCPU_public_search_science_20261105_v4.zip")) as public:
                corpus.write_bytes(public.read("retrieval/combined_train_wikidata_first.jsonl"))
        require_sha(corpus, old.CORPUS_SHA)
        return cached, source, corpus, original, old, runner
    # Reuse the exact historical composition, including its checks. This may
    # read the original v43/bridge inputs when the Colab temporary cache expired.
    launch.checked_28000_parent(study, old)
    parent28000, _, corpus, public_dev, source = launch.rebuild_28000_dataset(old, support, scratch, outer, runner)
    data = original / "curriculum/new_curriculum.jsonl"
    n, traces, source_sha = launch.validate_new_manifest(original / "curriculum/new_manifest.json", data)
    if (n, traces, source_sha) != (372, 72, "0e4f626d0b4d4ac491b4074d3a18def29636788a23b6aba84e4d5dcbff53246e"):
        raise RuntimeError("historical 30500 curriculum changed")
    destination = scratch / "train_30500.jsonl"
    manifest = scratch / "recomposed_30500_manifest.json"
    command = [sys.executable, "-u", str(original / "training/prepare_30500.py"),
               "--parent", str(parent28000), "--parent-sha256", launch.PARENT_TRAIN_SHA,
               "--parent-rows", str(launch.PARENT_TRAIN_ROWS), "--new", str(data), "--new-sha256", source_sha,
               "--meta", str(source / "base_metadata.json"),
               "--normalizer-source", str(source / "qwenputer_controller_v30.py"),
               "--min-new-rows", str(n), "--min-trace-rows", str(traces),
               "--output", str(destination), "--manifest", str(manifest)]
    heldouts = [outer / name for name in launch.BASE_DEV] + [source / name for name in launch.SOURCE_EVAL] + [public_dev]
    heldouts += [original / "engineering/engineering_heldout.jsonl", original / "visual/visual_heldout.jsonl"]
    for path in heldouts:
        command.extend(("--heldout", str(path)))
    subprocess.run(command, check=True)
    require_sha(destination, PARENT_TRAIN_SHA)
    if read_json(manifest).get("output_rows") != PARENT_ROWS:
        raise RuntimeError("recomposed 30500 row count changed")
    return destination, source, corpus, original, old, runner


def check_a100():
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is unavailable. CPU preparation does not imply GPU training was run.") from error
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or torch.cuda.get_device_name(0) != "NVIDIA A100-SXM4-40GB":
        raise RuntimeError("exact-state continuation needs the recorded single NVIDIA A100-SXM4-40GB; no fallback GPU is selected")
    return torch


def build_command(*, source: Path, train: Path, corpus: Path, local: Path, output: Path,
                  steps: int, variant: str, repeat: int) -> list[str]:
    stop = PARENT_STEP + steps
    actual_repeat = 1 if variant == "replay_only" else repeat
    floor = 0 if variant == "replay_only" else 1000
    return [sys.executable, "-u", str(source / "runtime_practical_v1.py"),
            "--train_path", str(train), "--meta_path", str(source / "base_metadata.json"),
            "--corpus_path", str(corpus), "--warmstart_path", str(local / "checkpoint_step_30500.pt"),
            "--required_start_step", "30500", "--resume_state", str(local / "train_state_step_30500.pt"),
            "--output_dir", str(local), "--max_steps", str(stop), "--stop_at_step", str(stop),
            "--batch_size", "64", "--step_tokens", "36864", "--phase_lr", "3e-6",
            "--phase_floor", "3e-6", "--phase_start", "30500", "--phase_warmup", "0",
            "--skip_final_eval", "--mask_tool_output", "--save_optimizer", "--numina_repeat", "1",
            "--new_dataset_sampler", "--new_curriculum_repeat", str(actual_repeat),
            "--new_group_min_exposures", str(floor), "--distill_teacher", str(local / "checkpoint_step_30500.pt"),
            "--distill_weight", "1.0", "--distill_temperature", "1.0",
            "--practical_variant", variant, "--min_replay_share", "0.75",
            "--telemetry_jsonl", str(output / "training_events.jsonl")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=STUDY)
    parser.add_argument("--parent-run", default=PARENT_RUN)
    parser.add_argument("--parent-step", type=int, default=PARENT_STEP)
    parser.add_argument("--output", type=Path, required=True, help="new, previously nonexistent branch directory")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--new-repeat", type=int, default=16)
    parser.add_argument("--variant", choices=VARIANTS, default="multistep")
    parser.add_argument("--prepared-parent", type=Path, help="optional exact-SHA cached train_30500.jsonl")
    parser.add_argument("--python-data", type=Path, default=HERE / "data/practical_train.jsonl")
    parser.add_argument("--python-manifest", type=Path, default=HERE / "data/practical_manifest.json")
    parser.add_argument("--knowledge-data", type=Path, default=HERE / "data/knowledge_train.jsonl")
    parser.add_argument("--knowledge-manifest", type=Path, default=HERE / "data/knowledge_manifest.json")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--preflight-report", type=Path)
    args = parser.parse_args()
    if not 1 <= args.steps <= 2500 or not 1 <= args.new_repeat <= 16:
        parser.error("pilot steps must be 1..2500 and new-repeat 1..16")
    study, output = args.study.resolve(), args.output.resolve()
    if output.exists() or output == study or output.is_relative_to(study / args.parent_run):
        raise FileExistsError("the candidate output must be a new branch, outside the parent branch")
    # Package acceptance first; a corrupted new file never reaches old data or GPU.
    additions = [(args.python_data, args.python_manifest), (args.knowledge_data, args.knowledge_manifest)]
    for data, manifest in additions:
        manifest_train(data, manifest)
    heldouts = fixed_heldouts(HERE)
    parent, parent_receipt, parent_binding = checked_parent(study, args.parent_run, args.parent_step)
    if not args.preflight_only:
        refuse_active_work(study)
    scratch = Path(tempfile.mkdtemp(prefix="geocpu_practical_pilot_"))
    from frozen_eval import iter_heldout_prompts
    registry_views = scratch / "frozen_heldout_input_views.jsonl"
    with registry_views.open("x", encoding="utf-8") as f:
        for view in iter_heldout_prompts():
            f.write(json.dumps(view, ensure_ascii=False) + "\n")
    heldouts.append(registry_views)
    parent_train, source, corpus, original, old, runner = make_parent_dataset(study, scratch, args.prepared_parent)
    train, manifest_file = scratch / "train_practical.jsonl", scratch / "input_manifest.json"
    manifest = prepare_dataset(parent_train, additions, meta=source / "base_metadata.json",
            normalizer_source=source / "qwenputer_controller_v30.py", heldouts=heldouts,
            output=train, manifest_output=manifest_file, variant=args.variant)
    repeat = 1 if args.variant == "replay_only" else args.new_repeat
    preview = preview_plan(train, meta=source / "base_metadata.json",
            normalizer_source=source / "qwenputer_controller_v30.py",
            base_sampler_source=source / "runtime_cotraining_v41_247M.py",
            runtime_source=original / "training/runtime_30500.py", steps=args.steps,
            repeat=repeat, min_new=0 if args.variant == "replay_only" else 1000)
    runtime, diff = source / "runtime_practical_v1.py", scratch / "runtime_practical_v1.diff"
    runtime_binding = write_runtime(original / "training/runtime_30500.py", runtime, diff)
    report = {"event": "cpu_preflight", "created_utc": stamp(), "model_updates": 0,
              "parent_binding": parent_binding, "parent_checkpoint_sha256": parent_receipt["checkpoint_sha256"],
              "variant": args.variant, "steps": args.steps, "manifest": manifest, "preview": preview,
              "runtime": runtime_binding, "scratch": str(scratch),
              "gpu_execution": "not_started", "evaluation": "pending",
              "parent_checkpoint_sha_source": "pinned receipt; file bytes checked when copying for training"}
    new_json(scratch / "preflight.json", report)
    if args.preflight_report:
        new_json(args.preflight_report, report)
    print("GECPU_PRACTICAL_PREFLIGHT " + json.dumps(report, ensure_ascii=False), flush=True)
    if args.preflight_only:
        return
    refuse_active_work(study)
    torch = check_a100()
    local = scratch / "checkpoints"
    local.mkdir()
    model_bytes = (parent / "checkpoints/checkpoint_step_30500.pt").stat().st_size
    optimizer_bytes = (parent / "checkpoints/optimizer_step_30500.pt").stat().st_size
    check_root = output.parent
    while not check_root.exists():
        check_root = check_root.parent
    if shutil.disk_usage(check_root).free < model_bytes + optimizer_bytes + 1_500_000_000:
        raise RuntimeError("insufficient room for a separate candidate checkpoint, Adam and RNG state")
    for kind, destination in (("checkpoint", "checkpoint_step_30500.pt"),
                              ("optimizer", "optimizer_latest.pt"),
                              ("state", "train_state_step_30500.pt")):
        name = "train_state" if kind == "state" else kind
        old.copy_verified_once(parent / f"checkpoints/{name}_step_30500.pt", local / destination,
                               parent_receipt[kind + "_sha256"])
    output.mkdir(parents=True, exist_ok=False)
    active = output / "training_active.json"
    new_json(active, {"pid": os.getpid(), "started_utc": stamp(), "parent_sha256": PARENT_ARTIFACTS["checkpoint"]})
    stop = PARENT_STEP + args.steps
    fork = {"source_run": str(parent), "source_receipt_sha256": parent_binding["receipt_sha256"],
            "source_receipt": parent_receipt, "branch_start": 30500, "branch_stop": stop,
            "variant": args.variant, "new_repeat": repeat, "new_group": NEW_GROUP,
            "parent_training_data_sha256": PARENT_TRAIN_SHA,
            "new_training_data_sha256": manifest["output_sha256"],
            "teacher_sha256": parent_receipt["checkpoint_sha256"],
            "teacher_scope": "all old replay groups; new practical rows excluded",
            "teacher_frozen": True, "distill_weight": 1.0, "distill_temperature": 1.0,
            "learning_rate": 3e-6, "step_token_budget": 36864,
            "sampler_deliberately_reset": True, "model_adam_rng_restore_required": True,
            "runtime": runtime_binding, "deterministic_preview": preview,
            "retention": "not yet evaluated; training settings alone do not guarantee retention"}
    new_json(output / "fork_from_30500.json", fork)
    shutil.copyfile(manifest_file, output / "input_manifest.json")
    shutil.copyfile(scratch / "preflight.json", output / "preflight.json")
    shutil.copyfile(runtime, output / "runtime_practical_v1.py")
    shutil.copyfile(diff, output / "runtime_practical_v1.diff")
    command = build_command(source=source, train=train, corpus=corpus, local=local, output=output,
                            steps=args.steps, variant=args.variant, repeat=repeat)
    try:
        returncode = run_with_logging(command, output / "console", cwd=source)
        if returncode:
            raise RuntimeError(f"training child failed with exit {returncode}; inspect {output / 'console/console.log'}")
        state = torch.load(local / f"train_state_step_{stop}.pt", map_location="cpu", weights_only=True)
        actual = int(state["source_exposures"].get(NEW_GROUP, 0))
        if (state.get("step") != stop or state.get("practical_variant") != args.variant or
                state.get("new_dataset_sampler_started") is not True or
                state.get("new_curriculum_repeat") != repeat or actual != preview["new_presentations"]):
            raise RuntimeError("saved candidate state disagrees with the verified pilot preview")
        actual_share = state["practical_replay_supervised_tokens"] / max(1, state["practical_supervised_tokens"])
        if actual_share < 0.75:
            raise RuntimeError("saved candidate replay share is below the minimum")
        receipt = {"step": stop, "branch_from": 30500, "training_data_sha256": manifest["output_sha256"],
                   "parent_sha256": parent_receipt["checkpoint_sha256"],
                   "source_receipt_sha256": parent_binding["receipt_sha256"], "variant": args.variant,
                   "new_group": NEW_GROUP, "new_curriculum_presentations": actual,
                   "teacher_sha256": parent_receipt["checkpoint_sha256"],
                   "runtime_sha256": runtime_binding["candidate_sha256"],
                   "replay_supervised_share": actual_share, "phase_source_presentations": state["practical_phase_sources"]}
        for kind in ("checkpoint", "optimizer", "state"):
            name = "train_state" if kind == "state" else kind
            filename = f"{name}_step_{stop}.pt"
            receipt[kind + "_sha256"] = runner.atomic_copy(local / filename, output / "checkpoints" / filename)
        new_json(output / f"checkpoints/receipt_step_{stop}.json", receipt)
        new_json(output / "training_saved.json", {"step": stop, "branch_from": 30500, "full_model": True,
                    "parent_sha256": parent_receipt["checkpoint_sha256"], "variant": args.variant,
                    "training_data_sha256": manifest["output_sha256"], "evaluation": "pending",
                    "new_curriculum_presentations": actual})
        print("GECPU_PRACTICAL_SAVED_EVALUATION_PENDING " + json.dumps(receipt, ensure_ascii=False), flush=True)
    except BaseException as error:
        new_json(output / "training_incomplete.json", {"error_type": type(error).__name__, "error": str(error),
                  "timestamp_utc": stamp(), "scratch_preserved": str(scratch), "evaluation": "pending"})
        raise
    finally:
        if active.exists() and read_json(active).get("pid") == os.getpid():
            active.unlink()


if __name__ == "__main__":
    main()
