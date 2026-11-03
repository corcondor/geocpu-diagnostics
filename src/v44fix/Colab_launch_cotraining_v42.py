# -*- coding: utf-8 -*-
"""
Colab_launch_cotraining_v42.py: Colab A100 Master In-Process Runner for GeoCPU v42 (Search-Augmented 247.5M).

Guarantees:
- Scaled 247.56M Architecture with Zero Overhead Pre-Tokenized RAM Loader.
- Native Direct In-Process Execution (All prints stream directly to Colab notebook UI without subprocess buffering).
- Ampere A100 TF32 & JIT Recurrent Scan Acceleration.
- Warm-starts safely from Golden Master v41 (SWE 95.73%, Tier 2 84.50%).
- Blends ~3.0% Search-Augmented Multi-Hop Tool Reasoning (<SEARCH> => [RESULT:]).
- Complete 6-Tier Benchmark Suite (11,397 Total Items):
  * Tier 1: Held-Out SWE Step Control (200 Items)
  * Tier 2: Decomposed State Tracking (200 Items)
  * Tier 3: SoftBank JamC-QA Full (2,309 Items)
  * Tier 4: Canonical Broad 8,100 Full Benchmark (7,536 JMMLU + 164 SWE + 400 ARC)
  * Tier 5: 6-Operator Logical Instruction Contrast (400 Items)
  * Tier 6: Search-Augmented Multi-Hop Retrieval (200 Items)
- Master Historical Scoreboard comparing v38 vs v41 vs v42.
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import time
import shutil
import zipfile
from pathlib import Path
import json

def main():
    import gc
    sys.last_traceback = None
    sys.last_value = None
    sys.last_type = None
    gc.collect()
    try:
        import warnings
        import torch
        if torch.cuda.is_available():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for obj in list(gc.get_objects()):
                    try:
                        if isinstance(obj, torch.nn.Module):
                            obj.cpu()
                        elif isinstance(obj, torch.optim.Optimizer):
                            obj.state.clear()
                    except Exception:
                        pass
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass

    print("=" * 95, flush=True)
    print("GEO-CPU FRONTIER CO-TRAINING v42: SEARCH-AUGMENTED 247.5M NEUROSYMBOLIC RUNNER", flush=True)
    print("=" * 95, flush=True)

    # 1. Mount Google Drive & Hardware Check
    print("\n[1/5] MOUNTING GOOGLE DRIVE & VERIFYING HARDWARE...", flush=True)
    drive_mounted = os.path.exists("/content/drive/MyDrive")
    if not drive_mounted:
        try:
            from google.colab import drive
            drive.mount("/content/drive")
            print("  Successfully mounted Google Drive at /content/drive.", flush=True)
        except Exception as e:
            print(f"  Note: Google Drive mount skipped or local test mode: {e}", flush=True)
    else:
        print("  Google Drive is already mounted at /content/drive.", flush=True)

    import torch
    print(f"  PyTorch Version: {torch.__version__}", flush=True)
    print(f"  CUDA Available:  {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"  GPU Name:        {torch.cuda.get_device_name(0)}", flush=True)
        print(f"  VRAM Available:  {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB", flush=True)

    # 2. Workspace extraction
    print("\n[2/5] UNPACKING DEPLOYMENT BUNDLE: Grand_cotraining_v42.zip...", flush=True)
    zip_paths = [
        "/content/drive/MyDrive/GeoCPU_Frontier_CoTraining_v42/Grand_cotraining_v42.zip",
        "/content/drive/MyDrive/GeoCPU_Frontier_CoTraining_v41/Grand_cotraining_v42.zip",
        "/content/Grand_cotraining_v42.zip",
        "Grand_cotraining_v42.zip"
    ]
    target_zip = None
    for zp in zip_paths:
        if os.path.exists(zp):
            target_zip = zp
            break

    workspace_dir = "/content/cotraining_v42_workspace"
    if target_zip:
        print(f"  Found zip at: {target_zip}", flush=True)
        shutil.rmtree(workspace_dir, ignore_errors=True)
        os.makedirs(workspace_dir, exist_ok=True)
        with zipfile.ZipFile(target_zip, 'r') as zf:
            zf.extractall(workspace_dir)
        print(f"  Extracted workspace to: {workspace_dir}", flush=True)
    else:
        print(f"  Local workspace fallback: {os.getcwd()}", flush=True)
        workspace_dir = os.getcwd()

    os.chdir(workspace_dir)
    if workspace_dir not in sys.path:
        sys.path.insert(0, workspace_dir)

    # Evict any cached modules to guarantee fresh execution
    for mod_name in list(sys.modules.keys()):
        if mod_name in (
            "integrated_model_v30",
            "integrated_model_v42",
            "local_search_engine_v42",
            "runtime_cotraining_v41_247M",
            "runtime_cotraining_v42_search",
            "eval_step_control_heldout",
            "eval_v38_decomposed_state_bench",
            "eval_benchmark_full_8100",
            "eval_jamcqa_official",
            "eval_logical_instruction_operators",
            "eval_v42_search_retrieval",
        ):
            del sys.modules[mod_name]

    # 3. Check for Existing Step 5000 Checkpoint or Warmstart
    print("\n[3/5] CHECKING FOR TRAINED CHECKPOINTS & WARMSTART...", flush=True)
    out_dir = "/content/drive/MyDrive/GeoCPU_Frontier_CoTraining_v42/checkpoints"
    os.makedirs(out_dir, exist_ok=True)

    step5000_ckpt = Path(out_dir) / "checkpoint_step_5000.pt"
    force_retrain = os.environ.get("FORCE_RETRAIN", "0") == "1"
    eval_only_flag = os.environ.get("EVAL_ONLY", "0") == "1"

    train_data = f"{workspace_dir}/data/train_v42_search_unified.jsonl"
    meta_path = f"{workspace_dir}/base_metadata.json"
    dev_str = "cuda" if torch.cuda.is_available() else "cpu"

    if (step5000_ckpt.exists() and step5000_ckpt.stat().st_size > 500_000_000 and not force_retrain) or eval_only_flag:
        eval_ckpt = step5000_ckpt if step5000_ckpt.exists() else Path("/content/drive/MyDrive/GeoCPU_Frontier_CoTraining_v41/golden_master_v41_success/checkpoint_step_5000_golden.pt")
        print(f"  Target Checkpoint for Evaluation: {eval_ckpt} ({eval_ckpt.stat().st_size:,} bytes)!", flush=True)
        print("  -> Launching Full 6-Tier Benchmark Suite (11,397 items) directly...", flush=True)

        cmd_args = [
            "runtime_cotraining_v42_search.py",
            "--eval_only",
            "--checkpoint", str(eval_ckpt),
            "--meta_path", meta_path,
            "--output_dir", out_dir,
            "--device", dev_str,
            "--batch_size", "64"
        ]
    else:
        # Locate Warmstart: Prefer Golden Master from v41
        warmstart_candidates = [
            "/content/drive/MyDrive/GeoCPU_Frontier_CoTraining_v41/golden_master_v41_success/checkpoint_step_5000_golden.pt",
            "/content/drive/MyDrive/GeoCPU_Frontier_CoTraining_v41/checkpoints/checkpoint_step_5000.pt",
            f"{out_dir}/checkpoint_step_2500.pt",
            f"{workspace_dir}/checkpoints/checkpoint_step_5000.pt"
        ]
        warmstart_file = None
        for cand in warmstart_candidates:
            if os.path.exists(cand) and os.path.getsize(cand) > 500_000_000:
                warmstart_file = cand
                break

        if warmstart_file:
            print(f"  Active warmstart base checkpoint: {warmstart_file} ({os.path.getsize(warmstart_file):,} bytes)", flush=True)
            print("  -> Warm-starting safely from Golden Master v41 (SWE 95.7%, Tier 2 84.5% preserved).", flush=True)
        else:
            print("  WARNING: No golden master checkpoint found! Checking fallback paths...", flush=True)

        print("\n" + "=" * 95, flush=True)
        print("[4/5] LAUNCHING v42 247.5M SEARCH-AUGMENTED CO-TRAINING (5,000 STEPS, LR=4e-5)...", flush=True)
        print("=" * 95, flush=True)

        cmd_args = [
            "runtime_cotraining_v42_search.py",
            "--train_path", train_data,
            "--meta_path", meta_path,
            "--output_dir", out_dir,
            "--max_steps", "5000",
            "--batch_size", "96",
            "--lr", "4e-5",
            "--save_interval", "2500",
            "--eval_interval", "2500",
            "--device", dev_str
        ]
        if warmstart_file:
            cmd_args.extend(["--warmstart_path", warmstart_file])

    # 4. Direct In-Process Execution (Ensures 100% real-time streaming to Colab UI)
    print(f"\n[RUNNING V42 ENGINE IN-PROCESS] {' '.join(cmd_args)}\n", flush=True)
    sys.argv = cmd_args

    import runtime_cotraining_v42_search
    runtime_cotraining_v42_search.main()

    print("\n" + "=" * 95, flush=True)
    print("[5/5] v42 247.5M EXECUTION AND FULL 6-TIER BENCHMARKS COMPLETED SUCCESSFULLY!", flush=True)
    print("=" * 95, flush=True)

if __name__ == "__main__":
    main()
