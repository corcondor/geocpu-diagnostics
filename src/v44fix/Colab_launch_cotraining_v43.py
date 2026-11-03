# -*- coding: utf-8 -*-
"""
Colab_launch_cotraining_v43.py: Colab A100 Master In-Process Runner for GeoCPU v43 (Knowledge-Scaled & Dual A/B Evaluator).

Guarantees:
- Scaled 247.56M Architecture with Zero Overhead Pre-Tokenized RAM Loader.
- Native Direct In-Process Execution (All prints stream directly to Colab notebook UI without subprocess buffering).
- Ampere A100 TF32 & JIT Recurrent Scan Acceleration.
- Warm-starts safely from Milestone v42 checkpoint_step_5000.pt (preserving SWE 95.5%, Tier 2 85.5%, Search 100%).
- Knowledge-Scaled Co-Training: 268,032 Unified samples in train_v43_knowledge_unified.jsonl.
- Complete 7-Tier Benchmark Suite (11,597 Total Items):
  * Tier 1: Held-Out SWE Step Control (200 Items)
  * Tier 2: Decomposed State Tracking (200 Items)
  * Tier 3: SoftBank JamC-QA Full (2,309 Items)
  * Tier 4: Canonical Broad 8,100 Full Benchmark (7,536 JMMLU + 164 SWE + 400 ARC)
  * Tier 5: 6-Operator Logical Instruction Contrast (400 Items)
  * Tier 6: Search-Augmented Multi-Hop Retrieval (200 Items)
  * Tier 7: Formal A/B Benchmark: Closed-Book (学習のみ) vs Open-Book (学習＋検索あり)
- Master Historical Scoreboard comparing v38 vs v41 vs v42 vs v43.
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
    print("GEO-CPU FRONTIER CO-TRAINING v43: KNOWLEDGE-SCALED & DUAL A/B BENCHMARK RUNNER", flush=True)
    print("=" * 95, flush=True)

    # 1. Mount Google Drive and verify GPU
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
    print("\n[2/5] UNPACKING DEPLOYMENT BUNDLE: Grand_cotraining_v43.zip...", flush=True)
    zip_paths = [
        "/content/drive/MyDrive/GeoCPU_Frontier_CoTraining_v43/Grand_cotraining_v43.zip",
        "/content/drive/MyDrive/GeoCPU_Frontier_CoTraining_v42/Grand_cotraining_v43.zip",
        "/content/Grand_cotraining_v43.zip",
        "Grand_cotraining_v43.zip"
    ]
    target_zip = None
    for zp in zip_paths:
        if os.path.exists(zp):
            target_zip = zp
            break

    workspace_dir = "/content/cotraining_v43_workspace"
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

    # Evict cached modules to guarantee fresh execution
    for mod_name in list(sys.modules.keys()):
        if mod_name in (
            "integrated_model_v30",
            "integrated_model_v42",
            "local_search_engine_v42",
            "runtime_cotraining_v41_247M",
            "runtime_cotraining_v42_search",
            "runtime_cotraining_v43_knowledge",
            "eval_step_control_heldout",
            "eval_v38_decomposed_state_bench",
            "eval_benchmark_full_8100",
            "eval_jamcqa_official",
            "eval_logical_instruction_operators",
            "eval_v42_search_retrieval",
            "eval_v43_ab_benchmark"
        ):
            del sys.modules[mod_name]

    # 3. Check for Existing Step 15000 Checkpoint or Step 5000 Warmstart
    print("\n[3/5] CHECKING FOR TRAINED CHECKPOINTS & WARMSTART...", flush=True)
    out_dir = "/content/drive/MyDrive/GeoCPU_Frontier_CoTraining_v43/checkpoints"
    os.makedirs(out_dir, exist_ok=True)

    step15000_ckpt = Path(out_dir) / "checkpoint_step_15000.pt"
    force_retrain = os.environ.get("FORCE_RETRAIN", "0") == "1"
    eval_only_flag = os.environ.get("EVAL_ONLY", "0") == "1"

    warmstart_candidates = [
        "/content/drive/MyDrive/GeoCPU_Frontier_CoTraining_v42/checkpoints/checkpoint_step_5000.pt",
        "/content/drive/MyDrive/GeoCPU_Frontier_CoTraining_v41/golden_master_v41_success/checkpoint_step_5000_golden.pt",
        f"{workspace_dir}/checkpoints/checkpoint_step_5000.pt"
    ]
    warmstart_ckpt = next((p for p in warmstart_candidates if os.path.exists(p)), None)

    train_data = f"{workspace_dir}/data/train_v43_knowledge_unified.jsonl"
    meta_path = f"{workspace_dir}/base_metadata.json"
    dev_str = "cuda" if torch.cuda.is_available() else "cpu"

    if (step15000_ckpt.exists() and step15000_ckpt.stat().st_size > 500_000_000 and not force_retrain) or eval_only_flag:
        eval_ckpt = step15000_ckpt if step15000_ckpt.exists() else warmstart_ckpt
        print(f"  Target Checkpoint for Evaluation: {eval_ckpt} ({Path(eval_ckpt).stat().st_size:,} bytes)!", flush=True)
        print("  -> Launching Full 7-Tier Benchmark Suite (11,597 items) directly...", flush=True)

        cmd_args = [
            "runtime_cotraining_v43_knowledge.py",
            "--eval_only",
            "--checkpoint", str(eval_ckpt),
            "--meta_path", meta_path,
            "--output_dir", out_dir,
            "--device", dev_str,
            "--batch_size", "64"
        ]
    else:
        print(f"  Warmstart Checkpoint: {warmstart_ckpt} (Exists: {warmstart_ckpt is not None})", flush=True)
        print("  -> Launching Knowledge-Scaled Co-Training (Step 5,000 -> 15,000)...", flush=True)

        cmd_args = [
            "runtime_cotraining_v43_knowledge.py",
            "--train_path", train_data,
            "--meta_path", meta_path,
            "--warmstart_path", str(warmstart_ckpt) if warmstart_ckpt else "",
            "--output_dir", out_dir,
            "--max_steps", "15000",
            "--batch_size", "64",
            "--lr", "3.5e-5",
            "--device", dev_str
        ]

    # 4. Direct In-Process Invocation
    print(f"\n[RUNNING V43 ENGINE IN-PROCESS] {' '.join(cmd_args)}\n", flush=True)
    import runtime_cotraining_v43_knowledge

    saved_argv = sys.argv
    try:
        sys.argv = cmd_args
        runtime_cotraining_v43_knowledge.main()
    finally:
        sys.argv = saved_argv

    print("\n" + "=" * 95)
    print("GEO-CPU V43 CO-TRAINING & EVALUATION COMPLETE")
    print(f"Checkpoints and benchmark reports persisted to: {out_dir}")
    print("=" * 95)

if __name__ == "__main__":
    main()
