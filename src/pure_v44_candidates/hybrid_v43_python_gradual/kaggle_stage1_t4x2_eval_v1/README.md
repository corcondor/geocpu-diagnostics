# Private T4x2 stage-1 fixed44 evaluation package

This directory builds a private Kaggle package for **read-only, post-training** evaluation. It evaluates the sealed 250-update control and Python checkpoints from the paired stage-1 run concurrently. It does not train, submit a Kaggle job, or promote either arm. The parent fixed44 baseline is an input and is **not regenerated**.

The fixed44 diagnostic consists of 44 fixed cases and 84 case/profile generations: 44 native, 20 Python-cued, and 20 Python-uncued. All three arms use the unchanged `eval_fixed_diagnostic.py` and `fixed_interim_diagnostic_44_v2.jsonl`. The two stage-1 arms use the exact `eval_fp16_native_v1` profile and 2,048-token limit of the parent baseline. The original BF16 SWE200 capture is bound as historical provenance only; this package does not compare a new SWE200 score to it.

## Required inputs

- `--paired-dir`: local copy of the completed paired Kaggle output root `geocpu_hybrid_stage1_fp16_paired_20261104`. It must contain `pair_status.json`, both `hybrid_stage1_<arm>_update_0250.pt` files and receipts, and both `admitted_actual_target_plan.json` files. The builder hashes and checks them. The corresponding private Kaggle dataset must contain this same root directory and bytes.
- `--paired-dataset`: Kaggle `owner/slug` for that private paired-output dataset. The default is only a proposed slug; specify the actual dataset identifier if different. The package pins the checkpoint, receipt, plan, and paired status SHA-256 hashes and rechecks the mounted dataset before and after evaluation.
- `--v43-zip` and `--parent-checkpoint`: local original v43 source ZIP and step-15000 checkpoint. Their pinned hashes and the checkpoint size are checked during preparation and in Kaggle. Kaggle reads the originals from `geocpuresearch/geocpu-v43-v44fix-paired-eval-20261103`.
- `--swe-proof-dir`: original BF16 SWE200 `COMPLETE.json` and its seven raw/grade files. This proves the historical checkpoint binding.
- `--fixed44-proof-dir`: parent FP16 fixed44 `COMPLETE.json` and `raw_answers_44_profiles.jsonl`. The builder verifies the exact 84 raw case/profile rows and embeds this baseline for read-only comparison.
- `--control-claim` and `--python-claim`: distinct, fresh, one-use `prelaunch_contract_once.json` files with their three evidence files. Each claims one T4, `eval_fp16_native_v1`, 1,800 seconds of bounded work and 180 seconds of save reserve. Their common official free-time snapshot, after existing unreturned local holds, must cover both reservations without counting total or paid allowance. Both claims bind the same SWE200 and fixed44 parent COMPLETE hashes. Each root hold ledger must list this package's kernel ID, both sorted claim IDs, and the combined device seconds. The active-job snapshot must mark the parent fixed44 and paired stage-1 Kaggle jobs `COMPLETE`.
- `--staged`: a directory that does not exist yet.

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -B prepare.py `
  --paired-dir <PAIRED_STAGE1_OUTPUT_ROOT> `
  --paired-dataset <OWNER/PRIVATE_PAIRED_OUTPUT_DATASET_SLUG> `
  --v43-zip <ORIGINAL_V43_SOURCE_ZIP> `
  --parent-checkpoint <ORIGINAL_V43_STEP15000_PT> `
  --swe-proof-dir <BF16_SWE200_OUTPUT>/actual `
  --fixed44-proof-dir <FP16_FIXED44_OUTPUT>/actual `
  --control-claim <CONTROL_EVAL_CLAIM>/prelaunch_contract_once.json `
  --python-claim <PYTHON_EVAL_CLAIM>/prelaunch_contract_once.json `
  --staged <NEW_ONE_USE_STAGE_DIRECTORY>
```

`bootstrap.py`, `kernel-metadata.json`, and `package-manifest.json` are the outputs in the staging directory. The metadata requests a private T4x2 Kaggle runtime. This builder does not upload or run it. Before any later submission, the external resource ledger must consume both claims once; an in-job copy cannot consume shared claims atomically.

At runtime, `CUDA_VISIBLE_DEVICES=0` runs the control evaluator and `CUDA_VISIBLE_DEVICES=1` runs the Python evaluator. Both invoke `launch_fixed_diagnostic_bounded.py`. The runtime preserves each arm's `raw_answers_44_profiles.jsonl`, `COMPLETE.json` or `PARTIAL.json`, and log. If both arms complete, `matched_rows_84.jsonl` records each parent/control/Python answer for the identical case and profile. `comparison.json` records case-level Boolean flips and provenance hashes. Search, logic, and Japanese explanation cases retain raw answers but have no invented automatic grade. `evaluation_status.json` records partial or failed runs without presenting them as complete.

## CPU-only checks

```powershell
python -B -m unittest -v test_package.py
```

The checks parse the template and sources, verify pinned source hashes, and exercise sealed-checkpoint and exact-case comparison rejection paths with synthetic local data. They do not load a real checkpoint, call a GPU, or submit a job.
