# Private T4x2 stage-1 fixed44 evaluation package

This directory builds a private Kaggle package for **read-only, post-training** evaluation. It evaluates the sealed 250-update control and Python checkpoints from the v3 paired stage-1 notebook `geocpuresearch/geocpu-v43py-stage1-m4096-20261104` concurrently. It does not train, submit a Kaggle job, or promote either arm. The parent fixed44 baseline is an input and is **not regenerated**.

The fixed44 diagnostic consists of 44 fixed cases and 84 case/profile generations: 44 native, 20 Python-cued, and 20 Python-uncued. All three arms use the unchanged `eval_fixed_diagnostic.py` and `fixed_interim_diagnostic_44_v2.jsonl`. The two stage-1 arms use the exact `eval_fp16_native_v1` profile and 2,048-token limit of the parent baseline. The original BF16 SWE200 capture is bound as historical provenance only; this package does not compare a new SWE200 score to it.

## Required inputs

- `--paired-dir`: local **small proof bundle** from the completed v3 paired Kaggle output root `geocpu_hybrid_stage1_fp16_paired_m4096_20261104_v3`. It must contain `pair_status.json`, both `hybrid_stage1_<arm>_update_0250.pt.receipt.json` files, and both `admitted_actual_target_plan.json` files. The multi-GB `.pt` files may be absent locally. If they are present, the builder hashes them too. The two receipt hashes and byte counts are pinned in the package. The package attaches the completed v3 notebook as `kernel_sources` and hashes the mounted checkpoints before and after evaluation. It does not need a checkpoint dataset upload.
- `--v43-zip` and `--parent-checkpoint`: local original v43 source ZIP and step-15000 checkpoint. Their pinned hashes and the checkpoint size are checked during preparation and in Kaggle. Kaggle reads the originals from `geocpuresearch/geocpu-v43-v44fix-paired-eval-20261103`.
- `--swe-proof-dir`: original BF16 SWE200 `COMPLETE.json` and its seven raw/grade files. This proves the historical checkpoint binding.
- `--fixed44-proof-dir`: parent FP16 fixed44 `COMPLETE.json` and `raw_answers_44_profiles.jsonl`. The builder verifies the exact 84 raw case/profile rows and embeds this baseline for read-only comparison.
- `--control-claim` and `--python-claim`: distinct, fresh, one-use `prelaunch_contract_once.json` files with their three evidence files. Each claims one T4, `eval_fp16_native_v1`, 1,800 seconds of bounded work and 180 seconds of save reserve. Their common official free-time snapshot, after existing unreturned local holds, must cover both reservations without counting total or paid allowance. Both claims bind the same SWE200 and fixed44 parent COMPLETE hashes. Each root hold ledger must list this package's kernel ID, both sorted claim IDs, and the combined device seconds. The active-job snapshot must mark BF16 SWE200, FP16 parent fixed44, and v3 paired stage-1 Kaggle jobs `COMPLETE`.
- `--staged`: a directory that does not exist yet.

The [Kaggle notebook documentation](https://www.kaggle.com/docs/notebooks) permits a completed notebook's saved output as another notebook's input. The [kernel metadata specification](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md) defines `kernel_sources`. Kaggle's [mount resolver](https://github.com/Kaggle/kagglehub/blob/main/src/kagglehub/kaggle_cache_resolver.py) receives a provider-chosen mount slug, so the runtime searches for exactly one output root with the pinned status hash. The Kaggle CLI's `--file-pattern` option can fetch only the five small proof files for local package preparation:

```powershell
kaggle kernels output geocpuresearch/geocpu-v43py-stage1-m4096-20261104 `
  -p <NEW_SMALL_PROOF_DOWNLOAD_DIRECTORY> `
  --file-pattern 'pair_status\.json|admitted_actual_target_plan\.json|hybrid_stage1_(control|python)_update_0250\.pt\.receipt\.json'
```

After the v3 stage-1 notebook has completed and its small proof files have been copied locally, issue fresh evaluation claims. The issuer only reads the official Kaggle quota and job statuses. It refuses a running or failed v3 job, a nonterminal recent job, a reused evaluation ID, and insufficient guaranteed free GPU time. Historical jobs older than 24 hours are recorded without querying status endpoints that may return provider 404; the required proof jobs are always status checked. The issuer conservatively carries forward 15,840 device seconds of prior local holds from the v3 claim ledger. A lower hold value requires a fresh `geocpu_kaggle_hold_reconciliation_v1` evidence file passed with `--reconciliation-evidence`.

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -B issue_claims.py `
  --swe-proof-dir <BF16_SWE200_OUTPUT>/actual `
  --fixed44-proof-dir <FP16_FIXED44_OUTPUT>/actual `
  --existing-holds-device-seconds 15840 `
  --hard-stop-at-utc <FUTURE_UTC_ISO_TIMESTAMP> `
  --output-dir <NEW_ONE_USE_EVAL_CLAIM_DIRECTORY>
```

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -B prepare.py `
  --paired-dir <PAIRED_STAGE1_SMALL_PROOF_ROOT> `
  --v43-zip <ORIGINAL_V43_SOURCE_ZIP> `
  --parent-checkpoint <ORIGINAL_V43_STEP15000_PT> `
  --swe-proof-dir <BF16_SWE200_OUTPUT>/actual `
  --fixed44-proof-dir <FP16_FIXED44_OUTPUT>/actual `
  --control-claim <CONTROL_EVAL_CLAIM>/prelaunch_contract_once.json `
  --python-claim <PYTHON_EVAL_CLAIM>/prelaunch_contract_once.json `
  --staged <NEW_ONE_USE_STAGE_DIRECTORY>
```

`bootstrap.py`, `kernel-metadata.json`, and `package-manifest.json` are the outputs in the staging directory. The metadata requests a private T4x2 Kaggle runtime and attaches the completed v3 notebook output directly. Kaggle chooses the mount slug; the bootstrap locates one matching output root by its pinned `pair_status.json` hash, then verifies every mounted checkpoint, receipt, and plan hash. This builder does not upload or run the evaluation. Before any later submission, the external resource ledger must consume both claims once; an in-job copy cannot consume shared claims atomically.

At runtime, `CUDA_VISIBLE_DEVICES=0` runs the control evaluator and `CUDA_VISIBLE_DEVICES=1` runs the Python evaluator. Both invoke `launch_fixed_diagnostic_bounded.py`. The runtime preserves each arm's `raw_answers_44_profiles.jsonl`, `COMPLETE.json` or `PARTIAL.json`, and log. If both arms complete, `matched_rows_84.jsonl` records each parent/control/Python answer for the identical case and profile. `comparison.json` records case-level Boolean flips and provenance hashes. Search, logic, and Japanese explanation cases retain raw answers but have no invented automatic grade. `evaluation_status.json` records partial or failed runs without presenting them as complete.

## CPU-only checks

```powershell
python -B -m unittest -v test_package.py
```

The checks parse the template and sources, verify pinned source hashes, and exercise sealed-checkpoint, mounted-output selection, claim issuance, and exact-case comparison rejection paths with synthetic local data. They do not load a real checkpoint, call a GPU, or submit a job.
