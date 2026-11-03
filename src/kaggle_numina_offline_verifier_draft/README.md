# NuminaMath-TIR offline verifier draft

This directory contains a **prepared, unlaunched** verifier for the public
NuminaMath-TIR train Parquet pinned by the v44fix source lock. It has not run on
public rows or on Kaggle. No existing v44fix repair release was edited.
This draft intentionally cannot run public rows: `kaggle_entrypoint.main()`
and `verifier.run_shard(production=True)` stop unconditionally before input
reading or worker creation. A separately reviewed distribution is required.
The separate environment probe cannot authorize this verifier kernel.

## Files and output

- `verifier.py` checks the pinned train and official test Parquet hashes, then
  processes a source-row range. It uses the v44fix question split and target
  format. Each accepted row has an exact stdout comparison for every Python
  block in a fresh problem namespace. Every source row gets a hash-only
  receipt with its status. Failed, timed-out, mismatched and truncated rows
  stay out of the curriculum.
- `executor_worker_v44r1.py` is an unchanged copy of the v44fix bounded
  Python worker. The verifier launches it as a child with an empty allowlisted
  environment, Linux resource limits, a private temporary directory, and a
  non-root user when the controller starts as root. It recycles after a timeout,
  after 100 rows, or at 20 cumulative CPU seconds. A problem has at most eight
  blocks with five-second wall limits, leaving headroom under the worker's
  90-second hard CPU cap. These restrictions are **not a security proof**.
- `probe.py` and `probe-kernel-metadata.json` are a separate private CPU
  environment probe with no attached datasets. The probe checks one external
  TCP destination, Drive-like mount paths, and credential presence using
  booleans only. It does not execute public code or print secret values.
- `verifier-kernel-metadata.template.json` declares private, GPU-disabled,
  internet-disabled execution. Its owner and public input dataset fields must
  be filled only after review. `kaggle_entrypoint.py` also contains a public
  dataset name placeholder. Neither kernel has been created or uploaded.

`run-shard` writes 64-row immutable chunks and updates a manifest after each
chunk. A new session can pass the previous output through `--prior`; every
copied chunk is checked by size, SHA-256, contiguous row indices, and exact
accepted receipt-to-record provenance and tool-observation hashes. A
deadline is checked between rows, with at most the current incomplete chunk
lost on interruption. `merge` requires complete, non-overlapping ranges from
row zero through the Parquet row count. It writes `row_receipts.jsonl`,
`train.jsonl`, `dev.jsonl`, `test.jsonl`, `dataset_manifest.json`, and a
provisional `offline_verification_receipt.json`.

## Input contract

The single public Kaggle input dataset must contain the pinned Numina train
and official test Parquet files, a character vocabulary, a hash-only evaluation
exclusion index, and a hash binding JSON file. A `run_config.json` maps those
five relative filenames and sets `shard_start`, `shard_end`, and
`wall_budget_seconds`. The source lock is bundled here and checked against the
hard-coded SHA-256. The source Parquet files are measured against its byte
counts and SHA-256 values before any child starts.

The hash binding uses schema `geocpu_numina_hash_binding_v1` and contains
`source_lock_sha256`, the `vocabulary` and `exclusion_index` file pins
(`bytes`, `sha256`), an `anchor` pin (`bytes`, `sha256`), and an
`evaluation_inputs` list of `{name, bytes, sha256}`. The exclusion index uses
schema `geocpu_evaluation_hash_index_v1`, repeats `evaluation_inputs`, and
lists `question_sha256` and `id_sha256` values. It contains no evaluation text.
The Numina official test questions are read directly from the pinned public
test Parquet and excluded from training.

The evaluation hash index and anchor pin are externally asserted inputs. This
package does not prove that the index matches the underlying evaluation files,
and it does not read or attach the v43 anchor corpus. The code and Japanese
public buckets are outside this verifier. Therefore its manifest always sets
`training_eligible=false` and `stage_a_admissible=false`.
The provisional manifest also keeps `accepted_rows_all_blocks_stdout_exact`
false. The synthetic merge validates receipt-to-record correspondence, but no
public execution has taken place.

## Existing Stage A admission conflict

The existing v44fix Stage A checks a pinned
`OFFLINE_VERIFICATION_RECEIPT_SHA256`, the hashes of its three original GPT
scripts, all four public source pins, all four curriculum buckets, vocabulary,
evaluation and anchor pins, output hashes, and the literal isolation claim
`credential_free_disposable_vm`. This draft makes no such isolation claim.
Its provisional receipt has a different schema, so Stage A rejects it. A
reviewed integration must resolve the source-code and four-bucket contracts
and establish the required isolation evidence before a new distribution can
be pinned. No training or evaluation may use this draft as an admission token.

## Local verification

`python -m pytest -q test_verifier.py` uses only synthetic Parquet rows and a
fake executor. It does not run any source-row Python. The probe test uses an
injected fake network result. The local tests do not establish actual Kaggle
network, credential, UID-drop, or filesystem isolation behavior.
