# GeoCPU v44fix TIR consistency: isolated Colab gate

This branch retains the v43 neural model and ALU while adding Python TIR. It is
separate from the pure-Python/no-ALU v44 branch. Neither the v43 source ZIP nor
its step-15000 checkpoint is modified.

The distribution is `GeoCPU_v44fix_TIR_consistency_bundle.zip`, verified by the
SHA-256 value in the accompanying one-cell launcher. The package also verifies
every extracted member against `release_manifest.json` before importing code.

Training is blocked (`training_ready=false`) in this distribution. Public
Numina Python blocks must first be verified in a disposable, credential-free
environment with network disabled and no Drive mount. The bundled Python
worker is not a security sandbox. The resulting curriculum and offline
verification receipt must be reviewed, then the receipt SHA-256 must be pinned
in a new distribution. Colab accepts only that pinned receipt, the exact
source lock and source file hashes, evaluation and vocabulary hashes, the v43
anchor hash, and matching output hashes. It never executes public code during
Stage A. The old `GeoCPU_v44fix_20261103/run_1` data is never reused.

Model-generated Python at evaluation time also needs an OS security boundary
before a Colab run with Drive mounted. The isolated worker now has a Linux
4 GiB address-space and 60-second CPU limit, but those resource caps do not
isolate credentials, files, or network access. This release therefore remains
blocked even if its source files compile and its CPU fixtures pass.
The public preparation worker also has a cumulative 60-second CPU cap. The
current verify-all loop keeps one worker across problems, so a future isolated
verifier must recycle it safely without carrying state between problems.

After that prerequisite, the first cell builds a fresh mixture with tool
observation spans, evaluates the v43 baseline, runs an independent 1,000-update
gate trial, and evaluates the gate. It stops after PASS or FAIL. It writes only
to `MyDrive/GeoCPU_v44fix_TIR_consistency_20261103/run_1/`.

The gate and full training are separate runs from the same pinned v43 parent.
Passing the gate does not imply that the full run will keep its quality. Once
the gate has passed, an explicit later invocation of `colab_v44fix.main(full=True)`
starts the 15,000→25,000 run independently. Within each run, a completed
checkpoint restores weights, AdamW, CPU/CUDA/Python/NumPy RNG and the pending
batch order. A missing, corrupted or half-committed checkpoint stops the run
for inspection; it is never treated as a successful continuation.

The checkpoint format holds about 3 GB per completed step. One gate checkpoint
and four full-run checkpoints need roughly 15 GB, plus a temporary write of
about 3 GB. The entry checks that the Drive mount reports at least 18 GB free
before data preparation or training. The reported mount capacity is only a
preflight; a write failure still stops without accepting a partial checkpoint.

Before packaging, local CPU fixtures cover tool masking, executable Python
fences, resumed sampler order and RNG, checkpoint hash/strict restore, gate
cache keys, and rejection of every OOM update. The distribution receipt records
file hashes; test results are reported separately. This document does not
claim any Colab training or new model performance.

Data and evaluation scope: the public TIR answer following a tool result is
still the author's answer, not an independently proved mathematical solution.
The code and Japanese instruction buckets are also not independently checked
for factual correctness. The retained v43 anchor has known prompt overlap
with the aligned Tier-6 search set, so that score is retention diagnostics,
not clean unseen generalization. Tier-1 through Tier-7 and the separate TIR
gate must be reported separately. This hybrid does not yet implement the
original pure-Python v44 Tier-1 through Tier-8 suite as one full evaluation.
