"""Read-only actual-pool comparison of the 0% and 1% stage-one plans.

This audit deliberately bypasses the independent Python replay receipt.  It
never calls the training entry or creates a model, optimizer, or GPU workload.
The production training entry still requires the full replay receipt.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys


CANDIDATE = (Path(__file__).parents[2] / "src" / "pure_v44_candidates" /
             "hybrid_v43_python_gradual")
sys.path.insert(0, str(CANDIDATE))

from hybrid_data import plan_exposure  # noqa: E402
from source_pools import file_sha, load_verified_pools  # noqa: E402
from train_hybrid_stage import BATCH_SIZE, STEPS_PER_STAGE, _microbatches  # noqa: E402


SEED = 20261103
MICRO_TOKENS = 24_576
FIRST_BARRIER = 250


def audit(v43_zip: Path, converted: Path) -> dict:
    pools = load_verified_pools(v43_zip=v43_zip, converted_jsonl=converted,
                                require_independent_replay=False)
    control = plan_exposure(pools.old, pools.python, cap=0.0,
                            updates=STEPS_PER_STAGE, batch_size=BATCH_SIZE, seed=SEED)
    treatment = plan_exposure(pools.old, pools.python, cap=0.01,
                              updates=STEPS_PER_STAGE, batch_size=BATCH_SIZE, seed=SEED)
    if control.paired_old_batches != treatment.paired_old_batches:
        raise AssertionError("The paired old-row slot schedules differ")

    totals = Counter()
    first = Counter()
    families = Counter()
    first_families = Counter()
    for step in range(STEPS_PER_STAGE):
        old_slots = control.paired_old_batches[step]
        control_batch = control.batches[step]
        treatment_batch = treatment.batches[step]
        if any(kind != "v43" or index != old_index
               for old_index, (kind, index) in zip(old_slots, control_batch)):
            raise AssertionError("Control differs from its paired old-row slots")
        common_lengths = [max(len(pools.old[old_index].tokens),
                              len((pools.python[index] if kind == "python" else pools.old[index]).tokens))
                          for old_index, (kind, index) in zip(old_slots, treatment_batch)]
        control_rows = [pools.old[index] for _, index in control_batch]
        treatment_rows = [(pools.python[index] if kind == "python" else pools.old[index])
                          for kind, index in treatment_batch]
        control_parts = [len(part) for part in _microbatches(control_rows, common_lengths, MICRO_TOKENS)]
        treatment_parts = [len(part) for part in _microbatches(treatment_rows, common_lengths, MICRO_TOKENS)]
        if control_parts != treatment_parts:
            raise AssertionError(f"Microbatch boundaries differ at update {step}")
        totals["microbatch_boundaries_equal_updates"] += 1
        if step < FIRST_BARRIER:
            first["microbatch_boundaries_equal_updates"] += 1
        for old_index, (kind, index) in zip(old_slots, treatment_batch):
            row = pools.python[index] if kind == "python" else pools.old[index]
            bucket = first if step < FIRST_BARRIER else None
            if kind == "python":
                totals["python_replaced_slots"] += 1
                totals["python_active_targets"] += row.active_positions
                families[row.source_family] += 1
                if bucket is not None:
                    bucket["python_replaced_slots"] += 1
                    bucket["python_active_targets"] += row.active_positions
                    first_families[row.source_family] += 1
            else:
                if index != old_index:
                    raise AssertionError(f"Unreplaced old row differs at update {step}")
                totals["same_unreplaced_old_slots"] += 1
                totals["old_active_targets"] += row.active_positions
                if bucket is not None:
                    bucket["same_unreplaced_old_slots"] += 1
                    bucket["old_active_targets"] += row.active_positions
    totals["updates"] = STEPS_PER_STAGE
    first["updates"] = FIRST_BARRIER

    def fraction(counts: Counter) -> float:
        return counts["python_active_targets"] / (counts["python_active_targets"] + counts["old_active_targets"])

    if totals["python_active_targets"] != treatment.python_active_positions or totals["old_active_targets"] != treatment.old_active_positions:
        raise AssertionError("Audited target counts differ from the admitted plan")
    return {
        "schema": "geocpu_hybrid_real_pool_plan_readonly_audit_v1",
        "inputs": {"v43_zip_sha256": file_sha(v43_zip), "converted_sha256": file_sha(converted),
                   "hybrid_data_sha256": file_sha(CANDIDATE / "hybrid_data.py"),
                   "source_pools_sha256": file_sha(CANDIDATE / "source_pools.py"),
                   "train_stage_sha256": file_sha(CANDIDATE / "train_hybrid_stage.py")},
        "pool_rows": {"old": len(pools.old), "python": len(pools.python)},
        "seed": SEED, "batch_size": BATCH_SIZE, "max_micro_tokens": MICRO_TOKENS,
        "full_1000_updates": {**totals, "python_actual_target_fraction": fraction(totals),
                              "python_rows_by_original_family": dict(families)},
        "first_250_updates": {**first, "python_actual_target_fraction": fraction(first),
                              "python_rows_by_original_family": dict(first_families)},
        "independent_replay_verification": "BYPASSED_FOR_READ_ONLY_AUDIT_ONLY",
        "training_updates": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("v43_zip", "converted", "output"):
        parser.add_argument("--" + name.replace("_", "-"), required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.v43_zip, args.converted)
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps({"full_fraction": result["full_1000_updates"]["python_actual_target_fraction"],
                      "first_fraction": result["first_250_updates"]["python_actual_target_fraction"],
                      "first_python_slots": result["first_250_updates"]["python_replaced_slots"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
