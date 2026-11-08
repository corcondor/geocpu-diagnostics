"""Position-bounded v43/Python schedule with a measured SWE bridge share."""
from __future__ import annotations

import random
from typing import Sequence

from hybrid_data import EncodedExample, ExposurePlan, STAGE_PYTHON_CAPS

FAMILY_TARGETS = {"algebra": 0.40, "comparison": 0.40, "swe": 0.20}
FAMILY_CYCLE = ("algebra", "comparison", "algebra", "comparison", "swe")


def group(row: EncodedExample) -> str:
    if row.source_family == "Algebraic_Composite_Aligned":
        return "algebra"
    if row.source_family == "Comparative_Conditional_Aligned":
        return "comparison"
    if row.source_family.startswith("swe_alu_"):
        return "swe"
    raise ValueError("Unverified Python source family: " + row.source_family)


def plan_stratified(old: Sequence[EncodedExample], python: Sequence[EncodedExample],
                    *, cap: float, updates: int, batch_size: int, seed: int,
                    min_fraction_factor: float = 0.70) -> ExposurePlan:
    if cap not in (0.0, *STAGE_PYTHON_CAPS) or updates < 1 or batch_size < 1:
        raise ValueError("Unapproved Python target-position share")
    if not old or not python or any(row.kind != "v43" for row in old):
        raise ValueError("Both checked training pools are required")
    groups = {name: [] for name in FAMILY_TARGETS}
    for index, row in enumerate(python):
        if row.kind != "python":
            raise ValueError("A Python pool row has the wrong target mask kind")
        groups[group(row)].append(index)
    if any(not indices for indices in groups.values()):
        raise ValueError("The Python bridge lacks a required source family")
    old_rng = random.Random(seed)
    py_rng = random.Random(seed ^ 0x5A17C0DE)
    old_ids = list(range(len(old)))
    old_rng.shuffle(old_ids)
    for indices in groups.values():
        py_rng.shuffle(indices)
    old_cursor = 0
    py_cursor = {name: 0 for name in groups}
    old_active = py_active = 0
    family_active = {name: 0 for name in groups}
    next_python_slot = 0
    batches = []
    paired_old_batches = []
    for _ in range(updates):
        selected = []
        paired_old = []
        for _ in range(batch_size):
            if old_cursor == len(old_ids):
                old_rng.shuffle(old_ids)
                old_cursor = 0
            old_index = old_ids[old_cursor]
            old_cursor += 1
            paired_old.append(old_index)
            old_count = old[old_index].active_positions
            old_gap = abs(cap - py_active / (old_active + old_count + py_active))
            # Hold the next family slot until it fits.  Substituting an easier
            # short row here would starve the long SWE bridge at a 1% cap.
            name = FAMILY_CYCLE[next_python_slot % len(FAMILY_CYCLE)]
            indices = groups[name]
            cursor = py_cursor[name]
            if cursor == len(indices):
                py_rng.shuffle(indices)
                cursor = 0
                py_cursor[name] = 0
            index = indices[cursor]
            count = python[index].active_positions
            fraction = (py_active + count) / (old_active + py_active + count)
            if fraction <= cap and abs(cap - fraction) < old_gap:
                selected.append(("python", index))
                py_cursor[name] += 1
                next_python_slot += 1
                py_active += count
                family_active[name] += count
            else:
                selected.append(("v43", old_index))
                old_active += old_count
        batches.append(tuple(selected))
        paired_old_batches.append(tuple(paired_old))
    plan = ExposurePlan(cap, tuple(batches), tuple(paired_old_batches),
                        old_active, py_active, seed)
    if not cap * min_fraction_factor <= plan.measured_fraction <= cap:
        raise ValueError("Declared Python target-position share was not delivered")
    if cap and (sum(family_active.values()) != py_active
                or any(family_active[name] == 0 for name in FAMILY_TARGETS)):
        raise ValueError("A required Python bridge family was absent from the schedule")
    return plan
