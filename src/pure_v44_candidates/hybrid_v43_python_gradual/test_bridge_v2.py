"""Admission checks for the new SWE bridge and position-level mixture."""
from __future__ import annotations

from collections import Counter
import unittest

from hybrid_data import EncodedExample
from stratified_exposure_v2 import group, plan_stratified


def example(kind: str, family: str, active: int) -> EncodedExample:
    return EncodedExample((1,) * (active + 2), 1, (), active,
                          kind, "test", family, False)


class StratifiedExposureTest(unittest.TestCase):
    def setUp(self):
        self.old = [example("v43", "retained_public", 100) for _ in range(1_000)]
        self.python = [example("python", family, 110) for family in (
            "Algebraic_Composite_Aligned", "Comparative_Conditional_Aligned",
            "swe_alu_accumulate_sum") for _ in range(100)]

    def test_old_slots_match_zero_percent_control(self):
        control = plan_stratified(self.old, self.python, cap=0.0,
                                  updates=250, batch_size=64, seed=20261104)
        bridge = plan_stratified(self.old, self.python, cap=0.01,
                                 updates=250, batch_size=64, seed=20261104)
        self.assertEqual(control.paired_old_batches, bridge.paired_old_batches)
        self.assertEqual(control.python_active_positions, 0)
        self.assertLessEqual(bridge.measured_fraction, 0.01)
        self.assertGreaterEqual(bridge.measured_fraction, 0.007)

    def test_swe_is_present_in_first_barrier_without_dominating(self):
        bridge = plan_stratified(self.old, self.python, cap=0.01,
                                 updates=250, batch_size=64, seed=20261104)
        counts = Counter(group(self.python[index]) for batch in bridge.batches
                         for kind, index in batch if kind == "python")
        self.assertGreater(counts["swe"], 0)
        self.assertGreater(counts["algebra"], counts["swe"])
        self.assertGreater(counts["comparison"], counts["swe"])

    def test_unknown_python_family_is_rejected(self):
        self.python[0] = example("python", "unverified", 110)
        with self.assertRaises(ValueError):
            plan_stratified(self.old, self.python, cap=0.01,
                            updates=250, batch_size=64, seed=20261104)


if __name__ == "__main__":
    unittest.main()
