"""CPU-only checks of the explicit fp16 overflow and exact-resume path."""
import json
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from train_hybrid_stage import (_finish_fp16_attempt, _load_stage_checkpoint,
                                _needs_partial_save, _retry_same_slot,
                                _save_checkpoint)


class _Model:
    def parameters(self):
        return []

    def state_dict(self):
        return {"weight": 7}


class _Optimizer:
    def __init__(self):
        self.steps = 0

    def step(self):
        self.steps += 1

    def state_dict(self):
        return {"steps": self.steps}


class _Torch:
    def __init__(self, norms=()):
        self.norms = iter(norms)
        self.current_norm = None
        self.nn = SimpleNamespace(utils=SimpleNamespace(clip_grad_norm_=self._clip))

    def _clip(self, parameters, maximum, error_if_nonfinite):
        self.current_norm = next(self.norms)
        return self.current_norm

    @staticmethod
    def isfinite(value):
        return math.isfinite(value)

    @staticmethod
    def save(payload, path):
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    @staticmethod
    def load(path, map_location, weights_only):
        return json.loads(Path(path).read_text(encoding="utf-8"))


class _Scaler:
    def __init__(self, torch, scale=1024):
        self.torch = torch
        self.scale_value = scale

    def unscale_(self, optimizer):
        pass

    def get_scale(self):
        return self.scale_value

    def step(self, optimizer):
        if math.isfinite(self.torch.current_norm):
            optimizer.step()

    def update(self, new_scale=None):
        if new_scale is not None:
            self.scale_value = new_scale
        elif not math.isfinite(self.torch.current_norm):
            self.scale_value /= 2

    def state_dict(self):
        return {"scale": self.scale_value}


class FP16RetryTests(unittest.TestCase):
    def test_infinite_aggregate_norm_never_calls_optimizer_step(self):
        class MisleadingScaler(_Scaler):
            # Simulates GradScaler seeing finite individual gradients even
            # though their float32 aggregate norm overflowed.
            def step(self, optimizer):
                optimizer.step()

        torch = _Torch([float("inf")])
        optimizer = _Optimizer()
        scaler = MisleadingScaler(torch)
        accepted = _finish_fp16_attempt(torch, _Model(), optimizer, scaler)
        self.assertFalse(accepted)
        self.assertEqual(optimizer.steps, 0)
        self.assertEqual(scaler.get_scale(), 512)

    def test_scale_down_after_an_earlier_checkpoint_needs_a_new_receipt(self):
        self.assertTrue(_needs_partial_save(50, 3, (50, 0)))
        self.assertTrue(_needs_partial_save(0, 3, None))
        self.assertFalse(_needs_partial_save(50, 3, (50, 3)))
        self.assertFalse(_needs_partial_save(250, 3, (250, 0)))

    def test_nonfinite_gradient_retries_the_same_slot_before_one_real_step(self):
        torch = _Torch([float("inf"), 0.5])
        optimizer = _Optimizer()
        scaler = _Scaler(torch)
        restored = []
        accepted, skipped = _retry_same_slot(
            attempt=lambda: _finish_fp16_attempt(torch, _Model(), optimizer, scaler),
            restore_after_skip=lambda: restored.append("same-slot"),
            can_continue=lambda: True)
        self.assertEqual((accepted, skipped), (True, 1))
        self.assertEqual(restored, ["same-slot"])
        self.assertEqual(optimizer.steps, 1)
        self.assertEqual(scaler.get_scale(), 512)

    def test_three_skips_keep_zero_accepted_and_checkpoint_the_reduced_scale(self):
        torch = _Torch([float("inf")] * 3)
        optimizer = _Optimizer()
        scaler = _Scaler(torch)
        accepted, skipped = _retry_same_slot(
            attempt=lambda: _finish_fp16_attempt(torch, _Model(), optimizer, scaler),
            restore_after_skip=lambda: None, can_continue=lambda: True)
        self.assertEqual((accepted, skipped), (False, 3))
        self.assertEqual(optimizer.steps, 0)
        self.assertEqual(scaler.get_scale(), 128)
        binding = {"schema": "geocpu_v43_python_gradual_stage_v1",
                   "precision_profile": "train_fp16_gradscaler_v1"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "update_0000_claim_unique.pt"
            with patch("train_hybrid_stage._rng_capture", return_value={"fixed_rng": 1}):
                receipt = _save_checkpoint(path, model=_Model(), optimizer=optimizer,
                    torch=torch, np=None, binding=binding, update=0,
                    old_active=0, python_active=0, status="PARTIAL", scaler=scaler,
                    fp16_skipped_attempts=skipped)
            saved = _load_stage_checkpoint(path, torch=torch, expected_binding=binding)
            self.assertEqual(saved["accepted_updates"], 0)
            self.assertEqual(saved["fp16_skipped_attempts"], 3)
            self.assertEqual(saved["scaler_state_dict"]["scale"], 128)
            self.assertEqual(receipt["fp16_skipped_attempts"], 3)


if __name__ == "__main__":
    unittest.main()
