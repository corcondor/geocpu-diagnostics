"""CPU-only checkpoint fixtures; no GeoCPU model or training data is used."""
import json
import random
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

from resume_v44fix import (ResumableTokenBucketBatchSampler, capture_rng,
                           load_checkpoint, receipt_path, restore_rng,
                           save_checkpoint, verify_checkpoint)
from runtime_cotraining_v41_247M import DynamicTokenBucketBatchSampler


class TinyDataset:
    def __init__(self, count=64):
        self.samples = [([0] * (20 + (i * 17) % 109), 1, ()) for i in range(count)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class ExactResumeTests(unittest.TestCase):
    def test_original_sampler_policy_and_next_batch_after_resume(self):
        dataset = TinyDataset()
        random.seed(812)
        original = DynamicTokenBucketBatchSampler(dataset, max_batch_size=8, max_tokens=2048,
                                                  megabucket_mult=4)
        original_batches = list(iter(original))
        random.seed(812)
        sampler = ResumableTokenBucketBatchSampler(dataset, max_batch_size=8, max_tokens=2048,
                                                    megabucket_mult=4)
        self.assertEqual(list(iter(sampler)), original_batches)

        random.seed(614)
        reference = ResumableTokenBucketBatchSampler(dataset, max_batch_size=8, max_tokens=2048,
                                                      megabucket_mult=4)
        iterator = iter(reference)
        for _ in range(3):
            next(iterator)
        saved = reference.state_dict()
        rng = random.getstate()
        remaining = list(iterator)
        next_epoch = list(iter(reference))

        random.seed(999)
        resumed = ResumableTokenBucketBatchSampler(dataset, max_batch_size=8, max_tokens=2048,
                                                    megabucket_mult=4)
        resumed.load_state_dict(saved)
        random.setstate(rng)
        self.assertEqual(list(iter(resumed)), remaining)
        self.assertEqual(list(iter(resumed)), next_epoch)

    def test_rng_state_roundtrip(self):
        random.seed(73)
        np.random.seed(73)
        torch.manual_seed(73)
        state = capture_rng()
        expected = (random.random(), float(np.random.random()), torch.rand(1).item())
        random.random(); np.random.random(); torch.rand(1)
        restore_rng(state)
        observed = (random.random(), float(np.random.random()), torch.rand(1).item())
        self.assertEqual(expected, observed)

    def test_epoch_boundary_resume_preserves_dataloader_rng_consumption(self):
        dataset = TinyDataset()
        random.seed(517)
        torch.manual_seed(517)
        reference = ResumableTokenBucketBatchSampler(dataset, max_batch_size=8, max_tokens=2048,
                                                      megabucket_mult=4)
        loader = DataLoader(dataset, batch_sampler=reference, collate_fn=lambda batch: batch)
        iterator = iter(loader)
        for _ in range(len(reference.batches)):
            next(iterator)
        saved_sampler = reference.state_dict()
        saved_rng = capture_rng()
        with self.assertRaises(StopIteration):
            next(iterator)
        expected = next(iter(loader))
        expected_torch_rng = torch.get_rng_state().clone()

        random.seed(99)
        torch.manual_seed(99)
        resumed = ResumableTokenBucketBatchSampler(dataset, max_batch_size=8, max_tokens=2048,
                                                    megabucket_mult=4)
        resumed.load_state_dict(saved_sampler)
        new_loader = DataLoader(dataset, batch_sampler=resumed, collate_fn=lambda batch: batch)
        new_iterator = iter(new_loader)
        restore_rng(saved_rng)  # Runtime restores after first DataLoader iterator setup.
        with self.assertRaises(StopIteration):
            next(new_iterator)
        actual = next(iter(new_loader))
        self.assertEqual(actual, expected)
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_torch_rng))

    def test_checkpoint_roundtrip_integrity_and_no_overwrite(self):
        random.seed(13)
        np.random.seed(13)
        torch.manual_seed(13)
        dataset = TinyDataset()
        sampler = ResumableTokenBucketBatchSampler(dataset, max_batch_size=8, max_tokens=2048,
                                                    megabucket_mult=4)
        next(iter(sampler))
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        for parameter in model.parameters():
            optimizer.state[parameter] = {
                "step": torch.tensor(1.), "exp_avg": torch.full_like(parameter, 0.1),
                "exp_avg_sq": torch.full_like(parameter, 0.01)}
        saved_weights = {key: tensor.detach().clone() for key, tensor in model.state_dict().items()}
        saved_sampler = sampler.state_dict()
        binding = {"profile": "tiny_fixture", "source": "static"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint_step_15001.pt"
            receipt = save_checkpoint(path, step=15001, model=model, optimizer=optimizer,
                                      sampler=sampler, binding=binding, config={}, meta={})
            self.assertEqual(verify_checkpoint(path, binding, 15001), receipt)
            self.assertTrue(receipt_path(path).is_file())
            with self.assertRaises(FileExistsError):
                save_checkpoint(path, step=15001, model=model, optimizer=optimizer,
                                sampler=sampler, binding=binding, config={}, meta={})
            for parameter in model.parameters():
                parameter.data.zero_()
            changed = ResumableTokenBucketBatchSampler(dataset, max_batch_size=8, max_tokens=2048,
                                                        megabucket_mult=4)
            new_optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
            step, rng = load_checkpoint(path, expected_binding=binding, model=model,
                                        optimizer=new_optimizer, sampler=changed)
            self.assertEqual(step, 15001)
            self.assertEqual(changed.state_dict(), saved_sampler)
            self.assertGreater(len(new_optimizer.state), 0)
            for key, value in saved_weights.items():
                self.assertTrue(torch.equal(model.state_dict()[key], value))
            restore_rng(rng)
            self.assertEqual(set(rng), {"torch_cpu", "torch_cuda", "python", "numpy"})
            with path.open("ab") as stream:
                stream.write(b"tamper")
            with self.assertRaises(ValueError):
                verify_checkpoint(path, binding, 15001)


if __name__ == "__main__":
    unittest.main()
