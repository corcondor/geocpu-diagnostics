"""Small CPU checks; no GeoCPU checkpoint or GPU is opened."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from adapter import (AdapterSpec, adapter_parameters, adapter_state, file_sha256,
                     install_adapters, resume_adapter_checkpoint, save_adapter_checkpoint)
from train_stage import _relative_adapter_norms


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ffn = nn.Sequential(nn.Linear(4, 7), nn.GELU(), nn.Linear(7, 4))

    def forward(self, x):
        return x + self.ffn(x)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fine_decoder = nn.ModuleList([Block(), Block()])
        self.base = nn.Module()
        self.base.head = nn.Linear(4, 5)

    def forward(self, x):
        for block in self.fine_decoder:
            x = block(x)
        return self.base.head(x)


class AdapterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.parent = TinyModel()
        self.x = torch.randn(3, 4)
        self.spec = AdapterSpec(rank=2, alpha=2.0)

    def test_identity_and_only_adapter_gradients(self):
        baseline = self.parent(self.x).detach().clone()
        saved_parent = {k: v.detach().clone() for k, v in self.parent.state_dict().items()}
        names = install_adapters(self.parent, self.spec)
        self.assertTrue(torch.equal(baseline, self.parent(self.x)))
        optimizer = torch.optim.AdamW(adapter_parameters(self.parent, names), lr=0.01)
        loss = self.parent(self.x).square().mean()
        loss.backward()
        self.assertTrue(all(parameter.grad is None for name, parameter in self.parent.named_parameters()
                            if name.endswith("parent.weight") or name.endswith("parent.bias")))
        optimizer.step()
        self.assertFalse(torch.equal(baseline, self.parent(self.x)))
        for name, value in saved_parent.items():
            wrapped_name = name.replace("base.head.", "base.head.parent.")
            for i in range(2):
                wrapped_name = wrapped_name.replace(f"fine_decoder.{i}.ffn.2.",
                                                    f"fine_decoder.{i}.ffn.2.parent.")
            self.assertTrue(torch.equal(value, self.parent.state_dict()[wrapped_name]), name)

    def test_save_resume_lineage_and_optimizer(self):
        trained = copy.deepcopy(self.parent)
        names = install_adapters(trained, self.spec)
        optimizer = torch.optim.AdamW(adapter_parameters(trained, names), lr=0.01)
        trained(self.x).square().mean().backward()
        optimizer.step()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapter.pt"
            sha = save_adapter_checkpoint(path, model=trained, names=names, spec=self.spec,
                                          parent_sha256="a" * 64, train_manifest_sha256="b" * 64,
                                          optimizer=optimizer, step=1)
            self.assertEqual(sha, file_sha256(path))
            restored = copy.deepcopy(self.parent)
            restored_names = install_adapters(restored, self.spec)
            resumed_optimizer = torch.optim.AdamW(adapter_parameters(restored, restored_names), lr=0.01)
            with self.assertRaises(ValueError):
                resume_adapter_checkpoint(path, expected_sha256=sha,
                                          expected_parent_sha256="c" * 64,
                                          expected_train_manifest_sha256="b" * 64,
                                          model=restored, names=restored_names,
                                          spec=self.spec, optimizer=resumed_optimizer)
            self.assertNotEqual(adapter_state(trained, names)["base.head.up.weight"].tolist(),
                                adapter_state(restored, restored_names)["base.head.up.weight"].tolist())
            receipt = resume_adapter_checkpoint(path, expected_sha256=sha,
                                                expected_parent_sha256="a" * 64,
                                                expected_train_manifest_sha256="b" * 64,
                                                model=restored, names=restored_names,
                                                spec=self.spec, optimizer=resumed_optimizer)
            self.assertEqual(receipt["step"], 1)
            self.assertTrue(torch.equal(trained(self.x), restored(self.x)))
            self.assertEqual(len(resumed_optimizer.state_dict()["state"]),
                             len(optimizer.state_dict()["state"]))

    def test_effective_weight_norm_matches_explicit_product(self):
        names = install_adapters(self.parent, self.spec)
        norms = {name: self.parent.get_submodule(name).parent.weight.detach().float().norm()
                 for name in names}
        with torch.no_grad():
            for name in names:
                self.parent.get_submodule(name).up.weight.fill_(0.05)
        compact = _relative_adapter_norms(self.parent, names, norms)
        for name in names:
            module = self.parent.get_submodule(name)
            explicit = (module.scale * (module.up.weight @ module.down.weight)).float().norm() / norms[name]
            self.assertAlmostEqual(compact[name], explicit.item(), places=6)


if __name__ == "__main__":
    unittest.main()
