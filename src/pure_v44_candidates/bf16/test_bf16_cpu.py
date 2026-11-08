"""CPU-only smoke test of the isolated pure-v44 autocast candidate."""

from types import SimpleNamespace

import torch

from tir_engine_v44r1 import TorchModelAdapter


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(max_positions=8192)
        self.calls = []

    def prefill(self, ids, arm):
        self.calls.append(("prefill", ids.device.type, torch.is_autocast_enabled("cpu"), torch.is_grad_enabled()))
        return SimpleNamespace(last_logits=torch.tensor([[[0.0, 1.0]]]))

    def step_state(self, ids, state, compute_logits):
        self.calls.append(("advance", ids.device.type, torch.is_autocast_enabled("cpu"), torch.is_grad_enabled()))
        return state


if __name__ == "__main__":
    model = FakeModel()
    adapter = TorchModelAdapter(model)
    state = adapter.prefill([1, 2])
    assert adapter.next_token(state) == 1
    adapter.advance(state, 1)
    assert model.calls == [
        ("prefill", "cpu", False, False),
        ("advance", "cpu", False, False),
    ], model.calls
    print("PASS: CPU prefill/advance run without CUDA or gradients")
