import unittest
from types import SimpleNamespace

import torch
from torch import nn

from collect_stepwise_activations import (
    PairedContinuation, capture_branch, collect_discovery_pairs,
)


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm2 = nn.LayerNorm(3)
        self.ffn = nn.Sequential(nn.Linear(3, 4), nn.GELU(), nn.Linear(4, 3))

    def forward(self, x):
        return x + self.ffn(self.norm2(x))


class TinyStepModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(max_positions=128)
        self.embedding = nn.Embedding(8, 3)
        self.base = nn.Module()
        self.base.encoder = nn.ModuleList([TinyBlock()])
        self.base.decoder = nn.ModuleList([TinyBlock()])
        self.head = nn.Linear(3, 8)

    def prefill(self, ids, *, arm):
        if arm != "learned":
            raise ValueError("Wrong arm")
        self.base.encoder[0](self.embedding(ids))
        return SimpleNamespace(iterations=1, last_logits=torch.zeros(1, 1, 8))

    def step_state(self, token, state, *, compute_logits):
        if compute_logits is not True:
            raise ValueError("Wrong generation path")
        result = self.base.encoder[0](self.embedding(token))
        if int(token[0, 0]) == 5:
            result = self.base.decoder[0](result)
        return SimpleNamespace(iterations=1, last_logits=self.head(result))


class CaptureTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        self.model = TinyStepModel().eval()
        self.vocab = {"P": 4, "a": 5, "b": 6, "c": 7}

    def test_stepwise_capture_and_no_weight_change(self):
        original = self.model.base.encoder[0].ffn[2].weight.detach().clone()
        capture = capture_branch(
            self.model, self.vocab, "P", "aaaabbbb", "base.encoder.0",
            tail_window=4, expected_shape=(3, 4),
        )
        self.assertEqual(capture.suffix_tokens, 8)
        self.assertEqual(capture.site_calls, 8)
        self.assertEqual(capture.tail_site_calls, 4)
        self.assertEqual(tuple(capture.residual_mean.shape), (3,))
        self.assertEqual(tuple(capture.ffn_input_mean.shape), (4,))
        self.assertTrue(torch.equal(self.model.base.encoder[0].ffn[2].weight, original))
        self.assertEqual(len(self.model.base.encoder[0].ffn[2]._forward_hooks), 0)

    def test_discovery_pairs_and_proposal(self):
        rows = [PairedContinuation(
            f"train-{index}", "train", "P", "a", "bbbb", "cccc",
        ) for index in range(4)]
        pairs, diagnostics = collect_discovery_pairs(
            self.model, self.vocab, rows, "base.encoder.0",
            tail_window=4, expected_shape=(3, 4),
        )
        self.assertEqual(len(pairs), 4)
        self.assertEqual(len(diagnostics), 4)
        self.assertTrue(all(row["productive"]["tail_site_calls"] == 4 for row in diagnostics))
        # A real direction can be zero for an artificial symmetric fixture. The
        # collector still must return valid activation pairs and never update W.
        self.assertTrue(torch.isfinite(pairs[0].productive_residual).all())

    def test_invalid_source_and_vocabulary_rejected(self):
        rows = [PairedContinuation(f"eval-{i}", "heldout", "P", "", "aaaa", "bbbb")
                for i in range(4)]
        with self.assertRaises(ValueError):
            collect_discovery_pairs(self.model, self.vocab, rows, "base.encoder.0",
                                    tail_window=4, expected_shape=(3, 4))
        with self.assertRaises(ValueError):
            capture_branch(self.model, self.vocab, "P", "unknown", "base.encoder.0",
                           tail_window=4, expected_shape=(3, 4))

    def test_decoder_skipped_calls_are_measured_not_zero_filled(self):
        capture = capture_branch(
            self.model, self.vocab, "P", "abababab", "base.decoder.0",
            tail_window=4, min_tail_calls=2, expected_shape=(3, 4),
        )
        self.assertEqual(capture.site_calls, 4)
        self.assertEqual(capture.tail_site_calls, 2)
        with self.assertRaises(ValueError):
            capture_branch(self.model, self.vocab, "P", "abababab", "base.decoder.0",
                           tail_window=4, min_tail_calls=4, expected_shape=(3, 4))

    def test_refuses_training_mode_and_short_tail(self):
        self.model.train()
        with self.assertRaises(ValueError):
            capture_branch(self.model, self.vocab, "P", "aaaa", "base.encoder.0",
                           tail_window=4, expected_shape=(3, 4))
        self.model.eval()
        with self.assertRaises(ValueError):
            capture_branch(self.model, self.vocab, "P", "aa", "base.encoder.0",
                           tail_window=4, expected_shape=(3, 4))


if __name__ == "__main__":
    unittest.main()
