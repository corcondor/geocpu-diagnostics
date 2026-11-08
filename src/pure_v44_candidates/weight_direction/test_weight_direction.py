import unittest

import torch
from torch import nn

from weight_direction import (
    GEOCPU_FFN_SHAPE, PairedActivation, Proposal, assert_disjoint, edited_weight,
    evaluate_heldout, orthogonal_null, propose_from_pairs, record_generation,
    resolve_ffn_output, temporary_output_hook,
)


class TinyBlock(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.ffn = nn.Sequential(nn.Linear(4, 4), nn.GELU(), nn.Linear(4, 3))
        with torch.no_grad():
            self.ffn[2].weight.copy_(weight)


class TinyGeoCPU(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.base = nn.Module()
        self.base.encoder = nn.ModuleList([TinyBlock(weight)])


def fixture():
    weight = torch.tensor([[1., 4., 3., 0.], [2., 1., 0., 2.], [0., 1., 1., 1.]])
    pairs = [PairedActivation(
        case_id=f"practice-{i}", source="practice",
        productive_residual=torch.zeros(3),
        repetitive_residual=torch.tensor([1., 0., 0.]),
        productive_ffn_input=torch.zeros(4),
        repetitive_ffn_input=torch.tensor([0., 2., 1., 0.]),
    ) for i in range(4)]
    proposal = propose_from_pairs(weight, "base.encoder.0", pairs, top_k=2)
    return weight, pairs, proposal


class WeightDirectionTests(unittest.TestCase):
    def test_real_shape_constant_and_discovery_column_ranking(self):
        self.assertEqual(GEOCPU_FFN_SHAPE, (768, 3072))
        weight, _pairs, proposal = fixture()
        self.assertEqual(proposal.columns, (1, 2))
        self.assertTrue(torch.allclose(proposal.direction, torch.tensor([1., 0., 0.])))
        expected = weight.clone()
        expected[0, 1:3] = 0
        self.assertTrue(torch.allclose(edited_weight(weight, proposal), expected))
        self.assertTrue(torch.equal(weight, fixture()[0]))

    def test_zero_dose_and_orthogonal_norm_matched(self):
        from dataclasses import replace
        weight, _pairs, proposal = fixture()
        zero = replace(proposal, alpha=0.0)
        self.assertTrue(torch.equal(edited_weight(weight, zero), weight))
        null = orthogonal_null(weight, proposal)
        self.assertAlmostEqual(float(null.direction @ proposal.direction), 0.0, places=6)
        target_size = (edited_weight(weight, proposal) - weight).norm()
        null_size = (edited_weight(weight, null) - weight).norm()
        self.assertTrue(torch.allclose(target_size, null_size, atol=1e-6))

    def test_hook_matches_materialized_edit_and_restores_original(self):
        weight, _pairs, proposal = fixture()
        model = TinyGeoCPU(weight).eval()
        output = resolve_ffn_output(model, "base.encoder.0", expected_shape=(3, 4))
        x = torch.tensor([[1., 2., 3., 4.]])
        original_weight = output.weight.detach().clone()
        with torch.inference_mode():
            original = output(x)
            with temporary_output_hook(output, proposal):
                actual = output(x)
                expected = torch.nn.functional.linear(x, edited_weight(original_weight, proposal), output.bias)
                self.assertTrue(torch.allclose(actual, expected, atol=1e-6))
            self.assertTrue(torch.equal(output(x), original))
        self.assertTrue(torch.equal(output.weight, original_weight))
        with self.assertRaises(ValueError):
            resolve_ffn_output(model, "base.encoder.0")  # Real GeoCPU shape is required by default.

    def test_heldout_separation_and_original_answer_record(self):
        assert_disjoint(["practice-1"], ["heldout-1"])
        with self.assertRaises(ValueError):
            assert_disjoint(["practice-1"], ["practice-1"])
        generation = {"output_text": "abcdabcdabcdabcd", "model_generated_text": "abcdabcdabcdabcd",
                      "stop_reason": "new_token_limit", "ended_with_eos": False,
                      "sampled_token_count": 16, "tool_invocations": 0}
        record = record_generation("heldout-1", "prompt", "reference", generation, False)
        self.assertEqual(record["raw_answer"], generation["output_text"])
        self.assertEqual(record["repetition"]["span_chars"], 16)
        self.assertFalse(record["correct"])
        hash_generation = dict(generation, output_text="#" * 400,
                               model_generated_text="#" * 400, sampled_token_count=400)
        self.assertEqual(record_generation("heldout-2", "prompt", "reference",
                                           hash_generation, False)["repetition"]["period_chars"], 1)

    def test_rejects_heldout_discovery_and_invalid_contrast(self):
        weight, pairs, proposal = fixture()
        bad = [PairedActivation(**{**p.__dict__, "source": "heldout"}) for p in pairs]
        with self.assertRaises(ValueError):
            propose_from_pairs(weight, "base.encoder.0", bad, top_k=2)
        zero = [PairedActivation(**{**p.__dict__, "repetitive_residual": torch.zeros(3)}) for p in pairs]
        with self.assertRaises(ValueError):
            propose_from_pairs(weight, "base.encoder.0", zero, top_k=2)

    def test_optional_heldout_runner_records_all_four_arms(self):
        # This is still a synthetic module; it does not load the 247M model.
        model = nn.Module()
        model.base = nn.Module()
        model.base.encoder = nn.ModuleList([nn.Module()])
        block = model.base.encoder[0]
        block.ffn = nn.Sequential(nn.Linear(1, 1), nn.GELU(), nn.Linear(3072, 768))
        with torch.no_grad():
            block.ffn[2].weight.normal_(0, 0.01)
        direction = torch.zeros(768)
        direction[0] = 1
        proposal = Proposal("base.encoder.0", direction, (0, 1), 1.0,
                            GEOCPU_FFN_SHAPE, ("practice-1",), torch.zeros(3072))

        def generate(model, _prompt):
            out = model.base.encoder[0].ffn[2](torch.ones(1, 3072))
            text = "a" if out[0, 0] >= 0 else "b"
            return {"output_text": text, "model_generated_text": text,
                    "stop_reason": "eos", "ended_with_eos": True,
                    "sampled_token_count": 1, "tool_invocations": 0}

        rows = evaluate_heldout(model, proposal,
                                [{"case_id": "heldout-1", "prompt": "prompt", "reference": "a"}],
                                generate, lambda case, result: result["output_text"] == case["reference"])
        arms = rows[0]["arms"]
        self.assertEqual(set(arms), {"baseline", "zero_dose", "orthogonal_null", "candidate"})
        self.assertEqual(arms["baseline"]["raw_answer"], arms["zero_dose"]["raw_answer"])
        self.assertEqual(arms["candidate"]["stop_reason"], "eos")


if __name__ == "__main__":
    unittest.main()
