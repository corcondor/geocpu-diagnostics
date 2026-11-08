"""CPU-only contract tests for T4 full-budget runtime patching."""
import ast
import hashlib
from pathlib import Path
import unittest

from t4_runtime_patch import patch_runtime

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/"research/v47_practical_20261107/runtime_practical_v1.py"
EXPECTED_SOURCE_SHA256="fc196cf9b7d22fae7f0128b5b0b1bd1e8b7630d6f9e27ba0fd310d0db49605c2"

class T4RuntimePatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw=SOURCE.read_bytes()
        assert hashlib.sha256(raw).hexdigest()==EXPECTED_SOURCE_SHA256, "v47 source changed"
        cls.source=raw.decode("utf-8").replace('"practical_multistep_v1"','"v48_additive_only"')
        cls.patched=patch_runtime(cls.source)
        cls.tree=ast.parse(cls.patched)

    def test_compiles(self):
        compile(self.patched,"runtime_v48_t4.py","exec")

    def test_no_training_step_or_batch_budget_reduction(self):
        self.assertIn("max_tokens_per_micro = int(args.step_tokens)",self.patched)
        self.assertIn("max_micro_batch = min(96, args.batch_size)",self.patched)
        self.assertIn("max_tokens=max_tokens_per_micro",self.patched)
        self.assertIn("step_tokens",self.patched)
        self.assertIn("torch.cuda.empty_cache()",self.patched)

    def test_gradient_scaler_and_no_skipped_optimizer_step(self):
        self.assertIn("torch.float16",self.patched)
        self.assertIn("torch.amp.GradScaler('cuda'",self.patched)
        self.assertIn("scaler.scale(loss).backward()",self.patched)
        self.assertIn("scaler.unscale_(optimizer)",self.patched)
        self.assertIn("scaler.step(optimizer)",self.patched)
        self.assertIn("scaler.update()",self.patched)
        self.assertIn("re-running full SAME batch",self.patched)
        self.assertIn("do not record exposure",self.patched)

    def test_checkpoint_does_not_change_parameter_names_or_count(self):
        self.assertIn("use_reentrant=False",self.patched)
        self.assertIn("preserve_rng_state=True",self.patched)
        self.assertIn("model.fine_decoder",self.patched)
        self.assertIn("model.base.encoder",self.patched)
        self.assertIn("model.base.decoder",self.patched)
        self.assertIn("activation checkpointing",self.patched)

    def test_only_explicit_a100_to_t4_transition(self):
        self.assertIn("args.new_dataset_sampler and 'T4' in current_gpu",self.patched)
        self.assertIn("'A100' in state['gpu_name']",self.patched)
        self.assertIn("GPU model changed outside declared",self.patched)

    def test_preserves_reference_loss_and_kl(self):
        for needle in (
            "compute_cut_ce_loss(model, out.fine_states",
            "distill_kl(model, teacher, out.fine_states",
            "loss = loss * weight",
            "n_ce = max(1, int(act.sum()))",
            "phase_source_supervised[origin] = phase_source_supervised.get(origin, 0) + tokens",
        ):
            self.assertIn(needle,self.source)
            self.assertIn(needle,self.patched)

    def test_rejects_modified_source(self):
        with self.assertRaises(ValueError):
            patch_runtime(self.source.replace("loss.backward()", "loss.backward_renamed()"))

if __name__=="__main__":
    unittest.main()
