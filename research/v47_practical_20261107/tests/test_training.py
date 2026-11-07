"""CPU/AST checks of the isolated training preparation, never GPU experiments."""
from __future__ import annotations
import ast
import copy
from contextlib import redirect_stderr
import hashlib
import io
import json
from pathlib import Path
import random
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_training import (NEW_GROUP, manifest_train, normalization_table,
                              original_classes, patched_runtime, prepare_dataset,
                              preview_plan, sha, split_sequence, validate_new_row)
from train_candidate import active_work, build_command, fixed_heldouts, parse_args

PACKAGE = Path(__file__).resolve().parents[1]
RUNTIME = PACKAGE / "frozen/continue/training/runtime_30500.py"
BASE = PACKAGE / "frozen/r6/runtime_cotraining_v41_247M.py"


class TrainingPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.normalizer = self.root / "normalizer.py"
        self.normalizer.write_text('_BASE_NORM_DICT = {"−": "-"}\n', encoding="utf-8")
        self.meta = self.root / "meta.json"
        self.prompt = "Question: calculate invoice sum.\nThink:"
        target = '\n```python\nprint(sum([1, 2]))\n```\n```output\n3\n```\n答え: 3 円。合計です。'
        full = self.prompt + target
        a = full.index("```output")
        b = full.index("```\n答え", a) + 4
        self.newrow = {"id": "training_1", "prompt": self.prompt, "full_text": full,
                       "target": target, "loss_mask_spans": [[a, b]],
                       "split": "train", "source": "authored", "license": "CC0",
                       "_v46_group": "practical_python_multistep_v1",
                       "reference_final_answer": "3 円", "final_answer_line": "答え: 3 円。合計です。"}
        self.parent = self.root / "parent.jsonl"
        parent_rows = [dict(id=f"old_{i}", prompt=f"Old{i}: ", full_text=f"Old{i}: " + "retained skill. " * 20,
                            _v46_group="original_parent") for i in range(48)]
        self.parent.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in parent_rows), encoding="utf-8")
        vocab = set(full + self.parent.read_text() + "答え: ")
        self.meta.write_text(json.dumps({"vocab": {char: i + 4 for i, char in enumerate(sorted(vocab))}}), encoding="utf-8")
        self.data = self.root / "new.jsonl"
        self.data.write_text(json.dumps(self.newrow, ensure_ascii=False) + "\n", encoding="utf-8")
        self.manifest = self.root / "new_manifest.json"
        self.manifest.write_text(json.dumps({"files": {"train": {"sha256": sha(self.data), "rows": 1}}}))
        self.holdout = self.root / "heldout.jsonl"
        self.holdout.write_text(json.dumps({"id": "heldout", "prompt": "Different heldout question"}) + "\n")

    def tearDown(self):
        self.temp.cleanup()

    def prepare(self, variant="multistep", output="composed.jsonl"):
        return prepare_dataset(self.parent, [(self.data, self.manifest)], meta=self.meta,
            normalizer_source=self.normalizer, heldouts=[self.holdout], output=self.root / output,
            manifest_output=self.root / (output + ".manifest.json"), variant=variant,
            expected_parent_sha=sha(self.parent), expected_parent_rows=48)

    def test_parent_bytes_are_preserved_and_new_groups_are_teacher_exempt(self):
        before = self.parent.read_bytes()
        report = self.prepare()
        composed = (self.root / "composed.jsonl").read_bytes()
        self.assertTrue(composed.startswith(before))
        self.assertEqual(self.parent.read_bytes(), before)
        self.assertEqual((report["parent_rows_preserved_byte_exact"], report["new_rows"]), (48, 1))
        appended = json.loads(composed[len(before):])
        self.assertEqual(appended["_v46_group"], NEW_GROUP)
        self.assertEqual(appended["_practical_source"], "practical_python_multistep_v1")
        self.assertEqual(appended["loss_mask_spans"], self.newrow["loss_mask_spans"])

    def test_answer_only_keeps_question_but_has_no_observation_targets(self):
        self.prepare("answer_only")
        appended = json.loads((self.root / "composed.jsonl").read_text().splitlines()[-1])
        self.assertEqual(appended["prompt"], self.prompt)
        self.assertEqual(appended["full_text"], self.prompt + "\n" + self.newrow["final_answer_line"])
        self.assertEqual(appended["loss_mask_spans"], [])
        self.assertNotIn("```output", appended["full_text"])

    def test_replay_control_is_byte_identical_parent(self):
        report = self.prepare("replay_only")
        self.assertEqual(report["new_rows"], 0)
        self.assertEqual((self.root / "composed.jsonl").read_bytes(), self.parent.read_bytes())

    def test_eval_prompt_leakage_and_missing_mask_are_rejected(self):
        self.holdout.write_text(json.dumps({"prompt": self.prompt}) + "\n")
        with self.assertRaisesRegex(ValueError, "overlaps heldout"):
            self.prepare()
        row = dict(self.newrow, loss_mask_spans=[])
        normalize = normalization_table(self.normalizer)
        vocab = set(json.loads(self.meta.read_text())["vocab"])
        with self.assertRaisesRegex(ValueError, "unmasked runtime observation"):
            validate_new_row(row, vocab, normalize, set(), set())

    def test_changed_manifest_bytes_are_rejected(self):
        self.data.write_text(self.data.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "missing or changed"):
            manifest_train(self.data, self.manifest)

    def test_patched_forward_kl_and_optimizer_retry_are_unchanged(self):
        original = RUNTIME.read_text(encoding="utf-8")
        updated = patched_runtime(original)
        first, second = ast.parse(original), ast.parse(updated)
        for name in ("forward_backward_chunked", "step_with_retry", "distill_kl", "phase_lr"):
            a = next(n for n in first.body if isinstance(n, ast.FunctionDef) and n.name == name)
            b = next(n for n in second.body if isinstance(n, ast.FunctionDef) and n.name == name)
            self.assertEqual(ast.dump(a, include_attributes=False), ast.dump(b, include_attributes=False), name)
        self.assertIn('str(item.get("_v46_group", "legacy")) != "practical_multistep_v1"', updated)
        self.assertIn('p_.requires_grad_(False)', updated)
        self.assertIn('planned_replay_share < args.min_replay_share', updated)
        self.assertIn('log_interval = 10', updated)
        self.assertIn('torch.cuda.set_rng_state_all(state["torch_cuda"])', updated)
        self.assertIn('optimizer.load_state_dict(oc["optimizer_state_dict"])', updated)
        with self.assertRaisesRegex(ValueError, "pinned"):
            patched_runtime(original + "\n")

    def test_cpu_sequence_count_matches_actual_original_chunk_algorithm(self):
        tree = ast.parse(RUNTIME.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ToolMaskedDataset")
        func = copy.deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "split_sequence"))
        func.decorator_list = []
        env = {}
        exec(compile(ast.Module(body=[func], type_ignores=[]), "original_split", "exec"), env)
        original = env["split_sequence"]
        for length, pl, spans in ((100, 12, [(30, 44)]), (5000, 300, [(1000, 1300), (3000, 3200)]),
                                   (2048, 200, [(1800, 2000)]), (30, 1, [])):
            expected = [(len(chunk), p, s) for chunk, p, s in original(range(length), pl, spans, 2048)]
            self.assertEqual(list(split_sequence(range(length), pl, spans)), expected)

    def test_actual_copied_dataset_marks_every_old_group_for_teacher_and_new_rows_off(self):
        self.prepare()
        tree = ast.parse(patched_runtime(RUNTIME.read_text()))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ToolMaskedDataset")
        helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "practical_supervised_count")
        env = {"Dataset": object, "time": time, "json": json, "now_iso": lambda: "cpu-check",
               "canonical_normalize_text": lambda s: s.translate(normalization_table(self.normalizer)),
               "print": lambda *a, **k: None}
        exec(compile(ast.Module(body=[helper, cls], type_ignores=[]), "candidate_dataset_only", "exec"), env)
        vocab = json.loads(self.meta.read_text())["vocab"]
        dataset = env["ToolMaskedDataset"](self.root / "composed.jsonl", vocab)
        self.assertEqual(len(dataset.samples), 49)
        self.assertTrue(all(s[3] for s in dataset.samples[:48]))
        self.assertFalse(dataset.samples[-1][3])
        self.assertEqual(dataset.origins[-1], "practical_python_multistep_v1")
        self.assertEqual(dataset.stats["rows_with_unaligned_spans_dropped"], 0)

    def test_preview_preserves_global_rng_and_rejects_insufficient_exposure(self):
        self.prepare()
        kwargs = dict(meta=self.meta, normalizer_source=self.normalizer, base_sampler_source=BASE,
                      runtime_source=RUNTIME, steps=4, repeat=1)
        before = random.getstate()
        report = preview_plan(self.root / "composed.jsonl", min_new=0, **kwargs)
        self.assertEqual(random.getstate(), before)
        self.assertTrue(report["passed"])
        self.assertGreaterEqual(report["replay_supervised_share"], .75)
        with self.assertRaisesRegex(ValueError, "pilot preview rejected"):
            preview_plan(self.root / "composed.jsonl", min_new=1000, **kwargs)

    def test_live_process_scan_does_not_signal_or_match_bash_command_text(self):
        proc = self.root / "proc"
        for pid, command in ((900001, b"/usr/bin/python3\0/tmp/runtime_30500.py\0--max_steps\0"),
                             (900002, b"/bin/bash\0-c\0cat runtime_30500.py\0"),
                             (900003, b"/usr/bin/python3\0-s\0other.py\0")):
            path = proc / str(pid)
            path.mkdir(parents=True)
            (path / "cmdline").write_bytes(command)
        found = active_work(proc)
        self.assertEqual(found, [{"pid": 900001, "script": "runtime_30500.py"}])
        self.assertTrue((proc / "900001/cmdline").is_file())

    def test_all_controls_have_same_budget_lr_parent_and_restored_state(self):
        for variant in ("multistep", "answer_only", "replay_only"):
            cmd = build_command(source=self.root, train=self.data, corpus=self.holdout,
                                local=self.root, output=self.root / "result", steps=1000,
                                variant=variant, repeat=16)
            def arg(name):
                return cmd[cmd.index(name) + 1]
            self.assertEqual(arg("--max_steps"), "31500")
            self.assertEqual(arg("--step_tokens"), "36864")
            self.assertEqual(arg("--phase_lr"), "3e-6")
            self.assertEqual(arg("--distill_teacher"), arg("--warmstart_path"))
            self.assertEqual(arg("--new_group_min_exposures"), "0" if variant == "replay_only" else "1000")
            self.assertIn("--save_optimizer", cmd)
            self.assertIn("--new_dataset_sampler", cmd)
            self.assertTrue(arg("--resume_state").endswith("train_state_step_30500.pt"))

    def test_both_knowledge_dev_formats_are_held_out(self):
        paths = fixed_heldouts(PACKAGE)
        self.assertIn(PACKAGE / "data/knowledge_dev.jsonl", paths)
        self.assertIn(PACKAGE / "data/knowledge_free_dev.jsonl", paths)
        self.assertNotIn(PACKAGE / "data/knowledge_train.jsonl", paths)

    def test_cli_default_is_ten_thousand_additional_updates_without_side_effects(self):
        destination = self.root / "new_branch"
        args = parse_args(["--output", str(destination), "--preflight-only"])
        self.assertEqual((args.parent_step, args.steps, args.new_repeat), (30500, 10000, 16))
        self.assertEqual(args.variant, "multistep")
        self.assertTrue(args.preflight_only)
        self.assertFalse(destination.exists())

    def test_cli_accepts_additional_update_boundaries_and_short_controls(self):
        for steps in (1, 1000, 2500, 9999, 10000):
            with self.subTest(steps=steps):
                args = parse_args(["--output", str(self.root / "new_branch"), "--steps", str(steps)])
                self.assertEqual(args.steps, steps)

    def test_cli_rejects_out_of_range_updates_before_any_file_access(self):
        for steps in (-1, 0, 10001):
            with self.subTest(steps=steps), redirect_stderr(io.StringIO()) as errors:
                with self.assertRaises(SystemExit) as stopped:
                    parse_args(["--output", str(self.root / "new_branch"), "--steps", str(steps)])
                self.assertEqual(stopped.exception.code, 2)
                self.assertIn("steps must be 1..10000", errors.getvalue())
        self.assertFalse((self.root / "new_branch").exists())

    def test_ten_thousand_update_cli_keeps_repeat_limit(self):
        for repeat in (0, 17):
            with self.subTest(repeat=repeat), redirect_stderr(io.StringIO()) as errors:
                with self.assertRaises(SystemExit) as stopped:
                    parse_args(["--output", str(self.root / "new_branch"), "--steps", "10000",
                                "--new-repeat", str(repeat)])
                self.assertEqual(stopped.exception.code, 2)
                self.assertIn("new-repeat 1..16", errors.getvalue())

    def test_ten_thousand_update_command_targets_40500_and_keeps_protection(self):
        for variant in ("multistep", "answer_only", "replay_only"):
            with self.subTest(variant=variant):
                args = parse_args(["--output", str(self.root / "new_branch"), "--variant", variant])
                cmd = build_command(source=self.root, train=self.data, corpus=self.holdout,
                                    local=self.root, output=args.output, steps=args.steps,
                                    variant=args.variant, repeat=args.new_repeat)
                def arg(name):
                    return cmd[cmd.index(name) + 1]
                self.assertEqual((arg("--max_steps"), arg("--stop_at_step")), ("40500", "40500"))
                self.assertEqual((arg("--phase_start"), arg("--required_start_step")), ("30500", "30500"))
                self.assertEqual((arg("--phase_lr"), arg("--phase_floor")), ("3e-6", "3e-6"))
                self.assertEqual(arg("--step_tokens"), "36864")
                self.assertEqual(arg("--min_replay_share"), "0.75")
                self.assertEqual(arg("--distill_teacher"), arg("--warmstart_path"))
                self.assertEqual((arg("--distill_weight"), arg("--distill_temperature")), ("1.0", "1.0"))
                self.assertEqual(arg("--new_group_min_exposures"), "0" if variant == "replay_only" else "1000")
                self.assertEqual(arg("--new_curriculum_repeat"), "1" if variant == "replay_only" else "16")
                for flag in ("--mask_tool_output", "--save_optimizer", "--new_dataset_sampler"):
                    self.assertIn(flag, cmd)
                self.assertTrue(arg("--resume_state").endswith("train_state_step_30500.pt"))


if __name__ == "__main__":
    unittest.main()
