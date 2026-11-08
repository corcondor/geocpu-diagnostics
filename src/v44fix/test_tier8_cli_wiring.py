"""Static CLI/wiring checks; no GeoCPU model is imported or executed."""

import ast
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
OVERLAY = HERE / "runtime_v44fix.py"
MODEL = HERE / "integrated_model_v44fix.py"


def function(tree, name):
    return next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                and node.name == name)


class Tier8CliWiringTests(unittest.TestCase):
    def test_cli_flags_two_profiles_and_json_result_are_reachable(self):
        tree = ast.parse(OVERLAY.read_text(encoding="utf-8"))
        imports = [node for node in tree.body if isinstance(node, ast.ImportFrom)]
        self.assertTrue(any(node.module == "tier8_readonly_v44fix"
                            and any(alias.name == "evaluate_model" and alias.asname == "evaluate_tier8_model"
                                    for alias in node.names) for node in imports))
        main = function(tree, "main")
        flags = [node.args[0].value for node in ast.walk(main)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                 and node.func.attr == "add_argument" and node.args
                 and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)]
        self.assertTrue({"--eval_only", "--tier8_only", "--tier8_max_tokens", "--tier8_output"} <= set(flags))
        probe = function(tree, "run_tier8_probe")
        calls = [node for node in ast.walk(probe) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == "evaluate_tier8_model"]
        self.assertEqual(len(calls), 2)
        profiles = {keyword.value.value for call in calls for keyword in call.keywords
                    if keyword.arg == "profile" and isinstance(keyword.value, ast.Constant)}
        self.assertEqual(profiles, {"original_v44", "aligned_v43"})
        self.assertTrue(any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                            and node.func.attr == "dump" and isinstance(node.func.value, ast.Name)
                            and node.func.value.id == "json" for node in ast.walk(probe)))

    def test_model_adds_no_new_trainable_forward_or_constructor(self):
        model_class = next(node for node in ast.parse(MODEL.read_text(encoding="utf-8")).body
                           if isinstance(node, ast.ClassDef) and node.name == "IntegratedModelV44Fix")
        self.assertFalse({"__init__", "forward"} & {node.name for node in model_class.body
                                                   if isinstance(node, ast.FunctionDef)})


if __name__ == "__main__":
    unittest.main()
