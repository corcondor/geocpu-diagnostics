"""Compare retained neural methods in the actual v43 and pure-v44 sources.

Read-only source audit. It does not construct a model or infer behavior.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path


METHODS = (
    "calibrate_router",
    "_valid",
    "_run_enc_block",
    "_encoder_stack",
    "_main",
    "_project_residual",
    "forward",
    "prefill",
    "step_state",
    "step",
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def methods(path: Path) -> dict[str, ast.FunctionDef]:
    root = ast.parse(path.read_text(encoding="utf-8"))
    model = next(
        node for node in root.body
        if isinstance(node, ast.ClassDef) and node.name == "IntegratedModelV27"
    )
    return {node.name: node for node in model.body if isinstance(node, ast.FunctionDef)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v43", type=Path, required=True)
    parser.add_argument("--pure-v44", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    left, right = methods(args.v43), methods(args.pure_v44)
    checks = []
    for name in METHODS:
        original = ast.dump(left[name], include_attributes=False)
        revised = ast.dump(right[name], include_attributes=False)
        checks.append({
            "method": name,
            "ast_identical": original == revised,
            "v43_ast_sha256": hashlib.sha256(original.encode()).hexdigest(),
            "pure_v44_ast_sha256": hashlib.sha256(revised.encode()).hexdigest(),
        })
    result = {
        "schema": "geocpu_v43_pure_v44_retained_neural_ast_v1",
        "scope": "IntegratedModelV27 retained methods only; excludes constructor, wrappers and imports",
        "v43_source": {"path": str(args.v43), "sha256": file_sha256(args.v43)},
        "pure_v44_source": {"path": str(args.pure_v44), "sha256": file_sha256(args.pure_v44)},
        "checks": checks,
        "all_identical": all(check["ast_identical"] for check in checks),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"all_identical": result["all_identical"], "checks": [(c["method"], c["ast_identical"]) for c in checks]}))


if __name__ == "__main__":
    main()
