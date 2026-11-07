"""Frozen GeoCPU evaluation inputs and unchanged original scoring entry points.

Importing this module performs no model loading, GPU allocation, training, or
filesystem mutation.  A registry row count is not an observed model score.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
from typing import Iterator


FROZEN_ROOT = Path(__file__).resolve().parent / "frozen"
# Filled from the byte-preserved source inventory and generated registry.
SOURCE_FILES_SHA256 = "1a579f01d6269084bc428e7e67d139fa4bdd6a3aff142d5e1912f944489bbb88"
REGISTRY_SHA256 = "527f0f1cb4b97ce83fdc79e247f64ec20653befa9b6964d3e2ce65f589e99928"
_ALIASES = {"python": "python120", "engineering": "engineering64",
            "visual": "visual10", "search_final": "search_final198",
            "search_decision": "search_decision200"}
_SCORERS = {
    "python120": "continue/eval_python/eval_python_tir_30500.py",
    "engineering64": "continue/engineering/scorer.py",
    "visual10": "continue/visual/eval_visual.py",
    "public_dev80": "public/code/score_traces.py",
    "T8": "base/code/tier8_readonly_v44fix.py",
    "new_public": "base/code/eval_new_public_v46.py",
}


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _root(root=None) -> Path:
    return Path(root) if root is not None else FROZEN_ROOT


def _pinned_json(path: Path, digest: str):
    data = path.read_bytes()
    if _digest(data) != digest:
        raise ValueError(f"frozen manifest changed: {path}")
    return json.loads(data)


def source_inventory(root=None) -> dict:
    return _pinned_json(_root(root) / "source_files.json", SOURCE_FILES_SHA256)


def registry(root=None) -> dict:
    """Return the frozen suite/condition registry, with no generated scores."""
    return _pinned_json(_root(root) / "registry.json", REGISTRY_SHA256)


def source_path(relative: str, root=None) -> Path:
    """Resolve and verify only an inventoried original file; reject traversal."""
    base = _root(root).resolve()
    path = (base / relative).resolve()
    if not path.is_relative_to(base):
        raise ValueError("source path escapes frozen root")
    info = source_inventory(base).get(relative)
    if info is None:
        raise KeyError(f"not an inventoried original: {relative}")
    data = path.read_bytes()
    if len(data) != info["bytes"] or _digest(data) != info["sha256"]:
        raise ValueError(f"frozen original changed: {relative}")
    return path


def verify_frozen(root=None) -> dict:
    """Verify all original bytes and the registry, without executing scorers."""
    inventory = source_inventory(root)
    for name in inventory:
        source_path(name, root)
    data = registry(root)
    return {"state": "verified_sources_only", "files": len(inventory),
            "bytes": sum(info["bytes"] for info in inventory.values()),
            "suite_count": len(data["suites"]),
            "registry_sha256": REGISTRY_SHA256,
            "source_files_sha256": SOURCE_FILES_SHA256,
            "model_evaluation_executed": False}


def get_suite(suite_id: str, root=None) -> dict:
    suite_id = _ALIASES.get(suite_id, suite_id)
    for suite in registry(root)["suites"]:
        if suite["id"] == suite_id:
            return suite
    raise KeyError(f"unknown frozen suite: {suite_id}")


def _jsonl(relative: str, root=None) -> list[dict]:
    with source_path(relative, root).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _load_module(relative: str, root=None):
    path = source_path(relative, root)
    module_name = "geocpu_frozen_" + _digest(str(path).encode())[:16]
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import original scorer: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def load_original_scorer(suite_id: str, root=None):
    """Load the byte-preserved CPU scorer, never its model-generating main().

    python120: score_generation(case, generation), summarize(rows)
    engineering64: score_case(case, generation)
    visual10: evaluate(case, model_generated_text, artifact_dir)
    public_dev80: score(reference_jsonl_path, generated_jsonl_path)
    T8: evaluate_tier8(generate_callable, profile=...)

    Scorer errors are deliberately propagated; callers must record unmeasured
    items separately from correct/incorrect items instead of inventing a score.
    """
    suite_id = _ALIASES.get(suite_id, suite_id)
    if suite_id.startswith("new_public:"):
        suite_id = "new_public"
    if suite_id not in _SCORERS:
        raise KeyError(f"no standalone original scorer for {suite_id}")
    return _load_module(_SCORERS[suite_id], root)


def iter_cases(suite_id: str, mode: str | None = None, root=None) -> Iterator[dict]:
    """Yield the original ordered case dictionaries for a frozen formal mode.

    For T8, original tuples are represented as dictionaries with id, prompt,
    language, and profile.  All other rows retain their original fields.
    T3 defaults to its full logit set; mode='generative' selects the original
    first 100 rows.  New-public groups default to mode='raw' (12 or fewer);
    mode='nll' selects the original 64 or fewer reference-loss rows.
    """
    suite = get_suite(suite_id, root)
    if suite.get("availability") != "frozen":
        raise RuntimeError(f"suite inputs unavailable: {suite['id']}")
    selector = suite["selection"]
    kind = selector["kind"]
    if kind == "tier8_profiles":
        original = load_original_scorer("T8", root)
        profiles = {"original_v44": original.CASES, "aligned_v43": original.ALIGNED_CASES}
        if mode is not None and mode not in profiles:
            raise ValueError(f"unknown T8 mode: {mode}")
        for profile, cases in profiles.items():
            if mode is None or mode == profile:
                for case_id, prompt, language in cases:
                    yield {"id": case_id, "prompt": prompt, "language": language, "profile": profile}
        return
    if kind == "json_cases":
        data = json.loads(source_path(selector["source"], root).read_text(encoding="utf-8"))
        rows = data[selector["key"]]
    else:
        rows = _jsonl(selector["source"], root)
    if kind == "tier7_first_mc":
        rows = [row for row in rows if row.get("task_type") == "multiple_choice" or
                "choices" in row or ("A" in row.get("prompt", "") and "B" in row.get("prompt", ""))][:200]
    elif kind == "sha_id_first":
        rows = sorted(rows, key=lambda row: _digest(row["id"].encode()))[:selector["count"]]
    elif kind == "public_depth":
        module = _load_module("r6/eval_public_r6.py", root)
        rows = module.select_rows(source_path(selector["source"], root), selector["per_depth"])
    elif kind == "new_public":
        mode = mode or "raw"
        if mode not in ("raw", "nll"):
            raise ValueError("new-public mode must be raw or nll")
        module = load_original_scorer("new_public", root)
        rows = module.chosen(selector["group"], rows, 12 if mode == "raw" else 64)
    elif kind not in ("jsonl_all", "json_cases"):
        raise ValueError(f"unknown frozen selector: {kind}")
    if suite["id"] == "T3":
        if mode == "generative":
            rows = rows[:100]
        elif mode not in (None, "logit"):
            raise ValueError("T3 mode must be logit or generative")
    elif suite["id"] == "T7":
        if mode not in (None, "closed", "open"):
            raise ValueError("T7 mode must be closed or open")
    elif kind not in ("new_public",) and mode not in (None, "formal"):
        raise ValueError(f"unexpected mode for {suite['id']}: {mode}")
    yield from rows


def generation_kwargs(suite_id: str, mode: str | None = None, root=None) -> dict:
    """Return explicit native v44fix generation kwargs for auxiliary suites.

    Tier1-7 must use the preserved stage (IntegratedModelV42); substituting
    v44fix/Python generation would change their original conditions.
    """
    suite = get_suite(suite_id, root)
    settings = suite.get("generation_kwargs")
    if settings is None:
        raise ValueError(f"use unchanged original stage for {suite['id']}")
    return dict(settings)


def iter_heldout_prompts(root=None) -> Iterator[dict]:
    """Expose every available held-out prompt, including unselected dev rows.

    This corpus intentionally includes archived alternate Tier manifests and
    all public dev families, so an added curriculum cannot leak evaluation
    rows merely because today's scorer subsamples them.  Exact prompt checks
    alone do not certify semantic/family-disjointness.
    """
    data = registry(root)
    for entry in data["leakage_sources"]:
        path = entry["path"]
        if entry.get("format") == "selection_json":
            rows = json.loads(source_path(path, root).read_text(encoding="utf-8"))["cases"]
        else:
            rows = _jsonl(path, root)
        for index, row in enumerate(rows):
            prompt = row.get("prompt")
            if not isinstance(prompt, str) or not prompt:
                continue
            families = {key: row[key] for key in data["family_fields"] if key in row and row[key] is not None}
            yield {"suite": entry.get("suite", "heldout_source"), "source": path,
                   "row_index": index, "id": row.get("id", str(index)),
                   "prompt": prompt, "prompt_sha256": _digest(prompt.encode()),
                   "families": families}
            # Some source records carry a natural-language question separate
            # from the labeled prompt.  Expose it as an additional leakage
            # view, never as a replacement for the frozen formal prompt.
            for key in ("question", "plain_prompt"):
                view = row.get(key)
                if isinstance(view, str) and view and view != prompt:
                    yield {"suite": entry.get("suite", "heldout_source"),
                           "source": path, "row_index": index,
                           "id": row.get("id", str(index)), "view": key,
                           "prompt": view, "prompt_sha256": _digest(view.encode()),
                           "families": families}
    for row in iter_cases("T8", root=root):
        yield {"suite": "T8", "source": _SCORERS["T8"], "id": row["id"],
               "profile": row["profile"], "prompt": row["prompt"],
               "prompt_sha256": _digest(row["prompt"].encode()),
               "families": {"original_case_id": row["id"]}}
    # Native T3/T7 construct these model-input prompts from their source row.
    # Include the actual input strings as well as the archived prompt fields.
    for row in iter_cases("T3", mode="generative", root=root):
        prompt = ("指示:: 質問と回答の選択肢を入力として受け取り、選択肢から回答を選択してください。回答の他には何も含めないことを厳守してください。\n\n"
                  f"質問:: {row['question']}\n"
                  f"選択肢:: {' '.join(row['choices'])}\n回答::")
        yield {"suite": "T3", "source": "r6/eval_jamcqa_official.py",
               "id": row["id"], "view": "native_generative_prompt",
               "prompt": prompt, "prompt_sha256": _digest(prompt.encode()), "families": {}}
    for row in iter_cases("T7", root=root):
        prompt = row.get("prompt", "").strip() + "\n思考: "
        yield {"suite": "T7", "source": "r6/eval_v43_ab_benchmark.py",
               "id": row["id"], "view": "native_ab_prompt",
               "prompt": prompt, "prompt_sha256": _digest(prompt.encode()), "families": {}}


def heldout_prompt_hashes(root=None) -> set[str]:
    return {row["prompt_sha256"] for row in iter_heldout_prompts(root)}


def _native_search_scorer(final: bool, root=None):
    """Compile the original per-case scoring AST, omitting only generation.

    Original search scripts have scoring inline in main(), not a callable.
    The same AST statements are reused verbatim; no scoring formula, prompt,
    answer normalization, or terminal criterion is rewritten here.
    """
    relative = ("searchprobe/eval_final_result_probe.py" if final else
                "base/code/eval_search_decision_v46.py")
    path = source_path(relative, root)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    loop = next(node for node in main.body if isinstance(node, ast.For) and
                isinstance(node.target, ast.Name) and node.target.id == ("case" if final else "item"))
    body = [node for node in loop.body if not (isinstance(node, ast.Assign) and
            any(isinstance(target, ast.Name) and target.id == "output" for target in node.targets))]
    prefix = ast.parse("raw = []\noutput = generation\n" + ("" if final else "item = case\n")).body
    fn = ast.FunctionDef(name="score_native", args=ast.arguments(
        posonlyargs=[], args=[ast.arg(arg="case"), ast.arg(arg="generation")],
        kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=prefix + body + ast.parse("return raw[0]").body, decorator_list=[])
    helpers = []
    if final:
        helpers = [node for node in tree.body if
                   (isinstance(node, ast.FunctionDef) and node.name == "normalized") or
                   (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and
                       t.id == "EXPLICIT_ANSWER" for t in node.targets))]
    module = ast.fix_missing_locations(ast.Module(body=helpers + [fn], type_ignores=[]))
    namespace = {"re": re, "hashlib": hashlib}
    exec(compile(module, str(path) + ":unchanged-scoring-AST", "exec"), namespace)
    return namespace["score_native"]


def score_search_decision(case: dict, generation: dict, root=None) -> dict:
    return _native_search_scorer(False, root)(case, generation)


def score_search_final(case: dict, generation: dict, root=None) -> dict:
    return _native_search_scorer(True, root)(case, generation)


def build_official_tier_command(*, study, run: str, step: int,
                                checkpoint_sha: str, output, python=sys.executable,
                                preflight_only=False, root=None) -> list[str]:
    """Build (do not execute) the unchanged saved-checkpoint Tier1-8 stage."""
    if re.fullmatch(r"[A-Za-z0-9_-]+", run) is None or step <= 0:
        raise ValueError("invalid saved run/step")
    if re.fullmatch(r"[0-9a-fA-F]{64}", checkpoint_sha) is None:
        raise ValueError("checkpoint_sha must be the actual 64-hex digest")
    stage = source_path("continue/tier_stage/stage_official_tier_v47.py", root)
    command = [str(python), "-u", str(stage), "--study", str(study), "--run", run,
               "--step", str(step), "--checkpoint-sha", checkpoint_sha.lower(),
               "--output", str(output)]
    if preflight_only:
        command.append("--preflight-only")
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify")
    sub.add_parser("list")
    prompt = sub.add_parser("prompts", help="emit read-only JSONL for leakage checks")
    prompt.add_argument("--hash-only", action="store_true")
    command = sub.add_parser("tier-command", help="print original argv; never execute")
    command.add_argument("--study", required=True)
    command.add_argument("--run", required=True)
    command.add_argument("--step", required=True, type=int)
    command.add_argument("--checkpoint-sha", required=True)
    command.add_argument("--output", required=True)
    command.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.command == "verify":
        print(json.dumps(verify_frozen(), ensure_ascii=False, indent=2))
    elif args.command == "list":
        print(json.dumps(registry(), ensure_ascii=False, indent=2))
    elif args.command == "prompts":
        for row in iter_heldout_prompts():
            if args.hash_only:
                row.pop("prompt")
            print(json.dumps(row, ensure_ascii=False))
    else:
        print(json.dumps(build_official_tier_command(study=args.study, run=args.run,
            step=args.step, checkpoint_sha=args.checkpoint_sha, output=args.output,
            preflight_only=args.preflight_only), ensure_ascii=False))


if __name__ == "__main__":
    main()
