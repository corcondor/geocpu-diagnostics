"""Read-only v44 Tier 8 formatting probe using the original four prompts.

This measures generated code and explanation form. It does not train the model,
execute C++/Rust, or claim correctness of the mathematical answer.
"""

from __future__ import annotations

import re
from typing import Callable


CASES = (
    ("python_derivative", "ユーザー: 関数 f(x) = 3x^3 + 2sin(x) - 5 について、導関数 f'(x) を求め、x = 0 における微分係数 f'(0) を計算してください。\nアシスタント: ", "python"),
    ("python_projectile", "ユーザー: 初速 20 m/s、仰角 45° で斜方投射された物体の滞空時間と水平到達距離を求めてください（重力加速度 g = 9.8 m/s^2、空気抵抗は無視）。\nアシスタント: ", "python"),
    ("cpp_search", "ユーザー: C++ を用いて、効率的な『二分探索 (Binary Search)』の完全な実装コードを作成し、その計算量と動作原理を解説してください。\nアシスタント: ", "cpp"),
    ("rust_fibonacci", "ユーザー: Rust を用いて、効率的な『フィボナッチ数列のメモ化DP』の完全な実装コードを作成し、その計算量と動作原理を解説してください。\nアシスタント: ", "rust"),
)

ALIGNED_CASES = tuple(
    (case_id,
     ("問題: " if language == "python" else "指示: ")
     + original_prompt.removeprefix("ユーザー: ").removesuffix("\nアシスタント: ")
     + ("\n思考:" if language == "python" else "\n回答:"),
     language)
    for case_id, original_prompt, language in CASES
)

FENCE = re.compile(r"(?ms)^```([^\n`]*)\n(.*?)^```[ \t]*(?:\n|$)")
LANGUAGE = {
    "python": {"python", "py"},
    "cpp": {"cpp", "c++", "cxx"},
    "rust": {"rust", "rs"},
}


def _score(case_id: str, language: str, generation: dict) -> dict:
    model_text = generation.get("model_generated_text")
    if not isinstance(model_text, str):
        raise ValueError("Tier 8 requires model_generated_text, separate from injected observations")
    matches = list(FENCE.finditer(model_text))
    labels = [m.group(1).strip().lower() for m in matches]
    outside = FENCE.sub("", model_text)
    # The first two fields retain v44's historical display heuristics.
    legacy_code = ("```" in model_text or "def " in model_text
                   or "int " in model_text or "fn " in model_text)
    legacy_narrative = any(char in model_text for char in "起承転結求")
    trace = generation.get("python_trace", [])
    if not isinstance(trace, list):
        raise ValueError("python_trace must be a list")
    return {
        "case_id": case_id, "expected_language": language,
        "has_code": legacy_code, "has_narrative": legacy_narrative,
        "expected_language_fence": any(label in LANGUAGE[language] for label in labels),
        "has_prose_outside_code": bool(re.search(r"[ぁ-んァ-ン一-龯]{4,}", outside)),
        "python_invocations": int(generation.get("python_invocations", 0)),
        "successful_python_invocations": sum(bool(item.get("success")) for item in trace),
        "ended_with_eos": bool(generation.get("ended_with_eos", False)),
        "model_generated_text": model_text,
        "output_text": generation.get("output_text"),
    }


def evaluate_tier8(generate: Callable[[str], dict], *, profile: str = "original_v44") -> dict:
    """Evaluate either original v44 prompts or content-matched v43-format prompts."""
    if profile not in ("original_v44", "aligned_v43"):
        raise ValueError("Unknown Tier 8 prompt profile")
    cases = CASES if profile == "original_v44" else ALIGNED_CASES
    details = [_score(case_id, language, generate(prompt))
               for case_id, prompt, language in cases]
    n = len(details)
    return {
        "tier": 8, "profile": profile, "n": n,
        "tir_code_rate": 100 * sum(row["has_code"] for row in details) / n,
        "narrative_rate": 100 * sum(row["has_narrative"] for row in details) / n,
        "expected_language_fence_rate": 100 * sum(row["expected_language_fence"] for row in details) / n,
        "prose_outside_code_rate": 100 * sum(row["has_prose_outside_code"] for row in details) / n,
        "python_invocation_count": sum(row["python_invocations"] for row in details),
        "successful_python_invocation_count": sum(row["successful_python_invocations"] for row in details),
        "details": details,
    }


def evaluate_both_profiles(generate: Callable[[str], dict]) -> dict:
    """Keep original and aligned results separate; never pool their rates."""
    return {"original_v44": evaluate_tier8(generate, profile="original_v44"),
            "aligned_v43": evaluate_tier8(generate, profile="aligned_v43")}


def evaluate_model(model, vocab, inv_vocab, *, max_tokens: int = 2048,
                   profile: str = "original_v44") -> dict:
    """Native v44fix decode. The caller loads a pinned checkpoint and sets eval mode."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    return evaluate_tier8(lambda prompt: model.generate_v44fix(
        prompt, vocab, inv_vocab, max_tokens=max_tokens,
        enable_python=True, loop_guard=False, repetition_penalty=1.0), profile=profile)
