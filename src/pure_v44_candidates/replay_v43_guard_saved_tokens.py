"""Read-only replay of v43's static repetition guard on saved pure-v44 tokens.

This does not run either model. It tests only whether the exact v43 guard
would have stopped the already-saved pure-v44 token sequences sooner.
"""

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def guard_stop(tokens: list[int]) -> tuple[int | None, str | None]:
    """v43 integrated_model_v30.py lines 776-782, with EOS checked first."""
    generated_tokens: list[int] = []
    for sampled_position, next_tok in enumerate(tokens, 1):
        if next_tok == 3:  # v43 checks EOS before appending/guarding.
            break
        generated_tokens.append(next_tok)
        n_gen = len(generated_tokens)
        if n_gen >= 32:
            if generated_tokens[-8:] == generated_tokens[-16:-8] == generated_tokens[-24:-16] == generated_tokens[-32:-24]:
                return sampled_position, "four_equal_8_token_blocks"
            if n_gen >= 64 and generated_tokens[-16:] == generated_tokens[-32:-16] == generated_tokens[-48:-32] == generated_tokens[-64:-48]:
                return sampled_position, "four_equal_16_token_blocks"
    return None, None


def replay(path: Path) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    result = []
    for row in document["records"]:
        generation = row["generation"]
        tokens = generation["generated_tokens"]
        assert all(isinstance(token, int) for token in tokens)
        assert generation["sampled_token_count"] == len(tokens)
        assert generation["injected_token_count"] == 0
        assert generation["tool_invocations"] == 0
        assert "<" not in generation["model_generated_text"]
        assert (tokens[-1] == 3) == (generation["stop_reason"] == "eos")
        stop, reason = guard_stop(tokens)
        result.append({
            "case_index": row["case_index"],
            "saved_sampled_tokens": len(tokens),
            "saved_stop_reason": generation["stop_reason"],
            "v43_guard_first_sampled_position": stop,
            "v43_guard_reason": reason,
        })
    return {"path": str(path), "sha256": sha256(path), "cases": result}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v43-source", type=Path, required=True)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.v43_source.read_text(encoding="utf-8")
    assert "generated_tokens[-8:] == generated_tokens[-16:-8] == generated_tokens[-24:-16] == generated_tokens[-32:-24]" in source
    assert "generated_tokens[-16:] == generated_tokens[-32:-16] == generated_tokens[-48:-32] == generated_tokens[-64:-48]" in source
    payload = {
        "schema": "geocpu_v43_guard_replay_saved_pure_v44_v1",
        "scope": "Saved tokens only; no model inference, no token injection, no training",
        "v43_source": {"path": str(args.v43_source), "sha256": sha256(args.v43_source), "guard_lines": "776-782"},
        "before": replay(args.before),
        "after": replay(args.after),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    print(json.dumps({"before": [case["v43_guard_first_sampled_position"] for case in payload["before"]["cases"]],
                      "after": [case["v43_guard_first_sampled_position"] for case in payload["after"]["cases"]]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
