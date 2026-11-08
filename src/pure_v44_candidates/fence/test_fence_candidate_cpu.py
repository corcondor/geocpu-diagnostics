"""CPU-only regression fixtures for the isolated pure-v44 fence candidate."""

from prepare_public_v44 import normalize_tir, sha_text
from tir_engine_v44r1 import PYTHON_BLOCK


def check(source: str, expected_opening: str, expected_blocks: int = 1) -> None:
    target, segments, receipts = normalize_tir(source)
    matches = list(PYTHON_BLOCK.finditer(target))
    assert len(matches) == expected_blocks, (source, target, len(matches))
    assert all(match.group(1).strip() == "print(1)" for match in matches)
    assert len(receipts) == expected_blocks
    assert all(receipt["original_python_opening_fence"] == expected_opening for receipt in receipts)
    assert source != target
    assert sha_text(source) != sha_text(target)
    tool_segments = [part for part in segments if part["role"] == "tool"]
    assert len(tool_segments) == expected_blocks
    for part in tool_segments:
        assert part["train"] is False
        assert target[part["start"]:part["end"]] == "\n```output\n1\n```\n"
    assert segments[0]["role"] == "assistant" and segments[0]["train"] is True
    assert target.startswith("```python\n")


if __name__ == "__main__":
    for opening in ("```py\n", "```Python\n", "``` python \n"):
        source = opening + "print(1)\n```\n```output\n1\n```\n"
        check(source, opening)
        check(source + source, opening, 2)
    print("PASS: aliases and two-block offsets match the pure-v44 executable fence and tool mask")
