"""Read-only count of target conflicts between v43 and Python conversions.

Both records use the same question.  If their supervised answer suffixes differ,
the same next-token context has two different labels.  This script counts the
first differing position without loading the model or changing either corpus.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import json
from pathlib import Path
import zipfile


TRAIN_MEMBER = "data/train_v43_knowledge_unified.jsonl"


def common_prefix_length(left, right) -> int:
    index = 0
    for a, b in zip(left, right):
        if a != b:
            break
        index += 1
    return index


def original_normalizer(source: str):
    tree = ast.parse(source)
    table = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "_BASE_NORM_DICT"
            for target in node.targets
        ):
            table = ast.literal_eval(node.value)
            break
    if not isinstance(table, dict):
        raise ValueError("Original normalization table is missing")
    for code in range(0xFF01, 0xFF5F):
        char = chr(code)
        if char not in ("（", "）", "！", "？", "：", "～"):
            table[char] = chr(code - 0xFEE0)
    translation = str.maketrans(table)
    return lambda value: value.translate(translation)


def target_tokens(full: str, prompt: str, normalize, vocab: dict[str, int]) -> list[int]:
    normalized_full = normalize(full)
    normalized_prompt = normalize(prompt)
    full_ids = [vocab[char] for char in normalized_full if char in vocab]
    prompt_length = sum(char in vocab for char in normalized_prompt)
    if len(full_ids) < 2048:
        full_ids.append(3)
    else:
        full_ids = full_ids[:2048]
    return full_ids[prompt_length:]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v43-zip", type=Path, required=True)
    parser.add_argument("--converted", type=Path, required=True)
    args = parser.parse_args()

    converted: dict[int, dict] = {}
    with args.converted.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            ref = row["source_ref"]
            if ref["file"] != TRAIN_MEMBER or ref["line"] in converted:
                raise ValueError("Converted source reference is invalid or repeated")
            converted[ref["line"]] = row

    counts: Counter[str] = Counter()
    by_source: dict[str, Counter[str]] = defaultdict(Counter)
    matched: set[int] = set()
    with zipfile.ZipFile(args.v43_zip) as archive:
        metadata = json.loads(archive.read("base_metadata.json"))
        vocab = metadata["vocab"]
        normalize = original_normalizer(
            archive.read("qwenputer_controller_v30.py").decode("utf-8")
        )
        with archive.open(TRAIN_MEMBER) as stream:
            for line_number, raw in enumerate(stream, 1):
                newer = converted.get(line_number)
                if newer is None:
                    continue
                old = json.loads(raw)
                if old["prompt"] != newer["prompt"]:
                    raise ValueError("Original and Python prompts differ")
                old_full = old.get("full_text") or old["prompt"] + old["target"]
                if not old_full.startswith(old["prompt"]):
                    raise ValueError("Original full text does not contain the prompt")
                old_answer = old_full[len(old["prompt"]):]
                new_answer = newer["target"]
                common = common_prefix_length(old_answer, new_answer)
                old_ids = target_tokens(old_full, old["prompt"], normalize, vocab)
                new_ids = target_tokens(newer["full_text"], newer["prompt"], normalize, vocab)
                token_common = common_prefix_length(old_ids, new_ids)
                source = str(old.get("source", "unknown"))
                for bucket in (counts, by_source[source]):
                    bucket["rows"] += 1
                    bucket["common_prefix_chars_sum"] += common
                    bucket["different_target"] += old_answer != new_answer
                    bucket["different_first_char"] += common == 0
                    bucket["different_within_first_4_chars"] += common < 4
                    bucket["different_within_first_16_chars"] += common < 16
                    bucket["common_prefix_tokens_sum"] += token_common
                    bucket["different_supervised_tokens"] += old_ids != new_ids
                    bucket["different_first_token"] += token_common == 0
                    bucket["different_within_first_16_tokens"] += token_common < 16
                matched.add(line_number)

    if matched != set(converted):
        raise ValueError("Some converted source lines were not in original training")
    print(json.dumps({"schema": "geocpu_v43_python_same_prompt_target_diagnostic_v1",
                      "counts": dict(counts),
                      "by_source": {name: dict(values) for name, values in by_source.items()}},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
