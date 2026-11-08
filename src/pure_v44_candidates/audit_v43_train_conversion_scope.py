"""Read-only census of the v43 train JSONL inside its ZIP; no model execution."""

import collections
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path


def main(zip_path: str, output_path: str) -> None:
    source_count = collections.Counter()
    tags = collections.Counter()
    opcodes = collections.Counter()
    unique_prompt = set()
    unique_pair = set()
    train_eval_prompt_overlap = 0
    eval_counts = collections.Counter()
    overlap_by_file = collections.Counter()
    overlap_by_source = collections.Counter()
    examples = {}

    with zipfile.ZipFile(zip_path) as archive:
        eval_names = [
            name
            for name in archive.namelist()
            if name.startswith("data/")
            and name.endswith(".jsonl")
            and ("eval" in name or "manifest" in name)
        ]
        eval_prompts = set()
        eval_membership = collections.defaultdict(set)
        for name in eval_names:
            with archive.open(name) as lines:
                for raw in lines:
                    prompt = json.loads(raw).get("prompt")
                    if isinstance(prompt, str):
                        eval_counts[name] += 1
                        eval_prompts.add(prompt)
                        eval_membership[prompt].add(name)

        with archive.open("data/train_v43_knowledge_unified.jsonl") as lines:
            for raw in lines:
                row = json.loads(raw)
                source = row.get("source", "?")
                prompt = row.get("prompt", "")
                target = row.get("target", "")
                source_count[source] += 1
                unique_prompt.add(hashlib.sha256(prompt.encode("utf-8")).digest())
                unique_pair.add(hashlib.sha256((prompt + "\0" + target).encode("utf-8")).digest())
                if prompt in eval_prompts:
                    train_eval_prompt_overlap += 1
                    overlap_by_source[source] += 1
                    for name in eval_membership[prompt]:
                        overlap_by_file[name] += 1
                if "<SEARCH " in target:
                    tags["search_rows"] += 1
                if re.search(r"<(?:SET|ADD|SUB|MUL|DIV|MOD|DEC|INC|AND|OR|XOR|SHL|SHR)\b", target):
                    tags["cpu_rows"] += 1
                if "```python" in target:
                    tags["python_fence_rows"] += 1
                if "[RESULT:" in target:
                    tags["search_result_rows"] += 1
                for opcode in re.findall(r"<([A-Z]+)\b", target):
                    opcodes[opcode] += 1
                if source not in examples:
                    examples[source] = {
                        "prompt_prefix": prompt[:120],
                        "target_prefix": target[:240],
                    }

    result = {
        "zip_path": zip_path,
        "train_member": "data/train_v43_knowledge_unified.jsonl",
        "train_rows": sum(source_count.values()),
        "unique_prompts_sha256": len(unique_prompt),
        "unique_prompt_target_sha256": len(unique_pair),
        "bundled_eval_names": eval_names,
        "bundled_eval_unique_prompts": len(eval_prompts),
        "train_rows_exact_prompt_matching_bundled_eval": train_eval_prompt_overlap,
        "bundled_eval_prompt_rows_by_file": dict(eval_counts),
        "train_prompt_overlap_rows_by_eval_file": dict(overlap_by_file),
        "train_prompt_overlap_rows_by_source": dict(overlap_by_source),
        "sources": dict(source_count.most_common()),
        "tags": dict(tags),
        "opcode_occurrences": dict(opcodes.most_common()),
        "one_prefix_example_per_source": examples,
    }
    Path(output_path).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("bundled_eval_names", "one_prefix_example_per_source")}, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
