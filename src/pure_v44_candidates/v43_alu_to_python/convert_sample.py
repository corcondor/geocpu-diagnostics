"""Bounded, read-only v43 ALU trace -> pure-v44 Python TIR sample converter.

This is an admission prototype, not a training launcher. It uses the original
v43 CPU circuit to verify every accepted instruction and the v44 Python worker
to generate every tool observation. Unsupported or ambiguous examples fail shut.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import sys
import time
import unicodedata
import zipfile
from pathlib import Path


TRAIN_MEMBER = "data/train_v43_knowledge_unified.jsonl"
SOURCE_DIR = Path(r"C:\Users\81808\.openclaw\reports\jmmlu_v1\ag_fused_fullqa_training_v1\Grand_cotraining_v43_zip_source_exact_20261103")
V44_DIR = Path(r"C:\Users\81808\.openclaw\reports\jmmlu_v1\ag_fused_fullqa_training_v1\v44_hotfix_source_readonly_20261103")
ACCEPTED_SOURCES = {"Algebraic_Composite_Aligned", "Comparative_Conditional_Aligned", "swe_alu_accumulate_sum"}
OPS = "SET|MOV|ADD|SUB|MUL|DIV|MOD|INC|DEC"
ARG = r"(?:R\d+|[+-]?\d+)"
PAIR = re.compile(rf"<(?P<op>{OPS})\s+(?P<dst>R\d+)(?:\s+(?P<a>{ARG}))?(?:\s+(?P<b>{ARG}))?>\s*=>\s*\[(?P<observed>R\d+)=(?P<value>\d+)\]")
ANGLE = re.compile(r"<[A-Z][A-Z0-9]*\b")
RESULT = re.compile(r"\[R\d+=\d+\]")
ALGEBRAIC = re.compile(r"数式 \((\d+) \* (\d+) \+ (\d+)\) を (\d+) で割った余り")
COMPARATIVE = re.compile(r"問題: (.+?)には(.+?)が(\d+)個、(.+?)には(.+?)が(\d+)個入っています。どちらに何個多く")
SWE = re.compile(r"```python\nR0 = (\d+)\nfor i in range\((\d+)\):\n    R0 \+= i \* 2\n```", re.MULTILINE)


class Reject(ValueError):
    pass


def normalized_prompt(prompt: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", prompt).split())


def evaluation_prompts(archive: zipfile.ZipFile) -> tuple[set[str], set[str], dict[str, int]]:
    exact: set[str] = set()
    normalized: set[str] = set()
    counts: dict[str, int] = {}
    for name in archive.namelist():
        if not (name.startswith("data/") and name.endswith(".jsonl") and ("eval" in name or "manifest" in name)):
            continue
        count = 0
        with archive.open(name) as lines:
            for raw in lines:
                prompt = json.loads(raw).get("prompt")
                if isinstance(prompt, str):
                    exact.add(prompt)
                    normalized.add(normalized_prompt(prompt))
                    count += 1
        counts[name] = count
    return exact, normalized, counts


def parse_pairs(target: str) -> list[re.Match[str]]:
    if "<SEARCH" in target or "[RESULT:" in target:
        raise Reject("search_trace")
    pairs = list(PAIR.finditer(target))
    if not pairs:
        raise Reject("no_supported_trace")
    if len(pairs) > 8:
        raise Reject("over_tool_call_limit")
    if len(list(ANGLE.finditer(target))) != len(pairs) or len(list(RESULT.finditer(target))) != len(pairs):
        raise Reject("unpaired_or_unsupported_trace")
    for pair in pairs:
        op, a, b = pair.group("op", "a", "b")
        if op in {"SET", "MOV"} and (a is None or b is not None):
            raise Reject("invalid_arity")
        if op in {"INC", "DEC"} and (a is not None or b is not None):
            raise Reject("invalid_arity")
        if op in {"ADD", "SUB", "MUL", "DIV", "MOD"} and a is None:
            raise Reject("invalid_arity")
        if int(pair.group("value")) >= 2**32:
            raise Reject("invalid_observation")
    return pairs


def source_semantics(row: dict, pairs: list[re.Match[str]]) -> None:
    """Reject rows whose task answer cannot be checked independently."""
    prompt, target, source = row["prompt"], row["target"], row["source"]
    last = int(pairs[-1].group("value"))
    if source == "Algebraic_Composite_Aligned":
        m = ALGEBRAIC.search(prompt)
        if not m or not re.search(rf"答え:\s*{last}\s*$", target):
            raise Reject("unverified_final_answer")
        x, y, add, mod = map(int, m.groups())
        if mod == 0 or (x * y + add) % mod != last:
            raise Reject("problem_answer_mismatch")
    elif source == "Comparative_Conditional_Aligned":
        m = COMPARATIVE.search(prompt)
        if not m:
            raise Reject("unverified_task_form")
        place1, _, n1, place2, _, n2 = m.groups()
        n1, n2 = int(n1), int(n2)
        if n1 == n2:
            raise Reject("ambiguous_tie")
        winner = place1 if n1 > n2 else place2
        difference = abs(n1 - n2)
        if difference != last or not target.rstrip().endswith(f"答え: {winner}に{difference}個多く入っています。"):
            raise Reject("problem_answer_mismatch")
    elif source == "swe_alu_accumulate_sum":
        m = SWE.search(prompt)
        if not m:
            raise Reject("unverified_task_form")
        initial, count = map(int, m.groups())
        last_r0 = next((int(p.group("value")) for p in reversed(pairs) if p.group("observed") == "R0"), None)
        if count > 32 or (initial + count * (count - 1)) & 0xFFFFFFFF != last_r0:
            raise Reject("problem_answer_mismatch")
        if not re.search(rf"\s{last_r0}\s*$", target):
            raise Reject("unverified_final_answer")
    else:
        raise Reject("source_not_selected")


def operand_code(raw: str) -> str:
    if raw.startswith("R"):
        return f"R[{int(raw[1:]) % 8}]"
    return str(int(raw) & 0xFFFFFFFF)


def instruction_code(pair: re.Match[str], first: bool) -> str:
    op, dest_raw, a, b = pair.group("op", "dst", "a", "b")
    dest = int(dest_raw[1:]) % 8
    left = operand_code(a) if b is not None else f"R[{dest}]"
    right = operand_code(b) if b is not None else operand_code(a) if a is not None else None
    if op in {"SET", "MOV"}:
        expression = operand_code(a)
    elif op in {"INC", "DEC"}:
        expression = f"(R[{dest}] {'+' if op == 'INC' else '-'} 1) & 0xFFFFFFFF"
    else:
        symbol = {"ADD": "+", "SUB": "-", "MUL": "*", "DIV": "//", "MOD": "%"}[op]
        if op in {"DIV", "MOD"}:
            right = f"({right} if {right} != 0 else 1)"
        expression = f"({left} {symbol} {right}) & 0xFFFFFFFF"
    lines = ["R = [0] * 8"] if first else []
    lines += [f"R[{dest}] = {expression}", f'print(f"[R{dest}={{R[{dest}]}}]")']
    return "\n".join(lines) + "\n"


def convert_row(row: dict, controller, executor, vocab: dict) -> dict:
    from training_data_v44 import encode_supervision
    from tir_engine_v44r1 import observation_text

    if row.get("source") not in ACCEPTED_SOURCES:
        raise Reject("source_not_selected")
    prompt = row.get("prompt")
    target = row.get("target")
    if not isinstance(prompt, str) or not isinstance(target, str) or not isinstance(row.get("full_text"), str):
        raise Reject("invalid_row_schema")
    if not row["full_text"].startswith(prompt) or row["full_text"] not in (prompt + target, prompt + " " + target):
        raise Reject("unverified_prompt_boundary")
    original_suffix = row["full_text"][len(prompt):]
    if "```" in original_suffix:
        raise Reject("preexisting_code_or_observation_fence")
    pairs = parse_pairs(original_suffix)
    source_semantics(row, pairs)
    state = controller.initial_cpu_state()
    executor.reset()
    out: list[str] = []
    spans: list[dict] = []
    cursor = 0
    for index, pair in enumerate(pairs):
        command = pair.group(0).split("=>", 1)[0].rstrip()
        parsed = controller.parse_command(command)
        if parsed is None or parsed["raw"] != command:
            raise Reject("original_cpu_parse_mismatch")
        state, value, observed = controller.execute_step(state, parsed)
        if observed != f'[{pair.group("observed")}={pair.group("value")}]':
            raise Reject("original_cpu_observation_mismatch")
        code = instruction_code(pair, index == 0)
        result = executor.execute(code)
        if not result.get("success") or result.get("truncated") or not result.get("worker_alive"):
            raise Reject("python_tool_execution_failed")
        if result.get("stdout") != observed + "\n" or result.get("stderr"):
            raise Reject("python_tool_observation_mismatch")
        out.append(original_suffix[cursor:pair.start()])
        out.append("\n```python\n" + code + "```\n")
        offset = sum(map(len, out))
        obs = observation_text(result)
        out.append(obs)
        spans.append({"start": offset, "end": offset + len(obs), "role": "tool", "train": False})
        cursor = pair.end()
    out.append(original_suffix[cursor:])
    converted_target = "".join(out)
    if ANGLE.search(converted_target):
        raise Reject("legacy_trace_remains")
    item = {"prompt": prompt, "target": converted_target, "full_text": prompt + converted_target,
            "target_segments": spans, "source": "v43_verified_alu_to_pure_python",
            "original_source": row["source"], "original_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
    try:
        ids, tags = encode_supervision(item, vocab, max_positions=8192)
    except ValueError as exc:
        raise Reject("v44_supervision_reject:" + str(exc).split(":", 1)[0]) from exc
    if tags[-1] != 1 or any(any(tags[len(prompt) + s["start"]:len(prompt) + s["end"]]) for s in spans):
        raise AssertionError("Observation or EOS mask failure")
    return item


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--max-accepted", type=int, default=12)
    args = parser.parse_args()
    if not 1 <= args.limit <= 300000 or not 1 <= args.max_accepted <= 30000:
        raise ValueError("Prototype has strict sample caps")
    started = time.perf_counter()
    sys.path.insert(0, str(SOURCE_DIR))
    sys.path.insert(0, str(V44_DIR))
    from qwenputer_controller_v30 import QwenputerExecutionController
    from cpu_circuit_v30 import CompiledOperators32, ExactRAM8x32
    from tir_engine_v44r1 import PythonExecutor
    import torch
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    counters = collections.Counter()
    accepted_sources = collections.Counter()
    items: list[dict] = []
    output = args.output_dir / "converted_sample.jsonl"
    output_writer = output.open("w", encoding="utf-8", newline="\n")
    with zipfile.ZipFile(args.zip_path) as archive:
        # The imported circuit/controller must be byte-for-byte the source in
        # the actual archive whose training rows are being converted.
        original_cpu_sources = {}
        for name in ("qwenputer_controller_v30.py", "cpu_circuit_v30.py"):
            archived = archive.read(name)
            if archived != (SOURCE_DIR / name).read_bytes():
                raise RuntimeError("Extracted v43 CPU source differs from training ZIP: " + name)
            original_cpu_sources[name] = hashlib.sha256(archived).hexdigest()
        exact_eval, normalized_eval, eval_counts = evaluation_prompts(archive)
        metadata = json.loads(archive.read("base_metadata.json"))
        vocab = metadata["vocab"]
        cpu = QwenputerExecutionController(ExactRAM8x32(), CompiledOperators32())
        with PythonExecutor() as executor, archive.open(TRAIN_MEMBER) as lines:
            for line_number, raw in enumerate(lines, 1):
                if line_number > args.limit:
                    break
                if line_number % 128 == 0:
                    print(json.dumps({"rows_completed": line_number - 1, "accepted": len(items),
                                      "elapsed_seconds": round(time.perf_counter() - started, 3)}, ensure_ascii=False), flush=True)
                if line_number % 5000 == 0:
                    output_writer.flush()
                    partial = {"rows_completed": line_number - 1, "accepted": len(items), "counts": dict(counters),
                               "elapsed_seconds": round(time.perf_counter()-started, 3),
                               "status": "INCOMPLETE; only rows present in converted_sample.jsonl passed full CPU/Python verification"}
                    (args.output_dir / "progress.json").write_text(json.dumps(partial, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                row = json.loads(raw)
                prompt = row.get("prompt")
                if not isinstance(prompt, str):
                    counters["invalid_prompt"] += 1
                    continue
                if prompt in exact_eval:
                    counters["eval_prompt_exact_excluded"] += 1
                    continue
                if normalized_prompt(prompt) in normalized_eval:
                    counters["eval_prompt_normalized_excluded"] += 1
                    continue
                if len(items) >= args.max_accepted:
                    counters["accepted_cap_not_converted"] += 1
                    continue
                try:
                    item = convert_row(row, cpu, executor, vocab)
                except Reject as exc:
                    counters[str(exc)] += 1
                    continue
                item["source_ref"] = {"kind": "v43_zip_member", "file": TRAIN_MEMBER,
                                      "line": line_number,
                                      "raw_sha256": hashlib.sha256(raw).hexdigest()}
                items.append(item)
                counters["accepted"] += 1
                accepted_sources[row["source"]] += 1
                output_writer.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
    output_writer.close()
    with args.zip_path.open("rb") as source_file:
        source_sha256 = hashlib.file_digest(source_file, "sha256").hexdigest()
    report = {"source_zip": str(args.zip_path), "source_zip_sha256": source_sha256,
              "verified_original_cpu_source_sha256": original_cpu_sources,
              "source_member": TRAIN_MEMBER, "bounded_rows_examined": args.limit,
              "max_accepted": args.max_accepted, "counts": dict(counters),
              "elapsed_wall_seconds": round(time.perf_counter() - started, 3),
              "accepted_sources": dict(accepted_sources),
              "bundled_eval_prompt_counts": eval_counts, "bundled_eval_unique_exact": len(exact_eval),
              "converted_sample_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
              "limitations": ["Exact and NFKC-normalized prompt overlap only; semantic-family leakage is not certified.",
                              "Only three source families and nine simple ALU opcodes are considered.",
                              "CPU trace and independently recomputed final answer must agree; no training or model test performed."]}
    (args.output_dir / "admission_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"counts": report["counts"], "converted_sample_sha256": report["converted_sample_sha256"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
