"""Read-only, fail-closed v43 SWE ALU trace to pure Python TIR converter.

The only code sent to the v44 executor is assembled from fully parsed,
allowlisted ALU commands. The Python shown in v43 prompts is parsed as data,
never executed. Every admitted row is checked against the original v43 CPU.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import sys
import time
import zipfile
from pathlib import Path

from src.pure_v44_candidates.v43_alu_to_python.convert_sample import (
    SOURCE_DIR, V44_DIR, TRAIN_MEMBER, evaluation_prompts, normalized_prompt,
    Reject,
)

SOURCE_FAMILIES = frozenset({
    "swe_alu_accumulate_sum", "swe_alu_modulo_loop", "swe_alu_multiply_power",
    "swe_alu_parity_branch", "swe_alu_conditional_filter_loop",
    "swe_alu_bxor_product", "swe_alu_modulo_expr", "swe_alu_conditional_filter_py",
})
PAIR = re.compile(
    r"<(?P<op>SET|MOV|ADD|SUB|MUL|DIV|MOD|INC|DEC|BXOR)\s+"
    r"(?P<dst>R\d+)(?:\s+(?P<a>R\d+|[+-]?\d+))?"
    r"(?:\s+(?P<b>R\d+|[+-]?\d+))?>\s*=>\s*"
    r"\[(?P<observed>R\d+)=(?P<value>\d+)\]"
)
CODE_FENCE = re.compile(r"```python\n(?P<code>.*?)\n```\n最終値 R0 =$", re.S)
M = 0xFFFFFFFF


def prompt_program(prompt: str) -> str:
    matches = list(CODE_FENCE.finditer(prompt))
    if len(matches) != 1 or matches[0].end() != len(prompt):
        raise Reject("unverified_prompt_program_boundary")
    return matches[0].group("code")


def task_result(source: str, code: str) -> tuple[int, list[int]]:
    """Independent whitelist parser for the eight exact v43 SWE task forms.

    Returns the final R0 and the expected R0 writes by each source statement.
    The latter detects targets that enumerate non-executed conditional branches.
    """
    def match(pattern: str):
        found = re.fullmatch(pattern, code)
        if not found:
            raise Reject("unverified_task_form")
        return tuple(map(int, found.groups()))

    if source == "swe_alu_accumulate_sum":
        initial, count, stride = match(r"R0 = (\d+)\nfor i in range\((\d+)\):\n    R0 \+= i \* (\d+)")
        if count > 32: raise Reject("task_loop_limit")
        values = [initial] + [initial + stride * i * (i + 1) // 2 for i in range(count)]
        return values[-1], values
    if source == "swe_alu_modulo_loop":
        initial, count, add, modulus = match(r"R0 = (\d+)\nfor i in range\((\d+)\):\n    R0 = \(R0 \+ (\d+)\) % (\d+)")
        if count > 32 or modulus == 0: raise Reject("task_loop_or_zero_modulus")
        values = [initial]
        for _ in range(count): values.append((values[-1] + add) % modulus)
        return values[-1], values
    if source == "swe_alu_multiply_power":
        initial, count, factor = match(r"R0 = (\d+)\nfor _ in range\((\d+)\):\n    R0 \*= (\d+)")
        if count > 32: raise Reject("task_loop_limit")
        values = [initial]
        for _ in range(count): values.append(values[-1] * factor)
        return values[-1], values
    if source == "swe_alu_modulo_expr":
        initial, a, b, modulus = match(r"R0 = \((\d+) \+ (\d+) \* (\d+)\) % (\d+)")
        if modulus == 0: raise Reject("task_zero_modulus")
        return (initial + a * b) % modulus, [(initial + a * b) % modulus]
    if source == "swe_alu_bxor_product":
        a, b, c, d = match(r"R0 = \((\d+) \* (\d+)\) \^ \((\d+) \* (\d+)\)")
        return (a * b) ^ (c * d), [(a * b) ^ (c * d)]
    if source == "swe_alu_parity_branch":
        initial, x, even_factor, y, odd_factor = match(
            r"R0 = (\d+)\nif R0 % 2 == 0: R0 \+= (\d+) \* (\d+)\n"
            r"else: R0 \+= (\d+) \* (\d+)"
        )
        if x != y: raise Reject("unverified_parity_branch_operands")
        result = initial + x * (even_factor if initial % 2 == 0 else odd_factor)
        return result, [initial, result]
    if source in {"swe_alu_conditional_filter_loop", "swe_alu_conditional_filter_py"}:
        # Parse the limited generated if/else language without executing text.
        lines = code.split("\n")
        first = re.fullmatch(r"R0 = (\d+)", lines[0]) if lines else None
        if first is None: raise Reject("unverified_task_form")
        current = int(first.group(1)); values = [current]; i = 1
        while i < len(lines):
            cond = re.fullmatch(r"if (\d+) % 2 == 0: R0 \+= (\d+)", lines[i])
            if cond is None: raise Reject("unverified_task_form")
            index, plus = map(int, cond.groups()); i += 1
            minus = None
            if i < len(lines) and lines[i].startswith("else:"):
                other = re.fullmatch(r"else: R0 -= (\d+)", lines[i])
                if other is None: raise Reject("unverified_task_form")
                minus = int(other.group(1)); i += 1
            if index % 2 == 0: current += plus
            elif minus is not None: current -= minus
            values.append(current)
            if len(values) > 33: raise Reject("task_loop_limit")
        return current, values
    raise Reject("source_not_selected")


def parse_swe_suffix(suffix: str):
    """Require nothing except whitespace, complete CPU pairs, and final int."""
    pairs = list(PAIR.finditer(suffix))
    if not pairs: raise Reject("no_supported_trace")
    cursor = 0
    for pair in pairs:
        if suffix[cursor:pair.start()].strip(): raise Reject("unpaired_or_unsupported_trace")
        if pair.group("observed") != f"R{int(pair.group('dst')[1:]) % 8}":
            raise Reject("wrong_observed_register")
        if int(pair.group("value")) > M: raise Reject("invalid_observation")
        op, a, b = pair.group("op", "a", "b")
        if op in {"SET", "MOV"} and (a is None or b is not None): raise Reject("invalid_arity")
        if op in {"INC", "DEC"} and (a is not None or b is not None): raise Reject("invalid_arity")
        if op in {"ADD", "SUB", "MUL", "DIV", "MOD", "BXOR"} and a is None: raise Reject("invalid_arity")
        cursor = pair.end()
    tail = suffix[cursor:]
    final = re.fullmatch(r"\s+(-?\d+)\s*", tail)
    if final is None: raise Reject("unverified_final_answer")
    return pairs, int(final.group(1))


def validate_r0_writes(source: str, pairs, expected_values: list[int]) -> None:
    writes = [int(p.group("value")) for p in pairs if p.group("observed") == "R0"]
    if source == "swe_alu_modulo_loop":
        # Each source assignment compiles to ADD followed by MOD on R0.
        if len(writes) != 1 + 2 * (len(expected_values)-1):
            raise Reject("task_trace_structure_mismatch")
        writes = writes[:1] + writes[2::2]
    if source == "swe_alu_modulo_expr":
        # One source expression compiles to SET, ADD, MOD; require the MOD result.
        if len(writes) != 3: raise Reject("task_trace_structure_mismatch")
        writes = writes[-1:]
    if writes != [value & M for value in expected_values]:
        raise Reject("task_trace_r0_mismatch")


def expected_commands(source: str, code: str) -> list[str]:
    """Derive the exact ALU program from the limited prompt grammar alone."""
    def args(pattern: str) -> tuple[int, ...]:
        found = re.fullmatch(pattern, code)
        if found is None: raise Reject("unverified_task_form")
        return tuple(map(int, found.groups()))

    if source == "swe_alu_accumulate_sum":
        initial, count, stride = args(r"R0 = (\d+)\nfor i in range\((\d+)\):\n    R0 \+= i \* (\d+)")
        commands = [f"<SET R0 {initial}>",f"<SET R1 {count}>","<SET R2 0>"]
        for _ in range(count):
            commands += ["<ADD R0 R0 R2>",f"<ADD R2 R2 {stride}>","<DEC R1>"]
        return commands
    if source == "swe_alu_modulo_loop":
        initial, count, add, modulus = args(r"R0 = (\d+)\nfor i in range\((\d+)\):\n    R0 = \(R0 \+ (\d+)\) % (\d+)")
        commands = [f"<SET R0 {initial}>",f"<SET R1 {count}>"]
        for _ in range(count):
            commands += [f"<ADD R0 R0 {add}>",f"<MOD R0 R0 {modulus}>","<DEC R1>"]
        return commands
    if source == "swe_alu_multiply_power":
        initial, count, factor = args(r"R0 = (\d+)\nfor _ in range\((\d+)\):\n    R0 \*= (\d+)")
        commands = [f"<SET R0 {initial}>",f"<SET R1 {count}>"]
        for _ in range(count): commands += [f"<MUL R0 R0 {factor}>","<DEC R1>"]
        return commands
    if source == "swe_alu_modulo_expr":
        initial, a, b, modulus = args(r"R0 = \((\d+) \+ (\d+) \* (\d+)\) % (\d+)")
        return [f"<SET R0 {initial}>",f"<MUL R2 {a} {b}>","<ADD R0 R0 R2>",f"<MOD R0 R0 {modulus}>"]
    if source == "swe_alu_bxor_product":
        a,b,c,d = args(r"R0 = \((\d+) \* (\d+)\) \^ \((\d+) \* (\d+)\)")
        return [f"<MUL R1 {a} {b}>",f"<MUL R2 {c} {d}>","<BXOR R0 R1 R2>"]
    if source == "swe_alu_parity_branch":
        initial, x, even_factor, y, odd_factor = args(
            r"R0 = (\d+)\nif R0 % 2 == 0: R0 \+= (\d+) \* (\d+)\n"
            r"else: R0 \+= (\d+) \* (\d+)"
        )
        if x != y: raise Reject("unverified_parity_branch_operands")
        factor = even_factor if initial % 2 == 0 else odd_factor
        return [f"<SET R0 {initial}>","<MOD R2 R0 2>",f"<MUL R3 {x} {factor}>",
                "<SET R1 1>","<ADD R0 R0 R3>","<DEC R1>"]
    if source in {"swe_alu_conditional_filter_loop", "swe_alu_conditional_filter_py"}:
        lines = code.split("\n")
        first = re.fullmatch(r"R0 = (\d+)",lines[0]) if lines else None
        if first is None: raise Reject("unverified_task_form")
        operations=[];i=1
        while i<len(lines):
            test=re.fullmatch(r"if (\d+) % 2 == 0: R0 \+= (\d+)",lines[i])
            if test is None: raise Reject("unverified_task_form")
            index,plus=map(int,test.groups());i+=1
            minus=None
            if i<len(lines) and lines[i].startswith("else:"):
                alt=re.fullmatch(r"else: R0 -= (\d+)",lines[i])
                if alt is None:raise Reject("unverified_task_form")
                minus=int(alt.group(1));i+=1
            if index%2==0:operations.append(f"<ADD R0 R0 {plus}>")
            elif minus is not None:operations.append(f"<SUB R0 R0 {minus}>")
        commands=[f"<SET R0 {int(first.group(1))}>",f"<SET R1 {len(operations)}>"]
        for operation in operations:commands += [operation,"<DEC R1>"]
        return commands
    raise Reject("source_not_selected")


def operand(raw: str) -> str:
    return f"R[{int(raw[1:]) % 8}]" if raw.startswith("R") else str(int(raw) & M)


def instruction(pair) -> str:
    op, dst, a, b = pair.group("op", "dst", "a", "b")
    d = int(dst[1:]) % 8
    left = operand(a) if b is not None else f"R[{d}]"
    right = operand(b) if b is not None else operand(a) if a is not None else None
    if op in {"SET", "MOV"}: value = operand(a)
    elif op in {"INC", "DEC"}: value = f"(R[{d}]{'+' if op=='INC' else '-'}1)&M"
    else:
        symbol = {"ADD":"+", "SUB":"-", "MUL":"*", "DIV":"//", "MOD":"%", "BXOR":"^"}[op]
        if op in {"DIV", "MOD"}: right = f"({right} or 1)"
        value = f"({left}{symbol}{right})&M"
    return f"R[{d}]={value};p({d})"


def convert_swe(row: dict, raw: bytes, line_number: int, cpu, executor, vocab: dict) -> dict:
    from training_data_v44 import encode_supervision
    from tir_engine_v44r1 import observation_text, PYTHON_BLOCK

    source = row.get("source")
    if source not in SOURCE_FAMILIES: raise Reject("source_not_selected")
    prompt, target, full = row.get("prompt"), row.get("target"), row.get("full_text")
    if not all(isinstance(x, str) for x in (prompt, target, full)):
        raise Reject("invalid_row_schema")
    if full not in (prompt + target, prompt + " " + target):
        raise Reject("unverified_prompt_boundary")
    suffix = full[len(prompt):]
    code = prompt_program(prompt)
    task_final, expected_values = task_result(source, code)
    pairs, stated_final = parse_swe_suffix(suffix)
    actual_commands = [pair.group(0).split("=>",1)[0].rstrip() for pair in pairs]
    if actual_commands != expected_commands(source, code):
        raise Reject("task_instruction_trace_mismatch")
    if stated_final != task_final or stated_final < 0 or stated_final > M:
        raise Reject("task_final_mismatch")
    validate_r0_writes(source, pairs, expected_values)
    state = cpu.initial_cpu_state()
    for pair in pairs:
        command = pair.group(0).split("=>", 1)[0].rstrip()
        parsed = cpu.parse_command(command)
        if parsed is None or parsed["raw"] != command:
            raise Reject("original_cpu_parse_mismatch")
        state, _, observed = cpu.execute_step(state, parsed)
        if observed != f'[{pair.group("observed")}={pair.group("value")}]':
            raise Reject("original_cpu_observation_mismatch")
    executor.reset()
    program = "R=[0]*8\nM=4294967295\ndef p(i):print(f'[R{i}={R[i]}]')\n"
    program += "\n".join(instruction(pair) for pair in pairs) + "\n"
    result = executor.execute(program)
    expected_stdout = "".join(f'[{p.group("observed")}={p.group("value")}]\n' for p in pairs)
    if not result.get("success") or result.get("truncated") or not result.get("worker_alive"):
        raise Reject("python_tool_execution_failed")
    if result.get("stdout") != expected_stdout or result.get("stderr"):
        raise Reject("python_tool_observation_mismatch")
    prefix = suffix[:pairs[0].start()]
    trailing = suffix[pairs[-1].end():]
    # The native TIR reader only recognizes a Python fence at line start.
    # Preserve the original leading space at the prompt/answer boundary, then
    # start a new generated line before the fence.
    answer = prefix + "\n```python\n" + program + "```\n"
    blocks = list(PYTHON_BLOCK.finditer(answer))
    if len(blocks) != 1 or blocks[0].group(1) != program:
        raise Reject("v44_native_python_fence_not_detected")
    start = len(answer)
    observation = observation_text(result)
    answer += observation + trailing
    item = {
        "prompt": prompt, "target": answer, "full_text": prompt + answer,
        "target_segments": [{"start": start, "end": start + len(observation), "role": "tool", "train": False}],
        "source": "v43_verified_swe_to_pure_python", "original_source": source,
        "original_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "source_ref": {"kind": "v43_zip_member", "file": TRAIN_MEMBER, "line": line_number,
                       "raw_sha256": hashlib.sha256(raw).hexdigest()},
    }
    try: ids, tags = encode_supervision(item, vocab, max_positions=2048)
    except ValueError as exc: raise Reject("v44_2048_supervision_reject:" + str(exc).split(":",1)[0]) from exc
    if tags[-1] != 1 or any(tags[len(prompt)+start:len(prompt)+start+len(observation)]):
        raise AssertionError("Observation/EOS mask failure")
    return item


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=268032)
    parser.add_argument("--sources", nargs="*", choices=sorted(SOURCE_FAMILIES))
    args = parser.parse_args()
    if not 1 <= args.limit <= 268032: raise ValueError("Invalid scan limit")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(SOURCE_DIR));sys.path.insert(0,str(V44_DIR))
    from qwenputer_controller_v30 import QwenputerExecutionController
    from cpu_circuit_v30 import CompiledOperators32, ExactRAM8x32
    from tir_engine_v44r1 import PythonExecutor
    import torch
    torch.set_num_threads(1);torch.set_grad_enabled(False)
    started = time.perf_counter(); counts=collections.Counter(); sources=collections.Counter(); examined=0
    with zipfile.ZipFile(args.zip_path) as archive:
        for name in ("qwenputer_controller_v30.py", "cpu_circuit_v30.py"):
            if archive.read(name)!=(SOURCE_DIR/name).read_bytes():
                raise RuntimeError("Original CPU source ZIP mismatch: "+name)
        eval_exact, eval_norm, _ = evaluation_prompts(archive)
        vocab=json.loads(archive.read("base_metadata.json"))["vocab"]
        cpu=QwenputerExecutionController(ExactRAM8x32(),CompiledOperators32())
        out_path=args.output_dir/"converted_swe.jsonl"
        with PythonExecutor(startup_timeout_seconds=180.) as executor, archive.open(TRAIN_MEMBER) as lines, out_path.open("w",encoding="utf-8",newline="\n") as out:
            for number,raw in enumerate(lines,1):
                if number>args.limit:break
                row=json.loads(raw);source=row.get("source")
                if source not in SOURCE_FAMILIES or (args.sources and source not in args.sources):continue
                examined+=1
                prompt=row.get("prompt")
                if prompt in eval_exact or (isinstance(prompt,str) and normalized_prompt(prompt) in eval_norm):
                    counts["eval_prompt_excluded"]+=1;continue
                try:item=convert_swe(row,raw,number,cpu,executor,vocab)
                except Reject as exc:counts[str(exc)]+=1;continue
                out.write(json.dumps(item,ensure_ascii=False,separators=(",",":"))+"\n")
                counts["accepted"]+=1;sources[source]+=1
                if examined%1000==0:
                    out.flush();print(json.dumps({"examined":examined,"accepted":counts["accepted"],"seconds":round(time.perf_counter()-started,2)}),flush=True)
    manifest={"status":"complete","source_zip_sha256":hashlib.sha256(args.zip_path.read_bytes()).hexdigest(),
              "member":TRAIN_MEMBER,"max_row_number":min(args.limit,268032),"candidate_rows_examined":examined,
              "counts":dict(counts),"accepted_by_source":dict(sources),"output_sha256":hashlib.sha256(out_path.read_bytes()).hexdigest(),
              "output_bytes":out_path.stat().st_size,"elapsed_wall_seconds":round(time.perf_counter()-started,3),
              "limits":{"context_positions":2048,"executor_calls_per_row":1,"model_updates":0}}
    (args.output_dir/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(manifest,ensure_ascii=True),flush=True)


if __name__=="__main__":main()
