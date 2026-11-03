# -*- coding: utf-8 -*-
"""
integrated_model_v44fix.py: v43 model + Python TIR as a THIRD tool (additive, not a replacement).

Guarantees:
1. Inherits IntegratedModelV42 unchanged: the fixed ALU (<SET>/<ADD>/<MUL>/...) and <SEARCH> keep working
   exactly as in v43. Nothing is removed.
2. Python: when the model itself closes a ```python ... ``` block, the code runs in the bounded worker
   (tir_engine_v44r1.PythonExecutor) and the observation is injected with the SAME text protocol the
   public TIR data uses (tir_engine_v44r1.PYTHON_BLOCK / observation_text). Code inside the PROMPT is
   never executed (SWE prompts contain ```python``` listings as input).
3. With no Python block emitted, the token stream is identical to generate_with_qwenputer_v42.
4. Optional v30 static-loop guard (8 tokens x4 / 16 tokens x4). Off by default so evaluation numbers stay
   comparable with v43; turn it on for interactive use.
"""
import re
from typing import Optional, Dict, Any, List

import torch

from integrated_model_v42 import IntegratedModelV42, SEARCH_PATTERN
from qwenputer_controller_v30 import CPUState, canonical_normalize_text, COMMAND_PATTERN
from tir_engine_v44r1 import PythonExecutor, PYTHON_BLOCK, FORGED_OUTPUT, observation_text


class IntegratedModelV44Fix(IntegratedModelV42):

    def generate_v44fix(
        self,
        prompt: str,
        vocab: Dict[str, int],
        inv_vocab: Dict[int, str],
        max_tokens: Optional[int] = None,
        cpu_state: Optional[CPUState] = None,
        arm: str = 'learned',
        repetition_penalty: float = 1.0,
        enable_search: bool = True,
        enable_python: bool = True,
        loop_guard: bool = False,
        max_python_calls: int = 8,
        python_timeout: float = 5.0,
        python_startup_timeout: float = 60.0,
        python_max_output_chars: int = 1024,
        executor: Optional[PythonExecutor] = None,
    ) -> Dict[str, Any]:
        device = next(self.parameters()).device
        cpu_dev = torch.device('cpu')
        if self.ram.word_sum.weight.device.type != 'cpu':
            self.ram.to(cpu_dev)
            self.alu.to(cpu_dev)
        if cpu_state is None:
            cpu_state = self.controller.initial_cpu_state(batch_size=1, device=cpu_dev)
        elif cpu_state.registers.device.type != 'cpu':
            cpu_state = cpu_state.clone()
            cpu_state.registers = cpu_state.registers.to(cpu_dev)

        cpu_invocations = 0
        search_invocations = 0
        search_trace: List[Dict[str, str]] = []
        python_trace: List[Dict[str, Any]] = []
        prompt = canonical_normalize_text(prompt)

        # 1. Pre-execute CPU commands already present in the prompt (identical to v42)
        matches = list(COMMAND_PATTERN.finditer(prompt))
        if matches:
            parts, last = [], 0
            for m in matches:
                parts.append(prompt[last:m.start()])
                parsed = self.controller.parse_command(m.group(0))
                if parsed is not None:
                    cpu_state, _, res_str = self.controller.execute_step(cpu_state, parsed)
                    cpu_invocations += 1
                    parts.append(f"{m.group(0)} => {res_str}")
                else:
                    parts.append(m.group(0))
                last = m.end()
            parts.append(prompt[last:])
            prompt = "".join(parts)

        # 2. Pre-execute SEARCH commands already present in the prompt (identical to v42)
        if enable_search:
            s_matches = list(SEARCH_PATTERN.finditer(prompt))
            if s_matches:
                parts, last = [], 0
                for sm in s_matches:
                    parts.append(prompt[last:sm.start()])
                    query = sm.group(1).strip()
                    res_snippet = self.search_engine.search(query)
                    search_invocations += 1
                    search_trace.append({"query": query, "result": res_snippet})
                    parts.append(f"{sm.group(0)} => {res_snippet}")
                    last = sm.end()
                parts.append(prompt[last:])
                prompt = "".join(parts)

        input_ids = [vocab[c] for c in prompt if c in vocab]
        if not input_ids:
            input_ids = [1]

        use_amp = (device.type == 'cuda')
        prompt_char_len = len(prompt)
        generated_tokens: List[int] = []
        full_text = prompt
        model_text = ""          # characters the MODEL sampled (no injected observations), for reporting
        py_scan = 0              # generated-text index right after the last injected python observation
        in_cmd = False
        cmd_buffer = ""
        ended_with_eos = False
        stop_reason = "token_limit"
        owned_executor = executor is None
        tool = executor

        layout_chars = set(" \n\t|")
        layout_toks = {vocab[c] for c in layout_chars if c in vocab}
        effective_horizon = int(self.config.max_positions) if max_tokens is None else int(max_tokens)

        def inject(state, text, normalize):
            # ALU/SEARCH: byte-for-byte the v42 behaviour (raw text, out-of-vocab chars skipped).
            # Python: normalized like the training loader normalizes full_text.
            src = canonical_normalize_text(text) if normalize else text
            toks = [vocab[c] for c in src if c in vocab]
            for i, itok in enumerate(toks):
                state = self.step_state(torch.tensor([[itok]], dtype=torch.long, device=device), state,
                                        compute_logits=(i == len(toks) - 1))
                generated_tokens.append(itok)
            return state, (src if not normalize else "".join(inv_vocab.get(t, "") for t in toks))

        try:
            with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=use_amp, dtype=torch.bfloat16):
                t = torch.tensor([input_ids], dtype=torch.long, device=device)
                state = self.prefill(t, arm=arm)

                for _ in range(effective_horizon):
                    if state.compact_prefix.shape[1] >= (self.config.max_positions - 5):
                        stop_reason = "context_limit"
                        break

                    logits_1d = state.last_logits[0, -1].float().cpu()
                    if repetition_penalty != 1.0 and generated_tokens:
                        for prev_tok in set(generated_tokens[-16:]):
                            if prev_tok in layout_toks:
                                continue
                            val = float(logits_1d[prev_tok])
                            logits_1d[prev_tok] = val / repetition_penalty if val > 0.0 else val * repetition_penalty
                        last_tok = generated_tokens[-1]
                        is_quad_repeat = len(generated_tokens) >= 4 and len(set(generated_tokens[-4:])) == 1
                        if last_tok not in layout_toks or is_quad_repeat:
                            val = float(logits_1d[last_tok])
                            logits_1d[last_tok] = val / repetition_penalty if val > 0.0 else val * repetition_penalty

                    next_tok = int(logits_1d.argmax().item())
                    if next_tok == 3:  # EOS
                        ended_with_eos = True
                        stop_reason = "eos"
                        break

                    char = inv_vocab.get(next_tok, "")
                    generated_tokens.append(next_tok)
                    full_text += char
                    model_text += char

                    # ---- tool 1/2: ALU and SEARCH, unchanged from v42 ----
                    if char == '<':
                        in_cmd = True
                        cmd_buffer = "<"
                    elif in_cmd:
                        cmd_buffer += char
                        if (len(cmd_buffer) == 2 and not ('A' <= char <= 'Z')) or ('\n' in cmd_buffer) or (len(cmd_buffer) > 64):
                            in_cmd = False
                            cmd_buffer = ""
                        elif char == '>':
                            in_cmd = False
                            if enable_search and cmd_buffer.startswith("<SEARCH"):
                                m_s = re.match(r"<SEARCH[ \t]+([^<>]+)>", cmd_buffer)
                                if m_s:
                                    query = m_s.group(1).strip()
                                    state = self.step_state(torch.tensor([[next_tok]], dtype=torch.long, device=device), state, compute_logits=False)
                                    res_snippet = self.search_engine.search(query)
                                    search_invocations += 1
                                    search_trace.append({"query": query, "result": res_snippet})
                                    state, injected = inject(state, f" => {res_snippet} ", normalize=False)
                                    full_text += injected
                                    continue
                            parsed = self.controller.parse_command(cmd_buffer)
                            if parsed is not None:
                                state = self.step_state(torch.tensor([[next_tok]], dtype=torch.long, device=device), state, compute_logits=False)
                                cpu_state, _, res_str = self.controller.execute_step(cpu_state, parsed)
                                cpu_invocations += 1
                                state, injected = inject(state, f" => {res_str} ", normalize=False)
                                full_text += injected
                                continue

                    # ---- tool 3: Python, same trigger/observation protocol as tir_engine_v44r1 ----
                    # Scanned on the generated text INCLUDING injected observations, from the end of the last
                    # observation, so a ```python block that follows an observation directly is still found.
                    if enable_python and not in_cmd:
                        gen_text = full_text[prompt_char_len:]
                        if FORGED_OUTPUT.search(gen_text[py_scan:]):
                            stop_reason = "model_generated_observation"
                            break
                        match = PYTHON_BLOCK.search(gen_text, py_scan)
                        if match:
                            if len(python_trace) >= max_python_calls:
                                stop_reason = "python_call_limit"
                                break
                            state = self.step_state(torch.tensor([[next_tok]], dtype=torch.long, device=device), state, compute_logits=False)
                            if tool is None:
                                try:
                                    tool = PythonExecutor(python_timeout, python_max_output_chars, 32768,
                                                          startup_timeout_seconds=python_startup_timeout)
                                except (TimeoutError, RuntimeError):
                                    stop_reason = "tool_worker_startup_error"
                                    break
                            result = tool.execute(match.group(1))
                            obs = observation_text(result)
                            if state.compact_prefix.shape[1] + len(obs) >= self.config.max_positions - 5:
                                stop_reason = "context_limit"
                                break
                            state, injected = inject(state, obs, normalize=True)
                            full_text += injected
                            python_trace.append({"code": match.group(1), "success": bool(result.get("success")),
                                                 "stdout": result.get("stdout", ""), "error": result.get("error")})
                            py_scan = len(full_text) - prompt_char_len
                            if not result.get("worker_alive", True):   # e.g. killed by a timeout
                                if owned_executor:
                                    tool.close()
                                    tool = None                      # next block starts a fresh worker
                                else:
                                    stop_reason = "tool_worker_terminated"
                                    break
                            continue

                    # ---- optional v30 static-loop guard ----
                    if loop_guard and not in_cmd:
                        n_gen = len(generated_tokens)
                        if n_gen >= 32 and generated_tokens[-8:] == generated_tokens[-16:-8] == generated_tokens[-24:-16] == generated_tokens[-32:-24]:
                            stop_reason = "loop_guard"
                            break
                        if n_gen >= 64 and generated_tokens[-16:] == generated_tokens[-32:-16] == generated_tokens[-48:-32] == generated_tokens[-64:-48]:
                            stop_reason = "loop_guard"
                            break

                    state = self.step_state(torch.tensor([[next_tok]], dtype=torch.long, device=device), state, compute_logits=True)
        finally:
            if owned_executor and tool is not None:
                tool.close()

        return {
            "output_text": full_text[prompt_char_len:],
            "full_text": full_text,
            "model_generated_text": model_text,
            "generated_tokens": generated_tokens,
            "cpu_state": cpu_state,
            "cpu_invocations": cpu_invocations,
            "cpu_trace": cpu_state.trace,
            "search_invocations": search_invocations,
            "search_trace": search_trace,
            "python_invocations": len(python_trace),
            "python_trace": python_trace,
            "ended_with_eos": ended_with_eos,
            "stop_reason": stop_reason,
        }
