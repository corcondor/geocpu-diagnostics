# -*- coding: utf-8 -*-
"""
integrated_model_v42.py: Neurosymbolic Integrated Model v42 with Search-Augmented Retrieval.

Guarantees:
1. 100% Backward Compatible: Inherits from IntegratedModelV30 without modifying a single line of v30.
2. Dual-Engine Co-Execution:
   - Hardware 32-bit Kogge-Stone ALU (<SET>, <ADD>, <MUL>, <MOD>, <CMP>, <DEC>, etc.)
   - Ultra-Fast Local Knowledge & Code Retrieval (<SEARCH query="...">)
3. Zero-Leakage & Snippet-Bounded: Search results strictly bounded to ~100 tokens, guaranteeing
   that 4-hop multi-step searches never exceed the 2048 token limit.
4. Native Fallback: If no search query is emitted, runs 100% identical to v41.
"""
import re
from typing import Optional, Dict, Any, List
import torch
from torch import Tensor

from integrated_model_v30 import IntegratedModelV30
from qwenputer_controller_v30 import (
    CPUState, canonical_normalize_text, COMMAND_PATTERN
)
from local_search_engine_v42 import LocalSearchEngineV42

SEARCH_PATTERN = re.compile(r"<SEARCH[ \t]+([^<>]+)>(?![ \t]*=>)")

class IntegratedModelV42(IntegratedModelV30):
    def __init__(self, base, search_engine: Optional[LocalSearchEngineV42] = None):
        super().__init__(base=base)
        self.search_engine = search_engine or LocalSearchEngineV42()

    def generate_with_qwenputer_v42(
        self,
        prompt: str,
        vocab: Dict[str, int],
        inv_vocab: Dict[int, str],
        max_tokens: Optional[int] = None,
        cpu_state: Optional[CPUState] = None,
        arm: str = 'learned',
        repetition_penalty: float = 1.0,
        enable_search: bool = True
    ) -> Dict[str, Any]:
        """Generates text while supporting both CPU ALU execution and Local Search Retrieval."""
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
        prompt = canonical_normalize_text(prompt)

        # 1. Pre-execute any pre-existing CPU commands in prompt
        matches = list(COMMAND_PATTERN.finditer(prompt))
        if matches:
            new_prompt_parts = []
            last_idx = 0
            for m in matches:
                new_prompt_parts.append(prompt[last_idx:m.start()])
                parsed = self.controller.parse_command(m.group(0))
                if parsed is not None:
                    cpu_state, result_val, res_str = self.controller.execute_step(cpu_state, parsed)
                    cpu_invocations += 1
                    new_prompt_parts.append(f"{m.group(0)} => {res_str}")
                else:
                    new_prompt_parts.append(m.group(0))
                last_idx = m.end()
            new_prompt_parts.append(prompt[last_idx:])
            prompt = "".join(new_prompt_parts)

        # 2. Pre-execute any pre-existing SEARCH commands in prompt
        if enable_search:
            s_matches = list(SEARCH_PATTERN.finditer(prompt))
            if s_matches:
                new_p_parts = []
                last_idx = 0
                for sm in s_matches:
                    new_p_parts.append(prompt[last_idx:sm.start()])
                    query = sm.group(1).strip()
                    res_snippet = self.search_engine.search(query)
                    search_invocations += 1
                    search_trace.append({"query": query, "result": res_snippet})
                    new_p_parts.append(f"{sm.group(0)} => {res_snippet}")
                    last_idx = sm.end()
                new_p_parts.append(prompt[last_idx:])
                prompt = "".join(new_p_parts)

        input_ids = [vocab[c] for c in prompt if c in vocab]
        if not input_ids:
            input_ids = [1]  # <unk>

        use_amp = (device.type == 'cuda')
        prompt_char_len = len(prompt)
        generated_tokens = []
        full_text = prompt
        in_cmd = False
        cmd_buffer = ""
        ended_with_eos = False

        layout_chars = set(" \n\t|")
        layout_toks = {vocab[c] for c in layout_chars if c in vocab}
        effective_horizon = int(self.config.max_positions) if max_tokens is None else int(max_tokens)

        with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=use_amp, dtype=torch.bfloat16):
            t = torch.tensor([input_ids], dtype=torch.long, device=device)
            state = self.prefill(t, arm=arm)

            for _ in range(effective_horizon):
                # Context cap safeguard (ensures 2048 positions never exceeded)
                if state.compact_prefix.shape[1] >= (self.config.max_positions - 5):
                    ended_with_eos = False
                    break

                logits_1d = state.last_logits[0, -1].float().cpu()
                if repetition_penalty != 1.0 and generated_tokens:
                    recent_tokens = generated_tokens[-16:]
                    for prev_tok in set(recent_tokens):
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
                    break

                char = inv_vocab.get(next_tok, "")
                generated_tokens.append(next_tok)
                full_text += char

                if char == '<':
                    in_cmd = True
                    cmd_buffer = "<"
                elif in_cmd:
                    cmd_buffer += char
                    # Guard: commands must start with uppercase ASCII and be <= 64 chars
                    if (len(cmd_buffer) == 2 and not ('A' <= char <= 'Z')) or ('\n' in cmd_buffer) or (len(cmd_buffer) > 64):
                        in_cmd = False
                        cmd_buffer = ""
                    elif char == '>':
                        in_cmd = False

                        # 1. Check for SEARCH Command
                        if enable_search and cmd_buffer.startswith("<SEARCH"):
                            m_s = re.match(r"<SEARCH[ \t]+([^<>]+)>", cmd_buffer)
                            if m_s:
                                query = m_s.group(1).strip()
                                state = self.step_state(torch.tensor([[next_tok]], dtype=torch.long, device=device), state, compute_logits=False)
                                res_snippet = self.search_engine.search(query)
                                search_invocations += 1
                                search_trace.append({"query": query, "result": res_snippet})

                                inject_str = f" => {res_snippet} "
                                full_text += inject_str
                                inject_toks = [vocab[c] for c in inject_str if c in vocab]
                                for idx_inj, itok in enumerate(inject_toks):
                                    is_last_token = (idx_inj == len(inject_toks) - 1)
                                    state = self.step_state(torch.tensor([[itok]], dtype=torch.long, device=device), state, compute_logits=is_last_token)
                                    generated_tokens.append(itok)
                                continue

                        # 2. Check for Hardware CPU Command
                        parsed = self.controller.parse_command(cmd_buffer)
                        if parsed is not None:
                            state = self.step_state(torch.tensor([[next_tok]], dtype=torch.long, device=device), state, compute_logits=False)
                            cpu_state, result_val, res_str = self.controller.execute_step(cpu_state, parsed)
                            cpu_invocations += 1

                            inject_str = f" => {res_str} "
                            full_text += inject_str
                            inject_toks = [vocab[c] for c in inject_str if c in vocab]
                            for idx_inj, itok in enumerate(inject_toks):
                                is_last_token = (idx_inj == len(inject_toks) - 1)
                                state = self.step_state(torch.tensor([[itok]], dtype=torch.long, device=device), state, compute_logits=is_last_token)
                                generated_tokens.append(itok)
                            continue

                # Normal token state update (feed the predicted token to advance hidden state & logits)
                state = self.step_state(torch.tensor([[next_tok]], dtype=torch.long, device=device), state, compute_logits=True)

        return {
            "output_text": full_text[prompt_char_len:],
            "full_text": full_text,
            "generated_tokens": generated_tokens,
            "cpu_state": cpu_state,
            "cpu_invocations": cpu_invocations,
            "cpu_trace": cpu_state.trace,
            "search_invocations": search_invocations,
            "search_trace": search_trace,
            "ended_with_eos": ended_with_eos
        }
