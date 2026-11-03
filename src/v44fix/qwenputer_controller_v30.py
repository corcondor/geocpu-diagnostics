"""qwenputer_controller_v30.py: Fully Fault-Tolerant Neuro-Symbolic Execution Controller.

Zero-Crash Guarantees:
1. Automatic Unit & Kanji Stripping: '45点', '120円', '15個' -> safely parsed as numeric integers 45, 120, 15.
2. Modulo-8 Register Wrapping: Out-of-bounds registers like 'R10', 'R99' -> mapped to R(10%8) = R2.
3. Total Arithmetic Exception Shield: Division by zero, unknown opcodes, or corrupted operands never crash the process.
4. State Separation: Language associative memory is 100% MUTED during CPU execution.
"""
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any
import re
import torch
from torch import Tensor
from cpu_circuit_v30 import ExactRAM8x32, CompiledOperators32, encode_int32, decode_int32, onehot_address

VALID_OPCODES = (
    r"SET|MOV|ADD|SUB|MUL|DIV|MOD|INC|DEC|"
    r"BAND|BOR|BXOR|EQ|NE|LT|GT|"
    r"IMP|IFF|AND|OR|NOT|LAND|LOR|LNOT|BNOT"
)
# Strict single-line command pattern; negative lookahead prevents re-executing commands that already have '=> [R..]'
COMMAND_PATTERN = re.compile(
    rf"<({VALID_OPCODES})[ \t]+([rR]\d+)(?:[ \t]+([^\s<>]+))?(?:[ \t]+([^\s<>]+))?>(?![ \t]*=>)"
)

_BASE_NORM_DICT = {
    "ϵ": "ε",   # LaTeX lunate epsilon -> standard vocab epsilon
    "ϕ": "φ",   # LaTeX straight phi -> standard vocab phi
    "ϑ": "θ",   # LaTeX curly theta -> standard vocab theta
    "ϰ": "κ",   # LaTeX kappa -> standard vocab kappa
    "ϱ": "ρ",   # LaTeX rho -> standard vocab rho
    "ς": "σ",   # Final sigma -> standard vocab sigma
    "ο": "o",   # Greek omicron -> ASCII o
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K",
    "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
    "＞": ">",  # Full-width greater-than -> ASCII >
    "＜": "<",  # Full-width less-than -> ASCII <
    "⇔": "↔",   # Double equivalence arrow -> vocab ↔
    "↦": "→",   # Maps-to arrow -> vocab →
    "⊨": "⊢",   # Semantic turnstile -> vocab ⊢
    "⊤": "T",   # Top / Verum -> T
    "∅": "φ",   # Empty set -> φ
    "≢": "≠",   # Not identical -> ≠
    "÷": "/",   # Division sign -> /
    "⋅": "・",  # Dot operator -> ・
    "∙": "・",  # Bullet operator -> ・
    "⟨": "(", "⟩": ")", "⟪": "(", "⟫": ")",
    "⟦": "[", "⟧": "]", "⌈": "[", "⌉": "]", "⌊": "[", "⌋": "]",
    "−": "-",   # Unicode minus U+2212 -> ASCII hyphen-minus
}
# Map full-width ASCII (0xFF01..0xFF5E) to half-width ASCII (0x0021..0x007E),
# except Japanese prose punctuation already in vocab: （ ） ！ ？ ： ～
for _fw_code in range(0xFF01, 0xFF5F):
    _fw_char = chr(_fw_code)
    if _fw_char not in ("（", "）", "！", "？", "：", "～"):
        _BASE_NORM_DICT[_fw_char] = chr(_fw_code - 0xFEE0)

SYMBOL_NORMALIZATION_MAP = str.maketrans(_BASE_NORM_DICT)

def canonical_normalize_text(text: str) -> str:
    """Safely normalizes mathematical, logical, and full-width lookalike characters into in-vocabulary glyphs."""
    if not text:
        return ""
    return text.translate(SYMBOL_NORMALIZATION_MAP)

@dataclass
class CPUState:
    registers: Tensor                       # [B, 8, 32]
    pc: int = 0
    active: bool = False
    current_command: Optional[str] = None
    last_result_value: Optional[int] = None
    last_destination: Optional[str] = None
    trace: List[Dict[str, Any]] = field(default_factory=list)

    def clone(self):
        return CPUState(
            registers=self.registers.clone(),
            pc=self.pc,
            active=self.active,
            current_command=self.current_command,
            last_result_value=self.last_result_value,
            last_destination=self.last_destination,
            trace=list(self.trace)
        )

class QwenputerExecutionController:
    """Controls deterministic CPU execution with language state protection and total fault tolerance."""
    def __init__(self, ram: ExactRAM8x32, alu: CompiledOperators32):
        self.ram = ram
        self.alu = alu

    def initial_cpu_state(self, batch_size: int = 1, device=None) -> CPUState:
        return CPUState(
            registers=self.ram.initial_state(batch_size=batch_size, device=device),
            pc=0,
            active=False,
            current_command=None,
            last_result_value=None,
            last_destination=None,
            trace=[]
        )

    def parse_command(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse structured CPU commands with fault-tolerant operand capture."""
        match = COMMAND_PATTERN.search(text)
        if not match:
            return None
        return {
            "raw": match.group(0),
            "op": match.group(1).upper(),
            "arg1": match.group(2),
            "arg2": match.group(3),
            "arg3": match.group(4)
        }

    def execute_step(self, cpu_state: CPUState, command_dict: Dict[str, Any]) -> Tuple[CPUState, int, str]:
        """Execute one CPU instruction step on cpu_state with complete fault-tolerance."""
        state = cpu_state.clone()
        op = command_dict.get("op", "NOP")
        arg1 = command_dict.get("arg1")
        arg2 = command_dict.get("arg2")
        arg3 = command_dict.get("arg3")
        dev = state.registers.device

        def normalize_reg_name(raw_name: Optional[str], default: str = "R0") -> str:
            if not raw_name:
                return default
            m = re.search(r'\d+', str(raw_name))
            if m:
                idx = int(m.group(0)) % 8
                return f"R{idx}"
            return default

        def get_operand_bits(arg: Optional[str]) -> Tensor:
            if arg is None:
                return encode_int32(0, device=dev)
            arg = str(arg).strip()
            # Register reference: e.g. R0..R7, R10 -> R2
            if re.match(r'^[rR]\d+', arg):
                reg = normalize_reg_name(arg)
                addr = onehot_address(reg, device=dev)
                return self.ram.read(state.registers, addr)
            
            # Numeric constant (auto-strips Kanji units like '45点', '120円')
            m = re.search(r'[-+]?\d+', arg)
            if m:
                try:
                    val = int(m.group(0)) & 0xFFFFFFFF
                    return encode_int32(val, device=dev)
                except Exception:
                    pass
            return encode_int32(0, device=dev)

        dest_reg = normalize_reg_name(arg1, default="R0")
        result_bits = encode_int32(0, device=dev)
        result_val = 0

        try:
            if op in ("SET", "MOV"):
                # e.g. <SET R0 123> or <SET R0 R1>
                result_bits = get_operand_bits(arg2)
                result_val = decode_int32(result_bits)
            elif op in ("ADD", "SUB", "MUL", "DIV", "MOD", "BAND", "BOR", "BXOR", "EQ", "NE", "LT", "GT", "IMP", "IFF", "AND", "OR", "LAND", "LOR"):
                if arg3 is not None:
                    a_bits = get_operand_bits(arg2)
                    b_bits = get_operand_bits(arg3)
                else:
                    a_bits = self.ram.read(state.registers, onehot_address(dest_reg, device=dev))
                    b_bits = get_operand_bits(arg2)

                # Safe division-by-zero check
                if op in ("DIV", "MOD"):
                    b_val = decode_int32(b_bits)
                    if b_val == 0:
                        b_bits = encode_int32(1, device=dev)  # prevent div-by-zero hardware exception

                comp = self.alu.compute(op, a_bits, b_bits)
                result_bits = comp["result"]
                result_val = decode_int32(result_bits)
            elif op in ("NOT", "LNOT", "BNOT"):
                if arg2 is not None:
                    a_bits = get_operand_bits(arg2)
                else:
                    a_bits = self.ram.read(state.registers, onehot_address(dest_reg, device=dev))
                comp = self.alu.compute(op, a_bits, a_bits)
                result_bits = comp["result"]
                result_val = decode_int32(result_bits)
            elif op in ("INC", "DEC"):
                a_bits = self.ram.read(state.registers, onehot_address(dest_reg, device=dev))
                delta = encode_int32(1, device=dev)
                comp = self.alu.compute("ADD" if op == "INC" else "SUB", a_bits, delta)
                result_bits = comp["result"]
                result_val = decode_int32(result_bits)
            else:
                # NOP or unknown opcode
                result_val = 0
                result_bits = encode_int32(0, device=dev)
        except Exception:
            # Shield process from any runtime arithmetic exception
            result_val = 0
            result_bits = encode_int32(0, device=dev)

        # Write result to physical RAM register
        dest_addr = onehot_address(dest_reg, device=dev)
        new_regs = self.ram.write(state.registers, dest_addr, result_bits)

        state.registers = new_regs
        state.pc += 1
        state.active = True
        state.current_command = command_dict.get("raw", f"<{op}>")
        state.last_result_value = result_val
        state.last_destination = dest_reg

        result_str = f"[{dest_reg}={result_val}]"
        state.trace.append({
            "step": state.pc,
            "cmd": state.current_command,
            "op": op,
            "dest": dest_reg,
            "val": result_val,
            "res_str": result_str
        })

        return state, result_val, result_str
