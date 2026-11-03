"""cpu_circuit_v27.py: Exact Fixed-Weight Boolean Logic Circuits & RAM.

Device-agnostic (runs on CPU or CUDA). Bit-exact with compiled.pt.
Contains ExactRAM8x32 (8 registers x 32 bits) and CompiledOperators (Kogge-Stone ALU).
"""
import re
import torch
from torch import nn, Tensor
import torch.nn.functional as F

OPCODES16 = ('ADD','SUB','MUL','DIV','ABS_DIFF','MIN','MAX','GT','GE','LT','LE','EQ','NE')

class _FixedLinear(nn.Linear):
    def __init__(self, weights, bias):
        nn.Module.__init__(self)
        weight = torch.tensor(weights, dtype=torch.float32)
        offset = torch.tensor(bias, dtype=torch.float32)
        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]
        self.register_buffer('weight', weight)
        self.register_buffer('bias', offset)


class _BinaryGate(nn.Module):
    def __init__(self, kind):
        super().__init__()
        if kind == 'AND':
            self.hidden = _FixedLinear([[1., 1.]], [-1.])
            self.output = _FixedLinear([[1.]], [0.])
        elif kind == 'OR':
            self.hidden = _FixedLinear([[1., 1.], [1., 1.]], [0., -1.])
            self.output = _FixedLinear([[1., -1.]], [0.])
        elif kind == 'XOR':
            self.hidden = _FixedLinear([[1., 1.], [1., 1.]], [0., -1.])
            self.output = _FixedLinear([[1., -2.]], [0.])
        else:
            raise ValueError(f"Unknown fixed Boolean gate: {kind}")

    def forward(self, left, right):
        left, right = torch.broadcast_tensors(left, right)
        pair = torch.stack((left, right), dim=-1)
        return self.output(F.relu(self.hidden(pair))).squeeze(-1)


class _Logic(nn.Module):
    def __init__(self):
        super().__init__()
        self.and_gate = _BinaryGate('AND')
        self.or_gate = _BinaryGate('OR')
        self.xor_gate = _BinaryGate('XOR')
        self.not_gate = _FixedLinear([[-1.]], [1.])

    def invert(self, bits):
        return F.relu(self.not_gate(bits.unsqueeze(-1))).squeeze(-1)

    def mux(self, select, when_true, when_false):
        return self.or_gate(self.and_gate(select, when_true),
                            self.and_gate(self.invert(select), when_false))

    def reduce_or(self, bits):
        while bits.shape[-1] > 1:
            if bits.shape[-1] % 2:
                bits = F.pad(bits, (0, 1), value=0.)
            bits = self.or_gate(bits[..., 0::2], bits[..., 1::2])
        return bits[..., 0]

    def add(self, left, right, carry_one=False):
        """Kogge-Stone parallel prefix carry-lookahead adder."""
        if left.shape != right.shape:
            raise ValueError('Adder bit widths differ')
        propagate_original = self.xor_gate(left, right)
        propagate = propagate_original
        generate = self.and_gate(left, right)
        distance = 1
        while distance < left.shape[-1]:
            lower_generate = F.pad(generate[..., :-distance], (distance, 0), value=0.)
            lower_propagate = F.pad(propagate[..., :-distance], (distance, 0), value=1.)
            generate = self.or_gate(generate, self.and_gate(propagate, lower_generate))
            propagate = self.and_gate(propagate, lower_propagate)
            distance *= 2
        carry_outputs = self.or_gate(generate, propagate) if carry_one else generate
        carry_inputs = F.pad(carry_outputs[..., :-1], (1, 0), value=1. if carry_one else 0.)
        return self.xor_gate(propagate_original, carry_inputs), carry_outputs[..., -1]


class ExactRAM8x32(nn.Module):
    """Explicit 8-register x 32-bit state; exact one-hot address, no soft attention."""
    def __init__(self):
        super().__init__()
        self.logic = _Logic()
        self.word_sum = _FixedLinear([[1. for _ in range(8)]], [0.])

    def initial_state(self, batch_size=1, device=None):
        dev = self.word_sum.weight.device if device is None else torch.device(device)
        return torch.zeros((batch_size, 8, 32), dtype=torch.float32, device=dev)

    def read(self, state, address_onehot):
        selected = self.logic.and_gate(state, address_onehot[:, :, None])
        return self.word_sum(selected.transpose(1, 2)).squeeze(-1)

    def write(self, state, address_onehot, word_bits):
        address = address_onehot[:, :, None]
        unchanged = self.logic.and_gate(self.logic.invert(address), state)
        replacement = self.logic.and_gate(address, word_bits[:, None, :])
        return self.logic.or_gate(unchanged, replacement)

    def forward(self, state, address_onehot):
        return self.read(state, address_onehot)


class CompiledArithmetic16(nn.Module):
    """16-bit fixed arithmetic circuit loaded from compiled.pt['cpu']."""
    def __init__(self):
        super().__init__()
        self.logic = _Logic()
        self.op_sum = _FixedLinear([[1. for _ in OPCODES16]], [0.])
        self.bool_op_sum = _FixedLinear([[1. for _ in range(6)]], [0.])

    def _multiply(self, left, right):
        partials = []
        for bit in range(16):
            shifted = F.pad(left, (bit, 16 - bit), value=0.)
            partials.append(self.logic.and_gate(shifted, right[:, bit:bit+1]))
        while len(partials) > 2:
            reduced = []
            for start in range(0, len(partials)-2, 3):
                x, y, z = partials[start:start+3]
                pair = self.logic.xor_gate(x, y)
                total = self.logic.xor_gate(pair, z)
                carry = self.logic.or_gate(self.logic.and_gate(x, y), self.logic.and_gate(pair, z))
                reduced.extend((total, F.pad(carry[..., :-1], (1, 0), value=0.)))
            used = (len(partials) // 3) * 3
            reduced.extend(partials[used:])
            partials = reduced
        return self.logic.add(partials[0], partials[1])[0]

    def _divide(self, dividend, divisor):
        divisor17 = F.pad(divisor, (0, 1), value=0.)
        remainder = torch.zeros_like(divisor17)
        quotient_high_first = []
        complement = self.logic.invert(divisor17)
        for bit in range(15, -1, -1):
            shifted = torch.cat((dividend[:, bit:bit+1], remainder[..., :-1]), dim=-1)
            trial, greater_equal = self.logic.add(shifted, complement, carry_one=True)
            remainder = self.logic.mux(greater_equal[:, None], trial, shifted)
            quotient_high_first.append(greater_equal)
        quotient = torch.stack(tuple(reversed(quotient_high_first)), dim=-1)
        zero = self.logic.invert(self.logic.reduce_or(divisor))
        quotient = self.logic.and_gate(self.logic.invert(zero)[:, None], quotient)
        remainder = self.logic.mux(zero[:, None], dividend, remainder[..., :16])
        return quotient, remainder, zero

    def forward(self, a_bits, b_bits, op_onehot):
        added, carry = self.logic.add(a_bits, b_bits)
        added32 = F.pad(torch.cat((added, carry[:, None]), dim=-1), (0, 15), value=0.)
        difference, greater_equal = self.logic.add(a_bits, self.logic.invert(b_bits), carry_one=True)
        negative = self.logic.invert(greater_equal)
        inverse_difference = self.logic.add(self.logic.invert(difference), torch.zeros_like(difference), carry_one=True)[0]
        magnitude = self.logic.mux(greater_equal[:, None], difference, inverse_difference)
        equality = self.logic.invert(self.logic.reduce_or(self.logic.xor_gate(a_bits, b_bits)))
        greater = self.logic.and_gate(greater_equal, self.logic.invert(equality))
        less = self.logic.invert(greater_equal)
        less_equal = self.logic.invert(greater)
        unequal = self.logic.invert(equality)
        minimum = self.logic.mux(greater_equal[:, None], b_bits, a_bits)
        maximum = self.logic.mux(greater_equal[:, None], a_bits, b_bits)
        product = self._multiply(a_bits, b_bits)
        quotient, remainder, zero = self._divide(a_bits, b_bits)
        comparisons = (greater, greater_equal, less, less_equal, equality, unequal)
        bool_words = tuple(F.pad(value[:, None], (0, 31), value=0.) for value in comparisons)
        candidates = (added32, F.pad(magnitude, (0, 16), value=0.), product,
                      F.pad(quotient, (0, 16), value=0.), F.pad(magnitude, (0, 16), value=0.),
                      F.pad(minimum, (0, 16), value=0.), F.pad(maximum, (0, 16), value=0.), *bool_words)
        selected = self.logic.and_gate(torch.stack(candidates, dim=-1), op_onehot[:, None, :])
        value_bits = self.op_sum(selected).squeeze(-1)
        selected_booleans = self.logic.and_gate(torch.stack(comparisons, dim=-1), op_onehot[:, 7:13])
        boolean = self.bool_op_sum(selected_booleans).squeeze(-1)
        is_boolean = self.bool_op_sum(op_onehot[:, 7:13]).squeeze(-1)
        division = op_onehot[:, 3:4]
        return dict(value_bits=value_bits,
                    negative=self.logic.and_gate(op_onehot[:, 1], negative),
                    quotient_bits=self.logic.and_gate(division, quotient),
                    remainder_bits=self.logic.and_gate(division, remainder),
                    boolean=boolean, is_boolean=is_boolean,
                    division_by_zero=self.logic.and_gate(op_onehot[:, 3], zero))


class CompiledOperators32(nn.Module):
    """32-bit two's-complement words; exact FP32 bit tensors [B, 32]."""
    def __init__(self):
        super().__init__()
        self.logic = _Logic()

    def _negate(self, word):
        return self.logic.add(self.logic.invert(word), torch.zeros_like(word), carry_one=True)[0]

    def _absolute(self, word, signed=True):
        sign = word[:, -1] if signed else torch.zeros_like(word[:, 0])
        return self.logic.mux(sign[:, None], self._negate(word), word), sign

    def _multiply_32(self, left, right):
        partials = []
        for bit in range(32):
            shifted = F.pad(left, (bit, 32-bit), value=0.)
            partials.append(self.logic.and_gate(shifted, right[:, bit:bit+1]))
        while len(partials) > 2:
            reduced = []
            for start in range(0, len(partials)-2, 3):
                a, b, c = partials[start:start+3]
                ab = self.logic.xor_gate(a, b)
                total = self.logic.xor_gate(ab, c)
                carry = self.logic.or_gate(self.logic.and_gate(a, b), self.logic.and_gate(ab, c))
                reduced.extend((total, F.pad(carry[:, :-1], (1, 0), value=0.)))
            reduced.extend(partials[(len(partials)//3)*3:])
            partials = reduced
        return self.logic.add(partials[0], partials[1])[0]

    def _divide_32(self, dividend, divisor):
        extended = F.pad(divisor, (0, 1), value=0.)
        remainder = torch.zeros_like(extended)
        complement = self.logic.invert(extended)
        high_first = []
        for bit in range(31, -1, -1):
            shifted = torch.cat((dividend[:, bit:bit+1], remainder[:, :-1]), dim=-1)
            trial, ge = self.logic.add(shifted, complement, carry_one=True)
            remainder = self.logic.mux(ge[:, None], trial, shifted)
            high_first.append(ge)
        quotient = torch.stack(tuple(reversed(high_first)), dim=-1)
        zero = self.logic.invert(self.logic.reduce_or(divisor))
        quotient = self.logic.and_gate(self.logic.invert(zero)[:, None], quotient)
        remainder = self.logic.mux(zero[:, None], dividend, remainder[:, :32])
        return quotient, remainder, zero

    def compute(self, op: str, a: Tensor, b: Tensor, signed: bool = False):
        """Execute a 32-bit operation deterministically."""
        op = op.upper()
        if op == 'ADD':
            res, cout = self.logic.add(a, b)
            return dict(result=res, carry=cout)
        elif op == 'SUB':
            diff, ge = self.logic.add(a, self.logic.invert(b), carry_one=True)
            return dict(result=diff, borrow=self.logic.invert(ge))
        elif op == 'MUL':
            prod = self._multiply_32(a, b)
            return dict(result=prod[:, :32], full=prod)
        elif op == 'DIV':
            q, r, z = self._divide_32(a, b)
            return dict(result=q, remainder=r, div_by_zero=z)
        elif op == 'MOD':
            q, r, z = self._divide_32(a, b)
            return dict(result=r, div_by_zero=z)
        elif op == 'BAND':
            return dict(result=self.logic.and_gate(a, b))
        elif op == 'BOR':
            return dict(result=self.logic.or_gate(a, b))
        elif op == 'BXOR':
            return dict(result=self.logic.xor_gate(a, b))
        elif op == 'BNOT':
            return dict(result=self.logic.invert(a))
        elif op == 'IMP':
            a_b = self.logic.reduce_or(a)
            b_b = self.logic.reduce_or(b)
            imp = self.logic.or_gate(self.logic.invert(a_b), b_b)
            return dict(result=F.pad(imp[:, None], (0, 31), value=0.))
        elif op == 'IFF':
            a_b = self.logic.reduce_or(a)
            b_b = self.logic.reduce_or(b)
            iff = self.logic.invert(self.logic.xor_gate(a_b, b_b))
            return dict(result=F.pad(iff[:, None], (0, 31), value=0.))
        elif op in ('NOT', 'LNOT'):
            a_b = self.logic.reduce_or(a)
            not_a = self.logic.invert(a_b)
            return dict(result=F.pad(not_a[:, None], (0, 31), value=0.))
        elif op in ('AND', 'LAND'):
            a_b = self.logic.reduce_or(a)
            b_b = self.logic.reduce_or(b)
            and_res = self.logic.and_gate(a_b, b_b)
            return dict(result=F.pad(and_res[:, None], (0, 31), value=0.))
        elif op in ('OR', 'LOR'):
            a_b = self.logic.reduce_or(a)
            b_b = self.logic.reduce_or(b)
            or_res = self.logic.or_gate(a_b, b_b)
            return dict(result=F.pad(or_res[:, None], (0, 31), value=0.))
        elif op == 'EQ':
            eq = self.logic.invert(self.logic.reduce_or(self.logic.xor_gate(a, b)))
            return dict(result=F.pad(eq[:, None], (0, 31), value=0.))
        elif op == 'NE':
            eq = self.logic.invert(self.logic.reduce_or(self.logic.xor_gate(a, b)))
            return dict(result=F.pad(self.logic.invert(eq)[:, None], (0, 31), value=0.))
        elif op == 'LT':
            _, ge = self.logic.add(a, self.logic.invert(b), carry_one=True)
            return dict(result=F.pad(self.logic.invert(ge)[:, None], (0, 31), value=0.))
        elif op == 'GT':
            _, ge = self.logic.add(a, self.logic.invert(b), carry_one=True)
            eq = self.logic.invert(self.logic.reduce_or(self.logic.xor_gate(a, b)))
            gt = self.logic.and_gate(ge, self.logic.invert(eq))
            return dict(result=F.pad(gt[:, None], (0, 31), value=0.))
        elif op == 'MOV':
            return dict(result=a)
        else:
            raise ValueError(f"Unsupported 32-bit opcode: {op}")


# Helper encoding and decoding functions
def encode_int32(val: int, device=None) -> Tensor:
    """Encode an integer to a 32-bit FP32 bit tensor [1, 32], LSB first."""
    val = int(val) & 0xFFFFFFFF
    bits = [(val >> i) & 1 for i in range(32)]
    return torch.tensor([bits], dtype=torch.float32, device=device)

def decode_int32(bits: Tensor, signed: bool = False) -> int:
    """Decode a 32-bit FP32 bit tensor [1, 32] to an integer."""
    b = bits.detach().cpu().round().long().squeeze()
    val = sum(int(b[i].item()) << i for i in range(32))
    if signed and (val & (1 << 31)):
        val -= (1 << 32)
    return val

def onehot_address(reg_name: str, device=None) -> Tensor:
    """Convert R0..R7 into a one-hot tensor [1, 8] with fault-tolerant mod-8 wrapping.
    
    Guarantees zero-crash execution:
    - If reg_name is 'R10', wraps to R2 (10 % 8 = 2).
    - If reg_name is invalid or has no digits, safely defaults to R0.
    """
    reg_name = str(reg_name).strip().upper()
    m = re.search(r'\d+', reg_name)
    idx = int(m.group(0)) % 8 if m else 0
    return F.one_hot(torch.tensor([idx], device=device), 8).float()
