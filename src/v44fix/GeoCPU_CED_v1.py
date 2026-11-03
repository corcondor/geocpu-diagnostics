"""GeoCPU_CED_v1: causal encoder, query-independent Delta tape, execution decoder.

Research prototype, NOT a pretrained CED or a replica of YOCO / Gated DeltaNet.
The write tape is constructed once and reused by every decoder layer/iteration.
Default chunk backend uses PyTorch matmul + triangular solves, not a fused CUDA kernel.
"""
from __future__ import annotations

from dataclasses import dataclass, replace, asdict
from typing import Optional, Callable, List
import math
import torch
from fast_encoder_scan import triton_recurrent_scan, torch_chunked_scan, HAVE_TRITON
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class CEDConfig:
    vocab_size: int = 3892
    d_model: int = 256
    encoder_layers: int = 4
    decoder_layers: int = 6
    ffn_dim: int = 1024
    heads: int = 4
    key_dim: int = 64
    value_dim: int = 64
    max_positions: int = 256
    position_mode: str = 'none'  # none, frozen, learned; migration must select explicitly
    head_bias: bool = False
    chunk_size: int = 32
    alpha_init: float = 0.999
    beta_init: float = 0.5
    read_scale: float = 1.0
    scan_backend: str = 'auto'  # auto / jit / triton / chunked (frozen encoder only)

    def __post_init__(self):
        for name in ('vocab_size', 'd_model', 'encoder_layers', 'decoder_layers',
                     'ffn_dim', 'heads', 'key_dim', 'value_dim', 'max_positions', 'chunk_size'):
            if getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive')
        if self.heads * self.key_dim != self.d_model or self.heads * self.value_dim != self.d_model:
            raise ValueError('Stage 1 requires heads*key_dim = heads*value_dim = d_model')
        if self.scan_backend not in ('auto', 'jit', 'triton', 'chunked'):
            raise ValueError('Unsupported encoder scan backend')
        if self.position_mode not in ('none', 'frozen', 'learned'):
            raise ValueError('Unknown position_mode')
        if not (0 < self.alpha_init < 1 and 0 < self.beta_init < 1):
            raise ValueError('Initial gates must lie strictly between zero and one')


def accumulator_dtype(x: Tensor) -> torch.dtype:
    return torch.float64 if x.dtype == torch.float64 else torch.float32


def check_delta_inputs(k: Tensor, v: Tensor, log_alpha: Tensor, beta: Tensor, m: Tensor):
    if k.ndim != 4 or v.ndim != 4 or k.shape[:3] != v.shape[:3]:
        raise ValueError('k/v must be [B,T,H,K/V] with identical B,T,H')
    if log_alpha.shape != k.shape[:3] or beta.shape != k.shape[:3]:
        raise ValueError('log_alpha and beta must be [B,T,H]')
    if m.shape != (k.shape[0], k.shape[2], k.shape[3], v.shape[3]):
        raise ValueError('initial memory must be [B,H,K,V]')
    if k.shape[1] == 0:
        raise ValueError('Empty sequences are not supported')


def delta_step(k: Tensor, v: Tensor, log_alpha: Tensor, beta: Tensor, memory: Tensor) -> Tensor:
    """One write. Shapes k/v [B,H,K/V], gates [B,H], memory [B,H,K,V].
    No in-place update; differentiable through state. k must be unit-normalized
    for the usual bounded directional-overwrite interpretation.
    """
    with torch.autocast(device_type=k.device.type, enabled=False):
        dt = accumulator_dtype(k)
        k, v, g, b, m = k.to(dt), v.to(dt), log_alpha.to(dt), beta.to(dt), memory.to(dt)
        decayed = g.exp()[..., None, None] * m
        error = v - torch.einsum('bhkv,bhk->bhv', decayed, k)
        return decayed + b[..., None, None] * k[..., :, None] * error[..., None, :]


@dataclass(frozen=True)
class DeltaChunk:
    start: int
    stop: int
    keys: Tensor          # [B,H,C,K]
    innovations: Tensor   # [B,H,C,V], query-independent
    decay: Tensor         # [B,H,C]
    causal_decay: Tensor  # [B,H,C,C], zero for j>i
    initial_memory: Tensor

    def read(self, q: Tensor) -> Tensor:
        # q [B,C,H,K]; every position reads ITS OWN prefix state, not final M_T.
        with torch.autocast(device_type=q.device.type, enabled=False):
            qh = q.to(self.keys.dtype).transpose(1, 2)
            base = (qh @ self.initial_memory) * self.decay[..., None]
            local = ((qh @ self.keys.transpose(-1, -2)) * self.causal_decay) @ self.innovations
            return (base + local).transpose(1, 2)


@dataclass(frozen=True)
class DeltaTape:
    chunks: tuple[DeltaChunk, ...]
    final_memory: Tensor
    length: int
    scale: float = 1.0
    batched_kh: Optional[Tensor] = None
    batched_u: Optional[Tensor] = None
    batched_decay: Optional[Tensor] = None
    batched_ratios: Optional[Tensor] = None
    batched_mem: Optional[Tensor] = None
    chunk_size: int = 32

    def read(self, q: Tensor) -> Tensor:
        if q.ndim != 4 or q.shape[1] != self.length:
            raise ValueError('Query length must equal tape length')
        if self.batched_kh is not None:
            with torch.autocast(device_type=q.device.type, enabled=False):
                B, T, H, K = q.shape
                V = self.batched_u.shape[-1]
                C = self.chunk_size
                rem = T % C
                pad_len = (C - rem) % C
                qp = F.pad(q, (0, 0, 0, 0, 0, pad_len)) if pad_len > 0 else q
                nc = qp.shape[1] // C
                qh = qp.to(self.batched_kh.dtype).view(B, nc, C, H, K).permute(0, 1, 3, 2, 4)
                base = (qh @ self.batched_mem) * self.batched_decay.unsqueeze(-1)
                local = ((qh @ self.batched_kh.transpose(-1, -2)) * self.batched_ratios) @ self.batched_u
                out = (base + local).permute(0, 1, 3, 2, 4).reshape(B, nc * C, H, V)
                return (out[:, :T] if pad_len > 0 else out) * self.scale
        return torch.cat([c.read(q[:, c.start:c.stop]) for c in self.chunks], dim=1) * self.scale


@dataclass(frozen=True)
class SequentialTape:
    """Small-test oracle only: explicitly stores every prefix matrix."""
    states: tuple[Tensor, ...]
    final_memory: Tensor
    scale: float = 1.0

    def read(self, q: Tensor) -> Tensor:
        if q.shape[1] != len(self.states):
            raise ValueError('Query length mismatch')
        with torch.autocast(device_type=q.device.type, enabled=False):
            if q.shape[1] == 1:
                m = self.states[0]
                return (q[:, 0].to(m.dtype).unsqueeze(-2) @ m).squeeze(-2).unsqueeze(1) * self.scale
            return torch.stack([torch.einsum('bhk,bhkv->bhv',
                q[:, t].to(m.dtype), m) for t, m in enumerate(self.states)], dim=1) * self.scale


_MASK_CACHE: dict[tuple, tuple[Tensor, Tensor, Tensor]] = {}


def _get_tri_masks(chunk_size: int, dt: torch.dtype, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    key = (chunk_size, dt, device)
    if key not in _MASK_CACHE:
        causal = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=device).tril()
        strict = causal & ~torch.eye(chunk_size, dtype=torch.bool, device=device)
        eye = torch.eye(chunk_size, dtype=dt, device=device)
        _MASK_CACHE[key] = (causal, strict, eye)
    return _MASK_CACHE[key]


def build_delta_tape(k: Tensor, v: Tensor, log_alpha: Tensor, beta: Tensor,
                     initial_memory: Tensor, chunk_size: int = 32, scale: float = 1.0) -> DeltaTape:
    """Exact algebraic chunk factorization (vectorized across all chunks).

    L_ij = delta_ij + 1[j<i] beta_i exp(d_i-d_j) <k_i,k_j>
    R_i = beta_i (v_i - exp(d_i) k_i^T M_start)
    L U = R.  U is shared by all future reads of this tape.
    """
    check_delta_inputs(k, v, log_alpha, beta, initial_memory)
    if chunk_size <= 0:
        raise ValueError('chunk_size must be positive')
    with torch.autocast(device_type=k.device.type, enabled=False):
        dt = accumulator_dtype(k)
        k, v, g, b = (x.to(dt) for x in (k, v, log_alpha, beta))
        memory = initial_memory.to(dt)
        B, T, H, K = k.shape
        V = v.shape[-1]
        rem = T % chunk_size
        pad_len = (chunk_size - rem) % chunk_size
        if pad_len > 0:
            kp = F.pad(k, (0, 0, 0, 0, 0, pad_len))
            vp = F.pad(v, (0, 0, 0, 0, 0, pad_len))
            gp = F.pad(g, (0, 0, 0, pad_len))
            bp = F.pad(b, (0, 0, 0, pad_len))
        else:
            kp, vp, gp, bp = k, v, g, b

        Tp = kp.shape[1]
        nc = Tp // chunk_size
        kh_all = kp.view(B, nc, chunk_size, H, K).permute(0, 1, 3, 2, 4).contiguous()
        vh_all = vp.view(B, nc, chunk_size, H, V).permute(0, 1, 3, 2, 4).contiguous()
        gh_all = gp.view(B, nc, chunk_size, H).permute(0, 1, 3, 2).contiguous()
        bh_all = bp.view(B, nc, chunk_size, H).permute(0, 1, 3, 2).contiguous()

        d_all = gh_all.cumsum(-1)
        decay_all = d_all.exp()
        causal, strict, eye = _get_tri_masks(chunk_size, dt, k.device)

        diff_all = d_all.unsqueeze(-1) - d_all.unsqueeze(-2)
        ratios_all = torch.where(causal, diff_all, torch.zeros_like(diff_all)).exp() * causal
        gram_all = kh_all @ kh_all.transpose(-1, -2)
        lower_all = eye + bh_all.unsqueeze(-1) * ratios_all * gram_all * strict

        # Single batched triangular solve across all chunks simultaneously
        rhs_v = bh_all.unsqueeze(-1) * vh_all
        rhs_k = (bh_all * decay_all).unsqueeze(-1) * kh_all
        rhs_cat = torch.cat([rhs_v, rhs_k], dim=-1)
        uk_cat = torch.linalg.solve_triangular(lower_all, rhs_cat, upper=False, unitriangular=True)
        u_v = uk_cat[..., :V]
        u_k = uk_cat[..., V:]

        mem_list = []
        u_list = []
        decay_last = decay_all[..., -1, None, None]
        ratios_last = ratios_all[..., -1, :, None]
        kh_t = kh_all.transpose(-1, -2)
        for c in range(nc):
            mem_list.append(memory)
            u = u_v[:, c] - u_k[:, c] @ memory
            u_list.append(u)
            memory = decay_last[:, c] * memory + kh_t[:, c] @ (ratios_last[:, c] * u)

        mem_stack = torch.stack(mem_list, dim=1)
        u_stack = torch.stack(u_list, dim=1)

    return DeltaTape(
        chunks=(),
        final_memory=memory,
        length=T,
        scale=scale,
        batched_kh=kh_all,
        batched_u=u_stack,
        batched_decay=decay_all,
        batched_ratios=ratios_all,
        batched_mem=mem_stack,
        chunk_size=chunk_size
    )


def build_sequential_tape(k: Tensor, v: Tensor, log_alpha: Tensor, beta: Tensor,
                          initial_memory: Tensor, scale: float = 1.0) -> SequentialTape:
    check_delta_inputs(k, v, log_alpha, beta, initial_memory)
    states, m = [], initial_memory
    for t in range(k.shape[1]):
        m = delta_step(k[:, t], v[:, t], log_alpha[:, t], beta[:, t], m)
        states.append(m)
    return SequentialTape(tuple(states), m, scale)


@torch.jit.script
def encoder_recurrent_scan(gamma: Tensor, kv: Tensor, initial_state: Tensor) -> tuple[Tensor, Tensor]:
    # TorchScript removes Python interpreter dispatch, not the time dependency.
    # No claim that this compiles into a single fused CUDA kernel.
    s = initial_state
    outputs = torch.jit.annotate(List[Tensor], [])
    for t in range(kv.size(1)):
        s = gamma[:, t] * s + kv[:, t]
        outputs.append(s)
    return torch.stack(outputs, dim=1), s


class EncoderBlock(nn.Module):
    """Clean block topology, with explicit initial/final recurrence state.
    Layer parameters keep the existing Clean names. TorchScript scan is a reference
    path, NOT claimed to be a fused/parallel CUDA implementation.
    """
    def __init__(self, cfg: CEDConfig):
        super().__init__()
        d = cfg.d_model
        self.scan_backend = cfg.scan_backend
        self.norm1 = nn.LayerNorm(d)
        self.proj_q = nn.Linear(d, d, bias=False)
        self.proj_k = nn.Linear(d, d, bias=False)
        self.proj_v = nn.Linear(d, d, bias=False)
        self.log_decay = nn.Parameter(torch.linspace(-0.001, -0.30, d)[None, None])
        self.proj_adapt = nn.Linear(d, d)
        self.out_norm = nn.RMSNorm(d)
        self.proj_out = nn.Linear(d, d, bias=False)
        self.norm2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, cfg.ffn_dim), nn.GELU(), nn.Linear(cfg.ffn_dim, d))
        self.gate = nn.Parameter(torch.full((1, 1, d), 0.1))

    def forward(self, x: Tensor, initial_state: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        h = self.norm1(x)
        q, k, v = self.proj_q(h), self.proj_k(h), self.proj_v(h)
        gamma = (self.log_decay.exp() * (0.98 + 0.02 * self.proj_adapt(h).sigmoid())).clamp(0.70, 0.999)
        kv = k * v
        s = torch.zeros_like(kv[:, 0]) if initial_state is None else initial_state
        # Forward-only Triton when the whole encoder is frozen.  The existing
        # TorchScript recurrence remains authoritative for trainable encoders.
        frozen = not (gamma.requires_grad or kv.requires_grad or s.requires_grad)
        if frozen and kv.is_cuda and self.scan_backend in ('auto', 'triton') and HAVE_TRITON:
            states, s = triton_recurrent_scan(gamma, kv, s)
        elif frozen and self.scan_backend == 'chunked':
            states, s = torch_chunked_scan(gamma, kv, s)
        else:
            states, s = encoder_recurrent_scan(gamma, kv, s)
        context = self.out_norm(q * states)
        x = x + self.proj_out(context) * self.gate
        return x + self.ffn(self.norm2(x)), s


class DeltaWriter(nn.Module):
    def __init__(self, cfg: CEDConfig):
        super().__init__()
        self.cfg = cfg
        self.norm = nn.LayerNorm(cfg.d_model)
        self.key = nn.Linear(cfg.d_model, cfg.heads * cfg.key_dim, bias=False)
        self.value = nn.Linear(cfg.d_model, cfg.heads * cfg.value_dim, bias=False)
        self.alpha = nn.Linear(cfg.d_model, cfg.heads)
        self.beta = nn.Linear(cfg.d_model, cfg.heads)
        nn.init.zeros_(self.alpha.weight)
        nn.init.constant_(self.alpha.bias, math.log(cfg.alpha_init / (1-cfg.alpha_init)))
        nn.init.zeros_(self.beta.weight)
        nn.init.constant_(self.beta.bias, math.log(cfg.beta_init / (1-cfg.beta_init)))

    def parameters_for(self, e: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        h = self.norm(e)
        shape = (*h.shape[:2], self.cfg.heads, -1)
        k = F.normalize(self.key(h).reshape(shape).to(accumulator_dtype(h)), dim=-1, eps=1e-8)
        v = self.value(h).reshape(shape).to(accumulator_dtype(h))
        g = F.logsigmoid(self.alpha(h).to(accumulator_dtype(h)))
        beta = self.beta(h).to(accumulator_dtype(h)).sigmoid()
        return k, v, g, beta


class ExecutionBlock(nn.Module):
    """No private time cache and no memory writes. Query depends on work state."""
    def __init__(self, cfg: CEDConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.norm1 = nn.LayerNorm(d)
        self.proj_q = nn.Linear(d, d, bias=False)
        self.out_norm = nn.RMSNorm(d)
        self.proj_out = nn.Linear(d, d, bias=False)
        self.norm2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, cfg.ffn_dim), nn.GELU(), nn.Linear(cfg.ffn_dim, d))
        self.gate = nn.Parameter(torch.full((1, 1, d), 0.1))
        nn.init.zeros_(self.proj_out.weight)  # ONLY this projection is zeroed

    def forward(self, x: Tensor, read: Callable[[Tensor], Tensor]) -> Tensor:
        q = self.proj_q(self.norm1(x)).reshape(*x.shape[:2], self.cfg.heads, self.cfg.key_dim)
        q = F.normalize(q.to(accumulator_dtype(x)), dim=-1, eps=1e-8)
        r = read(q).reshape(*x.shape[:2], self.cfg.d_model).to(x.dtype)
        x = x + self.proj_out(self.out_norm(r)) * self.gate
        return x + self.ffn(self.norm2(x))


@dataclass(frozen=True)
class CEDState:
    encoder_states: tuple[Tensor, ...]
    memory: Tensor
    positions: Tensor       # next per-row input position, NOT latent iteration count
    last_encoded: Tensor

    def detach(self) -> 'CEDState':
        return CEDState(tuple(t.detach() for t in self.encoder_states), self.memory.detach(),
                        self.positions.detach(), self.last_encoded.detach())

    def reset_rows(self, reset: Tensor) -> 'CEDState':
        if reset.dtype != torch.bool or reset.shape != self.positions.shape:
            raise ValueError('reset must be a boolean [batch] tensor')
        keep = ~reset
        return CEDState(tuple(s * keep[:, None] for s in self.encoder_states),
            self.memory * keep[:, None, None, None], self.positions * keep,
            self.last_encoded * keep[:, None])

    def to(self, device) -> 'CEDState':
        return CEDState(tuple(t.to(device) for t in self.encoder_states), self.memory.to(device),
                        self.positions.to(device), self.last_encoded.to(device))

    def tensor_dict(self) -> dict:
        return {'encoder_states': list(self.encoder_states), 'memory': self.memory,
                'positions': self.positions, 'last_encoded': self.last_encoded}

    @staticmethod
    def from_tensor_dict(data: dict) -> 'CEDState':
        return CEDState(tuple(data['encoder_states']), data['memory'], data['positions'], data['last_encoded'])


@dataclass(frozen=True)
class CEDOutput:
    logits: Tensor
    state: CEDState
    work_state: Tensor


class GeoCPU_CED_v1(nn.Module):
    def __init__(self, cfg: CEDConfig):
        super().__init__()
        self.config = cfg
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        if cfg.position_mode == 'frozen':
            self.register_buffer('position_table', torch.zeros(cfg.max_positions, cfg.d_model))
        elif cfg.position_mode == 'learned':
            self.position_table = nn.Parameter(torch.zeros(cfg.max_positions, cfg.d_model))
        else:
            self.register_buffer('position_table', None)
        self.encoder = nn.ModuleList([EncoderBlock(cfg) for _ in range(cfg.encoder_layers)])
        self.writer = DeltaWriter(cfg)
        self.decoder = nn.ModuleList([ExecutionBlock(cfg) for _ in range(cfg.decoder_layers)])
        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=cfg.head_bias)

    def initial_state(self, batch_size: int) -> CEDState:
        if batch_size <= 0:
            raise ValueError('batch_size must be positive')
        w, c = self.token_embed.weight, self.config
        return CEDState(tuple(w.new_zeros(batch_size, c.d_model) for _ in self.encoder),
            torch.zeros(batch_size, c.heads, c.key_dim, c.value_dim, device=w.device, dtype=accumulator_dtype(w)),
            torch.zeros(batch_size, device=w.device, dtype=torch.long), w.new_zeros(batch_size, c.d_model))

    def _encode(self, tokens: Tensor, state: CEDState, fresh: bool = False) -> tuple[Tensor, tuple[Tensor, ...], Tensor]:
        if tokens.ndim != 2 or tokens.dtype != torch.long or tokens.shape[1] == 0:
            raise ValueError('tokens must be non-empty int64 [B,T]')
        if tokens.shape[0] != state.memory.shape[0] or len(state.encoder_states) != len(self.encoder):
            raise ValueError('State batch/layer count mismatch')
        positions = state.positions[:, None] + torch.arange(tokens.shape[1], device=tokens.device)
        if (tokens.shape[1] > self.config.max_positions) or (not fresh and bool((positions >= self.config.max_positions).any())):
            raise ValueError('Position limit exceeded: no silent wrap or length-extrapolation claim; reset only at a real document boundary')
        e = self.token_embed(tokens)
        if self.position_table is not None:
            e = e + self.position_table[positions]
        enc_states = []
        for block, s in zip(self.encoder, state.encoder_states):
            e, end = block(e, s)
            enc_states.append(end)
        return e, tuple(enc_states), state.positions + tokens.shape[1]

    def _execute(self, e: Tensor, read: Callable[[Tensor], Tensor], iterations: int) -> Tensor:
        if not isinstance(iterations, int) or iterations < 1:
            raise ValueError('iterations must be a positive integer')
        z = e
        for _ in range(iterations):
            for block in self.decoder:
                z = block(z, read)
        return z

    def forward(self, tokens: Tensor, state: Optional[CEDState] = None,
                iterations: int = 1, backend: str = 'chunk') -> CEDOutput:
        fresh = state is None
        state = self.initial_state(tokens.shape[0]) if fresh else state
        e, enc, positions = self._encode(tokens, state, fresh=fresh)
        k, v, g, beta = self.writer.parameters_for(e)
        if backend == 'chunk':
            tape = build_delta_tape(k, v, g, beta, state.memory, self.config.chunk_size, self.config.read_scale)
        elif backend == 'sequential':
            tape = build_sequential_tape(k, v, g, beta, state.memory, self.config.read_scale)
        else:
            raise ValueError('backend must be chunk or sequential')
        z = self._execute(e, tape.read, iterations)
        end = CEDState(enc, tape.final_memory, positions, e[:, -1])
        return CEDOutput(self.head(self.final_norm(z)), end, z)

    def prefill(self, tokens: Tensor, state: Optional[CEDState] = None) -> CEDState:
        """Encode and write ONCE. Does not call the decoder or produce prompt logits."""
        fresh = state is None
        state = self.initial_state(tokens.shape[0]) if fresh else state
        e, enc, positions = self._encode(tokens, state, fresh=fresh)
        k, v, g, beta = self.writer.parameters_for(e)
        tape = build_delta_tape(k, v, g, beta, state.memory, self.config.chunk_size, self.config.read_scale)
        return CEDState(enc, tape.final_memory, positions, e[:, -1])

    def think(self, state: CEDState, iterations: int = 1, work_state: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        """Pure read/compute on the LAST encoded token. Neither t nor M changes.
        To continue a latent computation pass work_state from the previous call.
        To change an external question, encode its actual tokens first.
        """
        if bool((state.positions == 0).any()):
            raise ValueError('At least one input token per row is needed before think()')
        e = state.last_encoded[:, None] if work_state is None else work_state
        def read(q: Tensor) -> Tensor:
            with torch.autocast(device_type=q.device.type, enabled=False):
                return torch.einsum('bthk,bhkv->bthv', q.to(state.memory.dtype), state.memory) * self.config.read_scale
        z = self._execute(e, read, iterations)
        return self.head(self.final_norm(z))[:, -1], z

    def step(self, token: Tensor, state: CEDState, iterations: int = 1) -> CEDOutput:
        if token.ndim == 1:
            token = token[:, None]
        if token.shape[1] != 1:
            raise ValueError('step() accepts one token per row')
        # One external event advances t once, independent of iterations.
        end = self.prefill(token, state)
        logits, z = self.think(end, iterations)
        return CEDOutput(logits[:, None], end, z)
