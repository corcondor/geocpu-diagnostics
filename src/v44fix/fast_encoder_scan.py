"""Optional A100 Triton *forward-only* recurrent scan.

Use only when encoder and token embeddings are frozen. For full fine-tuning,
fall back to the autograd-safe TorchScript recurrence in GeoCPU_CED_v1.
Numerical behavior is FP32 accumulation; BF16 inputs are accepted. GPU test
must be run on device before selecting this kernel in production.
"""
from __future__ import annotations
import torch
try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False

if HAVE_TRITON:
    @triton.jit
    def _scan(G, KV, S0, Y, SF, T: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
        batch = tl.program_id(0)
        d = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = d < D
        off0 = batch * D + d
        s = tl.load(S0 + off0, mask=mask, other=0).to(tl.float32)
        for t in range(T):
            off = batch * T * D + t * D + d
            g = tl.load(G + off, mask=mask, other=1).to(tl.float32)
            v = tl.load(KV + off, mask=mask, other=0).to(tl.float32)
            s = g * s + v
            tl.store(Y + off, s, mask=mask)
        tl.store(SF + off0, s, mask=mask)

    @triton.jit
    def _scan_fwd_dyn(G, KV, S0, Y, SF, T, D: tl.constexpr, BLOCK: tl.constexpr):
        batch = tl.program_id(0)
        d = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = d < D
        off0 = batch * D + d
        s = tl.load(S0 + off0, mask=mask, other=0.0).to(tl.float32)
        base_off = batch * T * D + d
        for t in range(T):
            off = base_off + t * D
            g = tl.load(G + off, mask=mask, other=1.0).to(tl.float32)
            v = tl.load(KV + off, mask=mask, other=0.0).to(tl.float32)
            s = g * s + v
            tl.store(Y + off, s, mask=mask)
        tl.store(SF + off0, s, mask=mask)

    @triton.jit
    def _scan_bwd_dyn(G, Y, S0, DY, DSF, DG, DKV, DS0, T, D: tl.constexpr, BLOCK: tl.constexpr):
        batch = tl.program_id(0)
        d = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = d < D
        off0 = batch * D + d
        ds = tl.load(DSF + off0, mask=mask, other=0.0).to(tl.float32)
        base_off = batch * T * D + d
        for step in range(T):
            t = (T - 1) - step
            off = base_off + t * D
            dy = tl.load(DY + off, mask=mask, other=0.0).to(tl.float32)
            g = tl.load(G + off, mask=mask, other=1.0).to(tl.float32)
            ds = ds + dy
            s_prev = tl.load(Y + off - D, mask=mask & (t > 0), other=0.0).to(tl.float32)
            s0_val = tl.load(S0 + off0, mask=mask & (t == 0), other=0.0).to(tl.float32)
            s_prev = tl.where(t > 0, s_prev, s0_val)
            tl.store(DG + off, ds * s_prev, mask=mask)
            tl.store(DKV + off, ds, mask=mask)
            ds = g * ds
        tl.store(DS0 + off0, ds, mask=mask)


class _TritonRecurrentScanFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gamma: torch.Tensor, kv: torch.Tensor, initial_state: torch.Tensor):
        g = gamma.contiguous()
        v = kv.contiguous()
        s0 = initial_state.contiguous()
        b, t, d = v.shape
        out = torch.empty((b, t, d), dtype=torch.float32, device=v.device)
        sf = torch.empty((b, d), dtype=torch.float32, device=v.device)
        BLOCK = 64
        _scan_fwd_dyn[(b, triton.cdiv(d, BLOCK))](g, v, s0, out, sf, t, d, BLOCK, num_warps=4)
        ctx.save_for_backward(g, out, s0)
        ctx.orig_dtypes = (gamma.dtype, kv.dtype, initial_state.dtype)
        return out, sf

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor, grad_sf: torch.Tensor):
        g, out, s0 = ctx.saved_tensors
        g_dtype, v_dtype, s0_dtype = ctx.orig_dtypes
        b, t, d = out.shape
        dy = grad_out.contiguous() if grad_out is not None else torch.zeros_like(out)
        dsf = grad_sf.contiguous() if grad_sf is not None else torch.zeros_like(s0)
        dg = torch.empty((b, t, d), dtype=torch.float32, device=out.device)
        dkv = torch.empty((b, t, d), dtype=torch.float32, device=out.device)
        ds0 = torch.empty((b, d), dtype=torch.float32, device=out.device)
        BLOCK = 64
        _scan_bwd_dyn[(b, triton.cdiv(d, BLOCK))](g, out, s0, dy, dsf, dg, dkv, ds0, t, d, BLOCK, num_warps=4)
        return dg.to(g_dtype), dkv.to(v_dtype), ds0.to(s0_dtype)


def triton_recurrent_scan(gamma, kv, initial_state):
    if not HAVE_TRITON:
        raise RuntimeError('Triton is unavailable')
    if not kv.is_cuda:
        raise RuntimeError('Triton scan needs CUDA')
    if any(x.requires_grad for x in (gamma, kv, initial_state)):
        raise RuntimeError('Forward-only scan cannot be used with encoder gradients')
    if kv.ndim != 3 or kv.shape != gamma.shape or initial_state.shape != (kv.shape[0], kv.shape[-1]):
        raise ValueError('scan shape mismatch')
    g = gamma.contiguous()
    v = kv.contiguous()
    s0 = initial_state.contiguous()
    b, t, d = v.shape
    out = torch.empty((b, t, d), dtype=torch.float32, device=v.device)
    sf = torch.empty((b, d), dtype=torch.float32, device=v.device)
    _scan[(b, triton.cdiv(d, 32))](g, v, s0, out, sf, t, d, 32, num_warps=4)
    return out, sf


def torch_chunked_scan(gamma, kv, initial_state, chunk_size=32):
    """Vectorized autograd-compatible FP32 chunked associative recurrence.

    Computes intra-chunk cumprod/cumsum across all chunks in parallel [B, nc, C, D],
    then propagates only the [B, D] chunk-boundary states across nc chunks.
    """
    if chunk_size > 32 or chunk_size < 1:
        raise ValueError('chunk_size must be between 1 and 32')
    B, T, D = kv.shape
    if T == 1:
        g = gamma.float() if gamma.dtype != torch.float64 else gamma
        v = kv.float() if kv.dtype != torch.float64 else kv
        s = initial_state.to(v.dtype)
        s_next = g[:, 0] * s + v[:, 0]
        return s_next.unsqueeze(1), s_next
    if HAVE_TRITON and kv.is_cuda and gamma.dtype != torch.float64:
        try:
            return _TritonRecurrentScanFn.apply(gamma, kv, initial_state)
        except Exception:
            pass
    g = gamma.float() if gamma.dtype != torch.float64 else gamma
    v = kv.float() if kv.dtype != torch.float64 else kv
    s = initial_state.to(v.dtype)
    rem = T % chunk_size
    if rem == 0:
        nc = T // chunk_size
        gc = g.view(B, nc, chunk_size, D)
        vc = v.view(B, nc, chunk_size, D)
        gp = gc.cumprod(dim=2)
        v_sum = (vc / gp).cumsum(dim=2)
        gp_end = gp[:, :, -1, :]
        local_end = gp_end * v_sum[:, :, -1, :]
        s_list = []
        for c in range(nc):
            s_list.append(s)
            s = gp_end[:, c] * s + local_end[:, c]
        s_starts = torch.stack(s_list, dim=1)
        out = gp * (s_starts.unsqueeze(2) + v_sum)
        return out.reshape(B, T, D), s
    chunks = []
    for i in range(0, T, chunk_size):
        gp = g[:, i:i+chunk_size].cumprod(dim=1)
        local = gp * (s[:, None, :] + (v[:, i:i+chunk_size] / gp).cumsum(dim=1))
        chunks.append(local)
        s = local[:, -1]
    return torch.cat(chunks, dim=1), s


def check_triton_numerics(device='cuda', length=64):
    if not HAVE_TRITON or not torch.cuda.is_available():
        return {'available': False}
    torch.manual_seed(193)
    results = {}
    for dtype in (torch.float32, torch.bfloat16):
        gam = (0.75 + torch.rand(3, length, 256, device=device) * 0.249).to(dtype)
        kv = (torch.randn(3, length, 256, device=device) * .1).to(dtype)
        s0 = torch.randn(3, 256, device=device).to(dtype)
        fast_y, fast_s = triton_recurrent_scan(gam, kv, s0)
        # Reference uses float32 accumulation and same quantized inputs.
        y = []
        s = s0.float()
        for t in range(length):
            s = s * gam[:, t].float() + kv[:, t].float()
            y.append(s)
        err = (fast_y - torch.stack(y, 1)).abs().max().item()
        ferr = (fast_s-s).abs().max().item()
        results[str(dtype)] = {'max_abs_output_error':err,'max_abs_final_error':ferr}
        if max(err,ferr) > 2e-4:
            raise AssertionError(f'Triton scan numerics failed for {dtype}: {results}')
    return {'available': True, 'tested_length': length,'results':results}
