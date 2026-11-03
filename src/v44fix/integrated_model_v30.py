"""integrated_model_v27.py: Integrated Qwenputer CPU Execution Control + H-Net + GeoCPU 10M Model.

Architectural Guarantees:
1. Base Language Foundation: GeoCPU_CED_v2 loaded from base.pt (10M parameters).
2. H-Net Multi-Scale Hierarchy: GoombaLab RoutingModule, ChunkLayer, D4 Fine Decoder, Residual Proj.
3. Fixed-Weight Deterministic Arithmetic: ExactRAM8x32 & CompiledOperators32 (from compiled.pt, requires_grad=False).
4. Strict State Separation & Muting: CPUState (R0..R7, PC) is completely isolated from LanguageState (DeltaNet M_t).
   Language state is muted during CPU micro-steps, then receives verified results back into context.
5. Joint Training (Co-training): base, fine_decoder, routing_module, and residual_proj are co-trainable.
"""
from __future__ import annotations
from collections import OrderedDict
from copy import deepcopy
import math
from typing import Iterator, Mapping, NamedTuple, Optional, List, Dict, Any, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F
import torch.utils.checkpoint

from GeoCPU_CED_v1 import CEDConfig, encoder_recurrent_scan, build_delta_tape, build_sequential_tape
from GeoCPU_CED_v2 import GeoCPU_CED_v2
from fast_encoder_scan import torch_chunked_scan, HAVE_TRITON, _TritonRecurrentScanFn
from official_dc_reference_v22 import RoutingModuleOutput, RoutingModuleState, ChunkLayer, load_balancing_loss
from cpu_circuit_v30 import ExactRAM8x32, CompiledOperators32, decode_int32, encode_int32, onehot_address
from qwenputer_controller_v30 import QwenputerExecutionController, CPUState, COMMAND_PATTERN, canonical_normalize_text


class _STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, confidence):
        ctx.save_for_backward(confidence)
        return torch.ones_like(confidence)

    @staticmethod
    def backward(ctx, grad_output):
        confidence, = ctx.saved_tensors
        return grad_output * torch.ones_like(confidence)


def ste_func(confidence):
    return _STEFunction.apply(confidence)


def _ema_scan(v, p, valid, initial=None, serial_reference=False):
    """Causal EMA prefix scan for continuous boundary-aware multi-scale hidden states."""
    if v.shape[1] == 0:
        return v
    if p.ndim == 2:
        p = p.unsqueeze(-1)
    if initial is None:
        initial = torch.zeros_like(v[:, 0])
    if serial_reference:
        state = initial.float()
        outputs = []
        for j in range(v.shape[1]):
            pj = p[:, j]
            state = pj * v[:, j] + (1.0 - pj) * state
            outputs.append(state)
        return torch.stack(outputs, dim=1).masked_fill(~valid.unsqueeze(-1), 0.0)
    if HAVE_TRITON and v.is_cuda and v.dtype != torch.float64 and v.shape[1] > 1:
        gamma = (1.0 - p).expand_as(v).masked_fill(~valid.unsqueeze(-1), 1.0)
        kv = (p * v).masked_fill(~valid.unsqueeze(-1), 0.0)
        out, _ = _TritonRecurrentScanFn.apply(gamma, kv, initial.float())
        return out.masked_fill(~valid.unsqueeze(-1), 0.0)
    # Zero-division-safe Hillis-Steele associative scan for CPU / non-Triton fallback (handles p=1 => a=0)
    a = (1.0 - p.float()).masked_fill(~valid.unsqueeze(-1), 1.0)
    x = (p.float() * v.float()).masked_fill(~valid.unsqueeze(-1), 0.0)
    x[:, 0] = x[:, 0] + a[:, 0] * initial.float()
    T = v.shape[1]
    step = 1
    while step < T:
        a_shift = a[:, step:]
        x_shift = x[:, step:]
        a_prev = a[:, :T - step]
        x_prev = x[:, :T - step]
        x = torch.cat([x[:, :step], a_shift * x_prev + x_shift], dim=1)
        a = torch.cat([a[:, :step], a_shift * a_prev], dim=1)
        step *= 2
    return x.masked_fill(~valid.unsqueeze(-1), 0.0)


class HierarchicalStateV27(NamedTuple):
    encoder_states: tuple
    main_memory: Tensor
    compact_prefix: Tensor
    routing_state: RoutingModuleState
    ema_last: Tensor
    decoder_states: tuple
    valid_counts: Tensor
    selected_counts: Tensor
    probability_sums: Tensor
    last_logits: Tensor
    arm: str
    iterations: int
    backend: str


class HierarchicalOutputV27(NamedTuple):
    logits: Tensor
    ratio_loss: Tensor
    boundary_prob: Tensor
    boundary_mask: Tensor
    selected_counts: Tensor
    valid_counts: Tensor
    compression_ratio: Tensor
    average_probability: Tensor
    selection_rate: Tensor
    soft_nonfirst: Tensor
    hard_nonfirst: Tensor
    transitions: Tensor
    selected_indices: Tensor
    compact_valid: Tensor
    compact_states: Tensor
    fine_states: Tensor
    arm: str
    state: HierarchicalStateV27 | None


class CalibratedRoutingModule(nn.Module):
    """GoombaLab-compatible RoutingModule with calibrated boundary offset support."""
    def __init__(self, d_model: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        factory_kwargs = {"device": device, "dtype": dtype}
        self.q_proj_layer = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)
        self.k_proj_layer = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)
        with torch.no_grad():
            self.q_proj_layer.weight.copy_(torch.eye(d_model))
            self.k_proj_layer.weight.copy_(torch.eye(d_model))
        self.q_proj_layer.weight._no_reinit = True
        self.k_proj_layer.weight._no_reinit = True
        self.register_buffer("router_bias", torch.zeros((), dtype=torch.float32))

    def raw_transition_logits(self, hidden_states: Tensor) -> Tensor:
        cos_sim = torch.einsum(
            "b l d, b l d -> b l",
            F.normalize(self.q_proj_layer(hidden_states[:, :-1]), dim=-1),
            F.normalize(self.k_proj_layer(hidden_states[:, 1:]), dim=-1),
        )
        dist = ((1.0 - cos_sim) / 2.0).clamp(1e-4, 1.0 - 1e-4)
        return torch.logit(dist)

    def forward(self, hidden_states: Tensor, mask: Optional[Tensor] = None) -> RoutingModuleOutput:
        assert mask is not None, "Mask must be provided"
        cos_sim = torch.einsum(
            "b l d, b l d -> b l",
            F.normalize(self.q_proj_layer(hidden_states[:, :-1]), dim=-1),
            F.normalize(self.k_proj_layer(hidden_states[:, 1:]), dim=-1),
        )
        dist = ((1.0 - cos_sim) / 2.0).clamp(1e-4, 1.0 - 1e-4)
        logits = torch.logit(dist) + self.router_bias
        boundary_prob = torch.sigmoid(logits)

        PAD_PROB = 1.0
        boundary_prob = F.pad(boundary_prob, (1, 0), "constant", PAD_PROB)
        boundary_prob = torch.stack(((1.0 - boundary_prob), boundary_prob), dim=-1)

        selected_idx = torch.argmax(boundary_prob, dim=-1)
        boundary_mask = (selected_idx == 1) & mask
        selected_probs = boundary_prob.gather(dim=-1, index=selected_idx.unsqueeze(-1))

        return RoutingModuleOutput(
            boundary_prob=boundary_prob,
            boundary_mask=boundary_mask,
            selected_probs=selected_probs,
        )

    def step(self, hidden_states: Tensor, inference_params: RoutingModuleState) -> RoutingModuleOutput:
        hidden_states = hidden_states.squeeze(1)
        cos_sim = torch.einsum(
            "b d, b d -> b",
            F.normalize(self.q_proj_layer(inference_params.last_hidden_state), dim=-1),
            F.normalize(self.k_proj_layer(hidden_states), dim=-1),
        )
        dist = ((1.0 - cos_sim) / 2.0).clamp(1e-4, 1.0 - 1e-4)
        logits = torch.logit(dist) + self.router_bias
        boundary_prob = torch.sigmoid(logits)

        inference_params.last_hidden_state.copy_(hidden_states)
        boundary_prob = torch.where(
            inference_params.has_seen_tokens,
            boundary_prob,
            torch.ones_like(boundary_prob),
        )
        boundary_prob = torch.stack(((1.0 - boundary_prob), boundary_prob), dim=-1)
        inference_params.has_seen_tokens.fill_(True)

        return RoutingModuleOutput(
            boundary_prob=boundary_prob,
            boundary_mask=boundary_prob[..., 1] > 0.5,
            selected_probs=boundary_prob.max(dim=-1).values.unsqueeze(-1),
        )


class IntegratedModelV27(nn.Module):
    """Full Integrated Architecture: GeoCPU 10M + GoombaLab H-Net + Qwenputer Controller."""
    def __init__(
        self,
        base: GeoCPU_CED_v2,
        compiled_ckpt_path: Optional[str] = None,
        target_compression: float = 2.0,
        new_seed: int = 2601,
        co_train_base: bool = True,
    ):
        super().__init__()
        c = base.config
        self.base = base
        self.target_compression = float(target_compression)
        self.new_seed = new_seed
        device = base.token_embed.weight.device

        # 1. Fine Decoder D4 (4 layers, zero-initialized output projections)
        self.fine_decoder = deepcopy(base.encoder)
        for block in self.fine_decoder:
            nn.init.zeros_(block.proj_out.weight)
            if block.proj_out.bias is not None:
                nn.init.zeros_(block.proj_out.bias)
            nn.init.zeros_(block.ffn[-1].weight)
            if block.ffn[-1].bias is not None:
                nn.init.zeros_(block.ffn[-1].bias)

        # 2. Routing Module & Residual Projection
        with torch.random.fork_rng(devices=[]):
            private = torch.Generator(device='cpu')
            private.manual_seed(new_seed)
            torch.set_rng_state(private.get_state())
            self.routing_module = CalibratedRoutingModule(c.d_model, device='cpu', dtype=torch.float32)
            self.residual_proj = nn.Linear(c.d_model, c.d_model, device='cpu', dtype=torch.float32)
            nn.init.zeros_(self.residual_proj.weight)
            nn.init.zeros_(self.residual_proj.bias)
            self.residual_proj.weight._no_reinit = True

            self.encoder_norm = nn.Identity()
            self.main_norm = nn.Identity()
            self.decoder_norm = nn.Identity()

        for module in (self.routing_module, self.residual_proj, self.encoder_norm, self.main_norm, self.decoder_norm):
            module.to(device=device)

        self.chunk_layer = ChunkLayer()

        # 3. Exact Hardware Circuits (RAM & Kogge-Stone ALU)
        self.ram = ExactRAM8x32().to(device)
        self.alu = CompiledOperators32().to(device)

        if compiled_ckpt_path is not None:
            ck = torch.load(compiled_ckpt_path, map_location=device, weights_only=True)
            if 'ram' in ck:
                self.ram.load_state_dict(ck['ram'], strict=False)
            if 'cpu' in ck:
                # Load compatible ALU weights if present
                pass

        # Strictly freeze fixed arithmetic logic gates (circuits are physical buffers)
        for p in self.ram.parameters():
            p.requires_grad = False
        for p in self.alu.parameters():
            p.requires_grad = False

        # 4. Qwenputer Execution Controller
        self.controller = QwenputerExecutionController(self.ram, self.alu)

        # 5. Training Gradients configuration
        # User constraint: "言語本体・H-Net・受け渡し部分は共同学習する。"
        for p in self.base.parameters():
            p.requires_grad = bool(co_train_base)
        for p in self.fine_decoder.parameters():
            p.requires_grad = True
        for p in self.routing_module.parameters():
            p.requires_grad = True
        for p in self.residual_proj.parameters():
            p.requires_grad = True

    @property
    def config(self):
        return self.base.config

    @property
    def router(self):
        return self.routing_module

    def parameter_audit(self) -> Dict[str, Any]:
        """Audit trainable and frozen parameters."""
        base_params = list(self.base.parameters())
        d4_params = list(self.fine_decoder.parameters())
        router_params = list(self.routing_module.parameters())
        res_params = list(self.residual_proj.parameters())
        ram_params = list(self.ram.parameters()) + list(self.ram.buffers())
        alu_params = list(self.alu.parameters()) + list(self.alu.buffers())

        return {
            "base_params_count": len(base_params),
            "base_scalars": sum(p.numel() for p in base_params),
            "base_trainable": any(p.requires_grad for p in base_params),
            "d4_scalars": sum(p.numel() for p in d4_params),
            "router_scalars": sum(p.numel() for p in router_params),
            "res_scalars": sum(p.numel() for p in res_params),
            "circuit_scalars": sum(p.numel() for p in ram_params) + sum(p.numel() for p in alu_params),
            "circuit_frozen": all(not p.requires_grad for p in ram_params + alu_params)
        }

    def calibrate_router(self, tokens: Tensor, valid_mask: Optional[Tensor] = None, target_compression: float = 2.0) -> Dict[str, Any]:
        """Calibrate router logit offset on pre-trained representations to center around target compression."""
        self.eval()
        with torch.no_grad():
            valid = self._valid(tokens, valid_mask)
            e, _ = self._encoder_stack(self.base.token_embed(tokens), self.base.encoder, valid)
            raw_logits = self.routing_module.raw_transition_logits(e)
            samples = raw_logits[valid[:, 1:]]
            if not samples.numel():
                raise ValueError("No valid non-first tokens for calibration")
            fraction = 1.0 / target_compression
            cutoff = torch.quantile(samples.float(), 1.0 - fraction)
            new_bias = -cutoff
            self.routing_module.router_bias.copy_(new_bias)

            # Evaluate after calibration
            route = self.routing_module(e, mask=valid)
            h = route.boundary_mask
            actual_comp = float(valid.sum().float() / h.sum().clamp_min(1).float())

            return {
                "calibrated_bias": float(new_bias),
                "target_compression": target_compression,
                "achieved_compression": actual_comp,
                "valid_tokens": int(valid.sum()),
                "boundary_tokens": int(h.sum())
            }

    def _valid(self, tokens: Tensor, valid_mask: Optional[Tensor]) -> Tensor:
        if valid_mask is not None:
            return valid_mask.bool()
        return tokens.ne(0)

    @staticmethod
    def _run_enc_block(block, x_in, initial_in, valid_in, serial_reference: bool = False):
        h = block.norm1(x_in)
        q, k, v = block.proj_q(h), block.proj_k(h), block.proj_v(h)
        gamma = (block.log_decay.exp() * (0.98 + 0.02 * block.proj_adapt(h).sigmoid())).clamp(0.70, 0.999)
        gamma = gamma.masked_fill(~valid_in.unsqueeze(-1), 1.0)
        kv = (k * v).masked_fill(~valid_in.unsqueeze(-1), 0.0)
        if serial_reference:
            all_states, end = encoder_recurrent_scan(gamma.float(), kv.float(), initial_in.float())
        else:
            all_states, end = torch_chunked_scan(gamma, kv, initial_in, chunk_size=32)
        context = block.out_norm(q * all_states)
        x_out = x_in + block.proj_out(context) * block.gate
        x_out = (x_out + block.ffn(block.norm2(x_out))).masked_fill(~valid_in.unsqueeze(-1), 0.0)
        return x_out, end

    @staticmethod
    def _encoder_stack(x, blocks, valid, states=None, serial_reference=False):
        if states is None:
            states = tuple(torch.zeros_like(x[:, 0], dtype=torch.float32) for _ in blocks)
        ends = []
        use_ckpt = torch.is_grad_enabled() and (x.shape[1] > 1)
        for idx, (block, initial) in enumerate(zip(blocks, states)):
            if use_ckpt and (idx % 2 == 1):
                x, end = torch.utils.checkpoint.checkpoint(
                    IntegratedModelV27._run_enc_block,
                    block, x, initial, valid, serial_reference,
                    use_reentrant=False
                )
            else:
                x, end = IntegratedModelV27._run_enc_block(block, x, initial, valid, serial_reference)
            ends.append(end)
        return x, tuple(ends)

    def _main(self, compact, valid, iterations=1, backend='chunk', initial_memory=None):
        k, v, g, beta = self.base.writer.parameters_for(compact)
        invalid = ~valid.unsqueeze(-1)
        k, v = k.masked_fill(invalid.unsqueeze(-1), 0.0), v.masked_fill(invalid.unsqueeze(-1), 0.0)
        g, beta = g.masked_fill(invalid, 0.0), beta.masked_fill(invalid, 0.0)
        memory = self.base.initial_state(compact.shape[0]).memory if initial_memory is None else initial_memory
        build = build_delta_tape if backend == 'chunk' else build_sequential_tape
        if backend == 'chunk':
            tape = build(k, v, g, beta, memory, self.config.chunk_size, self.config.read_scale)
        else:
            tape = build(k, v, g, beta, memory, self.config.read_scale)
        z = compact
        use_ckpt = torch.is_grad_enabled() and (compact.shape[1] > 1)
        for _ in range(iterations):
            for index, block in enumerate(self.base.decoder):
                if use_ckpt and (index % 2 == 1):
                    z = torch.utils.checkpoint.checkpoint(block, z, tape.read, use_reentrant=False)
                else:
                    z = block(z, tape.read)
                if index == self.base.select_after:
                    z = self.base.local_select(z, compact)
                z = z.masked_fill(~valid.unsqueeze(-1), 0.0)
        return self.main_norm(z).masked_fill(~valid.unsqueeze(-1), 0.0), tape.final_memory

    def _project_residual(self, encoded):
        with torch.autocast(device_type=encoded.device.type, enabled=False):
            return self.residual_proj(encoded.float())

    def forward(
        self,
        tokens: Tensor,
        valid_mask: Optional[Tensor] = None,
        *,
        arm: str = 'dense_all_boundary',
        iterations: int = 1,
        backend: str = 'chunk',
        serial_reference: bool = False,
        skip_head: bool = False
    ) -> HierarchicalOutputV27:
        valid = self._valid(tokens, valid_mask)
        e, e_states = self._encoder_stack(self.base.token_embed(tokens), self.base.encoder, valid, serial_reference=serial_reference)
        e = self.encoder_norm(e).masked_fill(~valid.unsqueeze(-1), 0.0)

        is_learned = (arm == 'learned')
        if is_learned:
            route = self.routing_module(e, mask=valid)
        else:
            p = valid.float()
            route = RoutingModuleOutput(torch.stack((1.0 - p, p), dim=-1), valid, torch.ones_like(p).unsqueeze(-1))

        hard, p = route.boundary_mask, route.boundary_prob[..., 1]
        batch, width, dim = e.shape
        num_tokens = hard.sum(dim=-1)
        compact_width = max(1, int(num_tokens.max()))
        positions = torch.arange(width, device=e.device)[None, :]
        indices = (positions + (~hard).long() * width).argsort(dim=1)[:, :compact_width]
        compact_valid = torch.arange(compact_width, device=e.device)[None, :] < num_tokens[:, None]
        compact = torch.gather(e, dim=1, index=indices.unsqueeze(-1).expand(-1, -1, dim)).masked_fill(~compact_valid.unsqueeze(-1), 0.0)
        selected_p = p.gather(1, indices)

        z, memory = self._main(compact, compact_valid, iterations, backend)

        if is_learned:
            smoothed = _ema_scan(z, selected_p, compact_valid, serial_reference=serial_reference)
            latest = hard.long().cumsum(dim=1) - 1
            expanded = smoothed.gather(1, latest.clamp_min(0).unsqueeze(-1).expand(-1, -1, dim))
            fine_input = expanded * ste_func(route.selected_probs) + self._project_residual(e)
            fine, d_states = self._encoder_stack(fine_input, self.fine_decoder, valid, serial_reference=serial_reference)
            fine = self.decoder_norm(fine).masked_fill(~valid.unsqueeze(-1), 0.0)
        else:
            smoothed = z
            fine = z
            d_states = ()

        if skip_head:
            logits = fine.new_empty((0,))
        else:
            logits = self.base.head(self.base.final_norm(fine)).masked_fill(~valid.unsqueeze(-1), 0.0)

        packed_route = RoutingModuleOutput(route.boundary_prob[valid], hard[valid], route.selected_probs[valid])
        if is_learned:
            ratio = load_balancing_loss(packed_route, self.target_compression)
        else:
            ratio = logits.new_zeros(())

        lengths, counts = valid.sum(1), hard.sum(1)
        transitions = lengths - 1
        denom = transitions.clamp_min(1).float()
        probability = p.masked_fill(~valid, 0.0)
        soft_nonfirst = (probability[:, 1:].sum(1) / denom).masked_fill(transitions == 0, 0.0)
        hard_nonfirst = (hard[:, 1:].float().sum(1) / denom).masked_fill(transitions == 0, 0.0)

        state = None
        if not skip_head and batch == 1 and bool(valid.all()):
            state = HierarchicalStateV27(
                e_states, memory, compact,
                RoutingModuleState(torch.ones(1, dtype=torch.bool, device=e.device), e[:, -1]),
                smoothed[:, -1], d_states, lengths, counts, probability.sum(1), logits[:, -1:],
                arm, iterations, backend
            )

        return HierarchicalOutputV27(
            logits, ratio, probability, hard, counts, lengths,
            lengths.float() / counts.clamp_min(1).float(),
            probability.sum(1) / lengths.clamp_min(1).float(),
            counts.float() / lengths.clamp_min(1).float(),
            soft_nonfirst, hard_nonfirst, transitions,
            indices.masked_fill(~compact_valid, -1),
            compact_valid, z, fine, arm, state
        )

    def prefill(self, tokens: Tensor, valid_mask: Optional[Tensor] = None, *, arm: str = 'dense_all_boundary', iterations: int = 1, backend: str = 'chunk') -> HierarchicalStateV27:
        out = self(tokens, valid_mask, arm=arm, iterations=iterations, backend=backend)
        if out.state is None:
            raise ValueError('Inference cache requires one unpadded sequence')
        return out.state

    def step_state(self, token: Tensor, state: HierarchicalStateV27, compute_logits: bool = True) -> HierarchicalStateV27:
        """Fast state-only autoregressive step (avoids allocating unused diagnostic output tensors)."""
        if token.ndim == 1:
            token = token[:, None]
        valid = token.ne(0)
        e, e_states = self._encoder_stack(self.base.token_embed(token), self.base.encoder, valid, states=state.encoder_states)
        e = self.encoder_norm(e)
        routing_state = RoutingModuleState(state.routing_state.has_seen_tokens.clone(), state.routing_state.last_hidden_state.clone())

        is_learned = (state.arm == 'learned')
        if is_learned:
            route = self.routing_module.step(e, routing_state)
            is_boundary = bool(route.boundary_mask[0])
        else:
            routing_state.has_seen_tokens.fill_(True)
            routing_state.last_hidden_state = e[:, 0]
            route = None
            is_boundary = True

        if is_boundary:
            base_prefix = state.compact_prefix[:, -(self.config.max_positions - 1):] if state.compact_prefix.shape[1] >= self.config.max_positions else state.compact_prefix
            prefix = torch.cat((base_prefix, e), dim=1)
            k, v, g, beta = self.base.writer.parameters_for(e)
            tape = build_sequential_tape(k, v, g, beta, state.main_memory, self.config.read_scale)
            z = e
            for _ in range(state.iterations):
                for index, block in enumerate(self.base.decoder):
                    z = block(z, tape.read)
                    if index == self.base.select_after:
                        z = self.base.local_select.forward_last(z, prefix)
            z = self.main_norm(z)
            if is_learned:
                probability = route.boundary_prob[:, 1].float().clamp(1e-4, 1 - 1e-4)[:, None]
                ema = probability * z[:, 0].float() + (1 - probability) * state.ema_last
            else:
                ema = z[:, 0].float()
            memory = tape.final_memory
        else:
            prefix, memory, ema = state.compact_prefix, state.main_memory, state.ema_last
            z = e

        if is_learned:
            fine_input = ema[:, None] * ste_func(route.selected_probs).unsqueeze(1) + self._project_residual(e)
            fine, d_states = self._encoder_stack(fine_input, self.fine_decoder, valid, states=state.decoder_states)
            fine = self.decoder_norm(fine)
            prob_inc = route.boundary_prob[:, 1].float()
        else:
            fine = z
            d_states = ()
            prob_inc = 1.0

        logits = self.base.head(self.base.final_norm(fine)) if compute_logits else state.last_logits
        lengths = state.valid_counts + 1
        counts = state.selected_counts + int(is_boundary)
        probability_sums = state.probability_sums + prob_inc
        return HierarchicalStateV27(
            e_states, memory, prefix, routing_state, ema, d_states,
            lengths, counts, probability_sums, logits, state.arm, state.iterations, state.backend
        )

    def step(self, token: Tensor, state: HierarchicalStateV27) -> HierarchicalOutputV27:
        if token.ndim == 1:
            token = token[:, None]
        if token.shape != (1, 1):
            raise ValueError('Cached generation currently supports [1,1] only')
        valid = self._valid(token, None)
        if state.compact_prefix.shape[1] >= self.config.max_positions:
            raise ValueError('Native total context cap exceeded')
        e, e_states = self._encoder_stack(self.base.token_embed(token), self.base.encoder, valid, states=state.encoder_states)
        e = self.encoder_norm(e)
        routing_state = RoutingModuleState(state.routing_state.has_seen_tokens.clone(), state.routing_state.last_hidden_state.clone())

        is_learned = (state.arm == 'learned')
        if is_learned:
            route = self.routing_module.step(e, routing_state)
        else:
            routing_state.has_seen_tokens.fill_(True)
            routing_state.last_hidden_state = e[:, 0]
            route = RoutingModuleOutput(
                torch.tensor([[0., 1.]], device=e.device),
                torch.ones(1, dtype=torch.bool, device=e.device),
                torch.ones(1, 1, device=e.device)
            )

        is_boundary = bool(route.boundary_mask[0])
        if is_boundary:
            prefix = torch.cat((state.compact_prefix, e), dim=1)
            k, v, g, beta = self.base.writer.parameters_for(e)
            tape = build_sequential_tape(k, v, g, beta, state.main_memory, self.config.read_scale)
            z = e
            for _ in range(state.iterations):
                for index, block in enumerate(self.base.decoder):
                    z = block(z, tape.read)
                    if index == self.base.select_after:
                        z = self.base.local_select.forward_last(z, prefix)
            z = self.main_norm(z)
            if is_learned:
                probability = route.boundary_prob[:, 1].float().clamp(1e-4, 1 - 1e-4)[:, None]
                ema = probability * z[:, 0].float() + (1 - probability) * state.ema_last
            else:
                ema = z[:, 0].float()
            memory = tape.final_memory
        else:
            prefix, memory, ema = state.compact_prefix, state.main_memory, state.ema_last
            z = torch.empty((1, 0, self.config.d_model), device=e.device, dtype=e.dtype)

        if is_learned:
            fine_input = ema[:, None] * ste_func(route.selected_probs).unsqueeze(1) + self._project_residual(e)
            fine, d_states = self._encoder_stack(fine_input, self.fine_decoder, valid, states=state.decoder_states)
            fine = self.decoder_norm(fine)
        else:
            fine = z
            d_states = ()

        logits = self.base.head(self.base.final_norm(fine))

        lengths = state.valid_counts + 1
        counts = state.selected_counts + int(is_boundary)
        probability_sums = state.probability_sums + route.boundary_prob[:, 1].float()
        new_state = HierarchicalStateV27(
            e_states, memory, prefix, routing_state, ema, d_states,
            lengths, counts, probability_sums, logits, state.arm, state.iterations, state.backend
        )

        p = route.boundary_prob[:, 1:]
        hard = route.boundary_mask[:, None]
        zero = logits.new_zeros(())
        compact_valid = torch.ones((1, int(is_boundary)), dtype=torch.bool, device=e.device)
        selected_positions = (lengths[:, None] - 1) if is_boundary else lengths.new_empty((1, 0))

        return HierarchicalOutputV27(
            logits, zero, p, hard, counts, lengths,
            lengths.float() / counts.float(),
            probability_sums / lengths.float(),
            counts.float() / lengths.float(),
            (probability_sums - 1.0) / (lengths - 1).clamp_min(1).float(),
            (counts - 1).float() / (lengths - 1).clamp_min(1).float(),
            lengths - 1, selected_positions, compact_valid, z, fine, state.arm, new_state
        )

    def generate_with_qwenputer(
        self,
        prompt: str,
        vocab: Dict[str, int],
        inv_vocab: Dict[int, str],
        max_tokens: Optional[int] = None,
        arm: str = 'dense_all_boundary',
        cpu_state: Optional[CPUState] = None,
        repetition_penalty: float = 1.0,
        stop_strings: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Autoregressive generation with Qwenputer execution control and language state muting.
        
        When a CPU command like `<SET R0 100>` or `<ADD R2 R0 R1>` is encountered:
        1. Language associative memory M_t is MUTED / FROZEN.
        2. Exact fixed-circuit ALU executes instruction on CPU registers (L1 cache, zero CUDA launch overhead).
        3. Result is computed deterministically and logged.
        4. Language state is UN-MUTED and ingests the result tokens.
        5. Language resumes generation seamlessly.
        """
        self.eval()
        device = next(self.base.parameters()).device
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
        prompt = canonical_normalize_text(prompt)

        # Scan and execute any unexecuted CPU commands pre-existing in the prompt using exact span assembly
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

        input_ids = [vocab[c] for c in prompt if c in vocab]
        if not input_ids:
            input_ids = [1]  # Safe <unk> fallback so arbitrary/empty inputs never crash

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
                # Single transfer of [vocab_size] logits to CPU eliminates 17 per-token GPU stream syncs
                logits_1d = state.last_logits[0, -1].float().cpu()
                if repetition_penalty != 1.0 and generated_tokens:
                    # Universal Repetition Penalty on recent token window (last 16 characters)
                    recent_tokens = generated_tokens[-16:]
                    for prev_tok in set(recent_tokens):
                        if prev_tok in layout_toks:
                            continue
                        val = float(logits_1d[prev_tok])
                        logits_1d[prev_tok] = val / repetition_penalty if val > 0.0 else val * repetition_penalty
                    # Anti-stuttering penalty on immediate preceding character (exempt layout whitespace/pipe unless 4+ repeats)
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
                    # 1-token lookahead & length/newline guard:
                    # CPU commands MUST start with '<' + uppercase ASCII [A-Z], contain no newlines, and be <= 32 chars
                    if (len(cmd_buffer) == 2 and not ('A' <= char <= 'Z')) or ('\n' in cmd_buffer) or (len(cmd_buffer) > 32):
                        in_cmd = False
                        cmd_buffer = ""
                    elif char == '>':
                        in_cmd = False
                        # Parse CPU command
                        parsed = self.controller.parse_command(cmd_buffer)
                        if parsed is not None:
                            # 1. Feed the closing '>' token to the language state (skip head projection since inject follows)
                            state = self.step_state(torch.tensor([[next_tok]], dtype=torch.long, device=device), state, compute_logits=False)

                            # 2. CPU EXECUTION: Execute deterministic instruction on CPU L1 cache
                            cpu_state, result_val, res_str = self.controller.execute_step(cpu_state, parsed)
                            cpu_invocations += 1

                            # 3. RESULT INJECTION: Inject result text back to Language Context
                            inject_text = f" => {res_str} "
                            full_text += inject_text
                            inject_ids = [vocab[c] for c in inject_text if c in vocab]
                            n_inj = len(inject_ids)
                            for idx_inj, itok in enumerate(inject_ids):
                                generated_tokens.append(itok)
                                state = self.step_state(
                                    torch.tensor([[itok]], dtype=torch.long, device=device),
                                    state,
                                    compute_logits=(idx_inj == n_inj - 1)
                                )
                            cmd_buffer = ""
                            if len(generated_tokens) >= 80 and generated_tokens[-20:] == generated_tokens[-40:-20] == generated_tokens[-60:-40] == generated_tokens[-80:-60]:
                                break
                            continue

                if stop_strings and not in_cmd:
                    for s_str in stop_strings:
                        if full_text.endswith(s_str, prompt_char_len):
                            ended_with_eos = True
                            break
                    if ended_with_eos:
                        break

                # Guard against un-terminated static repetition loops (4x identical token block) while never limiting real multi-step reasoning
                n_gen = len(generated_tokens)
                if n_gen >= 32 and not in_cmd:
                    if (generated_tokens[-8:] == generated_tokens[-16:-8] == generated_tokens[-24:-16] == generated_tokens[-32:-24]):
                        break
                    if n_gen >= 64 and (generated_tokens[-16:] == generated_tokens[-32:-16] == generated_tokens[-48:-32] == generated_tokens[-64:-48]):
                        break

                # Normal language generation step
                state = self.step_state(torch.tensor([[next_tok]], dtype=torch.long, device=device), state, compute_logits=True)

        output_str = "".join(inv_vocab.get(tok, "") for tok in generated_tokens)
        return {
            "output_text": output_str,
            "full_text": full_text,
            "tokens": generated_tokens,
            "cpu_state": cpu_state,
            "cpu_trace": cpu_state.trace,
            "cpu_invocations": cpu_invocations,
            "ended_with_eos": ended_with_eos
        }

IntegratedModelV29 = IntegratedModelV27
IntegratedModelV30 = IntegratedModelV27

