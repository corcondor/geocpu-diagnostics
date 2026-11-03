"""AST-identical official H-Net routing/chunk/STE/ratio primitives.

Extracted from the pinned MIT-licensed upstream files identified below. Only
imports and provenance/AST verification are added. No Mamba/FlashAttention is
needed for these primitives. This module does not construct or execute a model.
"""
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

UPSTREAM_LICENSE = 'MIT License\n\nCopyright (c) 2025 brwa-cartesia\n\nPermission is hereby granted, free of charge, to any person obtaining a copy\nof this software and associated documentation files (the "Software"), to deal\nin the Software without restriction, including without limitation the rights\nto use, copy, modify, merge, publish, distribute, sublicense, and/or sell\ncopies of the Software, and to permit persons to whom the Software is\nfurnished to do so, subject to the following conditions:\n\nThe above copyright notice and this permission notice shall be included in all\ncopies or substantial portions of the Software.\n\nTHE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\nIMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\nFITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\nAUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\nLIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\nOUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE\nSOFTWARE.\n'

OFFICIAL_REFERENCE = {'repository': 'https://github.com/goombalab/hnet', 'commit': '3673fe1217ebeb0d1438c7c71d58d32bdd190ec2', 'source_sha256': {'hnet/modules/dc.py': 'e068281e3d27226f0d233b1507073209900c0de1099f64ada6a8f8481a0ccc86', 'hnet/models/hnet.py': '32058971e89c6fc39e8b00dc6af567d3222ac57f1dd6e599adc8deb43b7cd86e', 'hnet/utils/train.py': 'cf1b599709ad85ecb815f7e405f56b27f32a468c7179e72eb0c921f0b66e8127'}, 'ast_sha256': {'RoutingModuleOutput': '37c2e1e9f249b6a176aef3124ac9a56b36d41ca1e41cae9a9642425815a53f8d', 'RoutingModuleState': 'a68ed1f5a8ac5bd69bb64413aa67707fc9f5df476d9383b226ce4b5b1eeef40c', 'RoutingModule': 'b7b74ce2dbecb61ae8952730c07ba0e7213a96bacc2dbc1b808e72942d22c30d', 'ChunkLayer': '17bc76967bf4e87efb60c608ce654a71d07e6a0b440a7e5e559030907c0509be', 'STE': 'e7cd1c6b9249b558634f08caf8191cfd042ca40159f31a37273ac03cd15d613a', 'ste_func': '968a88a89f1246092a3c8f5f88088e0f3231f9b1c84ae92307f80593489b463b', 'load_balancing_loss': '4009d7eade57deca14e8fd378aa90ef51bc4b6bd5d643db2e163b59457cd1f7d'}, 'source_nodes': {'hnet/modules/dc.py': ['RoutingModuleOutput', 'RoutingModuleState', 'RoutingModule', 'ChunkLayer'], 'hnet/models/hnet.py': ['STE', 'ste_func'], 'hnet/utils/train.py': ['load_balancing_loss']}}

@dataclass
class RoutingModuleOutput:
    boundary_prob: torch.Tensor
    boundary_mask: torch.Tensor
    selected_probs: torch.Tensor


@dataclass
class RoutingModuleState:
    """
    The state of the routing module.

    Contains
        - [has_seen_tokens] (batch_size,) bool tensor. Whether that batch element has processed any tokens yet.
        - [last_hidden_state] (batch_size, d_model) tensor. The last hidden state of the batch element (used for boundary prediction).
    """

    has_seen_tokens: torch.Tensor  # (batch_size,)
    last_hidden_state: torch.Tensor  # (batch_size, d_model)


class RoutingModule(nn.Module):

    def __init__(self, d_model, device=None, dtype=None):
        self.d_model = d_model
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.q_proj_layer = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)
        self.k_proj_layer = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)
        with torch.no_grad():
            self.q_proj_layer.weight.copy_(torch.eye(d_model))
            self.k_proj_layer.weight.copy_(torch.eye(d_model))
        self.q_proj_layer.weight._no_reinit = True
        self.k_proj_layer.weight._no_reinit = True

    def allocate_inference_cache(self, batch_size, max_seqlen, device, dtype=None):
        return RoutingModuleState(
            has_seen_tokens=torch.zeros(batch_size, device=device, dtype=torch.bool),
            last_hidden_state=torch.zeros(
                batch_size, self.d_model, device=device, dtype=dtype
            ),
        )

    def forward(self, hidden_states, cu_seqlens=None, mask=None, inference_params=None):
        assert (mask is not None) or (
            cu_seqlens is not None
        ), "Either mask or cu_seqlens must be provided"

        if inference_params is not None:
            assert (
                mask is not None
            ), "Mask must be provided if inference_params is provided"
            assert (
                ~inference_params.has_seen_tokens
            ).all(), "Cannot have seen tokens when inference_params is not provided"

        if cu_seqlens is not None:
            # We are in packed mode, so hidden_states is (T, D). Make it (B, T, D)
            hidden_states = hidden_states.unsqueeze(0)

        cos_sim = torch.einsum(
            "b l d, b l d -> b l",
            F.normalize(self.q_proj_layer(hidden_states[:, :-1]), dim=-1),
            F.normalize(self.k_proj_layer(hidden_states[:, 1:]), dim=-1),
        )
        # this clamp should no-op as long as no precision issues are encountered
        boundary_prob = torch.clamp(((1 - cos_sim) / 2), min=0.0, max=1.0)

        # Force boundary probability of the first element to 1.0
        PAD_PROB = 1.0
        boundary_prob = F.pad(boundary_prob, (1, 0), "constant", PAD_PROB)

        if cu_seqlens is not None:
            boundary_prob = boundary_prob.squeeze(0)
            boundary_prob[cu_seqlens[:-1]] = PAD_PROB

        boundary_prob = torch.stack(((1 - boundary_prob), boundary_prob), dim=-1)

        selected_idx = torch.argmax(boundary_prob, dim=-1)

        boundary_mask = selected_idx == 1  # (shape hidden_states.shape[:-1])
        if mask is not None:
            # No invalid tokens can be selected
            boundary_mask = boundary_mask & mask

        if inference_params is not None:
            has_mask = mask.any(dim=-1)
            inference_params.has_seen_tokens.copy_(
                has_mask | inference_params.has_seen_tokens
            )
            last_mask = torch.clamp(mask.sum(dim=-1) - 1, min=0)
            inference_params.last_hidden_state.copy_(
                torch.where(
                    has_mask,
                    hidden_states[
                        torch.arange(
                            hidden_states.shape[0], device=hidden_states.device
                        ),
                        last_mask,
                    ],
                    inference_params.last_hidden_state,
                )
            )

        selected_probs = boundary_prob.gather(
            dim=-1, index=selected_idx.unsqueeze(-1)
        )  # (shape hidden_states.shape[:-1], 1)

        return RoutingModuleOutput(
            boundary_prob=boundary_prob,  # (shape hidden_states.shape[:-1], 2)
            boundary_mask=boundary_mask,  # (shape hidden_states.shape[:-1])
            selected_probs=selected_probs,  # (shape hidden_states.shape[:-1], 1)
        )

    def step(self, hidden_states, inference_params):
        # hidden_states is (B, 1, D)
        hidden_states = hidden_states.squeeze(1)
        cos_sim = torch.einsum(
            "b d, b d -> b",
            F.normalize(self.q_proj_layer(inference_params.last_hidden_state), dim=-1),
            F.normalize(self.k_proj_layer(hidden_states), dim=-1),
        )
        boundary_prob = torch.clamp(((1 - cos_sim) / 2), min=0.0, max=1.0)
        inference_params.last_hidden_state.copy_(hidden_states)
        boundary_prob = torch.where(
            inference_params.has_seen_tokens,
            boundary_prob,
            torch.ones_like(boundary_prob),
        )
        boundary_prob = torch.stack(((1 - boundary_prob), boundary_prob), dim=-1)

        inference_params.has_seen_tokens.copy_(
            torch.ones_like(inference_params.has_seen_tokens)
        )
        return RoutingModuleOutput(
            boundary_prob=boundary_prob,  # (B, 2)
            boundary_mask=boundary_prob[..., 1] > 0.5,  # (B,)
            selected_probs=boundary_prob.max(dim=-1).values.unsqueeze(-1),  # (B, 1)
        )


class ChunkLayer(nn.Module):

    def forward(self, hidden_states, boundary_mask, cu_seqlens=None, mask=None):
        assert (mask is not None) or (
            cu_seqlens is not None
        ), "Either mask or cu_seqlens must be provided"

        if cu_seqlens is not None:
            next_hidden_states = hidden_states[boundary_mask]
            next_cu_seqlens = F.pad(
                boundary_mask.cumsum(dim=0)[cu_seqlens[1:] - 1], (1, 0)
            )
            next_max_seqlen = int((next_cu_seqlens[1:] - next_cu_seqlens[:-1]).max())
            next_mask = None
        else:
            next_cu_seqlens = None
            num_tokens = boundary_mask.sum(dim=-1)
            next_max_seqlen = int(num_tokens.max())

            device = hidden_states.device
            L = hidden_states.shape[1]
            token_idx = (
                torch.arange(L, device=device)[None, :] + (~boundary_mask).long() * L
            )
            seq_sorted_indices = torch.argsort(token_idx, dim=1)

            next_hidden_states = torch.gather(
                hidden_states,
                dim=1,
                index=seq_sorted_indices[:, :next_max_seqlen, None].expand(
                    -1, -1, hidden_states.shape[-1]
                ),
            )

            next_mask = (
                torch.arange(next_max_seqlen, device=device)[None, :]
                < num_tokens[:, None]
            )
            next_max_seqlen = None

        return next_hidden_states, next_cu_seqlens, next_max_seqlen, next_mask

    def step(self, hidden_states, boundary_mask):
        return hidden_states[boundary_mask]


class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.ones_like(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = grad_output
        return grad_x


def ste_func(x):
    return STE.apply(x)


def load_balancing_loss(
    router_output: RoutingModuleOutput,
    N: float,
) -> torch.Tensor:
    """
    Compute the load balancing loss.
    
    NOTE: This is the loss we used for all experiments. It computes the loss on each device/minibatch, and then averages the loss over all devices/minibatches.
    It is possible that computing the loss on each example is better, or that computing the statistics over the entire (global) batch would have been better.

    Args:
        router_output: The output of the routing module.
        N: The number of "experts", i.e. the downsampling factor. Can be a float (e.g. 2.5) or an integer (e.g. 3), but must be greater than 1.

    Returns:
        A single tensor, the load balancing loss.
    """
    boundary_prob = router_output.boundary_prob
    tokenized_prob = boundary_prob[..., -1]
    boundary_mask = router_output.boundary_mask

    true_ratio = boundary_mask.float().mean()
    average_prob = tokenized_prob.float().mean()

    return (
        (1 - true_ratio) * (1 - average_prob) +
        (true_ratio) * (average_prob) * (N-1)
    ) * N / (N-1)

def verify_reference_ast(source_texts):
    """Compare official node ASTs; source_texts uses pinned repository paths."""
    import ast
    import hashlib
    actual = {}
    for path, names in OFFICIAL_REFERENCE['source_nodes'].items():
        source = source_texts[path]
        if hashlib.sha256(source.encode('utf8')).hexdigest() != OFFICIAL_REFERENCE['source_sha256'][path]:
            raise ValueError('Official source bytes differ: ' + path)
        tree = ast.parse(source)
        for name in names:
            node = next(n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name)
            actual[name] = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
    if actual != OFFICIAL_REFERENCE['ast_sha256']:
        raise ValueError('Official node AST differs')
    return {'passed': True, 'commit': OFFICIAL_REFERENCE['commit'], 'nodes': len(actual)}
