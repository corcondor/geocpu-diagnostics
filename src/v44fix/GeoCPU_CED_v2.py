"""GeoCPU_CED_v2: one causal local retrieval branch inside an existing CED decoder.

All v1 parameters are bitwise inherited. Only the local attention output projection
is initialized to zero. No parser, offset trick, special token logits, or pointer
position labels. This is full-prefix causal attention when T <= 256; not a
production sliding-window streaming implementation for arbitrary long sequences.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from GeoCPU_CED_v1 import GeoCPU_CED_v1, CEDConfig, CEDOutput, CEDState, build_delta_tape, build_sequential_tape


class LocalSelect(nn.Module):
    def __init__(self, dim: int = 256, heads: int = 4):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.head_dim = dim // heads
        self.query_norm = nn.LayerNorm(dim)
        self.key_norm = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim,dim,bias=False)
        self.k_proj = nn.Linear(dim,dim,bias=False)
        self.v_proj = nn.Linear(dim,dim,bias=False)
        self.out_proj = nn.Linear(dim,dim,bias=False)
        nn.init.zeros_(self.out_proj.weight)  # sole zero initializer

    def forward_last(self,z:Tensor,encoded:Tensor) -> Tensor:
        """One new query; all encoder positions are past or current for that query."""
        b,t,d=z.shape
        if t!=1 or encoded.shape[0]!=b or encoded.shape[2]!=d:
            raise ValueError("forward_last needs one query and its cached encoder prefix")
        h=self.heads
        q=self.q_proj(self.query_norm(z)).reshape(b,1,h,d//h).transpose(1,2)
        k=self.k_proj(self.key_norm(encoded)).reshape(b,encoded.shape[1],h,d//h).transpose(1,2)
        v=self.v_proj(encoded).reshape(b,encoded.shape[1],h,d//h).transpose(1,2)
        ctx=F.scaled_dot_product_attention(q,k,v,dropout_p=0.,is_causal=False)
        return z+self.out_proj(ctx.transpose(1,2).reshape(b,1,d).to(z.dtype))

    def forward(self, z: Tensor, encoded: Tensor) -> Tensor:
        b,t,d = z.shape
        if encoded.shape != z.shape:
            raise ValueError('LocalSelect needs the matching encoder input prefix')
        h = self.heads
        q = self.q_proj(self.query_norm(z)).reshape(b,t,h,d//h).transpose(1,2)
        k = self.k_proj(self.key_norm(encoded)).reshape(b,t,h,d//h).transpose(1,2)
        v = self.v_proj(encoded).reshape(b,t,h,d//h).transpose(1,2)
        # torch SDPA is causal for full square q/k. Outputs at t never see j>t.
        context = F.scaled_dot_product_attention(q,k,v,attn_mask=None,
                                                  dropout_p=0.0,is_causal=True)
        context = context.transpose(1,2).reshape(b,t,d).to(z.dtype)
        return z + self.out_proj(context)


@dataclass(frozen=True)
class DualStreamState:
    ced: CEDState
    encoded_prefix: Tensor
    def detach(self):
        return DualStreamState(self.ced.detach(),self.encoded_prefix.detach())
    def to(self,device):
        return DualStreamState(self.ced.to(device),self.encoded_prefix.to(device))


class GeoCPU_CED_v2(GeoCPU_CED_v1):
    def __init__(self,cfg: CEDConfig,select_after: int = 2):
        super().__init__(cfg)
        if not 0 <= select_after < cfg.decoder_layers:
            raise ValueError('select_after should be a valid decoder block index')
        self.select_after = int(select_after)
        self.local_select = LocalSelect(cfg.d_model,heads=cfg.heads)

    def prefill(self,tokens:Tensor,state:DualStreamState|None=None)->DualStreamState:
        fresh=state is None
        underlying=self.initial_state(tokens.shape[0]) if fresh else state.ced
        encoded,enc,positions=self._encode(tokens,underlying,fresh=fresh)
        k,v,g,beta=self.writer.parameters_for(encoded)
        tape=build_delta_tape(k,v,g,beta,underlying.memory,
                              self.config.chunk_size,self.config.read_scale)
        next_ced=CEDState(enc,tape.final_memory,positions,encoded[:,-1])
        prefix=encoded if fresh else torch.cat([state.encoded_prefix,encoded],dim=1)
        if prefix.shape[1]>self.config.max_positions:
            raise ValueError('Cache would exceed configured window')
        return DualStreamState(next_ced,prefix)

    def think(self,state:DualStreamState,iterations:int=1,work_state:Tensor|None=None):
        if iterations<1:raise ValueError('iterations must be positive')
        z=state.ced.last_encoded[:,None] if work_state is None else work_state
        def read(q:Tensor):
            return torch.einsum('bthk,bhkv->bthv',q.to(state.ced.memory.dtype),state.ced.memory)*self.config.read_scale
        for _ in range(iterations):
            for i,block in enumerate(self.decoder):
                z=block(z,read)
                if i==self.select_after:
                    z=self.local_select.forward_last(z,state.encoded_prefix)
        return self.head(self.final_norm(z))[:,-1],z

    def step(self,token:Tensor,state:DualStreamState,iterations:int=1)->CEDOutput:
        if token.ndim==1:token=token[:,None]
        if token.ndim!=2 or token.shape[1]!=1:raise ValueError('step expects [B,1]')
        end=self.prefill(token,state)
        logits,z=self.think(end,iterations)
        return CEDOutput(logits[:,None],end,z)

    def forward(self,tokens: Tensor,state: CEDState | None = None,
                iterations: int = 1,backend: str = 'chunk') -> CEDOutput:
        fresh = state is None
        state = self.initial_state(tokens.shape[0]) if fresh else state
        e,enc,positions = self._encode(tokens,state,fresh=fresh)
        k,v,g,beta = self.writer.parameters_for(e)
        if backend == 'chunk':
            tape = build_delta_tape(k,v,g,beta,state.memory,
                                    self.config.chunk_size,self.config.read_scale)
        elif backend == 'sequential':
            tape = build_sequential_tape(k,v,g,beta,state.memory,self.config.read_scale)
        else:
            raise ValueError('backend must be chunk or sequential')
        z=e
        for _ in range(iterations):
            for i,block in enumerate(self.decoder):
                z = block(z,tape.read)
                if i==self.select_after:
                    z = self.local_select(z,e)
        end = CEDState(enc,tape.final_memory,positions,e[:,-1])
        return CEDOutput(self.head(self.final_norm(z)),end,z)


def make_pair_from_checkpoint(ck: dict,select_after:int=2):
    cfg=CEDConfig(**ck['config'])
    a=GeoCPU_CED_v1(cfg)
    b=GeoCPU_CED_v2(cfg,select_after=select_after)
    a.load_state_dict(ck['model'],strict=True)
    miss,extra=b.load_state_dict(ck['model'],strict=False)
    expected={k for k in b.state_dict() if k.startswith('local_select.')}
    if set(miss)!=expected or extra:
        raise RuntimeError(f'Unexpected checkpoint mismatch: {miss}, {extra}')
    # Test exact tensor equality for all inherited state weights.
    for k,v in a.state_dict().items():
        if not torch.equal(v,b.state_dict()[k]):
            raise AssertionError(f'V1 weight not preserved: {k}')
    return a,b
