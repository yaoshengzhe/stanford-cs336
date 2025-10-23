from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Optional

import math
import torch
import torch.nn as nn


def gradient_clipping(parameters: Iterable[torch.nn.Parameter],
                      max_l2_norm: float,
                      eps=1e-6):
    l2 = 0.0
    
    for p in parameters:
        if p.grad is None:
            continue
        l2 += (p.grad.data ** 2).sum()

    l2 = l2.sqrt()

    if l2 > max_l2_norm:
        scaling_factor = max_l2_norm / (l2 + eps)
        for p in parameters:
            if p.grad is None:
                continue
            p.grad.data = p.grad.data * scaling_factor
    
    return parameters

def lr_cosine_schedule(it: int,
                       max_learning_rate: float,
                       min_learning_rate: float,
                       warmup_iters: int,
                       cosine_cycle_iters: int):
    # warm-up
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate
    # cosine annealing
    elif it <= cosine_cycle_iters:
        return min_learning_rate + \
            (1 + math.cos(math.pi *
                          (it - warmup_iters) / (cosine_cycle_iters - warmup_iters))) * \
                          (max_learning_rate - min_learning_rate) / 2
    else:
        return min_learning_rate


def cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    batch_size, vocab_size = inputs.shape

    x = inputs - inputs.max(-1, keepdim=True).values # substract max for numerical stability

    return (-torch.gather(x, dim=1, index=targets.unsqueeze(1)) + \
            x.exp().sum(-1, keepdim=True).log()).mean()


def softmax(x: torch.Tensor) -> torch.Tensor:
    e = torch.exp(x - torch.max(x))
    return e / e.sum(dim=-1, keepdim=True)


def dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Float[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Float[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    # query, key
    QK = torch.einsum('... qd, ... kd -> ... qk', Q, K)
    masked = QK.masked_fill(~mask, float('-inf'))

    # query, key
    sm = softmax(masked / math.sqrt(Q.shape[-1]))

    return torch.einsum('... qk, ... kv -> ... qv', sm, V)


class Linear(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 device: torch.device=None,
                 dtype: torch.dtype=None):
        super().__init__()

        # [out, in]
        self.weights = nn.Parameter(torch.empty(out_features, in_features))

        std = math.sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(self.weights, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum('... i, ji -> ... j', x, self.weights)

    def flops(self, d_in):
        return 2 * d_in * self.weights.shape[0] * self.weights.shape[1]


class Embedding(nn.Module):
    def __init__(self,
                 vocab: int,
                 d_model: int,
                 device: torch.device=None,
                 dtype: torch.dtype=None):
        super().__init__()

        # vocab, d_model
        self.weights = nn.Parameter(torch.empty(vocab, d_model))

        nn.init.trunc_normal_(self.weights, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weights[token_ids] # seq_len, d_model

    def flops(self, d_in):
        return 0


class RMSNorm(nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5,
                 device = None,
                 dtype = None):
        super().__init__()

        self.d_model = d_model
        self.eps = eps

        self.weights = nn.Parameter(torch.ones(d_model))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype

        x = x.to(torch.float32)

        rmsnorm = torch.sqrt(torch.sum(x**2, dim=2, keepdim=True) / x.shape[2] + self.eps)

        result = x / rmsnorm * self.weights

        return result.to(in_dtype)

    def flops(self, d_in):
        return 3 * d_in # 1 multiply (x**2), 1 sum, and 1 division


class SwiGLU(nn.Module):
    def __init__(self,
                 d_model: int,
                 d_ff: int,
                 device = None,
                 dtype = None):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff

        self.w1 = nn.Parameter(torch.Tensor(d_ff, d_model))
        self.w2 = nn.Parameter(torch.Tensor(d_model, d_ff))
        self.w3 = nn.Parameter(torch.Tensor(d_ff, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., d_model]
        xw1 = torch.einsum('... i, ji -> ... j', x, self.w1)
        xw3 = torch.einsum('... i, ji -> ... j', x, self.w3)
        x = xw1 * torch.sigmoid(xw1) * xw3 # [..., d_ff]

        return torch.einsum('... j, ij -> ... i', x, self.w2)

    def flops(self, d_in):
        return 2 * (d_in * self.w1.shape[0] * self.w1.shape[1] + # xw1
                    d_in * self.w3.shape[0] * self.w3.shape[1] + # xw3
                    d_in * self.w2.shape[0] * self.w2.shape[1]) + \
               7 * d_in * self.w1.shape[0] # swiglu, assuming sigmoid take 5 flops per element)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self,
                 theta: float,
                 d_k: int,
                 max_seq_len: int,
                 device=None):
        super().__init__()

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

#        x1, x2, x3, x4, x5, x6

#        x1 cos(t1, 1) + x2 sin(t1, 1)
#        -x1 sin(t1, 1) + x2 cos(t1, 1)
#        x3 cos(t2, 2) + x4 sin(t2, 2)
#        -x3 sin(t2, 2) + x4 cos(t2, 2)
#        x5 cos(t3, 3) + x6 sin(t3, 3)
#        -x5 sin(t3, 3) + x6 cos(t3, 3)

#        x1 cos(t1, 1)    +   -x2 sin(t1, 1)
#        x2 cos(t1, 1)    +    x1 sin(t1, 1)
#        x3 cos(t2, 2)    +   -x4 sin(t2, 2)
#        x4 cos(t2, 2)    +    x3 sin(t2, 2)
#        x5 cos(t3, 3)    +   -x6 sin(t3, 3)
#        x6 cos(t3, 3)    +    x5 sin(t3, 3)

        #x1, x2 -> -x2, x1
        #         0   1
        #        -1   0
        # first cos tensor
        # shape: d_k // 2
        index_k = torch.arange(d_k // 2, dtype=torch.float32)
        # shape: d_k
        index_k = torch.repeat_interleave(index_k, repeats=2)

        # seq_len, d_k
        positions = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1).repeat(1, d_k)

        power = -2*index_k / d_k
        angles = positions * (theta ** power)

        cos = torch.cos(angles)
        sin = torch.sin(angles)

        # print(f'ddddd: cos = {cos.shape}, sin = {sin.shape}, max_seq_len = {max_seq_len}')
        # rotations: d_k, d_k
        self.register_buffer('sin', sin, persistent=False)
        self.register_buffer('cos', cos, persistent=False)


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x: [..., seq_len, d_k]
        # token_positions: [..., seq_len]

        return x * self.cos[token_positions] + self._rearrange(x) * self.sin[token_positions]

    def flops(self, d_in):
        return 2 * d_in

    def _rearrange(self, x: torch.Tensor):
        # take a tensor [q1, q2, q3, ..., qn] and rearrange it so that
        # every pair [q1, q2], [q3, q4], ... becomes [-q2, q1], [-q4, q3], ....

        # shape: [..., n] -> [..., n//2, 2]
        x_pairs = x.view(*x.shape[:-1], -1, 2)

        # apply transformation: [-q2, q1]
        transformed = torch.stack([-x_pairs[..., 1], x_pairs[..., 0]], dim=-1)

        # flatten last two dims back
        return transformed.view(*x.shape)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int,
                 num_heads: int,
                 max_seq_len: int = 0,
                 theta: float = -1.0):

        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        # d_k, d_model
        self.wq = nn.Parameter(torch.empty(d_model, d_model))
        nn.init.trunc_normal_(self.wq)

        # d_k, d_in
        self.wk = nn.Parameter(torch.empty(d_model, d_model))
        nn.init.trunc_normal_(self.wk)

        # d_k, d_in
        self.wv = nn.Parameter(torch.empty(d_model, d_model))
        nn.init.trunc_normal_(self.wv)

        # d_model d_k
        self.wo = nn.Parameter(torch.empty(d_model, d_model))
        nn.init.trunc_normal_(self.wo)

        if theta > 0:
            self.rope = RotaryPositionalEmbedding(theta=theta,
                                                  d_k=d_model // num_heads,
                                                  max_seq_len=max_seq_len)


    def _split_heads(self, qkv: torch.Tensor):
        d_head = self.d_model // self.num_heads

        *leading, seq_len, d_model = qkv.shape

        x = qkv.reshape(*leading, seq_len, self.num_heads, d_head)
        x = x.transpose(-3, -2)
        # now x: ..., num_heads, seq_len, d_head
        return x

    def _merge_heads(self, attention: torch.Tensor):
        *leading, num_heads, seq_len, d_head = attention.shape

        out = attention.transpose(-3, -2)
        return out.reshape(*leading, seq_len, num_heads * d_head)


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        # x: Float[Tensor, " ... sequence_length d_model"],

        Q = self._split_heads(x @ self.wq.T)
        K = self._split_heads(x @ self.wk.T)
        V = self._split_heads(x @ self.wv.T)

        if token_positions is not None:
            # apply RoPE
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        seq_len = x.shape[-2]
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))

        # wo: d_model, d_model
        # dot product attention: d_k, d_k
        attention = dot_product_attention(Q, K, V, mask)
        # ..., seq_len, d_model
        out = self._merge_heads(attention)

        return out @ self.wo.T

    def flops(self, d_in):
        return 2 * self.rope.flops(d_in) + 2 * (d_in * self.d_model * self.d_model) + \
               2 * d_in * self.d_model * self.d_model


class TransformerBlock(nn.Module):
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 max_seq_len: int,
                 theta: int,
                 d_ff: int):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.d_ff = d_ff

        self.attn_rms = RMSNorm(self.d_model)
        self.attn = MultiHeadAttention(self.d_model,
                                       self.num_heads,
                                       self.max_seq_len,
                                       self.theta)

        self.ff_rms = RMSNorm(self.d_model)
        self.ff = SwiGLU(self.d_model, self.d_ff)

    def forward(self, x: torch.Tensor):
        # x: batch, sequence_length, d_model
        seq_len = x.shape[1]

        x = x + self.attn(self.attn_rms(x), torch.arange(seq_len))

        x = x + self.ff(self.ff_rms(x))

        return x

    def flops(self, d_in):
        return self.attn_rms.flops(d_in) + self.attn.flops(d_in) + \
               self.ff_rms.flops(d_in) + self.ff.flops(d_in)


    def load_weights(self, weights: dict[str, Tensor]):
        '''
            weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        '''
        self.attn.load_state_dict({'wq': weights['attn.q_proj.weight'],
                                   'wk': weights['attn.k_proj.weight'],
                                   'wv': weights['attn.v_proj.weight'],
                                   'wo': weights['attn.output_proj.weight']})

        self.attn_rms.load_state_dict({'weights': weights['ln1.weight']})

        self.ff.load_state_dict({'w1': weights['ffn.w1.weight'],
                                 'w2': weights['ffn.w2.weight'],
                                 'w3': weights['ffn.w3.weight']})

        self.ff_rms.load_state_dict({'weights': weights['ln2.weight']})

class TransformerLM(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 context_length: int,
                 d_model: int,
                 num_layers: int,
                 num_heads: int,
                 rope_theta: int,
                 d_ff: int):

        super().__init__()


        self.d_model = d_model

        self.embedding = Embedding(vocab=vocab_size, d_model=d_model)
        self.output = Linear(in_features=d_model, out_features=vocab_size)

        self.blocks = [TransformerBlock(d_model=d_model,
                         num_heads=num_heads,
                         max_seq_len=context_length,
                         theta=rope_theta,
                         d_ff=d_ff) for i in range(num_layers)]

        self.final_rms = RMSNorm(d_model)

    def forward(self, x: torch.Tensor):
        # x: Int[Tensor, " batch_size sequence_length"],
        x = self.embedding(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_rms(x)

        # x: Int[Tensor, " batch_size sequence_length vocab_size"],
        x = self.output(x)
        return x

    def flops(self, input_tokens):
        d_in = input_tokens * self.d_model

        return self.embedding.flops(d_in) + self.output.flops(d_in) + \
               len(self.blocks) * self.blocks[0].flops(d_in) + \
               self.final_rms.flops(d_in)

    def load_weights(self, weights: dict[str, Tensor]):
        keys_to_copy = ['attn.q_proj.weight',
                        'attn.k_proj.weight',
                        'attn.v_proj.weight',
                        'attn.output_proj.weight',
                        'ln1.weight',
                        'ffn.w1.weight',
                        'ffn.w2.weight',
                        'ffn.w3.weight',
                        'ln2.weight',
                        ]


        self.embedding.load_state_dict({'weights': weights['token_embeddings.weight']})

        self.output.load_state_dict({'weights': weights['lm_head.weight']})

        self.final_rms.load_state_dict({'weights': weights['ln_final.weight']})

        for i in range(len(self.blocks)):
            prefix = f"layers.{i}."
            layer_weights = {k: weights[prefix+k] for k in keys_to_copy}

            self.blocks[i].load_weights(layer_weights)


class AdamW(torch.optim.Optimizer):
    def __init__(self, params,
                 lr=1e-3,
                 weight_decay=0.01,
                 betas=(0.9, 0.999),
                 eps=1e-8):

        defaults = {'lr': lr,
                    'decay': weight_decay,
                    'betas': betas,
                    'eps': eps}

        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group['lr']
            decay = group['decay']
            betas = group['betas']
            eps = group['eps']
            
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]
                t = state.get('t', 1) # t must start with 1 in AdamW
                m = state.get('m', torch.zeros(p.shape))
                v = state.get('v', torch.zeros(p.shape))

                
                # AdamW
                m = betas[0] * m + (1 - betas[0]) * grad
                v = betas[1] * v + (1 - betas[1]) * grad**2

                lr_t = lr * math.sqrt(1 - betas[1]**t) / (1 - betas[0]**t)
                
                p.data = p.data - lr_t * m / (v.sqrt() + eps)
                p.data = p.data - lr * decay * p.data
                
                state['t'] = t + 1
                state['m'] = m
                state['v'] = v

        return loss
