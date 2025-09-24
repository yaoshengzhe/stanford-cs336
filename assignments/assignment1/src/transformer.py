from __future__ import annotations

import math
import torch
import torch.nn as nn

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
