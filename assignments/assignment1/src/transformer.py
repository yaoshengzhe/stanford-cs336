from __future__ import annotations

import math
import torch
import torch.nn as nn

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
        return self.weights[token_ids]

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
