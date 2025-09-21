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

