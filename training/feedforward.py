
from __future__ import annotations

import torch
import torch.nn as nn

from configs.config import Config


class FeedForward(nn.Module):
    """
    SwiGLU FeedForward Network.
    """

    def __init__(self):
        super().__init__()

        self.hidden_size = Config.HIDDEN_SIZE
        self.intermediate_size = Config.INTERMEDIATE_SIZE
        self.use_swiglu = Config.USE_SWIGLU

        self.gate_proj = nn.Linear(
            self.hidden_size,
            self.intermediate_size,
            bias=Config.MLP_BIAS,
        )

        self.up_proj = nn.Linear(
            self.hidden_size,
            self.intermediate_size,
            bias=Config.MLP_BIAS,
        )

        self.down_proj = nn.Linear(
            self.intermediate_size,
            self.hidden_size,
            bias=Config.MLP_BIAS,
        )

        self.dropout = nn.Dropout(Config.DROPOUT)

    def swiglu(
        self,
        gate: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        return torch.nn.functional.silu(gate) * value

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_swiglu:
            gate = self.gate_proj(x)
            value = self.up_proj(x)
            x = self.swiglu(gate, value)
        else:
            x = self.up_proj(x)
            x = torch.nn.functional.gelu(x)

        x = self.down_proj(x)
        x = self.dropout(x)

        return x

    def reset_parameters(self):
        nn.init.normal_(
            self.gate_proj.weight,
            mean=0.0,
            std=Config.INIT_STD,
        )

        nn.init.normal_(
            self.up_proj.weight,
            mean=0.0,
            std=Config.INIT_STD,
        )

        nn.init.normal_(
            self.down_proj.weight,
            mean=0.0,
            std=Config.INIT_STD,
        )

        if Config.MLP_BIAS:
            nn.init.zeros_(self.gate_proj.bias)
            nn.init.zeros_(self.up_proj.bias)
            nn.init.zeros_(self.down_proj.bias)

    def extra_repr(self):
        return (
            f"hidden_size={self.hidden_size}, "
            f"intermediate_size={self.intermediate_size}, "
            f"swiglu={self.use_swiglu}"
        )
        
