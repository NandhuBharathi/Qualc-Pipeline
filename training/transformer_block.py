
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from training.rmsnorm import RMSNorm
from training.attention import MultiHeadAttention
from training.feedforward import FeedForward

from configs.config import Config


class TransformerBlock(nn.Module):
    """
    Decoder Transformer Block.
    """

    def __init__(self):
        super().__init__()

        self.hidden_size = Config.HIDDEN_SIZE

        self.attention_norm = RMSNorm(
            self.hidden_size,
            eps=Config.RMS_NORM_EPS,
        )

        self.attention = MultiHeadAttention()

        self.feedforward_norm = RMSNorm(
            self.hidden_size,
            eps=Config.RMS_NORM_EPS,
        )

        self.feedforward = FeedForward()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
    ) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.attention_norm(hidden_states)
        hidden_states = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states

        hidden_states = self.feedforward_norm(hidden_states)
        hidden_states = self.feedforward(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

    def reset_parameters(self):
        self.attention.reset_parameters()
        self.feedforward.reset_parameters()

        if hasattr(self.attention_norm, "reset_parameters"):
            self.attention_norm.reset_parameters()

        if hasattr(self.feedforward_norm, "reset_parameters"):
            self.feedforward_norm.reset_parameters()

    def clear_cache(self):
        if hasattr(self.attention, "clear_cache"):
            self.attention.clear_cache()

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def extra_repr(self):
        return f"hidden_size={self.hidden_size}"
        
