
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from configs.config import Config
from training.rmsnorm import RMSNorm
from training.transformer_block import TransformerBlock


class QualcLM(nn.Module):
    """
    Decoder-only Transformer Language Model.
    """

    def __init__(self):
        super().__init__()

        self.vocab_size = Config.VOCAB_SIZE
        self.hidden_size = Config.HIDDEN_SIZE
        self.num_layers = Config.NUM_LAYERS

        self.embed_tokens = nn.Embedding(
            self.vocab_size,
            self.hidden_size,
        )

        self.layers = nn.ModuleList(
            [
                TransformerBlock()
                for _ in range(self.num_layers)
            ]
        )

        self.norm = RMSNorm(
            self.hidden_size,
            eps=Config.RMS_NORM_EPS,
        )

        self.lm_head = nn.Linear(
            self.hidden_size,
            self.vocab_size,
            bias=False,
        )

        self.lm_head.weight = self.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                use_cache=use_cache,
            )

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits

    def reset_parameters(self):
        nn.init.normal_(
            self.embed_tokens.weight,
            mean=0.0,
            std=Config.INIT_STD,
        )

        for layer in self.layers:
            layer.reset_parameters()

        if hasattr(self.norm, "reset_parameters"):
            self.norm.reset_parameters()

    def clear_cache(self):
        for layer in self.layers:
            if hasattr(layer, "clear_cache"):
                layer.clear_cache()

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def extra_repr(self):
        return (
            f"vocab_size={self.vocab_size}, "
            f"hidden_size={self.hidden_size}, "
            f"num_layers={self.num_layers}"
        )
        
