
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.config import Config

from training.rope import (
    RotaryEmbedding,
    apply_rotary_pos_emb,
)

from training.attention_core import (
    ScaledDotProductAttention,
)

from training.kv_cache import KVCache


class MultiHeadAttention(nn.Module):
    """
    Production Multi-Head Attention.

    Features
    --------
    - Multi Head Attention
    - Rotary Position Embedding (RoPE)
    - KV Cache
    - GQA Ready
    - SDPA Ready
    - FlashAttention Ready
    """

    def __init__(self):

        super().__init__()

        self.hidden_size = Config.HIDDEN_SIZE

        self.num_heads = Config.NUM_HEADS

        self.num_kv_heads = Config.NUM_KV_HEADS

        self.head_dim = (
            self.hidden_size //
            self.num_heads
        )

        self.dropout = Config.ATTENTION_DROPOUT

        self.use_gqa = Config.USE_GQA

        self.use_cache = Config.USE_KV_CACHE

        self.use_flash = Config.USE_FLASH_ATTENTION

        self.use_sdpa = Config.USE_SDPA

        self._validate_config()

        self.q_proj = nn.Linear(
            self.hidden_size,
            self.hidden_size,
            bias=Config.ATTENTION_BIAS,
        )

        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=Config.ATTENTION_BIAS,
        )

        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=Config.ATTENTION_BIAS,
        )

        self.out_proj = nn.Linear(
            self.hidden_size,
            self.hidden_size,
            bias=Config.ATTENTION_BIAS,
        )

        self.rope = RotaryEmbedding(
            self.head_dim
        )

        self.attention = (
            ScaledDotProductAttention(
                dropout=self.dropout
            )
        )

        self.kv_cache = KVCache()

        self.dropout_layer = nn.Dropout(
            self.dropout
        )

    def _validate_config(self):

        assert (
            self.hidden_size %
            self.num_heads == 0
        ), (
            "HIDDEN_SIZE must be divisible "
            "by NUM_HEADS."
        )

        assert (
            self.num_heads %
            self.num_kv_heads == 0
        ), (
            "NUM_HEADS must be divisible "
            "by NUM_KV_HEADS."
        )

    def clear_cache(self):

        self.kv_cache.clear()

    def split_heads(
        self,
        x: torch.Tensor,
        num_heads: int,
    ) -> torch.Tensor:

        batch_size, seq_len, _ = x.size()

        x = x.view(
            batch_size,
            seq_len,
            num_heads,
            self.head_dim,
        )

        return x.transpose(
            1,
            2,
        )

    def merge_heads(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        batch_size, _, seq_len, _ = x.size()

        x = (
            x.transpose(
                1,
                2,
            )
            .contiguous()
        )

        return x.view(
            batch_size,
            seq_len,
            self.hidden_size,
        )

    def repeat_kv(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        if (
            self.num_heads ==
            self.num_kv_heads
        ):
            return x

        repeat = (
            self.num_heads //
            self.num_kv_heads
        )

        return (
            x.unsqueeze(2)
            .expand(
                -1,
                -1,
                repeat,
                -1,
                -1,
            )
            .reshape(
                x.size(0),
                self.num_heads,
                x.size(2),
                self.head_dim,
            )
        )

    def apply_rope(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ):

        seq_len = query.size(-2)

        cos, sin = self.rope(
            seq_len=seq_len,
            device=query.device,
        )

        return apply_rotary_pos_emb(
            query,
            key,
            cos,
            sin,
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
    ) -> torch.Tensor:

        if use_cache is None:
            use_cache = (
                self.use_cache and
                not self.training
            )

        query = self.q_proj(x)
        key = self.k_proj(x)
        value = self.v_proj(x)

        query = self.split_heads(
            query,
            self.num_heads,
        )

        key = self.split_heads(
            key,
            self.num_kv_heads,
        )

        value = self.split_heads(
            value,
            self.num_kv_heads,
        )

        query, key = self.apply_rope(
            query,
            key,
        )

        key = self.repeat_kv(key)

        value = self.repeat_kv(value)

        if use_cache:

            key, value = (
                self.kv_cache.update(
                    key,
                    value,
                )
            )

        if (
            attention_mask is not None
            and attention_mask.dtype != torch.bool
        ):
            attention_mask = (
                attention_mask.bool()
            )

        if (
            self.use_sdpa and
            hasattr(
                F,
                "scaled_dot_product_attention",
            )
        ):

            attention_output = (
                F.scaled_dot_product_attention(
                    query=query,
                    key=key,
                    value=value,
                    attn_mask=attention_mask,
                    dropout_p=(
                        self.dropout
                        if self.training
                        else 0.0
                    ),
                    is_causal=(
                        attention_mask is None
                    ),
                )
            )

        else:

            attention_output = (
                self.attention(
                    query=query,
                    key=key,
                    value=value,
                    mask=attention_mask,
                )
            )

        attention_output = (
            self.merge_heads(
                attention_output
            )
        )

        attention_output = (
            self.out_proj(
                attention_output
            )
        )

        attention_output = (
            self.dropout_layer(
                attention_output
            )
        )

        return attention_output

    def reset_parameters(self):

        nn.init.normal_(
            self.q_proj.weight,
            mean=0.0,
            std=Config.INIT_STD,
        )

        nn.init.normal_(
            self.k_proj.weight,
            mean=0.0,
            std=Config.INIT_STD,
        )

        nn.init.normal_(
            self.v_proj.weight,
            mean=0.0,
            std=Config.INIT_STD,
        )

        nn.init.normal_(
            self.out_proj.weight,
            mean=0.0,
            std=Config.INIT_STD,
        )

        if Config.ATTENTION_BIAS:

            nn.init.zeros_(self.q_proj.bias)
            nn.init.zeros_(self.k_proj.bias)
            nn.init.zeros_(self.v_proj.bias)
            nn.init.zeros_(self.out_proj.bias)

    def extra_repr(self):

        return (
            f"hidden_size={self.hidden_size}, "
            f"num_heads={self.num_heads}, "
            f"num_kv_heads={self.num_kv_heads}, "
            f"head_dim={self.head_dim}"
        )

    @property
    def device(self):

        return next(
            self.parameters()
        ).device

    @property
    def dtype(self):

        return next(
            self.parameters()
        ).dtype

    def train(
        self,
        mode: bool = True,
    ):

        super().train(mode)

        if mode:
            self.clear_cache()

        return self
        
