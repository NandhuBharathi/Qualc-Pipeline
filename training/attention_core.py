
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):

    def __init__(
        self,
        dropout=0.0,
    ):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query,
        key,
        value,
        attention_mask=None,
    ):

        head_dim = query.size(-1)

        scores = torch.matmul(
            query,
            key.transpose(-2, -1),
        )

        scores = scores / math.sqrt(head_dim)

        if attention_mask is not None:

            scores = scores.masked_fill(
                attention_mask == 0,
                float("-inf"),
            )

        attention = F.softmax(
            scores,
            dim=-1,
        )

        attention = self.dropout(attention)

        output = torch.matmul(
            attention,
            value,
        )

        return output, attention
