
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.config import Config


class LanguageModelingLoss(nn.Module):
    """
    Causal Language Modeling Loss.
    """

    def __init__(self):
        super().__init__()

        self.ignore_index = -100
        self.label_smoothing = 0.0

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(
                -1,
                shift_logits.size(-1),
            ),
            shift_labels.view(-1),
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
        )

        return loss

    def extra_repr(self):
        return (
            f"ignore_index={self.ignore_index}, "
            f"label_smoothing={self.label_smoothing}"
        )
        
