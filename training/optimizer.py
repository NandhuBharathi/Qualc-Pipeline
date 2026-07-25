
from __future__ import annotations

import torch
from torch.optim import AdamW

from configs.config import Config


class OptimizerFactory:
    """
    Creates optimizer for Qualc-LM.
    """

    @staticmethod
    def get_parameter_groups(model):
        decay = []
        no_decay = []

        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue

            if (
                parameter.ndim == 1
                or name.endswith(".bias")
                or "norm" in name.lower()
                or "embedding" in name.lower()
            ):
                no_decay.append(parameter)
            else:
                decay.append(parameter)

        return [
            {
                "params": decay,
                "weight_decay": Config.WEIGHT_DECAY,
            },
            {
                "params": no_decay,
                "weight_decay": 0.0,
            },
        ]

    @staticmethod
    def create(model):
        parameter_groups = OptimizerFactory.get_parameter_groups(model)

        optimizer = AdamW(
            parameter_groups,
            lr=Config.LEARNING_RATE,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=Config.WEIGHT_DECAY,
        )

        return optimizer
        
