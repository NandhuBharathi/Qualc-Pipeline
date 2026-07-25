
from __future__ import annotations

import math

from torch.optim.lr_scheduler import LambdaLR

from configs.config import Config


class SchedulerFactory:
    """
    Creates learning rate scheduler.
    """

    @staticmethod
    def lr_lambda(
        current_step,
        total_steps,
    ):
        if current_step < Config.WARMUP_STEPS:
            return current_step / max(
                1,
                Config.WARMUP_STEPS,
            )

        progress = (current_step - Config.WARMUP_STEPS) / max(
            1,
            total_steps - Config.WARMUP_STEPS,
        )

        progress = min(
            max(progress, 0.0),
            1.0,
        )

        return 0.5 * (1.0 + math.cos(math.pi * progress))

    @staticmethod
    def create(
        optimizer,
        total_steps,
    ):
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda step: SchedulerFactory.lr_lambda(
                step,
                total_steps,
            ),
        )

        return scheduler
        
