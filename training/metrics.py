
from __future__ import annotations

import math
import time

import torch


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0.0
        self.sum = 0.0
        self.count = 0
        self.average = 0.0

    def update(
        self,
        value,
        n=1,
    ):
        self.value = float(value)
        self.sum += float(value) * n
        self.count += n
        self.average = self.sum / max(
            1,
            self.count,
        )


class TrainingMetrics:
    def __init__(self):
        self.loss = AverageMeter()
        self.start_time = time.time()
        self.total_tokens = 0

    def update(
        self,
        loss,
        batch_size,
        sequence_length,
    ):
        self.loss.update(
            loss,
            batch_size,
        )
        self.total_tokens += batch_size * sequence_length

    @property
    def elapsed_time(self):
        return time.time() - self.start_time

    @property
    def tokens_per_second(self):
        return self.total_tokens / max(
            self.elapsed_time,
            1e-8,
        )

    @property
    def perplexity(self):
        return math.exp(
            min(
                self.loss.average,
                20,
            )
        )

    def state_dict(self):
        return {
            "loss": self.loss.average,
            "tokens": self.total_tokens,
            "elapsed_time": self.elapsed_time,
        }

    def reset(self):
        self.loss.reset()
        self.start_time = time.time()
        self.total_tokens = 0
        
