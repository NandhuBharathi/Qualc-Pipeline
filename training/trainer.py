
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import torch
from torch.amp import GradScaler

from configs.config import Config
from training.loss import LanguageModelingLoss
from training.optimizer import OptimizerFactory
from training.scheduler import SchedulerFactory
from training.checkpoint import CheckpointManager
from training.metrics import TrainingMetrics


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        valid_loader=None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.device = (
            device
            if device is not None
            else torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        self.model.to(self.device)

        self.criterion = LanguageModelingLoss()

        self.optimizer = OptimizerFactory.create(self.model)

        total_steps = (
            len(train_loader)
            * Config.NUM_EPOCHS
        )

        self.scheduler = SchedulerFactory.create(
            optimizer=self.optimizer,
            total_steps=total_steps,
        )

        self.metrics = TrainingMetrics()

        self.checkpoint = CheckpointManager()

        self.scaler = GradScaler(
            enabled=(
                torch.cuda.is_available()
                and Config.USE_AMP
            )
        )

        self.start_epoch = 0
        self.global_step = 0
        self.best_loss = float("inf")

        self.start_time = time.time()
        if Config.AUTO_RESUME:
            checkpoint_path = self.checkpoint.auto_resume()

            if checkpoint_path is not None:
                checkpoint = self.load_checkpoint(checkpoint_path)

                if checkpoint is not None:
                    print(
                        f"[Resume] "
                        f"Epoch={self.start_epoch} "
                        f"Step={self.global_step}"
                    )

        self.model.train()

    def save_checkpoint(self):
        self.checkpoint.save(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            epoch=self.start_epoch,
            step=self.global_step,
            best_loss=self.best_loss,
        )

    def load_checkpoint(self, checkpoint_path):
        checkpoint = self.checkpoint.load(
            path=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
        )

        if checkpoint is not None:
            self.start_epoch = checkpoint.get(
                "epoch",
                0,
            )

            self.global_step = checkpoint.get(
                "step",
                0,
            )

            self.best_loss = checkpoint.get(
                "best_loss",
                float("inf"),
            )

        return checkpoint

    def state_dict(self):
        return {
            "epoch": self.start_epoch,
            "step": self.global_step,
            "best_loss": self.best_loss,
        }

    @property
    def device_type(self):
        return self.device.type

    @property
    def current_lr(self):
        return self.optimizer.param_groups[0]["lr"]

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"device={self.device}, "
            f"step={self.global_step}, "
            f"best_loss={self.best_loss:.6f})"
        )

    def train_epoch(self, epoch: int):
        self.model.train()
        self.metrics.reset()

        self.optimizer.zero_grad(set_to_none=True)

        accumulation_steps = Config.GRADIENT_ACCUMULATION_STEPS

        autocast_enabled = (
            torch.cuda.is_available()
            and Config.USE_AMP
        )

        for batch_index, batch in enumerate(self.train_loader):
            input_ids = batch["input_ids"].to(
                self.device,
                non_blocking=True,
            )

            labels = batch["labels"].to(
                self.device,
                non_blocking=True,
            )

            attention_mask = batch.get(
                "attention_mask",
                None,
            )

            if attention_mask is not None:
                attention_mask = attention_mask.to(
                    self.device,
                    non_blocking=True,
                )

            with torch.amp.autocast(
                device_type=self.device.type,
                enabled=autocast_enabled,
            ):
                logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                loss = self.criterion(
                    logits=logits,
                    labels=labels,
                )

                loss = loss / accumulation_steps

            self.scaler.scale(loss).backward()

            should_step = (
                (batch_index + 1) % accumulation_steps == 0
                or (batch_index + 1) == len(self.train_loader)
            )

            if should_step:
                self.scaler.unscale_(self.optimizer)

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    Config.MAX_GRAD_NORM,
                )

                self.scaler.step(
                    self.optimizer,
                )

                self.scaler.update()

                self.optimizer.zero_grad(
                    set_to_none=True,
                )

                self.scheduler.step()

                self.global_step += 1

            batch_loss = (
                loss.detach().item()
                * accumulation_steps
            )

            batch_size = input_ids.size(0)
            token_count = input_ids.numel()

            self.metrics.update(
                loss=batch_loss,
                batch_size=batch_size,
                sequence_length=token_count // batch_size,
            )

            if (
                self.global_step > 0
                and self.global_step % Config.LOG_INTERVAL == 0
                and should_step
            ):
                print(
                    f"[Epoch {epoch + 1}/{Config.NUM_EPOCHS}] "
                    f"Step {self.global_step} | "
                    f"Loss {self.metrics.loss.average:.4f} | "
                    f"Perplexity {self.metrics.perplexity:.4f} | "
                    f"Tokens/s {self.metrics.tokens_per_second:.2f}"
                )

            if (
                self.global_step > 0
                and self.global_step % Config.CHECKPOINT_INTERVAL == 0
                and should_step
            ):
                self.checkpoint.save(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    scaler=self.scaler,
                    epoch=epoch,
                    step=self.global_step,
                    best_loss=self.best_loss,
                )

        return {
            "loss": self.metrics.loss.average,
            "perplexity": self.metrics.perplexity,
            "tokens": self.metrics.total_tokens,
            "steps": self.global_step,
        }

    @torch.no_grad()
    def validate(self):
        if self.valid_loader is None:
            return None

        self.model.eval()

        total_loss = 0.0
        total_batches = 0

        for batch in self.valid_loader:
            input_ids = batch["input_ids"].to(
                self.device,
                non_blocking=True,
            )

            labels = batch["labels"].to(
                self.device,
                non_blocking=True,
            )

            attention_mask = batch.get(
                "attention_mask",
                None,
            )

            if attention_mask is not None:
                attention_mask = attention_mask.to(
                    self.device,
                    non_blocking=True,
                )

            with torch.amp.autocast(
                device_type=self.device.type,
                enabled=(
                    torch.cuda.is_available()
                    and Config.USE_AMP
                ),
            ):
                logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                loss = self.criterion(
                    logits=logits,
                    labels=labels,
                )

            total_loss += loss.item()
            total_batches += 1

        average_loss = total_loss / max(total_batches, 1)

        perplexity = torch.exp(
            torch.tensor(average_loss)
        ).item()

        self.model.train()

        return {
            "loss": average_loss,
            "perplexity": perplexity,
        }

    def train(self):
        print("=" * 60)
        print("Starting Qualc-LM Training")
        print("=" * 60)

        for epoch in range(
            self.start_epoch,
            Config.NUM_EPOCHS,
        ):
            train_metrics = self.train_epoch(
                epoch=epoch,
            )

            validation_metrics = self.validate()

            if validation_metrics is not None:
                print(
                    f"[Validation] "
                    f"Loss={validation_metrics['loss']:.4f} | "
                    f"Perplexity={validation_metrics['perplexity']:.4f}"
                )

                if (
                    validation_metrics["loss"]
                    < self.best_loss
                ):
                    self.best_loss = validation_metrics["loss"]

                    best_checkpoint = self.checkpoint.build_checkpoint(
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        scaler=self.scaler,
                        epoch=epoch,
                        step=self.global_step,
                        best_loss=self.best_loss,
                    )

                    self.checkpoint.save_best(best_checkpoint)

                    print(
                        f"[Best] "
                        f"Validation Loss={self.best_loss:.4f}"
                    )

            print(
                f"[Epoch {epoch + 1}] "
                f"Train Loss={train_metrics['loss']:.4f} | "
                f"Train Perplexity={train_metrics['perplexity']:.4f}"
            )

            self.checkpoint.save(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                epoch=epoch,
                step=self.global_step,
                best_loss=self.best_loss,
            )

        final_checkpoint = self.checkpoint.build_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            epoch=Config.NUM_EPOCHS,
            step=self.global_step,
            best_loss=self.best_loss,
        )

        self.checkpoint.save_final(final_checkpoint)

        total_time = time.time() - self.start_time

        print("=" * 60)
        print("Training Completed Successfully")
        print("=" * 60)
        print(f"Total Epochs      : {Config.NUM_EPOCHS}")
        print(f"Global Steps      : {self.global_step}")
        print(f"Best Loss         : {self.best_loss:.6f}")
        print(f"Training Time (s) : {total_time:.2f}")

        return {
            "epochs": Config.NUM_EPOCHS,
            "global_step": self.global_step,
            "best_loss": self.best_loss,
            "training_time": total_time,
        }
        
