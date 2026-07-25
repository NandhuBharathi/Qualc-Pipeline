
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from configs.config import Config


class CheckpointManager:
    def __init__(self):
        self.directory = Config.CHECKPOINT_DIR
        self.directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    def _checkpoint_path(
        self,
        step: int,
    ) -> Path:
        filename = (
            f"{Config.CHECKPOINT_PREFIX}"
            f"{step}"
            f"{Config.CHECKPOINT_EXTENSION}"
        )

        return self.directory / filename

    @property
    def latest_path(self) -> Path:
        return self.directory / Config.LATEST_CHECKPOINT_NAME

    @property
    def best_path(self) -> Path:
        return self.directory / Config.BEST_CHECKPOINT_NAME

    @property
    def final_path(self) -> Path:
        return self.directory / Config.FINAL_CHECKPOINT_NAME

    def build_checkpoint(
        self,
        *,
        model,
        optimizer,
        scheduler=None,
        scaler=None,
        epoch: int,
        step: int,
        best_loss: Optional[float] = None,
    ) -> dict:
        checkpoint = {
            "epoch": epoch,
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_loss": best_loss,
        }

        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        if scaler is not None:
            checkpoint["scaler_state_dict"] = scaler.state_dict()

        if Config.CHECKPOINT_METADATA:
            checkpoint["metadata"] = {
                "project_name": Config.PROJECT_NAME,
                "model_name": Config.MODEL_NAME,
                "version": Config.VERSION,
            }

        return checkpoint

    def save(
        self,
        *,
        model,
        optimizer,
        epoch: int,
        step: int,
        scheduler=None,
        scaler=None,
        best_loss: Optional[float] = None,
    ):
        checkpoint = self.build_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            step=step,
            best_loss=best_loss,
        )

        step_path = self._checkpoint_path(step)

        torch.save(
            checkpoint,
            step_path,
        )

        if Config.SAVE_LATEST:
            torch.save(
                checkpoint,
                self.latest_path,
            )

        self.cleanup()

    def load(
        self,
        path: Optional[Path] = None,
        *,
        model,
        optimizer=None,
        scheduler=None,
        scaler=None,
    ):
        if path is None:
            path = self.latest_path

        if not Path(path).exists():
            return None

        checkpoint = torch.load(
            path,
            map_location="cpu",
        )

        model.load_state_dict(checkpoint["model_state_dict"])

        if (
            optimizer is not None
            and "optimizer_state_dict" in checkpoint
        ):
            optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )

        if (
            scheduler is not None
            and "scheduler_state_dict" in checkpoint
        ):
            scheduler.load_state_dict(
                checkpoint["scheduler_state_dict"]
            )

        if (
            scaler is not None
            and "scaler_state_dict" in checkpoint
        ):
            scaler.load_state_dict(
                checkpoint["scaler_state_dict"]
            )

        return checkpoint

    def save_best(
        self,
        checkpoint: dict,
    ):
        if Config.SAVE_BEST:
            torch.save(
                checkpoint,
                self.best_path,
            )

    def save_final(
        self,
        checkpoint: dict,
    ):
        if Config.SAVE_FINAL:
            torch.save(
                checkpoint,
                self.final_path,
            )

    def cleanup(self):
        checkpoints = sorted(
            self.directory.glob(
                f"{Config.CHECKPOINT_PREFIX}*{Config.CHECKPOINT_EXTENSION}"
            ),
            key=lambda path: path.stat().st_mtime,
        )

        while (
            len(checkpoints)
            > Config.KEEP_LAST_CHECKPOINTS
        ):
            checkpoints[0].unlink()
            checkpoints.pop(0)

    def auto_resume(
        self,
    ) -> Optional[Path]:
        if (
            Config.AUTO_RESUME
            and self.latest_path.exists()
        ):
            return self.latest_path

        return None
        
