"""Training callbacks: EarlyStopping and ModelCheckpoint."""
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
import torch

from ..utils.logger import get_logger

logger = get_logger(__name__)


class EarlyStopping:
    """Monitors validation loss or accuracy and signals when to halt training."""

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.001,
        mode: str = "min",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False

    def __call__(self, current_val: float) -> bool:
        """Returns True if training should stop, False otherwise."""
        score = -current_val if self.mode == "min" else current_val

        if self.best_score is None:
            self.best_score = score
            return False

        if score < self.best_score + self.min_delta:
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info("EarlyStopping triggered: halting training.")
                return True
        else:
            self.best_score = score
            self.counter = 0

        return False


class ModelCheckpoint:
    """Saves the best model checkpoint to disk atomically."""

    def __init__(
        self,
        filepath: Union[str, Path],
        monitor: str = "val_loss",
        mode: str = "min",
    ):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.best_value = float("inf") if mode == "min" else float("-inf")

    def step(
        self,
        current_value: float,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: int = 0,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Check if metric improved and save checkpoint."""
        is_best = (
            current_value < self.best_value
            if self.mode == "min"
            else current_value > self.best_value
        )

        if is_best:
            old_best = self.best_value
            self.best_value = current_value
            logger.info(
                f"Metric {self.monitor} improved from {old_best:.4f} to {current_value:.4f}. "
                f"Saving checkpoint to {self.filepath.name}..."
            )

            state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                f"best_{self.monitor}": current_value,
            }
            if optimizer is not None:
                state["optimizer_state_dict"] = optimizer.state_dict()
            if extra_metadata:
                state.update(extra_metadata)

            # Atomic save: write to temp file then rename
            temp_path = self.filepath.with_suffix(".tmp")
            torch.save(state, temp_path)
            temp_path.replace(self.filepath)
            return True

        return False
