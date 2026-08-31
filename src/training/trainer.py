"""Core ModelTrainer class running train and evaluation epochs."""
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..utils.logger import get_logger

logger = get_logger(__name__)


class ModelTrainer:
    """Encapsulates training and validation step logic."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        scheduler: Optional[Any] = None,
    ):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Runs one full training epoch. Returns (avg_loss, accuracy)."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()

            # Gradient clipping to prevent exploding gradients during fine-tuning
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        epoch_loss = running_loss / max(total, 1)
        epoch_acc = (correct / max(total, 1)) * 100.0
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> Tuple[float, float, List[int], List[int]]:
        """Evaluates model on validation or test DataLoader without gradients."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds: List[int] = []
        all_targets: List[int] = []

        for inputs, targets in val_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            all_preds.extend(predicted.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())

        val_loss = running_loss / max(total, 1)
        val_acc = (correct / max(total, 1)) * 100.0
        return val_loss, val_acc, all_targets, all_preds
