"""Loss functions with class imbalance handling and label smoothing."""
from typing import Dict, List, Optional, Union
import torch
import torch.nn as nn


def build_loss_function(
    class_weights: Optional[Union[Dict[int, float], List[float], torch.Tensor]] = None,
    label_smoothing: float = 0.05,
    device: Optional[torch.device] = None,
) -> nn.CrossEntropyLoss:
    """
    Constructs CrossEntropyLoss with optional class weights and label smoothing.

    Args:
        class_weights: Optional weights per class to penalize false negatives on rare diseases (e.g. melanoma).
        label_smoothing: Regularization factor to prevent overconfidence on noisy dermatoscopic labels.
        device: Device to place weight tensor on.
    """
    weights_tensor = None
    if class_weights is not None:
        if isinstance(class_weights, dict):
            # Sort by class index
            sorted_weights = [class_weights[i] for i in sorted(class_weights.keys())]
            weights_tensor = torch.tensor(sorted_weights, dtype=torch.float32)
        elif isinstance(class_weights, (list, tuple)):
            weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
        elif isinstance(class_weights, torch.Tensor):
            weights_tensor = class_weights.float()

        if device and weights_tensor is not None:
            weights_tensor = weights_tensor.to(device)

    return nn.CrossEntropyLoss(
        weight=weights_tensor,
        label_smoothing=label_smoothing,
    )
