"""
Medical-grade data augmentation pipeline for dermatoscopic skin lesion classification.
Restricted to realistic variations that do NOT distort clinical pathology
(e.g., preserving lesion asymmetry, melanin coloration boundaries, and structural features).
"""
from typing import Optional, Tuple
import torch
import torchvision.transforms as T


def get_training_transforms(
    image_size: Tuple[int, int] = (224, 224),
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    horizontal_flip_prob: float = 0.5,
    vertical_flip_prob: float = 0.5,
    rotation_degrees: int = 30,
    zoom_scale: Tuple[float, float] = (0.85, 1.05),
    aspect_ratio: Tuple[float, float] = (0.90, 1.10),
    brightness_factor: float = 0.10,
    contrast_factor: float = 0.10,
    saturation_factor: float = 0.08,
    hue_factor: float = 0.02,
) -> T.Compose:
    """
    Returns the training transformation pipeline.
    
    Clinical Rationales:
    - Flips & Rotations: Dermatoscopes can capture lesions from any orientation (dermatoscope angle invariance).
    - Mild zoom/crop: Mimics varying camera distances without cutting off lesion margins.
    - Bounded ColorJitter: Light illumination differences occur between dermatoscopes, but hue is strictly
      bounded to 0.02 so erythematous or melanocytic pigmentation is never falsified into a different lesion type.
    """
    return T.Compose([
        T.Resize((int(image_size[0] * 1.12), int(image_size[1] * 1.12))),
        T.RandomResizedCrop(
            size=image_size,
            scale=zoom_scale,
            ratio=aspect_ratio,
            interpolation=T.InterpolationMode.BILINEAR,
        ),
        T.RandomHorizontalFlip(p=horizontal_flip_prob),
        T.RandomVerticalFlip(p=vertical_flip_prob),
        T.RandomRotation(
            degrees=rotation_degrees,
            interpolation=T.InterpolationMode.BILINEAR,
        ),
        T.ColorJitter(
            brightness=brightness_factor,
            contrast=contrast_factor,
            saturation=saturation_factor,
            hue=hue_factor,
        ),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def get_validation_transforms(
    image_size: Tuple[int, int] = (224, 224),
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> T.Compose:
    """
    Returns deterministic validation / testing transformation pipeline.
    Zero stochastic augmentations; strictly standardizes resolution and channel distribution.
    """
    return T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def denormalize_tensor(
    tensor: torch.Tensor,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> torch.Tensor:
    """Inverts ImageNet normalization for visual display/inspection."""
    inv_mean = [-m / s for m, s in zip(mean, std)]
    inv_std = [1.0 / s for s in std]
    inv_norm = T.Normalize(mean=inv_mean, std=inv_std)
    
    if tensor.ndim == 4:
        return torch.stack([inv_norm(t) for t in tensor]).clamp(0, 1)
    return inv_norm(tensor).clamp(0, 1)
