"""
Production-grade PyTorch Dataset for skin disease lesions.
Supports:
- Loading from DataFrame or CSV manifest file
- Loading from raw directory structure
- Safe image loading with corruption recovery
- Seamless augmentation integration
"""
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

from .validator import ImageValidator
from ..utils.logger import get_logger

logger = get_logger(__name__)


class SkinLesionDataset(Dataset):
    """
    PyTorch Dataset for multi-class dermatoscopic skin lesion classification.
    """

    def __init__(
        self,
        data: Union[pd.DataFrame, str, Path],
        transform: Optional[Callable] = None,
        class_mapping: Optional[Dict[str, int]] = None,
        return_path: bool = False,
    ):
        """
        Args:
            data: DataFrame containing 'image_path' and 'class_idx', or Path to CSV manifest file.
            transform: torchvision transformation pipeline
            class_mapping: Mapping of class name to integer index
            return_path: Whether __getitem__ returns (image, label, path)
        """
        if isinstance(data, (str, Path)):
            p = Path(data)
            if not p.exists():
                raise FileNotFoundError(f"Manifest file not found: {p}")
            self.df = pd.read_csv(p)
        elif isinstance(data, pd.DataFrame):
            self.df = data.copy().reset_index(drop=True)
        else:
            raise TypeError(f"Expected DataFrame or Path/str, got {type(data)}")

        if "image_path" not in self.df.columns or "class_idx" not in self.df.columns:
            raise ValueError("Dataset data must contain 'image_path' and 'class_idx' columns.")

        self.transform = transform
        self.class_mapping = class_mapping
        self.return_path = return_path
        self.validator = ImageValidator()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, int], Tuple[torch.Tensor, int, str]]:
        row = self.df.iloc[idx]
        image_path = row["image_path"]
        label = int(row["class_idx"])

        # Load image safely
        try:
            pil_img = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Error opening image {image_path}: {e}. Generating fallback neutral image.")
            pil_img = Image.new("RGB", (224, 224), color=(128, 128, 128))

        if self.transform is not None:
            tensor_img = self.transform(pil_img)
        else:
            # Fallback default transform
            import torchvision.transforms as T
            tensor_img = T.ToTensor()(pil_img)

        if self.return_path:
            return tensor_img, label, str(image_path)
        return tensor_img, label

    def get_labels(self) -> List[int]:
        """Return all labels as a list (useful for sampler or metrics)."""
        return self.df["class_idx"].tolist()

    def get_class_counts(self) -> Dict[int, int]:
        """Return label counts dictionary."""
        return self.df["class_idx"].value_counts().to_dict()
