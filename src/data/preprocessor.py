"""Image preprocessing pipeline for inference and evaluation."""
from pathlib import Path
from typing import Optional, Tuple, Union
import io
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T


class ImagePreprocessor:
    """Preprocesses raw skin lesion images for MobileNetV2."""

    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        self.image_size = image_size
        self.mean = mean
        self.std = std

        self.transform = T.Compose([
            T.Resize(self.image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(self.image_size),
            T.ToTensor(),
            T.Normalize(mean=self.mean, std=self.std),
        ])

    def preprocess_image(self, image: Union[str, Path, Image.Image, bytes]) -> Tuple[torch.Tensor, Image.Image]:
        """
        Takes an image input (file path, PIL Image, or raw bytes) and returns:
        1. Preprocessed PyTorch Tensor with shape [1, 3, H, W]
        2. RGB PIL Image (original resized to 224x224) for Grad-CAM overlay
        """
        if isinstance(image, (str, Path)):
            pil_img = Image.open(image).convert("RGB")
        elif isinstance(image, bytes):
            pil_img = Image.open(io.BytesIO(image)).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        # Resized PIL image for visualization overlay
        display_img = pil_img.resize(self.image_size, Image.Resampling.BILINEAR)

        # Tensor transformed for model input
        tensor_img = self.transform(pil_img).unsqueeze(0)  # Shape: [1, 3, 224, 224]

        return tensor_img, display_img

    def preprocess_pil(self, pil_img: Image.Image) -> torch.Tensor:
        """Preprocesses a PIL image directly to [3, H, W] tensor."""
        return self.transform(pil_img.convert("RGB"))
