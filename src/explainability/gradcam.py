"""
Grad-CAM: Gradient-Weighted Class Activation Mapping
(Selvaraju et al., 2017)
Tailored for MobileNetV2 skin lesion visual interpretability.
Features:
- Dynamic discovery of the final Conv2d layer (eliminating hardcoded indices).
- Mathematical gradient pooling and ReLU activation mapping.
- Arbitrary target-class selection (default: top predicted class).
- Resizing to match original image dimensions.
"""
from typing import Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from ..utils.logger import get_logger

logger = get_logger(__name__)


def find_last_conv_layer(model: nn.Module) -> Tuple[str, nn.Module]:
    """
    Recursively inspects model modules in reverse to automatically discover
    the final 2D convolutional layer.
    """
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, nn.Conv2d):
            logger.info(f"Dynamically identified final Conv2d layer: '{name}'")
            return name, module

    raise ValueError("No 2D convolutional layer found in model architecture.")


class GradCAM:
    """Computes Grad-CAM heatmaps for a target convolutional layer."""

    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[Union[nn.Module, str]] = None,
    ):
        self.model = model

        # Dynamically discover target layer if not explicitly provided
        if target_layer is None:
            self.target_layer_name, self.target_layer = find_last_conv_layer(model)
        elif isinstance(target_layer, str):
            self.target_layer_name = target_layer
            # Retrieve module by name
            module_dict = dict(model.named_modules())
            if target_layer not in module_dict:
                raise ValueError(f"Target layer name '{target_layer}' not found in model.")
            self.target_layer = module_dict[target_layer]
        else:
            self.target_layer_name = target_layer.__class__.__name__
            self.target_layer = target_layer

        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks."""
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class_idx: Optional[int] = None,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, int, float]:
        """
        Generate a 2D normalized Grad-CAM heatmap [0, 1].

        Args:
            input_tensor: Normalized image tensor of shape [1, 3, H, W]
            target_class_idx: Class index to compute CAM for. If None, highest predicted class is used.
            target_size: Optional (width, height) to resize heatmap to match original image dimensions.

        Returns:
            heatmap: 2D numpy array [H, W] normalized to [0, 1]
            target_class_idx: Class index used for Grad-CAM
            confidence: Probability score for the target class
        """
        # Ensure model is in eval mode so BatchNorm uses running stats with single sample
        self.model.eval()

        # Ensure parameters in target layer require grad
        for param in self.target_layer.parameters():
            param.requires_grad = True

        self.model.zero_grad()

        # Run forward pass with grad enabled
        with torch.enable_grad():
            tensor = input_tensor.clone().detach().requires_grad_(True)
            output = self.model(tensor)
            probabilities = F.softmax(output, dim=1)

            top_pred_idx = int(torch.argmax(probabilities, dim=1).item())

            if target_class_idx is None:
                target_class_idx = top_pred_idx

            if target_class_idx < 0 or target_class_idx >= output.shape[1]:
                raise ValueError(
                    f"Target class index {target_class_idx} out of range (model has {output.shape[1]} classes)."
                )

            score = output[0, target_class_idx]
            confidence = float(probabilities[0, target_class_idx].item())

            # Backward pass to calculate gradients w.r.t target layer activations
            score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Hooks did not capture gradients or activations.")

        # Global Average Pooling of gradients: [1, C, H, W] -> [1, C, 1, 1]
        pooled_gradients = torch.mean(self.gradients, dim=[2, 3], keepdim=True)

        # Weighted combination of forward activation maps
        cam = torch.sum(pooled_gradients * self.activations, dim=1, keepdim=True)  # [1, 1, H, W]

        # Apply ReLU to capture features with positive contribution to target class
        cam = F.relu(cam)

        # Interpolate resolution
        if target_size is not None:
            out_h, out_w = target_size[1], target_size[0]
        else:
            _, _, out_h, out_w = input_tensor.shape

        cam = F.interpolate(cam, size=(out_h, out_w), mode="bilinear", align_corners=False)

        # Normalize between 0 and 1
        cam_np = cam.squeeze().detach().cpu().numpy()
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max - cam_min > 1e-8:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np, target_class_idx, confidence


def apply_colormap_on_image(
    org_img: Image.Image,
    activation_map: np.ndarray,
    colormap_name: str = "jet",
    alpha: float = 0.45,
) -> Image.Image:
    """
    Overlay Grad-CAM heatmap on the original PIL image.
    Preserves original image dimensions.
    """
    w, h = org_img.size
    if activation_map.shape != (h, w):
        cam_img = Image.fromarray((activation_map * 255).astype(np.uint8))
        cam_img = cam_img.resize((w, h), Image.Resampling.BILINEAR)
        activation_map = np.array(cam_img) / 255.0

    cmap = plt.get_cmap(colormap_name)
    colored_cam = cmap(activation_map)[:, :, :3]  # [H, W, 3] in [0, 1]
    colored_cam = (colored_cam * 255).astype(np.uint8)

    org_arr = np.array(org_img.convert("RGB")).astype(float)
    blended = (1.0 - alpha) * org_arr + alpha * colored_cam.astype(float)
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return Image.fromarray(blended)


def create_standalone_heatmap_image(
    activation_map: np.ndarray,
    colormap_name: str = "jet",
    target_size: Optional[Tuple[int, int]] = None,
) -> Image.Image:
    """Generates a standalone colored RGB heatmap image without the original image."""
    cmap = plt.get_cmap(colormap_name)
    colored = cmap(activation_map)[:, :, :3]
    colored = (colored * 255).astype(np.uint8)
    heatmap_pil = Image.fromarray(colored)

    if target_size is not None and heatmap_pil.size != target_size:
        heatmap_pil = heatmap_pil.resize(target_size, Image.Resampling.BILINEAR)

    return heatmap_pil
