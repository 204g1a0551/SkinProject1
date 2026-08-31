#!/usr/bin/env python3
"""
CLI Tool: Visualize Medical-Grade Skin Lesion Augmentations
Generates side-by-side comparison of original vs 5 stochastic augmentations.
"""
import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ConfigManager
from src.data.augmentation import get_training_transforms, denormalize_tensor


def main():
    parser = argparse.ArgumentParser(description="Visualize dermatoscopic data augmentations.")
    parser.add_argument(
        "--image-path",
        type=str,
        default="data/sample_images/sample_lesion.jpg",
        help="Path to sample lesion image",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of augmented variations to generate",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="reports/figures/augmentation_samples.png",
        help="Path to output PNG image file",
    )
    args = parser.parse_args()

    cfg = ConfigManager()
    img_path = Path(args.image_path)
    if not img_path.is_absolute():
        img_path = PROJECT_ROOT / img_path

    out_path = Path(args.output_file)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not img_path.exists():
        print(f"❌ Error: Image not found at {img_path}")
        sys.exit(1)

    pil_img = Image.open(img_path).convert("RGB")

    # Get training transforms with medical bounds
    train_transform = get_training_transforms(
        image_size=cfg.dataset_config.image_size,
        mean=cfg.dataset_config.mean,
        std=cfg.dataset_config.std,
    )

    fig, axes = plt.subplots(1, args.num_samples + 1, figsize=(16, 3.5))

    # 1. Plot Original
    axes[0].imshow(pil_img.resize((224, 224)))
    axes[0].set_title("Original Lesion\n(Reference 224x224)", fontsize=10, fontweight="bold", color="#0f766e")
    axes[0].axis("off")

    # 2. Plot Augmented Variations
    for i in range(1, args.num_samples + 1):
        tensor = train_transform(pil_img)
        denorm = denormalize_tensor(tensor, mean=cfg.dataset_config.mean, std=cfg.dataset_config.std)
        # Tensor [3, H, W] -> Numpy [H, W, 3]
        np_img = denorm.permute(1, 2, 0).numpy()

        axes[i].imshow(np_img)
        axes[i].set_title(f"Augmented #{i}\n(Flip/Crop/Rot/Jitter)", fontsize=9)
        axes[i].axis("off")

    plt.suptitle("AI Skin: Clinically Bounded Dermatoscopic Augmentations (MobileNetV2)", fontsize=12, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"✅ Augmentation comparison grid saved to: {out_path}")


if __name__ == "__main__":
    main()
