#!/usr/bin/env python3
"""
CLI Tool: Explain Skin Lesion Predictions with Grad-CAM
Demonstrates:
- Automatic final conv layer discovery
- Target class selection (predicted or user-specified e.g. 'mel')
- Dimension alignment to original input image
- Exporting original, heatmap, and overlay images + clinical disclaimer
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.explainability.explainer import GradCAMExplainer
from src.utils.logger import get_logger

logger = get_logger("explain_cli")


def main():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM visual explanations for skin lesions.")
    parser.add_argument(
        "--image",
        type=str,
        default="data/sample_images/sample_lesion.jpg",
        help="Path to skin lesion image",
    )
    parser.add_argument(
        "--target-class",
        type=str,
        default=None,
        help="Target class code (e.g. 'mel', 'nv', 'bcc') or class index. Default: model's top predicted class.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/mobilenetv2_skin_disease_best.pth",
        help="Path to trained model weights",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/explainability",
        help="Directory to save explanation artifacts",
    )
    parser.add_argument(
        "--colormap",
        type=str,
        default="jet",
        help="Matplotlib colormap for heatmap (e.g. 'jet', 'viridis', 'magma')",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Overlay blending factor between 0.0 and 1.0",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device ('auto', 'mps', 'cpu')",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.is_absolute():
        image_path = PROJECT_ROOT / image_path

    if not image_path.exists():
        logger.error(f"Image not found at {image_path}")
        sys.exit(1)

    print(f"\n🔬 Initializing Grad-CAM Explainer on {args.device}...")
    explainer = GradCAMExplainer(
        weights_path=args.model_path if Path(args.model_path).exists() else None,
        device=args.device,
    )

    print(f"📷 Processing image: {image_path.name}...")
    explanation = explainer.explain_image(
        image_input=image_path,
        target_class=args.target_class,
        colormap_name=args.colormap,
        alpha=args.alpha,
    )

    prefix = image_path.stem
    if args.target_class:
        prefix = f"{prefix}_target_{args.target_class}"

    saved_paths = explainer.save_explanation_artifacts(
        explanation=explanation,
        output_dir=args.output_dir,
        prefix=prefix,
    )

    print("\n" + "=" * 65)
    print("      AI SKIN — GRAD-CAM EXPLAINABILITY REPORT")
    print("=" * 65)
    pred = explanation["predicted_class"]
    exp = explanation["explained_class"]
    print(f"Input Image         : {image_path.name} ({explanation['original_dimensions'][0]}x{explanation['original_dimensions'][1]} px)")
    print(f"Identified Layer    : {explanation['target_layer']}")
    print(f"Top Model Prediction: {pred['name']} (`{pred['code']}`) [Severity: {pred['severity']}]")
    print(f"Explained Class     : {exp['name']} (`{exp['code']}`) [Confidence: {exp['percentage']}%]")
    print("-" * 65)
    print("Exported Visual Artifacts:")
    print(f"  1. Original Image : {saved_paths['original']}")
    print(f"  2. Standalone Heat: {saved_paths['heatmap']}")
    print(f"  3. Grad-CAM Overlay: {saved_paths['overlay']}")
    print(f"  4. Metadata JSON  : {saved_paths['metadata']}")
    print("-" * 65)
    print(f"\n📢 {explanation['disclaimer']}\n")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
