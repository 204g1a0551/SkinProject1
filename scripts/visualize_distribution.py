#!/usr/bin/env python3
"""
CLI Tool: Generate Publication-Grade Dataset Distribution Charts
Visualizes class counts and split partitions, saving figure to reports/figures/.
"""
import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ConfigManager


def main():
    parser = argparse.ArgumentParser(description="Visualize class distribution of dataset splits.")
    parser.add_argument(
        "--manifest-dir",
        type=str,
        default="data/processed",
        help="Path to directory containing train_manifest.csv, val_manifest.csv, test_manifest.csv",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="reports/figures/class_distribution.png",
        help="Path to output PNG image file",
    )
    args = parser.parse_args()

    cfg = ConfigManager()
    manifest_dir = Path(args.manifest_dir)
    if not manifest_dir.is_absolute():
        manifest_dir = PROJECT_ROOT / manifest_dir

    output_path = Path(args.output_file)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    train_p = manifest_dir / "train_manifest.csv"
    val_p = manifest_dir / "val_manifest.csv"
    test_p = manifest_dir / "test_manifest.csv"

    if not (train_p.exists() and val_p.exists() and test_p.exists()):
        print("❌ Error: Split manifests not found. Please run `python scripts/prepare_splits.py` first.")
        sys.exit(1)

    train_df = pd.read_csv(train_p)
    train_df["Split"] = "Train (70%)"
    val_df = pd.read_csv(val_p)
    val_df["Split"] = "Val (15%)"
    test_df = pd.read_csv(test_p)
    test_df["Split"] = "Test (15%)"

    combined = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # Class name lookup
    class_map = {c.code: c.name for c in cfg.dataset_config.classes}
    combined["Disease"] = combined["class_name"].map(lambda c: class_map.get(c, c))

    # Set aesthetics
    sns.set_theme(style="whitegrid", font="sans-serif")
    fig, ax = plt.subplots(figsize=(12, 6))

    palette = {"Train (70%)": "#0d9488", "Val (15%)": "#f59e0b", "Test (15%)": "#6366f1"}

    order = combined["Disease"].value_counts().index

    sns.countplot(
        data=combined,
        x="Disease",
        hue="Split",
        order=order,
        palette=palette,
        ax=ax,
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_title("AI Skin: Multiclass Skin Lesion Distribution Across Splits", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Diagnostic Category", fontsize=11, fontweight="semibold", labelpad=10)
    ax.set_ylabel("Number of Dermatoscopic Images", fontsize=11, fontweight="semibold")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(title="Dataset Partition", frameon=True, facecolor="white", edgecolor="none")

    # Add count labels on top of bars
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(
                f"{int(height)}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=8,
                xytext=(0, 2),
                textcoords="offset points",
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"✅ Class distribution figure saved to: {output_path}")


if __name__ == "__main__":
    main()
