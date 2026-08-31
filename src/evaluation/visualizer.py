"""
High-resolution, publication-grade visualization tools for model evaluation.
Generates:
1. Raw confusion matrix heatmap
2. Row-normalized confusion matrix heatmap
3. Training vs Validation loss & accuracy learning curves
4. Per-class F1-score comparison bar chart
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class EvaluationVisualizer:
    """Generates and saves diagnostic evaluation charts."""

    def __init__(self, output_dir: Union[str, Path] = "reports/evaluation"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="white", font="sans-serif")

    def plot_confusion_matrix(
        self,
        cm: Union[List[List[Union[int, float]]], np.ndarray],
        class_names: List[str],
        normalized: bool = False,
        filename: Optional[str] = None,
    ) -> Path:
        """Plot and save confusion matrix heatmap."""
        cm_arr = np.array(cm)
        fname = filename or ("confusion_matrix_norm.png" if normalized else "confusion_matrix.png")
        out_path = self.output_dir / fname

        plt.figure(figsize=(9, 7.5))
        fmt = ".2f" if normalized else "d"
        cmap = "Blues" if not normalized else "YlGnBu"

        ax = sns.heatmap(
            cm_arr,
            annot=True,
            fmt=fmt,
            cmap=cmap,
            xticklabels=class_names,
            yticklabels=class_names,
            cbar=True,
            linewidths=0.5,
            linecolor="lightgray",
        )

        title = "Normalized Confusion Matrix (Recall)" if normalized else "Confusion Matrix (Sample Counts)"
        plt.title(f"AI Skin: {title}", fontsize=13, fontweight="bold", pad=12)
        plt.xlabel("Predicted Diagnostic Class", fontsize=11, fontweight="semibold", labelpad=8)
        plt.ylabel("True Diagnostic Class", fontsize=11, fontweight="semibold", labelpad=8)
        plt.xticks(rotation=35, ha="right", fontsize=9)
        plt.yticks(rotation=0, fontsize=9)

        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        return out_path

    def plot_learning_curves(
        self,
        history: Dict[str, List[Any]],
        filename: str = "learning_curves.png",
    ) -> Path:
        """Plot train vs validation loss and accuracy curves across stages."""
        out_path = self.output_dir / filename

        epochs = history.get("epoch", [])
        if not epochs:
            return out_path

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

        # Find boundary between Stage 1 and Stage 2
        stages = history.get("stage", [])
        stage2_start = None
        for i, s in enumerate(stages):
            if s == "stage2":
                stage2_start = epochs[i]
                break

        # Panel 1: Loss
        ax1.plot(epochs, history.get("train_loss", []), marker="o", color="#0d9488", label="Train Loss", linewidth=2)
        ax1.plot(epochs, history.get("val_loss", []), marker="s", color="#e11d48", label="Validation Loss", linewidth=2)
        ax1.set_title("Training vs Validation Loss", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Epoch", fontsize=10, fontweight="semibold")
        ax1.set_ylabel("Cross-Entropy Loss", fontsize=10, fontweight="semibold")
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend(frameon=True)

        if stage2_start is not None:
            ax1.axvline(x=stage2_start - 0.5, color="gray", linestyle=":", label="Stage 2 (Fine-Tuning)")
            ax1.text(stage2_start - 0.3, max(history.get("train_loss", [1])) * 0.9, "Fine-Tuning Unfrozen", color="gray", fontsize=8)

        # Panel 2: Accuracy
        ax2.plot(epochs, history.get("train_acc", []), marker="o", color="#0d9488", label="Train Acc (%)", linewidth=2)
        ax2.plot(epochs, history.get("val_acc", []), marker="s", color="#e11d48", label="Validation Acc (%)", linewidth=2)
        ax2.set_title("Training vs Validation Accuracy", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Epoch", fontsize=10, fontweight="semibold")
        ax2.set_ylabel("Accuracy (%)", fontsize=10, fontweight="semibold")
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend(frameon=True)

        if stage2_start is not None:
            ax2.axvline(x=stage2_start - 0.5, color="gray", linestyle=":")

        plt.suptitle("AI Skin: MobileNetV2 Two-Stage Transfer Learning Convergence", fontsize=13, fontweight="bold", y=0.98)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        return out_path

    def plot_per_class_metrics(
        self,
        per_class_data: List[Dict[str, Any]],
        filename: str = "per_class_f1_bar.png",
    ) -> Path:
        """Plot horizontal bar chart of per-class F1-scores."""
        out_path = self.output_dir / filename

        names = [item["class_name"] for item in per_class_data]
        f1_scores = [item["f1_score"] * 100 for item in per_class_data]
        supports = [item["support"] for item in per_class_data]

        plt.figure(figsize=(10, 5.5))
        y_pos = np.arange(len(names))

        # Color based on score
        colors = ["#0d9488" if s >= 50 else "#f59e0b" if s >= 20 else "#e11d48" for s in f1_scores]

        bars = plt.barh(y_pos, f1_scores, color=colors, edgecolor="black", linewidth=0.5)
        plt.yticks(y_pos, names, fontsize=10)
        plt.xlabel("F1-Score (%)", fontsize=11, fontweight="semibold")
        plt.title("AI Skin: Per-Class F1-Score Performance on Test Partition", fontsize=12, fontweight="bold", pad=12)
        plt.xlim(0, 105)
        plt.grid(axis="x", linestyle="--", alpha=0.5)

        # Annotate bars
        for bar, supp, f1 in zip(bars, supports, f1_scores):
            width = bar.get_width()
            plt.text(
                width + 1.5,
                bar.get_y() + bar.get_height() / 2,
                f"{f1:.1f}% (N={supp})",
                va="center",
                fontsize=9,
                color="#1e293b",
                fontweight="medium",
            )

        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        return out_path
