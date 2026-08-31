#!/usr/bin/env python3
"""
CLI Tool: Comprehensive Model Evaluation & Performance Analysis
Evaluates MobileNetV2 on untouched test dataset, exports metrics, plots, and markdown report.
"""
import argparse
import json
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.visualizer import EvaluationVisualizer
from src.utils.logger import get_logger

logger = get_logger("evaluate_cli")


def generate_markdown_report(
    metrics: dict,
    output_dir: Path,
) -> Path:
    """Generate concise academic summary markdown report for project thesis."""
    report_file = output_dir / "evaluation_report.md"

    strongest = metrics.get("strongest_classes", [])
    weakest = metrics.get("weakest_classes", [])
    confusion_pairs = metrics.get("top_confusion_pairs", [])

    lines = [
        "# AI Skin: Intelligent Skin Diseases Detection — Model Evaluation Report",
        "",
        "## 1. Executive Summary",
        "",
        f"- **Model Architecture**: MobileNetV2 (Transfer Learning, 2-Stage Fine-Tuning)",
        f"- **Evaluation Dataset**: Untouched Test Partition ({metrics.get('total_samples')} samples)",
        f"- **Hardware Platform**: {metrics.get('evaluation_device', 'Apple Silicon MPS')}",
        "",
        "| Metric | Macro Average | Weighted Average |",
        "| :--- | :---: | :---: |",
        f"| **Accuracy** | **{metrics.get('accuracy') * 100:.2f}%** | **{metrics.get('accuracy') * 100:.2f}%** |",
        f"| **Precision** | {metrics.get('macro_precision') * 100:.2f}% | {metrics.get('weighted_precision') * 100:.2f}% |",
        f"| **Recall** | {metrics.get('macro_recall') * 100:.2f}% | {metrics.get('weighted_recall') * 100:.2f}% |",
        f"| **F1-Score** | {metrics.get('macro_f1') * 100:.2f}% | {metrics.get('weighted_f1') * 100:.2f}% |",
        "",
        "---",
        "",
        "## 2. Per-Class Diagnostic Performance",
        "",
        "| Class Code | Diagnostic Category | Precision | Recall (Sens.) | F1-Score | Test Samples |",
        "| :--- | :--- | :---: | :---: | :---: | :---: |",
    ]

    for c in metrics.get("per_class", []):
        lines.append(
            f"| `{c['class_code']}` | {c['class_name']} | "
            f"{c['precision']*100:.1f}% | {c['recall']*100:.1f}% | {c['f1_score']*100:.1f}% | {c['support']} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Diagnostic Strengths, Weaknesses & Error Analysis",
        "",
        "### 🏆 Strongest-Performing Classes",
    ])
    for s in strongest:
        lines.append(f"- **{s['class_name']} (`{s['class_code']}`)**: F1-Score: **{s['f1_score']*100:.1f}%** (Recall: {s['recall']*100:.1f}%, Support: {s['support']})")

    lines.append("\n### ⚠️ Weakest-Performing Classes")
    for w in weakest:
        lines.append(f"- **{w['class_name']} (`{w['class_code']}`)**: F1-Score: **{w['f1_score']*100:.1f}%** (Recall: {w['recall']*100:.1f}%, Support: {w['support']})")

    lines.append("\n### 🔄 Common Diagnostic Confusion Pairs")
    if confusion_pairs:
        for p in confusion_pairs:
            lines.append(f"- **{p['true_class']}** predicted as **{p['pred_class']}**: {p['count']} occurrences ({p['percentage_of_true']}% of true class)")
    else:
        lines.append("- No off-diagonal misclassifications observed.")

    lines.extend([
        "",
        "### ⚖️ Class Imbalance Effects & Clinical Insights",
        "- High-frequency categories (such as Melanocytic Nevi) naturally exhibit higher precision due to sample dominance.",
        "- Class-weighted cross-entropy loss applied during Phase 3 mitigates majority bias by penalizing misclassifications on critical minority classes (such as Melanoma).",
        "",
        "---",
        "",
        "## 4. Visual Diagnostic Artifacts",
        "",
        "- Raw Confusion Matrix: `confusion_matrix.png`",
        "- Normalized Confusion Matrix: `confusion_matrix_norm.png`",
        "- Learning Curves (Stage 1 & Stage 2): `learning_curves.png`",
        "- Per-Class F1-Score Comparison: `per_class_f1_bar.png`",
        "",
    ])

    with open(report_file, "w") as f:
        f.write("\n".join(lines))

    return report_file


def main():
    parser = argparse.ArgumentParser(description="Evaluate MobileNetV2 on skin disease test set.")
    parser.add_argument(
        "--test-manifest",
        type=str,
        default="data/processed/test_manifest.csv",
        help="Path to test_manifest.csv",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/mobilenetv2_skin_disease_best.pth",
        help="Path to model checkpoint .pth",
    )
    parser.add_argument(
        "--mapping-path",
        type=str,
        default="models/class_mapping.json",
        help="Path to class_mapping.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/evaluation",
        help="Directory to save evaluation reports and figures",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run evaluation on ('auto', 'mps', 'cpu')",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n🔍 Initializing Model Evaluation Engine...")
    evaluator = ModelEvaluator(
        model_path=args.model_path,
        mapping_path=args.mapping_path,
        device=args.device,
    )

    print(f"📊 Running deterministic evaluation on test set: {args.test_manifest}...")
    metrics = evaluator.evaluate_test_set(test_manifest_path=args.test_manifest)

    # 1. Save machine-readable metrics JSON
    metrics_json_path = out_dir / "test_metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"💾 Saved raw test metrics: {metrics_json_path}")

    # 2. Save Classification Report CSV
    per_class_df = pd.DataFrame(metrics["per_class"])
    csv_report_path = out_dir / "classification_report.csv"
    per_class_df.to_csv(csv_report_path, index=False)
    print(f"💾 Saved classification report CSV: {csv_report_path}")

    # 3. Generate Visualizations
    visualizer = EvaluationVisualizer(output_dir=out_dir)

    # A. Raw Confusion Matrix
    cm_path = visualizer.plot_confusion_matrix(
        cm=metrics["confusion_matrix"],
        class_names=evaluator.class_names,
        normalized=False,
    )
    print(f"📈 Saved confusion matrix plot: {cm_path}")

    # B. Normalized Confusion Matrix
    cmn_path = visualizer.plot_confusion_matrix(
        cm=metrics["confusion_matrix_norm"],
        class_names=evaluator.class_names,
        normalized=True,
    )
    print(f"📈 Saved normalized confusion matrix plot: {cmn_path}")

    # C. Learning Curves
    history_file = PROJECT_ROOT / "models" / "training_history.json"
    if history_file.exists():
        with open(history_file, "r") as f:
            history = json.load(f)
        lc_path = visualizer.plot_learning_curves(history)
        print(f"📈 Saved learning curves plot: {lc_path}")

    # D. Per-Class F1-Score Bar Chart
    f1_bar_path = visualizer.plot_per_class_metrics(metrics["per_class"])
    print(f"📈 Saved per-class F1 chart: {f1_bar_path}")

    # 4. Generate Academic Markdown Summary
    report_md_path = generate_markdown_report(metrics, output_dir=out_dir)
    print(f"📝 Generated academic evaluation report: {report_md_path}")

    # Print Terminal Summary
    print("\n" + "=" * 65)
    print("        AI SKIN — MODEL EVALUATION SUMMARY (TEST SET)")
    print("=" * 65)
    print(f"Test Accuracy    : {metrics['accuracy'] * 100:.2f}%")
    print(f"Macro F1-Score   : {metrics['macro_f1'] * 100:.2f}%")
    print(f"Weighted F1-Score: {metrics['weighted_f1'] * 100:.2f}%")
    print("-" * 65)
    print(f"{'Class':<22} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'N':<4}")
    print("-" * 65)
    for c in metrics["per_class"]:
        print(f"{c['class_name']:<22} | {c['precision']*100:>6.1f}%    | {c['recall']*100:>6.1f}%    | {c['f1_score']*100:>6.1f}%    | {c['support']:<4}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
