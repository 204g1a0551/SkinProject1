"""Comprehensive evaluation metrics for multi-class skin disease classification."""
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


class MetricsCalculator:
    """Calculates classification metrics, confusion matrices, and diagnostic insights."""

    def __init__(
        self,
        class_names: Optional[List[str]] = None,
        class_codes: Optional[List[str]] = None,
    ):
        self.class_names = class_names
        self.class_codes = class_codes or class_names

    def compute(
        self,
        y_true: List[int],
        y_pred: List[int],
        average: str = "weighted",
    ) -> Dict[str, Any]:
        """
        Compute standard multi-class metrics:
        - Accuracy
        - Precision, Recall, F1 (macro, weighted)
        - Raw confusion matrix & Normalized confusion matrix
        - Per-class classification report
        - Diagnostic insights: strongest/weakest classes & top confusion pairs
        """
        y_t = np.array(y_true)
        y_p = np.array(y_pred)

        num_classes = len(self.class_names) if self.class_names else len(np.unique(np.concatenate([y_t, y_p])))
        labels_range = list(range(num_classes))

        acc = float(accuracy_score(y_t, y_p))
        prec = float(precision_score(y_t, y_p, average=average, zero_division=0))
        rec = float(recall_score(y_t, y_p, average=average, zero_division=0))
        f1 = float(f1_score(y_t, y_p, average=average, zero_division=0))

        macro_prec = float(precision_score(y_t, y_p, average="macro", zero_division=0))
        macro_rec = float(recall_score(y_t, y_p, average="macro", zero_division=0))
        macro_f1 = float(f1_score(y_t, y_p, average="macro", zero_division=0))

        # Confusion Matrices
        cm_raw = confusion_matrix(y_t, y_p, labels=labels_range)
        with np.errstate(divide="ignore", invalid="ignore"):
            row_sums = cm_raw.sum(axis=1, keepdims=True)
            cm_norm = np.where(row_sums > 0, cm_raw / row_sums, 0.0)

        # Classification report dict
        target_names = self.class_names if (self.class_names and len(self.class_names) == num_classes) else None
        report = classification_report(
            y_t,
            y_p,
            labels=labels_range,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        )

        # Extract per-class metrics
        per_class: List[Dict[str, Any]] = []
        for idx in labels_range:
            name = self.class_names[idx] if (self.class_names and idx < len(self.class_names)) else f"Class {idx}"
            code = self.class_codes[idx] if (self.class_codes and idx < len(self.class_codes)) else str(idx)
            stats = report.get(name, report.get(str(idx), {}))
            per_class.append({
                "class_idx": idx,
                "class_code": code,
                "class_name": name,
                "precision": round(float(stats.get("precision", 0.0)), 4),
                "recall": round(float(stats.get("recall", 0.0)), 4),
                "f1_score": round(float(stats.get("f1-score", 0.0)), 4),
                "support": int(stats.get("support", 0)),
            })

        # Identify Strongest & Weakest Performing Classes (sorted by F1-score)
        valid_per_class = [c for c in per_class if c["support"] > 0]
        sorted_by_f1 = sorted(valid_per_class, key=lambda x: x["f1_score"], reverse=True)
        strongest = sorted_by_f1[:2] if len(sorted_by_f1) >= 2 else sorted_by_f1
        weakest = sorted_by_f1[-2:][::-1] if len(sorted_by_f1) >= 2 else sorted_by_f1

        # Identify Top Confusion Pairs (Off-diagonal elements)
        confusion_pairs = []
        for i in range(num_classes):
            for j in range(num_classes):
                if i != j and cm_raw[i, j] > 0:
                    c_true_name = self.class_names[i] if self.class_names else f"Class {i}"
                    c_pred_name = self.class_names[j] if self.class_names else f"Class {j}"
                    confusion_pairs.append({
                        "true_class": c_true_name,
                        "pred_class": c_pred_name,
                        "count": int(cm_raw[i, j]),
                        "percentage_of_true": round(float(cm_norm[i, j] * 100), 2),
                    })
        confusion_pairs = sorted(confusion_pairs, key=lambda x: x["count"], reverse=True)

        return {
            "total_samples": len(y_true),
            "num_classes": num_classes,
            "accuracy": round(acc, 4),
            "weighted_precision": round(prec, 4),
            "weighted_recall": round(rec, 4),
            "weighted_f1": round(f1, 4),
            "macro_precision": round(macro_prec, 4),
            "macro_recall": round(macro_rec, 4),
            "macro_f1": round(macro_f1, 4),
            "confusion_matrix": cm_raw.tolist(),
            "confusion_matrix_norm": [[round(float(val), 4) for val in row] for row in cm_norm],
            "per_class": per_class,
            "strongest_classes": strongest,
            "weakest_classes": weakest,
            "top_confusion_pairs": confusion_pairs[:5],
            "classification_report": report,
        }
