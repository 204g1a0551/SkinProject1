"""
Class distribution analysis and severe class imbalance detection.
Computes imbalance ratio and balanced weights for loss functions.
"""
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd

from ..utils.logger import get_logger

logger = get_logger(__name__)


class ImbalanceAnalyzer:
    """Analyzes class distributions and detects severity of imbalance."""

    def __init__(
        self,
        severe_threshold: float = 10.0,
        moderate_threshold: float = 3.0,
    ):
        self.severe_threshold = severe_threshold
        self.moderate_threshold = moderate_threshold

    def analyze_distribution(
        self,
        labels: List[Union[int, str]],
        class_names: Optional[Dict[Any, str]] = None,
    ) -> Dict[str, Any]:
        """
        Compute frequency, percentages, imbalance ratio, Shannon entropy,
        and severity classification.
        """
        if not labels:
            return {"error": "Empty labels list provided."}

        series = pd.Series(labels)
        counts = series.value_counts().to_dict()
        total_samples = len(labels)
        num_classes = len(counts)

        sorted_classes = sorted(counts.keys())
        distribution = []
        for cls in sorted_classes:
            cnt = counts[cls]
            name = class_names.get(cls, str(cls)) if class_names else str(cls)
            pct = (cnt / total_samples) * 100
            distribution.append({
                "class_id": cls,
                "class_name": name,
                "count": cnt,
                "percentage": round(pct, 2),
            })

        counts_list = list(counts.values())
        max_count = max(counts_list)
        min_count = min(counts_list)
        imbalance_ratio = float(max_count / max(min_count, 1))

        # Classify severity
        if imbalance_ratio >= self.severe_threshold:
            severity = "Severe"
            recommendation = (
                "Severe class imbalance detected (IR >= 10.0). "
                "Recommendation: Apply Class-Weighted Cross-Entropy Loss, Focal Loss, "
                "or minority class oversampling (e.g. weighted random sampling)."
            )
        elif imbalance_ratio >= self.moderate_threshold:
            severity = "Moderate"
            recommendation = (
                "Moderate class imbalance detected. "
                "Recommendation: Use weighted loss or stratified batch sampling."
            )
        else:
            severity = "Balanced"
            recommendation = "Class distribution is relatively balanced."

        # Shannon Entropy (diversity metric): 1.0 = perfectly balanced
        proportions = np.array(counts_list) / total_samples
        entropy = -np.sum(proportions * np.log(proportions + 1e-12))
        max_entropy = np.log(num_classes) if num_classes > 1 else 1.0
        normalized_entropy = float(entropy / max_entropy) if max_entropy > 0 else 1.0

        # Balanced class weights: N / (num_classes * count_c)
        balanced_weights = {
            cls: round(float(total_samples / (num_classes * count)), 4)
            for cls, count in counts.items()
        }

        return {
            "total_samples": total_samples,
            "num_classes": num_classes,
            "imbalance_ratio": round(imbalance_ratio, 2),
            "severity": severity,
            "normalized_entropy": round(normalized_entropy, 3),
            "distribution": distribution,
            "balanced_weights": balanced_weights,
            "recommendation": recommendation,
        }

    def print_summary(self, analysis: Dict[str, Any]):
        """Pretty-print distribution analysis to stdout."""
        print("=" * 60)
        print("           DATASET CLASS DISTRIBUTION REPORT")
        print("=" * 60)
        print(f"Total Samples    : {analysis.get('total_samples')}")
        print(f"Number of Classes: {analysis.get('num_classes')}")
        print(f"Imbalance Ratio  : {analysis.get('imbalance_ratio')}:1")
        print(f"Imbalance Status : {analysis.get('severity')}")
        print(f"Shannon Diversity: {analysis.get('normalized_entropy')} / 1.000")
        print("-" * 60)
        print(f"{'Class':<20} | {'Count':<8} | {'Percentage':<10} | {'Weight':<8}")
        print("-" * 60)
        weights = analysis.get("balanced_weights", {})
        for item in analysis.get("distribution", []):
            cid = item["class_id"]
            w = weights.get(cid, 1.0)
            print(f"{item['class_name']:<20} | {item['count']:<8} | {item['percentage']:>6.2f}%    | {w:<8.4f}")
        print("-" * 60)
        print(f"Recommendation: {analysis.get('recommendation')}")
        print("=" * 60)
