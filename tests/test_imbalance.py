"""Test class imbalance detection and weights calculation."""
from src.data.imbalance import ImbalanceAnalyzer


def test_imbalance_analyzer_balanced():
    analyzer = ImbalanceAnalyzer()
    labels = [0] * 50 + [1] * 50 + [2] * 50
    analysis = analyzer.analyze_distribution(labels)

    assert analysis["total_samples"] == 150
    assert analysis["num_classes"] == 3
    assert analysis["imbalance_ratio"] == 1.0
    assert analysis["severity"] == "Balanced"


def test_imbalance_analyzer_severe():
    analyzer = ImbalanceAnalyzer()
    # 100 of class 0, 5 of class 1 -> IR = 20.0
    labels = [0] * 100 + [1] * 5
    analysis = analyzer.analyze_distribution(labels)

    assert analysis["total_samples"] == 105
    assert analysis["imbalance_ratio"] == 20.0
    assert analysis["severity"] == "Severe"
    # Minority class should have a much higher balanced weight
    weights = analysis["balanced_weights"]
    assert weights[1] > weights[0]
