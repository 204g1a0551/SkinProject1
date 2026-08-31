# AI Skin: Intelligent Skin Diseases Detection — Model Evaluation Report

## 1. Executive Summary

- **Model Architecture**: MobileNetV2 (Transfer Learning, 2-Stage Fine-Tuning)
- **Evaluation Dataset**: Untouched Test Partition (13 samples)
- **Hardware Platform**: mps

| Metric | Macro Average | Weighted Average |
| :--- | :---: | :---: |
| **Accuracy** | **38.46%** | **38.46%** |
| **Precision** | 23.61% | 50.64% |
| **Recall** | 23.81% | 38.46% |
| **F1-Score** | 22.42% | 41.68% |

---

## 2. Per-Class Diagnostic Performance

| Class Code | Diagnostic Category | Precision | Recall (Sens.) | F1-Score | Test Samples |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `mel` | Melanoma | 0.0% | 0.0% | 0.0% | 2 |
| `nv` | Melanocytic Nevi | 75.0% | 42.9% | 54.5% | 7 |
| `bcc` | Basal Cell Carcinoma | 66.7% | 100.0% | 80.0% | 2 |
| `akiec` | Actinic Keratoses / Intraepithelial Carcinoma | 0.0% | 0.0% | 0.0% | 0 |
| `bkl` | Benign Keratosis-like Lesions | 0.0% | 0.0% | 0.0% | 2 |
| `df` | Dermatofibroma | 0.0% | 0.0% | 0.0% | 0 |
| `vasc` | Vascular Lesions | 0.0% | 0.0% | 0.0% | 0 |

---

## 3. Diagnostic Strengths, Weaknesses & Error Analysis

### 🏆 Strongest-Performing Classes
- **Basal Cell Carcinoma (`bcc`)**: F1-Score: **80.0%** (Recall: 100.0%, Support: 2)
- **Melanocytic Nevi (`nv`)**: F1-Score: **54.5%** (Recall: 42.9%, Support: 7)

### ⚠️ Weakest-Performing Classes
- **Benign Keratosis-like Lesions (`bkl`)**: F1-Score: **0.0%** (Recall: 0.0%, Support: 2)
- **Melanoma (`mel`)**: F1-Score: **0.0%** (Recall: 0.0%, Support: 2)

### 🔄 Common Diagnostic Confusion Pairs
- **Melanocytic Nevi** predicted as **Actinic Keratoses / Intraepithelial Carcinoma**: 2 occurrences (28.57% of true class)
- **Melanocytic Nevi** predicted as **Benign Keratosis-like Lesions**: 2 occurrences (28.57% of true class)
- **Melanoma** predicted as **Basal Cell Carcinoma**: 1 occurrences (50.0% of true class)
- **Melanoma** predicted as **Actinic Keratoses / Intraepithelial Carcinoma**: 1 occurrences (50.0% of true class)
- **Benign Keratosis-like Lesions** predicted as **Melanocytic Nevi**: 1 occurrences (50.0% of true class)

### ⚖️ Class Imbalance Effects & Clinical Insights
- High-frequency categories (such as Melanocytic Nevi) naturally exhibit higher precision due to sample dominance.
- Class-weighted cross-entropy loss applied during Phase 3 mitigates majority bias by penalizing misclassifications on critical minority classes (such as Melanoma).

---

## 4. Visual Diagnostic Artifacts

- Raw Confusion Matrix: `confusion_matrix.png`
- Normalized Confusion Matrix: `confusion_matrix_norm.png`
- Learning Curves (Stage 1 & Stage 2): `learning_curves.png`
- Per-Class F1-Score Comparison: `per_class_f1_bar.png`
