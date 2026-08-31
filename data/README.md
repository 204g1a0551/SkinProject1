# AI Skin: Dataset Setup & Preparation Guide

This guide explains how to prepare and organize dermatoscopic skin disease datasets (such as **HAM10000** or **ISIC 2018/2019/2020**) for the AI Skin classification pipeline.

---

## 📁 Supported Directory Structure

Place raw dermatoscopic images into `data/raw/` organized by class code:

```
data/raw/
├── mel/                  # Melanoma
│   ├── ISIC_0024310.jpg
│   └── ...
├── nv/                   # Melanocytic Nevi
│   ├── ISIC_0024311.jpg
│   └── ...
├── bcc/                  # Basal Cell Carcinoma
├── akiec/                # Actinic Keratoses / Intraepithelial Carcinoma
├── bkl/                  # Benign Keratosis-like Lesions
├── df/                   # Dermatofibroma
└── vasc/                 # Vascular Lesions
```

*(Optional)* You can also place the official metadata CSV (e.g. `HAM10000_metadata.csv`) into `data/raw/metadata.csv` if you want patient-level grouping (`lesion_id` or `patient_id`) to completely eliminate patient leakage across splits.

---

## 🌐 Recommended Dataset Sources

1. **HAM10000 Dataset (Harvard Dataverse / Kaggle)**:
   - Contains 10,015 dermatoscopic images across the exact 7 diagnostic categories used in this project.
   - Harvard Dataverse: [https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)
   - Kaggle: `kmader/skin-cancer-mnist-ham10000`

2. **ISIC Archive**:
   - International Skin Imaging Collaboration: [https://www.isic-archive.com/](https://www.isic-archive.com/)

---

## ⚡ How to Inspect & Process Your Dataset

Once images are placed in `data/raw/`:

### 1. Inspect Dataset & Check for Corrupted Files
```bash
python scripts/inspect_dataset.py --data-dir data/raw
```
This scans all image files, checks PIL headers, reports corrupted/unreadable files, and computes the class imbalance ratio.

### 2. Generate Leakage-Free Splits (Train / Val / Test)
```bash
python scripts/prepare_splits.py --data-dir data/raw --output-dir data/processed
```
This performs a stratified split (70% train, 15% val, 15% test) and generates reproducible manifest files:
- `data/processed/train_manifest.csv`
- `data/processed/val_manifest.csv`
- `data/processed/test_manifest.csv`

### 3. Generate Visual Charts
```bash
python scripts/visualize_distribution.py --manifest-dir data/processed
python scripts/visualize_augmentations.py --image-path data/sample_images/sample_lesion.jpg
```
The figures will be saved in `reports/figures/`.
