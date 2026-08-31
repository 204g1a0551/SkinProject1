#!/usr/bin/env python3
"""
CLI Tool: Prepare Leakage-Free Dataset Splits
Partitions raw directory dataset into train, val, and test manifest CSVs.
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ConfigManager
from src.data.split import DatasetSplitter


def main():
    parser = argparse.ArgumentParser(description="Prepare train, validation, and test dataset splits.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/sample_dataset",
        help="Path to directory containing class subfolders (e.g. data/raw or data/sample_dataset)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Output directory where CSV manifests will be saved",
    )
    parser.add_argument(
        "--group-col",
        type=str,
        default=None,
        help="Optional column name for patient/lesion grouping to prevent leakage (e.g. patient_id)",
    )
    args = parser.parse_args()

    cfg = ConfigManager()
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    class_mapping = {c.code: idx for idx, c in enumerate(cfg.dataset_config.classes)}
    splitter = DatasetSplitter(
        train_ratio=cfg.dataset_config.train_ratio,
        val_ratio=cfg.dataset_config.val_ratio,
        test_ratio=cfg.dataset_config.test_ratio,
        random_seed=cfg.dataset_config.random_seed,
    )

    print(f"\n📂 Loading dataset from {data_dir}...")
    df, _ = splitter.discover_dataset(data_dir, class_mapping=class_mapping)

    print("\n🔀 Performing leakage-free stratified split...")
    train_df, val_df, test_df = splitter.split_dataframe(df, group_column=args.group_col)

    print(f"\n💾 Saving manifest files to {output_dir}...")
    manifests = splitter.save_split_manifests(train_df, val_df, test_df, output_dir=output_dir)

    print("\n✅ Manifests created successfully:")
    for split_name, path in manifests.items():
        print(f" - {split_name.capitalize():<5}: {path.name} ({len(pd.read_csv(path))} samples)")


if __name__ == "__main__":
    import pandas as pd
    main()
