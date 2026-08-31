#!/usr/bin/env python3
"""
CLI Tool: Inspect Skin Lesion Dataset
Scans directories, validates image integrity, detects class imbalance, and prints a diagnostic report.
"""
import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ConfigManager
from src.data.validator import ImageValidator
from src.data.split import DatasetSplitter
from src.data.imbalance import ImbalanceAnalyzer


def main():
    parser = argparse.ArgumentParser(description="Inspect dermatoscopic skin lesion dataset.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/sample_dataset",
        help="Path to root directory containing class subfolders (e.g. data/raw or data/sample_dataset)",
    )
    args = parser.parse_args()

    cfg = ConfigManager()
    target_dir = Path(args.data_dir)
    if not target_dir.is_absolute():
        target_dir = PROJECT_ROOT / target_dir

    print("\n🔍 Scanning skin disease dataset...")
    print(f"Target Directory: {target_dir}")

    if not target_dir.exists():
        print(f"❌ Error: Directory '{target_dir}' does not exist.")
        sys.exit(1)

    # 1. Validation & Corruption Scan
    validator = ImageValidator()
    scan_results = validator.scan_directory(target_dir)
    print(f"\n📂 File Scan:")
    print(f" - Total Files Scanned : {scan_results['total_scanned']}")
    print(f" - Valid Images        : {scan_results['valid_count']}")
    print(f" - Invalid / Corrupted : {scan_results['invalid_count']}")

    if scan_results['invalid_count'] > 0:
        print("\n⚠️ Corrupted Files Detected:")
        for item in scan_results['invalid_files']:
            print(f"   * {item['file']}: {item['reason']}")

    # 2. Discover classes and build dataframe
    class_mapping = {c.code: idx for idx, c in enumerate(cfg.dataset_config.classes)}
    splitter = DatasetSplitter(validate_files=True)
    try:
        df, mapping = splitter.discover_dataset(target_dir, class_mapping=class_mapping)
    except Exception as e:
        print(f"❌ Could not parse dataset: {e}")
        sys.exit(1)

    # 3. Class Imbalance Analysis
    class_names = {idx: c.name for idx, c in enumerate(cfg.dataset_config.classes)}
    analyzer = ImbalanceAnalyzer(
        severe_threshold=10.0,
        moderate_threshold=3.0,
    )
    analysis = analyzer.analyze_distribution(df["class_idx"].tolist(), class_names=class_names)
    analyzer.print_summary(analysis)


if __name__ == "__main__":
    main()
