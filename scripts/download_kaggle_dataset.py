#!/usr/bin/env python3
"""
CLI Tool: Download and Organize HAM10000 Skin Lesion Dataset from Kaggle.
Dataset: kmader/skin-cancer-mnist-ham10000
Organizes images automatically into:
  data/raw/<class_code>/*.jpg (mel, nv, bcc, akiec, bkl, df, vasc)
"""
import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger

logger = get_logger("kaggle_downloader")

DATASET_SLUG = "kmader/skin-cancer-mnist-ham10000"


def check_kaggle_auth() -> bool:
    """Verify kaggle credentials exist in ~/.kaggle/kaggle.json or env vars."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        return True
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return False


def organize_ham10000(download_dir: Path, target_raw_dir: Path):
    """
    Reads HAM10000_metadata.csv and sorts all .jpg images from
    HAM10000_images_part_1 and HAM10000_images_part_2 into class subfolders:
    data/raw/{dx}/{image_id}.jpg
    """
    logger.info("Locating HAM10000_metadata.csv...")
    metadata_candidates = list(download_dir.glob("**/HAM10000_metadata.csv"))
    if not metadata_candidates:
        # Fallback to any metadata CSV
        metadata_candidates = list(download_dir.glob("**/*metadata*.csv"))

    if not metadata_candidates:
        raise FileNotFoundError(f"Could not find HAM10000_metadata.csv in {download_dir}")

    meta_path = metadata_candidates[0]
    logger.info(f"Found metadata at: {meta_path}")
    df = pd.read_csv(meta_path)

    # Save a copy of metadata in data/raw
    target_raw_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(meta_path, target_raw_dir / "metadata.csv")

    # Map image_id to dx (class code)
    # Columns in HAM10000: lesion_id, image_id, dx, dx_type, age, sex, localization
    image_to_dx = dict(zip(df["image_id"], df["dx"]))
    logger.info(f"Loaded {len(image_to_dx)} image annotations across {df['dx'].nunique()} classes.")

    # Find all .jpg files in download_dir
    logger.info("Scanning for skin lesion image files (.jpg)...")
    image_files = list(download_dir.glob("**/*.jpg"))
    logger.info(f"Found {len(image_files)} image files. Organizing into class directories in {target_raw_dir}...")

    copied = 0
    missing_annotation = 0

    for img_path in image_files:
        stem = img_path.stem  # e.g. 'ISIC_0024306'
        dx = image_to_dx.get(stem)
        if not dx:
            missing_annotation += 1
            continue

        class_dir = target_raw_dir / dx
        class_dir.mkdir(parents=True, exist_ok=True)

        dest = class_dir / img_path.name
        if not dest.exists():
            shutil.copy(img_path, dest)
        copied += 1

    logger.info(f"Successfully organized {copied} images into {target_raw_dir}.")
    if missing_annotation > 0:
        logger.warning(f"{missing_annotation} images had no matching diagnosis in metadata.")

    # Print summary
    print("\n" + "=" * 50)
    print("      DATASET ORGANIZATION SUMMARY")
    print("=" * 50)
    for class_folder in sorted(target_raw_dir.iterdir()):
        if class_folder.is_dir() and not class_folder.name.startswith("."):
            count = len(list(class_folder.glob("*.jpg")))
            print(f" - {class_folder.name:<10}: {count:>5} images")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Download and organize HAM10000 dataset from Kaggle.")
    parser.add_argument(
        "--zip-path",
        type=str,
        default=None,
        help="Path to an already downloaded Kaggle zip file (e.g. skin-cancer-mnist-ham10000.zip)",
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default=None,
        help="Path to an already extracted Kaggle folder containing the dataset",
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default="data/raw",
        help="Target raw directory to organize class folders into (default: data/raw)",
    )
    args = parser.parse_args()

    target_raw_dir = Path(args.target_dir)
    if not target_raw_dir.is_absolute():
        target_raw_dir = PROJECT_ROOT / target_raw_dir

    # Option A: User provided an extracted directory
    if args.source_dir:
        src_path = Path(args.source_dir)
        if not src_path.exists():
            logger.error(f"Provided source directory does not exist: {src_path}")
            sys.exit(1)
        organize_ham10000(src_path, target_raw_dir)
        return

    # Option B: User provided an already downloaded zip file
    if args.zip_path:
        zip_path = Path(args.zip_path)
        if not zip_path.exists():
            logger.error(f"Provided zip file does not exist: {zip_path}")
            sys.exit(1)

        temp_extract = PROJECT_ROOT / "data" / "kaggle_temp_extract"
        logger.info(f"Extracting {zip_path} to {temp_extract}...")
        temp_extract.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(temp_extract)

        organize_ham10000(temp_extract, target_raw_dir)

        logger.info(f"Cleaning up temporary folder {temp_extract}...")
        shutil.rmtree(temp_extract, ignore_errors=True)
        return

    # Option C: Use Kaggle API to download automatically
    if not check_kaggle_auth():
        print("\n" + "!" * 65)
        print("⚠️  Kaggle API Credentials Not Found!")
        print("!" * 65)
        print("To download directly via script, you need your Kaggle API key:")
        print("1. Go to: https://www.kaggle.com/settings")
        print("2. Scroll to 'API' section -> Click 'Create New Token'")
        print("3. Place the downloaded 'kaggle.json' in: ~/.kaggle/kaggle.json")
        print("   chmod 600 ~/.kaggle/kaggle.json")
        print("\nAlternatively, you can manually download the dataset zip from:")
        print(" 👉 https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000")
        print("\nOnce downloaded, run:")
        print(" python scripts/download_kaggle_dataset.py --zip-path /path/to/downloaded.zip")
        print("!" * 65 + "\n")
        sys.exit(1)

    try:
        import kaggle
    except ImportError:
        logger.info("Installing kaggle python package...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
        import kaggle

    temp_download = PROJECT_ROOT / "data" / "kaggle_temp_download"
    temp_download.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading dataset '{DATASET_SLUG}' from Kaggle via API (approx 2.6 GB)...")
    kaggle.api.dataset_download_files(DATASET_SLUG, path=str(temp_download), unzip=True)

    organize_ham10000(temp_download, target_raw_dir)

    logger.info(f"Cleaning up temporary download folder...")
    shutil.rmtree(temp_download, ignore_errors=True)
    logger.info("Done!")


if __name__ == "__main__":
    main()
