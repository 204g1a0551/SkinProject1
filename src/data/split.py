"""
Leakage-free dataset splitting utility for skin disease datasets.
Supports:
1. Pure directory-based dataset (data/raw/<class_name>/*.jpg) with stratified splitting.
2. Metadata-guided dataset (with patient_id or lesion_id) with grouped stratified splitting
   to ensure multiple captures of the same lesion or patient never cross train/val/test splits.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedGroupKFold

from .validator import ImageValidator, SUPPORTED_IMAGE_EXTENSIONS
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DatasetSplitter:
    """Discovers, validates, and partitions skin disease datasets without leakage."""

    def __init__(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42,
        validate_files: bool = True,
    ):
        assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Ratios must sum to 1.0"
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        self.validator = ImageValidator() if validate_files else None

    def discover_dataset(
        self,
        data_dir: Union[str, Path],
        class_mapping: Optional[Dict[str, int]] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Scan directory for class folders.
        Expected structure:
            data_dir/
                mel/
                    img1.jpg
                    ...
                nv/
                    ...
        """
        root = Path(data_dir).resolve()
        if not root.exists():
            raise FileNotFoundError(f"Dataset root directory does not exist: {root}")

        records: List[Dict[str, Any]] = []

        # Discover subdirectories as classes
        class_dirs = [d for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")]

        if not class_dirs:
            raise ValueError(f"No class subdirectories found inside {root}.")

        # Auto-generate class mapping if not provided
        if class_mapping is None:
            sorted_classes = sorted([d.name for d in class_dirs])
            class_mapping = {cls_name: idx for idx, cls_name in enumerate(sorted_classes)}

        logger.info(f"Discovered {len(class_dirs)} class folders: {list(class_mapping.keys())}")

        corrupted_count = 0
        for class_dir in class_dirs:
            cls_name = class_dir.name
            if cls_name not in class_mapping:
                logger.warning(f"Skipping directory '{cls_name}' (not in defined class mapping).")
                continue

            cls_idx = class_mapping[cls_name]

            for img_path in class_dir.iterdir():
                if img_path.is_file() and img_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                    if self.validator:
                        is_valid, reason = self.validator.validate_file(img_path)
                        if not is_valid:
                            logger.warning(f"Quarantined corrupt file {img_path.name}: {reason}")
                            corrupted_count += 1
                            continue

                    records.append({
                        "image_path": str(img_path.resolve()),
                        "filename": img_path.name,
                        "class_name": cls_name,
                        "class_idx": cls_idx,
                    })

        if not records:
            raise ValueError(f"No valid images found in {root}.")

        df = pd.DataFrame(records)
        logger.info(f"Loaded {len(df)} valid images across {len(class_mapping)} classes. (Corrupted: {corrupted_count})")
        return df, class_mapping

    def split_dataframe(
        self,
        df: pd.DataFrame,
        group_column: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split DataFrame into train, val, and test splits without data leakage.

        Args:
            df: DataFrame containing image_path, class_idx, and optionally group_column
            group_column: Optional column name for grouping (e.g. 'patient_id' or 'lesion_id')
        """
        if df.empty:
            raise ValueError("Cannot split empty DataFrame.")

        # Scenario A: Patient / Lesion Grouped Stratified Split (prevents patient-level leakage)
        if group_column and group_column in df.columns:
            logger.info(f"Using Grouped Stratified Splitting on '{group_column}' to prevent leakage.")
            train_df, val_df, test_df = self._grouped_stratified_split(df, group_column=group_column)
        else:
            # Scenario B: Standard Stratified Split by Class Label
            logger.info("Using Stratified Splitting to preserve class distributions.")
            train_df, val_df, test_df = self._stratified_split(df)

        # Integrity Check: Assert mutual exclusivity (no data leakage)
        train_paths = set(train_df["image_path"])
        val_paths = set(val_df["image_path"])
        test_paths = set(test_df["image_path"])

        assert train_paths.isdisjoint(val_paths), "Data leakage detected: train and val share images!"
        assert train_paths.isdisjoint(test_paths), "Data leakage detected: train and test share images!"
        assert val_paths.isdisjoint(test_paths), "Data leakage detected: val and test share images!"

        if group_column and group_column in df.columns:
            train_groups = set(train_df[group_column])
            val_groups = set(val_df[group_column])
            test_groups = set(test_df[group_column])
            assert train_groups.isdisjoint(val_groups), f"Patient leakage: train and val share {group_column}!"
            assert train_groups.isdisjoint(test_groups), f"Patient leakage: train and test share {group_column}!"

        logger.info(
            f"Dataset split complete: Train={len(train_df)} ({len(train_df)/len(df):.1%}), "
            f"Val={len(val_df)} ({len(val_df)/len(df):.1%}), "
            f"Test={len(test_df)} ({len(test_df)/len(df):.1%})"
        )

        return train_df, val_df, test_df

    def _stratified_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Perform 2-step stratified splitting."""
        # 1. Split into Train and Temp (Val + Test)
        temp_ratio = self.val_ratio + self.test_ratio
        
        # Check minimum class frequency
        min_class_count = df["class_idx"].value_counts().min()
        stratify = df["class_idx"] if min_class_count >= 2 else None

        train_df, temp_df = train_test_split(
            df,
            test_size=temp_ratio,
            stratify=stratify,
            random_state=self.random_seed,
            shuffle=True,
        )

        # 2. Split Temp into Val and Test
        val_share = self.val_ratio / temp_ratio
        min_temp_count = temp_df["class_idx"].value_counts().min()
        stratify_temp = temp_df["class_idx"] if min_temp_count >= 2 else None

        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1.0 - val_share),
            stratify=stratify_temp,
            random_state=self.random_seed,
            shuffle=True,
        )

        return train_df.copy(), val_df.copy(), test_df.copy()

    def _grouped_stratified_split(
        self,
        df: pd.DataFrame,
        group_column: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Patient-level grouped splitting using StratifiedGroupKFold."""
        # Split into ~5 folds (e.g. 70% train = ~3.5 folds, 15% val, 15% test)
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=self.random_seed)
        folds = list(sgkf.split(df, df["class_idx"], df[group_column]))

        test_idx = folds[0][1]
        val_idx = folds[1][1]
        train_idx = np.concatenate([folds[2][1], folds[3][1], folds[4][1]])

        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()
        test_df = df.iloc[test_idx].copy()

        return train_df, val_df, test_df

    def save_split_manifests(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        output_dir: Union[str, Path],
    ) -> Dict[str, Path]:
        """Save split dataframes to CSV manifests."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        train_path = out / "train_manifest.csv"
        val_path = out / "val_manifest.csv"
        test_path = out / "test_manifest.csv"

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)

        logger.info(f"Saved split manifests to {out}")
        return {"train": train_path, "val": val_path, "test": test_path}
