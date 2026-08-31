"""Test dataset splitting and data leakage prevention."""
import pandas as pd
import pytest

from src.data.split import DatasetSplitter


def test_stratified_split_no_leakage():
    splitter = DatasetSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_seed=42)

    # Generate synthetic DataFrame
    records = []
    for cls in range(4):
        for i in range(50):
            records.append({
                "image_path": f"/dummy/path/{cls}_{i}.jpg",
                "class_name": f"class_{cls}",
                "class_idx": cls,
            })
    df = pd.DataFrame(records)

    train_df, val_df, test_df = splitter.split_dataframe(df)

    # Check non-empty
    assert len(train_df) > 0
    assert len(val_df) > 0
    assert len(test_df) > 0
    assert len(train_df) + len(val_df) + len(test_df) == len(df)

    # Check zero overlap (data leakage check)
    train_set = set(train_df["image_path"])
    val_set = set(val_df["image_path"])
    test_set = set(test_df["image_path"])

    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)


def test_patient_grouped_split_no_leakage():
    splitter = DatasetSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_seed=42)

    records = []
    # 20 patients, each patient has 4 images
    for p_id in range(20):
        cls = p_id % 3
        for img_id in range(4):
            records.append({
                "image_path": f"/dummy/path/p{p_id}_img{img_id}.jpg",
                "patient_id": f"patient_{p_id}",
                "class_name": f"class_{cls}",
                "class_idx": cls,
            })
    df = pd.DataFrame(records)

    train_df, val_df, test_df = splitter.split_dataframe(df, group_column="patient_id")

    # Assert zero patient overlap across splits
    train_patients = set(train_df["patient_id"])
    val_patients = set(val_df["patient_id"])
    test_patients = set(test_df["patient_id"])

    assert train_patients.isdisjoint(val_patients)
    assert train_patients.isdisjoint(test_patients)
    assert val_patients.isdisjoint(test_patients)
