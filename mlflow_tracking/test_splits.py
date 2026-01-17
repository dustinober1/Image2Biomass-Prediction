"""
Create canonical train/validation/test splits for Image2Biomass project.

This script generates stratified splits at the IMAGE level (not target level)
to prevent data leakage. Since each image has 5 target predictions,
splitting by individual targets would cause the same image to appear in
multiple splits, leaking information.

The splits use image-level stratification based on Dry_Total_g (primary target)
to ensure similar biomass distribution across train/val/test sets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mlflow_tracking.data_split import create_canonical_splits, DataSplitter


def main():
    """Generate canonical splits for the Image2Biomass dataset."""

    print("Loading training data...")
    df = pd.read_csv("csiro-biomass/train.csv")

    print(f"Loaded {len(df)} rows (5 targets per image)")

    # Extract unique image IDs (sample_ids without the target suffix)
    # Format: ID1011485656__Dry_Clover_g -> ID1011485656
    df['image_id'] = df['sample_id'].apply(lambda x: x.split('__')[0])

    # Get unique images
    unique_images = df['image_id'].unique()
    n_images = len(unique_images)
    print(f"Found {n_images} unique images")

    # Create image-level dataframe for splitting
    # We use Dry_Total_g as the primary target for stratification
    image_df = df[df['target_name'] == 'Dry_Total_g'].copy()
    print(f"Using {len(image_df)} Dry_Total_g targets for stratification")

    # Extract features (we only need indices, but maintaining the pattern)
    X = np.arange(len(image_df)).reshape(-1, 1)
    y = image_df['target'].values

    print(f"\nTarget statistics (Dry_Total_g):")
    print(f"  Mean: {y.mean():.2f}")
    print(f"  Std: {y.std():.2f}")
    print(f"  Min: {y.min():.2f}")
    print(f"  Max: {y.max():.2f}")
    print(f"  Median: {np.median(y):.2f}")

    # Create stratification bins using quantiles
    # This ensures each split has similar distribution of biomass values
    n_bins = 5
    try:
        y_binned = pd.qcut(y, q=n_bins, labels=False, duplicates='drop')
        print(f"\nCreated {len(np.unique(y_binned))} stratification bins using quantiles")
    except ValueError as e:
        print(f"\nWarning: Could not create {n_bins} bins: {e}")
        print("Falling back to unstratified split")
        y_binned = None

    # Create canonical splits
    print("\nCreating canonical splits (70/15/15 train/val/test)...")
    splits = create_canonical_splits(
        X, y,
        split_file="data/canonical_splits.json",
        stratify=y_binned
    )

    print(f"\nSplit summary:")
    print(f"  Train: {splits['n_train']} images ({splits['n_train']/n_images*100:.1f}%)")
    print(f"  Val: {splits['n_val']} images ({splits['n_val']/n_images*100:.1f}%)")
    print(f"  Test: {splits['n_test']} images ({splits['n_test']/n_images*100:.1f}%)")
    print(f"  Random state: {splits['random_state']}")

    # Validate splits
    print("\nValidating splits...")
    splitter = DataSplitter()
    splitter.load_splits()

    stats = splitter.validate_splits(X, y)

    print("\nValidation results:")
    print(f"  No overlap: {stats['no_overlap']}")
    print(f"  Train ratio: {stats['train_ratio']:.3f}")
    print(f"  Val ratio: {stats['val_ratio']:.3f}")
    print(f"  Test ratio: {stats['test_ratio']:.3f}")

    print("\nTarget distribution across splits:")
    print(f"  Train mean: {stats['train_mean']:.2f} (std: {stats['train_std']:.2f})")
    print(f"  Val mean: {stats['val_mean']:.2f} (std: {stats['val_std']:.2f})")
    print(f"  Test mean: {stats['test_mean']:.2f} (std: {stats['test_std']:.2f})")

    # Check distribution balance (means should be within 10%)
    overall_mean = y.mean()
    train_diff = abs(stats['train_mean'] - overall_mean) / overall_mean * 100
    val_diff = abs(stats['val_mean'] - overall_mean) / overall_mean * 100
    test_diff = abs(stats['test_mean'] - overall_mean) / overall_mean * 100

    print(f"\nDistribution balance (deviation from overall mean {overall_mean:.2f}):")
    print(f"  Train: {train_diff:.1f}%")
    print(f"  Val: {val_diff:.1f}%")
    print(f"  Test: {test_diff:.1f}%")

    if max(train_diff, val_diff, test_diff) < 10:
        print("  ✓ All splits within 10% - stratification successful!")
    else:
        print("  ⚠ Some splits exceed 10% - consider more bins or different stratification")

    # Get actual indices
    train_idx, val_idx, test_idx = splitter.get_split_indices()

    # Map back to image IDs
    train_images = image_df.iloc[train_idx]['image_id'].values
    val_images = image_df.iloc[val_idx]['image_id'].values
    test_images = image_df.iloc[test_idx]['image_id'].values

    print(f"\nImage-level splits:")
    print(f"  Train images: {len(train_images)}")
    print(f"  Val images: {len(val_images)}")
    print(f"  Test images: {len(test_images)}")

    # Verify no image appears in multiple splits
    train_set = set(train_images)
    val_set = set(val_images)
    test_set = set(test_images)

    assert len(train_set & val_set) == 0, "Train and val share images!"
    assert len(train_set & test_set) == 0, "Train and test share images!"
    assert len(val_set & test_set) == 0, "Val and test share images!"

    print("  ✓ No image overlap between splits")

    # Calculate total row counts (5 targets per image)
    train_rows = len(train_images) * 5
    val_rows = len(val_images) * 5
    test_rows = len(test_images) * 5
    total_rows = train_rows + val_rows + test_rows

    print(f"\nExpected row counts (5 targets per image):")
    print(f"  Train: {train_rows} rows")
    print(f"  Val: {val_rows} rows")
    print(f"  Test: {test_rows} rows")
    print(f"  Total: {total_rows} rows")

    print(f"\n✓ Canonical splits saved to data/canonical_splits.json")
    print(f"  Use: DataSplitter().load_splits() to load in experiments")


if __name__ == "__main__":
    main()
