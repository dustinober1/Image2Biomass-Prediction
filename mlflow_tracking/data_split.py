"""
Data splitting utilities for canonical train/validation/test splits.

This module provides tools to create and manage reproducible data splits
that prevent data leakage and enable fair model comparison across experiments.
"""

import json
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from pathlib import Path
from typing import Tuple, Dict, Optional


class DataSplitter:
    """
    Manages canonical three-way data splits for reproducible experiments.

    Ensures train/validation/test isolation to prevent data leakage and
    enables valid model comparisons across experiments using identical splits.
    """

    def __init__(self, split_file: str = "data/canonical_splits.json", random_state: int = 42):
        """
        Initialize with path to canonical split file.

        Args:
            split_file: Path to JSON file storing split indices
            random_state: Random seed for reproducibility
        """
        self.split_file = Path(split_file)
        self.random_state = random_state
        self.splits = None

    def create_splits(self, X: np.ndarray, y: np.ndarray,
                    train_size: float = 0.7,
                    val_size: float = 0.15,
                    test_size: float = 0.15,
                    stratify: Optional[np.ndarray] = None) -> Dict:
        """
        Create stratified three-way split.

        First split: separate test set (train_val vs test)
        Second split: separate train and validation from remaining

        Args:
            X: Feature array (n_samples, n_features)
            y: Target array (n_samples,)
            train_size: Fraction of data for training (default 0.7)
            val_size: Fraction of data for validation (default 0.15)
            test_size: Fraction of data for testing (default 0.15)
            stratify: Array for stratified splitting (e.g., binned targets)

        Returns:
            Dict with train_indices, val_indices, test_indices, and metadata

        Raises:
            AssertionError: If split sizes don't sum to 1.0
        """
        # Validate sizes sum to 1.0
        assert abs((train_size + val_size + test_size) - 1.0) < 1e-6, \
            f"Split sizes must sum to 1.0, got {train_size + val_size + test_size}"

        n_samples = len(X)
        indices = np.arange(n_samples)

        # First split: separate test
        if stratify is not None:
            X_temp, X_test, y_temp, y_test, idx_temp, idx_test = train_test_split(
                X, y, indices, test_size=test_size, random_state=self.random_state, stratify=stratify
            )
            stratify_temp = stratify[idx_temp]
        else:
            X_temp, X_test, y_temp, y_test, idx_temp, idx_test = train_test_split(
                X, y, indices, test_size=test_size, random_state=self.random_state
            )
            stratify_temp = None

        # Second split: separate train and validation
        # Adjust val_size to be relative to remaining data
        val_size_adjusted = val_size / (train_size + val_size)

        if stratify_temp is not None:
            X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
                X_temp, y_temp, idx_temp, test_size=val_size_adjusted, random_state=self.random_state, stratify=stratify_temp
            )
        else:
            X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
                X_temp, y_temp, idx_temp, test_size=val_size_adjusted, random_state=self.random_state
            )

        self.splits = {
            "train_indices": idx_train.tolist(),
            "val_indices": idx_val.tolist(),
            "test_indices": idx_test.tolist(),
            "n_train": len(idx_train),
            "n_val": len(idx_val),
            "n_test": len(idx_test),
            "random_state": self.random_state
        }

        return self.splits

    def save_splits(self):
        """Save splits to JSON file."""
        self.split_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.split_file, 'w') as f:
            json.dump(self.splits, f, indent=2)
        print(f"Saved canonical splits to {self.split_file}")

    def load_splits(self) -> Dict:
        """
        Load splits from JSON file.

        Returns:
            Dict containing split indices and metadata

        Raises:
            FileNotFoundError: If split file doesn't exist
        """
        if not self.split_file.exists():
            raise FileNotFoundError(
                f"Split file not found: {self.split_file}. Run create_splits() first."
            )
        with open(self.split_file) as f:
            self.splits = json.load(f)
        return self.splits

    def get_split_indices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return train, val, test indices as numpy arrays.

        Returns:
            Tuple of (train_indices, val_indices, test_indices)
        """
        if self.splits is None:
            self.load_splits()
        return (
            np.array(self.splits["train_indices"]),
            np.array(self.splits["val_indices"]),
            np.array(self.splits["test_indices"])
        )

    def validate_splits(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Dict:
        """
        Validate splits maintain distribution and no overlap.

        Args:
            X: Feature array
            y: Optional target array for distribution validation

        Returns:
            Dict with validation statistics

        Raises:
            AssertionError: If overlaps detected between splits
        """
        train_idx, val_idx, test_idx = self.get_split_indices()

        # Check no overlap
        assert len(set(train_idx) & set(val_idx)) == 0, "Train and val overlap"
        assert len(set(train_idx) & set(test_idx)) == 0, "Train and test overlap"
        assert len(set(val_idx) & set(test_idx)) == 0, "Val and test overlap"

        stats = {
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "n_test": len(test_idx),
            "train_ratio": len(train_idx) / len(X),
            "val_ratio": len(val_idx) / len(X),
            "test_ratio": len(test_idx) / len(X),
            "no_overlap": True
        }

        if y is not None:
            stats["train_mean"] = float(y[train_idx].mean())
            stats["val_mean"] = float(y[val_idx].mean())
            stats["test_mean"] = float(y[test_idx].mean())
            stats["train_std"] = float(y[train_idx].std())
            stats["val_std"] = float(y[val_idx].std())
            stats["test_std"] = float(y[test_idx].std())

        return stats


def create_canonical_splits(X: np.ndarray, y: np.ndarray,
                            split_file: str = "data/canonical_splits.json",
                            stratify: Optional[np.ndarray] = None) -> Dict:
    """
    Create and save canonical splits in one call.

    Convenience function for standalone usage without managing DataSplitter instance.

    Args:
        X: Feature array
        y: Target array
        split_file: Path to save splits
        stratify: Optional array for stratified splitting

    Returns:
        Dict with split indices and metadata
    """
    splitter = DataSplitter(split_file=split_file)
    splits = splitter.create_splits(X, y, stratify=stratify)
    splitter.save_splits()
    return splits
