# Phase 10: Canonical Splits Enforcement - Research

**Researched:** 2025-01-17
**Domain:** Training script adapters, argparse, data splitting, experiment reproducibility
**Confidence:** HIGH

## Summary

This phase requires enforcing canonical data splits across all ML experiments to ensure reproducibility and fair model comparison. The current codebase has:
- `DataSplitter` class that creates and persists canonical splits to JSON
- Existing `canonical_splits.json` with train/val/test indices for 357 images
- Two adapter classes (`PyTorchAdapter`, `SklearnAdapter`) that execute training scripts via subprocess
- Training scripts that currently create their own splits inline (e.g., `train_oof_effnet.py` uses KFold, `train_ridge_advanced.py` uses KFold)

The implementation requires:
1. **Adapters load canonical splits** - Modify adapters to read `canonical_splits.json` and pass indices to scripts
2. **Training scripts accept split indices via CLI** - Add `--train-indices`, `--val-indices`, `--test-indices` flags to training scripts
3. **Scripts use provided splits instead of creating their own** - Replace inline KFold/split logic with the passed indices

**Primary recommendation:** Pass indices as comma-separated strings via CLI args using argparse with `nargs='+'` or `type=str` with custom parsing, since the indices arrays (249 train, 54 val, 54 test) are manageable as CLI arguments and avoids adding JSON file dependencies to training scripts.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `argparse` | Python 3.14+ stdlib | CLI argument parsing | Built-in, battle-tested, standard for CLIs |
| `json` | Python stdlib | Load canonical_splits.json | Standard for JSON I/O in Python |
| `numpy` | Existing | Convert indices to arrays | Already in project, efficient for indexing |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `pathlib` | Python 3.4+ | Path handling for split file | Cleaner than os.path, already used in DataSplitter |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| CLI args (`--train-indices`) | JSON file path arg | File path is simpler but adds file I/O to every training run; CLI args enforce explicit split passing |
| Comma-separated string | `nargs='+'` | `nargs='+'` is cleaner but may hit shell arg limits; comma-separated is one string, safe for large arrays |

**Installation:**
```bash
# No additional packages needed - all are standard library or already installed
```

## Architecture Patterns

### Recommended Project Structure
```
mlflow_tracking/
├── adapters.py          # Modified: load_splits() method, pass indices to CLI
├── data_split.py        # Existing: DataSplitter class (no changes needed)
scripts/
├── train_oof_effnet.py  # Modified: add --*-indices args, use passed splits
├── train_ridge_advanced.py  # Modified: add --*-indices args, use passed splits
data/
└── canonical_splits.json  # Existing: pre-computed split indices
```

### Pattern 1: Adapter Loads and Passes Splits
**What:** Adapters load `canonical_splits.json` before executing training scripts, then pass indices as CLI args.

**When to use:** All adapter classes that execute training scripts via subprocess.

**Example:**
```python
# Source: Official Python argparse documentation
# https://docs.python.org/3/library/argparse.html

class PyTorchAdapter(BaseAdapter):
    def execute(self, config: ExperimentConfig, tracker) -> Dict[str, float]:
        # Load canonical splits
        from mlflow_tracking.data_split import DataSplitter
        splitter = DataSplitter(split_file="data/canonical_splits.json")
        splits = splitter.load_splits()

        # Build command args with split indices
        args = ["python3", script_path]

        # Add split indices as comma-separated strings
        for param_name, arg_name in param_mapping.items():
            if param_name in config.parameters:
                args.extend([arg_name, str(config.parameters[param_name])])

        # Add canonical split indices
        train_str = ",".join(map(str, splits["train_indices"]))
        val_str = ",".join(map(str, splits["val_indices"]))
        test_str = ",".join(map(str, splits["test_indices"]))

        args.extend([
            "--train-indices", train_str,
            "--val-indices", val_str,
            "--test-indices", test_str
        ])

        # Execute with splits
        result = subprocess.run(args, capture_output=True, text=True, check=True)
```

### Pattern 2: Training Scripts Accept Split Indices
**What:** Training scripts add argparse arguments for split indices and use them instead of creating splits inline.

**When to use:** All training scripts (`train_oof_effnet.py`, `train_ridge_advanced.py`, etc.)

**Example:**
```python
# Source: Official Python argparse documentation
# https://docs.python.org/3/library/argparse.html

import argparse

def parse_indices(indices_str: str) -> list[int]:
    """Parse comma-separated indices string to list of integers."""
    return [int(x) for x in indices_str.split(",")]

def main():
    parser = argparse.ArgumentParser(description="Train model with canonical splits")
    parser.add_argument("--train-indices", type=str, required=True,
                        help="Comma-separated train indices from canonical_splits.json")
    parser.add_argument("--val-indices", type=str, required=True,
                        help="Comma-separated validation indices from canonical_splits.json")
    parser.add_argument("--test-indices", type=str, required=True,
                        help="Comma-separated test indices from canonical_splits.json")

    # Existing arguments
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=1e-4)

    args = parser.parse_args()

    # Parse indices
    train_idx = parse_indices(args.train_indices)
    val_idx = parse_indices(args.val_indices)
    test_idx = parse_indices(args.test_indices)

    # Load data
    df = pd.read_csv(TRAIN_CSV)

    # Use canonical splits instead of KFold
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    test_df = df.iloc[test_idx]

    # Train model with fixed splits
    train_model(train_df, val_df, test_df)

if __name__ == "__main__":
    main()
```

### Pattern 3: SklearnAdapter Follows Same Pattern
**What:** Sklearn adapter also loads and passes canonical splits to sklearn training scripts.

**Example:**
```python
# Same pattern as PyTorchAdapter - load splits, pass as CLI args
class SklearnAdapter(BaseAdapter):
    def execute(self, config: ExperimentConfig, tracker) -> Dict[str, float]:
        from mlflow_tracking.data_split import DataSplitter
        splitter = DataSplitter(split_file="data/canonical_splits.json")
        splits = splitter.load_splits()

        args = ["python3", script_path]

        # Add split indices
        train_str = ",".join(map(str, splits["train_indices"]))
        val_str = ",".join(map(str, splits["val_indices"]))
        test_str = ",".join(map(str, splits["test_indices"]))

        args.extend([
            "--train-indices", train_str,
            "--val-indices", val_str,
            "--test-indices", test_str
        ])

        # Add other parameters
        for param_name, value in config.parameters.items():
            args.extend([f"--{param_name}", str(value)])

        result = subprocess.run(args, capture_output=True, text=True, check=True)
```

### Anti-Patterns to Avoid
- **Inline KFold in scripts:** Training scripts should NOT create their own splits via `KFold()` or `train_test_split()` - this defeats the purpose of canonical splits
- **Hardcoded split paths:** Don't hardcode `data/canonical_splits.json` in training scripts - let adapters handle it
- **Optional split arguments:** Split indices should be REQUIRED, not optional, to ensure reproducibility
- **Skipping validation:** Scripts should validate that split indices don't overlap and cover the full dataset

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| CLI parsing | Custom string splitting | `argparse` with custom `type` function | Handles validation, error messages, help generation |
| JSON loading | Custom file reading | `json.load()` with context manager | Handles encoding, errors, file closing |
| List parsing | Manual split logic | Function with list comprehension | Cleaner, handles edge cases, testable |
| Path handling | String concatenation | `pathlib.Path` | Cross-platform, handles edge cases |

**Key insight:** Using standard library tools (`argparse`, `json`, `pathlib`) reduces code, improves reliability, and follows Python best practices. Custom implementations often miss edge cases (e.g., trailing commas, whitespace, path separators).

## Common Pitfalls

### Pitfall 1: Shell Argument Length Limits
**What goes wrong:** Very long comma-separated index strings may exceed shell argument limits (typically ~2MB on Linux, ~32KB on Windows).

**Why it happens:** CLI arguments are passed through the shell, which has size limits.

**How to avoid:**
- Current splits are safe: 249+54+54 = 357 indices ≈ 1.5KB as CSV (well under limits)
- If dataset grows beyond ~10k images, consider passing a JSON file path instead
- Monitor: log the length of index strings in adapters

**Warning signs:** Scripts fail with "Argument list too long" errors

### Pitfall 2: Inconsistent Index Parsing
**What goes wrong:** Adapter passes `1,2,3` but script expects `[1, 2, 3]` or `"1 2 3"`.

**Why it happens:** No documented format for passing lists via CLI.

**How to avoid:**
- Document the expected format: "comma-separated integers, no spaces, e.g., `1,2,3`"
- Add validation in scripts: `parse_indices()` should handle whitespace
- Test with edge cases: empty string, single index, trailing comma

**Warning signs:** `ValueError: invalid literal for int()` when parsing indices

### Pitfall 3: Training Scripts Ignore Splits
**What goes wrong:** Scripts receive split indices via CLI but still create their own KFold splits.

**Why it happens:** Developer adds args but doesn't remove old split logic.

**How to avoid:**
- Add assertion in scripts: `assert args.train_indices is not None, "Must provide --train-indices"`
- Search for all `KFold`, `train_test_split`, `GroupKFold` usage and replace
- Add tests that verify splits are used (check that train/val/test sets match indices)

**Warning signs:** Different experiments report different train/val/test counts

### Pitfall 4: Duplicate Split Loading
**What goes wrong:** Both adapter AND training script load `canonical_splits.json`, potentially inconsistent versions.

**Why it happens:** Convenience - "let the script load it directly too."

**How to avoid:**
- Single source of truth: ONLY adapters load the split file
- Scripts receive indices as CLI args and use them blindly
- Validate: add test that checks adapter and script use same splits

**Warning signs:** Reproducibility tests fail with "split mismatch" errors

### Pitfall 5: Missing Test Set Usage
**What goes wrong:** Scripts train on train/val splits but never evaluate on test set.

**Why it happens:** Focus on training loop, forget final evaluation.

**How to avoid:**
- Require `--test-indices` argument (enforce usage)
- Add test set evaluation after training
- Log test metrics to MLflow (separate from val metrics)

**Warning signs:** MLflow runs have no `test.*` metrics, only `train.*` and `val.*`

## Code Examples

Verified patterns from official sources:

### Parse Comma-Separated Indices
```python
# Source: Python argparse documentation
# https://docs.python.org/3/library/argparse.html

def parse_indices(indices_str: str) -> list[int]:
    """Convert comma-separated indices string to list of integers.

    Args:
        indices_str: Comma-separated integers, e.g., "1,2,3,4,5"

    Returns:
        List of integer indices

    Raises:
        ValueError: If indices_str contains non-integer values
    """
    if not indices_str:
        return []
    # Strip whitespace and split by comma
    return [int(x.strip()) for x in indices_str.split(",") if x.strip()]

# Usage in argparse
parser.add_argument(
    "--train-indices",
    type=str,
    required=True,
    help="Comma-separated train indices, e.g., '0,1,2,3'"
)
args = parser.parse_args()
train_indices = parse_indices(args.train_indices)
```

### Adapter Loads Splits
```python
# Source: Existing mlflow_tracking/data_split.py
from mlflow_tracking.data_split import DataSplitter

def load_canonical_splits() -> dict:
    """Load canonical train/val/test splits from JSON file.

    Returns:
        Dict with 'train_indices', 'val_indices', 'test_indices' lists

    Raises:
        FileNotFoundError: If canonical_splits.json doesn't exist
    """
    splitter = DataSplitter(split_file="data/canonical_splits.json")
    splits = splitter.load_splits()
    return splits

# Usage in adapter
splits = load_canonical_splits()
train_idx = splits["train_indices"]  # List[int]
val_idx = splits["val_indices"]      # List[int]
test_idx = splits["test_indices"]    # List[int]
```

### Build Subprocess Args with Splits
```python
# Source: Existing mlflow_tracking/adapters.py pattern
import subprocess

def build_command_with_splits(script_path: str, splits: dict, params: dict) -> list[str]:
    """Build subprocess command including split indices.

    Args:
        script_path: Path to training script
        splits: Dict from DataSplitter.load_splits()
        params: Other parameters to pass

    Returns:
        List of command arguments for subprocess.run()
    """
    args = ["python3", script_path]

    # Add split indices as CSV strings
    for split_name in ["train_indices", "val_indices", "test_indices"]:
        arg_name = f"--{split_name.replace('_', '-')}"
        indices_str = ",".join(map(str, splits[split_name]))
        args.extend([arg_name, indices_str])

    # Add other parameters
    for key, value in params.items():
        args.extend([f"--{key}", str(value)])

    return args

# Usage
args = build_command_with_splits(
    script_path="scripts/train_oof_effnet.py",
    splits=splits,
    params={"batch_size": 16, "epochs": 30}
)
result = subprocess.run(args, capture_output=True, text=True, check=True)
```

### Training Script Uses Provided Splits
```python
# Before (OLD - using KFold):
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
    # Train on different splits each time
    pass

# After (NEW - using canonical splits):
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--train-indices", type=str, required=True)
parser.add_argument("--val-indices", type=str, required=True)
parser.add_argument("--test-indices", type=str, required=True)
args = parser.parse_args()

# Parse indices
train_idx = parse_indices(args.train_indices)
val_idx = parse_indices(args.val_indices)
test_idx = parse_indices(args.test_indices)

# Use fixed splits
train_df = df.iloc[train_idx]
val_df = df.iloc[val_idx]
test_df = df.iloc[test_idx]

# Train ONCE with canonical splits
model = train_model(train_df, val_df)
test_metrics = evaluate(model, test_df)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Inline KFold in each script | Canonical splits from DataSplitter | Phase 1 | All experiments use identical splits, enabling fair comparison |
| Script-specific split logic | Adapter-managed splits | Phase 10 | Scripts are simpler, splits are enforced by adapter layer |

**Deprecated/outdated:**
- **Inline `KFold()` in scripts:** Should be removed and replaced with canonical split indices
- **Script-generated `train_test_split()`:** Should use `--test-indices` from adapter instead

## Open Questions

1. **Should we support both canonical splits and custom splits?**
   - What we know: Success criteria require canonical splits for reproducibility
   - What's unclear: Whether users might want to test different split strategies
   - Recommendation: Start with canonical-only (required), add optional `--split-file` later if needed

2. **Should indices be passed as CSV or JSON strings?**
   - What we know: CSV is simpler (`"1,2,3"`), JSON is more explicit (`"[1,2,3]"`)
   - What's unclear: Which is more maintainable long-term
   - Recommendation: CSV is sufficient for current dataset size; JSON if dataset grows >10k images

3. **Should we validate splits in the adapter or training script?**
   - What we know: Both can validate (no overlap, correct counts)
   - What's unclear: Where validation should live to avoid duplication
   - Recommendation: Validate in both - adapter validates loaded splits, script validates received args

## Sources

### Primary (HIGH confidence)
- [Python argparse documentation](https://docs.python.org/3/library/argparse.html) - Official argparse API reference, verified 2025-01-17
- [StackOverflow: Pass list as CLI argument](https://stackoverflow.com/questions/15753701/how-can-i-pass-a-list-as-a-command-line-argument-with-argparse) - Community-validated patterns for passing lists
- [Existing codebase: mlflow_tracking/data_split.py](file:///Users/dustinober/Projects/Image2Biomass-Prediction/mlflow_tracking/data_split.py) - DataSplitter implementation (verified existing)
- [Existing codebase: mlflow_tracking/adapters.py](file:///Users/dustinober/Projects/Image2Biomass-Prediction/mlflow_tracking/adapters.py) - Adapter patterns (verified existing)
- [Existing codebase: data/canonical_splits.json](file:///Users/dustinober/Projects/Image2Biomass-Prediction/data/canonical_splits.json) - Split file format (verified existing)

### Secondary (MEDIUM confidence)
- [GeeksforGeeks: argparse with lists](https://www.geeksforgeeks.org/python/how-to-pass-a-list-as-a-command-line-argument-with-argparse/) - Examples of `nargs` usage (verified with official docs)
- [Real Python: argparse tutorial](https://realpython.com/python-command-line-arguments) - Comprehensive argparse guide (verified 2025-06-15)

### Tertiary (LOW confidence)
- [Medium: Combining JSON with argparse](https://medium.com/swlh/efficient-python-user-interfaces-combining-json-with-argparse-8bff716f31e4) - Alternative approach using JSON files (not used in recommendation)
- [jsonargparse library](https://jsonargparse.readthedocs.io/) - Extended argparse with JSON support (not needed for this phase)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All standard library or already in project
- Architecture: HIGH - Based on existing adapter patterns in codebase
- Pitfalls: HIGH - Based on official argparse docs and common CLI issues

**Research date:** 2025-01-17
**Valid until:** 2025-02-17 (30 days - stable domain, standard libraries)
