
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import json
import argparse

# Config
DATA_DIR = 'csiro-biomass'
TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')
OUTPUT_DIR = 'models/exp14_ridge_advanced'
os.makedirs(OUTPUT_DIR, exist_ok=True)

OOF_TABULAR = 'models/stacking/tabular/oof_tabular.csv'
OOF_KMEANS = 'models/stacking/kmeans/oof_kmeans.csv'
OOF_EFFNET = 'models/stacking/effnet/oof_effnet.csv'
FEATURES_GRID = 'models/features_grid/features_grid_train.csv'

TARGETS = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']

def parse_indices(indices_str):
    """Parse comma-separated indices string to list of integers."""
    return [int(x.strip()) for x in indices_str.split(",") if x.strip()]

def train_ridge_advanced(train_indices=None, val_indices=None, test_indices=None):
    print("Training Ridge Meta-Learner with Advanced Features...")
    df_train = pd.read_csv(TRAIN_CSV)
    wide_df = df_train.pivot_table(index='image_path', columns='target_name', values='target', aggfunc='first').reset_index()

    oof_tab = pd.read_csv(OOF_TABULAR)
    oof_km = pd.read_csv(OOF_KMEANS)
    oof_eff = pd.read_csv(OOF_EFFNET)
    grid_df = pd.read_csv(FEATURES_GRID)

    stacked_df = pd.merge(wide_df, oof_tab, on='image_path')
    stacked_df = pd.merge(stacked_df, oof_km, on='image_path')
    stacked_df = pd.merge(stacked_df, oof_eff, on='image_path')
    stacked_df = pd.merge(stacked_df, grid_df, on='image_path')

    meta_cols = ['Height_Ave_cm', 'Pre_GSHH_NDVI']
    meta_df = df_train.groupby('image_path')[meta_cols].first().reset_index()
    stacked_df = pd.merge(stacked_df, meta_df, on='image_path')

    drop_cols = ['image_path'] + TARGETS
    feature_cols = [c for c in stacked_df.columns if c not in drop_cols]

    # Use provided split indices if available, otherwise fall back to KFold
    if train_indices is not None and val_indices is not None and test_indices is not None:
        train_idx = parse_indices(train_indices) if isinstance(train_indices, str) else train_indices
        val_idx = parse_indices(val_indices) if isinstance(val_indices, str) else val_indices
        test_idx = parse_indices(test_indices) if isinstance(test_indices, str) else test_indices
        use_canonical_splits = True
    else:
        use_canonical_splits = False

    # Track all predictions for error analysis
    all_predictions = {}
    all_actuals = {}
    image_ids = stacked_df['image_path'].values

    if use_canonical_splits:
        # Use canonical splits (single train/val split, not KFold)
        print("Using canonical splits from DataSplitter")
        train_df = stacked_df.iloc[train_idx]
        val_df = stacked_df.iloc[val_idx]
        test_df = stacked_df.iloc[test_idx]

        # Train model on canonical splits
        for target in TARGETS:
            y = stacked_df[target].values
            X = stacked_df[feature_cols].values

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = Ridge(alpha=1.0)
            model.fit(X_train, y_train)

            preds = model.predict(X_val)
            all_predictions[target] = np.zeros(len(X))
            all_predictions[target][val_idx] = preds
            all_actuals[target] = y

            val_rmse = np.sqrt(mean_squared_error(y_val, preds))
            print(f"  -> {target} Ridge RMSE: {val_rmse:.4f}")

        print(f"\nValidation RMSE with canonical splits complete")

    else:
        # Original KFold code (keep as fallback)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        overall_rmse = []

        for target in TARGETS:
            y = stacked_df[target].values
            X = stacked_df[feature_cols].values

            target_preds = np.zeros(len(X))
            target_rmses = []

            for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                model = Ridge(alpha=1.0)
                model.fit(X_train, y_train)

                preds = model.predict(X_val)
                target_preds[val_idx] = preds
                rmse = np.sqrt(mean_squared_error(y_val, preds))
                target_rmses.append(rmse)

            all_predictions[target] = target_preds
            all_actuals[target] = y
            avg_rmse = np.mean(target_rmses)
            print(f"  -> {target} Ridge RMSE: {avg_rmse:.4f}")
            overall_rmse.append(avg_rmse)

        print(f"\nOverall Ridge + Advanced Feats RMSE: {np.mean(overall_rmse):.4f}")

    # Save predictions for error analysis (Dry_Total_g target)
    predictions_df = pd.DataFrame({
        'image_id': image_ids,
        'actual': all_actuals['Dry_Total_g'],
        'predicted': all_predictions['Dry_Total_g']
    })
    predictions_df.to_csv('predictions.csv', index=False)
    print(f"Saved predictions.csv for error analysis")

def main():
    parser = argparse.ArgumentParser(description="Train Ridge with canonical splits")
    parser.add_argument("--train-indices", type=str, help="Comma-separated train indices")
    parser.add_argument("--val-indices", type=str, help="Comma-separated val indices")
    parser.add_argument("--test-indices", type=str, help="Comma-separated test indices")
    # Note: Keep existing params for backward compatibility
    parser.add_argument("--model_type", type=str, default="ridge")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--random_seed", type=int, default=42)

    args = parser.parse_args()

    # Call train_ridge_advanced with split indices if provided
    if args.train_indices and args.val_indices and args.test_indices:
        train_ridge_advanced(
            train_indices=args.train_indices,
            val_indices=args.val_indices,
            test_indices=args.test_indices
        )
    else:
        # Fallback to KFold if no split indices provided
        train_ridge_advanced()

if __name__ == "__main__":
    main()
