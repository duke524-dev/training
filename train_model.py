"""
Training script for 6 Zeus subnet ensemble models.
Trains Ridge Regression models to predict ERA5 values from multiple forecast model outputs.

Zeus Subnet Variables:
1. 2m_temperature (K)
2. 2m_dewpoint_temperature (K)
3. surface_pressure (Pa)
4. total_precipitation (mm)
5. 100m_u_component_of_wind (m/s)
6. 100m_v_component_of_wind (m/s)

Each variable gets its own trained model.
"""

import pandas as pd
import numpy as np
import os
import json
import pickle
from datetime import datetime
from glob import glob
from typing import Tuple, Dict, Any

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

from config import TRAINING_CONFIG, ZEUS_VARIABLES, FEATURE_COLUMNS, TARGET_COLUMN


# Configuration
CONFIG = TRAINING_CONFIG.copy()


def find_integrated_file(variable: str, data_dir: str = "data") -> str:
    """Find the latest integrated data file for a specific variable."""
    safe_var_name = variable.replace("/", "_")
    pattern = os.path.join(data_dir, f"integrated_{safe_var_name}_*.csv")
    files = glob(pattern)
    
    if not files:
        return None
    
    return sorted(files)[-1]


def load_data(file_path: str) -> pd.DataFrame:
    """Load the integrated data."""
    df = pd.read_csv(file_path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def clean_data(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Remove rows with missing values in features or target."""
    required_cols = feature_cols + [TARGET_COLUMN]
    existing_cols = [c for c in required_cols if c in df.columns]
    
    initial_count = len(df)
    df_clean = df.dropna(subset=existing_cols)
    removed_count = initial_count - len(df_clean)
    
    if removed_count > 0:
        print(f"    Removed {removed_count} rows with missing values")
    
    return df_clean


def split_data(
    df: pd.DataFrame,
    feature_cols: list[str],
    test_size: float = 0.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Split data into training and test sets (time-based)."""
    
    df = df.sort_values("datetime").reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_size))
    
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    X_train = train_df[feature_cols].values
    X_test = test_df[feature_cols].values
    y_train = train_df[TARGET_COLUMN].values
    y_test = test_df[TARGET_COLUMN].values
    
    return X_train, X_test, y_train, y_test, feature_cols


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate evaluation metrics."""
    return {
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "bias": np.mean(y_pred - y_true)
    }


def train_model_for_variable(
    variable: str,
    data_dir: str,
    output_dir: str,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Train a Ridge model for a specific variable."""
    
    print(f"\n{'='*60}")
    print(f"Training: {variable}")
    print(f"{'='*60}")
    
    # Find and load data
    file_path = find_integrated_file(variable, data_dir)
    if file_path is None:
        print(f"  [ERROR] No integrated data file found for {variable}")
        return None
    
    print(f"  Data file: {os.path.basename(file_path)}")
    df = load_data(file_path)
    print(f"  Total records: {len(df)}")

    # Determine available feature columns (drop columns that are all NaN)
    available_features = []
    for col in FEATURE_COLUMNS:
        if col in df.columns and df[col].notna().any():
            available_features.append(col)

    if not available_features:
        print("  [ERROR] No usable feature columns (all NaN).")
        return None

    # Clean data
    df_clean = clean_data(df, available_features)
    print(f"  Clean records: {len(df_clean)}")

    if len(df_clean) < 10:
        print(f"  [ERROR] Not enough data to train")
        return None
    
    # Split data
    X_train, X_test, y_train, y_test, feature_cols = split_data(
        df_clean, available_features, config["test_size"]
    )
    print(f"  Training set: {len(X_train)}, Test set: {len(X_test)}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Baseline: Simple ensemble mean
    baseline_pred = np.mean(X_test, axis=1)
    baseline_metrics = evaluate_model(y_test, baseline_pred)
    print(f"\n  Baseline (Ensemble Mean):")
    print(f"    RMSE: {baseline_metrics['rmse']:.4f}")
    
    # Train Ridge model
    model = Ridge(alpha=config["alpha"])
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    test_metrics = evaluate_model(y_test, y_pred)
    
    print(f"\n  Ridge Model:")
    print(f"    RMSE: {test_metrics['rmse']:.4f}")
    print(f"    MAE:  {test_metrics['mae']:.4f}")
    print(f"    R²:   {test_metrics['r2']:.4f}")
    
    # Improvement
    improvement = (baseline_metrics['rmse'] - test_metrics['rmse']) / baseline_metrics['rmse'] * 100
    print(f"    Improvement: {improvement:+.2f}%")
    
    # Coefficients
    print(f"\n  Coefficients:")
    for feat, coef in zip(feature_cols, model.coef_):
        print(f"    {feat}: {coef:+.4f}")
    print(f"    Intercept: {model.intercept_:+.4f}")
    
    # Save model
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_var_name = variable.replace("/", "_")
    model_name = f"model_{safe_var_name}_{timestamp}"
    
    # Save pickle
    model_path = os.path.join(output_dir, f"{model_name}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": model,
            "scaler": scaler,
            "variable": variable,
            "feature_cols": feature_cols,
            "metrics": test_metrics,
            "config": config,
            "timestamp": timestamp
        }, f)
    
    # Save metadata JSON
    metadata = {
        "variable": variable,
        "feature_cols": feature_cols,
        "metrics": {k: round(v, 6) for k, v in test_metrics.items()},
        "baseline_rmse": round(baseline_metrics['rmse'], 6),
        "improvement_percent": round(improvement, 2),
        "coefficients": {feat: round(coef, 6) for feat, coef in zip(feature_cols, model.coef_)},
        "intercept": round(float(model.intercept_), 6),
        "alpha": config["alpha"],
        "timestamp": timestamp
    }
    
    metadata_path = os.path.join(output_dir, f"{model_name}_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n  Saved: {os.path.basename(model_path)}")
    
    return {
        "variable": variable,
        "model_path": model_path,
        "metrics": test_metrics,
        "baseline_rmse": baseline_metrics['rmse'],
        "improvement": improvement
    }


def main():
    """Main training function for all 6 Zeus variables."""
    
    print("=" * 70)
    print("Zeus Subnet Model Training (6 Variables)")
    print("=" * 70)
    
    config = CONFIG.copy()
    
    print("\nVariables to train:")
    for i, var in enumerate(ZEUS_VARIABLES, 1):
        print(f"  {i}. {var}")
    
    print(f"\nModel: Ridge Regression (alpha={config['alpha']})")
    print(f"Features: {FEATURE_COLUMNS}")
    print(f"Target: {TARGET_COLUMN}")
    
    # Train model for each variable
    results = []
    
    for variable in ZEUS_VARIABLES:
        result = train_model_for_variable(
            variable,
            config["data_dir"],
            config["output_dir"],
            config
        )
        if result:
            results.append(result)
    
    # Summary
    print("\n" + "=" * 70)
    print("Training Summary")
    print("=" * 70)
    
    if not results:
        print("\n[ERROR] No models were trained successfully.")
        return
    
    print(f"\nModels trained: {len(results)}/{len(ZEUS_VARIABLES)}")
    
    print("\nResults:")
    print(f"{'Variable':<35} {'RMSE':>10} {'R²':>8} {'Improvement':>12}")
    print("-" * 70)
    
    for r in results:
        print(f"{r['variable']:<35} {r['metrics']['rmse']:>10.4f} {r['metrics']['r2']:>8.4f} {r['improvement']:>+11.2f}%")
    
    print("\n" + "=" * 70)
    print("All Models Saved to: models/")
    print("=" * 70)
    
    # List saved files
    print("\nSaved files:")
    for r in results:
        print(f"  - {os.path.basename(r['model_path'])}")
    
    return results


if __name__ == "__main__":
    results = main()
