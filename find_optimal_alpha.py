"""
Script to find optimal alpha values for Ridge Regression models using cross-validation.
Tests multiple alpha values and selects the one with the best cross-validation performance.
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from glob import glob
from typing import Dict, List, Tuple, Any

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, make_scorer
import warnings
warnings.filterwarnings('ignore')

from config import TRAINING_CONFIG, ZEUS_VARIABLES, FEATURE_COLUMNS, TARGET_COLUMN


# Alpha values to test (logarithmic scale)
ALPHA_RANGE = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]

# Cross-validation settings
CV_FOLDS = 5  # Number of folds for cross-validation


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
    
    df_clean = df.dropna(subset=existing_cols)
    return df_clean


def prepare_data(df: pd.DataFrame, feature_cols: list[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare data for training (sort by time, scale features)."""
    df = df.sort_values("datetime").reset_index(drop=True)
    
    X = df[feature_cols].values
    y = df[TARGET_COLUMN].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler


def evaluate_alpha(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    cv_folds: int = 5
) -> Dict[str, float]:
    """
    Evaluate a specific alpha value using cross-validation.
    
    Returns:
        Dictionary with mean and std of RMSE scores
    """
    # Use TimeSeriesSplit for time-series data
    tscv = TimeSeriesSplit(n_splits=cv_folds)
    
    # Create Ridge model
    model = Ridge(alpha=alpha)
    
    # Cross-validation with RMSE scoring
    rmse_scorer = make_scorer(
        lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
        greater_is_better=False
    )
    
    scores = cross_val_score(model, X, y, cv=tscv, scoring=rmse_scorer, n_jobs=-1)
    
    # Convert to positive RMSE (since scorer returns negative)
    scores = -scores
    
    return {
        "mean_rmse": np.mean(scores),
        "std_rmse": np.std(scores),
        "min_rmse": np.min(scores),
        "max_rmse": np.max(scores),
        "scores": scores.tolist()
    }


def find_optimal_alpha_for_variable(
    variable: str,
    data_dir: str = "data",
    alpha_range: List[float] = None,
    cv_folds: int = 5
) -> Dict[str, Any]:
    """
    Find optimal alpha for a specific variable.
    
    Returns:
        Dictionary with optimal alpha and evaluation results
    """
    if alpha_range is None:
        alpha_range = ALPHA_RANGE
    
    print(f"\n{'='*70}")
    print(f"Finding Optimal Alpha: {variable}")
    print(f"{'='*70}")
    
    # Find and load data
    file_path = find_integrated_file(variable, data_dir)
    if file_path is None:
        print(f"  [ERROR] No integrated data file found for {variable}")
        return None
    
    print(f"  Data file: {os.path.basename(file_path)}")
    df = load_data(file_path)
    print(f"  Total records: {len(df)}")
    
    # Determine available feature columns
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
    
    if len(df_clean) < cv_folds * 2:
        print(f"  [ERROR] Not enough data for {cv_folds}-fold CV (need at least {cv_folds * 2} records)")
        return None
    
    # Prepare data
    X, y, scaler = prepare_data(df_clean, available_features)
    print(f"  Features: {available_features}")
    
    # Test each alpha value
    print(f"\n  Testing {len(alpha_range)} alpha values...")
    print(f"  Alpha range: {min(alpha_range)} to {max(alpha_range)}")
    
    results = []
    best_alpha = None
    best_rmse = float('inf')
    
    for alpha in alpha_range:
        eval_result = evaluate_alpha(X, y, alpha, cv_folds)
        mean_rmse = eval_result["mean_rmse"]
        
        results.append({
            "alpha": alpha,
            "mean_rmse": mean_rmse,
            "std_rmse": eval_result["std_rmse"],
            "min_rmse": eval_result["min_rmse"],
            "max_rmse": eval_result["max_rmse"]
        })
        
        if mean_rmse < best_rmse:
            best_rmse = mean_rmse
            best_alpha = alpha
        
        print(f"    Alpha {alpha:>6.2f}: RMSE = {mean_rmse:.4f} ± {eval_result['std_rmse']:.4f}")
    
    # Display results
    print(f"\n  {'='*60}")
    print(f"  Results Summary:")
    print(f"  {'='*60}")
    print(f"  Best Alpha: {best_alpha}")
    print(f"  Best RMSE:  {best_rmse:.4f}")
    
    # Show top 3 alphas
    sorted_results = sorted(results, key=lambda x: x["mean_rmse"])
    print(f"\n  Top 3 Alpha Values:")
    for i, result in enumerate(sorted_results[:3], 1):
        marker = "★" if result["alpha"] == best_alpha else " "
        print(f"    {marker} {i}. Alpha {result['alpha']:>6.2f}: RMSE = {result['mean_rmse']:.4f} ± {result['std_rmse']:.4f}")
    
    return {
        "variable": variable,
        "best_alpha": best_alpha,
        "best_rmse": best_rmse,
        "all_results": results,
        "feature_cols": available_features,
        "n_samples": len(df_clean)
    }


def main():
    """Main function to find optimal alpha for all variables."""
    
    print("=" * 70)
    print("Ridge Regression Alpha Optimization")
    print("=" * 70)
    print(f"\nAlpha range to test: {ALPHA_RANGE}")
    print(f"Cross-validation folds: {CV_FOLDS}")
    print(f"Variables: {len(ZEUS_VARIABLES)}")
    
    all_results = []
    
    # Find optimal alpha for each variable
    for variable in ZEUS_VARIABLES:
        result = find_optimal_alpha_for_variable(
            variable,
            TRAINING_CONFIG["data_dir"],
            ALPHA_RANGE,
            CV_FOLDS
        )
        if result:
            all_results.append(result)
    
    # Summary
    print("\n" + "=" * 70)
    print("OPTIMIZATION SUMMARY")
    print("=" * 70)
    
    if not all_results:
        print("\n[ERROR] No results found.")
        return
    
    print(f"\n{'Variable':<35} {'Optimal Alpha':>15} {'CV RMSE':>12} {'Improvement':>12}")
    print("-" * 70)
    
    # Calculate improvement vs default alpha (1.0)
    default_alpha = TRAINING_CONFIG.get("alpha", 1.0)
    
    for result in all_results:
        variable = result["variable"]
        best_alpha = result["best_alpha"]
        best_rmse = result["best_rmse"]
        
        # Find RMSE for default alpha
        default_result = next(
            (r for r in result["all_results"] if r["alpha"] == default_alpha),
            None
        )
        
        if default_result:
            improvement = ((default_result["mean_rmse"] - best_rmse) / default_result["mean_rmse"]) * 100
            improvement_str = f"{improvement:+.2f}%"
        else:
            improvement_str = "N/A"
        
        print(f"{variable:<35} {best_alpha:>15.2f} {best_rmse:>12.4f} {improvement_str:>12}")
    
    # Save detailed results
    output_file = f"alpha_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "alpha_range": ALPHA_RANGE,
            "cv_folds": CV_FOLDS,
            "default_alpha": default_alpha,
            "results": all_results
        }, f, indent=2)
    
    print(f"\n  Results saved to: {output_file}")
    
    # Remove old optimal_alphas.json file if it exists
    optimal_alphas_file = "optimal_alphas.json"
    if os.path.exists(optimal_alphas_file):
        try:
            os.remove(optimal_alphas_file)
            print(f"  Removed old {optimal_alphas_file}")
        except OSError as e:
            print(f"  [WARNING] Could not remove old {optimal_alphas_file}: {e}")
    
    # Save optimal alphas in simple format for training script
    optimal_alphas_dict = {r["variable"]: r["best_alpha"] for r in all_results}
    with open(optimal_alphas_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "optimal_alphas": optimal_alphas_dict
        }, f, indent=2)
    
    print(f"  Optimal alphas saved to: {optimal_alphas_file}")
    
    # Recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    
    # Check if all variables prefer the same alpha
    optimal_alphas = [r["best_alpha"] for r in all_results]
    unique_alphas = set(optimal_alphas)
    
    if len(unique_alphas) == 1:
        common_alpha = optimal_alphas[0]
        print(f"\n  All variables prefer alpha = {common_alpha}")
        print(f"  Recommendation: Update config.py with alpha = {common_alpha}")
    else:
        print(f"\n  Variables prefer different alpha values:")
        for result in all_results:
            print(f"    {result['variable']}: {result['best_alpha']}")
        print(f"\n  Recommendation: Consider variable-specific alpha values")
        print(f"  Or use a common value: {np.median(optimal_alphas):.2f} (median)")
    
    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)
    
    return all_results


if __name__ == "__main__":
    results = main()

