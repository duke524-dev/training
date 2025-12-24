"""
Script to view and display model accuracy metrics from saved model files.
Shows RMSE, MAE, R², Bias, and improvement over baseline for all trained models.
"""

import pickle
import os
from glob import glob
from datetime import datetime
from config import ZEUS_VARIABLES


def load_model_metrics(model_file: str) -> dict:
    """Load metrics from a saved model file."""
    try:
        with open(model_file, "rb") as f:
            model_data = pickle.load(f)
        
        return {
            "variable": model_data.get("variable", "unknown"),
            "metrics": model_data.get("metrics", {}),
            "timestamp": model_data.get("timestamp", "unknown"),
            "file": os.path.basename(model_file)
        }
    except Exception as e:
        print(f"Error loading {model_file}: {e}")
        return None


def view_model_metrics(models_dir: str = "models", sort_by: str = "variable"):
    """
    Display metrics for all saved models.
    
    Args:
        models_dir: Directory containing model files
        sort_by: How to sort results ("variable", "rmse", "r2")
    """
    model_files = glob(os.path.join(models_dir, "model_*.pkl"))
    
    if not model_files:
        print("=" * 80)
        print("No model files found!")
        print(f"Looking in: {os.path.abspath(models_dir)}")
        print("=" * 80)
        return
    
    print("=" * 80)
    print("MODEL ACCURACY METRICS")
    print("=" * 80)
    print(f"Found {len(model_files)} model file(s)")
    print(f"Directory: {os.path.abspath(models_dir)}")
    print("=" * 80)
    
    # Load all model data
    all_models = []
    for model_file in model_files:
        model_data = load_model_metrics(model_file)
        if model_data:
            all_models.append(model_data)
    
    if not all_models:
        print("\nNo valid model data found!")
        return
    
    # Sort models
    if sort_by == "variable":
        all_models.sort(key=lambda x: ZEUS_VARIABLES.index(x["variable"]) if x["variable"] in ZEUS_VARIABLES else 999)
    elif sort_by == "rmse":
        all_models.sort(key=lambda x: x["metrics"].get("rmse", float('inf')))
    elif sort_by == "r2":
        all_models.sort(key=lambda x: x["metrics"].get("r2", -1), reverse=True)
    
    # Display summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Variable':<35} {'RMSE':>12} {'MAE':>12} {'R²':>8} {'Bias':>12}")
    print("-" * 80)
    
    for model in all_models:
        metrics = model["metrics"]
        var_name = model["variable"]
        rmse = metrics.get("rmse", 0)
        mae = metrics.get("mae", 0)
        r2 = metrics.get("r2", 0)
        bias = metrics.get("bias", 0)
        
        print(f"{var_name:<35} {rmse:>12.4f} {mae:>12.4f} {r2:>8.4f} {bias:>12.4f}")
    
    # Display detailed metrics for each model
    print("\n" + "=" * 80)
    print("DETAILED METRICS")
    print("=" * 80)
    
    for model in all_models:
        print(f"\n{model['variable']}:")
        print(f"  File: {model['file']}")
        print(f"  Timestamp: {model['timestamp']}")
        
        metrics = model["metrics"]
        print(f"  RMSE: {metrics.get('rmse', 0):.6f} (Root Mean Squared Error - lower is better)")
        print(f"  MAE:  {metrics.get('mae', 0):.6f} (Mean Absolute Error - lower is better)")
        print(f"  R²:   {metrics.get('r2', 0):.6f} (Coefficient of Determination - closer to 1.0 is better)")
        print(f"  Bias: {metrics.get('bias', 0):.6f} (Average prediction bias - closer to 0 is better)")
    
    # Statistics summary
    print("\n" + "=" * 80)
    print("STATISTICS SUMMARY")
    print("=" * 80)
    
    if all_models:
        rmse_values = [m["metrics"].get("rmse", 0) for m in all_models]
        mae_values = [m["metrics"].get("mae", 0) for m in all_models]
        r2_values = [m["metrics"].get("r2", 0) for m in all_models]
        
        print(f"\nRMSE Statistics:")
        print(f"  Average: {sum(rmse_values) / len(rmse_values):.4f}")
        print(f"  Min:     {min(rmse_values):.4f} ({all_models[rmse_values.index(min(rmse_values))]['variable']})")
        print(f"  Max:     {max(rmse_values):.4f} ({all_models[rmse_values.index(max(rmse_values))]['variable']})")
        
        print(f"\nMAE Statistics:")
        print(f"  Average: {sum(mae_values) / len(mae_values):.4f}")
        print(f"  Min:     {min(mae_values):.4f} ({all_models[mae_values.index(min(mae_values))]['variable']})")
        print(f"  Max:     {max(mae_values):.4f} ({all_models[mae_values.index(max(mae_values))]['variable']})")
        
        print(f"\nR² Statistics:")
        print(f"  Average: {sum(r2_values) / len(r2_values):.4f}")
        print(f"  Min:     {min(r2_values):.4f} ({all_models[r2_values.index(min(r2_values))]['variable']})")
        print(f"  Max:     {max(r2_values):.4f} ({all_models[r2_values.index(max(r2_values))]['variable']})")
    
    print("\n" + "=" * 80)
    print("METRICS VIEWER COMPLETE")
    print("=" * 80)


def view_single_model(model_file: str):
    """View detailed metrics for a single model file."""
    if not os.path.exists(model_file):
        print(f"Error: File not found: {model_file}")
        return
    
    model_data = load_model_metrics(model_file)
    if not model_data:
        return
    
    print("=" * 80)
    print("MODEL DETAILS")
    print("=" * 80)
    print(f"\nVariable: {model_data['variable']}")
    print(f"File: {model_data['file']}")
    print(f"Timestamp: {model_data['timestamp']}")
    
    metrics = model_data["metrics"]
    print(f"\nMetrics:")
    print(f"  RMSE: {metrics.get('rmse', 0):.6f}")
    print(f"  MAE:  {metrics.get('mae', 0):.6f}")
    print(f"  R²:   {metrics.get('r2', 0):.6f}")
    print(f"  Bias: {metrics.get('bias', 0):.6f}")
    print("=" * 80)


def main():
    """Main function."""
    import sys
    
    if len(sys.argv) > 1:
        # View specific model file
        model_file = sys.argv[1]
        view_single_model(model_file)
    else:
        # View all models
        sort_by = sys.argv[2] if len(sys.argv) > 2 else "variable"
        view_model_metrics(sort_by=sort_by)


if __name__ == "__main__":
    main()

