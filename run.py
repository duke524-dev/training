"""
Main pipeline script to run all steps in order.

Steps:
1. Collect forecast data from multiple weather models (6 variables)
2. Collect ERA5 observation data (6 variables)
3. Integrate all data into single datasets
4. Train ensemble models (6 models, one per variable)

Zeus Subnet Variables:
- 2m_temperature
- 2m_dewpoint_temperature
- surface_pressure
- total_precipitation
- 100m_u_component_of_wind
- 100m_v_component_of_wind
"""

import subprocess
import sys
import time
from datetime import datetime
import os
from glob import glob
import re


def run_step(step_num: int, script_name: str, description: str) -> bool:
    """
    Run a single pipeline step (blocking).
    
    Args:
        step_num: Step number for display
        script_name: Python script to run
        description: Description of what the step does
    
    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*70}")
    print(f"STEP {step_num}: {description}")
    print(f"Script: {script_name}")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            text=True
        )
        
        elapsed = time.time() - start_time
        print(f"\n[OK] Step {step_num} completed in {elapsed:.1f} seconds")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Step {step_num} failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\n[ERROR] Script not found: {script_name}")
        return False


def _keep_latest_by_prefix(directory: str, prefixes: list[str], extension: str = ".csv") -> list[str]:
    """Keep only the newest file per prefix, delete older ones. Returns kept files."""
    kept = []
    for prefix in prefixes:
        pattern = os.path.join(directory, f"{prefix}*{extension}")
        files = sorted(glob(pattern))
        if not files:
            continue
        latest = files[-1]
        kept.append(latest)
        for old in files[:-1]:
            try:
                os.remove(old)
            except OSError:
                pass
    return kept


def _cleanup_models(directory: str = "models") -> list[str]:
    """Keep only the newest .pkl per variable; delete all metadata json."""
    kept = []

    # Delete all metadata JSON files
    for meta in glob(os.path.join(directory, "model_*_*.json")):
        try:
            os.remove(meta)
        except OSError:
            pass

    pattern = os.path.join(directory, "model_*_*.pkl")
    files = glob(pattern)
    grouped = {}
    for f in files:
        name = os.path.basename(f)
        m = re.match(r"(model_.+)_([0-9]{8}_[0-9]{6})\.pkl", name)
        if not m:
            continue
        base = m.group(1)
        ts = m.group(2)
        grouped.setdefault(base, []).append((ts, f))

    for base, items in grouped.items():
        items.sort(key=lambda x: x[0])
        newest_ts, newest_file = items[-1]
        kept.append(newest_file)
        for ts, f in items[:-1]:
            try:
                os.remove(f)
            except OSError:
                pass

    return kept


def cleanup_outputs():
    """Remove old data/model artifacts, keep only the latest per prefix."""
    print("\n" + "=" * 70)
    print("Cleanup")
    print("=" * 70)

    # Data cleanup
    data_dir = "data"
    data_prefixes = [
        "forecast_gem_seamless_",
        "forecast_gfs_seamless_",
        "forecast_jma_seamless_",
        "forecast_meteofrance_seamless_",
        "era5_data_",
        "integrated_2m_temperature_",
        "integrated_2m_dewpoint_temperature_",
        "integrated_surface_pressure_",
        "integrated_total_precipitation_",
        "integrated_100m_u_component_of_wind_",
        "integrated_100m_v_component_of_wind_",
    ]
    kept_data = _keep_latest_by_prefix(data_dir, data_prefixes, ".csv")
    print("Kept data files:")
    for f in kept_data:
        print(f"  - {os.path.basename(f)}")

    # Models cleanup
    kept_models = _cleanup_models("models")
    print("\nKept model files:")
    for f in kept_models:
        print(f"  - {os.path.basename(f)}")


def main():
    """Run the complete pipeline."""
    
    print("="*70)
    print("ZEUS SUBNET PIPELINE (6 Variables)")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    print("\nVariables:")
    print("  1. 2m_temperature")
    print("  2. 2m_dewpoint_temperature")
    print("  3. surface_pressure")
    print("  4. total_precipitation")
    print("  5. 100m_u_component_of_wind")
    print("  6. 100m_v_component_of_wind")
    
    # Define pipeline steps
    steps = [
        ("collect_forecast_data.py", "Collect forecast data (6 variables, 4 models)"),
        ("collect_era5_data.py", "Collect ERA5 observation data (6 variables)"),
        ("integrate_data.py", "Integrate all data into datasets"),
        ("find_optimal_alpha.py", "Find optimal alpha values for each variable"),
        ("train_model.py", "Train 6 ensemble models with optimized alphas"),
    ]
    
    total_start = time.time()
    results = []
    
    # Run each step sequentially (blocking)
    for i, (script, description) in enumerate(steps, 1):
        success = run_step(i, script, description)
        results.append((script, success))
        
        if not success:
            print(f"\n[ABORT] Pipeline stopped due to error in step {i}")
            break
    
    # Summary
    total_elapsed = time.time() - total_start
    
    print("\n" + "="*70)
    print("PIPELINE SUMMARY")
    print("="*70)
    
    for i, (script, success) in enumerate(results, 1):
        status = "[OK]" if success else "[FAILED]"
        print(f"  Step {i}: {status} {script}")
    
    successful = sum(1 for _, s in results if s)
    total = len(steps)
    
    print(f"\nCompleted: {successful}/{total} steps")
    print(f"Total time: {total_elapsed:.1f} seconds")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if successful == total:
        print("\n[SUCCESS] Pipeline completed successfully!")
        print("\nOutput:")
        print("  - data/integrated_*.csv (6 integrated datasets)")
        print("  - models/model_*.pkl (6 trained models)")

        # Cleanup old artifacts, keep latest
        cleanup_outputs()
        return 0
    else:
        print("\n[FAILED] Pipeline completed with errors")
        return 1


if __name__ == "__main__":
    sys.exit(main())
