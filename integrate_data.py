"""
Script to integrate forecast data from multiple weather models with ERA5 data.
Creates integrated datasets for each of the 6 Zeus subnet variables.

Output: One integrated CSV per variable with columns:
- location_name, latitude, longitude, timezone, elevation, datetime
- gem_seamless, gfs_seamless, jma_seamless, meteofrance (forecast model values)
- era5 (ground truth)

Zeus Subnet Variables:
1. 2m_temperature (K)
2. 2m_dewpoint_temperature (K)
3. surface_pressure (Pa)
4. total_precipitation (mm)
5. 100m_u_component_of_wind (m/s)
6. 100m_v_component_of_wind (m/s)
"""

import pandas as pd
import os
from glob import glob
from datetime import datetime

from config import ZEUS_VARIABLES

# Models to include
MODELS = [
    "gem_seamless",
    "gfs_seamless",
    "jma_seamless",
    "meteofrance_seamless"
]

# Column name mapping (shorter names for output)
MODEL_COLUMN_NAMES = {
    "gem_seamless": "gem_seamless",
    "gfs_seamless": "gfs_seamless",
    "jma_seamless": "jma_seamless",
    "meteofrance_seamless": "meteofrance"
}


def find_latest_model_files(data_dir: str = "data") -> dict[str, str]:
    """
    Find the latest CSV file for each model in the data directory.
    """
    model_files = {}
    
    for model in MODELS:
        pattern = os.path.join(data_dir, f"forecast_{model}_*.csv")
        files = glob(pattern)
        
        if files:
            latest_file = sorted(files)[-1]
            model_files[model] = latest_file
            print(f"  [OK] {model}: {os.path.basename(latest_file)}")
        else:
            print(f"  [--] {model}: No file found")
    
    return model_files


def find_latest_era5_file(data_dir: str = "data") -> str:
    """
    Find the latest ERA5 CSV file in the data directory.
    """
    pattern = os.path.join(data_dir, "era5_data_*.csv")
    files = glob(pattern)
    
    if files:
        latest_file = sorted(files)[-1]
        print(f"  [OK] era5: {os.path.basename(latest_file)}")
        return latest_file
    else:
        print(f"  [--] era5: No file found")
        return None


def load_model_data(file_path: str, model: str, variable: str, is_first: bool = False) -> pd.DataFrame:
    """
    Load model data for a specific variable and prepare it for merging.
    """
    df = pd.read_csv(file_path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    
    # Rename variable column to model name
    column_name = MODEL_COLUMN_NAMES.get(model, model)
    df = df.rename(columns={variable: column_name})
    
    # Select columns
    if is_first:
        columns_to_keep = [
            "location_name", "latitude", "longitude", "timezone", "elevation",
            "datetime", column_name
        ]
    else:
        columns_to_keep = ["location_name", "datetime", column_name]
    
    # Only keep columns that exist
    columns_to_keep = [c for c in columns_to_keep if c in df.columns]
    
    return df[columns_to_keep]


def load_era5_data(file_path: str, variable: str) -> pd.DataFrame:
    """
    Load ERA5 data for a specific variable and prepare it for merging.
    """
    df = pd.read_csv(file_path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    
    # Rename variable column to 'era5'
    df = df.rename(columns={variable: "era5"})
    
    columns_to_keep = ["location_name", "datetime", "era5"]
    columns_to_keep = [c for c in columns_to_keep if c in df.columns]
    
    return df[columns_to_keep]


def integrate_variable(
    model_files: dict[str, str],
    era5_file: str,
    variable: str
) -> pd.DataFrame:
    """
    Integrate data from multiple models and ERA5 for a single variable.
    """
    if not model_files:
        return pd.DataFrame()
    
    merge_keys = ["location_name", "datetime"]
    integrated_df = None
    is_first = True
    
    # Merge model data
    for model, file_path in model_files.items():
        df = load_model_data(file_path, model, variable, is_first=is_first)
        
        if integrated_df is None:
            integrated_df = df
            is_first = False
        else:
            integrated_df = pd.merge(
                integrated_df,
                df,
                on=merge_keys,
                how="outer"
            )
    
    # Merge ERA5 data
    if era5_file:
        era5_df = load_era5_data(era5_file, variable)
        integrated_df = pd.merge(
            integrated_df,
            era5_df,
            on=merge_keys,
            how="left"
        )
    
    return integrated_df


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reorder columns to standard format.
    """
    column_order = [
        "location_name",
        "latitude",
        "longitude",
        "timezone",
        "elevation",
        "datetime",
        "gem_seamless",
        "gfs_seamless",
        "jma_seamless",
        "meteofrance",
        "era5"
    ]
    
    existing_columns = [col for col in column_order if col in df.columns]
    return df[existing_columns]


def save_integrated_data(df: pd.DataFrame, variable: str, output_dir: str = "data") -> str:
    """
    Save the integrated dataset to CSV format.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create safe filename from variable name
    safe_var_name = variable.replace("/", "_")
    csv_filename = f"integrated_{safe_var_name}_{timestamp}.csv"
    csv_path = os.path.join(output_dir, csv_filename)
    
    # Round numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'float32']).columns
    df[numeric_cols] = df[numeric_cols].round(4)
    
    df.to_csv(csv_path, index=False)
    
    return csv_path


def main():
    """Main function to integrate forecast and ERA5 data for all 6 Zeus variables."""
    
    print("=" * 70)
    print("Data Integration (6 Zeus Variables)")
    print("=" * 70)
    
    data_dir = "data"
    
    # Find latest files
    print("\nSearching for data files...")
    model_files = find_latest_model_files(data_dir)
    era5_file = find_latest_era5_file(data_dir)
    
    if not model_files:
        print("\n[ERROR] No model data files found.")
        return
    
    print(f"\nFound {len(model_files)} model files" + (" + ERA5" if era5_file else ""))
    
    # Integrate each variable
    print("\n" + "=" * 70)
    print("Integrating Variables")
    print("=" * 70)
    
    all_integrated = {}
    
    for variable in ZEUS_VARIABLES:
        print(f"\nProcessing: {variable}")
        
        integrated_df = integrate_variable(model_files, era5_file, variable)
        
        if integrated_df.empty:
            print(f"  [ERROR] No data to integrate for {variable}")
            continue
        
        # Reorder and sort
        integrated_df = reorder_columns(integrated_df)
        integrated_df = integrated_df.sort_values(
            ["location_name", "datetime"]
        ).reset_index(drop=True)
        
        all_integrated[variable] = integrated_df
        print(f"  Records: {len(integrated_df)}")
    
    # Save all integrated data
    print("\n" + "=" * 70)
    print("Saving Integrated Data")
    print("=" * 70)
    
    saved_files = []
    for variable, df in all_integrated.items():
        csv_path = save_integrated_data(df, variable)
        saved_files.append((variable, csv_path))
        print(f"  [OK] {variable}: {os.path.basename(csv_path)}")
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    for variable, df in all_integrated.items():
        print(f"\n{variable}:")
        print(f"  Records: {len(df)}")
        print(f"  Locations: {df['location_name'].nunique()}")
        print(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        
        # Show sample statistics
        model_cols = ["gem_seamless", "gfs_seamless", "jma_seamless", "meteofrance", "era5"]
        existing_cols = [c for c in model_cols if c in df.columns]
        if existing_cols:
            means = df[existing_cols].mean()
            print(f"  Means: " + ", ".join([f"{c}={means[c]:.2f}" for c in existing_cols]))
    
    print("\n" + "=" * 70)
    print("Integration Complete!")
    print("=" * 70)
    
    return all_integrated


if __name__ == "__main__":
    all_data = main()

