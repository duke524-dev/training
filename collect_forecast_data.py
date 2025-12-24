"""
Script to collect historical forecast data for 6 Zeus subnet variables from multiple weather models.
Uses Open-Meteo Historical Forecast API for accessing archived weather forecasts.

Zeus Subnet Required Variables:
1. 2m_temperature (K)
2. 2m_dewpoint_temperature (K)
3. surface_pressure (Pa)
4. total_precipitation (mm)
5. 100m_u_component_of_wind (m/s)
6. 100m_v_component_of_wind (m/s)

Supported Models:
- GFS (Global Forecast System - NOAA)
- JMA (Japan Meteorological Agency)
- GEM (Environment Canada)
- Météo-France ARPEGE
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

from config import DATE_CONFIG, LOCATIONS, ZEUS_VARIABLES


# Conversion constants
CELSIUS_TO_KELVIN = 273.15
HPA_TO_PA = 100.0

# Variables to collect from Open-Meteo
OPENMETEO_VARIABLES = [
    "temperature_2m",
    "dew_point_2m",
    "surface_pressure",
    "precipitation",
    "wind_speed_100m",
    "wind_direction_100m",
]

# Available weather models from Open-Meteo Historical Forecast API
AVAILABLE_MODELS = {
    "gfs_seamless": {
        "name": "GFS",
        "provider": "NOAA (USA)",
        "resolution": "0.25° (~25km)",
    },
    "jma_seamless": {
        "name": "JMA",
        "provider": "Japan Meteorological Agency",
        "resolution": "0.2° (~20km)",
    },
    "gem_seamless": {
        "name": "GEM",
        "provider": "Environment Canada",
        "resolution": "0.25° (~25km)",
    },
    "meteofrance_seamless": {
        "name": "ARPEGE",
        "provider": "Météo-France",
        "resolution": "0.25° (~25km)",
    }
}


def get_date_range(start_days_ago: int = 12, end_days_ago: int = 5) -> tuple[str, str]:
    """
    Calculate the date range for data collection.
    Default: 12 days ago to 5 days ago (aligned with ERA5 availability).
    """
    today = datetime.now()
    start_date = today - timedelta(days=start_days_ago)
    end_date = today - timedelta(days=end_days_ago)
    
    return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")


def calculate_wind_components(speed: float, direction: float) -> tuple[float, float]:
    """
    Calculate U and V wind components from speed and direction.
    
    Args:
        speed: Wind speed in m/s
        direction: Wind direction in degrees (meteorological convention: from where wind blows)
    
    Returns:
        Tuple of (u_component, v_component) in m/s
        U: positive = eastward, V: positive = northward
    """
    if speed is None or direction is None:
        return None, None
    
    direction_rad = np.radians(direction)
    u_component = -speed * np.sin(direction_rad)
    v_component = -speed * np.cos(direction_rad)
    
    return u_component, v_component


def fetch_historical_forecast_data(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    model: str,
    timezone: str = "auto"
) -> dict:
    """
    Fetch historical forecast data from Open-Meteo API.
    """
    base_url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(OPENMETEO_VARIABLES),
        "timezone": timezone,
        "temperature_unit": "celsius",
        "wind_speed_unit": "ms",
        "precipitation_unit": "mm",
        "models": model
    }
    
    response = requests.get(base_url, params=params, timeout=30)
    response.raise_for_status()
    
    return response.json()


def process_to_dataframe(data: dict, model: str) -> pd.DataFrame:
    """
    Process the API response into a pandas DataFrame with Zeus variable names.
    Converts units and calculates wind components.
    """
    hourly_data = data.get("hourly", {})
    
    # Get raw data
    times = hourly_data.get("time", [])
    temp_celsius = hourly_data.get("temperature_2m", [])
    dewpoint_celsius = hourly_data.get("dew_point_2m", [])
    pressure_hpa = hourly_data.get("surface_pressure", [])
    precipitation = hourly_data.get("precipitation", [])
    wind_speed = hourly_data.get("wind_speed_100m", [])
    wind_direction = hourly_data.get("wind_direction_100m", [])
    
    # Convert units and calculate wind components
    temp_kelvin = [t + CELSIUS_TO_KELVIN if t is not None else None for t in temp_celsius]
    dewpoint_kelvin = [t + CELSIUS_TO_KELVIN if t is not None else None for t in dewpoint_celsius]
    pressure_pa = [p * HPA_TO_PA if p is not None else None for p in pressure_hpa]
    
    # Calculate U and V wind components
    u_components = []
    v_components = []
    for speed, direction in zip(wind_speed, wind_direction):
        u, v = calculate_wind_components(speed, direction)
        u_components.append(u)
        v_components.append(v)
    
    df = pd.DataFrame({
        "datetime": pd.to_datetime(times),
        "2m_temperature": temp_kelvin,
        "2m_dewpoint_temperature": dewpoint_kelvin,
        "surface_pressure": pressure_pa,
        "total_precipitation": precipitation,
        "100m_u_component_of_wind": u_components,
        "100m_v_component_of_wind": v_components,
    })
    
    # Add metadata columns
    df["model"] = model
    df["model_name"] = AVAILABLE_MODELS.get(model, {}).get("name", model)
    df["latitude"] = data.get("latitude")
    df["longitude"] = data.get("longitude")
    df["timezone"] = data.get("timezone")
    df["elevation"] = data.get("elevation")
    
    return df


def collect_data_for_model(
    model: str,
    locations: list[dict],
    start_days_ago: int = 12,
    end_days_ago: int = 5
) -> pd.DataFrame:
    """
    Collect historical forecast data for a specific model across multiple locations.
    """
    start_date, end_date = get_date_range(start_days_ago, end_days_ago)
    all_data = []
    
    for location in locations:
        try:
            data = fetch_historical_forecast_data(
                latitude=location["latitude"],
                longitude=location["longitude"],
                start_date=start_date,
                end_date=end_date,
                model=model
            )
            
            df = process_to_dataframe(data, model)
            df["location_name"] = location["name"]
            all_data.append(df)
            
            print(f"    [OK] {location['name']}: {len(df)} records")
            
        except requests.exceptions.RequestException as e:
            print(f"    [ERROR] {location['name']}: {e}")
            continue
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()


def save_model_data(df: pd.DataFrame, model: str, output_dir: str = "data") -> str:
    """
    Save the collected data for a specific model to CSV format.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    csv_filename = f"forecast_{model}_{timestamp}.csv"
    csv_path = os.path.join(output_dir, csv_filename)
    
    # Round numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'float32']).columns
    df[numeric_cols] = df[numeric_cols].round(4)
    
    df.to_csv(csv_path, index=False)
    
    return csv_path


def collect_all_models(
    locations: list[dict],
    models: list[str] = None,
    start_days_ago: int = 12,
    end_days_ago: int = 5
) -> dict[str, pd.DataFrame]:
    """
    Collect data from all specified models.
    """
    if models is None:
        models = list(AVAILABLE_MODELS.keys())
    
    start_date, end_date = get_date_range(start_days_ago, end_days_ago)
    
    print(f"\nFetching data from {start_date} to {end_date}")
    print(f"Models to collect: {len(models)}")
    print(f"Locations: {len(locations)}")
    print(f"Variables: {len(ZEUS_VARIABLES)}")
    print(f"  - {', '.join(ZEUS_VARIABLES)}")
    
    all_model_data = {}
    
    for model in models:
        model_info = AVAILABLE_MODELS.get(model, {})
        model_name = model_info.get("name", model)
        
        print(f"\n{'='*60}")
        print(f"Collecting: {model_name} ({model})")
        print(f"Provider: {model_info.get('provider', 'Unknown')}")
        print(f"{'='*60}")
        
        df = collect_data_for_model(
            model=model,
            locations=locations,
            start_days_ago=start_days_ago,
            end_days_ago=end_days_ago
        )
        
        if not df.empty:
            all_model_data[model] = df
            print(f"\n  Total: {len(df)} records")
        else:
            print(f"  [ERROR] No data retrieved for {model_name}")
    
    return all_model_data


def main():
    """Main function to collect forecast data for 6 Zeus variables."""
    
    print("=" * 70)
    print("Multi-Model Forecast Data Collection (6 Zeus Variables)")
    print("=" * 70)
    
    print("\nZeus Variables:")
    for i, var in enumerate(ZEUS_VARIABLES, 1):
        print(f"  {i}. {var}")
    
    print("\nAvailable Models:")
    for model_id, info in AVAILABLE_MODELS.items():
        print(f"  - {info['name']} ({model_id})")
    
    # Use config values
    locations = LOCATIONS
    start_days_ago = DATE_CONFIG["start_days_ago"]
    end_days_ago = DATE_CONFIG["end_days_ago"]
    
    # Models to collect
    models_to_collect = list(AVAILABLE_MODELS.keys())
    
    print(f"\nLocations: {len(locations)}")
    print(f"Models: {len(models_to_collect)}")
    
    # Collect data
    all_model_data = collect_all_models(
        locations=locations,
        models=models_to_collect,
        start_days_ago=start_days_ago,
        end_days_ago=end_days_ago
    )
    
    if not all_model_data:
        print("\n[ERROR] No data collected.")
        return
    
    # Save data
    print("\n" + "=" * 70)
    print("Saving Data")
    print("=" * 70)
    
    for model, df in all_model_data.items():
        model_name = AVAILABLE_MODELS.get(model, {}).get("name", model)
        csv_path = save_model_data(df, model)
        print(f"  [OK] {model_name}: {csv_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    for model, df in all_model_data.items():
        model_name = AVAILABLE_MODELS.get(model, {}).get("name", model)
        print(f"\n{model_name}:")
        print(f"  Records: {len(df)}")
        for var in ZEUS_VARIABLES:
            if var in df.columns:
                print(f"  {var}: min={df[var].min():.2f}, max={df[var].max():.2f}, mean={df[var].mean():.2f}")
    
    print("\n" + "=" * 70)
    print("Data Collection Complete!")
    print("=" * 70)
    
    return all_model_data


if __name__ == "__main__":
    all_data = main()

