"""
Script to collect ERA5 reanalysis data for 6 Zeus subnet variables.
Uses Open-Meteo Archive API which provides access to ERA5 reanalysis data.

ERA5: The fifth generation ECMWF atmospheric reanalysis of the global climate.
Resolution: 0.25° x 0.25° (~25km), Hourly data

Zeus Subnet Required Variables:
1. 2m_temperature (K)
2. 2m_dewpoint_temperature (K)
3. surface_pressure (Pa)
4. total_precipitation (mm)
5. 100m_u_component_of_wind (m/s)
6. 100m_v_component_of_wind (m/s)
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

from config import DATE_CONFIG, LOCATIONS, ZEUS_VARIABLES


# Conversion constants for Zeus Subnet unit requirements
# Zeus Subnet requires: K (Kelvin), Pa (Pascals), mm (millimeters), m/s (meters per second)
CELSIUS_TO_KELVIN = 273.15  # Convert Celsius to Kelvin for temperature variables
HPA_TO_PA = 100.0           # Convert hPa to Pa for pressure (1 hPa = 100 Pa)

# Variables to collect from Open-Meteo ERA5 Archive
OPENMETEO_VARIABLES = [
    "temperature_2m",
    "dew_point_2m",
    "surface_pressure",
    "precipitation",
    "wind_speed_100m",
    "wind_direction_100m",
]


def get_date_range(start_days_ago: int = 12, end_days_ago: int = 5) -> tuple[str, str]:
    """
    Calculate the date range for data collection.
    Default: 12 days ago to 5 days ago (ERA5 has ~5 day delay).
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
        direction: Wind direction in degrees (meteorological convention)
    
    Returns:
        Tuple of (u_component, v_component) in m/s
    """
    if speed is None or direction is None:
        return None, None
    
    direction_rad = np.radians(direction)
    u_component = -speed * np.sin(direction_rad)
    v_component = -speed * np.cos(direction_rad)
    
    return u_component, v_component


def fetch_era5_data(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    timezone: str = "auto"
) -> dict:
    """
    Fetch ERA5 reanalysis data from Open-Meteo Archive API.
    """
    base_url = "https://archive-api.open-meteo.com/v1/archive"
    
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(OPENMETEO_VARIABLES),
        "timezone": timezone,
        "temperature_unit": "celsius",
        "wind_speed_unit": "ms",
        "precipitation_unit": "mm"
    }
    
    response = requests.get(base_url, params=params, timeout=30)
    response.raise_for_status()
    
    return response.json()


def process_to_dataframe(data: dict) -> pd.DataFrame:
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
    
    # Convert units to match Zeus Subnet requirements:
    # - Temperature: Open-Meteo returns Celsius → Convert to Kelvin (K)
    # - Pressure: Open-Meteo returns hPa → Convert to Pascals (Pa)
    # - Precipitation: Open-Meteo returns mm → Already in correct unit (mm)
    # - Wind: Open-Meteo returns m/s → Already in correct unit (m/s), will calculate U/V components
    temp_kelvin = [t + CELSIUS_TO_KELVIN if t is not None else None for t in temp_celsius]
    dewpoint_kelvin = [t + CELSIUS_TO_KELVIN if t is not None else None for t in dewpoint_celsius]
    pressure_pa = [p * HPA_TO_PA if p is not None else None for p in pressure_hpa]
    
    # Calculate U and V wind components (Zeus Subnet requires m/s)
    # U: positive = eastward, negative = westward
    # V: positive = northward, negative = southward
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
    df["latitude"] = data.get("latitude")
    df["longitude"] = data.get("longitude")
    df["timezone"] = data.get("timezone")
    df["elevation"] = data.get("elevation")
    df["data_source"] = "ERA5"
    
    return df


def collect_multi_location_data(
    locations: list[dict],
    start_days_ago: int = 12,
    end_days_ago: int = 5
) -> pd.DataFrame:
    """
    Collect ERA5 data for multiple locations.
    """
    start_date, end_date = get_date_range(start_days_ago, end_days_ago)
    all_data = []
    
    print(f"\nNote: ERA5 data has ~5 day delay. Fetching from {start_date} to {end_date}")
    
    for location in locations:
        try:
            print(f"\nCollecting ERA5 data for: {location['name']}")
            
            data = fetch_era5_data(
                latitude=location["latitude"],
                longitude=location["longitude"],
                start_date=start_date,
                end_date=end_date
            )
            
            df = process_to_dataframe(data)
            df["location_name"] = location["name"]
            all_data.append(df)
            
            print(f"  [OK] Retrieved {len(df)} hourly records")
            
        except requests.exceptions.RequestException as e:
            print(f"  [ERROR] Error fetching data for {location['name']}: {e}")
            continue
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()


def save_data(df: pd.DataFrame, output_dir: str = "data") -> str:
    """
    Save the collected ERA5 data to CSV format.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    csv_filename = f"era5_data_{timestamp}.csv"
    csv_path = os.path.join(output_dir, csv_filename)
    
    # Round numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'float32']).columns
    df[numeric_cols] = df[numeric_cols].round(4)
    
    df.to_csv(csv_path, index=False)
    
    return csv_path


def main():
    """Main function to collect ERA5 data for 6 Zeus variables."""
    
    print("=" * 70)
    print("ERA5 Reanalysis Data Collection (6 Zeus Variables)")
    print("=" * 70)
    print("\nData Source: ERA5 - ECMWF Reanalysis v5")
    print("Resolution: 0.25° x 0.25° (~25km)")
    print("Temporal: Hourly data")
    
    print("\nZeus Variables:")
    for i, var in enumerate(ZEUS_VARIABLES, 1):
        print(f"  {i}. {var}")
    
    # Use config values
    locations = LOCATIONS
    start_days_ago = DATE_CONFIG["start_days_ago"]
    end_days_ago = DATE_CONFIG["end_days_ago"]
    
    print(f"\nLocations: {len(locations)}")
    
    # Collect data
    df = collect_multi_location_data(
        locations,
        start_days_ago=start_days_ago,
        end_days_ago=end_days_ago
    )
    
    if df.empty:
        print("\n[ERROR] No data collected.")
        return
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary Statistics")
    print("=" * 70)
    print(f"Total records: {len(df)}")
    print(f"Locations: {df['location_name'].nunique()}")
    print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    
    print("\nVariable Statistics:")
    for var in ZEUS_VARIABLES:
        if var in df.columns:
            print(f"  {var}:")
            print(f"    min={df[var].min():.2f}, max={df[var].max():.2f}, mean={df[var].mean():.2f}")
    
    # Save data
    print("\n" + "=" * 70)
    print("Saving Data")
    print("=" * 70)
    
    csv_path = save_data(df)
    print(f"  [OK] CSV: {csv_path}")
    
    print("\n" + "=" * 70)
    print("ERA5 Data Collection Complete!")
    print("=" * 70)
    
    return df


if __name__ == "__main__":
    df = main()

