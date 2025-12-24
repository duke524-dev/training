"""
Configuration file for Zeus Subnet training pipeline.
All configurable constants are defined here.
"""

# Date range settings for data collection
DATE_CONFIG = {
    "start_days_ago": 12,  # Days ago to start collecting data
    "end_days_ago": 5,      # Days ago to end collecting data (ERA5 has ~5 day delay)
}

# Location settings
LOCATIONS = [
    {"name": "New York", "latitude": 40.7128, "longitude": -74.0060},
    {"name": "London", "latitude": 51.5074, "longitude": -0.1278},
    {"name": "Tokyo", "latitude": 35.6762, "longitude": 139.6503},
    {"name": "Sydney", "latitude": -33.8688, "longitude": 151.2093},
    {"name": "Berlin", "latitude": 52.5200, "longitude": 13.4050},
]

# Model training configuration
TRAINING_CONFIG = {
    "data_dir": "data",
    "output_dir": "models",
    "alpha": 1.0,           # Ridge regularization strength
    "test_size": 0.2,       # Train/test split ratio
    "random_state": 42,     # For reproducibility
}

# Zeus subnet variables
ZEUS_VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "surface_pressure",
    "total_precipitation",
    "100m_u_component_of_wind",
    "100m_v_component_of_wind",
]

# Forecast model columns
FEATURE_COLUMNS = ["gem_seamless", "gfs_seamless", "jma_seamless", "meteofrance"]

# Target column
TARGET_COLUMN = "era5"

