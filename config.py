"""
Configuration file for Zeus Subnet training pipeline.
All configurable constants are defined here.
"""

# Date range settings for data collection
DATE_CONFIG = {
    "start_days_ago": 370,  # Days ago to start collecting data
    "end_days_ago": 5,      # Days ago to end collecting data (ERA5 has ~5 day delay)
}

# Location settings
LOCATIONS = [
    {"name": "New York", "latitude": 40.7128, "longitude": -74.0060},
    {"name": "London", "latitude": 51.5074, "longitude": -0.1278},
    {"name": "Tokyo", "latitude": 35.6762, "longitude": 139.6503},
    {"name": "Sydney", "latitude": -33.8688, "longitude": 151.2093},
    {"name": "Berlin", "latitude": 52.5200, "longitude": 13.4050},
    {"name": "Beijing", "latitude": 39.9042, "longitude": 116.4074},
    {"name": "Moscow", "latitude": 55.7558, "longitude": 37.6173},
    {"name": "Mexico City", "latitude": 19.4326, "longitude": -99.1332},
    {"name": "Taipei", "latitude": 25.0330, "longitude": 121.5654},
    {"name": "New Delhi", "latitude": 28.6139, "longitude": 77.2090},
    {"name": "Cairo", "latitude": 30.0444, "longitude": 31.2357},
    {"name": "Brasilia", "latitude": -15.7942, "longitude": -47.8822},
    {"name": "Mawsynram", "latitude": 25.3000, "longitude": 91.5833},
    {"name": "Cherrapunji", "latitude": 25.3000, "longitude": 91.7000},
    {"name": "Debundscha", "latitude": 4.1000, "longitude": 9.0167},
    {"name": "Quibdó", "latitude": 5.6900, "longitude": -76.6600},
    {"name": "Milford Sound", "latitude": -44.6417, "longitude": 167.9000},
    {"name": "Vancouver Island", "latitude": 48.4284, "longitude": -123.3656},
    {"name": "Amazon Basin", "latitude": -3.4653, "longitude": -62.2159},
    {"name": "Hilo", "latitude": 19.7297, "longitude": -155.0900},
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

