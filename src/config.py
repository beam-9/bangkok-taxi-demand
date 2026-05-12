"""Shared project configuration."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

TAXI_ANALYTICS_URL = (
    "https://otp.gdcatalog.go.th/dataset/bdc9e36a-b7c2-4cac-b36d-be7e4f7eb596/"
    "resource/79dbb88d-337c-4a9e-82b4-72da3ebb593a/download/otp_69_04.csv"
)
TAXI_ANALYTICS_PAGE = "https://datagov.mot.go.th/dataset/otp_69_04"
TAXI_HOTSPOTS_URL = (
    "https://otp.gdcatalog.go.th/dataset/b1b05f94-f1b3-43c6-8aa0-9c28ee836b64/"
    "resource/d6385049-4678-41d4-8633-5e7304dcac77/download/otp_69_05.csv"
)
TAXI_HOTSPOTS_PAGE = "https://datagov.mot.go.th/dataset/otp_69_05"
WEATHER_URL = (
    "https://archive-api.open-meteo.com/v1/archive?"
    "latitude=13.7563&longitude=100.5018&start_date=2023-01-01&end_date=2026-03-31&"
    "hourly=temperature_2m,relative_humidity_2m,precipitation,rain,weather_code&"
    "timezone=Asia%2FBangkok"
)

ANALYSIS_START_DATE = "2023-01-01"
ANALYSIS_END_DATE = "2026-12-31"

RAW_TAXI_ANALYTICS_PATH = RAW_DATA_DIR / "otp_69_04_taxi_analytics.csv"
RAW_TAXI_HOTSPOTS_PATH = RAW_DATA_DIR / "otp_69_05_taxi_hotspots.csv"
RAW_WEATHER_PATH = RAW_DATA_DIR / "bangkok_weather_hourly.csv"

CLEAN_TAXI_PATH = PROCESSED_DATA_DIR / "taxi_analytics_clean.csv"
CLEAN_WEATHER_PATH = PROCESSED_DATA_DIR / "weather_monthly_clean.csv"
MODELING_DATA_PATH = PROCESSED_DATA_DIR / "modeling_dataset.csv"
MODEL_RESULTS_PATH = PROCESSED_DATA_DIR / "model_results.csv"
TEST_PREDICTIONS_PATH = PROCESSED_DATA_DIR / "test_predictions.csv"

BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.json"

TARGET_PRIORITY = [
    "passenger_pickup_trips",
    "avg_trips_per_day",
    "total_passenger_travel_time",
    "total_distance",
    "average_taxis_per_month",
]


def ensure_directories() -> None:
    """Create runtime directories used by scripts."""
    for path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR]:
        path.mkdir(parents=True, exist_ok=True)
