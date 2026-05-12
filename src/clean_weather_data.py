"""Clean and aggregate Open-Meteo hourly weather data to monthly features."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import CLEAN_WEATHER_PATH, RAW_WEATHER_PATH, ensure_directories  # noqa: E402


def clean_weather_data(input_path: Path = RAW_WEATHER_PATH, output_path: Path = CLEAN_WEATHER_PATH) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Missing {input_path}. Run python src/download_data.py first.")

    df = pd.read_csv(input_path)
    if "time" not in df.columns:
        raise ValueError("Weather file must contain a 'time' column.")

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).copy()
    df["date"] = df["time"].dt.to_period("M").dt.to_timestamp()
    df["day"] = df["time"].dt.date
    rain = df["rain"] if "rain" in df.columns else pd.Series(0, index=df.index)
    df["is_rainy_hour"] = rain.fillna(0) > 0
    rainy_days = df.loc[df["is_rainy_hour"]].groupby("date")["day"].nunique().rename("monthly_rainy_days")

    monthly = (
        df.groupby("date")
        .agg(
            monthly_avg_temp=("temperature_2m", "mean"),
            monthly_avg_humidity=("relative_humidity_2m", "mean"),
            monthly_total_precipitation=("precipitation", "sum"),
            monthly_total_rain=("rain", "sum"),
            monthly_rainy_hours=("is_rainy_hour", "sum"),
        )
        .reset_index()
    )
    monthly = monthly.merge(rainy_days.reset_index(), on="date", how="left")
    monthly["monthly_rainy_days"] = monthly["monthly_rainy_days"].fillna(0).astype(int)

    ensure_directories()
    monthly.to_csv(output_path, index=False)
    print(f"Saved {output_path}")
    return monthly


if __name__ == "__main__":
    clean_weather_data()
