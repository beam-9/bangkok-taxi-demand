"""Create monthly modeling features for taxi demand forecasting."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import (  # noqa: E402
    ANALYSIS_END_DATE,
    ANALYSIS_START_DATE,
    CLEAN_TAXI_PATH,
    CLEAN_WEATHER_PATH,
    MODELING_DATA_PATH,
    ensure_directories,
)


def create_features(
    taxi_path: Path = CLEAN_TAXI_PATH,
    weather_path: Path = CLEAN_WEATHER_PATH,
    output_path: Path = MODELING_DATA_PATH,
) -> pd.DataFrame:
    if not taxi_path.exists():
        raise FileNotFoundError(f"Missing {taxi_path}. Run python src/clean_taxi_data.py first.")
    if not weather_path.exists():
        raise FileNotFoundError(f"Missing {weather_path}. Run python src/clean_weather_data.py first.")

    taxi = pd.read_csv(taxi_path, parse_dates=["date"])
    weather = pd.read_csv(weather_path, parse_dates=["date"])
    if "target" not in taxi.columns:
        raise ValueError("Clean taxi data must contain a 'target' column.")

    start_date = pd.Timestamp(ANALYSIS_START_DATE)
    end_date = pd.Timestamp(ANALYSIS_END_DATE)
    taxi = taxi.loc[taxi["date"].between(start_date, end_date)].copy()
    weather = weather.loc[weather["date"].between(start_date, end_date)].copy()

    df = taxi.merge(weather, on="date", how="left").sort_values("date").copy()
    df["target_original"] = df["target"]
    df["target_was_imputed"] = df["target"].isna().astype(int)
    df["target"] = df["target"].interpolate(method="linear", limit_area="inside")
    df = df.dropna(subset=["target"]).copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["is_rainy_season"] = df["month"].between(5, 10).astype(int)
    df["is_high_tourism_season"] = df["month"].isin([11, 12, 1, 2]).astype(int)
    df["is_songkran_month"] = (df["month"] == 4).astype(int)
    df["is_new_year_month"] = (df["month"] == 1).astype(int)

    df["target_lag_1"] = df["target"].shift(1)
    df["target_lag_2"] = df["target"].shift(2)
    df["target_lag_3"] = df["target"].shift(3)
    df["target_rolling_3_mean"] = df["target"].shift(1).rolling(3).mean()
    df["target_rolling_6_mean"] = df["target"].shift(1).rolling(6).mean()
    df["target_pct_change_1"] = df["target"].pct_change(1).shift(1).replace([float("inf"), float("-inf")], pd.NA)

    required_lags = [
        "target_lag_1",
        "target_lag_2",
        "target_lag_3",
        "target_rolling_3_mean",
        "target_rolling_6_mean",
        "target_pct_change_1",
    ]
    df = df.dropna(subset=required_lags + ["target"]).reset_index(drop=True)

    ensure_directories()
    df.to_csv(output_path, index=False)
    print(f"Saved {output_path}")
    return df


if __name__ == "__main__":
    create_features()
