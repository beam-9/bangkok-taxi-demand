"""Load the saved model and predict future monthly taxi demand."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import BEST_MODEL_PATH, FEATURE_COLUMNS_PATH, MODELING_DATA_PATH  # noqa: E402


def load_artifacts() -> tuple[object, list[str], str]:
    if not BEST_MODEL_PATH.exists() or not FEATURE_COLUMNS_PATH.exists():
        raise FileNotFoundError("Missing model artifacts. Run python src/train_model.py first.")
    artifact = joblib.load(BEST_MODEL_PATH)
    feature_columns = json.loads(FEATURE_COLUMNS_PATH.read_text(encoding="utf-8"))
    return artifact["model"], feature_columns, artifact.get("model_name", "model")


def predict_demand(features: dict[str, float | int]) -> float:
    model, feature_columns, _ = load_artifacts()
    row = pd.DataFrame([{col: features.get(col, 0) for col in feature_columns}])
    return float(model.predict(row)[0])


def example_prediction() -> float:
    if not MODELING_DATA_PATH.exists():
        raise FileNotFoundError("Missing modeling dataset. Run python src/feature_engineering.py first.")
    model, feature_columns, model_name = load_artifacts()
    df = pd.read_csv(MODELING_DATA_PATH, parse_dates=["date"]).sort_values("date")
    latest = df.iloc[-1].copy()
    next_date = latest["date"] + pd.DateOffset(months=1)
    latest["year"] = next_date.year
    latest["month"] = next_date.month
    latest["quarter"] = next_date.quarter
    latest["is_rainy_season"] = int(5 <= next_date.month <= 10)
    latest["is_high_tourism_season"] = int(next_date.month in [11, 12, 1, 2])
    latest["is_songkran_month"] = int(next_date.month == 4)
    latest["is_new_year_month"] = int(next_date.month == 1)
    row = pd.DataFrame([{col: latest.get(col, 0) for col in feature_columns}])
    prediction = float(model.predict(row)[0])
    print(f"Model: {model_name}")
    print(f"Next month: {next_date:%Y-%m}")
    print(f"Predicted taxi demand: {prediction:,.2f}")
    return prediction


if __name__ == "__main__":
    example_prediction()

