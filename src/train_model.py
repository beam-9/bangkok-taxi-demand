"""Train baseline and classical ML demand forecasting models."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import (  # noqa: E402
    BEST_MODEL_PATH,
    FEATURE_COLUMNS_PATH,
    MODELING_DATA_PATH,
    MODEL_RESULTS_PATH,
    TEST_PREDICTIONS_PATH,
    ensure_directories,
)


EXCLUDED_FEATURES = {"date", "target"}


def mape(y_true: pd.Series, y_pred: np.ndarray) -> float:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    mask = y_true_arr != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true_arr[mask] - y_pred_arr[mask]) / y_true_arr[mask])) * 100)


def metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": mape(y_true, y_pred),
        "r2": r2_score(y_true, y_pred) if len(y_true) > 1 else float("nan"),
    }


def feature_columns(df: pd.DataFrame) -> list[str]:
    candidates = [
        col
        for col in df.columns
        if col not in EXCLUDED_FEATURES and pd.api.types.is_numeric_dtype(df[col])
    ]
    return candidates


def train_test_split_time(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = max(1, int(len(df) * 0.8))
    if split_idx >= len(df):
        raise ValueError("Need at least two modeling rows for a train/test split.")
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def seasonal_baseline(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    history = pd.concat([train[["date", "target"]], test[["date", "target"]]], ignore_index=True)
    predictions: list[float] = []
    train_mean = train["target"].mean()
    for date in test["date"]:
        previous_year = date - pd.DateOffset(years=1)
        match = history.loc[history["date"] == previous_year, "target"]
        predictions.append(float(match.iloc[0]) if not match.empty else float(train_mean))
    return np.asarray(predictions)


def xgboost_model() -> Any | None:
    try:
        from xgboost import XGBRegressor
    except Exception as exc:
        print(f"Skipping XGBoost because it is unavailable in this environment: {exc}")
        return None
    return XGBRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
    )


def train_models(input_path: Path = MODELING_DATA_PATH) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Missing {input_path}. Run python src/feature_engineering.py first.")

    df = pd.read_csv(input_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    train, test = train_test_split_time(df)
    features = feature_columns(df)
    if not features:
        raise ValueError("No numeric feature columns available for modeling.")

    X_train, y_train = train[features], train["target"]
    X_test, y_test = test[features], test["target"]

    model_specs: dict[str, Any] = {
        "ridge_regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        min_samples_leaf=2,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }
    xgb = xgboost_model()
    if xgb is not None:
        model_specs["xgboost"] = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", xgb)])

    predictions = pd.DataFrame({"date": test["date"], "actual": y_test})
    result_rows: list[dict[str, float | str]] = []

    baselines = {
        "naive_previous_month": test["target_lag_1"].to_numpy(),
        "seasonal_previous_year": seasonal_baseline(train, test),
    }
    for name, pred in baselines.items():
        result_rows.append({"model": name, **metrics(y_test, pred)})
        predictions[name] = pred

    fitted_models: dict[str, Any] = {}
    for name, model in model_specs.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        result_rows.append({"model": name, **metrics(y_test, pred)})
        predictions[name] = pred
        fitted_models[name] = model

    results = pd.DataFrame(result_rows).sort_values(["mae", "rmse"]).reset_index(drop=True)
    best_model_name = str(results.iloc[0]["model"])
    if best_model_name in fitted_models:
        best_model = fitted_models[best_model_name]
    else:
        best_model_name = min(fitted_models, key=lambda model_name: results.loc[results["model"] == model_name, "mae"].iloc[0])
        best_model = fitted_models[best_model_name]

    ensure_directories()
    results.to_csv(MODEL_RESULTS_PATH, index=False)
    predictions.to_csv(TEST_PREDICTIONS_PATH, index=False)
    joblib.dump({"model_name": best_model_name, "model": best_model}, BEST_MODEL_PATH)
    FEATURE_COLUMNS_PATH.write_text(json.dumps(features, indent=2), encoding="utf-8")

    print("Model results:")
    print(results.to_string(index=False))
    print(f"Saved best ML model: {best_model_name}")
    return results


if __name__ == "__main__":
    train_models()
