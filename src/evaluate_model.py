"""Evaluate trained model outputs and save report figures."""

from __future__ import annotations

import sys
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"))

import matplotlib.pyplot as plt
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import FIGURES_DIR, MODEL_RESULTS_PATH, TEST_PREDICTIONS_PATH, ensure_directories  # noqa: E402


def best_prediction_column(results: pd.DataFrame, predictions: pd.DataFrame) -> str:
    for model_name in results.sort_values(["mae", "rmse"])["model"]:
        if model_name in predictions.columns:
            return str(model_name)
    raise ValueError("Could not find a prediction column matching model_results.csv.")


def evaluate() -> None:
    if not MODEL_RESULTS_PATH.exists():
        raise FileNotFoundError(f"Missing {MODEL_RESULTS_PATH}. Run python src/train_model.py first.")
    if not TEST_PREDICTIONS_PATH.exists():
        raise FileNotFoundError(f"Missing {TEST_PREDICTIONS_PATH}. Run python src/train_model.py first.")

    ensure_directories()
    results = pd.read_csv(MODEL_RESULTS_PATH)
    predictions = pd.read_csv(TEST_PREDICTIONS_PATH, parse_dates=["date"])
    best_col = best_prediction_column(results, predictions)

    print("Ranked model results:")
    print(results.sort_values(["mae", "rmse"]).to_string(index=False))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(predictions["date"], predictions["actual"], marker="o", label="Actual")
    ax.plot(predictions["date"], predictions[best_col], marker="o", label=best_col)
    ax.set_title("Actual vs Predicted Taxi Demand")
    ax.set_xlabel("Date")
    ax.set_ylabel("Taxi demand")
    ax.legend()
    fig.tight_layout()
    output_path = FIGURES_DIR / "actual_vs_predicted.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")

    residuals = predictions["actual"] - predictions[best_col]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axhline(0, color="black", linewidth=1)
    ax.plot(predictions["date"], residuals, marker="o")
    ax.set_title("Forecast Residuals Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Residual")
    fig.tight_layout()
    output_path = FIGURES_DIR / "residuals_over_time.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    evaluate()
