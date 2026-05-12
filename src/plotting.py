"""Shared plotting utilities for reports and Streamlit."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"))

import matplotlib.pyplot as plt
import pandas as pd


def save_demand_trend(df: pd.DataFrame, output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["date"], df["target"], marker="o")
    ax.set_title("Bangkok Taxi Demand Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Taxi demand")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_actual_vs_predicted(predictions: pd.DataFrame, model_column: str, output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(predictions["date"], predictions["actual"], marker="o", label="Actual")
    ax.plot(predictions["date"], predictions[model_column], marker="o", label="Predicted")
    ax.set_title(f"Actual vs Predicted Demand: {model_column}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Taxi demand")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
