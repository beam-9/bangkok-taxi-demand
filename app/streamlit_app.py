"""Streamlit dashboard for Bangkok taxi demand analytics."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    BEST_MODEL_PATH,
    CLEAN_TAXI_PATH,
    FEATURE_COLUMNS_PATH,
    MODEL_RESULTS_PATH,
    MODELING_DATA_PATH,
    RAW_TAXI_HOTSPOTS_PATH,
    TEST_PREDICTIONS_PATH,
)


st.set_page_config(page_title="Bangkok Taxi Demand", layout="wide")


@st.cache_data
def load_csv(path: Path, parse_dates: list[str] | None = None) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=parse_dates)


def missing_file(path: Path, command: str) -> None:
    st.info(f"Missing `{path.relative_to(PROJECT_ROOT)}`. Run `{command}` to generate it.")


def page_overview(modeling: pd.DataFrame | None) -> None:
    st.title("Bangkok Taxi Demand Forecasting and Mobility Analytics")
    st.write(
        "A reproducible public-data project for analyzing Bangkok taxi mobility patterns "
        "and forecasting next-month passenger demand with classical time-series ML."
    )
    st.subheader("Data sources")
    st.markdown(
        "- OTP Passenger Trip Analytics from GPS taxi data, Thailand Ministry of Transport\n"
        "- OTP Taxi Pickup/Dropoff Top 20 Hotspot Dataset\n"
        "- Open-Meteo Historical Weather API for Bangkok"
    )
    st.caption("This is not Grab/Bolt proprietary data.")
    if modeling is not None and not modeling.empty:
        st.metric("Target variable", "target")
        st.metric("Date range", f"{modeling['date'].min():%Y-%m} to {modeling['date'].max():%Y-%m}")


def page_demand_trends(modeling: pd.DataFrame | None) -> None:
    st.title("Demand Trends")
    if modeling is None:
        missing_file(MODELING_DATA_PATH, "python src/feature_engineering.py")
        return
    st.plotly_chart(px.line(modeling, x="date", y="target", markers=True, title="Historical Taxi Demand"), use_container_width=True)
    monthly = modeling.groupby("month", as_index=False)["target"].mean()
    st.plotly_chart(px.bar(monthly, x="month", y="target", title="Average Demand by Month"), use_container_width=True)
    rolling = modeling[["date", "target"]].copy()
    rolling["rolling_3_month"] = rolling["target"].rolling(3).mean()
    st.plotly_chart(px.line(rolling, x="date", y=["target", "rolling_3_month"], title="Demand and Rolling Average"), use_container_width=True)


def page_weather(modeling: pd.DataFrame | None) -> None:
    st.title("Weather Impact")
    if modeling is None:
        missing_file(MODELING_DATA_PATH, "python src/feature_engineering.py")
        return
    if "monthly_total_rain" in modeling.columns:
        st.plotly_chart(px.scatter(modeling, x="monthly_total_rain", y="target", trendline="ols", title="Rainfall vs Taxi Demand"), use_container_width=True)
    cols = [col for col in ["monthly_avg_temp", "monthly_avg_humidity", "target"] if col in modeling.columns]
    if len(cols) >= 2:
        st.plotly_chart(px.scatter_matrix(modeling, dimensions=cols, title="Temperature, Humidity, and Demand"), use_container_width=True)
    st.write("Interpretation: use these charts to compare weather conditions with monthly taxi demand after running the full pipeline.")


def page_hotspots() -> None:
    st.title("Hotspot Analysis")
    hotspots = load_csv(RAW_TAXI_HOTSPOTS_PATH)
    if hotspots is None:
        missing_file(RAW_TAXI_HOTSPOTS_PATH, "python src/download_data.py")
        return
    st.dataframe(hotspots.head(50), use_container_width=True)
    numeric_cols = hotspots.select_dtypes("number").columns.tolist()
    text_cols = hotspots.select_dtypes("object").columns.tolist()
    if numeric_cols and text_cols:
        metric = numeric_cols[0]
        label = text_cols[0]
        ranked = hotspots.sort_values(metric, ascending=False).head(20)
        st.plotly_chart(px.bar(ranked, x=metric, y=label, orientation="h", title="Top Hotspot Locations"), use_container_width=True)
    lat_cols = [col for col in hotspots.columns if "lat" in col.lower() or "latitude" in col.lower()]
    lon_cols = [col for col in hotspots.columns if "lon" in col.lower() or "lng" in col.lower() or "longitude" in col.lower()]
    if lat_cols and lon_cols:
        st.map(hotspots.rename(columns={lat_cols[0]: "lat", lon_cols[0]: "lon"}).dropna(subset=["lat", "lon"]))


def page_model_performance() -> None:
    st.title("Model Performance")
    results = load_csv(MODEL_RESULTS_PATH)
    predictions = load_csv(TEST_PREDICTIONS_PATH, parse_dates=["date"])
    if results is None:
        missing_file(MODEL_RESULTS_PATH, "python src/train_model.py")
        return
    st.dataframe(results, use_container_width=True)
    best = results.sort_values(["mae", "rmse"]).iloc[0]["model"]
    st.metric("Best model by MAE", best)
    if predictions is not None and best in predictions.columns:
        st.plotly_chart(px.line(predictions, x="date", y=["actual", best], markers=True, title="Actual vs Predicted"), use_container_width=True)
    if BEST_MODEL_PATH.exists() and FEATURE_COLUMNS_PATH.exists():
        artifact = joblib.load(BEST_MODEL_PATH)
        model = artifact["model"]
        feature_columns = json.loads(FEATURE_COLUMNS_PATH.read_text(encoding="utf-8"))
        estimator = model.named_steps.get("model") if hasattr(model, "named_steps") else model
        if hasattr(estimator, "feature_importances_"):
            importance = pd.DataFrame({"feature": feature_columns, "importance": estimator.feature_importances_}).sort_values("importance", ascending=False).head(20)
            st.plotly_chart(px.bar(importance, x="importance", y="feature", orientation="h", title="Feature Importance"), use_container_width=True)


def page_forecast(modeling: pd.DataFrame | None) -> None:
    st.title("Forecast Tool")
    if not BEST_MODEL_PATH.exists() or not FEATURE_COLUMNS_PATH.exists():
        missing_file(BEST_MODEL_PATH, "python src/train_model.py")
        return
    feature_columns = json.loads(FEATURE_COLUMNS_PATH.read_text(encoding="utf-8"))
    artifact = joblib.load(BEST_MODEL_PATH)
    model = artifact["model"]
    latest = modeling.iloc[-1] if modeling is not None and not modeling.empty else pd.Series(dtype=float)

    month = st.selectbox("Month", list(range(1, 13)), index=0)
    rainfall = st.number_input("Rainfall", min_value=0.0, value=float(latest.get("monthly_total_rain", 100.0)))
    avg_temp = st.number_input("Average temperature", value=float(latest.get("monthly_avg_temp", 29.0)))
    humidity = st.number_input("Humidity", min_value=0.0, max_value=100.0, value=float(latest.get("monthly_avg_humidity", 75.0)))
    previous_demand = st.number_input("Previous month demand", min_value=0.0, value=float(latest.get("target", 0.0)))

    row = {col: float(latest.get(col, 0)) for col in feature_columns}
    row.update(
        {
            "month": month,
            "quarter": ((month - 1) // 3) + 1,
            "is_rainy_season": int(5 <= month <= 10),
            "is_high_tourism_season": int(month in [11, 12, 1, 2]),
            "is_songkran_month": int(month == 4),
            "is_new_year_month": int(month == 1),
            "monthly_total_rain": rainfall,
            "monthly_total_precipitation": rainfall,
            "monthly_avg_temp": avg_temp,
            "monthly_avg_humidity": humidity,
            "target_lag_1": previous_demand,
        }
    )
    if st.button("Predict demand"):
        prediction = float(model.predict(pd.DataFrame([{col: row.get(col, 0) for col in feature_columns}]))[0])
        st.metric("Predicted taxi demand", f"{prediction:,.2f}")


def main() -> None:
    modeling = load_csv(MODELING_DATA_PATH, parse_dates=["date"])
    page = st.sidebar.radio(
        "Page",
        [
            "Project Overview",
            "Demand Trends",
            "Weather Impact",
            "Hotspot Analysis",
            "Model Performance",
            "Forecast Tool",
        ],
    )
    if page == "Project Overview":
        page_overview(modeling)
    elif page == "Demand Trends":
        page_demand_trends(modeling)
    elif page == "Weather Impact":
        page_weather(modeling)
    elif page == "Hotspot Analysis":
        page_hotspots()
    elif page == "Model Performance":
        page_model_performance()
    elif page == "Forecast Tool":
        page_forecast(modeling)


if __name__ == "__main__":
    main()

