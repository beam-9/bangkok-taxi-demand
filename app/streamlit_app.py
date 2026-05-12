"""Streamlit dashboard for Bangkok taxi demand analytics."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    BEST_MODEL_PATH,
    CLEAN_TAXI_PATH,
    CLEAN_WEATHER_PATH,
    FEATURE_COLUMNS_PATH,
    MODEL_RESULTS_PATH,
    MODELING_DATA_PATH,
    RAW_TAXI_ANALYTICS_PATH,
    RAW_TAXI_HOTSPOTS_PATH,
    RAW_WEATHER_PATH,
    TEST_PREDICTIONS_PATH,
)


st.set_page_config(page_title="Bangkok Taxi Demand", layout="wide")

DEPLOY_ARTIFACTS_DIR = PROJECT_ROOT / "deploy_artifacts"
DEPLOY_HOTSPOT_SUMMARY_PATH = DEPLOY_ARTIFACTS_DIR / "hotspot_summary.csv"


@st.cache_data
def load_csv(path: Path, parse_dates: list[str] | None = None) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=parse_dates)


def missing_file(path: Path, command: str) -> None:
    st.info(f"Missing `{path.relative_to(PROJECT_ROOT)}`. Run `{command}` to generate it.")


def restore_bundled_artifacts() -> None:
    """Restore small committed artifacts for deployments where OTP downloads time out."""
    artifact_map = {
        DEPLOY_ARTIFACTS_DIR / "taxi_analytics_clean.csv": CLEAN_TAXI_PATH,
        DEPLOY_ARTIFACTS_DIR / "weather_monthly_clean.csv": CLEAN_WEATHER_PATH,
        DEPLOY_ARTIFACTS_DIR / "modeling_dataset.csv": MODELING_DATA_PATH,
        DEPLOY_ARTIFACTS_DIR / "model_results.csv": MODEL_RESULTS_PATH,
        DEPLOY_ARTIFACTS_DIR / "test_predictions.csv": TEST_PREDICTIONS_PATH,
        DEPLOY_ARTIFACTS_DIR / "best_model.joblib": BEST_MODEL_PATH,
        DEPLOY_ARTIFACTS_DIR / "feature_columns.json": FEATURE_COLUMNS_PATH,
    }
    for source, destination in artifact_map.items():
        if source.exists() and not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)


@st.cache_resource(show_spinner=False)
def ensure_dashboard_artifacts() -> tuple[bool, str]:
    """Create deploy-time data/model artifacts if ignored files are absent."""
    required_outputs = [
        CLEAN_TAXI_PATH,
        CLEAN_WEATHER_PATH,
        MODELING_DATA_PATH,
        MODEL_RESULTS_PATH,
        TEST_PREDICTIONS_PATH,
        BEST_MODEL_PATH,
        FEATURE_COLUMNS_PATH,
    ]
    restore_bundled_artifacts()
    if all(path.exists() for path in required_outputs):
        return True, "Artifacts already available."

    try:
        from src.clean_taxi_data import clean_taxi_data
        from src.clean_weather_data import clean_weather_data
        from src.download_data import download_file, download_weather
        from src.evaluate_model import evaluate
        from src.feature_engineering import create_features
        from src.train_model import train_models
        from src.config import TAXI_ANALYTICS_PAGE, TAXI_ANALYTICS_URL, TAXI_HOTSPOTS_PAGE, TAXI_HOTSPOTS_URL, ensure_directories

        ensure_directories()
        if not RAW_TAXI_ANALYTICS_PATH.exists():
            download_file(TAXI_ANALYTICS_URL, RAW_TAXI_ANALYTICS_PATH, TAXI_ANALYTICS_PAGE)
        if not RAW_TAXI_HOTSPOTS_PATH.exists():
            download_file(TAXI_HOTSPOTS_URL, RAW_TAXI_HOTSPOTS_PATH, TAXI_HOTSPOTS_PAGE)
        if not RAW_WEATHER_PATH.exists():
            download_weather(RAW_WEATHER_PATH)

        if not CLEAN_TAXI_PATH.exists():
            clean_taxi_data()
        if not CLEAN_WEATHER_PATH.exists():
            clean_weather_data()
        if not MODELING_DATA_PATH.exists():
            create_features()
        if not MODEL_RESULTS_PATH.exists() or not BEST_MODEL_PATH.exists() or not FEATURE_COLUMNS_PATH.exists():
            train_models()
        if not TEST_PREDICTIONS_PATH.exists():
            train_models()
        evaluate()
    except Exception as exc:
        return False, str(exc)

    return True, "Dashboard artifacts generated successfully."


def infer_target_name(df: pd.DataFrame | None) -> str:
    if df is None or df.empty or "target" not in df.columns:
        return "Selected demand target"
    preferred_labels = {
        "passenger_pickup_trips": "Passenger pickup trips",
        "avg_trips_per_day": "Average trips per day",
        "total_passenger_travel_time": "Total passenger travel time",
        "total_distance": "Total distance",
        "average_taxis_per_month": "Average taxis per month",
    }
    for column, label in preferred_labels.items():
        if column in df.columns:
            comparable = df[[column, "target"]].dropna()
            if not comparable.empty and comparable[column].equals(comparable["target"]):
                return f"{label} ({column})"
    return "Selected taxi demand target"


def target_display_parts(df: pd.DataFrame | None) -> tuple[str, str, str]:
    target_name = infer_target_name(df)
    if "avg_trips_per_day" in target_name:
        return target_name, "Average taxi passenger trips per day", "trips/day"
    if "Passenger pickup trips" in target_name:
        return target_name, "Passenger pickup trips", "trips"
    if "Total passenger travel time" in target_name:
        return target_name, "Total passenger travel time", "time units"
    if "Total distance" in target_name:
        return target_name, "Total passenger trip distance", "distance units"
    if "Average taxis per month" in target_name:
        return target_name, "Average taxis per month", "taxis/month"
    return target_name, "Taxi demand", "index"


def pct_change_text(start_value: float, end_value: float) -> str:
    if pd.isna(start_value) or pd.isna(end_value) or start_value == 0:
        return "not enough usable data to calculate a percentage change"
    change = ((end_value - start_value) / start_value) * 100
    direction = "increased" if change > 0 else "decreased"
    return f"{direction} by {abs(change):.1f}%"


def shorten_label(value: str, max_length: int = 52) -> str:
    value = str(value)
    return value if len(value) <= max_length else f"{value[:max_length - 1]}..."


def demand_direction(start_value: float, end_value: float) -> str:
    if pd.isna(start_value) or pd.isna(end_value) or start_value == 0:
        return "unclear"
    change = ((end_value - start_value) / start_value) * 100
    if change <= -5:
        return "down"
    if change >= 5:
        return "up"
    return "flat"


def page_overview(modeling: pd.DataFrame | None) -> None:
    st.title("Bangkok Taxi Demand Forecasting and Mobility Analytics")
    st.markdown(
        """
        Bangkok's taxi market is changing as app-based mobility platforms such as Grab and Bolt become
        more common alternatives to traditional metered taxis. This project studies public GPS-derived
        taxi mobility indicators to understand how traditional taxi demand has moved over time, how
        weather and seasonality relate to demand, and whether classical forecasting models can produce
        useful next-month demand estimates.

        The analysis is framed as an urban mobility project: it connects passenger demand trends,
        pickup/dropoff hotspots, rainfall, temperature, humidity, and calendar effects into a dashboard
        that can support transport planning, market analysis, and portfolio-level data science storytelling.
        """
    )
    st.subheader("Data sources")
    st.markdown(
        "- OTP Passenger Trip Analytics from GPS taxi data, Thailand Ministry of Transport\n"
        "- OTP Taxi Pickup/Dropoff Top 20 Hotspot Dataset\n"
        "- Open-Meteo Historical Weather API for Bangkok"
    )
    clean_taxi = load_csv(CLEAN_TAXI_PATH, parse_dates=["date"])
    if modeling is not None and not modeling.empty:
        target_name = infer_target_name(clean_taxi if clean_taxi is not None else modeling)
        col1, col2 = st.columns([2, 1])
        col1.metric("Target variable", target_name)
        col2.metric("Modeling data range", f"{modeling['date'].min():%Y-%m} to {modeling['date'].max():%Y-%m}")
        if "target_was_imputed" in modeling.columns:
            imputed_months = int(modeling["target_was_imputed"].sum())
            st.caption(f"{imputed_months} modeling months use interpolated demand because the public source has missing monthly target values.")


def page_demand_trends(modeling: pd.DataFrame | None) -> None:
    st.title("Demand Trends")
    if modeling is None:
        missing_file(MODELING_DATA_PATH, "python src/feature_engineering.py")
        return
    clean_taxi = load_csv(CLEAN_TAXI_PATH, parse_dates=["date"])
    _, target_label, target_unit = target_display_parts(clean_taxi if clean_taxi is not None else modeling)
    y_axis_title = f"{target_label} ({target_unit})"

    trend_fig = px.line(
        modeling,
        x="date",
        y="target",
        markers=True,
        title="Monthly Bangkok Taxi Passenger Demand Trend",
        labels={"date": "Month", "target": y_axis_title},
    )
    trend_fig.update_layout(xaxis_title="Month", yaxis_title=y_axis_title)
    st.plotly_chart(trend_fig, use_container_width=True)
    first_row = modeling.dropna(subset=["target"]).iloc[0]
    last_row = modeling.dropna(subset=["target"]).iloc[-1]
    peak_row = modeling.loc[modeling["target"].idxmax()]
    low_row = modeling.loc[modeling["target"].idxmin()]
    trend_direction = demand_direction(first_row["target"], last_row["target"])
    if trend_direction == "down":
        trend_hypothesis = "A plausible explanation is that traditional taxi usage is being pressured by app-based ride-hailing, changing commuter habits, and shifts in post-pandemic travel demand."
    elif trend_direction == "up":
        trend_hypothesis = "A plausible explanation is that tourism recovery, office commuting, or airport-linked travel may be lifting traditional taxi activity over this period."
    else:
        trend_hypothesis = "A plausible explanation is that competing forces, such as ride-hailing adoption and tourism recovery, may be offsetting each other in the aggregate trend."
    st.caption(
        f"From {first_row['date']:%b %Y} to {last_row['date']:%b %Y}, demand "
        f"{pct_change_text(first_row['target'], last_row['target'])}. The highest modeled month is "
        f"{peak_row['date']:%b %Y} at {peak_row['target']:,.0f} {target_unit}, while the lowest is "
        f"{low_row['date']:%b %Y} at {low_row['target']:,.0f} {target_unit}. {trend_hypothesis}"
    )

    monthly = modeling.groupby("month", as_index=False)["target"].mean()
    seasonality_fig = px.bar(
        monthly,
        x="month",
        y="target",
        title="Average Taxi Demand by Calendar Month",
        labels={"month": "Calendar month", "target": f"Average {target_label.lower()} ({target_unit})"},
    )
    seasonality_fig.update_layout(xaxis_title="Calendar month", yaxis_title=f"Average {target_label.lower()} ({target_unit})")
    seasonality_fig.update_xaxes(dtick=1)
    st.plotly_chart(seasonality_fig, use_container_width=True)
    strongest_month = monthly.loc[monthly["target"].idxmax()]
    weakest_month = monthly.loc[monthly["target"].idxmin()]
    st.caption(
        f"Seasonality is strongest in month {int(strongest_month['month'])}, averaging "
        f"{strongest_month['target']:,.0f} {target_unit}. Month {int(weakest_month['month'])} is the softest, "
        f"averaging {weakest_month['target']:,.0f} {target_unit}. This pattern may reflect Bangkok's tourism calendar, "
        "holiday timing, school/work schedules, and seasonal weather conditions that affect when people choose taxis over other modes."
    )

    rolling = modeling[["date", "target"]].copy()
    rolling["3-month rolling average"] = rolling["target"].rolling(3).mean()
    rolling = rolling.rename(columns={"target": target_label})
    rolling_fig = px.line(
        rolling,
        x="date",
        y=[target_label, "3-month rolling average"],
        title="Taxi Demand Compared with 3-Month Rolling Average",
        labels={"date": "Month", "value": y_axis_title, "variable": "Series"},
    )
    rolling_fig.update_layout(xaxis_title="Month", yaxis_title=y_axis_title)
    st.plotly_chart(rolling_fig, use_container_width=True)
    rolling_valid = rolling.dropna(subset=["3-month rolling average"])
    if not rolling_valid.empty:
        rolling_first = rolling_valid.iloc[0]
        rolling_last = rolling_valid.iloc[-1]
        rolling_direction = demand_direction(rolling_first["3-month rolling average"], rolling_last["3-month rolling average"])
        if rolling_direction == "down":
            rolling_hypothesis = "The smoothed decline suggests the change is not only a single-month shock; it may reflect a structural shift in how Bangkok passengers choose between taxis, ride-hailing, transit, and private vehicles."
        elif rolling_direction == "up":
            rolling_hypothesis = "The smoothed increase suggests demand recovery is broad enough to persist beyond isolated high months, possibly linked to tourism, commuting, or city activity returning."
        else:
            rolling_hypothesis = "The relatively stable rolling average suggests monthly spikes and dips may be temporary, while underlying taxi demand is moving within a narrow band."
        st.caption(
            f"The 3-month rolling average smooths monthly volatility. It "
            f"{pct_change_text(rolling_first['3-month rolling average'], rolling_last['3-month rolling average'])} "
            f"from {rolling_first['date']:%b %Y} to {rolling_last['date']:%b %Y}, indicating the broader demand direction "
            f"after short-term month-to-month noise is reduced. {rolling_hypothesis}"
        )


def page_weather(modeling: pd.DataFrame | None) -> None:
    st.title("Weather Impact")
    if modeling is None:
        missing_file(MODELING_DATA_PATH, "python src/feature_engineering.py")
        return
    if "monthly_total_rain" in modeling.columns:
        rain_plot = modeling.dropna(subset=["monthly_total_rain", "target"])
        fig = px.scatter(
            rain_plot,
            x="monthly_total_rain",
            y="target",
            title="Monthly Rainfall and Taxi Demand",
            labels={
                "monthly_total_rain": "Monthly total rainfall (mm)",
                "target": "Average taxi passenger trips per day (trips/day)",
            },
        )
        if len(rain_plot) >= 2 and rain_plot["monthly_total_rain"].nunique() > 1:
            slope, intercept = np.polyfit(rain_plot["monthly_total_rain"], rain_plot["target"], 1)
            x_values = np.linspace(rain_plot["monthly_total_rain"].min(), rain_plot["monthly_total_rain"].max(), 100)
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=slope * x_values + intercept,
                    mode="lines",
                    name="Linear trend",
                )
            )
        st.plotly_chart(fig, use_container_width=True)
        if len(rain_plot) >= 2:
            corr = rain_plot["monthly_total_rain"].corr(rain_plot["target"])
            if pd.isna(corr):
                rain_summary = "The rainfall relationship is not clear enough to summarize from the available data. The available monthly points do not provide a stable direction for the relationship. This can happen when missing demand values, seasonal patterns, and weather variation overlap. More granular daily or hourly taxi data would be better for isolating short-term rain effects."
            elif corr > 0.25:
                rain_summary = "Higher-rainfall months tend to align with higher taxi demand in this view. One possible explanation is that rain makes walking, motorbike travel, and outdoor transit access less comfortable, pushing some travelers toward taxis. The relationship is still measured at monthly level, so it blends together many rainy and non-rainy days. It should be interpreted as a broad association rather than proof that rainfall directly causes demand increases."
            elif corr < -0.25:
                rain_summary = "Higher-rainfall months tend to align with lower taxi demand in this view. A plausible explanation is that heavy rain may reduce total travel activity, delay discretionary trips, or create traffic conditions that make taxi service less attractive. This does not mean rain always reduces taxi demand on a given day. At monthly scale, the chart captures the net effect after travel suppression, mode switching, and seasonality are all mixed together."
            else:
                rain_summary = "Rainfall does not show a strong linear relationship with taxi demand in this monthly dataset. This suggests rain is not the only dominant factor behind monthly taxi demand changes. Seasonal travel, tourism recovery, commuting habits, and ride-hailing competition may be more important at this level of aggregation. A daily or hourly dataset would likely reveal sharper rain-related effects than monthly totals can show."
            st.caption(rain_summary)
    cols = [col for col in ["monthly_avg_temp", "monthly_avg_humidity", "target"] if col in modeling.columns]
    if len(cols) >= 2:
        weather_matrix = px.scatter_matrix(
            modeling,
            dimensions=cols,
            title="Temperature, Humidity, and Taxi Demand Relationships",
            labels={
                "monthly_avg_temp": "Average temperature (deg C)",
                "monthly_avg_humidity": "Average relative humidity (%)",
                "target": "Average taxi passenger trips per day (trips/day)",
            },
        )
        st.plotly_chart(weather_matrix, use_container_width=True)
        temp_corr = modeling["monthly_avg_temp"].corr(modeling["target"]) if "monthly_avg_temp" in modeling.columns else float("nan")
        humidity_corr = modeling["monthly_avg_humidity"].corr(modeling["target"]) if "monthly_avg_humidity" in modeling.columns else float("nan")
        stronger_driver = "temperature" if abs(temp_corr) >= abs(humidity_corr) else "humidity"
        st.caption(
            f"This chart compares whether hotter or more humid months coincide with changes in taxi demand. In this dataset, {stronger_driver} has the stronger simple correlation with demand, but the relationship should not be read in isolation. Temperature and humidity are seasonal, so they can partly stand in for broader calendar effects such as school periods, holidays, tourism cycles, and rainy-season travel behavior. The main takeaway is that weather may help explain demand variation, but it works alongside larger mobility and market shifts."
        )


def prepare_hotspot_summary(hotspots: pd.DataFrame) -> pd.DataFrame:
    required = {"bus_stop", "district", "pickup_wd", "pickup_we", "dropoff_wd", "dropoff_we", "total_stops_wd", "total_stops_we"}
    missing = required.difference(hotspots.columns)
    if missing:
        raise ValueError(f"Hotspot data is missing required columns: {sorted(missing)}")

    df = hotspots.copy()
    df["location"] = df["bus_stop"].astype(str) + " | " + df["district"].astype(str)
    df["pickup_total"] = df["pickup_wd"] + df["pickup_we"]
    df["dropoff_total"] = df["dropoff_wd"] + df["dropoff_we"]
    df["all_stops_total"] = df["pickup_total"] + df["dropoff_total"]
    df["period"] = df["Year"].astype(str) + "-" + df["Month"].astype(str)

    summary = (
        df.groupby("location", as_index=False)
        .agg(
            pickup_total=("pickup_total", "sum"),
            dropoff_total=("dropoff_total", "sum"),
            all_stops_total=("all_stops_total", "sum"),
            weekday_total=("total_stops_wd", "sum"),
            weekend_total=("total_stops_we", "sum"),
            active_months=("period", "nunique"),
        )
        .sort_values("all_stops_total", ascending=False)
    )
    for col in ["pickup_total", "dropoff_total", "all_stops_total", "weekday_total", "weekend_total"]:
        summary[f"avg_monthly_{col}"] = summary[col] / summary["active_months"].clip(lower=1)
    summary["location_short"] = summary["location"].map(shorten_label)
    return summary


def model_display_name(model_name: str) -> str:
    names = {
        "naive_previous_month": "Naive previous-month baseline",
        "seasonal_previous_year": "Seasonal previous-year baseline",
        "ridge_regression": "Ridge regression",
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
    }
    return names.get(model_name, model_name.replace("_", " ").title())


def model_explanations() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Model": "Naive previous-month baseline",
                "What it does": "Uses last month's taxi demand as the forecast for next month.",
                "Why it matters": "A strong benchmark for slowly changing monthly demand.",
            },
            {
                "Model": "Seasonal previous-year baseline",
                "What it does": "Uses demand from the same month in the previous year when available.",
                "Why it matters": "Tests whether annual seasonality is more useful than recent demand.",
            },
            {
                "Model": "Ridge regression",
                "What it does": "Fits a regularized linear model using calendar, weather, lag, and rolling features.",
                "Why it matters": "Shows whether a simple explainable ML model improves on baselines.",
            },
            {
                "Model": "Random Forest",
                "What it does": "Combines many decision trees to learn nonlinear relationships between features and demand.",
                "Why it matters": "Can capture interactions, but may overfit when monthly data is limited.",
            },
            {
                "Model": "XGBoost",
                "What it does": "Uses boosted decision trees that sequentially correct earlier model errors.",
                "Why it matters": "Often strong on tabular data, but skipped here if the local runtime lacks the required library support.",
            },
        ]
    )


def metric_explanations() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Metric": "MAE", "Meaning": "Mean Absolute Error", "Interpretation": "Average absolute forecast miss in demand units. Lower is better."},
            {"Metric": "RMSE", "Meaning": "Root Mean Squared Error", "Interpretation": "Error metric that penalizes large misses more heavily. Lower is better."},
            {"Metric": "MAPE", "Meaning": "Mean Absolute Percentage Error", "Interpretation": "Average percentage miss relative to actual demand. Lower is better."},
            {"Metric": "R2", "Meaning": "Coefficient of determination", "Interpretation": "Share of test-period variation explained by the model. Higher is better, but it can be negative on small test sets."},
        ]
    )


def page_hotspots() -> None:
    st.title("Hotspot Analysis")
    hotspots = load_csv(RAW_TAXI_HOTSPOTS_PATH)
    summary = load_csv(DEPLOY_HOTSPOT_SUMMARY_PATH)
    if hotspots is not None:
        try:
            summary = prepare_hotspot_summary(hotspots)
        except ValueError as exc:
            st.warning(str(exc))
            st.dataframe(hotspots.head(50), use_container_width=True)
            return
    elif summary is None:
        missing_file(RAW_TAXI_HOTSPOTS_PATH, "python src/download_data.py")
        return

    st.write(
        "The hotspot dataset reports top taxi pickup and dropoff areas by month. Locations are aggregated by bus stop and district, "
        "then normalized as average monthly events so the chart compares real places instead of monthly rank IDs."
    )

    top_pickups = summary.nlargest(12, "avg_monthly_pickup_total").sort_values("avg_monthly_pickup_total")
    pickup_fig = px.bar(
        top_pickups,
        x="avg_monthly_pickup_total",
        y="location_short",
        orientation="h",
        title="Top Taxi Pickup Hotspots by Average Monthly Events",
        labels={"avg_monthly_pickup_total": "Average monthly pickup events", "location_short": "Hotspot location"},
        hover_data={"location": True, "avg_monthly_pickup_total": ":,.1f", "location_short": False},
        text="avg_monthly_pickup_total",
    )
    pickup_fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    pickup_fig.update_layout(xaxis_title="Average monthly pickup events", yaxis_title="Hotspot location", margin={"l": 220, "r": 40})
    st.plotly_chart(pickup_fig, use_container_width=True)
    pickup_leader = summary.nlargest(1, "avg_monthly_pickup_total").iloc[0]
    st.caption(
        f"The strongest pickup hotspot is {pickup_leader['location']}, averaging {pickup_leader['avg_monthly_pickup_total']:,.0f} pickup events per active month. "
        "Transport terminals, hospitals, malls, and transit-adjacent areas dominate because they concentrate predictable passenger origins. "
        "Using location names rather than IDs makes it clear which urban places are repeatedly generating taxi demand."
    )

    top_dropoffs = summary.nlargest(12, "avg_monthly_dropoff_total").sort_values("avg_monthly_dropoff_total")
    dropoff_fig = px.bar(
        top_dropoffs,
        x="avg_monthly_dropoff_total",
        y="location_short",
        orientation="h",
        title="Top Taxi Dropoff Hotspots by Average Monthly Events",
        labels={"avg_monthly_dropoff_total": "Average monthly dropoff events", "location_short": "Hotspot location"},
        hover_data={"location": True, "avg_monthly_dropoff_total": ":,.1f", "location_short": False},
        text="avg_monthly_dropoff_total",
    )
    dropoff_fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    dropoff_fig.update_layout(xaxis_title="Average monthly dropoff events", yaxis_title="Hotspot location", margin={"l": 220, "r": 40})
    st.plotly_chart(dropoff_fig, use_container_width=True)
    dropoff_leader = summary.nlargest(1, "avg_monthly_dropoff_total").iloc[0]
    st.caption(
        f"The leading dropoff hotspot is {dropoff_leader['location']}, averaging {dropoff_leader['avg_monthly_dropoff_total']:,.0f} dropoff events per active month. "
        "Dropoff-heavy areas often reflect destination demand, such as passengers arriving at terminals, hospitals, retail centers, or major transit connections. "
        "Comparing pickup and dropoff rankings helps separate places where trips begin from places where passengers are mainly arriving."
    )

    top_week = summary.nlargest(10, "avg_monthly_all_stops_total").copy()
    week_plot = top_week[["location_short", "avg_monthly_weekday_total", "avg_monthly_weekend_total"]].melt(
        id_vars="location_short",
        var_name="period_type",
        value_name="average_monthly_events",
    )
    week_plot["period_type"] = week_plot["period_type"].map(
        {
            "avg_monthly_weekday_total": "Weekday",
            "avg_monthly_weekend_total": "Weekend",
        }
    )
    week_fig = px.bar(
        week_plot,
        x="average_monthly_events",
        y="location_short",
        color="period_type",
        barmode="group",
        orientation="h",
        title="Weekday vs Weekend Activity at Major Taxi Hotspots",
        labels={"average_monthly_events": "Average monthly pickup + dropoff events", "location_short": "Hotspot location", "period_type": "Period"},
    )
    week_fig.update_layout(xaxis_title="Average monthly pickup + dropoff events", yaxis_title="Hotspot location", margin={"l": 220, "r": 40})
    st.plotly_chart(week_fig, use_container_width=True)
    weekday_total = summary["weekday_total"].sum()
    weekend_total = summary["weekend_total"].sum()
    period_bias = "weekday" if weekday_total >= weekend_total else "weekend"
    st.caption(
        f"Across the hotspot dataset, {period_bias} activity is higher overall. This comparison is useful because commuter, hospital, and terminal demand often behaves differently from leisure or shopping trips. "
        "Locations with strong weekday volume may be more tied to work and institutional travel, while weekend-heavy places may reflect retail, tourism, or discretionary movement."
    )

    with st.expander("View aggregated hotspot table"):
        st.dataframe(
            summary[
                [
                    "location",
                    "active_months",
                    "avg_monthly_pickup_total",
                    "avg_monthly_dropoff_total",
                    "avg_monthly_all_stops_total",
                ]
            ].head(50),
            use_container_width=True,
        )

    lat_cols = [col for col in hotspots.columns if "lat" in col.lower() or "latitude" in col.lower()] if hotspots is not None else []
    lon_cols = [col for col in hotspots.columns if "lon" in col.lower() or "lng" in col.lower() or "longitude" in col.lower()] if hotspots is not None else []
    if hotspots is not None and lat_cols and lon_cols:
        st.map(hotspots.rename(columns={lat_cols[0]: "lat", lon_cols[0]: "lon"}).dropna(subset=["lat", "lon"]))


def page_model_performance() -> None:
    st.title("Model Performance")
    results = load_csv(MODEL_RESULTS_PATH)
    predictions = load_csv(TEST_PREDICTIONS_PATH, parse_dates=["date"])
    if results is None:
        missing_file(MODEL_RESULTS_PATH, "python src/train_model.py")
        return
    st.write(
        "The forecasting task predicts next-month taxi demand from historical monthly demand, calendar features, and weather features. "
        "Models are evaluated on a time-based holdout period, so later months are used for testing instead of randomly mixing the data."
    )

    with st.expander("What each model does", expanded=True):
        st.dataframe(model_explanations(), use_container_width=True, hide_index=True)

    with st.expander("What the error metrics mean", expanded=True):
        st.dataframe(metric_explanations(), use_container_width=True, hide_index=True)

    display_results = results.copy()
    display_results["Model"] = display_results["model"].map(model_display_name)
    metric_cols = ["mae", "rmse", "mape", "r2"]
    display_results = display_results[["Model", *metric_cols]].rename(
        columns={"mae": "MAE", "rmse": "RMSE", "mape": "MAPE (%)", "r2": "R2"}
    )
    st.subheader("Model Comparison")
    st.dataframe(display_results, use_container_width=True, hide_index=True)
    best = results.sort_values(["mae", "rmse"]).iloc[0]["model"]
    best_label = model_display_name(str(best))
    st.metric("Best model by MAE", best_label)
    naive_row = results.loc[results["model"] == "naive_previous_month"]
    best_row = results.loc[results["model"] == best].iloc[0]
    if not naive_row.empty and best == "naive_previous_month":
        st.caption(
            f"The naive baseline performs best with an MAE of {best_row['mae']:,.0f}, meaning its average monthly miss is about "
            f"{best_row['mae']:,.0f} trips/day. This is plausible because the modeled taxi series changes gradually month to month, "
            "so last month's demand already contains much of the useful signal for next month. More complex models have relatively few monthly observations, "
            "and the added weather/calendar features may not provide enough extra information to overcome noise, interpolation, and structural changes in taxi usage."
        )
    else:
        st.caption(
            f"{best_label} performs best by MAE on the test period. The result should still be compared against the naive baseline because monthly demand forecasting often benefits strongly from recent lag values."
        )
    if predictions is not None and best in predictions.columns:
        prediction_plot = predictions.rename(columns={"actual": "Actual demand", best: f"{best_label} prediction"})
        st.plotly_chart(
            px.line(
                prediction_plot,
                x="date",
                y=["Actual demand", f"{best_label} prediction"],
                markers=True,
                title="Actual vs Predicted Taxi Demand on the Test Period",
                labels={"date": "Month", "value": "Average taxi passenger trips per day (trips/day)", "variable": "Series"},
            ),
            use_container_width=True,
        )
        st.caption(
            "This plot shows whether the best model follows the direction and magnitude of demand in the holdout months. "
            "When a simple baseline tracks actual demand closely, it suggests short-term persistence is stronger than complex feature effects. "
            "Large gaps between actual and predicted values highlight months where unusual travel behavior, missing source values, or external market shifts may not be fully captured."
        )
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

    month = st.selectbox("Forecast month (calendar month number)", list(range(1, 13)), index=0)
    rainfall = st.number_input("Expected monthly rainfall (mm)", min_value=0.0, value=float(latest.get("monthly_total_rain", 100.0)))
    avg_temp = st.number_input("Expected average temperature (deg C)", value=float(latest.get("monthly_avg_temp", 29.0)))
    humidity = st.number_input("Expected average relative humidity (%)", min_value=0.0, max_value=100.0, value=float(latest.get("monthly_avg_humidity", 75.0)))
    previous_demand = st.number_input("Previous month demand (average taxi passenger trips per day)", min_value=0.0, value=float(latest.get("target", 0.0)))

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
        raw_prediction = float(model.predict(pd.DataFrame([{col: row.get(col, 0) for col in feature_columns}]))[0])
        prediction = max(0.0, raw_prediction)
        st.metric("Predicted taxi demand", f"{prediction:,.2f} trips/day")
        if raw_prediction < 0:
            st.caption("The raw model output was below zero, so the displayed prediction is capped at 0 because negative demand is not meaningful.")


def main() -> None:
    with st.spinner("Preparing dashboard data and model artifacts..."):
        artifacts_ready, artifact_message = ensure_dashboard_artifacts()
    if not artifacts_ready:
        st.error("The dashboard could not prepare its required data files.")
        st.write(artifact_message)
        st.info("On deployment, the app needs internet access to download the public OTP and Open-Meteo files on first run.")
        return

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
