# Bangkok Taxi Demand Forecasting and Mobility Analytics

## Project overview

This is a reproducible end-to-end data science project for analyzing Bangkok taxi mobility patterns and forecasting next-month taxi passenger demand. It uses public GPS-derived taxi mobility analytics from Thailand's Ministry of Transport and historical Bangkok weather data from Open-Meteo.

This is not Grab, Bolt, or other proprietary ride-hailing data. The project focuses on public urban mobility analytics, transparent forecasting, and a recruiter-friendly Streamlit dashboard.

## Why this project matters

Taxi demand reflects commuter flows, tourism, weather, seasonality, and city activity. A monthly forecasting workflow can help planners and operators understand demand cycles, compare weather impacts, and evaluate whether machine learning improves over simple historical baselines.

## Data sources

- OTP Passenger Trip Analytics from GPS taxi data: https://datagov.mot.go.th/dataset/otp_69_04
- OTP Taxi Pickup/Dropoff Top 20 Hotspot Dataset: https://datagov.mot.go.th/dataset/otp_69_05
- Open-Meteo Historical Weather API: https://open-meteo.com/en/docs/historical-weather-api

## Methodology

1. Download public OTP taxi analytics, hotspot data, and Bangkok hourly weather.
2. Clean Thai/mixed taxi columns and select the best available demand target.
3. Aggregate hourly weather into monthly features.
4. Create calendar, seasonality, lag, rolling mean, and percentage-change features.
5. Train models with a time-based train/test split.
6. Compare classical ML models against naive and seasonal baselines.
7. Present trends, weather impact, hotspots, model performance, and forecasting in Streamlit.

No LSTM or deep learning is used. The dataset is monthly and relatively small, so baseline models and classical tabular/time-series ML are more appropriate and easier to interpret.

## Repository structure

```text
bangkok-taxi-demand/
  README.md
  requirements.txt
  .gitignore
  data/
    raw/
    processed/
  notebooks/
    01_data_inspection.ipynb
    02_eda.ipynb
    03_modeling.ipynb
  src/
    config.py
    download_data.py
    clean_taxi_data.py
    clean_weather_data.py
    feature_engineering.py
    train_model.py
    evaluate_model.py
    predict.py
    plotting.py
  app/
    streamlit_app.py
  models/
  reports/
    figures/
```

## How to run

Use Python 3.10 or newer.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python src/download_data.py
python src/clean_taxi_data.py
python src/clean_weather_data.py
python src/feature_engineering.py
python src/train_model.py
python src/evaluate_model.py
python src/predict.py

streamlit run app/streamlit_app.py
```

If an OTP direct CSV link fails, manually download the file from the dataset page and save it with the expected filename under `data/raw/`.

## Results

Results are generated after running the pipeline:

- `data/processed/model_results.csv`
- `data/processed/test_predictions.csv`
- `reports/figures/actual_vs_predicted.png`
- `reports/figures/residuals_over_time.png`

## Dashboard screenshots

Add screenshots from the Streamlit app after running the dashboard locally.

## Model performance

The training script evaluates:

- Naive previous-month baseline
- Seasonal previous-year baseline
- Ridge regression
- Random Forest Regressor
- XGBoost Regressor, if installed

The README should be updated with the generated `model_results.csv` table once the final data files are downloaded and processed.

## Key insights

Typical insights to document after running the data:

- Direction and volatility of monthly taxi demand
- Seasonal demand patterns by month
- Relationship between rainfall and demand
- Whether ML models outperform simple baselines
- Most frequent pickup/dropoff hotspot areas

## Limitations

- The core demand data is monthly, which limits the usefulness of deep learning and high-frequency forecasting.
- Public aggregated data may hide neighborhood-level and trip-level variation.
- Weather is measured for central Bangkok and may not capture all local conditions.
- Forecast accuracy depends on the stability of post-pandemic travel, tourism, and policy patterns.

## Future improvements

- Add holiday and event calendars for Bangkok.
- Improve hotspot standardization and geocoding.
- Add confidence intervals or scenario ranges.
- Compare SARIMAX or Prophet as optional non-required models.
- Deploy the Streamlit dashboard.

