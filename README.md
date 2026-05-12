# Bangkok Taxi Demand Forecasting and Mobility Analytics

## Project overview

Link to Dashboard: https://bangkok-taxi-demand.streamlit.app/


This project explores how traditional taxi demand in Bangkok has changed over time using public GPS-derived taxi mobility analytics from Thailand's Ministry of Transport. I built it as an end-to-end data science project that starts with raw public data, cleans and merges it with weather data, creates forecasting features, compares classical forecasting models, and presents the results in a Streamlit dashboard.

The main forecasting target is average taxi passenger trips per day. The dashboard looks at demand trends, seasonality, weather relationships, pickup/dropoff hotspots, model performance, and a simple forecast tool for testing future-month assumptions.

I intentionally avoided LSTM or deep learning methods. The dataset is monthly and relatively small, so classical baselines and tabular/time-series machine learning methods are more appropriate, easier to interpret, and more honest for this problem size.

## Methodology

The project follows a reproducible pipeline:

1. Download OTP taxi analytics, OTP hotspot data, and Bangkok historical weather data.
2. Clean Thai/mixed taxi data columns and select the best available demand target.
3. Aggregate hourly Open-Meteo weather data into monthly weather features.
4. Filter the analysis to the 2023-2026 period where the project focus is strongest.
5. Fill internal missing monthly demand values with linear interpolation and flag those months for transparency.
6. Create calendar, seasonality, lag, rolling-average, and percentage-change features.
7. Train models with a time-based train/test split instead of a random split.
8. Compare model performance against simple baselines.
9. Display the results in a Streamlit dashboard.

Models evaluated:

- Naive previous-month baseline
- Seasonal previous-year baseline
- Ridge regression
- Random Forest Regressor
- XGBoost Regressor, if the local environment supports it

The main evaluation metrics are MAE, RMSE, MAPE, and R2. In the current run, the naive previous-month baseline performs best, which suggests monthly taxi demand is highly persistent and that recent demand contains more useful signal than the added weather/calendar features for this small dataset.

Run the project:

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

## Data sources

- OTP Passenger Trip Analytics from GPS taxi data: https://datagov.mot.go.th/dataset/otp_69_04
- OTP Taxi Pickup/Dropoff Top 20 Hotspot Dataset: https://datagov.mot.go.th/dataset/otp_69_05
- Open-Meteo Historical Weather API: https://open-meteo.com/en/docs/historical-weather-api

This project does not use proprietary Grab, Bolt, or ride-hailing company data. The taxi data is public aggregated mobility data, which makes the project more reproducible but also limits the level of detail available for modeling.

## Intent

I wanted this project to connect data science with a real urban mobility question in Bangkok. Traditional taxis operate in a market where app-based ride-hailing platforms such as Grab and Bolt have become increasingly popular, so I was interested in whether public taxi data shows signs of changing demand patterns.

The goal was not just to train a model, but to build a full analytical workflow that a reader could follow end to end. The project is meant to show how I approach messy public data, feature engineering, baseline comparison, model interpretation, and dashboard communication.

The dashboard is designed for exploration rather than only prediction. It helps answer questions such as whether taxi demand is trending downward, which months are stronger or weaker, whether rain or humidity appear related to demand, which locations repeatedly appear as hotspots, and whether machine learning improves on simple historical baselines.

The current results also show an important modeling lesson: a simple baseline can outperform more complex models when the dataset is small and demand changes gradually. That is useful context because it keeps the project grounded in the data instead of forcing a more complicated model where it is not justified.
