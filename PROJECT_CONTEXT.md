# RespiGuard-AI — Project Context

## Summary
Predicts hospital respiratory admission surge 72hr ahead from AQI spikes
6 cities: Delhi Mumbai Chennai Kolkata Ahmedabad Bengaluru

## Version Status
V1 — Complete (June 2026)
V2 — Pending (July 2026) — adds live AQI + visualizations
V3 — Pending (August 2026) — adds self improving model

## Data Instructions
Real datasets go into data/ folder:
- data/aqi_data.csv      → from Kaggle (rohanrao Air Quality India)
- data/hospital_data.csv → from Kaggle respiratory dataset
- data/weather_data.csv  → from Open-Meteo historical download

If real data not present app auto uses data/sample_data/ folder

## Expected Dataset Columns
aqi_data.csv:      date, city, PM2.5, PM10, NO2, SO2, AQI
hospital_data.csv: date, city, respiratory_admissions, disease_type
weather_data.csv:  date, city, temperature, humidity, wind_speed, season
Cities:            Delhi, Mumbai, Chennai, Kolkata, Ahmedabad, Bengaluru
Date format:       YYYY-MM-DD

## Key Files
src/styles.py           → design system, all CSS, all UI functions
src/data_cleaning.py    → loads and merges 3 CSVs, auto detects real vs sample
src/feature_engineering.py → lag, rolling, season, festival, pollutant features
src/bayesian_model.py   → pgmpy primary, sklearn GaussianNB backup
src/expert_system.py    → IF-THEN risk rules, 7 priority rules
train_model.py          → run once to train model: python train_model.py
app.py                  → Streamlit app: python streamlit run app.py

## Engineered Features
AQI_lag_24hr, AQI_lag_48hr, AQI_lag_72hr
AQI_rolling_7day
season_name, season_num
festival_risk (0 or 1)
PM25_pct, PM10_pct, NO2_pct, SO2_pct
AQI_Category, AQI_Category_num
admission_surge (target variable)

## Model
Primary:  pgmpy BayesianNetwork with MaximumLikelihoodEstimator
Backup:   sklearn GaussianNB (auto fallback if pgmpy fails)
Saved as: models/trained_model_v1.pkl
Output:   LOW / LOW-MEDIUM / MEDIUM / HIGH / CRITICAL + confidence %

## Design System
Theme:    Dark clinical navy
Primary:  #0A1628
Accent:   #00D4AA (electric teal)
Fonts:    Space Grotesk + Inter + JetBrains Mono
Signature: Glowing pulse animation on CRITICAL badge

## How to Run
Step 1: python generate_sample_data.py
Step 2: python train_model.py
Step 3: streamlit run app.py

## V2 Context (for future reference)
Will add: src/live_aqi.py, src/visualizations.py, config.py
Will add: Folium map, timeline graph, pollutant pie, what-if simulator
Will add: 3 new Streamlit pages

## V3 Context (for future reference)
Will add: src/aqi_forecast.py, src/feedback_system.py
Will add: src/auto_retrain.py, src/report_generator.py
Will add: Doctor feedback loop, weekly retraining, PDF morning report
