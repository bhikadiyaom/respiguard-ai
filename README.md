# 🫁 RespiGuard-AI
### Hospital Respiratory Surge Predictor — Version 1.0

AI system that predicts hospital respiratory admission surges
72 hours in advance based on AQI spikes across 6 Indian cities.

---

## The Problem
When AQI spikes in Delhi, hospitals overflow 48-72 hours later.
Doctors have no warning. RespiGuard-AI changes that.

---

## How to Run

Step 1 — Install dependencies:

```bash
pip install -r requirements.txt
```

Step 2 — Generate sample data (if you don't have real datasets yet):

```bash
python generate_sample_data.py
```

Step 3 — Train model:

```bash
python train_model.py
```

Step 4 — Run app:

```bash
streamlit run app.py
```

---

## Upload Real Datasets
Drop these files into the `data/` folder:
- `data/aqi_data.csv`
- `data/hospital_data.csv`
- `data/weather_data.csv`

Sources:
- AQI data: kaggle.com search "Air Quality Data India rohanrao"
- Hospital data: kaggle.com search "respiratory hospital admissions"
- Weather data: open-meteo.com historical download

App auto-detects and switches to real data.

---

## Architecture

```
CSV Data → Clean → Engineer Features → Bayesian Model → Expert Rules → Streamlit UI
```

---

## Version Roadmap
- V1 (June 2026) — Core prediction system ← YOU ARE HERE
- V2 (July 2026) — Live AQI + visualizations + maps
- V3 (August 2026) — Self improving model + doctor feedback

---

## Tech Stack
Python, pgmpy, scikit-learn, Streamlit, Pandas, NumPy

---

Built by Om — BE Computer Engineering, LJ University, Ahmedabad
