"""Generate realistic sample datasets for RespiGuard-AI.

Run once with:  python generate_sample_data.py

This creates three CSV files in data/sample_data/ so the full project runs
end-to-end without any real datasets. When the developer later drops real
files into data/ the app switches over automatically.

Cities:     Delhi, Mumbai, Chennai, Kolkata, Ahmedabad, Bengaluru
Date range: January 2015 to December 2020 (monthly, 72 rows per city).
"""

import os
import random

import numpy as np
import pandas as pd

# Deterministic output so every run produces the same believable dataset.
random.seed(42)
np.random.seed(42)

CITIES = ["Delhi", "Mumbai", "Chennai", "Kolkata", "Ahmedabad", "Bengaluru"]

# Per-city AQI ranges (min, max) reflecting each city's real-world profile.
CITY_AQI_RANGE = {
    "Delhi": (150, 999),
    "Mumbai": (80, 300),
    "Chennai": (70, 250),
    "Kolkata": (100, 400),
    "Ahmedabad": (100, 450),
    "Bengaluru": (40, 200),
}

# Hospital respiratory-admission baselines (per month) for each city.
CITY_HOSPITAL_BASE = {
    "Delhi": 180,
    "Mumbai": 150,
    "Chennai": 120,
    "Kolkata": 160,
    "Ahmedabad": 130,
    "Bengaluru": 100,
}

# Per-city weather envelopes: temperature/humidity/wind (min, max) ranges.
CITY_WEATHER = {
    "Delhi": {"temp": (5, 45), "humidity": (20, 90), "wind": (5, 30)},
    "Mumbai": {"temp": (20, 35), "humidity": (60, 95), "wind": (10, 40)},
    "Chennai": {"temp": (22, 40), "humidity": (65, 95), "wind": (10, 35)},
    "Kolkata": {"temp": (12, 38), "humidity": (50, 95), "wind": (5, 25)},
    "Ahmedabad": {"temp": (8, 45), "humidity": (20, 80), "wind": (5, 35)},
    "Bengaluru": {"temp": (15, 32), "humidity": (40, 85), "wind": (8, 30)},
}

DISEASE_TYPES = ["asthma", "COPD", "bronchitis", "pneumonia"]

OUTPUT_DIR = os.path.join("data", "sample_data")


def get_season(month: int) -> str:
    """Return the season name for a given month number."""
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5, 6):
        return "Summer"
    if month in (7, 8, 9):
        return "Monsoon"
    return "PostMonsoon"  # months 10, 11


def build_date_range() -> pd.DatetimeIndex:
    """Build a monthly date range from January 2015 to December 2020."""
    return pd.date_range(start="2015-01-01", end="2020-12-01", freq="MS")


def make_city_aqi(city: str, month: int) -> float:
    """Compute a realistic AQI value for a city in a given month.

    Winter months (Oct-Feb) trap pollution, and October/November get an extra
    30% bump for Diwali firecrackers. Small random noise adds realism.
    """
    low, high = CITY_AQI_RANGE[city]
    mid = (low + high) / 2.0

    # Seasonal shaping: winter months push AQI toward the high end.
    if month in (10, 11, 12, 1, 2):
        base = mid + (high - mid) * random.uniform(0.4, 0.9)
    else:
        base = low + (mid - low) * random.uniform(0.2, 0.9)

    # Diwali firecracker spike in October and November: +30%.
    if month in (10, 11):
        base *= 1.30

    # Add random noise for realism.
    base += random.uniform(-15, 15)

    # Keep the value within the city's plausible bounds.
    return float(np.clip(round(base), low, high))


def generate_aqi_data(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Generate the AQI sample dataset for all cities and dates."""
    rows = []
    for city in CITIES:
        for date in dates:
            aqi = make_city_aqi(city, date.month)
            # Derive pollutant concentrations from AQI with small noise.
            pm25 = round(aqi * 0.4 + random.uniform(-5, 5), 1)
            pm10 = round(aqi * 0.3 + random.uniform(-5, 5), 1)
            no2 = round(aqi * 0.15 + random.uniform(-3, 3), 1)
            so2 = round(aqi * 0.10 + random.uniform(-2, 2), 1)
            rows.append({
                "date": date.strftime("%Y-%m-%d"),
                "city": city,
                "PM2.5": max(pm25, 0.0),
                "PM10": max(pm10, 0.0),
                "NO2": max(no2, 0.0),
                "SO2": max(so2, 0.0),
                "AQI": int(aqi),
            })
    return pd.DataFrame(rows)


def generate_hospital_data(aqi_df: pd.DataFrame) -> pd.DataFrame:
    """Generate hospital respiratory-admission data driven by AQI and season."""
    rows = []
    disease_index = 0
    for _, row in aqi_df.iterrows():
        city = row["city"]
        aqi = row["AQI"]
        month = int(row["date"][5:7])
        base = CITY_HOSPITAL_BASE[city]

        admissions = float(base)
        # Winter months worsen respiratory conditions.
        if month in (10, 11, 12, 1, 2):
            admissions *= 1.35
        # Severe and critical AQI drive additional admissions.
        if aqi > 400:
            admissions *= 1.50
        elif aqi > 300:
            admissions *= 1.30

        # Random noise for realism.
        admissions *= random.uniform(0.92, 1.08)

        # Rotate through disease types deterministically.
        disease = DISEASE_TYPES[disease_index % len(DISEASE_TYPES)]
        disease_index += 1

        rows.append({
            "date": row["date"],
            "city": city,
            "respiratory_admissions": int(round(admissions)),
            "disease_type": disease,
        })
    return pd.DataFrame(rows)


def generate_weather_data(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Generate weather data per city with seasonal temperature/wind shaping."""
    rows = []
    for city in CITIES:
        env = CITY_WEATHER[city]
        for date in dates:
            month = date.month
            season = get_season(month)

            t_low, t_high = env["temp"]
            h_low, h_high = env["humidity"]
            w_low, w_high = env["wind"]

            # Seasonal shaping of the weather envelope.
            if season == "Winter":
                # Lower temperature and lower wind (traps pollution).
                temperature = random.uniform(t_low, t_low + (t_high - t_low) * 0.4)
                wind_speed = random.uniform(w_low, w_low + (w_high - w_low) * 0.4)
                humidity = random.uniform(h_low, (h_low + h_high) / 2)
            elif season == "Monsoon":
                # Higher humidity, medium wind.
                temperature = random.uniform((t_low + t_high) / 2, t_high * 0.9)
                wind_speed = random.uniform((w_low + w_high) / 2, w_high * 0.7)
                humidity = random.uniform((h_low + h_high) / 2, h_high)
            elif season == "Summer":
                temperature = random.uniform((t_low + t_high) / 2, t_high)
                wind_speed = random.uniform(w_low, w_high)
                humidity = random.uniform(h_low, (h_low + h_high) / 2)
            else:  # PostMonsoon
                temperature = random.uniform(t_low + (t_high - t_low) * 0.3, t_high * 0.8)
                wind_speed = random.uniform(w_low, (w_low + w_high) / 2)
                humidity = random.uniform((h_low + h_high) / 2, h_high * 0.9)

            rows.append({
                "date": date.strftime("%Y-%m-%d"),
                "city": city,
                "temperature": round(temperature, 1),
                "humidity": round(humidity, 1),
                "wind_speed": round(wind_speed, 1),
                "season": season,
            })
    return pd.DataFrame(rows)


def main():
    """Generate all three sample CSVs and print a completion summary."""
    # Make sure the output directory exists.
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dates = build_date_range()

    # Generate the three linked datasets.
    aqi_df = generate_aqi_data(dates)
    hospital_df = generate_hospital_data(aqi_df)
    weather_df = generate_weather_data(dates)

    # Write to CSV.
    aqi_path = os.path.join(OUTPUT_DIR, "aqi_sample.csv")
    hospital_path = os.path.join(OUTPUT_DIR, "hospital_sample.csv")
    weather_path = os.path.join(OUTPUT_DIR, "weather_sample.csv")

    aqi_df.to_csv(aqi_path, index=False)
    hospital_df.to_csv(hospital_path, index=False)
    weather_df.to_csv(weather_path, index=False)

    # Report row counts per file.
    print(f"aqi_sample.csv      -> {len(aqi_df)} rows written to {aqi_path}")
    print(f"hospital_sample.csv -> {len(hospital_df)} rows written to {hospital_path}")
    print(f"weather_sample.csv  -> {len(weather_df)} rows written to {weather_path}")
    print("Sample data generated successfully — run app.py now")


if __name__ == "__main__":
    main()
