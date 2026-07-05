"""Feature engineering pipeline for RespiGuard-AI.

Builds lag features, rolling averages, season encodings, festival-risk flags,
pollutant contribution percentages, AQI categories, and the target
``admission_surge`` label from the cleaned + merged dataset.
"""

import os

import pandas as pd


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-city AQI lag features (24hr/48hr/72hr = 1/2/3 periods back)."""
    df = df.copy()
    # Sort so shifting respects chronological order within each city.
    df = df.sort_values(["city", "date"]).reset_index(drop=True)

    grouped = df.groupby("city")["AQI"]
    df["AQI_lag_24hr"] = grouped.shift(1)
    df["AQI_lag_48hr"] = grouped.shift(2)
    df["AQI_lag_72hr"] = grouped.shift(3)

    # Fill missing lag values (start of each city series) with current AQI.
    for col in ["AQI_lag_24hr", "AQI_lag_48hr", "AQI_lag_72hr"]:
        df[col] = df[col].fillna(df["AQI"])

    return df


def add_rolling_average(df: pd.DataFrame) -> pd.DataFrame:
    """Add a per-city 7-period rolling mean of AQI."""
    df = df.copy()
    df = df.sort_values(["city", "date"]).reset_index(drop=True)

    df["AQI_rolling_7day"] = (
        df.groupby("city")["AQI"]
        .transform(lambda s: s.rolling(window=7, min_periods=1).mean())
    )
    # Any residual NaN falls back to the raw AQI value.
    df["AQI_rolling_7day"] = df["AQI_rolling_7day"].fillna(df["AQI"])
    return df


def add_season_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """Add season_name and numeric season_num columns derived from the month."""
    df = df.copy()
    month = pd.to_datetime(df["date"], errors="coerce").dt.month

    def season_from_month(m):
        # Winter(12,1,2), Summer(3,4,5,6), Monsoon(7,8,9), PostMonsoon(10,11).
        if m in (12, 1, 2):
            return "Winter"
        if m in (3, 4, 5, 6):
            return "Summer"
        if m in (7, 8, 9):
            return "Monsoon"
        return "PostMonsoon"

    df["season_name"] = month.apply(season_from_month)
    season_num_map = {"Winter": 0, "Summer": 1, "Monsoon": 2, "PostMonsoon": 3}
    df["season_num"] = df["season_name"].map(season_num_map)
    return df


def add_festival_risk(df: pd.DataFrame) -> pd.DataFrame:
    """Flag festival-period rows (Diwali, New Year, Holi, Chhath Puja)."""
    df = df.copy()
    dates = pd.to_datetime(df["date"], errors="coerce")
    month = dates.dt.month
    day = dates.dt.day

    def is_festival(m, d):
        if pd.isna(m) or pd.isna(d):
            return 0
        m, d = int(m), int(d)
        # Diwali window: October 20-25.
        if m == 10 and 20 <= d <= 25:
            return 1
        # New Year: December 31 and January 1.
        if (m == 12 and d == 31) or (m == 1 and d == 1):
            return 1
        # Holi: March 20-25.
        if m == 3 and 20 <= d <= 25:
            return 1
        # Chhath Puja: November 1-5.
        if m == 11 and 1 <= d <= 5:
            return 1
        return 0

    df["festival_risk"] = [is_festival(m, d) for m, d in zip(month, day)]
    return df


def add_pollutant_contribution(df: pd.DataFrame) -> pd.DataFrame:
    """Add each pollutant's percentage contribution to the total load."""
    df = df.copy()
    # +1 avoids any division-by-zero when all pollutants are zero.
    total = df["PM2.5"] + df["PM10"] + df["NO2"] + df["SO2"] + 1
    df["PM25_pct"] = (df["PM2.5"] / total) * 100
    df["PM10_pct"] = (df["PM10"] / total) * 100
    df["NO2_pct"] = (df["NO2"] / total) * 100
    df["SO2_pct"] = (df["SO2"] / total) * 100
    return df


def add_aqi_category(df: pd.DataFrame) -> pd.DataFrame:
    """Add AQI_Category label and numeric AQI_Category_num."""
    df = df.copy()

    def categorize(aqi):
        if aqi <= 100:
            return "Low"
        if aqi <= 200:
            return "Moderate"
        if aqi <= 300:
            return "Poor"
        if aqi <= 400:
            return "Severe"
        return "Critical"

    df["AQI_Category"] = df["AQI"].apply(categorize)
    cat_num_map = {"Low": 0, "Moderate": 1, "Poor": 2, "Severe": 3, "Critical": 4}
    df["AQI_Category_num"] = df["AQI_Category"].map(cat_num_map)
    return df


def add_admission_surge_label(df: pd.DataFrame) -> pd.DataFrame:
    """Add the binary target ``admission_surge`` (above city mean = 1)."""
    df = df.copy()
    city_mean = df.groupby("city")["respiratory_admissions"].transform("mean")
    df["admission_surge"] = (df["respiratory_admissions"] > city_mean).astype(int)
    return df


def run_feature_engineering(merged_df: pd.DataFrame) -> pd.DataFrame:
    """Run all feature-engineering steps in order and save the result."""
    original_cols = set(merged_df.columns)

    df = add_lag_features(merged_df)
    df = add_rolling_average(df)
    df = add_season_encoding(df)
    df = add_festival_risk(df)
    df = add_pollutant_contribution(df)
    df = add_aqi_category(df)
    df = add_admission_surge_label(df)

    # Persist engineered dataset.
    os.makedirs("output", exist_ok=True)
    df.to_csv(os.path.join("output", "engineered_data.csv"), index=False)

    new_cols = [c for c in df.columns if c not in original_cols]
    print(f"New columns added: {new_cols}")
    print(f"Engineered dataset shape: {df.shape}")
    print("Feature engineering complete")

    return df


if __name__ == "__main__":
    from src.data_cleaning import run_cleaning

    merged, _ = run_cleaning()
    run_feature_engineering(merged)
