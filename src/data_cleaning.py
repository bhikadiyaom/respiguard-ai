"""Data cleaning pipeline for RespiGuard-AI.

Auto-detects whether real datasets are present in data/ and otherwise falls
back to the auto-generated sample data in data/sample_data/. Loads, cleans,
and merges the AQI, hospital, and weather datasets into a single DataFrame.
"""

import os

import pandas as pd

# The six cities the whole system is built around.
EXPECTED_CITIES = ["Delhi", "Mumbai", "Chennai", "Kolkata", "Ahmedabad", "Bengaluru"]


def _print_sample_banner():
    """Print the clear SAMPLE DATA MODE banner described in the spec."""
    print("========================================")
    print("SAMPLE DATA MODE")
    print("Real datasets not found in data/ folder")
    print("Upload these files to switch to real data:")
    print("  \u2192 data/aqi_data.csv")
    print("  \u2192 data/hospital_data.csv")
    print("  \u2192 data/weather_data.csv")
    print("")
    print("Expected columns:")
    print("aqi_data.csv:      date, city, PM2.5, PM10, NO2, SO2, AQI")
    print("hospital_data.csv: date, city, respiratory_admissions, disease_type")
    print("weather_data.csv:  date, city, temperature, humidity, wind_speed, season")
    print("")
    print("Cities expected: Delhi, Mumbai, Chennai, Kolkata, Ahmedabad, Bengaluru")
    print("Date format:     YYYY-MM-DD")
    print("========================================")


def check_data_source():
    """Determine which data folder to use.

    Returns a tuple ``(data_folder_path, using_sample_bool)``. If a non-empty
    ``data/aqi_data.csv`` exists we use the real data folder; otherwise we fall
    back to the sample data folder and print the SAMPLE DATA MODE banner.
    """
    real_path = os.path.join("data", "aqi_data.csv")

    if os.path.exists(real_path) and os.path.getsize(real_path) > 0:
        print("Real dataset found — loading from data/")
        return "data/", False

    print("Real dataset not found — using sample data for now")
    print("Upload your CSV files to data/ folder to use real data")
    _print_sample_banner()
    return os.path.join("data", "sample_data") + os.sep, True


def load_datasets(data_path: str):
    """Load the AQI, hospital, and weather CSVs from ``data_path``.

    Handles both the real filenames (aqi_data.csv, ...) and the sample
    filenames (aqi_sample.csv, ...). Prints the shape of each dataframe.
    Returns ``(aqi_df, hospital_df, weather_df)``.
    """
    # Decide filename set based on whether we're pointing at sample_data.
    if "sample_data" in data_path:
        aqi_file = "aqi_sample.csv"
        hospital_file = "hospital_sample.csv"
        weather_file = "weather_sample.csv"
    else:
        aqi_file = "aqi_data.csv"
        hospital_file = "hospital_data.csv"
        weather_file = "weather_data.csv"

    aqi_df = pd.read_csv(os.path.join(data_path, aqi_file))
    hospital_df = pd.read_csv(os.path.join(data_path, hospital_file))
    weather_df = pd.read_csv(os.path.join(data_path, weather_file))

    print(f"Loaded AQI data:      shape {aqi_df.shape}")
    print(f"Loaded hospital data: shape {hospital_df.shape}")
    print(f"Loaded weather data:  shape {weather_df.shape}")

    return aqi_df, hospital_df, weather_df


def _clean_common(df: pd.DataFrame) -> pd.DataFrame:
    """Shared cleaning steps: parse date, normalize city, ffill, dedupe."""
    df = df.copy()

    # Parse and re-format the date column to a consistent YYYY-MM-DD string.
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    # Strip whitespace and capitalize city names consistently.
    if "city" in df.columns:
        df["city"] = df["city"].astype(str).str.strip().str.title()

    # Forward fill missing values, then remove duplicate rows.
    df = df.ffill()
    df = df.drop_duplicates()

    return df


def clean_aqi(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the AQI dataframe."""
    df = _clean_common(df)
    # Make sure pollutant/AQI columns are numeric.
    for col in ["PM2.5", "PM10", "NO2", "SO2", "AQI"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.ffill()
    return df


def clean_hospital(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the hospital dataframe and ensure admissions are numeric."""
    df = _clean_common(df)
    if "respiratory_admissions" in df.columns:
        df["respiratory_admissions"] = pd.to_numeric(
            df["respiratory_admissions"], errors="coerce"
        )
    df = df.ffill()
    return df


def clean_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the weather dataframe and ensure numeric weather columns."""
    df = _clean_common(df)
    for col in ["temperature", "humidity", "wind_speed"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.ffill()
    return df


def merge_datasets(aqi_df: pd.DataFrame, hospital_df: pd.DataFrame,
                   weather_df: pd.DataFrame) -> pd.DataFrame:
    """Merge AQI + hospital + weather on (date, city) with inner joins."""
    merged = pd.merge(aqi_df, hospital_df, on=["date", "city"], how="inner")
    merged = pd.merge(merged, weather_df, on=["date", "city"], how="inner")

    print(f"Merged dataset shape:   {merged.shape}")
    print(f"Merged dataset columns: {list(merged.columns)}")
    print("Missing values per column:")
    print(merged.isna().sum().to_string())

    return merged


def run_cleaning():
    """Run the full cleaning pipeline and save the merged dataset.

    Returns ``(merged_df, using_sample_bool)``.
    """
    # 1. Decide data source.
    data_path, using_sample = check_data_source()

    # 2. Load raw datasets.
    aqi_df, hospital_df, weather_df = load_datasets(data_path)

    # 3. Clean each dataset.
    aqi_df = clean_aqi(aqi_df)
    hospital_df = clean_hospital(hospital_df)
    weather_df = clean_weather(weather_df)

    # 4. Merge into one dataframe.
    merged_df = merge_datasets(aqi_df, hospital_df, weather_df)

    # 5. Persist the merged output.
    os.makedirs("output", exist_ok=True)
    merged_df.to_csv(os.path.join("output", "merged_data.csv"), index=False)

    print("Data cleaning complete")
    return merged_df, using_sample


if __name__ == "__main__":
    run_cleaning()
