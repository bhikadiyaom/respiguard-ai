"""RespiGuard-AI — Streamlit web application.

A dark, clinical early-warning dashboard that predicts hospital respiratory
admission surges 72 hours in advance from AQI readings across six Indian cities.
"""

import os
from datetime import datetime

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration MUST be the first Streamlit call.
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RespiGuard-AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.styles import (  # noqa: E402
    apply_styles,
    render_page_header,
    render_sidebar,
    render_metric_card,
    render_data_mode_banner,
    render_risk_badge,
    render_recommendation_box,
    render_aqi_display,
    COLORS,
)
from src.data_cleaning import run_cleaning  # noqa: E402
from src.feature_engineering import (  # noqa: E402
    run_feature_engineering,
    add_season_encoding,
    add_festival_risk,
    add_pollutant_contribution,
    add_aqi_category,
)
from src.bayesian_model import (  # noqa: E402
    train_model,
    load_model,
    load_model_metrics,
    predict_surge_probability,
    MODEL_PATH,
    FEATURES,
)
from src.expert_system import apply_expert_rules, get_risk_color  # noqa: E402

# Inject the design system CSS.
apply_styles()

CITIES = ["Delhi", "Mumbai", "Chennai", "Kolkata", "Ahmedabad", "Bengaluru"]
SEASON_NUM = {"Winter": 0, "Summer": 1, "Monsoon": 2, "PostMonsoon": 3,
              "Post Monsoon": 3}
ENGINEERED_PATH = os.path.join("output", "engineered_data.csv")


# ---------------------------------------------------------------------------
# Startup: load or train the model (cached so it runs once per session).
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def initialize_system():
    """Ensure engineered data and a trained model exist; return everything."""
    # Train the model if it has never been trained.
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ENGINEERED_PATH):
        merged_df, using_sample = run_cleaning()
        engineered_df = run_feature_engineering(merged_df)
        train_model(engineered_df)
    else:
        # Determine data mode without re-running the full pipeline.
        real_path = os.path.join("data", "aqi_data.csv")
        using_sample = not (os.path.exists(real_path) and os.path.getsize(real_path) > 0)

    bundle, model_type = load_model()
    accuracy, f1, _ = load_model_metrics()
    engineered_df = pd.read_csv(ENGINEERED_PATH)
    return bundle, model_type, accuracy, f1, engineered_df, using_sample


def build_input_features(aqi, pm25, pm10, no2, so2, temperature, humidity,
                         wind_speed, season, festival_risk):
    """Build the full feature dict the model expects from raw UI inputs."""
    # Derive AQI category number.
    if aqi <= 100:
        cat_num = 0
    elif aqi <= 200:
        cat_num = 1
    elif aqi <= 300:
        cat_num = 2
    elif aqi <= 400:
        cat_num = 3
    else:
        cat_num = 4

    total = pm25 + pm10 + no2 + so2 + 1
    pm25_pct = (pm25 / total) * 100

    return {
        "AQI_Category_num": cat_num,
        "season_num": SEASON_NUM.get(season, 0),
        "festival_risk": int(festival_risk),
        # Without history we approximate lags/rolling by the current AQI.
        "AQI_lag_24hr": aqi,
        "AQI_lag_48hr": aqi,
        "AQI_lag_72hr": aqi,
        "AQI_rolling_7day": aqi,
        "PM25_pct": pm25_pct,
        "temperature": temperature,
        "humidity": humidity,
        "wind_speed": wind_speed,
    }


def compute_city_risk(row, bundle, model_type):
    """Compute the expert-system risk decision for one engineered data row."""
    features = {col: row.get(col, 0.0) for col in FEATURES}
    model_pred = predict_surge_probability(bundle, model_type, features)
    result = apply_expert_rules(
        aqi=row.get("AQI", 0),
        season=row.get("season_name", "Summer"),
        wind_speed=row.get("wind_speed", 15),
        humidity=row.get("humidity", 60),
        temperature=row.get("temperature", 25),
        festival_risk=row.get("festival_risk", 0),
        model_prediction=model_pred,
    )
    return result


# ===========================================================================
# PAGE 1 — DASHBOARD
# ===========================================================================
def page_dashboard(bundle, model_type, accuracy, f1, engineered_df, using_sample):
    """Render the dashboard page."""
    render_page_header("Hospital Respiratory Surge Predictor", version="V1.0")

    if using_sample:
        render_data_mode_banner(using_sample=True)
    else:
        render_data_mode_banner(using_sample=False)

    st.write("")

    # Row 1 — three metric cards.
    c1, c2, c3 = st.columns(3)
    with c1:
        render_metric_card("Cities Monitored", 6)
    with c2:
        render_metric_card("Model Accuracy", f"{accuracy * 100:.0f}", "%")
    with c3:
        render_metric_card("Model Type", model_type)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # City risk overview table.
    st.markdown(
        '<div class="main-subtitle" style="font-size:1.3rem;color:#F1F5F9;'
        'font-family:Space Grotesk;">Current City Risk Overview</div>',
        unsafe_allow_html=True,
    )

    rows_html = ""
    for city in CITIES:
        city_df = engineered_df[engineered_df["city"].str.title() == city]
        if city_df.empty:
            continue
        latest = city_df.sort_values("date").iloc[-1]
        result = compute_city_risk(latest, bundle, model_type)
        risk = result["risk_level"]
        color = get_risk_color(risk)
        festival = "Yes" if int(latest.get("festival_risk", 0)) == 1 else "No"
        rows_html += f"""
            <tr>
                <td>{city}</td>
                <td class="mono">{int(latest.get('AQI', 0))}</td>
                <td>{latest.get('AQI_Category', '-')}</td>
                <td><span style="color:{color};font-weight:600;">{risk}</span></td>
                <td>{latest.get('season_name', '-')}</td>
                <td>{festival}</td>
            </tr>
        """

    table_html = f"""
        <table class="risk-table">
            <thead>
                <tr>
                    <th>City</th>
                    <th>Latest AQI</th>
                    <th>AQI Category</th>
                    <th>Risk Level</th>
                    <th>Season</th>
                    <th>Festival Risk</th>
                </tr>
            </thead>
            <tbody>{rows_html}</tbody>
        </table>
    """
    st.markdown(table_html, unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Model performance section.
    st.markdown(
        '<div class="main-subtitle" style="font-size:1.3rem;color:#F1F5F9;'
        'font-family:Space Grotesk;">Model Performance</div>',
        unsafe_allow_html=True,
    )
    left, right = st.columns(2)
    with left:
        st.metric("Accuracy", f"{accuracy * 100:.1f}%")
        st.metric("F1 Score", f"{f1:.3f}")
    with right:
        st.markdown(
            f"""
            <div class="metric-card">
              <div class="card-label">What the model predicts</div>
              <p style="color:#94A3B8;font-family:Inter;margin-top:0.5rem;">
              RespiGuard-AI estimates the probability of a
              <span class="teal-accent">respiratory admission surge</span>
              in the next 24, 48, and 72 hours based on air quality, weather,
              seasonal, and festival signals. Expert clinical rules refine the
              output into an actionable risk level and hospital preparation
              recommendation. Current engine: <b>{model_type}</b>.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Last updated timestamp, bottom right.
    st.markdown(
        f'<div style="text-align:right;color:#94A3B8;font-family:JetBrains Mono;'
        f'font-size:0.75rem;margin-top:1rem;">Last updated: '
        f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>',
        unsafe_allow_html=True,
    )


# ===========================================================================
# PAGE 2 — 72 HOUR PREDICTOR
# ===========================================================================
def page_predictor(bundle, model_type):
    """Render the 72-hour admission-surge predictor page."""
    render_page_header(
        "72 Hour Admission Surge Predictor",
        subtitle="Enter today's AQI readings to predict hospital admission surge",
    )

    left, right = st.columns(2)

    with left:
        st.markdown(
            '<div class="card-label">AQI Inputs</div>', unsafe_allow_html=True)
        city = st.selectbox("City", CITIES)
        aqi = st.slider("AQI", 0, 500, 150)
        pm25 = st.slider("PM2.5", 0, 300, 60)
        pm10 = st.slider("PM10", 0, 300, 80)
        no2 = st.slider("NO2", 0, 200, 30)
        so2 = st.slider("SO2", 0, 200, 20)

    with right:
        st.markdown(
            '<div class="card-label">Weather Inputs</div>', unsafe_allow_html=True)
        temperature = st.slider("Temperature (°C)", 0, 50, 25)
        humidity = st.slider("Humidity (%)", 0, 100, 60)
        wind_speed = st.slider("Wind Speed (km/h)", 0, 60, 15)
        season = st.selectbox("Season", ["Winter", "Summer", "Monsoon", "Post Monsoon"])
        festival = st.checkbox("Is today a festival period? (Diwali, Holi etc.)")

    st.write("")
    predict = st.button("🔍 Predict Admission Surge")

    if not predict:
        return

    # Build features and run model + expert system.
    features = build_input_features(
        aqi, pm25, pm10, no2, so2, temperature, humidity, wind_speed,
        season, festival,
    )
    model_pred = predict_surge_probability(bundle, model_type, features)
    result = apply_expert_rules(
        aqi=aqi, season=season, wind_speed=wind_speed, humidity=humidity,
        temperature=temperature, festival_risk=int(festival),
        model_prediction=model_pred,
    )

    risk = result["risk_level"]
    confidence = result["confidence_pct"]

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown(
        '<div class="main-subtitle" style="font-size:1.3rem;color:#F1F5F9;'
        'font-family:Space Grotesk;">Prediction Results</div>',
        unsafe_allow_html=True,
    )

    # Row 1: AQI display + risk badge.
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        render_aqi_display(aqi, city)
    with r1c2:
        render_risk_badge(risk)
        st.markdown(
            f'<div style="font-family:JetBrains Mono;color:#00D4AA;'
            f'margin-top:0.75rem;">Confidence: {confidence}%</div>',
            unsafe_allow_html=True,
        )

    # Row 2: three forecast cards.
    def surge_color(val):
        if val < 30:
            return COLORS["risk_low"]
        if val < 55:
            return COLORS["risk_medium"]
        if val < 75:
            return COLORS["risk_high"]
        return COLORS["risk_critical"]

    s24 = model_pred["surge_24hr_pct"]
    s48 = model_pred["surge_48hr_pct"]
    s72 = model_pred["surge_72hr_pct"]

    forecast_html = f"""
    <div class="forecast-container">
        <div class="forecast-bar-item">
            <div class="forecast-time">24 Hour Surge</div>
            <div class="forecast-value" style="color:{surge_color(s24)};">{s24}%</div>
        </div>
        <div class="forecast-bar-item">
            <div class="forecast-time">48 Hour Surge</div>
            <div class="forecast-value" style="color:{surge_color(s48)};">{s48}%</div>
        </div>
        <div class="forecast-bar-item">
            <div class="forecast-time">72 Hour Surge</div>
            <div class="forecast-value" style="color:{surge_color(s72)};">{s72}%</div>
        </div>
    </div>
    """
    st.markdown(forecast_html, unsafe_allow_html=True)

    # Recommendation box.
    render_recommendation_box(result["recommendation"], risk)

    # Rule triggered note.
    st.markdown(
        f'<div style="color:#94A3B8;font-family:Inter;font-size:0.8rem;'
        f'margin-top:0.5rem;">Triggered by: {result["rule_triggered"]}</div>',
        unsafe_allow_html=True,
    )

    # Critical alert.
    if risk == "CRITICAL":
        st.warning("🚨 CRITICAL ALERT: Contact hospital administration immediately")


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    """Application entry point: initialize, render sidebar, route pages."""
    with st.spinner("Initializing RespiGuard-AI..."):
        bundle, model_type, accuracy, f1, engineered_df, using_sample = \
            initialize_system()

    pages = ["Dashboard", "72 Hour Predictor"]
    selected = render_sidebar(pages)

    # Data mode indicator in the sidebar.
    if using_sample:
        st.sidebar.markdown(
            '<div style="color:#7DD3FC;font-family:Inter;font-size:0.85rem;">'
            '🔵 Sample Data Mode</div>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown(
            '<div style="color:#86EFAC;font-family:Inter;font-size:0.85rem;">'
            '🟢 Real Data Active</div>', unsafe_allow_html=True)

    st.sidebar.markdown(
        '<div class="sidebar-footer">V1.0 | Built by Om</div>',
        unsafe_allow_html=True,
    )

    with st.container():
        if selected == "Dashboard":
            page_dashboard(bundle, model_type, accuracy, f1, engineered_df,
                           using_sample)
        else:
            page_predictor(bundle, model_type)


if __name__ == "__main__":
    main()
