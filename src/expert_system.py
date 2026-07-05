"""Expert system for RespiGuard-AI.

Applies a prioritized set of IF-THEN clinical rules on top of the model's
probabilistic prediction. High-severity rules can override the model, while
lower-severity rules blend with it.
"""

# Risk-level to hex-color map (mirrors the design system palette).
RISK_COLORS = {
    "LOW": "#22C55E",
    "LOW-MEDIUM": "#84CC16",
    "MEDIUM": "#EAB308",
    "HIGH": "#F97316",
    "CRITICAL": "#EF4444",
}

# Risk-level to emoji map for compact UI hints.
RISK_EMOJIS = {
    "LOW": "\u2705",          # ✅
    "LOW-MEDIUM": "\U0001F7E1",  # 🟡
    "MEDIUM": "\u26A0\uFE0F",   # ⚠️
    "HIGH": "\U0001F534",      # 🔴
    "CRITICAL": "\U0001F6A8",  # 🚨
}


def _normalize_season(season) -> str:
    """Normalize a season string for reliable comparison."""
    return str(season).strip().lower()


def apply_expert_rules(aqi, season, wind_speed, humidity, temperature,
                       festival_risk, model_prediction) -> dict:
    """Apply the prioritized clinical rules and return the final decision.

    ``model_prediction`` is the dict returned by
    ``predict_surge_probability``. Returns a dict describing the final risk
    level, recommendation, confidence, whether the model was overridden, and
    which rule triggered.
    """
    aqi = float(aqi)
    wind_speed = float(wind_speed)
    humidity = float(humidity)
    temperature = float(temperature)
    festival_risk = int(festival_risk)
    season_norm = _normalize_season(season)
    is_winter = season_norm.startswith("winter")

    # Base confidence taken from the model output (default 50%).
    model_conf = float(model_prediction.get("confidence_pct", 50.0)) \
        if isinstance(model_prediction, dict) else 50.0
    model_surge = float(model_prediction.get("surge_72hr_pct", 0.0)) \
        if isinstance(model_prediction, dict) else 0.0

    # ----- Rules in strict priority order -----

    # RULE 1: extreme AQI during a festival period.
    if aqi > 400 and festival_risk == 1:
        return _build_result(
            "CRITICAL",
            "Festival + extreme AQI detected. Activate full emergency "
            "respiratory protocol immediately.",
            override=True, model_conf=model_conf, model_surge=model_surge,
            rule="Rule 1: AQI > 400 during festival period")

    # RULE 2: critical AQI in stagnant winter air.
    if aqi > 400 and is_winter and wind_speed < 10:
        return _build_result(
            "CRITICAL",
            "Critical AQI in stagnant winter air. Activate emergency "
            "respiratory ward. Pre-position all respiratory equipment.",
            override=True, model_conf=model_conf, model_surge=model_surge,
            rule="Rule 2: AQI > 400, Winter, wind speed < 10 km/h")

    # RULE 3: critical AQI regardless of other conditions.
    if aqi > 400:
        return _build_result(
            "CRITICAL",
            "Critical AQI level. Emergency respiratory ward activation "
            "recommended immediately.",
            override=True, model_conf=model_conf, model_surge=model_surge,
            rule="Rule 3: AQI > 400")

    # RULE 4: severe AQI with high humidity.
    if aqi > 300 and humidity > 80:
        return _build_result(
            "HIGH",
            "Severe AQI with high humidity. Increase respiratory ward capacity "
            "by 30% and pre-stock nebulizers.",
            override=False, model_conf=model_conf, model_surge=model_surge,
            rule="Rule 4: AQI > 300 and humidity > 80%")

    # RULE 5: poor AQI in cold temperatures.
    if aqi > 200 and temperature < 15:
        return _build_result(
            "HIGH",
            "Poor AQI in cold temperatures greatly worsens respiratory "
            "conditions. Pre-stock nebulizers and bronchodilators.",
            override=False, model_conf=model_conf, model_surge=model_surge,
            rule="Rule 5: AQI > 200 and temperature < 15°C")

    # RULE 6: poor AQI.
    if aqi > 200:
        return _build_result(
            "MEDIUM",
            "Poor AQI detected. Issue public health advisory. Monitor ward "
            "capacity and prepare additional staff.",
            override=False, model_conf=model_conf, model_surge=model_surge,
            rule="Rule 6: AQI > 200")

    # RULE 7: moderate AQI.
    if aqi > 100:
        return _build_result(
            "LOW-MEDIUM",
            "Moderate AQI. Monitor situation. Keep respiratory ward on "
            "standby. No immediate action required.",
            override=False, model_conf=model_conf, model_surge=model_surge,
            rule="Rule 7: AQI > 100")

    # DEFAULT: good air quality.
    return _build_result(
        "LOW",
        "Good air quality. Normal hospital operations. No respiratory surge "
        "expected.",
        override=False, model_conf=model_conf, model_surge=model_surge,
        rule="Default: AQI <= 100")


def _build_result(risk_level: str, recommendation: str, override: bool,
                  model_conf: float, model_surge: float, rule: str) -> dict:
    """Assemble the standard expert-system result dict, computing confidence."""
    if override:
        # Overriding rules are high-certainty clinical mandates.
        confidence = min(max(model_conf, 90.0), 99.0)
    else:
        # Blend the model confidence with the surge signal for a final number.
        confidence = min(0.6 * model_conf + 0.4 * model_surge, 95.0)
        confidence = max(confidence, 50.0)

    return {
        "risk_level": risk_level,
        "recommendation": recommendation,
        "confidence_pct": round(float(confidence), 1),
        "override_applied": bool(override),
        "rule_triggered": rule,
    }


def get_risk_color(risk_level: str) -> str:
    """Return the hex color for a given risk level."""
    return RISK_COLORS.get(str(risk_level).strip().upper(), "#22C55E")


def get_risk_emoji(risk_level: str) -> str:
    """Return the emoji for a given risk level."""
    return RISK_EMOJIS.get(str(risk_level).strip().upper(), "\u2705")
