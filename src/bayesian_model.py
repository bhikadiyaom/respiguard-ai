"""Bayesian model for RespiGuard-AI.

Primary model: pgmpy BayesianNetwork trained with MaximumLikelihoodEstimator.
Backup model:  scikit-learn GaussianNB (used automatically if pgmpy is
unavailable or fails at any point).

The trained artifact is a "bundle" dict saved with pickle that carries the
model, its type, and any metadata (e.g. discretization bin edges) needed to
make predictions later.
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score

# ---------------------------------------------------------------------------
# Try to import pgmpy. If anything goes wrong we transparently fall back to
# the scikit-learn GaussianNB backup at training time.
# ---------------------------------------------------------------------------
PGMPY_AVAILABLE = False
# _MLE_NEW_API distinguishes pgmpy >= 1.x (initialized estimator instance,
# e.g. DiscreteMLE()) from older releases (pass the estimator class).
_MLE_NEW_API = False
try:
    try:
        # Newer pgmpy versions.
        from pgmpy.models import DiscreteBayesianNetwork as _BayesianNetwork
    except Exception:
        # Older pgmpy versions.
        from pgmpy.models import BayesianNetwork as _BayesianNetwork

    try:
        # pgmpy >= 1.x: estimator lives in pgmpy.parameter_estimator and must
        # be passed as an initialized instance.
        from pgmpy.parameter_estimator import DiscreteMLE as _MLE
        _MLE_NEW_API = True
    except Exception:
        # Older pgmpy: MaximumLikelihoodEstimator class passed directly.
        from pgmpy.estimators import MaximumLikelihoodEstimator as _MLE
        _MLE_NEW_API = False

    from pgmpy.inference import VariableElimination
    PGMPY_AVAILABLE = True
except Exception:
    PGMPY_AVAILABLE = False


# Feature columns fed to the model, and the binary target.
FEATURES = [
    "AQI_Category_num", "season_num", "festival_risk",
    "AQI_lag_24hr", "AQI_lag_48hr", "AQI_lag_72hr",
    "AQI_rolling_7day", "PM25_pct", "temperature", "humidity", "wind_speed",
]
TARGET = "admission_surge"

MODEL_PATH = os.path.join("models", "trained_model_v1.pkl")

# Discrete parent nodes used by the pgmpy Bayesian network.
_PGMPY_NODES = ["season_num", "festival_risk", "AQI_Category_num",
                "temperature", "humidity", "admission_surge"]


def prepare_model_data(df: pd.DataFrame):
    """Select the feature columns and target, dropping rows with any NaN.

    Returns ``(X, y)`` where X is a DataFrame of features and y is the target.
    """
    cols = FEATURES + [TARGET]
    data = df[cols].copy()
    data = data.dropna(axis=0, how="any")
    X = data[FEATURES]
    y = data[TARGET].astype(int)
    return X, y


def _discretize(series: pd.Series, bins: int = 4):
    """Discretize a continuous series into integer bins, returning edges too."""
    codes, edges = pd.cut(series, bins=bins, labels=False, retbins=True,
                          duplicates="drop")
    codes = pd.Series(codes, index=series.index).fillna(0).astype(int)
    return codes, edges


def _apply_bins(value: float, edges) -> int:
    """Map a single value to a bin index using previously computed edges."""
    # np.digitize returns 1..len(edges); shift to 0-based and clamp.
    idx = int(np.digitize([value], edges[1:-1])[0])
    max_bin = len(edges) - 2
    return int(np.clip(idx, 0, max(max_bin, 0)))


def train_bayesian_pgmpy(df: pd.DataFrame):
    """Train the pgmpy Bayesian network and return a model bundle dict.

    The network structure encodes how season/festival drive AQI category and
    how AQI category, temperature, and humidity drive the admission surge.
    Continuous temperature/humidity are discretized before fitting.
    """
    # Assemble the columns the network needs and drop incomplete rows.
    needed = ["season_num", "festival_risk", "AQI_Category_num",
              "temperature", "humidity", TARGET]
    data = df[needed].dropna(axis=0, how="any").copy()

    # Discretize the continuous parents into integer bins.
    data["temperature"], temp_edges = _discretize(data["temperature"], bins=4)
    data["humidity"], humid_edges = _discretize(data["humidity"], bins=4)

    # Ensure integer typing for all discrete nodes.
    for col in needed:
        data[col] = data[col].astype(int)

    # Define the network structure exactly as specified.
    model = _BayesianNetwork([
        ("season_num", "AQI_Category_num"),
        ("festival_risk", "AQI_Category_num"),
        ("AQI_Category_num", "admission_surge"),
        ("temperature", "admission_surge"),
        ("humidity", "admission_surge"),
    ])

    # Fit conditional probability tables via Maximum Likelihood Estimation.
    # Newer pgmpy expects an initialized estimator instance; older releases
    # accept the estimator class itself.
    if _MLE_NEW_API:
        model.fit(data, estimator=_MLE())
    else:
        model.fit(data, estimator=_MLE)

    bundle = {
        "model": model,
        "model_type": "pgmpy",
        "temp_edges": temp_edges,
        "humid_edges": humid_edges,
        "base_rate": float(df[TARGET].mean()),
    }
    return bundle


def train_sklearn_backup(X: pd.DataFrame, y: pd.Series):
    """Train a GaussianNB backup model. Returns ``(model, accuracy, f1)``."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    model = GaussianNB()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, preds))
    f1 = float(f1_score(y_test, preds, zero_division=0))
    return model, accuracy, f1


def _evaluate_pgmpy(bundle: dict, df: pd.DataFrame):
    """Evaluate a pgmpy bundle on a held-out split. Returns ``(accuracy, f1)``."""
    needed = ["season_num", "festival_risk", "AQI_Category_num",
              "temperature", "humidity", TARGET]
    data = df[needed].dropna(axis=0, how="any").copy()

    # Discretize using the bundle's stored edges for consistency.
    data["temperature"] = data["temperature"].apply(
        lambda v: _apply_bins(v, bundle["temp_edges"]))
    data["humidity"] = data["humidity"].apply(
        lambda v: _apply_bins(v, bundle["humid_edges"]))
    for col in needed:
        data[col] = data[col].astype(int)

    _, test = train_test_split(data, test_size=0.20, random_state=42)
    evidence = test.drop(columns=[TARGET])

    try:
        preds = bundle["model"].predict(evidence)[TARGET].values
        y_true = test[TARGET].values
        accuracy = float(accuracy_score(y_true, preds))
        f1 = float(f1_score(y_true, preds, zero_division=0))
        return accuracy, f1
    except Exception:
        # If pgmpy prediction fails for the eval set, report neutral metrics.
        return 0.0, 0.0


def train_model(df: pd.DataFrame):
    """Train the best available model and save it.

    Tries pgmpy first; on any failure falls back to sklearn GaussianNB.
    Returns ``(model_bundle, accuracy, f1, model_type_string)``.
    """
    X, y = prepare_model_data(df)

    bundle = None
    accuracy = 0.0
    f1 = 0.0
    model_type = "GaussianNB"

    if PGMPY_AVAILABLE:
        try:
            bundle = train_bayesian_pgmpy(df)
            accuracy, f1 = _evaluate_pgmpy(bundle, df)
            model_type = "pgmpy"
            # If pgmpy produced a degenerate model, prefer the backup.
            if accuracy <= 0.0:
                raise RuntimeError("pgmpy produced degenerate metrics")
        except Exception as exc:  # noqa: BLE001
            print(f"pgmpy training failed ({exc}); falling back to GaussianNB")
            bundle = None

    if bundle is None:
        sk_model, accuracy, f1 = train_sklearn_backup(X, y)
        bundle = {
            "model": sk_model,
            "model_type": "GaussianNB",
            "features": FEATURES,
            "base_rate": float(y.mean()),
        }
        model_type = "GaussianNB"

    bundle["model_type"] = model_type
    bundle.setdefault("features", FEATURES)

    print(f"Model type in use: {model_type}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"F1 score: {f1:.3f}")

    # Persist the bundle.
    os.makedirs("models", exist_ok=True)
    with open(MODEL_PATH, "wb") as fh:
        pickle.dump({"bundle": bundle, "accuracy": accuracy, "f1": f1,
                     "model_type": model_type}, fh)

    return bundle, accuracy, f1, model_type


def load_model():
    """Load the saved model bundle. Returns ``(bundle, model_type)``."""
    with open(MODEL_PATH, "rb") as fh:
        payload = pickle.load(fh)
    bundle = payload["bundle"]
    return bundle, payload.get("model_type", bundle.get("model_type", "GaussianNB"))


def load_model_metrics():
    """Load stored accuracy/f1/model_type without unpacking the model itself."""
    try:
        with open(MODEL_PATH, "rb") as fh:
            payload = pickle.load(fh)
        return (float(payload.get("accuracy", 0.0)),
                float(payload.get("f1", 0.0)),
                payload.get("model_type", "GaussianNB"))
    except Exception:
        return 0.0, 0.0, "GaussianNB"


def _predict_prob_sklearn(bundle: dict, input_features: dict) -> float:
    """Return P(surge=1) from the GaussianNB backup for the given features."""
    row = [float(input_features.get(col, 0.0)) for col in FEATURES]
    X = np.array(row, dtype=float).reshape(1, -1)
    model = bundle["model"]
    proba = model.predict_proba(X)[0]
    # Locate the probability for class label 1 (surge).
    classes = list(model.classes_)
    if 1 in classes:
        return float(proba[classes.index(1)])
    return float(proba.max())


def _predict_prob_pgmpy(bundle: dict, input_features: dict) -> float:
    """Return P(surge=1) from the pgmpy network for the given features."""
    model = bundle["model"]
    infer = VariableElimination(model)

    evidence = {
        "season_num": int(input_features.get("season_num", 0)),
        "festival_risk": int(input_features.get("festival_risk", 0)),
        "AQI_Category_num": int(input_features.get("AQI_Category_num", 0)),
        "temperature": _apply_bins(
            float(input_features.get("temperature", 25.0)), bundle["temp_edges"]),
        "humidity": _apply_bins(
            float(input_features.get("humidity", 60.0)), bundle["humid_edges"]),
    }

    # Only keep evidence for states the model actually learned.
    valid_evidence = {}
    for node, val in evidence.items():
        try:
            states = model.get_cpds(node).state_names[node]
            if val in states:
                valid_evidence[node] = val
        except Exception:
            continue

    result = infer.query(variables=[TARGET], evidence=valid_evidence,
                         show_progress=False)
    state_names = result.state_names[TARGET]
    values = result.values
    if 1 in state_names:
        return float(values[state_names.index(1)])
    return float(values.max())


def predict_surge_probability(model, model_type: str, input_features: dict) -> dict:
    """Predict the 24/48/72-hour admission-surge probabilities.

    ``model`` is the loaded bundle dict. Returns a dict with 24hr/48hr/72hr
    surge percentages and a confidence percentage, each capped at 95%.
    """
    bundle = model  # the "model" passed around is the bundle dict

    # Compute the base 24hr probability using whichever engine is available.
    try:
        if model_type == "pgmpy" and PGMPY_AVAILABLE:
            p24 = _predict_prob_pgmpy(bundle, input_features)
        else:
            p24 = _predict_prob_sklearn(bundle, input_features)
    except Exception:
        # Robust heuristic fallback so the app never crashes on prediction.
        cat = float(input_features.get("AQI_Category_num", 0))
        p24 = min(0.15 + cat * 0.20, 0.95)

    # Convert to percentages; surges build up over the horizon.
    surge_24 = min(p24 * 100.0, 95.0)
    surge_48 = min(p24 * 1.15 * 100.0, 95.0)
    surge_72 = min(p24 * 1.30 * 100.0, 95.0)

    # Confidence: distance of probability from the 50/50 decision boundary.
    confidence = min(50.0 + abs(p24 - 0.5) * 100.0, 95.0)

    return {
        "surge_24hr_pct": round(surge_24, 1),
        "surge_48hr_pct": round(surge_48, 1),
        "surge_72hr_pct": round(surge_72, 1),
        "confidence_pct": round(confidence, 1),
    }


if __name__ == "__main__":
    from src.data_cleaning import run_cleaning
    from src.feature_engineering import run_feature_engineering

    merged, _ = run_cleaning()
    engineered = run_feature_engineering(merged)
    train_model(engineered)
