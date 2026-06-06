import os
import pandas as pd
import joblib

from attack_mapper import get_attack_description
from risk_scoring import calculate_risk
from calibration import TemperatureScaledRF  # noqa: F401 — needed for pickle


# Load saved objects — prefer the dedicated Random Forest model
_rf_path = "models/rf_model.pkl"
_fallback_path = "models/intrusion_model.pkl"
model = joblib.load(_rf_path if os.path.exists(_rf_path) else _fallback_path)

scaler = joblib.load(
    "models/scaler.pkl"
)

target_encoder = joblib.load(
    "models/target_encoder.pkl"
)

feature_encoders = joblib.load(
    "models/feature_encoders.pkl"
)


def predict_attack(df):
    df = df.copy()

    # --- Normalise column names ---
    # Input may come in as integer-indexed (CSV upload) or named (live monitor).
    # Rename integer columns to feature names so the rest of the function is uniform.
    int_to_name = {
        0: "duration", 1: "protocol_type", 2: "service", 3: "flag",
        41: "label", 42: "difficulty",
    }
    df.rename(columns={k: v for k, v in int_to_name.items() if k in df.columns},
              inplace=True)

    # Drop label / difficulty columns if present
    for col in ("label", "difficulty"):
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # --- Encode categorical columns ---
    for col in ("protocol_type", "service", "flag"):
        if col not in df.columns:
            continue
        enc   = feature_encoders[col]
        known = set(enc.classes_)
        df[col] = df[col].astype(str).apply(
            lambda v: v if v in known else enc.classes_[0]
        )
        df[col] = enc.transform(df[col])

    # --- Scale ---
    scaled_data = scaler.transform(df)

    # Prediction
    import numpy as np

    prediction = model.predict(scaled_data)
    probabilities = model.predict_proba(scaled_data)

    # Confidence = calibrated probability of the top class
    confidence = probabilities.max(axis=1)[0]

    # Margin = gap between the top-1 and top-2 class probabilities.
    sorted_probs = np.sort(probabilities[0])[::-1]
    margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 1.0

    # Decode attack
    attack_name = target_encoder.inverse_transform(prediction)[0]
    attack_description = get_attack_description(attack_name)

    # Do not score normal traffic as a threat
    if attack_name.lower() == "normal":
        return {
            "attack_name": attack_name,
            "attack_description": attack_description,
            "confidence": round(confidence * 100, 2),
            "risk_score": 0,
            "threat_level": "None",
        }

    risk_score, threat_level = calculate_risk(confidence, margin)

    return {
        "attack_name": attack_name,
        "attack_description": attack_description,
        "confidence": round(confidence * 100, 2),
        "risk_score": risk_score,
        "threat_level": threat_level,
    }


if __name__ == "__main__":
    from data_preprocessing import FEATURE_NAMES

    sample = pd.read_csv(
        "data/raw/KDDTest+.txt",
        header=None,
        names=FEATURE_NAMES,
    )

    result = predict_attack(sample.head(1))

    print("\nPrediction Result\n")
    for key, value in result.items():
        print(f"{key}: {value}")