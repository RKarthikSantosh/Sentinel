import pandas as pd
import joblib

from attack_mapper import get_attack_description
from risk_scoring import calculate_risk


# Load saved objects

model = joblib.load(
    "models/intrusion_model.pkl"
)

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

    # Remove label and difficulty score

    if 41 in df.columns:
        df.drop(columns=[41], inplace=True)

    if 42 in df.columns:
        df.drop(columns=[42], inplace=True)

    # Encode categorical columns

    categorical_cols = [1, 2, 3]

    for col in categorical_cols:

        df[col] = (
            feature_encoders[col]
            .transform(df[col])
        )

    # Scale

    scaled_data = scaler.transform(df)

    # Prediction

    prediction = model.predict(
        scaled_data
    )

    probabilities = (
        model.predict_proba(
            scaled_data
        )
    )

    confidence = (
        probabilities.max(axis=1)[0]
    )

    # Decode attack

    attack_name = (
        target_encoder
        .inverse_transform(prediction)
    )[0]

    attack_description = (
        get_attack_description(
            attack_name
        )
    )

    # Do not score normal traffic as a threat
    if attack_name.lower() == "normal":
        return {
            "attack_name": attack_name,
            "attack_description": attack_description,
            "confidence": round(confidence * 100, 2),
            "risk_score": 0,
            "threat_level": "None",
        }

    risk_score, threat_level = (
        calculate_risk(
            confidence
        )
    )

    return {

        "attack_name":
            attack_name,

        "attack_description":
            attack_description,

        "confidence":
            round(
                confidence * 100,
                2
            ),

        "risk_score":
            risk_score,

        "threat_level":
            threat_level
    }


if __name__ == "__main__":

    sample = pd.read_csv(
        "data/raw/KDDTest+.txt",
        header=None
    )

    sample = sample.head(1)

    result = predict_attack(
        sample
    )

    print("\nPrediction Result\n")

    for key, value in result.items():

        print(
            f"{key}: {value}"
        )