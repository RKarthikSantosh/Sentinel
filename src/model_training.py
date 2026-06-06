import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from data_preprocessing import DataPreprocessing, _repo_path
from calibration import TemperatureScaledRF


# --------------------------------------------------
# 1. Load & preprocess training data
#    - Primary   : full KDDTrain+.txt  (125 k rows)
#    - Extra     : KDDTrain+_20Percent.txt (subset — dupes auto-dropped)
# --------------------------------------------------
preprocessing = DataPreprocessing(
    extra_train_paths=[
        _repo_path("NSL-KDD-Dataset-master", "KDDTrain+_20Percent.txt"),
    ],
)

X_train, X_val, y_train, y_val, scaler, target_encoder = (
    preprocessing.preprocess()
)

# Load the fitted encoders for test-set evaluation
import joblib as _jl
feature_encoders = _jl.load(_repo_path("models", "feature_encoders.pkl"))


# --------------------------------------------------
# 2. Models
# --------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree":        DecisionTreeClassifier(random_state=42),
    "Random Forest":        RandomForestClassifier(
                                n_estimators=100,
                                random_state=42,
                                n_jobs=-1,       # use all CPU cores
                            ),
}


# --------------------------------------------------
# 3. Training loop — evaluate on validation split
# --------------------------------------------------
best_model    = None
best_accuracy = 0

for name, model in models.items():
    print("\n" + "=" * 55)
    print(f"  {name}")
    print("=" * 55)

    model.fit(X_train, y_train)

    preds    = model.predict(X_val)
    accuracy = accuracy_score(y_val, preds)
    print(f"  Validation Accuracy : {accuracy:.4f}")
    print(classification_report(y_val, preds, zero_division=0))

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model    = model


# --------------------------------------------------
# 4. Wrap the Random Forest with temperature scaling
# --------------------------------------------------
rf_raw = models["Random Forest"]

TEMPERATURE = 3.0   # T > 1 softens; empirically chosen

print("\n" + "=" * 55)
print(f"  Temperature Scaling (T={TEMPERATURE})")
print("=" * 55)


calibrated_rf = TemperatureScaledRF(rf_raw, temperature=TEMPERATURE)

# Verify calibrated model retains accuracy
cal_preds = calibrated_rf.predict(X_val)
cal_accuracy = accuracy_score(y_val, cal_preds)
print(f"  Calibrated RF Validation Accuracy : {cal_accuracy:.4f}")

# Show probability distribution on validation set
cal_probas = calibrated_rf.predict_proba(X_val)
cal_max_probs = cal_probas.max(axis=1)
print(f"  Mean  max-probability (calibrated): {cal_max_probs.mean():.4f}")
print(f"  Median max-probability (calibrated): {np.median(cal_max_probs):.4f}")
print(f"  Preds with P >= 0.99: {(cal_max_probs >= 0.99).sum()} / {len(cal_max_probs)} "
      f"({(cal_max_probs >= 0.99).sum() / len(cal_max_probs) * 100:.1f}%)")
print(f"  Preds with P == 1.00: {(cal_max_probs >= 1.0 - 1e-9).sum()} / {len(cal_max_probs)} "
      f"({(cal_max_probs >= 1.0 - 1e-9).sum() / len(cal_max_probs) * 100:.1f}%)")


# --------------------------------------------------
# 5. Evaluate on BOTH NSL-KDD held-out test sets
# --------------------------------------------------
test_sets = [
    ("KDDTest+ (full)",     _repo_path("data", "raw", "KDDTest+.txt")),
    ("KDDTest-21 (harder)", _repo_path("NSL-KDD-Dataset-master", "KDDTest-21.txt")),
]

print("\n" + "=" * 55)
print("  Evaluation on held-out test sets (Calibrated Random Forest)")
print("=" * 55)

for label, path in test_sets:
    X_t, y_t = preprocessing.preprocess_test(
        path, scaler, target_encoder, feature_encoders
    )
    preds    = calibrated_rf.predict(X_t)
    accuracy = accuracy_score(y_t, preds)

    # Also check probability distribution on test set
    test_probas = calibrated_rf.predict_proba(X_t)
    test_max_probs = test_probas.max(axis=1)

    print(f"\n  {label}")
    print(f"  Accuracy : {accuracy:.4f}")
    print(f"  Mean  max-prob: {test_max_probs.mean():.4f}")
    print(f"  P >= 0.99: {(test_max_probs >= 0.99).sum()} / {len(test_max_probs)} "
          f"({(test_max_probs >= 0.99).sum() / len(test_max_probs) * 100:.1f}%)")
    print(classification_report(y_t, preds, zero_division=0))


# --------------------------------------------------
# 6. Save models
# --------------------------------------------------
joblib.dump(best_model, _repo_path("models", "intrusion_model.pkl"))
print(f"\nBest model saved  -> models/intrusion_model.pkl")
print(f"Best Val Accuracy -> {best_accuracy:.4f}")

joblib.dump(calibrated_rf, _repo_path("models", "rf_model.pkl"))
print("Calibrated Random Forest saved -> models/rf_model.pkl")