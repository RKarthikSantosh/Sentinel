import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

from data_preprocessing import DataPreprocessing


# --------------------------------------------------
# 1. Load & preprocess training data
#    - Primary   : full KDDTrain+.txt  (125 k rows)
#    - Extra     : KDDTrain+_20Percent.txt (subset — dupes auto-dropped)
# --------------------------------------------------
preprocessing = DataPreprocessing(
    train_path="data/raw/KDDTrain+.txt",
    extra_train_paths=[
        "NSL-KDD-Dataset-master/KDDTrain+_20Percent.txt",
    ],
)

X_train, X_val, y_train, y_val, scaler, target_encoder = (
    preprocessing.preprocess()
)

# Load the fitted encoders for test-set evaluation
import joblib as _jl
feature_encoders = _jl.load("models/feature_encoders.pkl")


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
# 4. Evaluate on BOTH NSL-KDD held-out test sets
# --------------------------------------------------
test_sets = [
    ("KDDTest+ (full)",     "data/raw/KDDTest+.txt"),
    ("KDDTest-21 (harder)", "NSL-KDD-Dataset-master/KDDTest-21.txt"),
]

print("\n" + "=" * 55)
print("  Evaluation on held-out test sets (Random Forest)")
print("=" * 55)

rf = models["Random Forest"]

for label, path in test_sets:
    X_t, y_t = preprocessing.preprocess_test(
        path, scaler, target_encoder, feature_encoders
    )
    preds    = rf.predict(X_t)
    accuracy = accuracy_score(y_t, preds)
    print(f"\n  {label}")
    print(f"  Accuracy : {accuracy:.4f}")
    print(classification_report(y_t, preds, zero_division=0))


# --------------------------------------------------
# 5. Save models
# --------------------------------------------------
joblib.dump(best_model, "models/intrusion_model.pkl")
print(f"\nBest model saved  -> models/intrusion_model.pkl")
print(f"Best Val Accuracy -> {best_accuracy:.4f}")

joblib.dump(rf, "models/rf_model.pkl")
print("Random Forest saved -> models/rf_model.pkl")