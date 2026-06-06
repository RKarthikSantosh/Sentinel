import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from data_preprocessing import DataPreprocessing


# -----------------------------
# Load Processed Data
# -----------------------------
preprocessing = DataPreprocessing(
    "data/raw/KDDTrain+.txt"
)

(
    X_train,
    X_test,
    y_train,
    y_test,
    scaler,
    target_encoder
) = preprocessing.preprocess()


# -----------------------------
# Models
# -----------------------------
models = {

    "Logistic Regression":
        LogisticRegression(max_iter=1000),

    "Decision Tree":
        DecisionTreeClassifier(
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
}


best_model = None
best_accuracy = 0


# -----------------------------
# Training Loop
# -----------------------------
for name, model in models.items():

    print("\n" + "="*50)
    print(name)
    print("="*50)

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print(
        classification_report(
            y_test,
            predictions
        )
    )

    if accuracy > best_accuracy:

        best_accuracy = accuracy

        best_model = model


# -----------------------------
# Save Best Model
# -----------------------------
joblib.dump(
    best_model,
    "models/intrusion_model.pkl"
)

print("\nBest Model Saved")
print(f"Best Accuracy: {best_accuracy:.4f}")

# -----------------------------
# Save Random Forest Explicitly
# -----------------------------
rf_model = models["Random Forest"]
joblib.dump(rf_model, "models/rf_model.pkl")
print("Random Forest model saved as rf_model.pkl")