import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from data_preprocessing import DataPreprocessing


# ----------------------------------
# Load Processed Data
# ----------------------------------

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


# ----------------------------------
# Load Saved Model
# ----------------------------------

model = joblib.load(
    "models/intrusion_model.pkl"
)


# ----------------------------------
# Predictions
# ----------------------------------

y_pred = model.predict(X_test)


# ----------------------------------
# Metrics
# ----------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted"
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted"
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted"
)


print("\nModel Evaluation")
print("-" * 50)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


# ----------------------------------
# Classification Report
# ----------------------------------

print("\nClassification Report\n")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ----------------------------------
# Confusion Matrix
# ----------------------------------

cm = confusion_matrix(
    y_test,
    y_pred
)

plt.figure(
    figsize=(10, 8)
)

sns.heatmap(
    cm,
    annot=False,
    cmap="Blues"
)

plt.title(
    "Confusion Matrix"
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.show()