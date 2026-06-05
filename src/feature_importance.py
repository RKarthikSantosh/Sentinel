import joblib
import pandas as pd
import matplotlib.pyplot as plt

from data_preprocessing import DataPreprocessing


# Load processed data
preprocessing = DataPreprocessing(
    "data/raw/KDDTrain+.txt"
)

X_train, X_test, y_train, y_test = (
    preprocessing.preprocess()
)

# Load model
model = joblib.load(
    "models/intrusion_model.pkl"
)

# Get feature importance
importance = model.feature_importances_

# Create dataframe
feature_df = pd.DataFrame({
    "Feature": range(len(importance)),
    "Importance": importance
})

# Sort descending
feature_df = feature_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 Features\n")
print(feature_df.head(10))

# Plot
plt.figure(figsize=(10,6))

plt.bar(
    feature_df.head(10)["Feature"].astype(str),
    feature_df.head(10)["Importance"]
)

plt.title(
    "Top 10 Important Features"
)

plt.xlabel(
    "Feature Number"
)

plt.ylabel(
    "Importance"
)

plt.show()