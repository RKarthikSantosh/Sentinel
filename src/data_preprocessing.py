import pandas as pd
import joblib
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class DataPreprocessing:

    def __init__(self, filepath):
        self.filepath = filepath

    def preprocess(self):

        # Load Dataset
        df = pd.read_csv(
            self.filepath,
            header=None
        )

        # Rename Target Column
        df.rename(
            columns={41: "label"},
            inplace=True
        )

        # Remove Difficulty Score
        if 42 in df.columns:
            df.drop(
                columns=[42],
                inplace=True
            )

        print("Dataset Shape:", df.shape)

        # Features & Target
        X = df.drop(
            columns=["label"]
        )

        y = df["label"]

        # Categorical Columns
        categorical_cols = [1, 2, 3]

        feature_encoders = {}

        for col in categorical_cols:

            encoder = LabelEncoder()

            X[col] = encoder.fit_transform(
                X[col]
            )

            feature_encoders[col] = encoder

        # Target Encoding
        target_encoder = LabelEncoder()

        y = target_encoder.fit_transform(y)

        # Train Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        # Scaling
        scaler = StandardScaler()

        X_train = scaler.fit_transform(X_train)

        X_test = scaler.transform(X_test)

        # Create models folder
        os.makedirs(
            "models",
            exist_ok=True
        )

        print("\nSaving Files...")

        # Save Scaler
        joblib.dump(
            scaler,
            "models/scaler.pkl"
        )

        # Save Target Encoder
        joblib.dump(
            target_encoder,
            "models/target_encoder.pkl"
        )

        # Save Feature Encoders
        joblib.dump(
            feature_encoders,
            "models/feature_encoders.pkl"
        )

        print("feature_encoders.pkl saved successfully")

        print("\nPreprocessing Completed Successfully")

        print(
            "X_train Shape:",
            X_train.shape
        )

        print(
            "X_test Shape:",
            X_test.shape
        )

        return (
            X_train,
            X_test,
            y_train,
            y_test,
            scaler,
            target_encoder
        )


if __name__ == "__main__":

    preprocessing = DataPreprocessing(
        "data/raw/KDDTrain+.txt"
    )

    preprocessing.preprocess()