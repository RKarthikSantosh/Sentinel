import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class DataPreprocessing:

    def __init__(self, filepath):
        self.filepath = filepath

    def preprocess(self):

        # Load dataset
        df = pd.read_csv(
            self.filepath,
            header=None
        )

        # Rename target column
        df.rename(
            columns={41: "label"},
            inplace=True
        )

        # Remove difficulty score
        df.drop(
            columns=[42],
            inplace=True
        )

        print("Dataset Shape:")
        print(df.shape)

        # Features and Target
        X = df.drop(
            columns=["label"]
        )

        y = df["label"]

        # Categorical columns
        categorical_cols = [1, 2, 3]

        # Store encoders
        feature_encoders = {}

        for col in categorical_cols:

            encoder = LabelEncoder()

            X[col] = encoder.fit_transform(
                X[col]
            )

            feature_encoders[col] = encoder

        # Encode target
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

        X_train = scaler.fit_transform(
            X_train
        )

        X_test = scaler.transform(
            X_test
        )

        # Save scaler
        joblib.dump(
            scaler,
            "models/scaler.pkl"
        )

        # Save target encoder
        joblib.dump(
            target_encoder,
            "models/target_encoder.pkl"
        )

        # NEW: Save feature encoders
        joblib.dump(
            feature_encoders,
            "models/feature_encoders.pkl"
        )

        print("\nPreprocessing Completed Successfully")

        print(
            "\nX_train Shape:",
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
            y_test
        )


if __name__ == "__main__":

    preprocessing = DataPreprocessing(
        "data/raw/KDDTrain+.txt"
    )

    X_train, X_test, y_train, y_test = (
        preprocessing.preprocess()
    )