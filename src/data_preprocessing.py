import pandas as pd
import joblib
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# Column names from the NSL-KDD feature spec
# --------------------------------------------------
FEATURE_NAMES = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count",
    "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "label",        # col 41 — attack type name
    "difficulty",   # col 42 — difficulty score (dropped)
]

CATEGORICAL_COLS = ["protocol_type", "service", "flag"]


def load_txt(filepath):
    """Load a NSL-KDD .txt file and return a clean DataFrame."""
    df = pd.read_csv(filepath, header=None, names=FEATURE_NAMES)
    if "difficulty" in df.columns:
        df.drop(columns=["difficulty"], inplace=True)
    return df


class DataPreprocessing:

    def __init__(
        self,
        train_path="data/raw/KDDTrain+.txt",
        extra_train_paths=None,
    ):
        """
        Parameters
        ----------
        train_path : str
            Path to the primary training file (KDDTrain+.txt).
        extra_train_paths : list[str] | None
            Additional .txt files whose rows are appended to the
            training set before fitting (e.g. KDDTrain+_20Percent.txt).
            Duplicates are dropped automatically.
        """
        self.train_path = train_path
        self.extra_train_paths = extra_train_paths or []

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def preprocess(self):
        """
        Load, combine, encode, scale and split the training data.

        Returns
        -------
        X_train, X_test, y_train, y_test, scaler, target_encoder
        """
        # 1. Load primary training set
        train_df = load_txt(self.train_path)
        print(f"Primary train file : {self.train_path} -> {train_df.shape}")

        # 2. Append any extra training files (drop exact duplicates)
        for path in self.extra_train_paths:
            extra = load_txt(path)
            print(f"Extra train file   : {path} -> {extra.shape}")
            train_df = pd.concat([train_df, extra], ignore_index=True)
            before = len(train_df)
            train_df.drop_duplicates(inplace=True)
            dropped = before - len(train_df)
            print(f"  After dedup: {len(train_df)} rows ({dropped} duplicates removed)")

        print(f"\nCombined training set: {train_df.shape}")

        # 3. Split features / target
        X = train_df.drop(columns=["label"])
        y = train_df["label"]

        # 4. Fit encoders on training data only
        feature_encoders = {}
        for col in CATEGORICAL_COLS:
            enc = LabelEncoder()
            X[col] = enc.fit_transform(X[col].astype(str))
            feature_encoders[col] = enc

        target_encoder = LabelEncoder()
        y_enc = target_encoder.fit_transform(y)

        print(f"Attack classes : {len(target_encoder.classes_)} "
              f"-> {sorted(target_encoder.classes_)}\n")

        # 5. Train / validation split (stratified)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_enc,
            test_size=0.2,
            random_state=42,
            stratify=y_enc,
        )

        # 6. Scale
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val   = scaler.transform(X_val)

        print(f"X_train : {X_train.shape}")
        print(f"X_val   : {X_val.shape}")

        # 7. Persist artefacts
        os.makedirs("models", exist_ok=True)
        joblib.dump(scaler,           "models/scaler.pkl")
        joblib.dump(target_encoder,   "models/target_encoder.pkl")
        joblib.dump(feature_encoders, "models/feature_encoders.pkl")
        print("\nArtefacts saved -> models/")

        return X_train, X_val, y_train, y_val, scaler, target_encoder

    def preprocess_test(self, test_path, scaler, target_encoder, feature_encoders):
        """
        Apply the already-fitted encoders and scaler to a test file.

        Unknown categories (services/flags not seen during training)
        are mapped to a safe fallback instead of crashing.

        Returns
        -------
        X_test : np.ndarray
        y_test : np.ndarray
        """
        df = load_txt(test_path)
        print(f"\nTest file: {test_path} -> {df.shape}")

        X = df.drop(columns=["label"])
        y = df["label"]

        for col in CATEGORICAL_COLS:
            enc = feature_encoders[col]
            # Map unseen values to the first known class (safe fallback)
            known = set(enc.classes_)
            X[col] = X[col].astype(str).apply(
                lambda v: v if v in known else enc.classes_[0]
            )
            X[col] = enc.transform(X[col])

        # Map unseen attack labels to "normal" so they don't crash inverse_transform
        known_labels = set(target_encoder.classes_)
        y = y.apply(lambda v: v if v in known_labels else "normal")
        y_enc = target_encoder.transform(y)

        X_scaled = scaler.transform(X)
        print(f"X_test  : {X_scaled.shape}")
        return X_scaled, y_enc


if __name__ == "__main__":
    pp = DataPreprocessing(
        train_path="data/raw/KDDTrain+.txt",
        extra_train_paths=[
            "NSL-KDD-Dataset-master/KDDTrain+_20Percent.txt",
        ],
    )
    pp.preprocess()