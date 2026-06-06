import numpy as np


class TemperatureScaledRF:
    """
    Wraps a pre-trained Random Forest with temperature scaling.
    """

    def __init__(self, rf_model, temperature=3.0):
        self.rf = rf_model
        self.classes_ = rf_model.classes_
        self.temperature = temperature

    def predict_proba(self, X):
        raw_probs = self.rf.predict_proba(X)
        raw_probs = np.clip(raw_probs, 1e-10, 1.0)
        log_probs = np.log(raw_probs) / self.temperature
        # Numerically stable softmax
        exp_probs = np.exp(log_probs - log_probs.max(axis=1, keepdims=True))
        calibrated = exp_probs / exp_probs.sum(axis=1, keepdims=True)
        return calibrated

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[probs.argmax(axis=1)]
