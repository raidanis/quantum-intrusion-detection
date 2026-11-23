import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class VariationalQuantumClassifier(BaseEstimator, ClassifierMixin):
    """Small NumPy-based variational classifier for dependency-light demos."""

    def __init__(self, learning_rate: float = 0.05, epochs: int = 100, random_state: int = 42):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_state = random_state

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        self.classes_, counts = np.unique(y, return_counts=True)
        if len(self.classes_) != 2:
            raise ValueError("VariationalQuantumClassifier requires binary labels.")

        positive_label = self.classes_[1]
        y_binary = (y == positive_label).astype(float)
        self.weights_ = rng.normal(0.0, 0.1, X.shape[1])

        X = np.clip(X, -2.5, 2.5)
        for _ in range(self.epochs):
            logits = np.tanh(X @ self.weights_)
            probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))
            gradient = X.T @ (probabilities - y_binary) / len(X)
            self.weights_ -= self.learning_rate * gradient
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        logits = np.tanh(X @ self.weights_)
        positive = 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X):
        positive = self.predict_proba(X)[:, 1]
        return np.asarray([self.classes_[1] if value >= 0.5 else self.classes_[0]
                           for value in positive])
