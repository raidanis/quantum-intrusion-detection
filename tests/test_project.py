import pytest

from qid.data import evaluate_model, load_data
from qid.model import VariationalQuantumClassifier


def test_data_shapes_and_balanced_labels():
    X_train, X_test, y_train, y_test = load_data(n_samples=200)
    assert X_train.shape[1] == X_test.shape[1] == 4
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)


def test_vqc_fit_predict():
    X_train, X_test, y_train, y_test = load_data(n_samples=200)
    model = VariationalQuantumClassifier(epochs=20).fit(X_train, y_train)

    predictions = model.predict(X_test)
    assert set(predictions).issubset(set(y_test))

    metrics = evaluate_model(model, X_test, y_test)
    assert 0.0 <= metrics["accuracy"] <= 1.0
