from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from qid.data import evaluate_model, load_data
from qid.model import VariationalQuantumClassifier


def run_experiment(output_dir: str = "results"):
    X_train, X_test, y_train, y_test = load_data()

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "quantum-inspired-vqc": VariationalQuantumClassifier(learning_rate=0.05, epochs=100),
    }

    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        rows.append({"model": name, **metrics})

    frame = pd.DataFrame(rows)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination / "metrics.csv", index=False)

    ax = frame.set_index("model")[["accuracy", "precision", "recall", "f1"]].plot.bar(figsize=(9, 5))
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title("Intrusion Detection Model Comparison")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(destination / "model_comparison.png", dpi=150)
    plt.close()

    print(frame.to_string(index=False))
    return frame


if __name__ == "__main__":
    run_experiment()
