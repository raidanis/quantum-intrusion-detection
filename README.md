# Quantum Machine Learning for Intrusion Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A reproducible Python project comparing a **quantum-inspired variational classifier** with classical models for binary network intrusion detection.

> **Scope:** The included classifier is dependency-light and quantum-inspired, not a circuit executed on quantum hardware. The optional Qiskit dependencies allow future replacement of the toy model with a real variational quantum circuit.

## What it does

- Generates a configurable, imbalance-aware binary intrusion dataset for repeatable experiments.
- Standardizes four extracted features.
- Trains logistic regression and random forest baselines.
- Trains a variational, parameterized binary classifier.
- Reports accuracy, precision, recall, and F1-score.
- Exports `results/metrics.csv` and `results/model_comparison.png`.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python main.py
pytest
```

## Project structure

```text
main.py                 Entry point
src/qid/data.py         Data generation and evaluation helpers
src/qid/model.py        Quantum-inspired variational classifier
src/qid/experiment.py   Experiment and result export
results/                Generated metrics and chart
tests/                  Automated tests
```

## Extend to real network data

Replace `load_data()` in `src/qid/data.py` with a loader for UNSW-NB15 or CICIDS2017. Encode only a small number of scaled features, split using stratification, and preserve the class names for binary or multiclass experiments. Never evaluate on the training split.

## Reproducibility

All random seeds are fixed. Metrics and output charts are generated from the same train/test split, making the comparison suitable for a first academic prototype. 

## License

MIT
