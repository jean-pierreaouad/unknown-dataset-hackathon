"""Generate nine reproducible synthetic development datasets (not official data)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs, make_classification, make_moons, make_regression
from sklearn.model_selection import train_test_split


def create_suite(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    cases = []

    def save(name, task, frame, y, metric, k=None, feature_types=None):
        folder = output / name
        folder.mkdir(parents=True, exist_ok=True)
        config = {"task": task, "train": "train.csv", "seed": 42, "time_budget": 15,
                  "metric": metric, "output": "results"}
        if feature_types:
            config["feature_types"] = feature_types
        if task == "clustering":
            frame.to_csv(folder / "train.csv", index=False)
            pd.DataFrame({"target": y}).to_csv(folder / "truth.csv", index=False)
            config["n_clusters"] = k
        else:
            train_idx, test_idx = train_test_split(np.arange(len(frame)), test_size=0.25, random_state=42,
                                                  stratify=y if task == "classification" else None)
            train = frame.iloc[train_idx].copy()
            train["target"] = y[train_idx]
            train.to_csv(folder / "train.csv", index=False)
            frame.iloc[test_idx].to_csv(folder / "test.csv", index=False)
            pd.DataFrame({"target": y[test_idx]}).to_csv(folder / "truth.csv", index=False)
            config.update({"test": "test.csv", "target": "target"})
        (folder / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        cases.append({"name": name, "task": task, "config": f"{name}/config.json"})

    X, y = make_classification(n_samples=900, n_features=12, n_informative=7, n_redundant=2,
                               weights=[0.8, 0.2], class_sep=1.3, random_state=11)
    save("classification_imbalanced", "classification", pd.DataFrame(X), y, "balanced_accuracy")
    X, y = make_moons(n_samples=900, noise=0.16, random_state=12)
    frame = pd.DataFrame(X, columns=["x", "y"])
    frame["irrelevant"] = rng.normal(size=len(frame))
    save("classification_nonlinear", "classification", frame, y, "accuracy")
    size = 1000
    age = rng.normal(40, 12, size)
    service = rng.choice(["basic", "premium", "family"], size)
    signal = age / 20 + (service == "premium") * 2 + rng.normal(0, 0.6, size)
    y = np.where(signal > np.median(signal), "yes", "no")
    frame = pd.DataFrame({"age": age, "service": service, "amount": rng.lognormal(3, 1, size), "constant": 1})
    frame.loc[rng.choice(size, 110, replace=False), "age"] = np.nan
    frame["age_copy"] = frame["age"]
    frame["all_missing"] = np.nan
    save("classification_mixed", "classification", frame, y, "accuracy")
    X, y = make_regression(n_samples=1000, n_features=10, n_informative=7, noise=10, random_state=21)
    X[:, 0] *= 1e5
    save("regression_linear", "regression", pd.DataFrame(X), y, "rmse")
    X = rng.uniform(-3, 3, (1200, 6))
    y = 8 * np.sin(X[:, 0]) + 3 * X[:, 1] ** 2 + X[:, 2] * X[:, 3] + rng.normal(0, 0.5, len(X))
    save("regression_nonlinear", "regression", pd.DataFrame(X), y, "rmse")
    category = rng.choice(["A", "B", "C"], 1100)
    area = rng.uniform(30, 200, 1100)
    y = np.exp(4 + area / 100 + (category == "A") * 0.7 + rng.normal(0, 0.08, 1100))
    frame = pd.DataFrame({"area": area, "zone": category, "noise": rng.normal(size=1100), "constant": 0})
    frame.loc[rng.choice(1100, 80, replace=False), "area"] = np.nan
    save("regression_mixed_positive", "regression", frame, y, "rmsle")
    X, y = make_blobs(n_samples=800, centers=4, n_features=4, cluster_std=0.55, random_state=31)
    X[:, 0] *= 1000
    save("clustering_scaled", "clustering", pd.DataFrame(X), y, "silhouette", k=4)
    X, y = make_blobs(n_samples=800, centers=3, n_features=2, cluster_std=0.7, random_state=32)
    X = X @ np.array([[0.6, -0.6], [-0.4, 1.0]])
    save("clustering_anisotropic", "clustering", pd.DataFrame(X), y, "silhouette", k=3)
    X, y = make_blobs(n_samples=900, centers=3, n_features=3, cluster_std=0.5, random_state=33)
    frame = pd.DataFrame(X, columns=["x", "y", "z"])
    frame["segment"] = np.array(["red", "green", "blue"])[y]
    frame["constant"] = 1
    frame.loc[rng.choice(900, 55, replace=False), "z"] = np.nan
    frame["x_copy"] = frame["x"]
    save("clustering_mixed", "clustering", frame, y, "silhouette", k=3)
    (output / "suite.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")
    return cases


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("practice"))
    args = parser.parse_args()
    create_suite(args.output)
    print(f"Created nine synthetic practice datasets in {args.output.resolve()}")
