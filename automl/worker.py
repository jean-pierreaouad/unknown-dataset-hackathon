"""Budgeted model selection. Only this process imports scientific libraries."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
import time
import warnings

PROCESS_STARTED = time.monotonic()

import numpy as np
from scipy import sparse
from sklearn.base import clone
from sklearn.cluster import KMeans, MiniBatchKMeans, SpectralClustering
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, log_loss,
                             mean_absolute_error, mean_squared_error, r2_score, roc_auc_score, silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from threadpoolctl import threadpool_limits

from .data import load_data
from .features import Cleaner, TreeEncoder, general_preprocessor


def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Refusing to emit nonfinite JSON numbers")
    return value


class Budget:
    def __init__(self, seconds):
        self.deadline = time.monotonic() + seconds

    def left(self):
        return max(0.0, self.deadline - time.monotonic())


class Recorder:
    def __init__(self, path, config, profile, row_ids):
        self.path = Path(path)
        self.config = config
        self.profile = profile
        self.row_ids = row_ids
        self.run = {"schema_version": "1.0", "task": config["task"], "seed": config["seed"],
                    "metric": config["metric"], "budget_seconds": config["time_budget"],
                    "threads": config["threads"], "candidates": [], "warnings": [],
                    "interface_status": "provisional_pending_official_kickoff_contract"}

    def save(self, predictions, name, score=None, probabilities=None, classes=None, refit=False):
        self.run.update({"selected_model": name, "validation_score": score, "refit_on_all_training_rows": refit})
        result = {"schema_version": "1.0", "predictions": predictions, "row_ids": self.row_ids,
                  "profile": self.profile, "run": self.run}
        if probabilities is not None:
            result.update({"probabilities": probabilities, "classes": classes})
        result = jsonable(result)
        temp = self.path.with_suffix(".tmp")
        temp.write_text(json.dumps(result, allow_nan=False), encoding="utf-8")
        os.replace(temp, self.path)


def score_supervised(metric, y, prediction, probability=None, classes=None):
    if metric == "accuracy":
        return float(accuracy_score(y, prediction))
    if metric == "balanced_accuracy":
        return float(balanced_accuracy_score(y, prediction))
    if metric in {"f1_macro", "f1_weighted"}:
        return float(f1_score(y, prediction, average=metric.removeprefix("f1_"), zero_division=0))
    if metric == "log_loss":
        return -float(log_loss(y, probability, labels=classes))
    if metric == "roc_auc":
        if len(classes) == 2:
            return float(roc_auc_score(y, probability[:, 1]))
        return float(roc_auc_score(y, probability, multi_class="ovr", average="weighted", labels=classes))
    if metric == "rmse":
        return -float(np.sqrt(mean_squared_error(y, prediction)))
    if metric == "mae":
        return -float(mean_absolute_error(y, prediction))
    if metric == "rmsle":
        return -float(np.sqrt(mean_squared_error(np.log1p(y), np.log1p(np.maximum(0, prediction)))))
    return float(r2_score(y, prediction))


def candidates(config, types, X, y):
    task = config["task"]
    seed = config["seed"]
    classify = task == "classification"
    balanced = classify and config["metric"] in {"balanced_accuracy", "f1_macro", "roc_auc"}
    weight = "balanced" if balanced else None
    leaf = max(2, min(20, len(X) // 40))
    hist_class = HistGradientBoostingClassifier if classify else HistGradientBoostingRegressor
    params = dict(max_iter=140, learning_rate=0.08, max_leaf_nodes=23,
                  min_samples_leaf=leaf, l2_regularization=1.0, categorical_features="from_dtype",
                  early_stopping=False, random_state=seed)
    if classify:
        params["class_weight"] = weight
    histogram = make_pipeline(Cleaner(types), TreeEncoder(), hist_class(**params))
    linear = LogisticRegression(C=1.0, max_iter=250, solver="lbfgs", class_weight=weight, random_state=seed) if classify else Ridge(alpha=1.0, solver="lsqr")
    yield "linear", make_pipeline(general_preprocessor(types), linear)
    yield "histogram_boosting", histogram
    trees = (ExtraTreesClassifier(n_estimators=96, min_samples_leaf=1, max_features=0.8, class_weight=weight, random_state=seed, n_jobs=1)
             if classify else ExtraTreesRegressor(n_estimators=96, min_samples_leaf=1, max_features=0.9, random_state=seed, n_jobs=1))
    yield "extra_trees", make_pipeline(general_preprocessor(types), trees)
    if not classify and np.min(y) >= 0 and float(np.std(y)) > 0 and float(np.mean(y)) > 0:
        yield "log_target_histogram", TransformedTargetRegressor(regressor=clone(histogram), func=np.log1p, inverse_func=np.expm1)
    if classify and weight is None and np.bincount(y).max() / max(1, np.bincount(y).min()) >= 3:
        alternative = clone(histogram)
        alternative.steps[-1][1].set_params(class_weight="balanced")
        yield "balanced_histogram", alternative


def supervised(config, X, evaluation, target, types, recorder, budget):
    classification = config["task"] == "classification"
    label_encoder = LabelEncoder().fit(target) if classification else None
    y = label_encoder.transform(target) if classification else target.to_numpy()
    classes = label_encoder.classes_ if classification else None
    indices = np.arange(len(X))
    can_validate = len(X) >= 15 and (not classification or (len(classes) >= 2 and np.bincount(y).min() >= 2))
    if can_validate:
        test_size = max(int(math.ceil(len(X) * 0.2)), len(classes) if classification else 2)
        can_validate = test_size < len(X) and (not classification or len(X) - test_size >= len(classes))
    if can_validate:
        fit_idx, validation_idx = train_test_split(indices, test_size=test_size, random_state=config["seed"], stratify=y if classification else None)
        if classification:
            # Proportional stratification can round a two-row minority entirely
            # into one side. Ensure probability metrics see every class twice:
            # at least once in training and at least once in validation.
            for label in range(len(classes)):
                train_members = fit_idx[y[fit_idx] == label]
                valid_members = validation_idx[y[validation_idx] == label]
                if len(valid_members) == 0:
                    moved = train_members[-1]
                    fit_idx = fit_idx[fit_idx != moved]
                    validation_idx = np.append(validation_idx, moved)
                elif len(train_members) == 0:
                    moved = valid_members[-1]
                    validation_idx = validation_idx[validation_idx != moved]
                    fit_idx = np.append(fit_idx, moved)
    else:
        fit_idx, validation_idx = indices, np.array([], dtype=int)
        recorder.run["warnings"].append("Too few rows or rare classes for a reliable holdout; using a fixed model without validation selection.")
    X_fit, y_fit = X.iloc[fit_idx], y[fit_idx]
    X_valid, y_valid = X.iloc[validation_idx], y[validation_idx]
    dummy = DummyClassifier(strategy="prior") if classification else DummyRegressor(strategy="median" if config["metric"] == "mae" else "mean")
    dummy.fit(np.zeros((len(y_fit), 1)), y_fit)

    def predict(model, frame, is_dummy=False):
        values = np.zeros((len(frame), 1)) if is_dummy else frame
        predictions = model.predict(values)
        if not classification and config["metric"] == "rmsle":
            predictions = np.maximum(0, predictions)
        if not np.isfinite(predictions).all():
            raise ValueError("Model produced nonfinite predictions")
        probabilities = model.predict_proba(values) if classification else None
        return predictions, probabilities

    def save_model(model, name, score, is_dummy=False, refit=False):
        predictions, probabilities = predict(model, evaluation, is_dummy)
        labels = label_encoder.inverse_transform(predictions.astype(int)) if classification else predictions
        recorder.save(labels, name, score, probabilities, classes, refit)

    best_score = -float("inf")
    if can_validate:
        pred, proba = predict(dummy, X_valid, True)
        best_score = score_supervised(config["metric"], y_valid, pred, proba, dummy.classes_ if classification else None)
    best = dummy
    best_name = "baseline_prior" if classification else "baseline_constant"
    save_model(best, best_name, best_score if can_validate else None, is_dummy=True)
    if classification and len(classes) == 1:
        recorder.run["warnings"].append("Training labels contain only one class; returning that class.")
        save_model(best, best_name, None, is_dummy=True, refit=True)
        return
    for name, model in candidates(config, types, X_fit, y_fit):
        if budget.left() < 3:
            break
        if not can_validate and name != "histogram_boosting":
            continue
        started = time.monotonic()
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model.fit(X_fit, y_fit)
            record = {"model": name, "fit_seconds": round(time.monotonic() - started, 4)}
            record["warnings"] = list(dict.fromkeys(str(w.message) for w in caught))[:5]
            value = None
            if can_validate:
                pred, proba = predict(model, X_valid)
                value = score_supervised(config["metric"], y_valid, pred, proba, np.arange(len(classes)) if classification else None)
                if not math.isfinite(value):
                    raise ValueError("Nonfinite validation score")
            record["score"] = value
            recorder.run["candidates"].append(record)
            if not can_validate or value > best_score + 1e-9:
                # Commit predictions before replacing the current best model.
                save_model(model, name, value)
                best, best_name = model, name
                best_score = value if can_validate else -float("inf")
            if not can_validate:
                break
        except Exception as exc:
            recorder.run["candidates"].append({"model": name, "error": str(exc)[:1000]})
    if budget.left() > 2:
        try:
            full_model = clone(best)
            full_model.fit(np.zeros((len(y), 1)) if isinstance(best, (DummyClassifier, DummyRegressor)) else X, y)
            save_model(full_model, best_name, best_score if can_validate else None,
                       is_dummy=isinstance(best, (DummyClassifier, DummyRegressor)), refit=True)
        except Exception as exc:
            recorder.run["warnings"].append("Full-data refit failed; retained validated model: " + str(exc)[:500])


def ensure_groups(labels, k):
    """Deterministically split duplicate points when an estimator collapses groups."""
    labels = np.asarray(labels, dtype=int).copy()
    unique, labels = np.unique(labels, return_inverse=True)
    for new in range(len(unique), k):
        counts = np.bincount(labels)
        largest = int(counts.argmax())
        location = int(np.flatnonzero(labels == largest)[-1])
        labels[location] = new
    return labels


def clustering(config, X, evaluation, types, recorder, budget):
    k, seed = config["n_clusters"], config["seed"]
    target_frame = evaluation if evaluation is not None else X
    initial = np.arange(len(target_frame)) % k
    recorder.save(initial, "deterministic_fallback")
    if k == 1:
        recorder.save(np.zeros(len(target_frame), dtype=int), "single_cluster", refit=True)
        return
    prep = general_preprocessor(types)
    train_matrix = prep.fit_transform(X)
    eval_matrix = prep.transform(evaluation) if evaluation is not None else train_matrix
    matrix_bytes = train_matrix.shape[0] * train_matrix.shape[1] * 8
    n = len(X)
    plans = [("kmeans", KMeans(n_clusters=k, n_init=5, max_iter=200, random_state=seed))]
    if n > 5000:
        plans.insert(0, ("minibatch_kmeans", MiniBatchKMeans(n_clusters=k, n_init=3, batch_size=1024, random_state=seed)))
    if n >= max(k * 3, 10) and matrix_bytes < 64_000_000 and train_matrix.shape[1] <= 100:
        plans.append(("gaussian_mixture_diagonal", GaussianMixture(n_components=k, covariance_type="diag", n_init=2, max_iter=100, reg_covar=1e-5, random_state=seed)))
        if train_matrix.shape[1] <= 20 and n <= 10000:
            plans.append(("gaussian_mixture_full", GaussianMixture(n_components=k, covariance_type="full", n_init=1, max_iter=100, reg_covar=1e-5, random_state=seed)))
    # Spectral embeddings provide a nonlinear candidate only on modest datasets.
    if evaluation is None and 30 <= n <= 1200 and k < n - 1 and matrix_bytes < 32_000_000:
        plans.append(("spectral_neighbors", SpectralClustering(n_clusters=k, affinity="nearest_neighbors", n_neighbors=min(12, n - 1), assign_labels="kmeans", n_init=5, random_state=seed, n_jobs=1)))
    best_score = -float("inf")
    sample_idx = np.random.default_rng(seed).choice(n, min(n, 1500), replace=False)
    repaired_any = False
    for name, model in plans:
        if budget.left() < 3:
            break
        started = time.monotonic()
        try:
            dense = isinstance(model, GaussianMixture)
            values = train_matrix.toarray() if dense and sparse.issparse(train_matrix) else train_matrix
            labels = model.fit_predict(values)
            repaired = len(np.unique(labels)) < k
            if repaired:
                labels = ensure_groups(labels, k)
            sample_labels = labels[sample_idx]
            groups = len(np.unique(sample_labels))
            value = float(silhouette_score(train_matrix[sample_idx], sample_labels)) if 1 < groups < len(sample_idx) else -1.0
            recorder.run["candidates"].append({"model": name, "score": value, "fit_seconds": round(time.monotonic() - started, 4), "groups_repaired": repaired})
            if value > best_score:
                if evaluation is not None:
                    eval_values = eval_matrix.toarray() if dense and sparse.issparse(eval_matrix) else eval_matrix
                    predictions = model.predict(eval_values)
                else:
                    predictions = labels
                recorder.save(predictions, name, value, refit=True)
                best_score, repaired_any = value, repaired
        except Exception as exc:
            recorder.run["candidates"].append({"model": name, "error": str(exc)[:1000]})
    if repaired_any:
        recorder.run["warnings"].append("Identical feature vectors collapsed groups; split duplicate points deterministically to meet the requested group count. These groups have no geometric separation.")
    # Persist final diagnostics even when later candidates did not beat the winner.
    if recorder.path.exists():
        previous = json.loads(recorder.path.read_text(encoding="utf-8"))
        recorder.save(previous["predictions"], previous["run"]["selected_model"], previous["run"]["validation_score"], refit=True)


def main():
    config_path, checkpoint, seconds = sys.argv[1:]
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    budget = Budget(float(seconds) - (time.monotonic() - PROCESS_STARTED) - 1.0)
    with threadpool_limits(limits=config["threads"]):
        X, evaluation, y, types, profile, row_ids = load_data(config)
        recorder = Recorder(checkpoint, config, profile, row_ids)
        if config["task"] == "clustering":
            clustering(config, X, evaluation, types, recorder, budget)
        else:
            supervised(config, X, evaluation, y, types, recorder, budget)
        # Keep diagnostics from every attempted candidate, even if it did not win.
        if recorder.path.exists():
            previous = json.loads(recorder.path.read_text(encoding="utf-8"))
            recorder.save(previous["predictions"], previous["run"]["selected_model"],
                          previous["run"]["validation_score"], previous.get("probabilities"),
                          previous.get("classes"), previous["run"]["refit_on_all_training_rows"])


if __name__ == "__main__":
    main()
