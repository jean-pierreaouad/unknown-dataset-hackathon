"""The small set of supervised models and their validation scores.

Start here to understand the model choices. The random forest is the one
ensemble extension: several decision trees vote (classification) or average
their predictions (regression). It does not need a separate feature encoder.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, log_loss,
    mean_absolute_error, mean_squared_error, r2_score, roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .features import general_preprocessor


def candidates(config, types):
    """Try a linear relationship, a decision tree, then a small forest."""
    seed = config["seed"]
    if config["task"] == "classification":
        # Give rare classes greater weight when the requested metric values
        # performance across classes rather than overall accuracy alone.
        balanced_metrics = {"balanced_accuracy", "f1_macro", "roc_auc"}
        weight = "balanced" if config["metric"] in balanced_metrics else None
        # "sqrt" checks a small random subset of features at each forest split.
        # This adds diversity and avoids checking almost every feature per split.
        models = [
            ("linear", LogisticRegression(max_iter=250, class_weight=weight,
                                           random_state=seed)),
            ("decision_tree", DecisionTreeClassifier(
                max_depth=8, min_samples_leaf=2, class_weight=weight,
                random_state=seed)),
            ("random_forest", RandomForestClassifier(
                n_estimators=60, max_depth=12, min_samples_leaf=2,
                max_features="sqrt", class_weight=weight, n_jobs=1,
                random_state=seed)),
        ]
    else:
        models = [
            ("linear", Ridge(alpha=1.0, solver="lsqr")),
            ("decision_tree", DecisionTreeRegressor(
                max_depth=8, min_samples_leaf=2, random_state=seed)),
            ("random_forest", RandomForestRegressor(
                n_estimators=60, max_depth=12, min_samples_leaf=2,
                max_features=1.0, n_jobs=1, random_state=seed)),
        ]
    # Each candidate gets its own preprocessor, fitted only on its training rows.
    for name, model in models:
        yield name, make_pipeline(general_preprocessor(types), model)


def score_supervised(metric, y, prediction, probability=None, classes=None):
    """Make larger scores better; negate metrics that measure prediction error."""
    if metric == "accuracy":
        return float(accuracy_score(y, prediction))
    if metric == "balanced_accuracy":
        return float(balanced_accuracy_score(y, prediction))
    if metric in {"f1_macro", "f1_weighted"}:
        average = "macro" if metric == "f1_macro" else "weighted"
        return float(f1_score(y, prediction, average=average, zero_division=0))
    if metric == "log_loss":
        return -float(log_loss(y, probability, labels=classes))
    if metric == "roc_auc":
        if len(classes) == 2:
            return float(roc_auc_score(y, probability[:, 1]))
        return float(roc_auc_score(y, probability, multi_class="ovr",
                                   average="weighted", labels=classes))
    if metric == "rmse":
        return -float(np.sqrt(mean_squared_error(y, prediction)))
    if metric == "mae":
        return -float(mean_absolute_error(y, prediction))
    if metric == "rmsle":
        actual_log = np.log1p(y)
        predicted_log = np.log1p(np.maximum(0, prediction))
        return -float(np.sqrt(mean_squared_error(actual_log, predicted_log)))
    if metric == "r2":
        return float(r2_score(y, prediction))
    raise ValueError(f"Unsupported supervised metric: {metric}")
