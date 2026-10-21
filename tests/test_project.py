from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs, make_classification, make_regression

import run
from automl.data import infer_types, load_data, profile_frame, read_csv
from automl.features import Cleaner, TreeEncoder, general_preprocessor
from automl.worker import ensure_groups, score_supervised
from check_output import check

ROOT = Path(__file__).resolve().parents[1]


class FeatureTests(unittest.TestCase):
    def test_types_respect_metadata_and_infer_numeric_strings(self):
        frame = pd.DataFrame({"number": ["1", " 2 ", None], "code": [1, 2, 1], "text": ["a", "b", "c"]})
        self.assertEqual(infer_types(frame, {"code": "categorical"}), {"number": "numeric", "code": "categorical", "text": "categorical"})
        with self.assertRaises(ValueError):
            infer_types(frame, {"absent": "numeric"})

    def test_profile_reports_exact_missing_constants_and_duplicates(self):
        frame = pd.DataFrame({"x": [1.0, np.nan, 3.0], "copy": [1.0, np.nan, 3.0], "constant": [2, 2, 2], "empty": [np.nan] * 3, "category": ["a", "b", None]})
        types = infer_types(frame, {})
        profile = profile_frame(frame, types)
        self.assertEqual(profile["missing_cells"], 6)
        self.assertEqual(profile["duplicate_columns"], {"copy": "x"})
        self.assertEqual(profile["constant_columns"], ["constant", "empty"])
        self.assertEqual(profile["columns"]["x"]["statistics"]["median"], 2)

    def test_validation_values_do_not_change_fitted_imputation(self):
        train = pd.DataFrame({"x": [0.0, 2.0, np.nan], "cat": ["a", "b", "a"]})
        prep = general_preprocessor({"x": "numeric", "cat": "categorical"}).fit(train)
        stats_before = prep.steps[-1][1].named_transformers_["numeric"].steps[0][1].statistics_.copy()
        transformed = prep.transform(pd.DataFrame({"x": [1e12, np.nan], "cat": ["unseen", None]}))
        self.assertEqual(transformed.shape[0], 2)
        np.testing.assert_array_equal(stats_before, [1.0])
        np.testing.assert_array_equal(prep.steps[-1][1].named_transformers_["numeric"].steps[0][1].statistics_, stats_before)

    def test_cleaner_removes_only_training_constants_and_duplicates(self):
        train = pd.DataFrame({"x": [0, 1, 2], "copy": [0, 1, 2], "constant": [5, 5, 5]})
        cleaner = Cleaner(infer_types(train, {})).fit(train)
        evaluation = pd.DataFrame({"x": [4], "copy": [100], "constant": [99]})
        self.assertEqual(list(cleaner.transform(evaluation).columns), ["x"])

    def test_tree_encoder_bounds_cardinality_and_handles_unknowns(self):
        frame = pd.DataFrame({"cat": [f"v{i}" for i in range(400)]})
        encoder = TreeEncoder().fit(frame)
        transformed = encoder.transform(frame)
        self.assertLessEqual(transformed["cat"].nunique(), 254)
        self.assertTrue(pd.isna(encoder.transform(pd.DataFrame({"cat": ["new"]})).iloc[0, 0]))

    def test_all_constant_features_become_bias(self):
        frame = pd.DataFrame({"constant": [1, 1, 1], "empty": [np.nan] * 3})
        output = general_preprocessor(infer_types(frame, {})).fit_transform(frame)
        self.assertEqual(output.shape, (3, 1))

    def test_nonfinite_and_invalid_numeric_values_are_reported(self):
        frame = pd.DataFrame({"x": ["1", "oops", np.inf, None]})
        p = profile_frame(frame, {"x": "numeric"})["columns"]["x"]
        self.assertEqual((p["invalid_numeric_count"], p["infinite_count"], p["missing_count"]), (1, 1, 1))

    def test_repair_collapsed_groups(self):
        labels = ensure_groups([0, 0, 0, 0, 0], 4)
        self.assertEqual(set(labels), {0, 1, 2, 3})

    def test_error_metrics_are_maximized_internally(self):
        self.assertGreater(score_supervised("rmse", [1, 2], [1, 2]), score_supervised("rmse", [1, 2], [4, 5]))
        self.assertGreater(score_supervised("log_loss", [0, 1], [0, 1], np.array([[0.9, 0.1], [0.1, 0.9]]), [0, 1]),
                           score_supervised("log_loss", [0, 1], [1, 0], np.array([[0.1, 0.9], [0.9, 0.1]]), [0, 1]))


class InputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix=".test-", dir=ROOT)
        self.folder = Path(self.temp.name)
        pd.DataFrame({"x": [1, 2, 3], "target": [0, 1, 0]}).to_csv(self.folder / "train.csv", index=False)
        pd.DataFrame({"x": [4, 5]}).to_csv(self.folder / "test.csv", index=False)
        self.config = {"task": "classification", "train": str(self.folder / "train.csv"), "test": str(self.folder / "test.csv"), "target": "target", "feature_types": {}, "metric": "accuracy"}

    def tearDown(self):
        self.temp.cleanup()

    def test_reject_evaluation_target(self):
        pd.DataFrame({"x": [4], "target": [1]}).to_csv(self.folder / "test.csv", index=False)
        with self.assertRaisesRegex(ValueError, "must not contain the target"):
            load_data(self.config)

    def test_reject_missing_target(self):
        pd.DataFrame({"x": [1, 2], "target": [1, None]}).to_csv(self.folder / "train.csv", index=False)
        with self.assertRaisesRegex(ValueError, "missing values"):
            load_data(self.config)

    def test_reject_schema_mismatch(self):
        pd.DataFrame({"wrong": [1, 2]}).to_csv(self.folder / "test.csv", index=False)
        with self.assertRaisesRegex(ValueError, "features differ"):
            load_data(self.config)

    def test_allow_reordered_evaluation_columns(self):
        pd.DataFrame({"x": [1, 2, 3], "z": [3, 2, 1], "target": [0, 1, 0]}).to_csv(self.folder / "train.csv", index=False)
        pd.DataFrame({"z": [1, 2], "x": [4, 5]}).to_csv(self.folder / "test.csv", index=False)
        _, test, *_ = load_data(self.config)
        self.assertEqual(list(test.columns), ["x", "z"])

    def test_reject_duplicate_headers(self):
        (self.folder / "bad.csv").write_text("x,x\n1,2\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            read_csv(self.folder / "bad.csv")

    def test_reject_empty_csv(self):
        (self.folder / "bad.csv").write_text("x,y\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "at least one"):
            read_csv(self.folder / "bad.csv")

    def test_config_paths_and_cli_override(self):
        config = {"task": "classification", "train": "train.csv", "test": "test.csv", "target": "target", "time_budget": 10}
        path = self.folder / "config.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        actual = run.parse_args(["--config", str(path), "--time-budget", "12"])
        self.assertEqual(actual["time_budget"], 12)
        self.assertEqual(Path(actual["train"]), self.folder / "train.csv")

    def test_reject_budget_over_contest_limit(self):
        with self.assertRaisesRegex(ValueError, "between 3 and 55"):
            run.parse_args(["--task", "classification", "--time-budget", "60"])

    def test_reject_unknown_config_keys(self):
        path = self.folder / "config.json"
        path.write_text('{"unexpected": true}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "documented fields"):
            run.parse_args(["--config", str(path)])

    def test_explicit_identifier_is_excluded_and_preserved(self):
        pd.DataFrame({"id": ["a", "b", "c"], "x": [1, 2, 3], "target": [0, 1, 0]}).to_csv(self.folder / "train.csv", index=False)
        pd.DataFrame({"id": ["d", "e"], "x": [4, 5]}).to_csv(self.folder / "test.csv", index=False)
        self.config["id_column"] = "id"
        train, _, _, _, _, ids = load_data(self.config)
        self.assertEqual(list(train.columns), ["x"])
        self.assertEqual(ids, ["d", "e"])

    def test_reject_duplicate_output_identifiers(self):
        pd.DataFrame({"id": [1, 2, 3], "x": [1, 2, 3], "target": [0, 1, 0]}).to_csv(self.folder / "train.csv", index=False)
        pd.DataFrame({"id": [4, 4], "x": [4, 5]}).to_csv(self.folder / "test.csv", index=False)
        self.config["id_column"] = "id"
        with self.assertRaisesRegex(ValueError, "unique"):
            load_data(self.config)

    def test_reject_negative_rmsle_target(self):
        self.config.update(task="regression", metric="rmsle")
        pd.DataFrame({"x": [1, 2, 3], "target": [-1, 1, 2]}).to_csv(self.folder / "train.csv", index=False)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            load_data(self.config)

    def test_reject_more_clusters_than_rows(self):
        self.config.update(task="clustering", n_clusters=10)
        self.config.pop("target")
        self.config.pop("test")
        with self.assertRaisesRegex(ValueError, "number of training rows"):
            load_data(self.config)


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix=".test-", dir=ROOT)
        self.folder = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def execute(self, task, train, test=None, **kwargs):
        train.to_csv(self.folder / "train.csv", index=False)
        config = {"task": task, "train": "train.csv", "output": "results", "time_budget": 30} | kwargs
        if test is not None:
            test.to_csv(self.folder / "test.csv", index=False)
            config["test"] = "test.csv"
        config_path = self.folder / "config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        result = subprocess.run([sys.executable, str(ROOT / "run.py"), "--config", str(config_path)], capture_output=True, text=True, timeout=40)
        self.assertEqual(result.returncode, 0, result.stderr)
        return check(self.folder / "results", len(test) if test is not None else len(train), kwargs.get("n_clusters") if test is None else None)

    def test_classification_with_mixed_missing_and_unseen_values(self):
        X, y = make_classification(n_samples=250, n_features=5, n_informative=3, random_state=42)
        frame = pd.DataFrame(X, columns=list("abcde"))
        frame["category"] = np.where(X[:, 0] > 0, "up", "down")
        frame.loc[:20, "a"] = np.nan
        frame["target"] = np.where(y == 1, "yes", "no")
        test = frame.iloc[-30:].drop(columns="target").copy()
        test["category"] = "unseen"
        result = self.execute("classification", frame.iloc[:-30], test, target="target")
        self.assertEqual(set(result["classes"]), {"yes", "no"})
        self.assertTrue(set(result["predictions"]) <= {"yes", "no"})
        self.assertGreater(len(result["run"]["candidates"]), 0)

    def test_regression_learns_heldout_linear_signal(self):
        X, y = make_regression(n_samples=300, n_features=5, noise=1, random_state=42)
        frame = pd.DataFrame(X)
        frame["target"] = y
        result = self.execute("regression", frame.iloc[:240], frame.iloc[240:].drop(columns="target"), target="target")
        rmse = np.sqrt(np.mean((np.array(result["predictions"]) - y[240:]) ** 2))
        self.assertLess(rmse, 10)

    def test_clustering_finds_requested_groups(self):
        X, y = make_blobs(n_samples=200, centers=3, cluster_std=0.3, random_state=42)
        result = self.execute("clustering", pd.DataFrame(X), n_clusters=3)
        from sklearn.metrics import adjusted_rand_score
        self.assertGreater(adjusted_rand_score(y, result["predictions"]), 0.95)

    def test_single_class_and_all_constant_features(self):
        train = pd.DataFrame({"x": [1] * 25, "target": ["only"] * 25})
        result = self.execute("classification", train, pd.DataFrame({"x": [1, 1, 1]}), target="target")
        self.assertEqual(result["predictions"], ["only"] * 3)
        self.assertEqual(result["probabilities"], [[1.0]] * 3)

    def test_clustering_duplicate_points_still_returns_required_groups(self):
        result = self.execute("clustering", pd.DataFrame({"constant": [1] * 20}), n_clusters=4)
        self.assertEqual(set(result["predictions"]), {0, 1, 2, 3})

    def test_rare_class_does_not_break_split(self):
        train = pd.DataFrame({"x": range(30), "target": ["majority"] * 29 + ["rare"]})
        result = self.execute("classification", train, pd.DataFrame({"x": [1, 29]}), target="target")
        self.assertIsNone(result["run"]["validation_score"])
        self.assertTrue(result["run"]["warnings"])

    def test_multiclass_auc_and_probability_alignment(self):
        X, y = make_classification(n_samples=300, n_features=7, n_classes=3, n_informative=5, random_state=42)
        frame = pd.DataFrame(X)
        frame["target"] = np.array(["alpha", "beta", "gamma"])[y]
        result = self.execute("classification", frame.iloc[:250], frame.iloc[250:].drop(columns="target"), target="target", metric="roc_auc")
        self.assertEqual(result["classes"], ["alpha", "beta", "gamma"])
        self.assertTrue(0 <= result["run"]["validation_score"] <= 1)

    def test_high_cardinality_categories_fit_histogram_successfully(self):
        frame = pd.DataFrame({"category": [f"label{i}" for i in range(350)], "number": np.arange(350), "target": np.arange(350) % 2})
        test = pd.DataFrame({"category": ["unseen", "label1"], "number": [20, 30]})
        result = self.execute("classification", frame, test, target="target", feature_types={"category": "categorical"})
        hist = [item for item in result["run"]["candidates"] if item["model"] == "histogram_boosting"]
        self.assertTrue(hist)
        self.assertNotIn("error", hist[0])

    def test_clustering_evaluation_assignments(self):
        X, _ = make_blobs(n_samples=160, centers=3, random_state=42)
        result = self.execute("clustering", pd.DataFrame(X[:140]), pd.DataFrame(X[140:]), n_clusters=3)
        self.assertTrue(set(result["predictions"]) <= {0, 1, 2})

    def test_one_row_regression_has_finite_fallback(self):
        result = self.execute("regression", pd.DataFrame({"x": [1], "target": [4.5]}), pd.DataFrame({"x": [2, 3]}), target="target")
        self.assertEqual(result["predictions"], [4.5, 4.5])

    def test_extreme_minority_auc_keeps_each_class_in_both_partitions(self):
        train = pd.DataFrame({"x": np.arange(1000), "target": ["majority"] * 998 + ["rare"] * 2})
        result = self.execute("classification", train, pd.DataFrame({"x": [2, 998]}), target="target", metric="roc_auc")
        self.assertTrue(0 <= result["run"]["validation_score"] <= 1)
        self.assertEqual(result["classes"], ["majority", "rare"])

    def test_watchdog_retains_complete_checkpoint(self):
        config = {"task": "regression", "train": str(self.folder / "unused.csv"), "output": str(self.folder / "results"), "seed": 42, "threads": 1, "time_budget": 3}
        checkpoint = {"schema_version": "1.0", "predictions": [7.0], "row_ids": [0], "profile": {"training": {"rows": 1, "features": 1, "columns": {"x": {}}}}, "run": {"task": "regression"}}
        actual_popen = subprocess.Popen

        def delayed_worker(command, **kwargs):
            script = "import json,pathlib,sys,time; pathlib.Path(sys.argv[1]).write_text(sys.argv[2],encoding='utf-8'); time.sleep(30)"
            return actual_popen([sys.executable, "-c", script, command[4], json.dumps(checkpoint)], **kwargs)

        started = time.monotonic()
        with patch.object(run.subprocess, "Popen", side_effect=delayed_worker):
            output = run.execute(config)
        self.assertLess(time.monotonic() - started, 5)
        result = check(output, 1)
        self.assertTrue(result["run"]["time_limit_reached"])
        self.assertEqual(result["predictions"], [7.0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
