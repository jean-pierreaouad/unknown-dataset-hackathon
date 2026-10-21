"""Local checker for our provisional output contract, not the official checker."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def strict_json(path):
    def reject(value):
        raise ValueError(f"Nonfinite JSON number: {value}")

    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            reject(value)
        return number

    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject, parse_float=finite_float)


def check(output, expected_rows=None, n_clusters=None):
    output = Path(output)
    result = strict_json(output / "result.json")
    prediction = result["predictions"]
    require(isinstance(prediction, list), "predictions must be an array")
    require(isinstance(result["row_ids"], list), "row_ids must be an array")
    require(len(prediction) == len(result["row_ids"]), "Row identifier count differs")
    require(all(value is not None for value in prediction), "Predictions must not be missing")
    task = result["run"]["task"]
    require(task in {"classification", "regression", "clustering"}, "Unknown result task")
    if task == "regression":
        require(all(type(value) in (int, float) and math.isfinite(value) for value in prediction), "Regression predictions must be finite numbers")
    if task == "clustering":
        require(all(type(value) is int and value >= 0 for value in prediction), "Cluster labels must be nonnegative integers")
    if expected_rows is not None:
        require(len(prediction) == expected_rows, "Prediction count differs from expected row count")
    if n_clusters is not None:
        require(set(prediction) == set(range(n_clusters)), "Expected every requested group on the training dataset")
    with (output / "predictions.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    require(bool(rows) and rows[0] == ["row_id", "prediction"], "Invalid prediction CSV header")
    require(len(rows) - 1 == len(prediction), "CSV and JSON prediction counts differ")
    require(rows[1:] == [[str(row_id), str(value)] for row_id, value in zip(result["row_ids"], prediction, strict=True)], "CSV and JSON predictions differ")
    require(result["schema_version"] == "1.0", "Unknown result schema version")
    profile = result["profile"]
    require(profile["training"]["rows"] > 0, "Training profile must contain rows")
    require(profile["training"]["features"] == len(profile["training"]["columns"]), "Profile feature count differs")
    require(profile == strict_json(output / "profile.json"), "Profile files disagree")
    require(result["run"] == strict_json(output / "run.json"), "Run files disagree")
    if task == "classification":
        require("probabilities" in result and "classes" in result, "Classification requires classes and probabilities")
    if "probabilities" in result:
        classes = result["classes"]
        require(isinstance(classes, list) and len(classes) > 0, "Classes must be a nonempty array")
        require(len(result["probabilities"]) == len(prediction), "Probability row count differs")
        for row in result["probabilities"]:
            require(len(row) == len(classes), "Probability column count differs")
            require(all(type(number) in (int, float) and math.isfinite(number) and 0 <= number <= 1 for number in row), "Invalid probability value")
            require(abs(sum(row) - 1) < 1e-5, "Probabilities must sum to one")
        require(all(label in classes for label in prediction), "Prediction is absent from the class list")
        with (output / "probabilities.csv").open(newline="", encoding="utf-8") as stream:
            probability_rows = list(csv.reader(stream))
        require(len(probability_rows) == len(prediction) + 1, "Probability CSV row count differs")
        require(probability_rows[0] == ["row_id"] + [f"class_{index}" for index in range(len(classes))], "Probability CSV header differs")
        for csv_row, row_id, expected in zip(probability_rows[1:], result["row_ids"], result["probabilities"], strict=True):
            require(len(csv_row) == len(classes) + 1 and csv_row[0] == str(row_id), "Probability CSV row identifiers or columns differ")
            require(all(math.isfinite(float(actual)) and math.isclose(float(actual), value, rel_tol=1e-12, abs_tol=1e-15) for actual, value in zip(csv_row[1:], expected, strict=True)), "CSV and JSON probabilities differ")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--n-clusters", type=int)
    args = parser.parse_args()
    check(args.output, args.expected_rows, args.n_clusters)
    print("Outputs passed the provisional checker.")
