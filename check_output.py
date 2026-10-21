"""Local checker for our provisional output contract, not the official checker."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def check(output, expected_rows=None, n_clusters=None):
    output = Path(output)
    result = json.loads((output / "result.json").read_text(encoding="utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"Nonfinite JSON value: {value}")))
    prediction = result["predictions"]
    assert isinstance(prediction, list), "predictions must be an array"
    assert len(prediction) == len(result["row_ids"]), "Row identifier count differs"
    if expected_rows is not None:
        assert len(prediction) == expected_rows, "Prediction count differs from expected row count"
    if n_clusters is not None:
        assert set(prediction) == set(range(n_clusters)), "Expected every requested group on the training dataset"
    with (output / "predictions.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    assert rows[0] == ["row_id", "prediction"]
    assert len(rows) - 1 == len(prediction), "CSV and JSON prediction counts differ"
    assert rows[1:] == [[str(row_id), str(value)] for row_id, value in zip(result["row_ids"], prediction, strict=True)], "CSV and JSON predictions differ"
    assert result["schema_version"] == "1.0"
    profile = result["profile"]
    assert profile["training"]["rows"] > 0
    assert profile["training"]["features"] == len(profile["training"]["columns"])
    assert profile == json.loads((output / "profile.json").read_text(encoding="utf-8"))
    assert result["run"] == json.loads((output / "run.json").read_text(encoding="utf-8"))
    if "probabilities" in result:
        classes = result["classes"]
        assert len(result["probabilities"]) == len(prediction)
        for row in result["probabilities"]:
            assert len(row) == len(classes)
            assert all(0 <= number <= 1 for number in row)
            assert abs(sum(row) - 1) < 1e-5
        assert all(label in classes for label in prediction)
        with (output / "probabilities.csv").open(newline="", encoding="utf-8") as stream:
            probability_rows = list(csv.reader(stream))
        assert len(probability_rows) == len(prediction) + 1
        assert probability_rows[0] == ["row_id"] + [f"class_{index}" for index in range(len(classes))]
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--n-clusters", type=int)
    args = parser.parse_args()
    check(args.output, args.expected_rows, args.n_clusters)
    print("Outputs passed the provisional checker.")
