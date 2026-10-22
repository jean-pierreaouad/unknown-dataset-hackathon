"""The command-line entry point, using Python's standard library.

parse_args() checks the requested task and paths. execute() launches the ML
worker, enforces the time budget, and writes its last completed predictions.
For the actual learning steps, read automl/models.py and automl/worker.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parent
FIELDS = {"task", "train", "test", "target", "n_clusters", "feature_types",
          "id_column", "metric", "seed", "time_budget", "threads", "output"}


def atomic_json(path, value):
    """Replace a JSON file only after its complete contents have been written."""
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def parse_args(argv=None):
    """Read JSON configuration, apply command-line overrides, and check inputs."""
    parser = argparse.ArgumentParser(description="Automatic classification, regression, and clustering")
    parser.add_argument("--config", type=Path, help="JSON configuration; paths resolve beside that file")
    parser.add_argument("--task", choices=["classification", "regression", "clustering"])
    parser.add_argument("--train", help="Training CSV, or the dataset to cluster")
    parser.add_argument("--test", help="Unlabelled evaluation CSV")
    parser.add_argument("--target", help="Training target column for supervised tasks")
    parser.add_argument("--n-clusters", type=int)
    parser.add_argument("--feature-types", help="JSON file mapping feature names to numeric or categorical")
    parser.add_argument("--id-column", help="Exclude this column from modelling; preserve it in predictions")
    parser.add_argument("--metric", help="Validation metric (see README)")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--time-budget", type=float, help="Total processing budget, default 55 seconds, maximum 55")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--output", help="Output directory, default results")
    args = vars(parser.parse_args(argv))
    config_path = args.pop("config")
    config = {}
    base = Path.cwd()
    if config_path:
        config_path = config_path.resolve()
        base = config_path.parent
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict) or set(config) - FIELDS:
            raise ValueError("Config must be an object containing only documented fields")
        for key in ["train", "test", "output", "feature_types"]:
            if isinstance(config.get(key), str):
                config[key] = str((base / config[key]).resolve())
    config.update({key: value for key, value in args.items() if value is not None})
    defaults = {"seed": 42, "time_budget": 55.0, "threads": 1, "output": str(base / "results")}
    config = defaults | config
    if config.get("task") not in {"classification", "regression", "clustering"}:
        raise ValueError("Provide --task classification, regression, or clustering")
    for key in ("seed", "threads"):
        if type(config[key]) is not int:
            raise ValueError(f"{key} must be an integer")
    if not 0 <= config["seed"] < 2**32:
        raise ValueError("seed must be between 0 and 2**32 - 1")
    if not 1 <= config["threads"] <= 8:
        raise ValueError("threads must be between 1 and 8")
    budget = config["time_budget"]
    if isinstance(budget, bool) or not isinstance(budget, (int, float)) or not math.isfinite(budget) or not 3 <= budget <= 55:
        raise ValueError("time_budget must be between 3 and 55 seconds")
    if not config.get("train"):
        raise ValueError("Provide a training CSV")
    if config["task"] != "clustering" and (not config.get("test") or not config.get("target")):
        raise ValueError("Supervised tasks require --target and --test")
    if config["task"] == "clustering":
        if config.get("target"):
            raise ValueError("Do not supply a target for clustering")
        if type(config.get("n_clusters")) is not int or config["n_clusters"] < 1:
            raise ValueError("Clustering requires a positive integer --n-clusters")
    for key in ("train", "test"):
        if config.get(key):
            path = Path(config[key]).resolve()
            if not path.is_file():
                raise ValueError(f"Missing {key} CSV: {path}")
            config[key] = str(path)
    kinds = config.get("feature_types", {})
    if isinstance(kinds, str):
        kinds = json.loads(Path(kinds).read_text(encoding="utf-8"))
    if not isinstance(kinds, dict) or any(v not in {"numeric", "categorical"} for v in kinds.values()):
        raise ValueError("feature_types must map column names to numeric or categorical")
    config["feature_types"] = kinds
    metrics = {"classification": {"accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "roc_auc", "log_loss"},
               "regression": {"rmse", "mae", "r2", "rmsle"}, "clustering": {"silhouette"}}
    config.setdefault("metric", {"classification": "accuracy", "regression": "rmse", "clustering": "silhouette"}[config["task"]])
    if config["metric"] not in metrics[config["task"]]:
        raise ValueError(f"Unsupported metric for {config['task']}: {config['metric']}")
    return config


def execute(config):
    """Run one dataset with a deadline and keep the last complete result."""
    started = time.monotonic()
    output = Path(config["output"]).resolve()
    protected = {Path(config[key]).resolve() for key in ("train", "test") if config.get(key)}
    if any(output / name in protected for name in ["result.json", "profile.json", "run.json", "predictions.csv", "probabilities.csv"]):
        raise ValueError("Output files must not overwrite an input CSV")
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".hackathon-", dir=output) as tmp:
        tmp = Path(tmp)
        config_file = tmp / "config.json"
        checkpoint = tmp / "checkpoint.json"
        atomic_json(config_file, config)
        env = os.environ.copy()
        for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"]:
            env[name] = str(config["threads"])
        env["PYTHONHASHSEED"] = str(config["seed"])
        # Leave time for termination and writing final outputs within the requested budget.
        worker_seconds = max(0.1, config["time_budget"] - (time.monotonic() - started) - 2.0)
        with (tmp / "worker.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen([sys.executable, "-m", "automl.worker", str(config_file), str(checkpoint), str(worker_seconds)], cwd=ROOT, env=env, stdout=log, stderr=log)
            timed_out = False
            try:
                process.wait(timeout=worker_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                process.kill()
                process.wait()
        logs = (tmp / "worker.log").read_text(encoding="utf-8", errors="replace")
        if not checkpoint.exists():
            detail = logs[-6000:] or "No initial prediction completed within the time budget"
            raise RuntimeError(detail)
        result = json.loads(checkpoint.read_text(encoding="utf-8"))
        if process.returncode and not timed_out:
            result["run"].setdefault("warnings", []).append("Worker exited unexpectedly; retained last complete checkpoint. " + logs[-2000:])
        result["run"]["time_limit_reached"] = timed_out
        predictions_path = output / "predictions.csv"
        temp_predictions = output / "predictions.csv.tmp"
        with temp_predictions.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["row_id", "prediction"])
            writer.writerows(zip(result["row_ids"], result["predictions"], strict=True))
        os.replace(temp_predictions, predictions_path)
        probability_path = output / "probabilities.csv"
        if "probabilities" in result:
            temp_proba = output / "probabilities.csv.tmp"
            with temp_proba.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["row_id"] + [f"class_{i}" for i in range(len(result["classes"]))])
                writer.writerows([row_id] + row for row_id, row in zip(result["row_ids"], result["probabilities"], strict=True))
            os.replace(temp_proba, probability_path)
        elif probability_path.exists():
            probability_path.unlink()
        atomic_json(output / "profile.json", result["profile"])
        result["run"]["elapsed_seconds"] = round(time.monotonic() - started, 4)
        atomic_json(output / "run.json", result["run"])
        atomic_json(output / "result.json", result)
    return output


def main(argv=None):
    try:
        output = execute(parse_args(argv))
    except (ValueError, OSError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Completed. Outputs: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
