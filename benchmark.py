"""Run nine synthetic cases and compare predictions with held-out local truth."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, adjusted_rand_score, balanced_accuracy_score, mean_squared_error, r2_score
from check_output import check
from make_practice import create_suite


def benchmark(output, budget=15):
    output = Path(output).resolve()
    cases = create_suite(output)
    report = {"scope": "Synthetic local development tests; not official practice data or contest scores", "time_budget_per_case": budget, "cases": []}
    for case in cases:
        config_path = output / case["config"]
        config = json.loads(config_path.read_text(encoding="utf-8"))
        started = time.monotonic()
        process = subprocess.run([sys.executable, str(Path(__file__).parent / "run.py"), "--config", str(config_path), "--time-budget", str(budget)], capture_output=True, text=True, timeout=budget + 10)
        elapsed = time.monotonic() - started
        if process.returncode:
            raise RuntimeError(f"{case['name']}: {process.stderr}")
        truth = pd.read_csv(config_path.parent / "truth.csv")["target"].to_numpy()
        result = check(config_path.parent / "results", len(truth), config.get("n_clusters"))
        predicted = np.array(result["predictions"])
        metrics = {}
        if case["task"] == "classification":
            metrics = {"accuracy": accuracy_score(truth, predicted), "balanced_accuracy": balanced_accuracy_score(truth, predicted)}
        elif case["task"] == "regression":
            metrics = {"rmse": float(np.sqrt(mean_squared_error(truth, predicted))), "r2": r2_score(truth, predicted)}
            if config["metric"] == "rmsle":
                metrics["rmsle"] = float(np.sqrt(mean_squared_error(np.log1p(truth), np.log1p(np.maximum(0, predicted)))))
        else:
            metrics = {"adjusted_rand_index": adjusted_rand_score(truth, predicted)}
        record = {"name": case["name"], "task": case["task"], "elapsed_seconds": round(elapsed, 3),
                  "selected_model": result["run"]["selected_model"], "metrics": metrics,
                  "time_limit_reached": result["run"]["time_limit_reached"],
                  "candidates": result["run"]["candidates"]}
        report["cases"].append(record)
        print(json.dumps({key: value for key, value in record.items() if key != "candidates"}), flush=True)
    report["all_under_60_seconds"] = all(case["elapsed_seconds"] < 60 for case in report["cases"])
    (output / "benchmark_report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("practice"))
    parser.add_argument("--time-budget", type=float, default=15)
    args = parser.parse_args()
    benchmark(args.output, args.time_budget)
