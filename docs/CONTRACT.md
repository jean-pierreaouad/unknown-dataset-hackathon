# Provisional execution contract

This is the project's own interface, awaiting the kickoff specification.

## Configuration

| Field | Type | Meaning |
|---|---|---|
| `task` | string | Required: classification, regression, or clustering |
| `train` | path string | Required CSV; supervised rows include the target |
| `test` | path string | Required for supervised tasks; optional clustering evaluation features |
| `target` | string | Required supervised target column; forbidden for clustering |
| `n_clusters` | integer | Required clustering group count, at least 1 and at most the number of training rows |
| `feature_types` | object or path string | Optional feature-to-type mapping; values numeric or categorical; partial declarations allowed |
| `id_column` | string | Optional column present in every supplied CSV; excluded from features, retained for output row identifiers |
| `metric` | string | Optional supported validation metric from README |
| `seed` | integer | Default 42; between 0 and 2**32 - 1 |
| `time_budget` | number | Processing budget between 3 and 55 seconds, default 55 |
| `threads` | integer | CPU thread limit between 1 and 8, default 1 |
| `output` | path string | Output directory, default results |

Unknown config keys are rejected. Command-line flags use hyphens, such as `--n-clusters`; JSON fields use underscores. Input paths in JSON resolve relative to the config file. Explicit flags resolve relative to the process working directory.

Headers must be unique and nonempty. Empty tables, missing targets, nonfinite regression targets, mismatched feature sets, and evaluation target columns fail with exit code 1. Explicit output identifiers must be nonmissing and unique. Without an identifier column, row IDs are zero-based positions. Supervised labels are not inferred or synthesized.

## Result

`result.json` is a JSON object with these fields:

```json
{
  "schema_version": "1.0",
  "predictions": ["no", "yes"],
  "row_ids": [0, 1],
  "classes": ["no", "yes"],
  "probabilities": [[0.8, 0.2], [0.1, 0.9]],
  "profile": {},
  "run": {}
}
```

`classes` and `probabilities` appear only for classification. Regression predictions are finite numbers. Clustering labels are integers; clustering the training table produces exactly the required number of nonempty groups. A separate clustering evaluation table need not contain every group. Predictions always follow the output table's original row order.

The profile contains schema version, task, feature types, a training profile, an evaluation profile when applicable, and a target summary for supervised tasks. Training and evaluation feature profiles each contain row and feature counts, per-column type, missing counts and fractions, nonmissing uniqueness, constant and all-missing flags, raw exact duplicate columns, duplicate-row count, and missing-cell total. Numeric fields include finite statistics, infinite counts, invalid numeric counts, and extreme-value clipping counts. Categorical fields include the ten most frequent nonmissing values and counts. Labels from the evaluation set never enter the program.

Diagnostics contain the selected model, seed, task, metric, attempted candidates and errors, score, full-training-refit indicator, warnings, configured budget and threads, elapsed processing time, and timeout flag. Loss scores are negated for internal maximization. The reported elapsed processing time is measured by the parent near output completion; end-to-end process time is measured independently in the benchmark report.

The worker replaces its checkpoint only after a complete prediction is ready. A timeout returns the most recent completed result. No checkpoint means an explicit failure. Individual files are atomically replaced; the directory is not an atomic multi-file transaction. Read outputs only after a successful exit. Previous output files can remain after a failed invocation, so callers must check the exit code.
