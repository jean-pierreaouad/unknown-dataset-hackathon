# Local validation results

Tests were run on 21 October 2026 with an isolated Python 3.12.14 environment on Windows AMD64, an Intel Family 6 Model 126 processor, and 8 logical CPUs. The prediction worker was limited to one CPU thread. The pinned dependency set passed its dependency consistency check. Linux and the organizer's assessment environment have not yet been tested.

The release ZIP was extracted to a separate directory. All three shipped examples ran successfully from that copy and passed the output checker. Every installed direct and transitive dependency matched its exact pinned version.

## Automated checks

All **34 tests passed** in the final full-suite run, which took 68.880 seconds for the entire suite. They cover numerical and categorical preprocessing, exact profiling, declared types, high cardinality, unknown categories, preprocessing isolation, malformed schemas, missing targets, duplicate identifiers, RMSLE constraints, clustering count constraints, mixed-data classification, regression quality, cluster recovery, single-class targets, collapsed groups, rare classes, multiclass probability alignment, one-row regression, optional clustering evaluation, and forced worker termination with checkpoint recovery.

One additional imbalance case checks that a class with just two examples remains present in both fitting and validation partitions, allowing ROC AUC to be evaluated. A single proportional stratified split can otherwise round the entire minority into one side.

Run the suite from the project directory:

```sh
python -m unittest discover -s tests -v
```

## Nine synthetic development datasets

These are generated data, not the official practice or private assessment datasets. The benchmark uses separately held-out truth for supervised metrics and reference labels only for evaluating clustering after predictions. The production entry point receives no truth files.

Every case used a 15-second processing budget. Timings below are independent end-to-end subprocess measurements, including interpreter startup and output writing. Several cases ran while other local verification work was active, so these timings are descriptive rather than a controlled performance comparison.

| Case | Rows supplied for learning | Features | End-to-end seconds | Held-out or external quality |
|---|---:|---:|---:|---|
| Classification with imbalance | 675 | 12 | 8.515 | Accuracy 0.9422; balanced accuracy 0.8748 |
| Nonlinear classification | 675 | 3 | 6.406 | Accuracy 0.9778 |
| Classification with mixed types and missingness | 750 | 6 | 7.391 | Accuracy 0.8840 |
| Linear regression with unequal scales | 750 | 10 | 7.093 | R² 0.9956; RMSE 10.8059 |
| Nonlinear regression | 900 | 6 | 7.078 | R² 0.9734; RMSE 1.6699 |
| Positive regression with mixed types | 825 | 4 | 7.218 | R² 0.9302; RMSLE 0.1721 |
| Clustering with unequal scales | 800 | 4 | 6.110 | Adjusted Rand index 1.0000 |
| Anisotropic clustering | 800 | 2 | 6.140 | Adjusted Rand index 1.0000 |
| Clustering with mixed types and missingness | 900 | 6 | 6.234 | Adjusted Rand index 1.0000 |

All nine passed the provisional output checker, finished below 60 seconds, and completed without worker termination. No model-candidate errors were recorded. High clustering scores reflect deliberately separable synthetic groups; they do not predict performance on arbitrary hidden datasets.

## Larger cases and timeout behavior

Two further generated cases used a 20-second processing budget:

| Task | Learning rows | Features | Output rows | End-to-end seconds | Quality | Behavior |
|---|---:|---:|---:|---:|---|---|
| Classification | 20,000 | 41 | 4,000 | 18.562 | Held-out accuracy 0.9545 | Watchdog stopped a later candidate and returned the completed histogram model checkpoint |
| Clustering | 8,000 | 12 | 8,000 | 6.218 | Adjusted Rand index 1.0000 | Completed model comparisons normally |

The classification checkpoint was fitted on its training partition; the interrupted run did not complete a full-data refit. This is an intentional recovery path, and `run.json` reports the timeout and refit status. The separate watchdog test starts a worker that writes a valid checkpoint and then sleeps; the parent terminates it and validates the retained outputs.

These measurements establish working behavior at the listed sizes on this machine. The official dataset limits, hardware, runtime accounting, profile schema, metrics, and checker remain unknown. No official score or general 60-second guarantee is claimed.

The machine-readable benchmark report is included beside this file. Recreate the generated data and measurements with:

```sh
python benchmark.py --output practice --time-budget 15
```
