# Unknown Dataset Hackathon

This project automatically profiles a tabular CSV dataset and solves its declared classification, regression, or clustering task. It selects preparation and models from the supplied data, then writes predictions, a structured profile, and execution diagnostics. It runs on a CPU without internet access, AI services, external datasets, or downloaded pretrained models.

Public repository: https://github.com/jean-pierreaouad/unknown-dataset-hackathon

The implementation follows the supplied hackathon overview. **The official kickoff input/output contract, environment, size limits, checker, and scoring formulas were not supplied. This repository therefore defines a provisional interface.** It is a working project that must be compared with those materials before claiming assessment compatibility. See [the requirement checklist](docs/REQUIREMENTS.md) and [interface specification](docs/CONTRACT.md).

## Team

- Jean-Pierre Aouad
- Maher Karam
- Alex Yammine
- Marc Abou Mikheal

The overview specifies three students per team. Jean-Pierre confirmed that this four-person team has approval. The reported submission deadline is **21 October 2026 at 23:00, Asia/Beirut**.

## Install and run

Python **3.12** is the tested runtime. Install pinned dependencies into an isolated environment. Dependency installation requires internet access; running the program does not.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run.py --config examples/classification/config.json
.\.venv\Scripts\python.exe check_output.py examples/classification/results --expected-rows 20
```

Linux or macOS:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run.py --config examples/classification/config.json
.venv/bin/python check_output.py examples/classification/results --expected-rows 20
```

The examples contain small generated datasets. Run the regression and clustering examples by replacing the config path with `examples/regression/config.json` or `examples/clustering/config.json`.

The commands below use `python` for the Python executable in the installed environment.

```sh
python run.py --task classification --train train.csv --test test.csv --target outcome --output results
python run.py --task regression --train train.csv --test test.csv --target price --metric rmse --output results
python run.py --task clustering --train customers.csv --n-clusters 4 --output results
```

Every CSV needs a header. Supervised training includes the target; evaluation contains features only. Classification returns original labels, regression returns finite numbers, and clustering returns integer group identifiers. Supply `--id-column customer_id` to exclude a row identifier from modelling and retain its values in the prediction file. Otherwise output identifiers are zero-based row positions, preserving input order.

## Inputs and outputs

`--config` accepts a JSON object. Paths inside a config resolve relative to that config; explicit command-line arguments override config fields. For example:

```json
{
  "task": "classification",
  "train": "train.csv",
  "test": "test.csv",
  "target": "outcome",
  "feature_types": {"age": "numeric", "contract": "categorical"},
  "metric": "accuracy",
  "seed": 42,
  "time_budget": 55,
  "threads": 1,
  "output": "results"
}
```

Feature types may be supplied as a dictionary in the config or a path to a JSON dictionary with `--feature-types types.json`. Metadata takes precedence; undeclared types are inferred from training features. Numeric-looking strings are inferred as numeric when at least 98% of nonempty, nonmissing entries parse as numbers.

Explicit identifiers and declared categorical columns are read as text before numeric inference, preserving distinctions such as `001` versus `1`. The output checker stays active under `python -O` and checks CSV probability values against the complete JSON result. See [the final audit](docs/FINAL_AUDIT.md) for review findings and remaining requirements.

Output files:

| File | Contents |
|---|---|
| `predictions.csv` | `row_id,prediction`, in evaluation order; clustering uses training order when no evaluation file is given |
| `probabilities.csv` | Classification probabilities, with `class_0`, `class_1`, etc. in the order of `result.json` → `classes` |
| `profile.json` | Raw feature statistics, types, missingness, constants, duplicates, and supervised target summary |
| `run.json` | Selected model, holdout score, candidate diagnostics, elapsed time, timeout flag, and warnings |
| `result.json` | Complete authoritative result containing predictions, identifiers, profile, diagnostics, and classification probabilities |

`result.json` is the integration point for an eventual official adapter. Each file is replaced atomically; a nonzero exit code indicates failure. Treat the latest successful `result.json` as authoritative. Do not read an output directory while a run is in progress. Input and output locations must be writable where appropriate, and separate concurrent runs must use different output directories.

## How the program works

The program first validates schemas and builds a raw-data profile. It respects declared feature types, excludes an explicitly declared identifier, and keeps target values out of the feature matrix. Training and evaluation CSVs must contain the same features, although column order may differ.

Preparation removes constants and exact duplicate columns using the current training partition. Numeric nonfinite or unparseable values become missing. Numeric magnitudes above `1e30` are clipped for numerical stability. For linear models, numeric values are median-imputed and standardized, missing indicators are added, and categories are imputed and one-hot encoded with bounded cardinality. Unknown categories are accepted. Histogram models learn low-cardinality categorical splits directly, pool rare known categories, and treat unseen categories as missing. Transformations are fitted on training data and then applied to validation or evaluation rows.

Supervised model selection uses one reproducible holdout, stratified for classification. Types are inferred from the complete labelled training table; learned imputation, scaling, column removal, and category vocabularies are fitted on the training partition. The program compares a constant baseline, a linear model, histogram gradient boosting, and extra trees. It also considers balanced classification and a log-target regression variant when applicable. It retains the best holdout score and attempts a full-training refit. No evaluation targets are used. Small datasets or classes with fewer than two examples bypass validation and use a fixed histogram model, with a constant fallback.

Clustering median-imputes numeric features, standardizes them, and encodes categories. It compares k-means, minibatch k-means for larger tables, Gaussian mixtures when dimensionality permits, and a spectral candidate on modest datasets. Selection uses silhouette on the same fixed sample, with at most 1,500 rows; no reference labels are read. Silhouette favors separated groups and is not a guarantee of agreement with hidden labels. If identical feature vectors collapse the requested groups, training assignments are split deterministically to return the requested count; such groups have no geometric separation and generate a warning. If a clustering evaluation file is supplied, supported fitted models assign its rows without requiring every group to appear in that file.

## Metrics and time control

| Task | Supported `metric` values | Default |
|---|---|---|
| Classification | `accuracy`, `balanced_accuracy`, `f1_macro`, `f1_weighted`, `roc_auc`, `log_loss` | `accuracy` |
| Regression | `rmse`, `mae`, `r2`, `rmsle` | `rmse` |
| Clustering | `silhouette` | `silhouette` |

Model selection maximizes its internal score. For RMSE, MAE, RMSLE, and log loss, the recorded score is the **negative loss**. RMSLE requires nonnegative training targets and clips predicted values to zero. Multiclass ROC AUC uses weighted one-vs-rest probabilities. Classification probabilities follow the original class order stored in the result.

The parent process uses only Python's standard library and launches a worker with controlled CPU threads. The default total processing budget is 55 seconds, leaving margin against the overview's 60-second assessment limit. The allowed configurable range is 3–55 seconds; the worker is stopped before the configured budget to leave time for output writing. Completed prediction checkpoints survive a later timeout or model failure. If loading, imports, or initial profiling fail before a checkpoint exists, the run fails explicitly. Extremely small budgets, very large tables, very wide outputs, or slow disks can prevent timely results. There is no universal runtime or memory guarantee outside measured sizes; the official environment and dataset limits still need verification.

The program defaults to one CPU thread and a fixed seed. Runtime can affect which candidates finish; operating systems and numerical libraries can also affect results. Reproduce with the same versions, seed, environment, and sufficient time. Training does not use GPUs or network services.

## Test and reproduce

```sh
python -m pip check
python -m unittest discover -s tests -v
python benchmark.py --output practice --time-budget 15
```

The benchmark generates **nine synthetic development datasets**, runs the real entry point, checks outputs, and computes local metrics using separately stored truth files. Truth files are never passed to the modelling program. The generator and benchmark have no role in production prediction. These are not the official nine practice datasets, and local metrics are not contest scores.

The tests cover train/evaluation schema checks, preprocessing isolation, declared types, high-cardinality and unknown categories, missingness, constants, duplicates, label preservation, rare classes, all three tasks, probability validity, and actual worker termination with checkpoint recovery. See [validation results](docs/VALIDATION.md) for the measured environment, sizes, and outcomes.

## Assumptions and limitations

- The task, target name, required cluster count, and any declared feature types are explicit inputs. The program does not guess the task or target from a table.
- This provisional contract uses UTF-8 comma-separated CSV plus JSON configuration. Only ordinary tabular numeric and categorical features are supported. Dates, text, images, and time-series forecasting receive no specialized interpretation.
- CSV missingness follows pandas' default parsing, including empty fields and standard NA tokens. A literal category such as `NA` can therefore be treated as missing. The official missing-value rules must be checked.
- Features with at most one nonmissing unique value are marked constant in the profile. Preparation can retain a value-plus-missing feature because missingness itself may contain information. Duplicate profile columns are exact raw columns, checked after hash shortlisting.
- Only explicitly identified row IDs are removed. High uniqueness alone is insufficient to remove a potentially useful feature.
- A single holdout saves time but can select an inferior model by chance. There is no claim of optimality or a guaranteed private-dataset score.
- For large feature spaces, high cardinality is bounded, but memory is not externally capped. Size and memory limits from kickoff may require further constraints.
- The dependency versions are tested on Python 3.12. The actual assessment image may require a revised compatible lockfile.
- The official profile schema, entry point, metrics, runtime accounting, and scoring formulas are pending. Replace this provisional adapter when they arrive.



## Library references

Implementation references: [histogram gradient boosting and categorical handling](https://scikit-learn.org/1.7/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html), [preprocessing](https://scikit-learn.org/1.7/modules/preprocessing.html), [clustering](https://scikit-learn.org/1.7/modules/clustering.html), and [CPU resource management](https://scikit-learn.org/stable/computing/parallelism.html).
