# Hackathon requirement checklist

The source is the supplied Unknown Dataset Hackathon overview. It describes the objective and submission but defers technical rules to kickoff. This checklist distinguishes implemented behavior from external information still needed.

| Requirement | Status and evidence |
|---|---|
| One program for classification, regression, and clustering | Implemented in `run.py`; tested through the real command-line entry point |
| Automatic dataset preparation and modelling | Implemented; preprocessing and candidate selection derive from data rather than dataset names |
| Predictions and structured profile | Implemented under the provisional contract; official schema pending |
| Task and feature metadata, required group count | Supported through JSON config and command-line options; official metadata format pending |
| Missing values and unequal scales | Imputation, scaling, missing indicators, and native histogram missing-value support |
| Categorical features and unknown categories | Bounded encoding and explicit handling of unseen values |
| Constant, duplicate, and irrelevant columns | Constants and duplicates profiled and removed during fitting; regularized and tree models address irrelevant features without guaranteeing removal |
| Imbalanced classification | Stratified validation, balanced-model candidates, and configurable metrics |
| Nonlinear relationships | Histogram boosting, extra trees, and bounded nonlinear clustering candidate |
| Independent execution | No network or AI service used during inference |
| 60 seconds per dataset | Default 55-second processing budget with worker termination and checkpoints; locally measured cases documented in VALIDATION.md; official hardware and limits pending |
| Pinned dependencies | `requirements.txt` pins direct and transitive dependencies tested on Python 3.12 |
| Standard entry point | Provisional `python run.py --config config.json`; official signature pending |
| Team names | All four supplied names included; four-person exception confirmed by Jean-Pierre |
| README approach, reproduction, and assumptions | Included |
| Nine official practice datasets and baseline | Not supplied; nine synthetic development cases included through a generator |
| Local official checker and public scoring rules | Not supplied; own checker validates the provisional outputs only |
| Public GitHub repository URL | Created at https://github.com/jean-pierreaouad/unknown-dataset-hackathon; final code publication verified separately |
| Frozen submitted commit | Requires final commit and submission through the organizer's channel |

## Before the deadline

1. Obtain any available kickoff package: exact entry point, input schema, output schema, profile definitions, official dependencies, CPU/memory and size limits, metric formulas, baseline, nine practice datasets, and checker.
2. Map the official inputs into `automl.data.load_data` or adapt the boundary around the modelling worker. Map `result.json` into the official prediction and profile outputs.
3. Select the official metric for each task. Defaults in this repository are assumptions.
4. Run the official checker and measure on the published CPU environment, including startup and output costs under its timing rules.
5. Resolve any dependency constraints and rerun tests from a clean environment.
6. Make the repository public, inspect its contents, confirm all four team names, and confirm the final commit.
7. Submit the public repository URL before the reported deadline of 21 October 2026 at 23:00 Asia/Beirut. Keep the submitted commit unchanged for reproducibility.

If the organizer never provides the detailed materials, submit the runnable project with these assumptions disclosed. Compatibility with an unspecified interface cannot be certified.
