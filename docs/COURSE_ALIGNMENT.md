# Course alignment and simplification decisions

Reviewed on 22 October 2026 against the supplied hackathon overview and the two
available course decks: *Machine Learning in the Real World* and *Data Science
Foundations*. We have not been given the detailed slides or labs for the other
chapters. This mapping is evidence of alignment, not instructor approval or a
claim that every method has already been taught.

| Course evidence | Project behavior | Where to read |
|---|---|---|
| Real World, slide 3: basic Python prerequisite | All project logic is Python; functions, loops, dictionaries, and small classes | All `.py` files |
| Real World, slides 27-32 and 39-43: data quality, training data, problem choice, cleaning, models, evaluation | A table-to-predictions pipeline with validation | `data.py`, `features.py`, `worker.py` |
| Foundations, slides 7-10: shape, types, nulls, statistics | Structured raw-data profile | `profile_frame()` |
| Foundations, slides 13, 24, 29: filling missing values and removing duplicates | Median/mode filling and removal of training constants and duplicate columns | `features.py` |
| Foundations, slide 38: mean-zero, unit-variance standardization | `StandardScaler` on numeric columns | `general_preprocessor()` |
| Foundations, slide 74: categorical encoding | One-hot encoding of features; label encoding of supervised classes | `features.py`, `supervised()` |
| Real World, slide 2: fundamentals, validation, decision trees, K-means in the syllabus | Linear candidates, validation split, decision trees, K-means | `models.py`, `worker.py` |
| Foundations, slides 63-65: problem, methods, results and documentation | README, student guide, reproducible benchmark and validation report | `README.md`, `docs/` |

The course also covers subjects such as images and language models. They are
not required for this assignment, which explicitly concerns numeric and
categorical tables and three declared task families.

## The one model extension

Random forests are retained to provide a practical nonlinear candidate. A
forest combines decision trees rather than introducing the previous boosting
and mixture-model searches. The supplied decks do not establish that random
forests were taught. We disclose this extension and explain voting, averaging,
and varying samples/features in the student guide. This is consistent with the
hackathon's permission to use open-source libraries, but that permission alone
does not show a student understands the method.

Logistic and Ridge regression are standard introductory model choices; the
available syllabus lists fundamentals, but the supplied two decks do not prove
those particular models have been covered in class. The guide explains their
role and the Ridge coefficient penalty rather than assuming prior knowledge.

## What became simpler

- One common, library-based imputation/scaling/one-hot pipeline.
- Three supervised candidates: linear, decision tree, and random forest.
- K-means, or its mini-batch variant based on table size, for clustering.
- No histogram-specific categorical transformer, boosting, Extra Trees,
  Gaussian mixtures, spectral clustering, or automatic log-target variant.
- Model choices and scoring live in a small `models.py` module.
- Comments explain purpose, and a reading guide follows one complete run.

The time-control subprocess, complete-result checkpoints, schema validation,
probability alignment, and finite-number checks remain. They are support code
needed for dependable timed execution, not extra machine-learning methods.
Removing them would simplify the file superficially while weakening explicit
assignment requirements.

## What this does not establish

This review cannot certify the lecturer's preferred coding style, the team's
actual understanding, an official grade, or compatibility with an unpublished
assessment interface. The final program still uses the provisional CSV/JSON
contract. The tests and benchmark establish only the behavior and sizes
reported in `VALIDATION.md`.
