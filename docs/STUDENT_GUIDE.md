# Understanding our hackathon project

This guide follows the simplified version. All project logic is Python. We use
pandas for tables, NumPy for numerical arrays, and scikit-learn for existing
machine-learning algorithms. We do not claim to have implemented the algorithms
inside these libraries ourselves.

## What the program does

Given a table and a task, the program prepares the columns, learns from the
available data, and writes predictions and a data summary. It does not ask an AI
service to make predictions. It works on the supplied table without identifying
which practice dataset it resembles.

| Task | Example | Training target available? | Result |
|---|---|---|---|
| Classification | Will a customer leave? | Yes: previous yes/no outcomes | A class and class probabilities |
| Regression | What is a house worth? | Yes: previous prices | A number |
| Clustering | Which customers are similar? | No | A group number for each row |

The task is supplied in the configuration. The program does not guess whether a
numeric target represents a class or a continuous measurement. For clustering,
the organizer supplies the number of groups, `n_clusters`.

## A sensible reading order

1. Read `examples/classification/config.json` and look at its two CSV files.
2. Read `automl/features.py`: the preparation steps are the same for all models.
3. Read `automl/models.py`: the small list of models and scoring functions.
4. Read `supervised()` and `clustering()` in `automl/worker.py`.
5. Read `automl/data.py` for loading, input checks, and profiling.
6. Read `run.py` for command-line options, time control, and output writing.
7. Read `check_output.py` and the tests to see what counts as valid output.

`make_practice.py` and `benchmark.py` generate and assess development examples.
The prediction program does not read their truth files.

## The variables

`X` is the feature table: the measurements used to learn. `y` is the target: the
answer we know for training rows. `evaluation` contains new rows whose answers
we need to predict. A `DataFrame` is a pandas table with named columns. A NumPy
array holds numbers or labels without the table's column names.

The target and an explicitly declared row-ID column are removed from `X`.
An ID tells us which row received a prediction; it is not automatically a useful
measurement. We preserve output row order and explicit IDs, including zeros in
IDs such as `007`.

## The Python structures you will see

| Structure | Meaning in this project |
|---|---|
| Dictionary, such as `config["task"]` | Store settings or report fields by name |
| Function, written with `def` | Give one step a reusable name |
| Class, such as `Recorder` | Keep related data and operations together |
| `try` / `except` | Handle an input or model failure without losing a completed result |
| `Pipeline` | Run preparation and a model in the right order |
| Sparse matrix | Store mostly zero values efficiently; mostly numeric tables normally stay dense |

`Cleaner` is a small class because scikit-learn expects preparation steps to
have `fit()` and `transform()` methods. Its `columns_` attribute remembers which
columns training retained. The trailing underscore is a library convention for
something learned during fitting. It does not change how Python stores it.

## Step 1: inspect the data

`load_data()` reads the CSV files and rejects inconsistent inputs: a missing
target, different feature sets, duplicate headers, or a target in the evaluation
file. Feature metadata takes priority. If a type is unspecified, numeric
columns and columns whose nonempty values are at least 98% numeric are treated
as numbers; other columns are categorical.

`profile_frame()` records rows, columns, missing cells, duplicates, constants,
and useful per-column statistics. Profiling reports the supplied data before
model preparation. This is our programmatic exploratory data analysis; the
hackathon asks for a structured profile rather than a graphical dashboard.

Duplicate-column detection uses a short fingerprint to find likely matches,
then checks their actual values. The extra numeric checks prevent infinity
from appearing in a JSON report when inputs contain extreme values.

## Step 2: prepare features

The `general_preprocessor()` function builds a scikit-learn `Pipeline`: an
ordered sequence of operations. `ColumnTransformer` applies one sequence to
numeric columns and another to categories.

| Problem | Our treatment | Reason |
|---|---|---|
| Repeated or unchanging columns | Remove them using training data | They add no new varying information |
| Missing numeric value | Fill with the training median | Keep the row with a typical value |
| Missingness itself contains information | Add a missing-value indicator | The model can distinguish a filled value |
| Different numeric scales | Standardize using training mean and standard deviation | Help linear models and distance calculations |
| Missing category | Fill with the most frequent training category | Supply a valid value |
| Categories such as red/blue | One-hot encode | Supply numbers without inventing an order |
| Very many categories | Pool rare values; at most 64 encoded values per column | Limit the number of new columns |
| A previously unseen category | Use the rare-value group, or all zeros if no such group exists | Keep predictions working |

For example, a rose's color might become separate red and blue indicator
columns. Encoding red as 1 and blue as 2 would imply an ordering that color does
not have, so we use one-hot encoding for input features.

Scaling is unnecessary for decision-tree splits, but it does not change the
ordering of numeric values. Using the same preparation for all supervised
models makes the project easier to follow.

`fit()` learns preparation values from training rows. `transform()` applies
those values to other rows. Fitting the imputer or scaler on validation data
would leak information about the data used to judge the model. A Pipeline
keeps preparation inside each candidate's training process. Type inference
uses the supplied training feature table; it never reads evaluation targets.

## Step 3: learn and compare supervised models

For ordinary datasets, we reserve about 20% of the supplied labelled rows for
validation. The other approximately 80% trains the candidates. Classification
uses a stratified split to preserve classes as far as possible. Additional
checks keep classes represented on both sides when they have at least two
examples.

There is also a simple baseline: predict class frequencies for classification
or a constant for regression. It gives us a reference and an early result if a
later model fails or runs out of time.

| Model | Plain explanation | Why it is included |
|---|---|---|
| Logistic regression | Combine features to estimate class probabilities | A simple classification starting point |
| Ridge regression | Fit a linear relationship with a penalty on large coefficients | A stable regression starting point |
| Decision tree | Follow learned questions such as “is area greater than 100?” | Capture nonlinear patterns |
| Random forest | Train 60 decision trees on varied samples/features, then vote or average | Reduce the instability of a single tree |

Despite its name, logistic regression is used for classification. Ridge is the
linear-regression variant used for continuous targets.

Decision trees and K-means appear explicitly in the supplied course syllabus.
The supplied overview also lists fundamentals and validation. Only two course
decks are available here, so we cannot establish every algorithm you have
already studied. The random forest is an explicitly disclosed extension of
decision trees, not a claim about what your lecturer taught. The project no
longer uses boosting, Extra Trees, Gaussian mixtures, spectral clustering,
special histogram category encoding, or a log-target model search.

Each candidate predicts the reserved validation rows. The metric in the config
decides which candidate is best. We then try to train a fresh copy of that model
on all the supplied labelled rows, including the former validation rows, before
predicting evaluation rows. The validation score describes the selection step;
it is not a score measured on the final private evaluation data.

Very small datasets or classes with only one example cannot support our normal
validation split. Those cases use a fixed decision tree and report that no
validation comparison was possible. A single-class target simply returns that
class. The baseline remains available if fitting fails.

## Step 4: understand the scores

| Metric | Interpretation |
|---|---|
| Accuracy | Fraction of correct class predictions |
| Balanced accuracy | Average recall across classes, giving each class equal importance |
| F1 | Balance of precision and recall; macro averages classes equally |
| ROC AUC | How well probabilities rank classes across decision thresholds |
| Log loss | Penalize wrong probabilities, especially confident mistakes |
| RMSE | Typical numeric error, with greater penalty for large errors |
| MAE | Average absolute numeric error |
| R-squared | Improvement relative to predicting the target mean; it can be negative |
| RMSLE | Error after taking `log(1 + value)`; requires nonnegative targets |

The code chooses the largest internal score. It negates error metrics so a
smaller error becomes a larger score: RMSE 2 becomes -2 and beats RMSE 5 (-5).
The class list specifies the order of probability columns. Label encoding
temporarily gives target classes integer IDs; predictions are converted back
to the original parsed labels before writing.

When the requested classification metric emphasizes rare classes, we use
balanced class weights. This makes errors on rare classes count more during
fitting; it cannot guarantee good predictions for every minority class.

## Step 5: clustering

We prepare the features, then use K-means with the requested group count.
K-means repeatedly assigns rows to their nearest center and updates the centers
from those assignments. Scaling matters because distance should not be
dominated by a feature just because it uses larger units.

For more than 5,000 training rows, we use MiniBatchKMeans. It updates centers
using smaller batches and is a faster variant of the same central idea. There
is no clustering-model competition in this version.

Silhouette compares within-group closeness with separation from other groups.
We calculate it on up to 1,500 rows as a diagnostic. It does not use target
labels and does not certify meaningful real-world groups. K-means works best
for compact groups; curved or elongated groups may produce poorer results.

If every row is identical, there are no meaningful geometric groups to find.
The code still splits training assignments to meet the required count and
reports a warning. If clustering fails before a learned prediction is saved,
the output identifies the emergency fallback rather than claiming it learned
useful groups.

## Step 6: time control and outputs

The hackathon gives each dataset 60 seconds. `run.py` starts the ML worker in a
separate Python process and reserves a little time for writing outputs. The
default processing budget is 55 seconds. `Budget.left()` tells the worker how
much time remains before starting another model.

`Recorder.save()` writes a complete prediction to a temporary file and then
replaces the checkpoint. This prevents a partially written result from
replacing a good one. If a later fit times out, the parent keeps the last
complete result. These support details are more advanced Python, but the reason
for them is the assignment's time limit. If no initial result completed, the
program reports failure. We have measured local cases, not every possible
dataset or the organizer's CPU.

`predictions.csv` is the easiest output to inspect. `profile.json` describes the
data, `run.json` explains what ran, and `result.json` contains the complete
result. Classification also writes `probabilities.csv`. Check the exit code
and run the output checker before trusting files left in an output directory.

## Practice explaining the project

Run an example, open its `run.json`, and answer these questions together:

1. Which columns are features, which column is the target, and what is one row?
2. Why do classification and regression need targets but clustering does not?
3. Which preparation values are learned, and why are they learned only from training rows?
4. What does one-hot encoding do to a color column?
5. Why is validation different from training and from private evaluation?
6. Which model won this example, and which metric chose it?
7. Why might a single decision tree make different predictions from a forest?
8. Why can accuracy hide a problem when classes are imbalanced?
9. What assumptions does K-means make about useful groups?
10. What happens if a later candidate runs out of time?

Our tests check behavior, outputs, and selected learning examples. They do not
prove an official score or substitute for understanding the code. The official
kickoff interface and checker are still missing, so interface compatibility
must remain an explicit assumption.
