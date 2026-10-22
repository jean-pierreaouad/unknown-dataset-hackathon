# Simplified-version validation

Tested on 22 October 2026 using Python 3.12.14 on Windows AMD64, Intel
Family 6 Model 126, with 8 logical CPUs. The prediction worker used one thread.
All ten installed versions matched `requirements.txt`; all ten active package
dependency constraints passed a distribution-metadata check. The existing
isolated environment has no pip module, so this review does not claim a new
`pip check` run there. Fresh Windows and Ubuntu Linux installations subsequently passed GitHub Actions,
including `pip check`, all 47 tests, and all three examples. See [CI evidence](CI.md).
The organizer's image remains unverified. See `dependency_report.json` for the
original local environment check.

## Automated checks

**47 tests passed**, with no failures, errors, or skipped tests.
The suite took 95.906 seconds in total, not per dataset.

Behavior checks cover schemas, metadata, preprocessing isolation, missing and
extreme values, duplicates, categories, leading zeros, probabilities, rare
classes, tiny tables, all tasks, cluster counts, and forced timeout recovery.
Two implementation-specific checks were adapted: the removed custom encoder
test became a bounded one-hot/unseen-category test; the histogram test now
checks that all three simpler models fit high-cardinality data without errors.
They were replaced rather than deleted to obtain a passing result.

Five added integration tests check held-out nonlinear classification (accuracy
above 0.90), nonlinear regression (R-squared above 0.80), built-in breast cancer
teaching data (accuracy above 0.90), built-in diabetes teaching data (R-squared
above 0.30), and 5,200-row mini-batch clustering (adjusted Rand index above 0.95).
The production program does not receive these test datasets or their answers.

Run `python -m unittest discover -s tests -v` to repeat the tests.

## Same-data comparison

Both versions received byte-identical generated CSV inputs and separate truth
files, using the unchanged generator, seed 42, a 15-second budget, and the same
pinned environment. These are synthetic development cases, not official data.

| Case | Metric | Previous | Simplified | Selected model | Seconds |
|---|---|---:|---:|---|---:|
| classification_imbalanced | balanced_accuracy | 0.8748 | 0.8475 | random_forest | 7.109 |
| classification_nonlinear | accuracy | 0.9778 | 0.9644 | decision_tree | 5.828 |
| classification_mixed | accuracy | 0.8840 | 0.8840 | linear | 6.016 |
| regression_linear | r2 | 0.9956 | 0.9956 | linear | 6.312 |
| regression_nonlinear | r2 | 0.9734 | 0.8957 | random_forest | 6.453 |
| regression_mixed_positive | r2 | 0.9302 | 0.9397 | random_forest | 6.235 |
| clustering_scaled | adjusted_rand_index | 1.0000 | 1.0000 | kmeans | 5.766 |
| clustering_anisotropic | adjusted_rand_index | 1.0000 | 1.0000 | kmeans | 5.516 |
| clustering_mixed | adjusted_rand_index | 1.0000 | 1.0000 | kmeans | 5.250 |

All nine simpler runs passed the output checker, finished below 60 seconds,
and reported no timeout or candidate error. Times include startup and writing.
Runs occurred at different moments on a shared computer, so these are not a
controlled speed comparison. High clustering scores reflect separated groups.
Full metrics and diagnostics appear in `comparison_report.json`.

Some classification results and nonlinear regression quality decreased with
simplification. Linear regression and these clustering cases retained their
scores; the mixed positive regression case improved. No equivalent hidden-data
accuracy, official score, or grade is claimed. Recreate the current benchmark
with `python benchmark.py --output practice --time-budget 15`.

## Larger timed cases

Classification uses 20,000 training and 4,000 evaluation rows, 40 numeric
features plus a category, and missing values. Clustering uses 8,000 rows and
12 features. Exact generator parameters are in `larger_case_report.json`.
The 20-second budget exercises the earlier audit's case; 55 is the default.

| Task | Training rows | Features | Budget | Seconds | Accuracy or ARI | Selected model | Timed out | Full refit |
|---|---:|---:|---:|---:|---:|---|---|---|
| classification | 20,000 | 41 | 20 | 18.500 | 0.9397 | random_forest | True | False |
| classification | 20,000 | 41 | 55 | 21.312 | 0.9433 | random_forest | False | True |
| clustering | 8,000 | 12 | 20 | 5.891 | 1.0000 | minibatch_kmeans | False | True |

When time expires, the latest complete checkpoint can precede full-data
refitting; diagnostics report this. Constant/tree or deterministic-group
fallbacks are emergency results, not evidence of useful learned performance.
The forced-timeout test checks checkpoint recovery independently.

## Limits

K-means assumes compact groups; curved or elongated groups can be worse than
with the removed methods. Pooling rare categories can lose signal. A single
holdout has selection variance. Magnitudes over `1e30` are clipped. Memory is
not externally capped. A timeout before the first checkpoint is an explicit
failure. Local timing does not prove a universal 60-second guarantee.

The official contract, profile schema, checker, scoring formulas, dataset
limits, and CPU image remain unavailable. This is a tested implementation of
the supplied overview and disclosed assumptions. Version 0.2.0 is a later educational revision; earlier commits remain in
repository history. Publication does not establish a course submission or
which commit the organizer assesses.
