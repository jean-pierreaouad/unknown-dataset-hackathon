# Final project audit

The project was reviewed on 21 October 2026 against the supplied hackathon overview, the documented provisional contract, the code, dependency pins, tests, and published repository. The overview delegates technical details to kickoff; those materials remain unavailable.

## Findings fixed

| Finding | Reproduction and correction |
|---|---|
| Numeric-looking categories and identifiers lost leading zeros | CSV values such as `001` and `1` were both inferred as integer 1 before categorical metadata was applied. Declared categorical columns and explicit identifiers are now parsed as text before inference. Regression tests check distinct codes and preserved row IDs. |
| Extreme finite features produced an infinite median | Two values of `1e308` overflowed when the median implementation averaged them. A median calculation using bounded addition or subtraction keeps statistics finite. Positive, negative, and opposite-sign cases are tested. |
| Optimized Python disabled the output checker | `python -O` removes assertions, allowing malformed output to be reported as valid. Validation now uses explicit exceptions. Additional checks compare probability CSV contents with JSON, enforce task-specific prediction types, and reject nonfinite numbers including exponent overflow. |

## Verification

- All 42 tests passed in 50.585 seconds for the full suite on the documented Windows Python 3.12 environment.
- The pinned dependency set passed its consistency check.
- All 11 earlier benchmark and larger-case output folders passed the stricter checker.
- The nine synthetic benchmark cases and two larger cases retain their originally recorded timing and quality measurements; model-selection settings were unchanged by these fixes.
- The release ZIP and published source are checked for matching file contents after publication.

## Requirement assessment

The implementation covers the three task families, explicit task and feature metadata, required clustering count, automatic preparation, model selection, predictions, profiling, independent CPU execution, pinned dependencies, team names, and README instructions. Missingness, categories, unequal scales, duplicates, constants, imbalance, nonlinear patterns, rare classes, unseen categories, and timeout recovery have direct test coverage.

These conditions remain external and unverified:

- The official input/output signature and profile field definitions.
- The exact quality, profiling, and speed formulas.
- Published feature, row, category, memory, and CPU limits.
- The official dependency environment and local checker.
- The nine official practice datasets and baseline.
- End-to-end timing under the organizer's assessment environment.

Consequently, this is a tested implementation of the available overview and its declared assumptions. It is not certified compatible with a technical contract that has not been supplied. A 55-second processing budget and local timing evidence do not prove a universal 60-second guarantee for arbitrary inputs.

## Remaining engineering tradeoffs

A single holdout conserves time but has higher selection variance than repeated validation. Tiny or extremely rare-class datasets use a fixed-model fallback. Clustering selects by silhouette, which need not match a hidden label-based metric. High-cardinality categories are pooled, potentially losing signal. Numeric feature magnitudes above `1e30` are clipped. Memory is not externally capped, and a timeout before the initial checkpoint produces an explicit failure. These are documented design limits rather than evidence of optimal performance.

## Portfolio improvements to consider

Keep the assessed repository URL stable. Preserve the submitted commit before making changes for a portfolio.

1. Lead the README with a descriptive project title such as Budgeted Tabular AutoML, a concise explanation of the problem, and one working example. Keep team credits visible and move deadline and team-size administration into the hackathon documentation.
2. Add GitHub Actions to install the pinned environment and run the suite on Windows and Linux with Python 3.12. Add a status badge only after the workflow has run successfully. This supplies evidence beyond local test claims.
3. Show a short demo from input CSV through profiling to predictions. Include a compact results table with dataset size, model, quality, baseline, and wall-clock time. Clearly label synthetic data; later add public real-world datasets with separate held-out evaluation.
4. Add a small architecture diagram and explain decisions: training-only transformations, bounded category encoding, limited model search, full-data refitting, and checkpoint recovery under deadlines.
5. Improve repository discovery with relevant topics and a concise description. A useful topic set is `automl`, `tabular-data`, `scikit-learn`, `machine-learning`, and `python`.
6. Agree on a license with the team, record actual contributions without inventing roles, and create a versioned release. A short roadmap should distinguish implemented features from future ideas.

The first three improvements would provide the clearest evidence that the project is usable and maintained. Additional branding or a hosted website is optional and is not a contest requirement.

References: [GitHub README guidance](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes), [continuous integration](https://docs.github.com/en/actions/get-started/continuous-integration), and [repository topics](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/classifying-your-repository-with-topics).
