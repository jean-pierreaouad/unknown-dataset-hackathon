# Simplification audit: 22 October 2026

This revision implements the request to make the existing project easier to
understand at introductory-course level. The previous published project remains
separate. Synced course sources were read-only references.

## Changes

- Three supervised candidates: a linear model, a decision tree, and a small
  random forest. The forest is an explained extension of decision trees.
- One shared median/mode, scaling, and bounded one-hot preparation pipeline.
- Standard storage-density selection avoids forcing mostly numeric mixed
  tables into slow sparse tree fitting. Classifier forests consider a small
  random feature subset at each split.
- K-means or its mini-batch variant for clustering.
- Removed boosting, Extra Trees, Gaussian mixtures, spectral clustering,
  custom histogram category encoding, and log-target search.
- Kept input checks, profiling, labels and IDs, probabilities, time protection,
  complete checkpoints, dependencies, team details, and the existing interface.
- Added `STUDENT_GUIDE.md`, `COURSE_ALIGNMENT.md`, and a clearer README.

## Evidence and limits

All 47 tests passed. Two checks tied to removed implementations
were replaced by equivalent behavior checks, and five learning/integration
checks were added. Both versions were compared on identical generated data,
larger timed cases were measured, and exact dependencies were checked.
`VALIDATION.md` records results, including quality reductions.

The two supplied course decks support the workflow and preparation concepts,
and the syllabus names validation, decision trees, and K-means. They do not
establish every method already taught. The forest remains a disclosed extension.
All application logic is Python with NumPy, pandas, and scikit-learn.

Official assessment compatibility, a grade, private-data performance, and a
successful course submission cannot be certified from the available materials.
Version 0.2.0 follows the earlier audited commit `b45b06415b54ddc406be731cad110967b2989869`,
which remains in repository history. Publishing a later revision does not
establish which commit the organizer accepts or assesses.
