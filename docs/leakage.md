# Leakage safeguards

## Task 11 classification baseline

`train_classifier` supports a random stratified holdout only. It fixes original
integer row positions before schema-driven feature eligibility or fitting, then
calls `infer_schema` on `X_train` only. ID-like and constant decisions, numeric
imputation medians, categorical vocabularies, scaling state and fitted estimator
state therefore cannot observe holdout rows.

The API preserves source column order and never modifies the input DataFrame.
It retains only the feature-selected holdout table and aligned labels in
`TrainingResult`; `evaluate_classifier` predicts that holdout exactly once.

Users must explicitly pass known target-derived, posterior, future, or entity
fields in `exclude_columns`. A declared `time_column`, and any datetime or
timedelta dtype, is rejected before a random split. Numeric time encodings cannot
be inferred reliably and must also be declared with `time_column`.

Duplicate index and duplicate-row observations are recorded as warnings, not
silently corrected. Random holdout does not claim to solve entity/group leakage
or time ordering. Custom estimator randomness is caller-managed and is disclosed
in the returned warnings and limitations.
