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

## Task 20 audit handoff

The Task 20 integration keeps the raw-carrier boundary explicit:

- `V02WorkflowRequest.data` is the primary raw DataFrame carrier.
- `V02AuditRequest.reference` is the optional secondary DataFrame used only by
  the Task 16 audit for reference/current profile and missingness evidence.

The reference frame is not a score-validation reference, policy reference,
post-loan reference, or governance candidate source. Task 20 does not infer
those meanings from the presence of a second frame. Passing the same DataFrame
object as both `data` and `reference` is allowed by the carrier contract; both
references are read-only.

Task 16 remains the sole owner of audit, leakage, input-profile, and
missingness-drift semantics. Task 20 only passes the declared audit inputs to
that owner and forwards its frozen result to downstream opt-in owners. It does
not repair data, delete rows, alter features, or reinterpret audit findings as
policy or governance decisions.

`V02WorkflowResult` and the Task 20 report do not retain the primary or audit
reference raw DataFrames. They do retain typed owner results and their approved
summary/evidence tables; this boundary does not claim that every upstream
result type is raw-data-free. Any remaining leakage warning, limitation,
effective sample count, maturity status, and reference/current comparison must
be interpreted from the owning Task 15/16 result contract.
