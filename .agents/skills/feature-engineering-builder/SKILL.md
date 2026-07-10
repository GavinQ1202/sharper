---
name: feature-engineering-builder
description: Implement or review only Task 09 feature suggestions and safe stateless feature materialization approved for Sharper v0.1 in features.py. Use for suggest_feature_derivations, derive_features, FeatureSuggestion, FeatureSuggestionReport, and FeatureDerivationResult after prerequisite analytics tasks are complete. Do not use for Tasks 01-08, preprocessing.py, model preprocessing, fitted transformers, learned binning, group aggregates, target encoding, automatic feature search, or any feature API absent from SPEC.md.
---

# Feature Engineering Builder

Implement the locked Task 09 feature contract without turning v0.1 into a fitted feature-engineering or search system. Treat `SPEC.md` as authoritative, `AGENTS.md` as mandatory leakage policy, and `IMPLEMENTATION_PLAN.md` as the task and file boundary.

## Gate Before Work

1. Read `SPEC.md`, `AGENTS.md`, `README.md`, and Task 09 in `IMPLEMENTATION_PLAN.md`.
2. Confirm that Task 09 prerequisites are complete.
3. Inspect `features.py`, schema contracts, approved exports, feature tests, and API documentation relevant to Task 09.
4. If a requested module, API, result field, transformation, or workflow is absent from SPEC, propose a SPEC update and stop. Do not implement it.
5. Do not use this skill during Tasks 01-08. Do not create or modify `preprocessing.py`; it is not a v0.1 module in the locked architecture.

## Approved v0.1 Public API

Use exactly:

```python
suggest_feature_derivations(
    df: pd.DataFrame,
    *,
    schema: SchemaReport | None = None,
    target: str | None = None,
    max_suggestions: int = 50,
) -> FeatureSuggestionReport

derive_features(
    df: pd.DataFrame,
    suggestions: Sequence[FeatureSuggestion],
    *,
    copy: bool = True,
) -> FeatureDerivationResult
```

Use only the approved public result types:

- `FeatureSuggestion`
- `FeatureSuggestionReport`
- `FeatureDerivationResult`

Preserve the minimum fields frozen by SPEC and Task 09, including source columns, stable name, feature type, formula or parameters, rationale, risk, `requires_fit`, priority, and the applied, skipped, and warning information required by `FeatureDerivationResult`.

Do not add `create_ratio_features`, `create_difference_features`, `create_interaction_features`, `extract_datetime_features`, `create_binned_features`, `build_preprocessor`, `validate_no_target_leakage_risk`, or any other new public API. If another API is needed, recommend a SPEC update first.

## Suggestion Versus Transformation

`suggest_feature_derivations` suggests candidates and never modifies the input.

Approved v0.1 suggestion categories are:

- ratio;
- difference;
- product;
- deterministic datetime derivations;
- fixed or learned binning candidates;
- group aggregate candidates;
- target-aware candidates only as explicit risk-bearing suggestions.

All suggestion types must be structured, bounded, deduplicated, deterministically ordered, and carry rationale, risk, and `requires_fit`.

`derive_features` may materialize only the v0.1 stateless whitelist:

- ratio;
- difference;
- product;
- deterministic datetime features based on an explicit reference date when one is required.

It must reject every `requires_fit=True` suggestion with `ValueError` explaining that v0.1 supports suggestion only. Division by zero becomes missing and produces the approved warning. It must not silently mutate the input; honor the locked `copy` contract.

## Explicitly Deferred

Do not implement in v0.1:

- data-driven binning transformation;
- group aggregate transformer;
- target encoding, WOE, or supervised binning;
- target-aware transformation;
- sklearn-compatible public feature transformer;
- automatic feature search or best-interaction selection;
- genetic construction or deep feature synthesis;
- SHAP or model-driven feature selection;
- feature store, AutoML, MLflow, or registry infrastructure.

These remain suggestions, v0.2 work, or non-goals exactly as stated in SPEC. Do not create placeholder implementations.

## Leakage Invariants

- Never use target values as ordinary feature inputs.
- Never materialize target-derived or post-outcome features in v0.1.
- Never fit an imputer, encoder, scaler, binner, selector, category mapping, group mapping, or other preprocessing object before train/test split.
- Any future learned or target-aware transformation must fit on training rows or training folds only and transform validation/test without fitting.
- Test-set-only categories, extremes, or groups must never influence fitted state.
- Group aggregates, learned binning, target encoding, and any learned mapping are not v0.1 transformations.
- Date features that need an observation point must use an explicit reference date, never the current clock.
- Exclude target, IDs, constants, explicitly excluded columns, and obvious duplicates according to the locked contract.
- Bound each suggestion category and total suggestions; never generate an unbounded Cartesian product.

Because v0.1 has no fitted feature transformer, any leakage-sensitive capability must produce only a warning or structured suggestion. If it cannot be represented safely within `FeatureSuggestion`, recommend a SPEC update; do not execute it.

Model preprocessing, `ColumnTransformer`, and fitted `Pipeline` construction belong to `modeling.py` in Tasks 11-12, not to this skill or `features.py`.

## Module and Dependency Boundaries

- `features.py` may depend only on approved schema contracts and allowed core dependencies.
- `features.py` must not depend on modeling, evaluation, workflow, CLI, reporting, or visualization.
- `workflow.py` may later compose the approved feature APIs, but Task 09 must not move feature algorithms into workflow.
- Do not create `preprocessing.py`, registries, plugin systems, transformer hierarchies, or generic configuration engines.

## Required Tests

Write or update pytest in the same Task 09 implementation and only in files allowed by `IMPLEMENTATION_PLAN.md`.

Cover:

- every approved suggestion type;
- suggestion budgets, stable ordering, deduplication, and naming;
- exclusion of target, ID-like, constant, duplicate, and explicitly excluded inputs as required;
- ratio, difference, product, and deterministic datetime materialization;
- zero denominator becoming missing with an approved warning;
- missing values, non-finite values, and invalid column names;
- column-name collision behavior;
- explicit reference-date behavior and reproducibility;
- input non-mutation and the `copy` contract;
- `requires_fit=True` suggestions rejected by `derive_features`;
- learned binning, group aggregate, and target-aware candidates remaining suggestion-only;
- public exports, signatures, type hints, docstrings, result fields, and error messages;
- deterministic results and bounded combination counts.

Use small hand-checkable fixtures. Leakage regression tests must prove that target or held-out-only values cannot enter a materialized v0.1 feature. Do not add fit/transform lifecycle tests to Task 09 because no fitted public feature transformer is approved in v0.1.

## Review Mode

Treat as blockers:

- a public API or result field absent from SPEC;
- materialization of any `requires_fit=True` suggestion;
- target-aware, group aggregate, or learned-binning transformation;
- preprocessing fit before split;
- target, ID, future, or post-outcome leakage;
- unbounded candidate generation;
- hidden input mutation;
- Task 11-12 modeling behavior pulled into Task 09.

Lead with findings, cite exact paths and lines, explain the consequence, and recommend the smallest contract-preserving correction. Do not modify code during review-only work.

## Completion Output

After implementation, report:

1. files changed;
2. the approved public APIs implemented or changed;
3. the exact `FeatureSuggestion`, `FeatureSuggestionReport`, and `FeatureDerivationResult` contracts used;
4. which approved features remained suggestions and which stateless features were materialized;
5. tests added and commands run;
6. leakage risks rejected, warned about, or deferred;
7. v0.2 and non-goal capabilities deliberately not implemented;
8. SPEC, AGENTS, README, and Task 09 compliance.

