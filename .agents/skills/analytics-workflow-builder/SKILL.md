---
name: analytics-workflow-builder
description: Implement or review only the v0.1 analytics capabilities and workflow integration already approved in Sharper's SPEC.md and IMPLEMENTATION_PLAN.md. Use for Task 03 DataFrame summaries, Task 04 data-quality rules, Task 05 minimal workflow integration, Tasks 07-08 approved analysis APIs, or Task 13 final workflow integration in summary.py, quality.py, analysis.py, and workflow.py. Do not use for Task 01 packaging, unapproved profiling.py or mining.py modules, feature engineering, visualization implementation, model training, or APIs absent from SPEC.md.
---

# Analytics Workflow Builder

Implement the locked v0.1 analytical contracts without adding modules, APIs, result fields, or later-phase behavior. Treat `SPEC.md` as authoritative, `AGENTS.md` as mandatory engineering policy, and `IMPLEMENTATION_PLAN.md` as the task order and file boundary.

## Gate Before Work

1. Read `SPEC.md`, `AGENTS.md`, `README.md`, and the current task in `IMPLEMENTATION_PLAN.md`.
2. Confirm that all prerequisite tasks are complete and that the request belongs to an approved task listed below.
3. Inspect only the approved modules, public exports, result types, tests, workflow, and reporting consumers relevant to that task.
4. If the requested module, API, field, result type, or behavior is absent from `SPEC.md`, propose a SPEC update and stop. Do not implement it.
5. Do not use this skill for Task 01, Task 02, Task 06, or any feature, visualization, modeling, evaluation, packaging, release, or unrelated reporting task.

## Approved Tasks and Boundaries

### Task 03 — Schema and DataFrame Summary

The first use of this skill is the `summarize_dataframe` vertical slice after Task 01 has established the package contract.

Implement only:

```python
summarize_dataframe(
    df: pd.DataFrame,
    *,
    schema: SchemaReport | None = None,
) -> DataFrameSummary
```

Work in `summary.py` and only the public exports, tests, and API documentation allowed by Task 03. Return the frozen `DataFrameSummary`; do not substitute a plain or nested `dict`.

### Task 04 — Fixed Data-Quality Rules

Implement only:

```python
check_data_quality(
    df: pd.DataFrame,
    *,
    schema: SchemaReport | None = None,
    missing_threshold: float = 0.40,
) -> QualityReport
```

Work in `quality.py` and the files allowed by Task 04. Use the approved `QualityReport` and quality issue fields: code, severity, column, count or ratio, message, and suggestion. Report issues; never repair data automatically.

### Task 05 — Minimal Workflow Integration

Compose the already implemented schema, summary, and quality public APIs into the approved `AnalysisRun` through `run_analysis`. Keep `workflow.py` thin. Do not copy summary or quality algorithms into workflow. `reporting.py` consumes the structured results; workflow does not render or write reports.

### Task 07 — Univariate Analysis, Correlations, and Outliers

Implement only the signatures and result types frozen by SPEC:

- `analyze_numeric_features(...) -> NumericAnalysis`
- `analyze_categorical_features(...) -> CategoricalAnalysis`
- `compute_correlations(...) -> CorrelationAnalysis`
- `detect_outliers(...) -> OutlierAnalysis`

Do not add datetime-analysis, missingness-pattern, candidate-insight, generic bivariate, or generic multivariate public APIs. Those APIs are not part of locked v0.1.

### Task 08 — Group and Target Relationships

Implement only:

- `compare_groups(...) -> GroupComparison`
- `analyze_target_relationships(...) -> TargetAnalysis`

Keep classification and regression target behavior explicit. Do not train models. Respect the single group key, numeric value columns, group budget, and explicit target/task contract.

### Task 13 — Final Workflow Integration

Integrate only approved, already implemented domain results into `AnalysisRun`. Preserve the locked budgets, warnings, ordering, optional modeling behavior, and Python/CLI consistency. Do not add hidden analysis to `workflow.py`.

## Locked Module Boundaries

- `summary.py` owns dataset-level and column-level descriptive summaries.
- `quality.py` owns quality rules, severity, evidence, and structured findings.
- `analysis.py` owns approved numeric, categorical, correlation, outlier, group, and target analyses.
- `workflow.py` only composes public domain APIs into `AnalysisRun`.
- `reporting.py` consumes `AnalysisRun` and approved result objects without hidden recomputation.

Do not create `profiling.py`, `mining.py`, or `report.py`. Do not make domain modules depend on workflow, CLI, reporting, or visualization.

## Result Contracts

Use only the public result types and frozen fields defined by SPEC and the current implementation task:

- `DataFrameSummary`
- `QualityReport` and its approved quality issue type
- `NumericAnalysis`
- `CategoricalAnalysis`
- `CorrelationAnalysis`
- `OutlierAnalysis`
- `GroupComparison`
- `TargetAnalysis`
- `AnalysisRun`

Use approved DataFrame detail schemas where SPEC assigns tabular detail. Prefer the named immutable result dataclasses required by the project. Do not return an ad hoc `dict`, add convenience fields, change field meaning, or invent a common result hierarchy.

If implementation reveals that a result needs another field, stop and recommend updating the public contract in `SPEC.md` before changing implementation.

## Analytical Requirements

- Design behavior around the approved analytical task, not around exposing pandas methods.
- Do not reduce `summarize_dataframe` or analysis functions to a thin `DataFrame.describe()` wrapper.
- Use `describe()` internally only where it produces a verified component of the locked result contract.
- Preserve effective sample sizes, missing handling, limits, truncation, ordering, warnings, and skipped reasons required by SPEC.
- Keep results deterministic and suitable for direct consumption by `workflow.py` and `reporting.py`.
- Never print as the primary output, mutate the input silently, state causal conclusions, or hide analysis in rendering code.
- Require explicit target and `Literal["classification", "regression"]` for target-aware analysis.

## Phase Discipline

When executing a task:

- implement only that task's approved API and acceptance criteria;
- do not pull work forward from later tasks;
- do not implement Task 04 quality rules during Task 03;
- do not implement Tasks 07-08 analysis during Tasks 03-05;
- do not implement Task 10 plots or Task 13 final workflow behavior early;
- do not create future missingness-pattern, broad multivariate, or candidate-insight systems.

Use the smallest complete vertical slice allowed by the current task.

## Required Tests

Write or update pytest in the same implementation task and only in the files allowed by `IMPLEMENTATION_PLAN.md`.

For Task 03, cover:

- mixed numeric, categorical, datetime, boolean, text-like, ID-like, and unknown columns;
- nullable and mixed dtypes;
- empty-row DataFrame, zero-column rejection, all-missing and constant columns;
- shape, memory, missing, unique, and quantile results against explicit pandas baselines;
- deterministic roles, reasons, fields, and ordering;
- input non-mutation and public contract checks.

For Task 04, cover:

- every fixed v0.1 rule independently;
- duplicates, all-missing, missing thresholds, constant and near-constant data;
- high cardinality, suspected IDs, infinity, mixed Python types, and datetime parse warnings;
- threshold boundaries, empty rows, duplicate column names, and no-issue data;
- stable issue codes, severity, evidence, suggestion, and no automatic repair.

For Tasks 07-08, cover:

- approved numeric and categorical statistics;
- top-N and all configured budgets;
- Pearson and Spearman against hand calculations or SciPy with explicit tolerance;
- IQR and MAD outlier behavior;
- missing, constant, invalid-column, and small-sample cases;
- single-key group comparison, missing groups, group truncation, and multiple-key rejection;
- separate classification and regression target contracts;
- structured skipped reasons, effective sample sizes, deterministic ordering, and input non-mutation.

For Tasks 05 and 13, test that workflow composes approved results, aggregates warnings and budgets, remains deterministic, and contains no duplicate domain algorithm.

Run the commands configured by the repository and report any command not run. Do not claim success from coverage percentage alone.

## Review Mode

Lead with correctness and contract findings. Cite exact paths and lines. Flag:

- API or result fields not present in SPEC;
- work pulled forward from a later task;
- ad hoc dictionaries or unstable DataFrame schemas;
- describe-only implementations;
- hidden report or workflow computation;
- wrong denominators, missing sample context, unstable ordering, or input mutation;
- weak tests that do not establish the locked contract.

Do not modify code during a review-only request.

## Completion Output

After implementation, report:

1. files changed;
2. approved public API implemented or changed;
3. exact approved result types and schemas used;
4. tests added and the contracts they prove;
5. commands run and results;
6. SPEC, AGENTS, README, and implementation-task compliance;
7. later-task or out-of-scope work deliberately deferred.

