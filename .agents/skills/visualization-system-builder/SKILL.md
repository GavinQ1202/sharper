---
name: visualization-system-builder
description: "Implement or review only Sharper's approved report-ready seaborn-first visualization APIs with matplotlib Figure contracts in visualization.py: Task 10 analytical plots, Task 11 classification evaluation plots, Task 12 regression evaluation plots, and Task 13 reporting integration. Use only plot_distributions, plot_missingness, plot_correlations, plot_outliers, plot_group_comparison, plot_target_relationships, plot_classification_evaluation, and plot_regression_evaluation with PlotResult and PlotCollection. Do not use for Tasks 01-09, dashboards, interactive charts, extra backends, or visualization APIs absent from SPEC.md."
---

# Visualization System Builder

Implement the locked v0.1 task-oriented figures without creating a plotting framework or dashboard. Treat `SPEC.md` as authoritative, `AGENTS.md` as mandatory visualization policy, and `IMPLEMENTATION_PLAN.md` as the task order and file boundary.

## Gate Before Work

1. Read `SPEC.md`, `AGENTS.md`, `README.md`, and the relevant Task 10-13 section of `IMPLEMENTATION_PLAN.md`.
2. Confirm that the plot's producing analysis or evaluation result and all prerequisite tasks are complete.
3. Inspect `visualization.py`, approved result inputs, `reporting.py`, workflow integration, exports, tests, and API documentation relevant to the current task.
4. If a requested plot, module, result field, style system, export behavior, or workflow is absent from SPEC, propose a SPEC update and stop. Do not implement it.
5. Do not use this skill for Tasks 01-09 or to implement analysis, feature engineering, model training, evaluation calculations, packaging, or CLI behavior.

## Approved v0.1 Public API

Use exactly:

```python
plot_distributions(
    df: pd.DataFrame,
    *,
    max_plots: int = 20,
    sample_size: int = 10_000,
) -> PlotCollection

plot_missingness(
    df: pd.DataFrame,
    *,
    max_columns: int = 50,
) -> PlotResult

plot_correlations(result: CorrelationAnalysis) -> PlotResult

plot_outliers(
    result: OutlierAnalysis,
    *,
    max_plots: int = 20,
) -> PlotCollection

plot_group_comparison(result: GroupComparison) -> PlotCollection

plot_target_relationships(result: TargetAnalysis) -> PlotCollection

plot_classification_evaluation(
    result: ClassificationEvaluation,
) -> PlotCollection

plot_regression_evaluation(
    result: RegressionEvaluation,
) -> PlotCollection
```

Do not add separate public numeric, categorical, heatmap, confusion-matrix, diagnostics, feature-importance, save, or generic chart APIs. Those behaviors must remain inside the approved task-level functions and result contracts where SPEC assigns them.

## Approved Result Contract

Use the frozen:

- `PlotResult`: named matplotlib `Figure`, analytical task, sampling or truncation metadata, and skipped reason as defined by SPEC and the task that freezes its fields;
- `PlotCollection`: composition of approved plot results only.

The public return is `PlotResult` or `PlotCollection`, not a bare `Axes`, arbitrary `dict`, or new renderer type. The contained object is a matplotlib `Figure`. Do not add result fields without first updating SPEC and the public contract.

Every plot function must return its approved result even when a plot is not applicable; use the approved skipped reason rather than a misleading empty visualization.

## Approved Task Boundaries

### Task 10 — Analytical Visualizations

Implement:

- numeric and categorical distribution figures through `plot_distributions`;
- column missing-rate figure through `plot_missingness`;
- bounded correlation heatmap through `plot_correlations`;
- outlier figures consuming `OutlierAnalysis`;
- group figures consuming `GroupComparison`;
- classification or regression target-relationship figures consuming `TargetAnalysis`.

Prefer already computed analytical result objects. Do not hide a second analysis implementation inside plotting. Raw `DataFrame` input is allowed only where the locked signature requires it.

### Task 11 — Classification Evaluation Plot

Implement only `plot_classification_evaluation(ClassificationEvaluation) -> PlotCollection` after Task 11 evaluation results exist. Include only the classification plots approved by SPEC, such as confusion matrix and ROC when applicable. Do not fit, predict, select, or inspect models implicitly.

### Task 12 — Regression Evaluation Plot

Implement only `plot_regression_evaluation(RegressionEvaluation) -> PlotCollection` after Task 12 evaluation results exist. Include only the approved residual and predicted-versus-actual diagnostics. Keep it separate from classification.

### Task 13 — Reporting Integration

Allow `reporting.py` to save and embed figures already present in approved results. `reporting.py` owns output paths, filenames, formats, asset directories, overwrite behavior, and figure closing after export. `visualization.py` does not assemble reports or write files.

## Figure Behavior

- Prefer seaborn for statistical analytical charts. Use matplotlib as seaborn's
  underlying backend, the Figure/Axes object contract, and a low-level fallback
  when seaborn does not fit the approved chart.
- Create and return a matplotlib `Figure` inside `PlotResult`.
- Never call `plt.show()` in library code.
- Do not write files by default or add `output_path` to the approved plot signatures.
- Do not casually modify global matplotlib or seaborn style inside plot functions.
  Any approved style customization must be locally scoped and restored.
- Do not build a multi-visualization-backend abstraction.
- Do not close a figure before returning it; tests and `reporting.py` own cleanup.
- Do not mutate raw input or structured analysis results.
- Apply locked budgets: at most 20 plots per type, 20 displayed category levels, 50 correlation columns, and 10,000 sampled rows unless the approved API exposes a bounded override.
- Record requested and actual budgets, truncation, sampling, and reasons in the approved result metadata.
- Handle empty, all-missing, constant, high-cardinality, small-sample, and single-class inputs with an approved skipped reason or documented error.
- Keep ordering deterministic and labels interpretable.

Every public function must have the complete type hints and docstring required by AGENTS, including parameters, return type, exceptions, side effects, missing-value behavior, and minimal example.

## Analytical Purpose

Charts must answer the approved analytical task:

- distributions show supported numeric or categorical structure and disclose missing/sample context;
- missingness shows column rates with explicit denominators;
- correlation heatmaps visualize the supplied `CorrelationAnalysis` without recomputation or causal language;
- outlier plots visualize supplied methods and thresholds without declaring bad data;
- group plots preserve group sample counts and truncation context;
- target plots keep classification and regression behavior distinct;
- classification evaluation uses `ClassificationEvaluation`;
- regression diagnostics use `RegressionEvaluation`.

Do not wrap matplotlib primitives as public APIs, expose arbitrary plotting kwargs as a framework, or add decorative charts unrelated to the approved analysis.

## Dependencies and Deferred Scope

Prefer seaborn for statistical analytical charts. Use matplotlib as the
underlying backend, Figure/Axes contract, and low-level fallback. Both are
approved core runtime dependencies.

Do not add:

- plotly, altair, bokeh, or another visualization backend;
- an interactive dashboard, drag-and-drop BI, or Web UI;
- notebook widgets;
- a multi-backend abstraction;
- a large automatic layout or theme engine;
- maps or animation;
- feature-importance plots not present in the locked API;
- SHAP or model-explanation visualization.

Interactive Plotly static HTML is only a possible v0.3 extra, not v0.1 work. AutoML, feature store, MLflow, and dashboard systems remain non-goals. If a new visual capability is needed, recommend a SPEC update instead of implementing it.

## Module Boundaries

- `visualization.py` generates approved figures from raw data or existing result types.
- `reporting.py` saves assets and renders Markdown/HTML without recomputing analysis.
- `workflow.py` composes approved public APIs and carries plots, warnings, and budget metadata in `AnalysisRun`.
- analysis and evaluation modules compute statistics; visualization must not duplicate them.

Do not create `report.py`, renderer hierarchies, plotting registries, theme managers, or backend adapters.

## Required Tests

Write or update pytest in the same implementation task and only in files allowed by `IMPLEMENTATION_PLAN.md`.

For Task 10, cover:

- every approved function returning `PlotResult` or `PlotCollection` with contained `Figure` objects;
- numeric and categorical distribution behavior;
- missing values, all-missing, constant, empty, high-cardinality, and invalid inputs;
- correlation, outlier, group, and target result inputs;
- separate classification and regression target figures;
- exact plot, category, correlation-column, and sample budgets;
- deterministic labels, ordering, skipped reasons, and budget metadata;
- no hidden analysis recomputation and no input mutation;
- no `plt.show()` and no default filesystem write.

For Tasks 11-12, cover classification and regression evaluation figures separately, including unavailable ROC behavior, absent classes, non-finite regression values, labels, reference lines, and skipped reasons defined by SPEC.

For Task 13, test Markdown/HTML image assets, stable links, overwrite and I/O behavior, report-only saving, and figure cleanup after success or failure.

Use a headless matplotlib backend. Close every generated figure in test cleanup. Assert result type, contained figure type, axes, labels, plotted data, metadata, and artifacts; avoid brittle pixel snapshots.

## Review Mode

Treat as blockers:

- a plot API or result field absent from SPEC;
- a bare `Axes`, arbitrary dictionary, or unapproved result type;
- `plt.show()` or default writes;
- hidden analysis or evaluation recomputation;
- unreported sampling or truncation;
- a dashboard, extra backend, or visualization framework;
- global matplotlib or seaborn style mutation;
- Task 11-13 work pulled into Task 10;
- figures that leak resources or cannot be consumed by `reporting.py`.

Lead with findings, cite exact paths and lines, explain the consequence, and recommend the smallest contract-preserving correction. Do not modify code during review-only work.

## Completion Output

After implementation, report:

1. files changed;
2. approved public APIs implemented or changed;
3. exact `PlotResult` and `PlotCollection` contracts used;
4. the approved analytical task served by each figure;
5. tests added and commands run;
6. `reporting.py` compatibility and figure-cleanup behavior;
7. later-task and advanced visualization capabilities deferred;
8. SPEC, AGENTS, README, and current implementation-task compliance.
