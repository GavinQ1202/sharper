# sharper

Sharper is a Python toolkit for structured tabular-data analysis. It combines
data loading, schema inference, summaries, quality diagnostics, relationship
analysis, bounded feature suggestions, static visualizations, optional baseline
models, and deterministic reports behind typed result objects.

The current source version is **0.2.0**. It adds an opt-in integration layer
for score validation, pre-loan decision-strategy simulation, post-loan warning
and lifecycle monitoring, optional data audit, optional offline model
governance, closed JSON carriers, static Markdown/HTML reports, and the
`v02-run` CLI path. The existing analysis workflow remains available and
compatible.

Sharper is intended for analysts, data scientists, researchers, and risk teams
who need inspectable evidence from structured tables. It is not an AutoML
system, a deployment platform, an MLOps product, or a generic configuration
DSL.

## Contents

- [What sharper does](#what-sharper-does)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Core workflow](#core-workflow)
- [Public API overview](#public-api-overview)
- [v0.2 integrated workflow](#v02-integrated-workflow)
- [v0.2 JSON configuration](#v02-json-configuration)
- [Reporting](#reporting)
- [Command-line interface](#command-line-interface)
- [Practical examples](#practical-examples)
- [Complete root API catalog](#complete-root-api-catalog)
- [Compatibility, reproducibility, and boundaries](#compatibility-reproducibility-and-boundaries)
- [Development and status](#development-and-status)
- [License](#license)

## What sharper does

Sharper works with one local tabular DataFrame at a time. Its main capabilities
are:

- CSV and one-sheet XLSX loading;
- schema inference for numeric, categorical, datetime, boolean, text,
  identifier, and unknown columns, including non-binding target candidates;
- shape, memory, missingness, and column-level summaries;
- deterministic data-quality findings for duplicate rows, missingness,
  constants, cardinality, identifiers, infinity, mixed object types, and
  datetime parsing;
- numeric and categorical summaries, correlations, IQR outlier diagnostics,
  group comparison, and exploratory target relationships;
- bounded feature suggestions and safe stateless derivation of supported
  arithmetic and timezone-naive datetime features;
- optional classification and regression baselines using split-first
  scikit-learn pipelines and independent holdout evaluation;
- report-ready seaborn/matplotlib Figures and static Markdown/HTML reports;
- opt-in binary risk validation, data audit, decision-strategy simulation,
  lifecycle monitoring, and offline governance analytics;
- an integrated v0.2 workflow that carries those opt-in results into one
  typed result and one static report.

The outputs are structured dataclasses and typed pandas tables. Truncation,
skipping, warnings, limitations, effective sample sizes, and input-boundary
decisions are exposed in result metadata rather than silently hidden.

## Installation

The current source is release-ready but has **not** been formally published.
Install from a clone or another local source checkout:

```bash
git clone https://github.com/GavinQ1202/sharper.git
cd sharper
python -m pip install -e .
```

Sharper supports Python `>=3.10`; the package metadata declares classifiers for
Python 3.10, 3.11, 3.12, and 3.13. The build backend is Hatchling. Runtime
dependencies are pandas, NumPy, SciPy, scikit-learn, matplotlib, seaborn, and
Typer.

For local `.xlsx` input, install the optional Excel dependency:

```bash
python -m pip install -e ".[excel]"
```

The `excel` extra supplies `openpyxl`. Development tools are available through
the `dev` extra:

```bash
python -m pip install -e ".[dev]"
```

## Quick start

This is a complete base-workflow example. `outcome` is an example of a target
column that must be confirmed by the caller; sharper does not silently turn a
target candidate into a target-aware analysis.

```python
from sharper import (
    check_data_quality,
    generate_analysis_report,
    infer_schema,
    load_csv,
    run_analysis,
)

frame = load_csv("data/input.csv")
schema = infer_schema(frame, target="outcome")
quality = check_data_quality(frame, schema=schema)
run = run_analysis(frame, target="outcome", task="classification")
artifact = generate_analysis_report(run, "reports/analysis.md")

print(schema.logical_type_counts)
print(quality.issue_count)
print(artifact.path)
```

The report writer creates the requested Markdown or HTML file and a sibling
PNG asset directory. Figures are static; sharper does not start a server or
produce an interactive dashboard.

## Public API overview

The root package exports the stable functions and typed result/config classes
cataloged in [Complete root API catalog](#complete-root-api-catalog). The
sections below explain when to use each capability; exact fields and validation
contracts are in [docs/api.md](docs/api.md).

## Core workflow

`run_analysis` is the base orchestration entry point. It combines schema,
summary, quality, non-target analysis, group/target analysis when requested,
feature suggestions, visualizations, and optional classification or regression
baseline results into an `AnalysisRun`.

```python
from sharper import generate_analysis_report, run_analysis

run = run_analysis(
    frame,
    target="outcome",                 # optional; caller-confirmed
    task="classification",            # or "regression"
    include_model=False,               # opt in to a baseline explicitly
    id_columns=("customer_id",),
    exclude_columns=("future_value",),
    group_by="segment",
    reference_date="2025-01-01",
    max_suggestions=50,
    random_state=42,
)
artifact = generate_analysis_report(
    run,
    "reports/analysis.html",
    format="html",
)
```

The base workflow is deterministic for fixed inputs and `random_state`. It
does not modify the input DataFrame. Target-aware analysis and modeling require
an explicit target and task. `include_model=True` opts into a random holdout
baseline; it does not claim that a random split is safe for time-ordered or
entity-grouped data.

### Data loading and inspection

- `load_csv` accepts a path and supported read options, and returns one pandas DataFrame from a local
  CSV. The stable read options are `encoding`, `sep`, and `dtype`.
- `load_excel` accepts a path, `sheet_name`, and supported read options, and returns one DataFrame
  from one `.xlsx` sheet. It uses `openpyxl` and accepts the documented pandas
  options `header`, `names`, `usecols`, `dtype`, `na_values`,
  `keep_default_na`, `skiprows`, and `nrows`.
- `infer_schema` returns a `SchemaReport`; `summarize_dataframe` returns a
  `DataFrameSummary`; `check_data_quality` returns a `QualityReport`.

These functions inspect data without cleaning, renaming, or silently coercing
the caller's DataFrame. File failures use `OSError`; invalid input uses the
documented `ValueError` contracts.

### Analysis and diagnostics

The independent analysis APIs are useful when a full workflow is unnecessary:

```python
from sharper import (
    analyze_categorical_features,
    analyze_numeric_features,
    analyze_target_relationships,
    compare_groups,
    compute_correlations,
    detect_outliers,
)

numeric = analyze_numeric_features(frame)
categorical = analyze_categorical_features(frame, top_n=10)
correlations = compute_correlations(frame, method="spearman", max_columns=50)
outliers = detect_outliers(frame, method="iqr", threshold=1.5)
groups = compare_groups(frame, "segment", max_groups=20)
target = analyze_target_relationships(
    frame, "outcome", task="classification"
)
```

Numeric and categorical summaries use missing-value-aware pandas statistics.
Correlation supports Pearson and Spearman with a column budget. Outlier
detection currently uses IQR bounds. Group comparison accepts a categorical
group key and real numeric measures. Target relationships use explicit
classification or regression paths, exploratory tests, complete-case rules,
and fixed feature/category budgets; p-values are not causal evidence.

### Feature suggestions

The function `suggest_feature_derivations` returns bounded, deterministically
ordered `FeatureSuggestion` values with formulas, reasons, risk, priority, and
a `requires_fit` flag.

```python
from sharper import derive_features, suggest_feature_derivations

suggestions = suggest_feature_derivations(
    frame,
    schema=schema,
    target="outcome",
    exclude_columns=("customer_id",),
    reference_date="2025-01-01",
)
safe_suggestions = tuple(
    item for item in suggestions.suggestions if not item.requires_fit
)
materialized = derive_features(
    frame,
    safe_suggestions,
    copy=True,
)
```

Only ratio, difference, product, timezone-naive datetime components, and
explicit-reference-date days-since suggestions can be materialized. Learned or
fixed binning, group aggregates, and target encoding remain suggestions that
require fitting and are rejected by `derive_features`. `copy=True` returns a
deep pandas DataFrame copy; `copy=False` permits attaching derived columns to
the input only after validation and temporary computation succeed.

### Baseline modeling and evaluation

`train_classifier` and `train_regressor` are optional, split-first baseline
APIs. They fit preprocessing and estimators on the training partition only and
retain a holdout for evaluation. `evaluate_classifier`, `evaluate_regressor`,
and `evaluate_model` return independent evaluation results.

```python
from sharper import (
    evaluate_model,
    plot_classification_evaluation,
    train_classifier,
)

training = train_classifier(
    frame,
    "outcome",
    exclude_columns=("future_value",),
    random_state=42,
)
evaluation = evaluate_model(training)
plots = plot_classification_evaluation(evaluation)
```

Known target-derived, posterior, future, ID, or entity-risk columns must be
declared in `exclude_columns`. Datetime/timedelta inputs and a declared
`time_column` are not accepted by these random-holdout baseline APIs. The
returned warnings and limitations disclose that the split does not solve
time-order or entity/group leakage.

### Visualization

The visualization API returns `PlotResult` or `PlotCollection`, each containing
caller-owned matplotlib Figures. `plot_distributions` and `plot_missingness`
accept raw DataFrames. Result-only plots consume existing analysis or
evaluation results and do not recalculate statistics:

```python
from sharper import plot_correlations, plot_missingness

missingness = plot_missingness(frame)
correlation_figure = plot_correlations(correlations)
missingness.figure.savefig("reports/missingness.png")
correlation_figure.figure.savefig("reports/correlations.png")
```

Sharper does not call `show()`, save files automatically, close caller-owned
Figures, switch matplotlib backends, or establish global plotting styles.
Callers own Figure lifetime when using the plotting functions directly.

### Risk validation, audit, decision, lifecycle, and governance

These are independent, opt-in Python APIs. They are useful when the caller has
explicit metadata and wants evidence rather than an automated action:

- `validate_binary_risk` evaluates an estimator or external predictions with
  explicit label, score semantics, validation provenance, calibration/gains/
  threshold evidence, and maturity handling. Ranking scores are not silently
  treated as probabilities.
- `audit_data_quality` produces structured quality, missingness drift,
  leakage, schema, and point-in-time evidence without repairing data.
- `simulate_decision_strategy` executes a caller-frozen closed strategy against
  offline data and returns simulated actions, rule evidence, constraints, and
  business summaries. It does not approve, decline, contact, collect, or
  optimize a policy.
- `monitor_lifecycle` evaluates caller-frozen point-in-time warning scenarios,
  episodes, events, states, transitions, and lifecycle summaries. It does not
  send notifications or execute account or collection actions.
- `evaluate_governance` consumes frozen owner results and structured attribution,
  prediction-profile, and performance evidence. It does not inspect or execute
  models, automatically promote candidates, deploy, or certify fairness.

The corresponding result dataclasses preserve typed tables, warnings,
limitations, provenance, and resource/status information. See the [API
reference](docs/api.md) for exact fields and contracts.

## v0.2 integrated workflow

The v0.2 layer is an additive, opt-in integration surface. It does not replace
`run_analysis`, `generate_analysis_report`, or `sharper analyze`.

There are three independent primary paths:

1. **Score validation** — explicit Task 15 validation metadata and an estimator
   or external score source.
2. **Pre-loan** — a caller-frozen `DecisionStrategyConfig` carried by
   `V02PreLoanRequest`.
3. **Post-loan** — a caller-frozen `LifecycleMonitoringConfig` carried by
   `V02PostLoanRequest`.

`V02AuditRequest` attaches the optional data-audit diagnostic path.
`V02GovernanceRequest` is an optional final step that consumes compatible
upstream owner results and declared governance evidence. At least one primary
path is required; audit alone or governance alone is not a primary workflow.
Enabled owners run once in the fixed integration order:

```text
audit -> score validation -> pre-loan -> post-loan -> governance
```

### Python API

The exact nine root exports for the v0.2 integration are:

```text
V02ScoreValidationRequest
V02AuditRequest
V02PreLoanRequest
V02PostLoanRequest
V02GovernanceRequest
V02WorkflowRequest
V02WorkflowResult
run_v02_workflow
generate_v02_report
```

The smallest valid Python shape is a score-validation workflow. The score
source and all important semantics are caller-supplied; sharper does not infer
the target, positive label, score direction, or score kind.

```python
import numpy as np
import pandas as pd

from sharper import (
    BinaryRiskValidationConfig,
    ExternalRiskPredictions,
    V02ScoreValidationRequest,
    V02WorkflowRequest,
    run_v02_workflow,
)

frame = pd.DataFrame({"feature": range(12), "outcome": [0, 1] * 6})
scores = tuple(float(value) for value in np.linspace(0.05, 0.95, 12))
external = ExternalRiskPredictions(
    row_positions=tuple(range(12)),
    fold_ids=(0, 1, 0, 0, 2, 0, 1, 1, 2, 2, 1, 2),
    fold_fit_row_positions=(
        (0, (1, 4, 6, 7, 8, 9, 10, 11)),
        (1, (0, 2, 3, 4, 5, 8, 9, 11)),
        (2, (0, 1, 2, 3, 5, 6, 7, 10)),
    ),
    ranking_scores=scores,
    ranking_direction="higher_risk",
    event_probabilities=None,
    probability_positive_label=None,
    probability_provenance=None,
)
request = V02WorkflowRequest(
    data=frame,
    score_validation=V02ScoreValidationRequest(
        target="outcome",
        config=BinaryRiskValidationConfig(
            validation_mode="stratified_kfold",
            n_splits=3,
            thresholds=(0.5,),
            threshold_kind="ranking_score",
        ),
        positive_label=1,
        external_predictions=external,
    ),
)
result = run_v02_workflow(request)
print(result.enabled_paths)
print(result.path_status)
```

For a full, deterministic synthetic score-validation example with three folds,
see [`examples/v02_score_validation.py`](examples/v02_score_validation.py).

### Pre-loan and post-loan paths

The pre-loan path carries an existing `DecisionStrategyConfig`; it does not
reimplement Task 17 logic:

```python
from sharper import V02PreLoanRequest, V02WorkflowRequest, run_v02_workflow

result = run_v02_workflow(
    V02WorkflowRequest(
        data=frame,
        preloan=V02PreLoanRequest(config=preloan_config),
    )
)
```

The post-loan path carries an existing `LifecycleMonitoringConfig`:

```python
from sharper import V02PostLoanRequest, V02WorkflowRequest, run_v02_workflow

result = run_v02_workflow(
    V02WorkflowRequest(
        data=observations,
        postloan=V02PostLoanRequest(config=monitoring_config),
    )
)
```

The complete synthetic constructors are in
[`examples/v02_preloan.py`](examples/v02_preloan.py) and
[`examples/v02_postloan.py`](examples/v02_postloan.py). A combined pre-loan +
post-loan report is in
[`examples/v02_combined_report.py`](examples/v02_combined_report.py).

`V02WorkflowResult` contains the contract version, enabled paths, path-status
table, call trace, typed owner results, warnings, and limitations. It does not
retain the primary or audit reference raw DataFrames. This is an evidence
orchestration boundary, not a claim that every upstream result is raw-data-free.

## v0.2 JSON configuration

The v0.2 CLI accepts exactly these versioned, closed, pure-data JSON schemas:

```text
task20.policy.v1
task20.warning.v1
```

`task20.policy.v1` maps to `DecisionStrategyConfig`; `task20.warning.v1` maps
to `LifecycleMonitoringConfig`. The JSON adapters are intentionally not a
generic configuration language. They reject YAML/TOML, Python configuration,
callables, expressions, scripts, templates, `include`, URLs, `$ref`,
environment expansion, path expansion, comments, duplicate keys, unknown
fields/operators, non-finite numbers, and unsupported schema versions.

This is a minimal valid policy carrier, matching the shape used by the CLI
example:

```json
{
  "schema_version": "task20.policy.v1",
  "strategy_key": "example-cli-policy",
  "strategy_version": "v1",
  "effective_from": "2025-01-01T00:00:00.000000",
  "expires_at": null,
  "evaluation_time": "2025-01-02T00:00:00.000000",
  "rules": [],
  "default_action_name": "review",
  "unknown_action_name": "review",
  "action_role_mapping": [["review", "review"]]
}
```

The complete end-to-end JSON example creates its own temporary input and runs
the CLI in [`examples/v02_cli_json.py`](examples/v02_cli_json.py). The Python
API remains the correct choice when the caller already has typed Task 15--19
configuration objects.

## Reporting

### Base analysis reports

`generate_analysis_report(run, output_path, *, title=..., format="markdown",
overwrite=True)` returns a `ReportArtifact`. It supports `markdown` and `html`,
writes a static report, and writes PNG Figures to a sibling directory named
`<report-stem>_assets`. Reporting consumes the stored `AnalysisRun` results and
does not rerun analysis, fit models, or recompute metrics.

### v0.2 integration reports

`generate_v02_report(result, output_path, *, title=..., format="markdown",
overwrite=True)` returns the same `ReportArtifact` type. It supports exactly
Markdown and HTML and writes a sibling PNG asset directory. The report covers
run context, path status, score validation, audit/leakage, pre-loan, post-loan,
governance, cross-path comparison, reason/override trace, stability/business
evidence, warnings/limitations, and provenance/release readiness.

```python
from sharper import generate_v02_report, run_v02_workflow

result = run_v02_workflow(request)
artifact = generate_v02_report(
    result,
    "reports/v02-run.html",
    format="html",
)
print(artifact.path)
```

The v0.2 report consumes stored owner result tables and caller-owned Figures.
It does not split, fit, predict, evaluate, execute rules, reconstruct alerts,
or recalculate drift, metrics, or importance. At most nine deterministic PNG
plot slots are acquired when valid evidence exists.

## Command-line interface

The installed console command and module form are equivalent:

```bash
sharper --version
sharper analyze data/input.csv --output reports/analysis.md
# equivalent during a source checkout:
.venv/bin/python -m sharper.cli analyze data/input.csv --output reports/analysis.md
```

### `analyze`

`analyze INPUT --output OUTPUT` accepts local `.csv` and single-sheet `.xlsx`
input. The main options are:

```text
--output, -o PATH                 required
--format TEXT                     markdown (default) or html
--target TEXT
--task TEXT                       classification or regression
--id-column TEXT                  repeatable
--exclude-column TEXT            repeatable
--feature TEXT                   repeatable
--time-column TEXT
--group-by TEXT
--reference-date TEXT
--max-suggestions INTEGER        default 50
--model / --no-model             default no-model
--test-size FLOAT                default 0.20
--random-state INTEGER           default 42
--overwrite / --no-overwrite     default overwrite
--debug / --no-debug             default no-debug
```

Without `--target` and `--task`, the command still produces the base analysis
but does not run target-aware analysis or modeling. `--model` requires a valid
explicit target/task pair.

Portable examples:

```bash
sharper analyze data/input.csv \\
  --target outcome \\
  --task classification \\
  --model \\
  --output reports/analysis.html \\
  --format html

sharper analyze data/input.xlsx --output reports/excel-analysis.md
```

### `v02-run`

`v02-run INPUT --output OUTPUT` is the closed, opt-in v0.2 command. It accepts
the following user-facing options:

```text
--output, -o PATH
--format TEXT                     markdown (default) or html
--policy-json PATH
--warning-json PATH
--audit / --no-audit              default no-audit
--reference-input PATH
--target TEXT
--external-ranking-score-column TEXT
--external-event-probability-column TEXT
--ranking-direction TEXT
--probability-provenance TEXT
--score-validation-mask-column TEXT
--positive-label-type TEXT        repeatable: str, int, or bool
--positive-label-value TEXT       repeatable: one value
--score-test-size FLOAT           default 0.20
--score-random-state INTEGER      default 42
--overwrite / --no-overwrite      default overwrite
```

Examples:

```bash
sharper v02-run data/input.csv \\
  --output reports/policy-report.md \\
  --policy-json config/policy.json

sharper v02-run data/input.csv \\
  --output reports/warning-report.html \\
  --warning-json config/warning.json \\
  --format html

sharper v02-run data/input.csv \\
  --output reports/score-report.md \\
  --target outcome \\
  --external-ranking-score-column risk_score \\
  --ranking-direction higher_risk \\
  --score-validation-mask-column is_validation \\
  --positive-label-type int \\
  --positive-label-value 1
```

Score validation requires explicit target, exactly one external score column,
a boolean validation mask, positive-label type/value, and matching score
direction or probability provenance. `--policy-json` and `--warning-json`
select pre-loan and post-loan paths. `--audit` enables the optional audit path;
`--reference-input` is only valid with it. There is no governance CLI option,
title option, YAML carrier, or alternate configuration format.

CLI exit behavior is stable:

- `0`: success;
- `1`: base `analyze` input, filesystem, or report error;
- `2`: CLI usage/validation/closed-contract error;
- `3`: v0.2 filesystem I/O error;
- `70`: unexpected v0.2 internal error.

The v0.2 command reports contract errors with a `sharper task20: ...` message
and does not emit a default traceback.

## Practical examples

The repository contains synthetic, deterministic examples that use the public
API:

| Example | Demonstrates |
| --- | --- |
| [`examples/basic_analysis.py`](examples/basic_analysis.py) | Base analysis and a static report |
| [`examples/baseline_modeling.py`](examples/baseline_modeling.py) | Optional classification baseline and evaluation |
| [`examples/v02_score_validation.py`](examples/v02_score_validation.py) | Explicit external score validation metadata |
| [`examples/v02_preloan.py`](examples/v02_preloan.py) | Offline pre-loan strategy simulation |
| [`examples/v02_postloan.py`](examples/v02_postloan.py) | Offline post-loan lifecycle monitoring |
| [`examples/v02_combined_report.py`](examples/v02_combined_report.py) | Combined pre-loan/post-loan static report |
| [`examples/v02_cli_json.py`](examples/v02_cli_json.py) | Closed policy JSON through `v02-run` |

The v0.2 examples create synthetic inputs and produce offline evidence. They do
not execute approvals, account changes, notifications, collections, model
promotion, deployment, or customer contact.

## Complete root API catalog

The following catalog covers every symbol in the current root
`sharper.__all__` (96 symbols including `__version__`). The
module-level reference documents linked below contain exact annotations,
dataclass field order, validation rules, table schemas, and examples.

### Package and I/O

| Symbol | Kind | Purpose / primary result |
| --- | --- | --- |
| `__version__` | version | Current package version, `0.2.0`. |
| `load_csv` | function | Read a local CSV into a pandas DataFrame. |
| `load_excel` | function | Read one local `.xlsx` sheet into a pandas DataFrame. |

### Schema, summary, quality, workflow, and reports

| Symbol | Kind | Purpose / primary result |
| --- | --- | --- |
| `ColumnSchema` | dataclass | Per-column dtype, logical role, missingness, uniqueness, and confidence evidence. |
| `TargetCandidate` | dataclass | Non-binding target/task suggestion from schema inference. |
| `SchemaReport` | dataclass | Ordered column schemas, logical-type counts, and target candidates. |
| `infer_schema` | function | Infer schema roles and return `SchemaReport`. |
| `DataFrameSummary` | dataclass | Shape, memory, missingness, schema, and column summary table. |
| `summarize_dataframe` | function | Build a `DataFrameSummary` without modifying input. |
| `QualityIssue` | dataclass | One coded quality finding with severity, evidence, message, and suggestion. |
| `QualityReport` | dataclass | Quality issue collection and severity counts. |
| `check_data_quality` | function | Run deterministic quality checks and return `QualityReport`. |
| `AnalysisRun` | dataclass | Complete base-workflow result, including analyses, plots, model/evaluation, warnings, and limitations. |
| `run_analysis` | function | Orchestrate the base analysis workflow and return `AnalysisRun`. |
| `ReportArtifact` | dataclass | Written report path, format, and title. |
| `generate_analysis_report` | function | Render an `AnalysisRun` as static Markdown or HTML plus PNG assets. |

Important signatures:

```python
infer_schema(df, *, target=None, id_threshold=0.98) -> SchemaReport
summarize_dataframe(df, *, schema=None) -> DataFrameSummary
check_data_quality(df, *, schema=None, missing_threshold=0.4) -> QualityReport
run_analysis(
    df, *, target=None, task=None, include_model=False,
    id_columns=(), exclude_columns=(), features=None, time_column=None,
    group_by=None, reference_date=None, max_suggestions=50,
    test_size=0.2, random_state=42,
) -> AnalysisRun
generate_analysis_report(
    run, output_path, *, title="Sharper Analysis Report",
    format="markdown", overwrite=True,
) -> ReportArtifact
```

### Analysis and feature engineering

| Symbol | Kind | Purpose / primary result |
| --- | --- | --- |
| `NumericAnalysis` | dataclass | Numeric summary table plus analyzed/skipped columns. |
| `CategoricalAnalysis` | dataclass | Categorical summary and top-category tables. |
| `CorrelationAnalysis` | dataclass | Bounded long-form pairwise correlation table and truncation metadata. |
| `OutlierAnalysis` | dataclass | IQR bounds, outlier summary, and detail table. |
| `analyze_numeric_features` | function | Summarize applicable non-boolean numeric columns. |
| `analyze_categorical_features` | function | Summarize categorical/boolean columns and top categories. |
| `compute_correlations` | function | Compute bounded Pearson or Spearman pairwise correlations. |
| `detect_outliers` | function | Detect IQR outliers without removing values. |
| `GroupComparison` | dataclass | Group selection, truncation, missing-group metadata, and comparison table. |
| `TargetAnalysis` | dataclass | Numeric/category target-relationship details, exploratory tests, and limitations. |
| `compare_groups` | function | Compare real numeric measures across selected categorical groups. |
| `analyze_target_relationships` | function | Run explicit classification or regression target analyses. |
| `FeatureSuggestion` | dataclass | One bounded candidate feature with formula, risk, reason, and fit requirement. |
| `FeatureSuggestionReport` | dataclass | Eligible/excluded/skipped columns, budgets, and ordered suggestions. |
| `FeatureDerivationResult` | dataclass | Derived DataFrame plus applied/skipped suggestion metadata. |
| `suggest_feature_derivations` | function | Suggest bounded, deterministic feature derivations; does not materialize them. |
| `derive_features` | function | Materialize only safe stateless suggestions under explicit copy semantics. |

Important signatures:

```python
analyze_numeric_features(df, *, columns=None) -> NumericAnalysis
analyze_categorical_features(df, *, columns=None, top_n=10) -> CategoricalAnalysis
compute_correlations(df, *, columns=None, method="pearson", max_columns=50,
                     min_periods=2) -> CorrelationAnalysis
detect_outliers(df, *, columns=None, method="iqr", threshold=1.5) -> OutlierAnalysis
compare_groups(df, group_by, *, values=None, max_groups=20) -> GroupComparison
analyze_target_relationships(df, target, *, task, features=None) -> TargetAnalysis
suggest_feature_derivations(df, *, schema=None, target=None,
                            exclude_columns=(), reference_date=None,
                            max_suggestions=50) -> FeatureSuggestionReport
derive_features(df, suggestions, *, copy=True) -> FeatureDerivationResult
```

### Modeling and evaluation

| Symbol | Kind | Purpose / primary result |
| --- | --- | --- |
| `TrainingResult` | dataclass | Fitted classification pipeline, split positions, holdout snapshot, and limits. |
| `train_classifier` | function | Fit a split-first classification baseline. |
| `RegressionTrainingResult` | dataclass | Fitted regression pipeline, split positions, holdout snapshot, and limits. |
| `train_regressor` | function | Fit a split-first regression baseline. |
| `ClassificationEvaluation` | dataclass | Holdout labels, predictions, metrics, confusion matrix, ROC data, and limits. |
| `evaluate_classifier` | function | Evaluate a `TrainingResult` once on its holdout. |
| `RegressionEvaluation` | dataclass | Holdout predictions, residuals, metrics, and limits. |
| `evaluate_regressor` | function | Evaluate a `RegressionTrainingResult` once on its holdout. |
| `evaluate_model` | function | Dispatch exactly once to the matching classifier/regressor evaluator. |

### Visualization

| Symbol | Kind | Purpose / primary result |
| --- | --- | --- |
| `PlotResult` | dataclass | One caller-owned matplotlib Figure with chart/source metadata. |
| `PlotCollection` | dataclass | Bounded collection of `PlotResult` values and truncation metadata. |
| `plot_distributions` | function | Plot bounded numeric/categorical distributions from a raw DataFrame. |
| `plot_missingness` | function | Plot column missingness from a raw DataFrame. |
| `plot_correlations` | function | Plot a stored `CorrelationAnalysis`. |
| `plot_outliers` | function | Plot a stored `OutlierAnalysis`. |
| `plot_group_comparison` | function | Plot a stored `GroupComparison`. |
| `plot_target_relationships` | function | Plot a stored `TargetAnalysis`. |
| `plot_classification_evaluation` | function | Plot stored classification evaluation results. |
| `plot_regression_evaluation` | function | Plot stored regression evaluation results. |

### Binary risk validation

| Symbol | Kind | Purpose / primary result |
| --- | --- | --- |
| `BinaryRiskValidationConfig` | config dataclass | Explicit validation mode, maturity, score/probability, threshold, and business settings. |
| `ExternalRiskPredictions` | result/input dataclass | Row-position, fold, fit-row, score, direction, and probability provenance. |
| `BinaryRiskValidationResult` | result dataclass | Fold, prediction, metric, gains, calibration, threshold, and business evidence tables. |
| `validate_binary_risk` | function | Validate one explicit estimator or external prediction source. |
| `plot_binary_risk_validation` | function | Plot stored gains, lift, calibration, or threshold evidence. |

### Data audit and leakage evidence

| Symbol | Kind | Purpose / primary result |
| --- | --- | --- |
| `DataAuditRoles` | config dataclass | Declares target, feature, time, partition, outcome, and leakage-related roles. |
| `ColumnAuditRule` | config dataclass | Closed per-column minimum, maximum, allowed-value, temporal, or monotonic rule. |
| `DataAuditConfig` | config dataclass | Budgets and thresholds for quality, drift, leakage, and point-in-time checks. |
| `DataAuditResult` | result dataclass | Typed quality, profile, missingness-drift, schema, collinearity, and timing evidence. |
| `audit_data_quality` | function | Produce offline audit evidence without repairing or modifying input data. |

### Pre-loan decision-strategy simulation

| Symbol | Kind | Purpose / primary result |
| --- | --- | --- |
| `StrategyCondition` | config dataclass | Closed pure-data condition tree for approved comparisons and boolean composition. |
| `DecisionRule` | config dataclass | Ordered eligibility or decision rule with a simulated action. |
| `DecisionConstraint` | config dataclass | Evidence-only metric constraint over frozen simulated actions. |
| `DecisionStrategyConfig` | config dataclass | Versioned strategy, rules, actions, mappings, timing, and optional evidence settings. |
| `DecisionStrategyResult` | result dataclass | Row decisions, rule/action summaries, business evidence, constraints, transitions, and provenance. |
| `simulate_decision_strategy` | function | Evaluate a caller-frozen offline strategy; never execute a real decision. |

### Post-loan lifecycle monitoring

| Symbol | Kind | Purpose / primary result |
| --- | --- | --- |
| `MonitoringCondition` | config dataclass | Closed condition tree for warning and lifecycle monitoring. |
| `EarlyWarningRule` | config dataclass | Priority, alert level, persistence, resolution, and cooldown rule. |
| `WarningScenario` | config dataclass | Named scenario containing warning rules. |
| `LifecycleState` | config dataclass | Ordered lifecycle state with condition, terminal flag, and description key. |
| `LifecycleMonitoringConfig` | config dataclass | Point-in-time observation, scenario, state, timing, and evidence settings. |
| `LifecycleMonitoringResult` | result dataclass | Observation, rule, notification, episode, event, state, transition, summary, and provenance tables. |
| `monitor_lifecycle` | function | Produce offline warning/lifecycle evidence without sending or executing actions. |

### Offline model governance

| Symbol | Kind | Purpose / primary result |
| --- | --- | --- |
| `GovernanceEvidenceRef` | dataclass | Typed locator for evidence in frozen owner result tables. |
| `GovernanceCandidate` | dataclass | Candidate identity, family, state, role, version, and evidence references. |
| `GovernanceCriterion` | dataclass | Candidate comparison criterion, direction, target, support, and promotion role. |
| `GovernanceExplanation` | dataclass | Structured explanation declaration for a candidate and feature/relation. |
| `GovernanceAttributionEvidence` | dataclass | Precomputed attribution value, method, support, uncertainty, and provenance. |
| `GovernancePredictionProfile` | dataclass | Declared prediction distribution/profile evidence for a snapshot. |
| `GovernancePerformanceEvidence` | dataclass | Declared time/scope performance evidence with aligned scores/probabilities. |
| `GovernanceMetadata` | dataclass | Ownership, materiality, assumptions, limitations, and remediation metadata. |
| `GovernancePolicy` | config dataclass | Versioned governance policy, candidates, criteria, evidence, and alignment settings. |
| `GovernanceResult` | result dataclass | Explanation, attribution, drift, stability, comparisons, recommendations, and provenance tables. |
| `evaluate_governance` | function | Evaluate structured offline governance evidence without inspecting or deploying models. |
| `plot_model_governance` | function | Plot stored governance importance, comparisons, drift, stability, or summary evidence. |

### v0.2 integration

| Symbol | Kind | Purpose / primary result |
| --- | --- | --- |
| `V02ScoreValidationRequest` | request dataclass | Carries explicit Task 15 score-validation config and estimator/external source. |
| `V02AuditRequest` | request dataclass | Carries optional audit reference, roles, and audit config. |
| `V02PreLoanRequest` | request dataclass | Carries a `DecisionStrategyConfig` for the pre-loan path. |
| `V02PostLoanRequest` | request dataclass | Carries a `LifecycleMonitoringConfig` for the post-loan path. |
| `V02GovernanceRequest` | request dataclass | Carries a `GovernancePolicy` and structured governance declarations. |
| `V02WorkflowRequest` | request dataclass | Carries the primary DataFrame and optional path requests. |
| `V02WorkflowResult` | result dataclass | Carries enabled paths, status/trace, typed owner results, warnings, and limitations. |
| `run_v02_workflow` | function | Run enabled v0.2 owners once in fixed order and return `V02WorkflowResult`. |
| `generate_v02_report` | function | Render a `V02WorkflowResult` as a static Markdown/HTML report. |

For exact contracts and full field lists, see the [API reference](docs/api.md),
the [analysis guide](docs/analysis-guide.md), and the [v0.2 integration
guide](docs/v02-integration-guide.md).

## Compatibility, reproducibility, and boundaries

- v0.2 is additive and opt-in. The base Python workflow, reports, CLI, and
  existing root exports remain available.
- Inputs are read rather than silently cleaned or rewritten. Quality,
  analysis, audit, and monitoring APIs expose missingness and limitations.
- Baseline modeling fits data-driven preprocessing only after the split and
  requires explicit exclusion of future, posterior, target-derived, ID, and
  entity-risk fields. Random holdout is not a time-aware or entity-aware
  guarantee.
- Score validation keeps ranking scores separate from event probabilities.
  Calibration and expected-loss arithmetic require a valid probability for an
  explicit positive class.
- v0.2 JSON is closed and versioned. It has no arbitrary Python execution,
  callables, dynamic operators, includes, or environment expansion.
- Reports are static artifacts. They do not imply causality, regulatory
  compliance, production safety, deployment readiness, realized revenue, or
  operational alert execution.
- Pre-loan and post-loan capabilities are offline simulations/evidence only;
  they do not perform approvals, account operations, customer contact,
  notifications, or collections.

See [leakage safeguards](docs/leakage.md) for the modeling and audit boundary.

## Development and status

This repository uses a uv-managed project environment at `.venv`. Before
running Python validation commands in the repository:

```bash
bash scripts/verify-uv-env.sh
.venv/bin/python -m pytest
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
.venv/bin/python -m build --no-isolation
```

Current source version: **0.2.0**.

Current implementation status: **Release Ready — Not Released**. The v0.2
implementation and validation are complete, but no package publication, GitHub
Release, tag, or PyPI release is implied by this repository state.

Additional maintainer evidence is available in
[release readiness](docs/release-readiness.md). That document is a readiness
checklist, not a release announcement.

## License

Sharper is licensed under the [MIT License](LICENSE).
