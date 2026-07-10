# Public API

## CSV input

```python
from sharper import load_csv

frame = load_csv("data.csv", encoding="utf-8", sep=",")
```

`load_csv(path, **read_options)` reads a local CSV into a
`pandas.DataFrame`. `path` accepts `str` and `pathlib.Path`. Task 02 supports
the pandas options `encoding`, `sep`, and `dtype`; other options raise
`ValueError` rather than becoming an accidental stable API.

The loader does not clean column names or values, infer a schema, summarize
data, or otherwise modify pandas' parsed result. Missing values use pandas
defaults. File-system failures raise `OSError`; empty or malformed input and
invalid parameters raise `ValueError`. Wrapped failures retain their original
exception as `__cause__`.

## Excel input

```python
from sharper import load_excel

frame = load_excel("data.xlsx", sheet_name=0, usecols=["name", "score"])
```

`load_excel(path, *, sheet_name=0, **read_options)` reads one sheet from a
local `.xlsx` file into a `pandas.DataFrame`. Install the optional Excel extra
before using it:

```bash
pip install "sharper[excel]"
```

For editable source installs:

```bash
python -m pip install -e ".[excel]"
```

Task 06 supports `str` and `pathlib.Path` paths, and `sheet_name` as a sheet
name or zero-based integer index. It rejects `None`, collections, and any
multi-sheet mode so the return value is always one DataFrame. Supported read
options are `header`, `names`, `usecols`, `dtype`, `na_values`,
`keep_default_na`, `skiprows`, and `nrows`. The engine is fixed to `openpyxl`
and cannot be overridden.

The loader preserves pandas' parsed column names and values, does not clean or
coerce data, and does not run schema inference, summary, quality checks,
workflow, reporting, visualization, or modeling. Task 06 does not add Excel
CLI support; `sharper analyze input.xlsx` is future workflow/CLI work.

Invalid path types, non-`.xlsx` suffixes, invalid sheet names, unsupported
read options, missing sheets, and pandas parser failures raise `ValueError`
with the stable Task 06 messages. Missing files and directory paths raise
`OSError`. Missing `openpyxl` raises `ImportError` with installation guidance.

## Schema inference

```python
from sharper import infer_schema

schema = infer_schema(frame, target="outcome")
```

`infer_schema(df, *, target=None, id_threshold=0.98)` returns a
`SchemaReport` containing ordered `ColumnSchema` results, counts for all seven
logical types, and non-binding `TargetCandidate` suggestions. Logical types are
`numeric`, `categorical`, `datetime`, `boolean`, `text`, `identifier`, and
`unknown`. Confidence values and reason codes follow the frozen
[Task 03 contract](decisions/task03-schema-summary-contract.md).

Task 03 requires unique string column names. It does not convert values or
column names, confirm a target, or run target-aware analysis. Empty DataFrames
are valid. An absent explicit target, duplicated names, non-string names, or an
invalid ID threshold raises `ValueError`.

## DataFrame summary

```python
from sharper import summarize_dataframe

summary = summarize_dataframe(frame, schema=schema)
```

`summarize_dataframe(df, *, schema=None)` returns `DataFrameSummary`: shape,
deep memory usage including the index, total missingness, the resolved schema,
and a column-level `pandas.DataFrame`. The detail table always has the frozen
17-column order and dtypes, including for 0-row and 0-column inputs. Only
logical numeric columns receive min, max, mean, sample standard deviation, and
quartiles. The function does not mutate the input.

## Data quality

```python
from sharper import check_data_quality

quality = check_data_quality(frame, schema=schema, missing_threshold=0.40)
```

`check_data_quality(df, *, schema=None, missing_threshold=0.40)` returns a
`QualityReport` containing deterministic `QualityIssue` entries. Task 04 checks
only empty input, duplicate rows, all/high missing columns, constant and
near-constant columns, high-cardinality categorical columns, identifier-like
columns, numeric infinite values, mixed Python object types, and partial
datetime parse failures.

The missing threshold must be greater than zero and at most one. Issues use only
the frozen `info`, `warning`, and `error` severities and preserve stable
table/code/original-column ordering. The function reports suggestions but does
not clean, convert, remove, or otherwise mutate input data. `QualityReport`
does not embed a `SchemaReport`, `DataFrameSummary`, timestamp, path, or random
identifier. Full fields, thresholds, evidence semantics, messages, and
mutual-exclusion rules follow the frozen
[Task 04 contract](decisions/task04-quality-contract.md).

## Non-target feature analysis

Task 07 adds four independent analysis functions. They require a pandas
DataFrame with unique string column names, preserve explicit column order, and
never mutate the input. With `columns=None`, each function selects applicable
columns by pandas dtype in original DataFrame order; it does not call schema
inference, summary, or quality APIs.

```python
from sharper import (
    analyze_categorical_features,
    analyze_numeric_features,
    compute_correlations,
    detect_outliers,
)

numeric = analyze_numeric_features(frame)
categorical = analyze_categorical_features(frame, top_n=10)
correlations = compute_correlations(
    frame,
    method="pearson",
    max_columns=50,
    min_periods=2,
)
outliers = detect_outliers(frame, method="iqr", threshold=1.5)
```

`NumericAnalysis` has fields `n_rows`, `requested_columns`,
`analyzed_columns`, `skipped_columns`, `skipped_reasons`, and `summary`.
Its summary columns are, in order: `column`, `count`, `missing_count`,
`missing_rate`, `mean`, `std`, `min`, `q25`, `median`, `q75`, `max`, `skew`,
`zero_count`, and `zero_rate`. Numeric selection excludes boolean dtypes.
Statistics use pandas sample standard deviation, quantile interpolation, and
skew defaults. Infinity is retained; missing values are excluded, and
`zero_rate` uses the non-missing count as its denominator.

`CategoricalAnalysis` has fields `n_rows`, `requested_columns`,
`analyzed_columns`, `skipped_columns`, `skipped_reasons`, `top_n`, `summary`,
and `top_categories`. It selects object, string, category, and boolean dtypes.
The summary columns are `column`, `count`, `missing_count`, `missing_rate`,
`unique_count`, `unique_rate`, `top`, `top_count`, and `top_rate`.
`top_categories` uses `column`, `category`, `count`, `rate`, and `rank`.
Missing values do not participate in category frequencies. Frequencies sort by
descending count, with ties resolved by first appearance in the original
column; `top_n` is a per-column display budget.

`CorrelationAnalysis` has fields `n_rows`, `requested_columns`,
`analyzed_columns`, `skipped_columns`, `skipped_reasons`, `method`,
`max_columns`, `min_periods`, `truncated`, and `correlations`. Its long-form
table columns are `column_a`, `column_b`, `method`, `correlation`, and
`n_pairs`. Pearson and Spearman are supported. The function first applies dtype,
all-missing, effective-sample, and constant checks, then retains the first
`max_columns` eligible columns. Excess eligible columns receive
`exceeds_max_columns`, and `truncated` becomes true. Each unordered pair appears
once in analyzed-column order; pairs below `min_periods` or with a pandas `NaN`
coefficient are omitted. There are no diagonal rows, p-values, or heatmaps.

`OutlierAnalysis` has fields `n_rows`, `requested_columns`,
`analyzed_columns`, `skipped_columns`, `skipped_reasons`, `method`, `threshold`,
`summary`, and `outliers`. Its summary columns are `column`, `method`,
`threshold`, `lower_bound`, `upper_bound`, `outlier_count`, and `outlier_rate`.
Detail columns are `column`, `row_index`, `value`, `lower_bound`, and
`upper_bound`. Task 07 supports IQR only: bounds use pandas' default quartiles,
and strict comparisons identify outliers. Missing values are excluded; a column
containing either infinity is skipped. Detail rows preserve original index
labels and DataFrame row order. Values are reported but never removed or
cleaned.

All four results are frozen dataclasses. The only skipped reason codes are
`not_numeric`, `not_categorical`, `all_missing`, `constant`,
`insufficient_non_missing`, `non_finite_values`, and `exceeds_max_columns`, as
applicable. Precedence is:

- numeric: `not_numeric` then `all_missing`;
- categorical: `not_categorical` then `all_missing`;
- correlation: `not_numeric`, `all_missing`, `insufficient_non_missing`,
  `constant`, then `exceeds_max_columns`;
- outliers: `not_numeric`, `all_missing`, `non_finite_values`,
  `insufficient_non_missing`, then `constant`.

Full field types, fixed table dtypes, validation messages, missing-value
semantics, and ordering rules follow the frozen
[Task 07 contract](decisions/task07-analysis-contract.md). Task 07 does not add
target or group analysis, feature engineering, visualization, modeling,
evaluation, report generation, workflow integration, CLI integration, data
cleaning, or custom exceptions.

## Group and target relationship analysis

Task 08 adds two independent Python APIs without changing workflow, reporting,
CLI, or I/O:

```python
from sharper import analyze_target_relationships, compare_groups

groups = compare_groups(frame, "segment", values=["revenue"], max_groups=20)
target = analyze_target_relationships(
    frame,
    "outcome",
    task="classification",
)
```

`compare_groups(df, group_by, *, values=None, max_groups=20)` accepts one
categorical group key and real numeric non-boolean values. Complex columns are
not selected automatically and are rejected when explicitly requested. Missing group keys are
excluded and disclosed. Groups rank by descending row frequency with
first-appearance tie breaking; the first `max_groups` are retained without an
Other bucket. `GroupComparison` records requested, analyzed, and skipped values,
group counts, missing-group count, truncation metadata, and a fixed long-form
`summary` table with `value`, `group`, `group_count`, `count`, `missing_count`,
`mean`, `q25`, `median`, and `q75`.

`analyze_target_relationships(df, target, *, task, features=None)` requires an
explicit classification or regression task. Classification uses Kruskal-Wallis
for numeric features and Chi-square with Cramer's V for categorical features.
Regression uses Pearson correlation for real numeric features and Kruskal-Wallis
for categorical features. Complex features are skipped as unsupported, and a
complex regression target is rejected before dispatch. Kruskal paths retain only groups with at least two complete
observations. The fixed budgets are 50 eligible features, 20 complete-case
categories per categorical feature, and 20 classification target classes.

`TargetAnalysis.numeric_details` uses `feature`, `target_category`,
`group_count`, `count`, `missing_count`, `mean`, `q25`, `median`, and `q75`.
`category_details` uses only `feature`, `feature_category`, `target_category`,
`count`, `rate`, `target_mean`, and `target_median`. `statistical_tests` uses
`feature`, `feature_kind`, `analysis`, `n_obs`, `group_count`, `statistic`,
`p_value`, `effect_size`, `effect_size_name`, and `limitation`. Empty tables keep
the same columns and dtypes.

All results are deterministic frozen dataclasses. Missing values are handled by
the path-specific complete-case rules, numeric infinity skips a feature, and
inapplicable tests emit a structured skipped reason rather than a NaN result row.
P-values are exploratory and unadjusted; no significance label, ranking, causal
claim, model, plot, cleaning, or post-hoc test is produced. Complete field types,
schemas, budgets, skipped-reason precedence, stable errors, and limitation codes
follow the frozen
[Task 08 contract](decisions/task08-group-target-analysis-contract.md).
