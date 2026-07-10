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
