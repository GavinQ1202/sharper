# Task 07 Non-Target Feature Analysis 公共契约

## 状态

已接受。本文是 Task 07 实现前的 API 决策记录。Task 07 的实现、测试和
API 文档必须遵守本文；修改本文冻结的 public API、结果字段、表格 schema、
跳过原因、错误行为、排序或范围前，必须先同步评审 `SPEC.md` 和
`IMPLEMENTATION_PLAN.md`。

## 范围

Task 名称冻结为：
**Task 07 — Non-target feature analysis**。

Task 07 只新增以下 public API：

```python
def analyze_numeric_features(
    df: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
) -> NumericAnalysis: ...

def analyze_categorical_features(
    df: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    top_n: int = 10,
) -> CategoricalAnalysis: ...

def compute_correlations(
    df: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    method: str = "pearson",
    max_columns: int = 50,
    min_periods: int = 2,
) -> CorrelationAnalysis: ...

def detect_outliers(
    df: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    method: str = "iqr",
    threshold: float = 1.5,
) -> OutlierAnalysis: ...
```

Task 07 只做：

1. numeric feature analysis；
2. categorical feature analysis；
3. numeric pairwise correlations；
4. numeric outlier detection。

Task 07 不做：

1. target relationship analysis；
2. grouped analysis；
3. feature engineering；
4. visualization；
5. modeling；
6. evaluation；
7. report generation；
8. workflow integration；
9. CLI integration；
10. automatic cleaning；
11. data mutation；
12. custom exceptions。

Task 07 不修改 Task 05 workflow、reporting 或 CLI。

## 共享输入规则

所有 Task 07 函数遵守以下规则：

- `df` 必须是 pandas DataFrame；否则抛出 `ValueError`，消息包含
  `df must be a pandas DataFrame`。
- DataFrame column names 必须全部是字符串；否则抛出 `ValueError`，消息
  包含 `DataFrame column names must all be strings`。
- 重复 DataFrame column names 不支持；否则抛出 `ValueError`，消息包含
  `duplicate DataFrame column names are not supported`。
- `columns=None` 表示基于 pandas dtype 自动选择适用列，不调用
  `infer_schema`。
- 不调用 `infer_schema`、`summarize_dataframe` 或 `check_data_quality`。
- 不修改输入 DataFrame。
- 如果提供 `columns`，每一项必须是字符串、不得重复、且必须存在于 df。
- 非字符串 column parameter 抛出 `ValueError`，消息包含
  `columns must contain only strings`。
- 重复 column parameter 抛出 `ValueError`，消息包含
  `duplicate column parameter`。
- 缺失 requested column 抛出 `ValueError`，消息包含 `column not found`。
- Explicit `columns` 保持调用者顺序。
- Auto-selected columns 保持原 DataFrame column order。
- 所有输出必须 deterministic。

## 结果类型

所有结果类型必须使用 `dataclass(frozen=True)`。

### `NumericAnalysis`

字段顺序和类型冻结为：

```python
n_rows: int
requested_columns: tuple[str, ...] | None
analyzed_columns: tuple[str, ...]
skipped_columns: tuple[str, ...]
skipped_reasons: dict[str, str]
summary: pd.DataFrame
```

### `CategoricalAnalysis`

字段顺序和类型冻结为：

```python
n_rows: int
requested_columns: tuple[str, ...] | None
analyzed_columns: tuple[str, ...]
skipped_columns: tuple[str, ...]
skipped_reasons: dict[str, str]
top_n: int
summary: pd.DataFrame
top_categories: pd.DataFrame
```

### `CorrelationAnalysis`

字段顺序和类型冻结为：

```python
n_rows: int
requested_columns: tuple[str, ...] | None
analyzed_columns: tuple[str, ...]
skipped_columns: tuple[str, ...]
skipped_reasons: dict[str, str]
method: str
max_columns: int
min_periods: int
truncated: bool
correlations: pd.DataFrame
```

### `OutlierAnalysis`

字段顺序和类型冻结为：

```python
n_rows: int
requested_columns: tuple[str, ...] | None
analyzed_columns: tuple[str, ...]
skipped_columns: tuple[str, ...]
skipped_reasons: dict[str, str]
method: str
threshold: float
summary: pd.DataFrame
outliers: pd.DataFrame
```

规则：

- `requested_columns` 在 `columns is None` 时为 `None`。
- 如果提供 `columns`，`requested_columns` 是保持调用者顺序的 tuple。
- `analyzed_columns` 是保持分析顺序的 tuple。
- `skipped_columns` 是保持 input/request order 的 tuple。
- `skipped_reasons` 将列名映射到一个稳定 reason code。
- 结果类型不得包含 timestamps、paths、random IDs、durations、plots、
  models 或 generated files。

## Skipped Reason Vocabulary

只允许以下 skipped reason codes：

- `not_numeric`
- `not_categorical`
- `all_missing`
- `constant`
- `insufficient_non_missing`
- `non_finite_values`
- `exceeds_max_columns`

各函数允许使用：

- Numeric analysis：`not_numeric`、`all_missing`。
- Categorical analysis：`not_categorical`、`all_missing`。
- Correlation analysis：`not_numeric`、`all_missing`、`constant`、
  `insufficient_non_missing`、`exceeds_max_columns`。
- Outlier analysis：`not_numeric`、`all_missing`、`constant`、
  `insufficient_non_missing`、`non_finite_values`。

不得创建额外 skipped reason codes。

## Skipped Reason Precedence

当同一列命中多个 skipped conditions 时，只能分配一个 skipped reason，并按
以下 precedence 选择。不得给同一列附加多个 reasons。

### Numeric Analysis

Precedence:

1. `not_numeric`
2. `all_missing`

Rules:

- Explicitly requested non-numeric columns 使用 `not_numeric` 跳过，即使该列
  全部值缺失。
- Numeric all-missing columns 使用 `all_missing` 跳过。

### Categorical Analysis

Precedence:

1. `not_categorical`
2. `all_missing`

Rules:

- Explicitly requested non-categorical columns 使用 `not_categorical` 跳过，
  即使该列全部值缺失。
- Categorical all-missing columns 使用 `all_missing` 跳过。

### Correlation Analysis

Precedence:

1. `not_numeric`
2. `all_missing`
3. `insufficient_non_missing`
4. `constant`
5. `exceeds_max_columns`

Rules:

- Non-numeric requested columns 使用 `not_numeric` 跳过。
- Numeric all-missing columns 使用 `all_missing` 跳过。
- Non-missing values 少于 `min_periods` 的 numeric columns 使用
  `insufficient_non_missing` 跳过。
- 至少有 `min_periods` 个 non-missing values 但 non-missing values 为
  constant 的 numeric columns 使用 `constant` 跳过。
- `exceeds_max_columns` 只在 eligibility 和以上 skip checks 之后应用。
- 如果一列已经因任何更早 reason 被跳过，不得再被跳过为
  `exceeds_max_columns`。

### Outlier Detection

Precedence:

1. `not_numeric`
2. `all_missing`
3. `non_finite_values`
4. `insufficient_non_missing`
5. `constant`

Rules:

- Non-numeric requested columns 使用 `not_numeric` 跳过。
- Numeric all-missing columns 使用 `all_missing` 跳过。
- 包含 positive 或 negative infinity 的 numeric columns 使用
  `non_finite_values` 跳过，除非已被跳过为 `all_missing`。
- 排除 missing values 并确认无 infinite values 后，finite non-missing
  values 少于 2 的 numeric columns 使用 `insufficient_non_missing` 跳过。
- 至少有 2 个 finite non-missing values 但 finite values 为 constant 的
  numeric columns 使用 `constant` 跳过。

## Numeric Feature Analysis

### Column Selection

Eligible columns:

- pandas numeric dtype；
- excluding boolean dtype。

Auto-selection 使用 DataFrame order 中所有 eligible numeric non-boolean
columns。Explicit columns 先验证名称，再分析 numeric non-boolean columns；
non-numeric columns 使用 `not_numeric` 跳过。全缺失 numeric columns 使用
`all_missing` 跳过。

### `NumericAnalysis.summary`

`summary` 必须是 pandas DataFrame，列顺序冻结为：

1. `column`
2. `count`
3. `missing_count`
4. `missing_rate`
5. `mean`
6. `std`
7. `min`
8. `q25`
9. `median`
10. `q75`
11. `max`
12. `skew`
13. `zero_count`
14. `zero_rate`

Dtypes:

- `column`: object
- `count`: int64
- `missing_count`: int64
- `missing_rate`: float64
- `mean`: float64
- `std`: float64
- `min`: float64
- `q25`: float64
- `median`: float64
- `q75`: float64
- `max`: float64
- `skew`: float64
- `zero_count`: int64
- `zero_rate`: float64

Rules:

- `count` 是 non-missing count，排除 NaN。
- `missing_count = n_rows - count`。
- `missing_rate = missing_count / n_rows`；`n_rows == 0` 时为 `0.0`。
- Quantiles 使用 pandas default quantile interpolation。
- `std` 使用 pandas default sample standard deviation。
- `skew` 使用 pandas `Series.skew` default behavior。
- `zero_count` 统计 non-missing finite/infinite values 中等于 0 的值。
- `zero_rate = zero_count / count`；`count == 0` 时为 `0.0`。
- 正负 infinity 在 numeric analysis 中允许，并遵循 pandas aggregation
  behavior。
- Summary rows 按 analyzed column order 排序。
- 无 analyzed columns 时返回 fixed columns and dtypes 的空 DataFrame。

## Categorical Feature Analysis

### Column Selection

Eligible columns:

- pandas object dtype；
- pandas string dtype；
- pandas category dtype；
- pandas boolean dtype。

Auto-selection 使用 DataFrame order 中所有 eligible categorical columns。
Explicit columns 先验证名称，再分析 eligible categorical columns；
ineligible columns 使用 `not_categorical` 跳过。全缺失 categorical columns
使用 `all_missing` 跳过。

`top_n` 必须是 int 且 `>= 1`；否则抛出 `ValueError`，消息包含
`top_n must be a positive integer`。

### `CategoricalAnalysis.summary`

`summary` 必须是 pandas DataFrame，列顺序冻结为：

1. `column`
2. `count`
3. `missing_count`
4. `missing_rate`
5. `unique_count`
6. `unique_rate`
7. `top`
8. `top_count`
9. `top_rate`

Dtypes:

- `column`: object
- `count`: int64
- `missing_count`: int64
- `missing_rate`: float64
- `unique_count`: int64
- `unique_rate`: float64
- `top`: object
- `top_count`: int64
- `top_rate`: float64

Rules:

- `count` 是 non-missing count。
- `missing_count = n_rows - count`。
- `missing_rate = missing_count / n_rows`；`n_rows == 0` 时为 `0.0`。
- `unique_count` 是 non-missing unique count。
- `unique_rate = unique_count / count`；`count == 0` 时为 `0.0`。
- `top` 是最频繁的 non-missing value。
- top ties 按该 column 中第一次出现顺序打破。
- `top_count` 是 top frequency。
- `top_rate = top_count / count`；`count == 0` 时为 `0.0`。
- Summary rows 按 analyzed column order 排序。
- 无 analyzed columns 时返回 fixed columns and dtypes 的空 DataFrame。

### `CategoricalAnalysis.top_categories`

`top_categories` 必须是 pandas DataFrame，列顺序冻结为：

1. `column`
2. `category`
3. `count`
4. `rate`
5. `rank`

Dtypes:

- `column`: object
- `category`: object
- `count`: int64
- `rate`: float64
- `rank`: int64

Rules:

- 每个 displayed category 一行。
- 只包含 non-missing values。
- 每列最多 `top_n` 行。
- Categories 按 descending count 排序。
- Ties 按该 column 中第一次出现顺序打破。
- `rank` 从每列 1 开始。
- `rate = count / non_missing_count`。
- Column groups 按 analyzed column order 排序。
- 无 analyzed columns 时返回 fixed columns and dtypes 的空 DataFrame。

## Correlation Analysis

### Column Selection

Eligible columns:

- pandas numeric dtype；
- excluding boolean dtype。

Auto-selection 使用 DataFrame order 中所有 eligible numeric non-boolean
columns。Explicit columns 先验证名称；eligible numeric non-boolean columns
可以分析，non-numeric columns 使用 `not_numeric` 跳过。

Additional skip rules:

- all-missing numeric columns 使用 `all_missing` 跳过；
- non-missing values 少于 `min_periods` 的列使用
  `insufficient_non_missing` 跳过；
- constant columns 使用 `constant` 跳过。

`method` 只支持 `"pearson"` 和 `"spearman"`；否则抛出 `ValueError`，消息
包含 `method must be pearson or spearman`。

`max_columns` 必须是 int 且 `>= 2`；否则抛出 `ValueError`，消息包含
`max_columns must be an integer >= 2`。

`min_periods` 必须是 int 且 `>= 2`；否则抛出 `ValueError`，消息包含
`min_periods must be an integer >= 2`。

Budget behavior:

- 先应用 eligibility/skips。
- 如果 eligible analyzable columns 超过 `max_columns`，保留
  request/DataFrame order 中前 `max_columns` 列。
- 超出 budget 的列进入 `skipped_columns`，reason 为
  `exceeds_max_columns`。
- 任何列因 `exceeds_max_columns` 跳过时 `truncated=True`，否则为 False。

### `CorrelationAnalysis.correlations`

使用 long-form pairwise table，不使用 matrix。列顺序冻结为：

1. `column_a`
2. `column_b`
3. `method`
4. `correlation`
5. `n_pairs`

Dtypes:

- `column_a`: object
- `column_b`: object
- `method`: object
- `correlation`: float64
- `n_pairs`: int64

Rules:

- 无 diagonal rows。
- 每个 unordered pair 只保留一行。
- Pair order 跟随 analyzed column order：`(col_i, col_j)` for `i < j`。
- `n_pairs` 是两列都 non-missing 的行数。
- 如果 `n_pairs < min_periods`，omit the pair。
- 使用 pandas `Series.corr` 和 requested method 计算 coefficient。
- 如果 pandas returns NaN，omit the pair。
- analyzed columns 少于 2 时返回 fixed-schema empty DataFrame。
- Task 07 不返回 p-values。
- Task 07 不生成 correlation heatmap。

## Outlier Detection

### Column Selection

Eligible columns:

- pandas numeric dtype；
- excluding boolean dtype。

Auto-selection 使用 DataFrame order 中所有 eligible numeric non-boolean
columns。Explicit columns 先验证名称；non-numeric columns 使用
`not_numeric` 跳过。

Additional skip rules:

- all-missing columns 使用 `all_missing` 跳过；
- fewer than 2 non-missing finite values 使用 `insufficient_non_missing` 跳过；
- constant finite values 使用 `constant` 跳过；
- 包含 positive 或 negative infinity 的列使用 `non_finite_values` 跳过。

`method` 在 Task 07 只支持 `"iqr"`；否则抛出 `ValueError`，消息包含
`method must be iqr`。

`threshold` 必须是 int 或 float 且 `> 0`；否则抛出 `ValueError`，消息包含
`threshold must be a positive number`。

IQR formula:

- `q1 = pandas quantile 0.25 default interpolation`；
- `q3 = pandas quantile 0.75 default interpolation`；
- `iqr = q3 - q1`；
- `lower_bound = q1 - threshold * iqr`；
- `upper_bound = q3 + threshold * iqr`；
- outlier if `value < lower_bound or value > upper_bound`；
- missing values are not outliers；
- infinity causes whole column to be skipped with `non_finite_values`；
- if `iqr == 0` but finite values are not constant,
  `lower_bound == upper_bound` and values outside that bound are outliers。

### `OutlierAnalysis.summary`

`summary` 必须是 pandas DataFrame，列顺序冻结为：

1. `column`
2. `method`
3. `threshold`
4. `lower_bound`
5. `upper_bound`
6. `outlier_count`
7. `outlier_rate`

Dtypes:

- `column`: object
- `method`: object
- `threshold`: float64
- `lower_bound`: float64
- `upper_bound`: float64
- `outlier_count`: int64
- `outlier_rate`: float64

Rules:

- 每个 analyzed column 一行。
- `outlier_rate = outlier_count / non_missing_finite_count`。
- Summary rows 按 analyzed column order 排序。
- 无 analyzed columns 时返回 fixed columns and dtypes 的空 DataFrame。

### `OutlierAnalysis.outliers`

`outliers` 必须是 pandas DataFrame，列顺序冻结为：

1. `column`
2. `row_index`
3. `value`
4. `lower_bound`
5. `upper_bound`

Dtypes:

- `column`: object
- `row_index`: object
- `value`: float64
- `lower_bound`: float64
- `upper_bound`: float64

Rules:

- 每个 outlier value 一行。
- `row_index` stores the original DataFrame index label。
- Outlier rows ordered by analyzed column order, then original DataFrame row
  order。
- Missing values omitted。
- 无 outliers 时返回 fixed columns and dtypes 的空 DataFrame。

## Deterministic Ordering

- Requested columns follow caller order。
- Auto-selected columns follow DataFrame column order。
- Skipped columns follow request/DataFrame order。
- Summary rows follow analyzed column order。
- Category rows sorted by descending count；ties by first appearance in original
  column。
- Correlation pairs follow analyzed column order with `i < j`。
- Outliers follow analyzed column order then original row order。
- Do not sort alphabetically unless explicitly stated。
- Do not include timestamps、random IDs 或 runtime-dependent fields。

## Error Behavior

只使用 built-in exceptions，不创建 custom exceptions。

Stable errors:

- non-DataFrame `df`: `ValueError`，消息包含
  `df must be a pandas DataFrame`。
- non-string DataFrame columns: `ValueError`，消息包含
  `DataFrame column names must all be strings`。
- duplicate DataFrame column names: `ValueError`，消息包含
  `duplicate DataFrame column names are not supported`。
- non-string column parameter: `ValueError`，消息包含
  `columns must contain only strings`。
- duplicate column parameter: `ValueError`，消息包含
  `duplicate column parameter`。
- missing requested column: `ValueError`，消息包含 `column not found`。
- invalid `top_n`: `ValueError`，消息包含
  `top_n must be a positive integer`。
- invalid correlation method: `ValueError`，消息包含
  `method must be pearson or spearman`。
- invalid `max_columns`: `ValueError`，消息包含
  `max_columns must be an integer >= 2`。
- invalid `min_periods`: `ValueError`，消息包含
  `min_periods must be an integer >= 2`。
- invalid outlier method: `ValueError`，消息包含 `method must be iqr`。
- invalid `threshold`: `ValueError`，消息包含
  `threshold must be a positive number`。

## Public API 与 `__all__`

Task 07 implementation must export:

- `NumericAnalysis`
- `CategoricalAnalysis`
- `CorrelationAnalysis`
- `OutlierAnalysis`
- `analyze_numeric_features`
- `analyze_categorical_features`
- `compute_correlations`
- `detect_outliers`

这些符号必须加入 `src/sharper/__init__.py` 和 `__all__`。不得删除或改变
Tasks 01-06 已有 public API。

## 测试合同

Task 07 tests must cover:

1. public API exports and signatures；
2. result dataclass frozen behavior；
3. non-DataFrame rejected；
4. non-string DataFrame columns rejected；
5. duplicate DataFrame columns rejected；
6. missing requested column rejected；
7. duplicate requested column rejected；
8. non-string requested column rejected；
9. input DataFrame not mutated；
10. numeric auto-select numeric non-boolean columns；
11. numeric explicit columns preserve order；
12. numeric skip non-numeric explicit column；
13. numeric skip all-missing numeric column；
14. numeric fixed summary columns and dtypes；
15. numeric `zero_count` / `zero_rate`；
16. numeric empty DataFrame returns fixed empty schema；
17. categorical auto-select object/string/category/bool columns；
18. categorical explicit columns preserve order；
19. categorical skip numeric explicit column；
20. categorical invalid `top_n` rejected；
21. categorical fixed summary columns and dtypes；
22. categorical fixed `top_categories` columns and dtypes；
23. categorical `top_n` budget；
24. categorical tie break by first appearance；
25. categorical all-missing skipped；
26. correlation auto-select numeric non-boolean columns；
27. correlation explicit columns preserve order；
28. correlation invalid method rejected；
29. correlation invalid `max_columns` rejected；
30. correlation invalid `min_periods` rejected；
31. correlation skip non-numeric/all-missing/constant/insufficient columns；
32. correlation `max_columns` truncation and `exceeds_max_columns`；
33. correlation long-form pair table order；
34. correlation `n_pairs` and `min_periods` behavior；
35. correlation no diagonal rows；
36. correlation empty fixed schema；
37. correlation all-missing precedence beats insufficient；
38. correlation insufficient precedence beats constant；
39. correlation `exceeds_max_columns` applies only after earlier skips；
40. outlier method only `iqr`；
41. outlier invalid `threshold` rejected；
42. outlier skip non-numeric/all-missing/constant/insufficient/non-finite columns；
43. outlier all-missing precedence beats insufficient；
44. outlier `non_finite_values` precedence beats insufficient；
45. outlier insufficient precedence beats constant where applicable；
46. outlier IQR lower/upper bound behavior；
47. outlier summary schema and dtypes；
48. outlier details schema and dtypes；
49. outlier `row_index` preserves original index label；
50. outlier deterministic order；
51. outlier no-outlier empty details schema；
52. Task 01-06 tests still pass；
53. no workflow/reporting/CLI changes。

## Verification Commands

Task 07 implementation and review must use:

```bash
.venv/bin/python -m pytest
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
.venv/bin/python -m build
```

Do not use system `python3`.
Do not rely on `python` command.

## Documentation Sync

Update `SPEC.md` and `IMPLEMENTATION_PLAN.md` to reflect:

- Task 07 is non-target analysis only；
- it does not modify CLI、workflow 或 reporting；
- it does not perform target relationship analysis；
- it exports four functions and four result types；
- it uses fixed result table schemas and deterministic ordering。

README may mention Task 07 only as planned/future until implementation is
complete.

## 明确推迟

以下能力明确推迟到后续任务：

- target relationship analysis；
- grouped analysis；
- feature engineering；
- visualization；
- modeling；
- evaluation；
- report generation；
- workflow integration；
- CLI integration；
- automatic cleaning；
- data mutation；
- custom exceptions。
