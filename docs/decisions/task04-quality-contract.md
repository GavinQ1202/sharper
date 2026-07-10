# Task 04 Minimal Data Quality 公共契约

## 状态

已接受。本文是 Task 04 实现前的 API 决策记录。Task 04 的实现、测试和
API 文档必须遵守本文；修改本文冻结的 public API、规则、阈值或稳定文本
需要先同步评审 `SPEC.md` 和 `IMPLEMENTATION_PLAN.md`。

## 范围

Task 04 只实现 minimal data quality reporting：报告 empty DataFrame、
duplicate rows、all-missing columns、high-missing columns、constant
columns、near-constant columns、high-cardinality categorical columns、
identifier-like columns、numeric infinite values、mixed Python object
types 和 datetime string parse failures。

Task 04 只报告问题和建议，不修改输入 DataFrame，也不自动执行建议。它
不实现 outlier detection、correlation analysis、target relationship
analysis、feature engineering、visualization、modeling、evaluation、
report generation、CLI、automatic data cleaning、duplicate index
detection、entity duplicate detection、data leakage detection 或 business
rule validation。

## 公开结果类型

所有公开结果类型均使用 `dataclass(frozen=True)`。容器字段仍为普通
`dict` 和 `list`，调用者不应依赖深度不可变性。

### `QualityIssue`

字段按以下顺序冻结：

| 字段 | 类型 | 含义 |
|---|---|---|
| `code` | `str` | 本文冻结的 issue code |
| `severity` | `str` | `"info"`、`"warning"` 或 `"error"` |
| `scope` | `str` | `"table"` 或 `"column"` |
| `column` | `str \| None` | 列级 issue 的原始字符串列名；表级 issue 为 `None` |
| `count` | `int \| None` | 触发该 issue 的数量；不适用时为 `None` |
| `ratio` | `float \| None` | 触发比例；不适用时为 `None` |
| `threshold` | `float \| None` | 规则使用的阈值；不适用时为 `None` |
| `message` | `str` | 稳定的问题说明 |
| `suggestion` | `str` | 稳定且不自动执行的建议 |

表级 issue 的 `column` 必须为 `None`；列级 issue 的 `column` 必须为
DataFrame 中的原始字符串列名。`message` 和 `suggestion` 不得包含
时间戳、路径、随机 ID 或其他非确定性内容。

### `QualityReport`

字段按以下顺序冻结：

| 字段 | 类型 | 含义 |
|---|---|---|
| `n_rows` | `int` | 输入行数 |
| `n_columns` | `int` | 输入列数 |
| `issue_count` | `int` | `len(issues)` |
| `severity_counts` | `dict[str, int]` | 各 severity 的 issue 数量 |
| `issues` | `list[QualityIssue]` | 按本文规则稳定排序的问题 |

`severity_counts` 必须按 `"info"`、`"warning"`、`"error"` 的顺序包含
全部三个 key，即使计数为 0。报告不包含 `SchemaReport`、
`DataFrameSummary`、`generated_at`、随机 ID、文件路径或其他非确定性
字段。

## Severity 与 issue code

Task 04 只允许以下 severity：

- `"info"`：值得注意但不一定阻塞分析；
- `"warning"`：常见数据质量风险；
- `"error"`：会让基本分析结果明显不可靠。

Task 04 只允许以下 issue code，并以此顺序作为排序优先级：

1. `empty_dataframe`
2. `duplicate_rows`
3. `all_missing_column`
4. `high_missing_column`
5. `constant_column`
6. `near_constant_column`
7. `high_cardinality_categorical`
8. `identifier_like_column`
9. `infinite_values`
10. `mixed_python_types`
11. `datetime_parse_failures`

不得生成其他 severity 或 issue code。

## `check_data_quality`

函数签名冻结为：

```python
def check_data_quality(
    df: pd.DataFrame,
    *,
    schema: SchemaReport | None = None,
    missing_threshold: float = 0.40,
) -> QualityReport: ...
```

`df` 必须是 pandas DataFrame。列名必须唯一；重复列名沿用 Task 03 的
`ValueError` 行为。非字符串列名抛出 `ValueError`，消息必须包含
`DataFrame column names must all be strings`，且不得转换列名。

`missing_threshold` 的合法区间为 `0 < missing_threshold <= 1`。非法值
抛出 `ValueError`，消息必须包含
`missing_threshold must be > 0 and <= 1`。

传入 `schema` 时，`schema.n_rows`、`schema.n_columns` 以及 schema 的
列名和顺序必须分别与 `len(df)`、`len(df.columns)` 和 DataFrame 列名
顺序完全一致；否则抛出 `ValueError`，消息必须包含
`schema does not match DataFrame`。未传入 schema 时可以调用
`infer_schema(df)`。

Task 04 不调用 `summarize_dataframe(df)`。它可以重复计算必要的 missing、
duplicate 和 constant 统计，但语义必须与 Task 03 一致。函数不得修改
输入 DataFrame。

## 固定质量规则

### Empty DataFrame

当 `n_rows == 0 and n_columns == 0` 时，只生成：

- `code="empty_dataframe"`、`severity="error"`、`scope="table"`；
- `column=None`、`count=0`、`ratio=None`、`threshold=None`；
- `message="DataFrame has no rows or columns"`；
- `suggestion="Provide data rows before running analysis"`。

当 `n_rows == 0 and n_columns > 0` 时，只生成相同 code 的表级
`warning`，字段同上，但
`message="DataFrame has no rows"`。0 行输入不生成任何列级 issue。

### Duplicate rows

使用全列判断，等价于 `df.duplicated(keep=False)`，并沿用 pandas 对
NaN 的默认重复语义。不忽略任何列，也不检查重复 index 或业务实体重复。
若属于重复组的行数大于 0，生成：

- `code="duplicate_rows"`、`severity="warning"`、`scope="table"`；
- `column=None`；
- `count` 为所有属于重复组的行数；
- `ratio=count / n_rows`、`threshold=None`；
- `message="Duplicate rows detected"`；
- `suggestion="Review duplicate rows and decide whether they should be removed or consolidated"`。

`n_rows == 0` 时不生成该 issue。

### All-missing column

若 `n_rows > 0` 且某列 `missing_count == n_rows`，生成：

- `code="all_missing_column"`、`severity="warning"`、`scope="column"`；
- `count=missing_count`、`ratio=1.0`、`threshold=None`；
- `message="Column contains only missing values"`；
- `suggestion="Consider dropping the column or investigating the data source"`。

全缺失列不再生成 high-missing 或 constant issue。

### High-missing column

若 `n_rows > 0`、`0 < missing_count < n_rows` 且
`missing_rate >= missing_threshold`，其中
`missing_rate = missing_count / n_rows`，生成：

- `code="high_missing_column"`、`severity="warning"`、
  `scope="column"`；
- `count=missing_count`、`ratio=missing_rate`、
  `threshold=missing_threshold`；
- `message="Column has a high missing rate"`；
- `suggestion="Review missingness before using this column for analysis or modeling"`。

不生成表级总体缺失 issue。

### Constant column

沿用 Task 03 语义：列恰有一个非缺失唯一值时为 constant；全缺失列不是
constant。生成：

- `code="constant_column"`、`severity="warning"`、`scope="column"`；
- `count=1`、`ratio=1.0`、`threshold=None`；
- `message="Column has a constant non-missing value"`；
- `suggestion="Consider excluding constant columns from analysis or modeling"`。

Constant 与 near-constant 互斥。

### Near-constant column

固定阈值 `near_constant_threshold = 0.95`，不作为 public 参数。只考虑
非缺失值；无非缺失值或已经是 constant 时不生成。若最常见非缺失值的
`top_count / non_null_count >= 0.95`，生成：

- `code="near_constant_column"`、`severity="info"`、
  `scope="column"`；
- `count=top_count`、`ratio=top_count / non_null_count`、
  `threshold=0.95`；
- `message="Column is near constant"`；
- `suggestion="Check whether the column adds useful variation before analysis or modeling"`。

### High-cardinality categorical

只适用于 schema 中 `logical_type == "categorical"` 的列，不适用于 text、
boolean、numeric、datetime、identifier 或 unknown。若
`unique_count > 50` 且 `unique_rate > 0.50`，生成：

- `code="high_cardinality_categorical"`、`severity="info"`、
  `scope="column"`；
- `count=unique_count`、`ratio=unique_rate`、`threshold=0.50`；
- `message="Categorical column has high cardinality"`；
- `suggestion="Consider grouping rare categories or using appropriate encoding strategies"`。

统计使用 Task 03 schema 中的 `unique_count` 和 `unique_rate`。同时判为
identifier 的列不生成该 issue。

### Identifier-like column

当 `logical_type == "identifier"` 或 `ColumnSchema.is_id_like is True`
时，生成：

- `code="identifier_like_column"`、`severity="info"`、
  `scope="column"`；
- `count=unique_count`、`ratio=unique_rate`、`threshold=None`；
- `message="Column appears to be an identifier"`；
- `suggestion="Avoid treating identifier-like columns as ordinary predictive features"`。

### Numeric infinite values

只检查非 boolean 的 pandas numeric dtype，使用 `np.isinf` 检查 `+inf`
和 `-inf`。当 infinite value 数量大于 0 时，生成：

- `code="infinite_values"`、`severity="warning"`、`scope="column"`；
- `count` 为 infinite value 数量；
- `ratio=count / n_rows`、`threshold=None`；
- `message="Numeric column contains infinite values"`；
- `suggestion="Replace or remove infinite values before analysis or modeling"`。

`n_rows == 0` 时不生成；不替换 inf，也不扩展为一般 outlier detection。

### Mixed Python types

只适用于 Task 03 schema 中 `logical_type == "unknown"` 且 `reasons`
包含 `mixed_object_unknown` 的列。`n_rows > 0` 时生成：

- `code="mixed_python_types"`、`severity="warning"`、
  `scope="column"`；
- `count=non_null_count`、`ratio=non_null_count / n_rows`、
  `threshold=None`；
- `message="Column contains mixed Python value types"`；
- `suggestion="Standardize the column values before analysis"`。

### Datetime parse failures

只适用于 object、string、category 或 `StringDtype` 列。对非缺失值调用
`pd.to_datetime(..., errors="coerce")`。仅当至少一个非缺失值可解析、
至少一个非缺失值解析失败，且 Task 03 未将该列判为 datetime string
时生成：

- `code="datetime_parse_failures"`、`severity="info"`、
  `scope="column"`；
- `count=parse_failure_count`；
- `ratio=parse_failure_count / non_null_count`、`threshold=None`；
- `message="Column contains partial datetime-like values"`；
- `suggestion="Review datetime parsing before using this column as a date or time feature"`。

检查不修改原始 DataFrame，也不执行类型转换。

## 重叠与互斥

以下规则冻结：

1. all-missing 与 high-missing 互斥；
2. all-missing 与 constant 互斥；
3. constant 与 near-constant 互斥；
4. identifier 与 high-cardinality categorical 互斥；
5. 0x0 DataFrame 只生成 `empty_dataframe`；
6. 0 行有列的 DataFrame 只生成 `empty_dataframe`；
7. `infinite_values` 可以与 high-missing、constant 等其他 issue 共存；
8. `mixed_python_types` 可以与 high-missing 共存；它不与 identifier
   冲突，因为 Task 03 已将该列判为 unknown。

## 稳定排序

`issues` 先排列表级 issue，再排列列级 issue；随后按本文冻结的 issue
code 顺序排列；同一 code 的列级 issue 按 DataFrame 原始列顺序排列，
不得按列名字母序排序。

## 无问题结果

没有任何 issue 时：

```python
issues = []
issue_count = 0
severity_counts = {"info": 0, "warning": 0, "error": 0}
```

相同输入和参数必须产生相同字段、issue code、证据、文本和排序。
