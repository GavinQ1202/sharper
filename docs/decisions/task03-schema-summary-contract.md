# Task 03 Schema 与 DataFrame Summary 公共契约

## 状态

已接受。本文是 Task 03 实现前的 API 决策记录。Task 03 的实现、测试和
API 文档必须遵守本文；修改本文冻结的 public API 需要先同步评审
`SPEC.md` 和 `IMPLEMENTATION_PLAN.md`。

## 范围

本文只冻结 `infer_schema`、`summarize_dataframe` 及其最小公开结果类型。
不定义质量问题、重复行、异常值、相关性、target relationship、特征工程、
可视化、建模、报告或 CLI 行为。

所有公开结果类型使用 dataclass。结果字段的容器本身仍是普通 `list`、
`dict` 和 `pandas.DataFrame`；调用者不应依赖深度不可变性。两个函数均不
修改输入 DataFrame。

## 公开结果类型

### `ColumnSchema`

字段按以下顺序冻结：

| 字段 | 类型 | 含义 |
|---|---|---|
| `name` | `str` | 原始列名 |
| `pandas_dtype` | `str` | `str(series.dtype)` 的结果 |
| `logical_type` | `str` | 本文定义的逻辑类型 |
| `nullable` | `bool` | 当前数据中是否至少观察到一个缺失值 |
| `missing_count` | `int` | 缺失单元格数 |
| `missing_rate` | `float` | `missing_count / n_rows`；`n_rows == 0` 时为 `0.0` |
| `unique_count` | `int` | 非缺失唯一值数，等价于 `nunique(dropna=True)` |
| `unique_rate` | `float` | `unique_count / non_null_count`；无非缺失值时为 `0.0` |
| `is_constant` | `bool` | 至少有一个非缺失值且 `unique_count == 1` |
| `is_id_like` | `bool` | 是否命中本文的 identifier 规则 |
| `confidence` | `float` | `[0.0, 1.0]` 内的确定性置信度 |
| `reasons` | `list[str]` | 按规则优先级排列、稳定且可读的判定依据 |

`nullable` 描述观察到的数据，不表示 pandas dtype 理论上能否承载缺失值。
全缺失列不是常量列。

Task 03 只支持字符串列名。任一列名不是 `str` 时，`infer_schema` 和
`summarize_dataframe` 均抛出 `ValueError`，错误消息必须包含
`DataFrame column names must all be strings`。不得自动将列名转换为
字符串，以免例如整数 `1` 与字符串 `"1"` 发生冲突。

### 逻辑类型

`logical_type` 只允许：

- `"numeric"`
- `"categorical"`
- `"datetime"`
- `"boolean"`
- `"text"`
- `"identifier"`
- `"unknown"`

Task 03 不增加其他逻辑类型。

### `TargetCandidate`

字段按以下顺序冻结：

| 字段 | 类型 | 含义 |
|---|---|---|
| `name` | `str` | 候选列名 |
| `suggested_task_type` | `str` | `"classification"`、`"regression"` 或 `"unknown"` |
| `confidence` | `float` | `[0.0, 1.0]` 内的建议置信度 |
| `reasons` | `list[str]` | 稳定且可读的候选依据 |

候选只是提示，不确认 target，不运行 target-aware 分析或统计检验。

### `SchemaReport`

字段按以下顺序冻结：

| 字段 | 类型 |
|---|---|
| `n_rows` | `int` |
| `n_columns` | `int` |
| `columns` | `list[ColumnSchema]` |
| `logical_type_counts` | `dict[str, int]` |
| `target_candidates` | `list[TargetCandidate]` |

`columns` 保持 DataFrame 原始列顺序。`logical_type_counts` 必须按本文逻辑
类型列出的固定顺序包含全部七个 key，包括计数为零的类型。
`target_candidates` 保持原始列顺序，但显式 `target`（如提供）排在首位。
不增加 `generated_at` 或其他非确定性字段。

### `DataFrameSummary`

字段按以下顺序冻结：

| 字段 | 类型 | 含义 |
|---|---|---|
| `n_rows` | `int` | 行数 |
| `n_columns` | `int` | 列数 |
| `memory_usage_bytes` | `int` | `df.memory_usage(index=True, deep=True).sum()` |
| `total_missing_cells` | `int` | 全表缺失单元格数 |
| `total_missing_rate` | `float` | 缺失单元格数除以 `n_rows * n_columns`；分母为零时为 `0.0` |
| `schema` | `SchemaReport` | 使用或生成的 schema |
| `column_summary` | `pandas.DataFrame` | 下文冻结的逐列摘要 |

若调用者传入 `schema`，它的 shape、列名和顺序必须与 `df` 一致，否则
`summarize_dataframe` 抛出带可操作消息的 `ValueError`。

## `column_summary` 明细契约

列名和顺序冻结为：

1. `column`
2. `pandas_dtype`
3. `logical_type`
4. `non_null_count`
5. `missing_count`
6. `missing_rate`
7. `unique_count`
8. `unique_rate`
9. `is_constant`
10. `is_id_like`
11. `min`
12. `max`
13. `mean`
14. `std`
15. `q25`
16. `median`
17. `q75`

每个输入列对应一行，顺序与输入 DataFrame 相同。各列 dtype 冻结如下：

| 列 | dtype |
|---|---|
| `column` | `object` |
| `pandas_dtype` | `object` |
| `logical_type` | `object` |
| `non_null_count` | `int64` |
| `missing_count` | `int64` |
| `missing_rate` | `float64` |
| `unique_count` | `int64` |
| `unique_rate` | `float64` |
| `is_constant` | `bool` |
| `is_id_like` | `bool` |
| `min` | `object` |
| `max` | `object` |
| `mean` | `float64` |
| `std` | `float64` |
| `q25` | `float64` |
| `median` | `float64` |
| `q75` | `float64` |

只有逻辑类型为 `"numeric"` 的列计算 `min`、`max`、`mean`、`std`、
`q25`、`median` 和 `q75`。计算忽略缺失值，`std` 使用 pandas 默认的
样本标准差 `ddof=1`。不适用、无法稳定比较或全缺失时，`min` 和 `max`
使用 `None`；不适用或不可定义的 `mean`、`std`、`q25`、`median` 和
`q75` 使用 `float("nan")`。Task 03 不为 datetime、boolean、categorical、
text、identifier 或 unknown 列计算这些统计。

0 行 0 列输入的 `column_summary` 必须是具有全部固定列名和上述固定
dtype 的空 DataFrame。0 行但有列时，每个输入列仍对应一行，统计缺失值
按上一段填充，整个结果仍严格保持上述 dtype。

## Schema 推断规则

规则按以下优先级执行。所有检测只读取数据，不转换或回写原列：

1. empty-row 与 all-missing；
2. pandas dtype direct rules；
3. boolean-like string rule；
4. datetime string detection；
5. mixed object unknown；
6. identifier detection；
7. categorical、text 与 fallback unknown。

第 2 步中 pandas bool、datetime 和 numeric dtype 可直接确定；category
dtype 有一个窄例外：必须先经过第 3 步的严格 boolean-token 检测，未命中
后才由 `pandas_category_dtype` 直接判为 categorical。这样 category
`"true"/"false"` 不会被过早截获。

### 共同计数

- 缺失值遵循 pandas `isna` 语义。
- `non_null_count = n_rows - missing_count`。
- `unique_count` 排除缺失值。
- 所有 rate 的分母和零分母行为按 `ColumnSchema` 定义。
- 重复列名使 `infer_schema` 和 `summarize_dataframe` 抛出 `ValueError`。
- 非字符串列名使两个函数抛出 `ValueError`，消息包含
  `DataFrame column names must all be strings`。
- `id_threshold` 必须在 `(0.0, 1.0]`，否则抛出 `ValueError`。

### 空行或全缺失

- 当整个 DataFrame 为 0 行时，每列均为 `"unknown"`、置信度 `0.5`，
  无论其物理 dtype 是什么；计数为 0、rate 为 `0.0`、`nullable` 为
  `False`、`is_constant` 和 `is_id_like` 均为 `False`。
- 有行但某列全缺失时，该列为 `"unknown"`、置信度低于明确类型；
  `nullable=True`、`missing_rate=1.0`、`unique_count=0`、
  `unique_rate=0.0`、`is_constant=False`、`is_id_like=False`。

### boolean

- pandas boolean dtype（包括 nullable BooleanDtype）直接判为
  `"boolean"`，confidence 为 `1.0`，reason 为
  `pandas_boolean_dtype`。
- object、string、category 或 pandas StringDtype 列，如果所有非缺失值
  经 `str(value).strip().casefold()` 规范化后都属于
  `{"true", "false"}`，则判为 `"boolean"`，confidence 为 `0.8`，
  reason 为 `boolean_values_only`。
- 该规则大小写不敏感并忽略前后空格，因此 `" TRUE "`、`"false"` 以及
  实际 Python/pandas/numpy 布尔标量都可命中。检测只读取规范化后的临时
  值，不修改原始值或 dtype。
- `"yes"/"no"`、`"y"/"n"`、`"0"/"1"` 和数字 `0/1` 不自动判为
  boolean。其他布尔样式推迟到 v0.2。
- 全缺失列已由更高优先级规则判为 unknown，不进入 boolean 检测。

### datetime

- pandas datetime dtype 直接判为 `"datetime"`。
- 对非缺失值全部为字符串的 object 或 string 列，使用
  `pandas.to_datetime(..., errors="coerce")` 做轻量、只读检测；可解析
  比例大于等于 `0.80` 时判为 `"datetime"`，reason 为
  `string_datetime_parse_rate_met`。
- category、混合 Python 类型和纯数值列不做日期字符串检测。
- 检测不得修改原始列；解析比例低于 `0.80` 时继续执行后续规则。

### mixed object unknown

object 或 string-like 列若非缺失值跨越多个 Python 类型族（boolean、
numeric、string、datetime-like、other），即视为 mixed object；在 pandas
direct、严格 boolean-token 和 datetime-string 规则均未命中时，必须先于
identifier 判为 `"unknown"`，confidence 为 `0.5`，reason 为
`mixed_object_unknown`，不做字符串强制转换。

mixed object 即使 `unique_rate` 达到 identifier 阈值、所有值唯一，或列名
包含 `id`、`uuid`、`key`，仍不得判为 identifier。identifier detection
只适用于非 mixed-object 列。

### identifier

identifier 只对非 mixed-object 列检测。非缺失值数量必须大于 0，且
`unique_rate >= id_threshold`，并至少满足一项：

- 规范化后的列名包含独立 token `id`、`uuid` 或 `key`；或
- 所有非缺失值唯一，且物理 dtype 为整数、object 或 string。

命中时 `logical_type="identifier"` 且 `is_id_like=True`。浮点列不会仅因
为所有值唯一而成为 identifier，但带有上述列名 token 且达到阈值时可以。
默认 `id_threshold` 为 public signature 中冻结的 `0.98`；不另设硬编码
的 `0.95` 阈值。

### numeric

pandas numeric dtype 且不是 boolean，并且未命中 identifier 时判为
`"numeric"`。

### categorical

- pandas category dtype 在未被前述规则识别时判为 `"categorical"`。
- 全部非缺失值为字符串的 object 或 string 列，在
  `unique_rate <= 0.50` 或 `unique_count <= 50` 时判为
  `"categorical"`。

Task 03 不编码或转换类别值。

### text

全部非缺失值为字符串的 object 或 string 列，在
`unique_rate > 0.50`、`unique_count > 50` 且未命中 identifier 时判为
`"text"`。

### fallback unknown

全缺失列、0 行 DataFrame 中的列，以及无法命中前述规则的列均为
`"unknown"`。未被更具体 unknown 规则覆盖时使用 confidence `0.5` 和
reason `fallback_unknown`。unknown 的 `confidence` 必须低于任何明确逻辑
类型。

### 置信度与 reasons

`ColumnSchema.confidence` 只允许以下固定值，不使用动态公式：

| confidence | 使用场景 |
|---|---|
| `1.0` | pandas numeric、boolean、datetime 或 category dtype 直接确定 |
| `0.9` | identifier 规则命中 |
| `0.85` | object/string 通过轻量日期解析规则判为 datetime |
| `0.8` | object/string/category/StringDtype 通过严格 boolean-token 规则，或 object/string 通过基数规则判为 categorical/text |
| `0.5` | empty、all-missing、mixed object 或 fallback unknown |

若 category 列的非缺失值经规范化后只包含 `"true"` 和 `"false"`，
boolean 规则优先，使用 `0.8`；否则 category dtype 直接判为 categorical
并使用 `1.0`。

`ColumnSchema.reasons` 只能包含以下 reason codes：

- `pandas_numeric_dtype`
- `pandas_boolean_dtype`
- `boolean_values_only`
- `pandas_datetime_dtype`
- `pandas_category_dtype`
- `string_datetime_parse_rate_met`
- `identifier_name_pattern`
- `identifier_high_unique_rate`
- `categorical_unique_threshold_met`
- `text_high_unique_rate`
- `all_missing`
- `empty_dataframe`
- `mixed_object_unknown`
- `fallback_unknown`

每个结果至少包含触发最终逻辑类型的主 reason。identifier 总是包含
`identifier_high_unique_rate`，命中列名 token 时再追加
`identifier_name_pattern`。reasons 按推断规则和上表出现顺序稳定输出，
不得添加自由文本、时间戳、比例文本或其他 code。

## Target candidate 规则

自动候选必须具有名称信号。匹配使用原始字符串列名的
case-insensitive 小写形式，结果中的 `name` 保持原始大小写：

- 名称恰好为 `target`：`target_name_exact`；
- 名称包含但不完全等于 `target`：`target_name_contains_target`；
- 名称包含 `label`：`target_name_contains_label`；
- 名称包含 `outcome`：`target_name_contains_outcome`；
- 名称恰好为 `y`：`target_name_is_y`。

普通单词中出现字母 `y` 不构成信号。名称完全等于 `label` 或 `outcome`
时仍分别使用对应的 contains reason code。

- identifier 默认不进入自动候选。`excluded_identifier` 只描述内部排除
  规则，不得出现在最终候选的 `reasons` 中。
- 显式 `target` 必须存在且列名唯一，否则抛出 `ValueError`；它无论逻辑
  类型如何都作为第一个候选，且 reasons 必须包含 `explicit_target`，但仍
  只是建议，不表示确认或触发分析。不存在时的错误消息必须包含
  `target column not found`。
- boolean、categorical，或 numeric 且 `unique_count <= 20`：
  `suggested_task_type="classification"`。
- numeric 且 `unique_count > 20`：
  `suggested_task_type="regression"`。
- datetime、text、identifier 和 unknown：
  `suggested_task_type="unknown"`。
- boolean candidate 追加 `classification_boolean`；categorical candidate
  追加 `classification_categorical`；低基数 numeric candidate 追加
  `classification_low_cardinality`；高基数 numeric candidate 追加
  `regression_numeric_high_cardinality`。
- explicit identifier 允许进入候选，但 task type 为 `"unknown"`。

`TargetCandidate.confidence` 只允许 `0.9`、`0.75` 和预留的 `0.6`：

- `0.9`：显式 target 且 task type 非 unknown；或名称 case-insensitive
  完全等于 `target`、`label`、`outcome` 或 `y`。
- `0.75`：显式 target 且 task type 为 unknown；或名称只是包含
  `target`、`label` 或 `outcome` 而非上述精确匹配。
- `0.6`：仅为未来 SPEC 明确定义的弱候选信号预留。Task 03 v0.1 没有
  弱候选信号，因此不得产生 confidence 为 `0.6` 的候选。

同时命中多个规则时取最高 confidence。因而显式 unknown target 通常为
`0.75`；若其名称还精确等于 `target`、`label`、`outcome` 或 `y`，名称
强信号使最终 confidence 为 `0.9`。

`TargetCandidate.reasons` 只能包含：

- `explicit_target`
- `target_name_exact`
- `target_name_contains_target`
- `target_name_contains_label`
- `target_name_contains_outcome`
- `target_name_is_y`
- `classification_low_cardinality`
- `classification_boolean`
- `classification_categorical`
- `regression_numeric_high_cardinality`
- `excluded_identifier`

最终候选一般不得包含 `excluded_identifier`；该 code 只用于文档或内部
排除逻辑。每个最终候选按“explicit、名称信号、task type”的顺序包含全部
适用 reason codes，不得添加自由文本或其他 code。

候选不运行统计检验、关系分析或模型训练。

## 空 DataFrame 行为

- 0 行但有列：`infer_schema` 返回逐列 unknown `ColumnSchema`；
  `summarize_dataframe` 返回逐列 `column_summary`，计数和 rate 采用本文
  的零分母规则，统计值为 pandas 缺失。
- 0 行 0 列：两个函数均成功。`SchemaReport.columns` 和
  `target_candidates` 为空，七个逻辑类型计数均为 0。
  `column_summary` 是具有全部 17 个固定列名、固定 dtype 和 0 行的
  DataFrame。
- 两种情况均不因“空”本身抛出异常。重复列名、非法阈值、缺失的显式
  target 或不匹配的传入 schema 仍按各自规则报 `ValueError`。

## 明确推迟

以下内容不属于 Task 03 结果类型：

- duplicate row、缺失阈值、常量/近常量质量 issue、高基数 issue、
  类型异常 issue 及其他 `QualityReport` 字段：推迟到 Task 04；
- outlier、correlation、group comparison 和 target relationship：
  推迟到批准的 analysis tasks；
- 特征建议、可视化、建模、评估、workflow、报告和 CLI：推迟到各自任务；
- 自动数据清洗、类型强制转换和 target 自动确认：v0.1 Task 03 不提供。
