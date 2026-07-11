# Task 09 Feature Suggestions and Safe Stateless Derivation 公共契约

## 状态

已接受。本文是 Task 09 实现前的 API 决策记录。Task 09 的实现、测试和
API 文档必须遵守本文；修改本文冻结的 public API、结果字段、vocabulary、
列选择、预算、命名、排序、错误、物化或 copy 行为前，必须先同步评审本文、
`SPEC.md` 和 `IMPLEMENTATION_PLAN.md`。

## 范围

Task 名称冻结为：
**Task 09 — Feature suggestions and safe stateless derivation**。

Task 09 只实现：

1. 确定性候选特征建议；
2. 固定 per-type budget、全局预算和稳定排序；
3. 安全无状态 arithmetic materialization；
4. 安全无状态 datetime component materialization；
5. 基于显式 reference date 的 days-since 衍生；
6. 对 fitted、target-aware 和 aggregate 候选只生成结构化建议，不物化。

Task 09 不实现 learned/fixed binning materialization、group aggregate
materialization、target encoding、WOE、监督分箱、target-aware
materialization、correlation-based candidate search、model-based feature
selection、transformer hierarchy、fit/transform state、workflow/reporting/CLI
integration、visualization、modeling、evaluation、HTML、cleaning 或 custom
exceptions。

## Public API

签名冻结为：

```python
def suggest_feature_derivations(
    df: pd.DataFrame,
    *,
    schema: SchemaReport | None = None,
    target: str | None = None,
    exclude_columns: Sequence[str] = (),
    reference_date: str | date | datetime | pd.Timestamp | None = None,
    max_suggestions: int = 50,
) -> FeatureSuggestionReport: ...

def derive_features(
    df: pd.DataFrame,
    suggestions: Sequence[FeatureSuggestion],
    *,
    copy: bool = True,
) -> FeatureDerivationResult: ...
```

`reference_date` 和 `exclude_columns` 只属于 suggestion generation。
reference date 规范化后写入 suggestion parameters；`derive_features` 不接受
隐式 reference date，也不得读取系统日期。

Task 09 实现后从 `sharper` 导出：

- `FeatureSuggestion`
- `FeatureSuggestionReport`
- `FeatureDerivationResult`
- `suggest_feature_derivations`
- `derive_features`

不得修改 Tasks 01–08 public API。

## Public Result Types

三个结果类型均使用 `@dataclass(frozen=True)`。其中 `dict` 和 DataFrame 字段
不承诺深度不可变性。

### `FeatureSuggestion`

字段按以下顺序冻结：

```python
@dataclass(frozen=True)
class FeatureSuggestion:
    name: str
    feature_type: str
    source_columns: tuple[str, ...]
    formula: str | None
    parameters: tuple[tuple[str, str], ...]
    reason: str
    risk: str
    requires_fit: bool
    priority: int
```

- `name` 是建议输出列名。
- `feature_type`、`reason` 和 `risk` 使用本文封闭 vocabulary。
- `source_columns` 保持公式使用顺序。
- `formula` 是稳定的人类可读表达式；纯参数型候选为 `None`。
- `parameters` 是稳定有序的字符串 `(key, value)` tuple，不使用 mutable dict。
- `priority` 为正整数，数值越小优先级越高。
- 结果不包含 timestamp、随机 ID、路径、duration 或 callable。

### `FeatureSuggestionReport`

字段按以下顺序冻结：

```python
@dataclass(frozen=True)
class FeatureSuggestionReport:
    n_rows: int
    requested_target: str | None
    requested_exclusions: tuple[str, ...]
    reference_date: str | None
    eligible_columns: tuple[str, ...]
    excluded_columns: tuple[str, ...]
    skipped_columns: tuple[str, ...]
    skipped_reasons: dict[str, str]
    max_suggestions: int
    type_budgets: dict[str, int]
    available_counts: dict[str, int]
    available_suggestion_count: int
    truncated: bool
    truncation_reason: str | None
    suggestions: tuple[FeatureSuggestion, ...]
```

- `reference_date` 是规范化的 `YYYY-MM-DD`，未提供时为 `None`。
- `available_counts` 是各类 candidate 完成名称冲突过滤和去重后、应用该类
  budget 前的数量。
- `available_suggestion_count` 是各类 budget 截断后、全局 budget 截断前的
  总数。
- `truncated=True` 当且仅当 global `max_suggestions` 进一步截断；此时
  `truncation_reason="max_suggestions"`。否则为 `False` 和 `None`。
- per-type truncation 只通过 `available_counts` 与 `type_budgets` 披露。

### `FeatureDerivationResult`

字段按以下顺序冻结：

```python
@dataclass(frozen=True)
class FeatureDerivationResult:
    data: pd.DataFrame
    applied_suggestions: tuple[str, ...]
    skipped_suggestions: tuple[str, ...]
    skipped_reasons: dict[str, str]
    copy: bool
```

合法且可物化的 suggestions 全部应用。合同验证错误 fail-fast。Task 09 正常
成功结果固定满足 `skipped_suggestions == ()` 和 `skipped_reasons == {}`；这两个
字段为后续版本保留，Task 09 不得发明 runtime skip 条件。计算结果中的 `NaN`
或 `pd.NA` 不表示 suggestion 被 skipped。

## Suggestion Vocabularies

### Feature types

可物化类型只允许：

- `ratio`
- `difference`
- `product`
- `datetime_year`
- `datetime_month`
- `datetime_quarter`
- `datetime_dayofweek`
- `datetime_is_weekend`
- `datetime_days_since_reference`

仅建议、不可物化类型只允许：

- `binning_candidate`
- `group_aggregate_candidate`
- `target_encoding_candidate`

不得新增其他 feature type。

### Risk

只允许 `low`、`medium`、`high`：

- arithmetic 与 datetime component：`low`
- days-since-reference：`medium`
- binning candidate：`medium`
- group aggregate candidate：`high`
- target encoding candidate：`high`

### Reason

只允许：

- `numeric_pair_arithmetic`
- `datetime_component`
- `explicit_reference_date`
- `numeric_binning_review`
- `categorical_group_aggregate_review`
- `target_aware_encoding_review`

不得生成自由文本 reason。

### `requires_fit`

- ratio、difference、product 和所有 datetime materializable types：`False`
- binning、group aggregate 和 target encoding candidates：`True`

`derive_features` 遇到任一 `requires_fit=True` suggestion 时整次调用 fail-fast，
抛出 `ValueError`，消息包含：

```text
requires_fit suggestions cannot be materialized in Task 09
```

不得降级为 skipped result。

## Shared DataFrame Validation

两个函数共同遵守：

- `df` 不是 pandas DataFrame：`ValueError`，消息包含
  `df must be a pandas DataFrame`；
- 任一 DataFrame 列名不是字符串：`ValueError`，消息包含
  `DataFrame column names must all be strings`；
- 重复 DataFrame 列名：`ValueError`，消息包含
  `duplicate DataFrame column names are not supported`。

不得修改 Task 07/08 的 validation 合同。

## Suggestion Input Validation

### Schema

`schema=None` 时调用 `infer_schema(df)`。Task 09 不调用 Task 07/08 analysis
functions，不计算 correlation，也不调用 `summarize_dataframe` 或
`check_data_quality`。

外部 schema 必须是 `SchemaReport`，否则 `ValueError`，消息包含：

```text
schema must be a SchemaReport
```

必须核对 schema 的 `n_rows`、`n_columns`、column names 和 column order；不匹配
时 `ValueError`，消息包含：

```text
schema does not match DataFrame
```

### Target

`target=None` 合法。非 `None` target 必须为字符串且存在，否则分别抛出
`ValueError`，消息包含：

```text
target must be a string
target column not found
```

target 永远不作为 suggestion source，不参与 pair arithmetic。Task 09 不读取
target values 来评分。当 target 非 `None` 时，可对 eligible categorical column
生成 `target_encoding_candidate`；它只依据 dtype/schema，固定
`requires_fit=True`、`risk="high"`。

### Exclusions

`exclude_columns` 中每项必须是字符串、不得重复且必须存在，否则分别抛出
`ValueError`，消息包含：

```text
columns must contain only strings
duplicate column parameter
column not found
```

target 与 exclusions 可以重叠，只记录一次 exclusion。requested exclusions 保持
调用者顺序；excluded columns 不作为任何 suggestion source。

### Reference date

允许严格 ISO `YYYY-MM-DD` 字符串、`datetime.date`、`datetime.datetime` 和
`pandas.Timestamp`。类型 dispatch 顺序冻结为：先单独处理 `pd.Timestamp`，
再处理 `datetime.datetime`，最后处理 `datetime.date`；字符串单独严格解析，
其他类型拒绝。`datetime.datetime` 是 `datetime.date` 的子类，不得先进入 date
分支。

- 在任何类型 branch 前先显式拒绝 `pd.NaT` sentinel；随后按上述类型顺序 dispatch；
- `pd.Timestamp`：拒绝 timezone-aware value；timezone-naive 时
  调用 `.normalize()`，只保留 calendar date；
- `datetime.datetime`：拒绝 timezone-aware value；timezone-naive 时忽略时分秒，
  使用 `.date()`；
- `datetime.date`：直接作为 calendar date；
- string：必须严格匹配并成功解析 `YYYY-MM-DD`。

所有合法输入最终序列化为 `YYYY-MM-DD` timezone-naive calendar date。

- timezone-aware datetime/Timestamp：`ValueError`，消息包含
  `reference_date must be timezone-naive`；
- `NaT`、非法日期或其他类型：`ValueError`，消息包含
  `reference_date must be a valid date`。

不允许把当前系统日期作为默认值。未提供时不生成
`datetime_days_since_reference`。即使没有 eligible timezone-naive datetime
column，report 仍保存 normalized reference date、不生成 days-since suggestion，
也不报错。提供后参数固定为：

```python
(("reference_date", "YYYY-MM-DD"),)
```

### Global budget

`max_suggestions` 必须是非 bool 的 int 且至少为 1，否则 `ValueError`，消息包含：

```text
max_suggestions must be a positive integer
```

Suggestion validation precedence 冻结为：

1. DataFrame validation；
2. schema type/match，`schema=None` 时在 DataFrame validation 后 inference；
3. target；
4. exclude columns；
5. reference date；
6. max suggestions。

## Eligible and Excluded Columns

Arithmetic numeric 只接受 real numeric non-boolean：integer、unsigned integer、
float、nullable integer 和 nullable float。bool、complex、datetime、timedelta、
object、string 和 category 不进入 arithmetic。

Datetime source 只接受 timezone-naive pandas datetime dtype。在没有命中更高
precedence 的 target、explicit exclusion、identifier-like、all-missing 或 constant
reason 时，timezone-aware datetime column 不解析、不转换、不移除 timezone，
固定使用 `unsupported_dtype` 进入 `skipped_columns`，且不生成任何 datetime
candidate。Task 09 不自动解析 object/string。

Categorical review candidates 接受 object、string、category 和 bool。

Candidate generation 固定为：每个 eligible datetime column 生成五个 component
candidates，并在存在 reference date 时再生成一个 days-since candidate；每个
eligible arithmetic numeric unordered pair 生成 ratio、difference 和 product；
每个 eligible arithmetic numeric column 生成一个 binning candidate；每个
eligible categorical column 生成一个 group aggregate candidate，并且仅在
target 非 `None` 时再生成一个 target encoding candidate。

以下列不生成任何 suggestion：

- target；
- explicit exclusions；
- schema logical type `identifier`；
- `is_id_like=True`；
- all-missing；
- constant；
- unsupported dtype；
- duplicate-content column。

Duplicate-content detection 只在依次完成 target、explicit exclusions、
identifier-like、all-missing、constant 和 unsupported dtype 六步前置过滤后的
剩余候选列之间执行。第六步判断列是否至少可参与一种批准 suggestion type；
timezone-aware datetime、complex、timedelta 和其他 unsupported dtype columns
直接使用 `unsupported_dtype`，不参加 duplicate comparison。前述步骤已
excluded/skipped 的列不得参与比较，也不得成为 retained representative。
按 DataFrame 原始列顺序遍历剩余候选列；当前列只与此前仍保留的候选列比较。
只有 dtype 字符串完全相同且 `Series.equals()` 为 `True` 才标记
`duplicate_content`；其 missing equality 语义继承 `Series.equals()`。保留首次
出现的合格候选列；已标记 duplicate 的列不作为后续比较的 representative。
不做跨 dtype、近似、相关性或名称重复检测。

Skipped column reason 只允许：

1. `target_column`
2. `explicitly_excluded`
3. `identifier_like`
4. `all_missing`
5. `constant`
6. `unsupported_dtype`
7. `duplicate_content`

以上顺序也是 precedence，每列至多一个 reason。target 同时显式排除时使用
`target_column`。某列只要可参与至少一种候选类型，就不算
`unsupported_dtype`。

- `excluded_columns` 只包含 reason 为 `target_column` 或
  `explicitly_excluded` 的列；target 同时显式排除时只出现一次并使用
  `target_column`；
- `skipped_columns` 只包含 reason 为 `identifier_like`、`all_missing`、
  `constant`、`unsupported_dtype` 或 `duplicate_content` 的列；
- `skipped_reasons` 按 DataFrame 原顺序包含 `excluded_columns` 与
  `skipped_columns` 中每一列的唯一 reason，不为 eligible column 提供 reason；
- `eligible_columns`、`excluded_columns` 和 `skipped_columns` 均按 DataFrame
  原顺序，`requested_exclusions` 保持 caller order。

对输入 DataFrame 的每个 column，三个状态 tuple 构成完整 partition：它恰好
属于 `eligible_columns`、`excluded_columns`、`skipped_columns` 之一；三者互不
重叠，并集等于全部 DataFrame columns。最终状态 tuple 不按 caller exclusions
顺序重排。

## Budgets, Enumeration, Naming and Ordering

### Per-type budgets

`type_budgets` 和 `available_counts` 始终按以下顺序包含全部 key：

| Key | Budget |
|---|---:|
| `datetime` | 20 |
| `ratio` | 10 |
| `difference` | 10 |
| `product` | 10 |
| `binning_candidate` | 5 |
| `group_aggregate_candidate` | 5 |
| `target_encoding_candidate` | 5 |

生成流程冻结为：先生成每类全部 deterministic candidates；完成名称冲突过滤和
identity 去重；应用 per-type budget；按表中类型顺序连接；最后应用 global
`max_suggestions`。不做随机 sampling，不按数据统计或 target 结果排序。

### Pair enumeration

Arithmetic sources 按 DataFrame 顺序。对每个 unordered pair `i < j`：

- ratio 只生成 `col_i / col_j`；
- difference 只生成 `col_i - col_j`；
- product 生成 `col_i * col_j`；
- 不生成反向或 self-pair。

### Naming

固定模板：

| Type | Name |
|---|---|
| ratio | `{left}__div__{right}` |
| difference | `{left}__minus__{right}` |
| product | `{left}__times__{right}` |
| datetime year | `{column}__year` |
| datetime month | `{column}__month` |
| datetime quarter | `{column}__quarter` |
| datetime dayofweek | `{column}__dayofweek` |
| datetime weekend | `{column}__is_weekend` |
| days since | `{column}__days_since__{YYYY_MM_DD}` |
| binning | `{column}__binning_candidate` |
| group aggregate | `{column}__group_aggregate_candidate` |
| target encoding | `{column}__target_encoding_candidate` |

reference date 名称将 `YYYY-MM-DD` 转为 `YYYY_MM_DD`。不做其他字符清洗或
slugification。

候选名与原 DataFrame 列名冲突时丢弃，不生成替代名、不记入
`skipped_columns`；available count 在冲突过滤后计算。候选之间同名时保留生成
顺序中的第一个。

Suggestion identity 冻结为：

```text
(name, feature_type, source_columns, parameters)
```

去重发生在 per-type 和 global budget 前。

### Priority and within-type order

`priority` 是全局类型优先级，不是最终 row number：

1. 所有 datetime types；
2. ratio；
3. difference；
4. product；
5. binning candidate；
6. group aggregate candidate；
7. target encoding candidate。

以下六种 type 均固定 `priority == 1`：`datetime_year`、`datetime_month`、
`datetime_quarter`、`datetime_dayofweek`、`datetime_is_weekend` 和
`datetime_days_since_reference`。Days-since 不是 datetime component，但属于同一
datetime priority group。

Datetime 同类按 source column order，再按 year、month、quarter、dayofweek、
is_weekend、days_since_reference。相同 priority 不改变此稳定顺序。Arithmetic
按 pair enumeration order；单列 candidate 按 source column order。

## Canonical Formula and Parameters

| Type | Formula | Parameters |
|---|---|---|
| ratio | `{left} / {right}` | `()` |
| difference | `{left} - {right}` | `()` |
| product | `{left} * {right}` | `()` |
| datetime year | `{column}.dt.year` | `()` |
| datetime month | `{column}.dt.month` | `()` |
| datetime quarter | `{column}.dt.quarter` | `()` |
| datetime dayofweek | `{column}.dt.dayofweek` | `()` |
| datetime weekend | `{column}.dt.is_weekend` | `()` |
| days since | `{reference_date} - {column}` | `(("reference_date", "YYYY-MM-DD"),)` |
| binning candidate | `None` | `(("strategy", "learned"),)` |
| group aggregate candidate | `None` | `(("strategy", "fit_on_train_only"),)` |
| target encoding candidate | `None` | `(("strategy", "fit_on_train_only"),)` |

Suggestion-only canonical fields 进一步冻结为：

- `binning_candidate`：`source_columns` 恰有一个 eligible real numeric non-bool
  column；name 为 `{column}__binning_candidate`；formula 为 `None`；parameters 为
  `(("strategy", "learned"),)`；reason/risk/requires-fit/priority 分别为
  `numeric_binning_review`、`medium`、`True`、`5`；
- `group_aggregate_candidate`：`source_columns` 恰有一个 eligible categorical
  column，表示未来需要评审的 group key；不包含 numeric value column，Task 09
  不定义 aggregate function、value column 或训练统计；name 为
  `{column}__group_aggregate_candidate`；formula 为 `None`；parameters 为
  `(("strategy", "fit_on_train_only"),)`；reason/risk/requires-fit/priority 分别为
  `categorical_group_aggregate_review`、`high`、`True`、`6`；
- `target_encoding_candidate`：只在 target 非 `None` 时生成；`source_columns`
  恰有一个 eligible categorical column 且不包含 target；name 为
  `{column}__target_encoding_candidate`；formula 为 `None`；parameters 为
  `(("strategy", "fit_on_train_only"),)`；reason/risk/requires-fit/priority 分别为
  `target_aware_encoding_review`、`high`、`True`、`7`。

## Derivation Validation

`suggestions` 必须是 `collections.abc.Sequence`。`str`、`bytes` 和 `bytearray`
显式不合法；generator/iterator 因不是 Sequence 也不合法。非法 container 抛出
`ValueError`，消息包含：

```text
suggestions must be a sequence of FeatureSuggestion
```

list、tuple 及其 empty value 合法。每项必须为 `FeatureSuggestion`，否则
`ValueError`，消息包含：

```text
suggestions must contain only FeatureSuggestion values
```

其他稳定错误：

| Condition | Exception/message contains |
|---|---|
| duplicate suggestion name | `ValueError`: `duplicate suggestion name` |
| missing source | `ValueError`: `source column not found` |
| output name already exists | `ValueError`: `derived feature name already exists` |
| non-bool copy | `ValueError`: `copy must be a boolean` |
| unsupported materialization type | `ValueError`: `unsupported feature type for Task 09 materialization` |
| invalid canonical fields | `ValueError`: `suggestion fields do not match the Task 09 contract` |

外部构造的 materializable suggestion 也必须逐字段遵守对应 feature type 的
canonical name、feature type、source order、formula、parameters、reason、risk、
requires-fit 和 priority。Source count 属于下述第 8 步 source validation，不在
第 9 步重复验证；count 不匹配仍使用统一 canonical-field 错误。

验证 precedence 保证 `requires_fit=True` 先使用 requires-fit 错误。只有通过该
检查且属于 materializable type 才执行完整 canonical validation。Suggestion-only
type 即使被外部错误设置为 `requires_fit=False`，也先在 supported materializable
type 检查处使用 unsupported-type 错误，不进入 canonical validation。

验证 precedence 冻结为：

1. DataFrame validation；
2. copy；
3. suggestions container；
4. suggestion element types；
5. duplicate suggestion names；
6. requires_fit；
7. supported feature type；
8. source columns，内部顺序固定为：
   1. source count；
   2. source existence；
   3. source dtype compatibility；
9. parameters/formula 及其余 canonical fields；
10. output name conflicts。

Source count 对 ratio/difference/product 必须恰为两个不同列，对全部六种 datetime
types 必须恰为一列；不匹配时 `ValueError` 消息包含
`suggestion fields do not match the Task 09 contract`。只有 count 合法后才检查
source existence；只有 source 存在后才检查 dtype。

Arithmetic dtype compatibility 要求两个 source 均为 real numeric non-bool，失败
使用 `arithmetic source columns must be real numeric`。Datetime dtype
compatibility 要求 source 为 timezone-naive pandas datetime dtype，失败使用
`datetime source column must have timezone-naive datetime dtype`。Dtype 通过后才
进入第 9 步 canonical validation，output collision 始终最后检查。

所有 validation 必须在任何临时计算和列写入前完成，避免 validation failure
造成部分修改。

## Materialization

`derive_features` 只允许本文列出的九个 stateless materializable types：三个
arithmetic types、五个 datetime components 和一个 days-since type。

### Arithmetic

Source dtype 必须是 real numeric non-bool，否则 `ValueError`，消息包含：

```text
arithmetic source columns must be real numeric
```

ratio、difference、product 在任何 arithmetic operation 前，两个 source 必须先
转换为 `float64` representation；pandas nullable missing 在转换时变为 `NaN`，
然后才执行运算。不得先以 integer/nullable integer dtype 做 difference/product
再转换，以免 integer overflow 产生伪有限结果。输出严格为 `float64`，原始
missing 传播为 `NaN`。

- ratio denominator 为零时为 `NaN`；计算后所有非有限值转为 `NaN`，不得产生
  正负 infinity；
- difference/product 计算后所有非有限值，包括 overflow infinity，转为
  `NaN`；
- overflow warning 不是 public contract，可局部使用 `numpy.errstate`。

### Datetime

Source 必须为 timezone-naive pandas datetime dtype。非 datetime 或
timezone-aware datetime source 均 fail-fast `ValueError`，消息包含：

```text
datetime source column must have timezone-naive datetime dtype
```

输出 dtype 固定为：

| Type | dtype |
|---|---|
| year | `Int64` |
| month | `Int64` |
| quarter | `Int64` |
| dayofweek | `Int64` |
| is_weekend | `boolean` |
| days since reference | `Int64` |

缺失 datetime 输出 `pd.NA`。

`datetime_dayofweek` 使用 pandas 语义：Monday=0、Tuesday=1、…、Sunday=6。
`datetime_is_weekend` 定义为 `dayofweek >= 5`：仅 Saturday/Sunday 为 `True`，
Monday–Friday 为 `False`；missing datetime 为 `pd.NA`。

Days-since 使用：

```text
reference_date.normalize() - source.dt.normalize()
```

输出整数日；source 晚于 reference date 时允许负数。parameters 必须恰好包含
合法 reference date；不得读取当前日期。

## Output and Copy Contract

`derive_features` 具有 transaction-like 行为。先完成全部合同 validation，再对
所有 suggestions 在临时结果结构中完成计算；只有全部计算成功后，才按
suggestions 输入顺序统一追加新列。任一 validation 或 unexpected computation
exception 都不得向输入 DataFrame 写入任何新列；合同不要求把 unexpected
internal exception 转换为新的 public exception type。

成功时保留原 index（包括重复 labels）、原列顺序、原 dtype 和原数据值。

- `copy=True`：使用 pandas `df.copy(deep=True)` 语义，返回新 DataFrame 对象，
  复制 DataFrame 数据管理结构以及 index/columns objects；对普通 numeric、
  datetime 和 extension arrays 的后续 DataFrame-level assignment 不影响输入；
  不承诺递归复制 object-dtype cells 引用的任意 Python mutable object，也不承诺
  通用 `copy.deepcopy` 语义；输入始终不修改；
- `copy=False`：调用者明确授权成功时原地修改，`result.data is df`；计算阶段
  不得逐列写入 df，全部临时结果成功后才统一赋值；任一失败时原 DataFrame 的
  columns、values 和 dtypes 保持调用前状态；
- 空 suggestions 在 `copy=True` 下返回 pandas `deep=True` 副本，在
  `copy=False` 下返回同一对象；
  applied 和 skipped 字段均为空。

`applied_suggestions` 按 suggestions 输入顺序保存成功应用的 suggestion name。

## Dependency and Regression Boundaries

Task 09 只可依赖 pandas、numpy、Task 03 schema contracts 和 `infer_schema`。
Task 07 只是 sequencing prerequisite。Task 09 不得 import/call Task 07 或 Task 08
public analysis functions，也不实现 correlation heuristic。

Task 09 不修改 I/O、schema、summary、quality、analysis、Task 05
workflow/reporting/CLI、Task 07 non-target analysis、Task 08 group/target analysis、
`pyproject.toml` entry point 或 dependency groups。Task 09 不接入 workflow、
reporting 或 CLI；完整集成留给 Task 13。

不新增 runtime/optional/dev dependency，不添加 lock file。

## Testing Contract

Task 09 实现必须覆盖并锁定：

1. exact signatures、type hints、docstrings、exports 和 `__all__`；
2. 三个 frozen dataclass 的 exact fields/order/types；
3. non-DataFrame、非字符串/重复 DataFrame columns；
4. external schema 类型与 mismatch；
5. target、exclusions、reference date、max suggestions 验证，包括 bool budget；
6. `schema=None` 只调用 `infer_schema`，不调用 Task 07/08 analysis；
7. target、explicit exclusion、identifier、all-missing、constant、unsupported dtype
   和 duplicate-content 的 exact precedence；
8. real numeric、nullable numeric、bool、complex、datetime、timedelta、object 的
   eligibility；
9. pair direction/order、per-type budgets、global 49/50/51 boundaries；
10. naming、name conflict、identity dedup、priority 和 deterministic order；
11. target-aware suggestions 不读取 target values；held-out-only category/extreme
    不影响 materialized formulas；
12. reference date 的所有允许类型、invalid/timezone-aware 拒绝、normalized
    parameter/name、无 current-date dependency；
13. suggestions container/element、duplicate name、requires-fit、unsupported type、
    source count/existence/dtype、canonical fields、output collision、copy 的
    fail-fast validation；
14. 所有 validation 在 temporary computation 和 mutation 前完成；
15. ratio/difference/product 值与 `float64`、zero/missing/infinity/overflow 行为；
16. datetime component/days-since 值与 extension dtypes、missing 和正负 day count；
17. 原 index/columns/dtypes、新列顺序、copy true/false、empty suggestions；
18. 正常结果 skipped fields 固定为空；
19. Tasks 01–08 regression tests 保持通过，且无 workflow/reporting/CLI/I/O/
    analysis、dependency 或 lock-file 变化。

Conditional-Go 修复还必须以客观断言覆盖：

1. 每列恰好属于 eligible/excluded/skipped 之一，三个 tuple 无重叠且并集为全部
   columns；
2. target 与 explicit exclusion 重叠时只进入 excluded，reason 为
   `target_column`；
3. `skipped_reasons` 同时覆盖 excluded 与 skipped，不包含 eligible；
4. duplicate comparison 只在 target、explicit exclusion、identifier-like、
   all-missing、constant、unsupported dtype 六步前置过滤后的剩余候选列中执行；
5. target、explicit exclusion、ID-like、all-missing、constant columns 不得成为
   duplicate retained representative；
6. timezone-aware datetime source 在 generation 使用 `unsupported_dtype`；
7. timezone-aware datetime external materialization 使用冻结错误；
8. dayofweek 的 Monday=0、Sunday=6，以及 weekend 仅 Saturday/Sunday；
9. Timestamp、datetime、date dispatch 顺序和 timezone-naive Timestamp
   normalization；
10. 无 eligible datetime column 时 report 仍保存 normalized reference date；
11. binning 的 canonical single numeric source 和全部字段；
12. group aggregate 只有一个 categorical group-key source，不含 value column；
13. target encoding 只有一个 categorical source，不含 target；
14. generated suggestion-only canonical fields 以及 external malformed
    suggestion-only requires-fit/unsupported precedence；
15. `str`、`bytes`、`bytearray` container 和 generator 均被拒绝，empty
    list/tuple 合法；
16. arithmetic sources 在运算前转为 float64；
17. large integer product/difference 不发生整数 overflow 后的伪有限结果；
18. `copy=False` validation 或 unexpected computation failure 不部分修改输入；
19. `copy=True` 任意失败不修改输入；
20. pandas `deep=True` 不承诺递归复制 object-cell mutable objects；
21. duplicate index 不影响结果、顺序或原子性；
22. derive contract validation 对每个 materializable type 检查全部九个字段，其中
    source count 位于第 8 步，其余 canonical field checks 位于第 9 步；
23. 所有 suggestion vocabularies 和 priority/requires-fit 映射；
24. 49/50/51 global budget 与每类 boundary；
25. Tasks 01–08 regression 保持通过且外围模块、依赖和 lock-file 无变化。

Duplicate-content 专项测试还必须精确断言：

- unsupported dtype columns 不进入 duplicate comparison，也不能成为 retained
  representative；
- 两个内容相同且 dtype 相同的 unsupported columns 均使用
  `unsupported_dtype`，后一列不得使用 `duplicate_content`；
- unsupported column 与 eligible column 内容相同时不比较，eligible column 不得
  因该 unsupported column 被跳过；
- `duplicate_content` 只适用于通过完整六步前置检查的 dtype-eligible candidates；
- skipped precedence 精确断言 `unsupported_dtype` 优先于
  `duplicate_content`。

## Verification

实现和 review 必须使用项目本地环境：

```bash
.venv/bin/python -m pytest
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
.venv/bin/python -m build
```

不得使用系统 `python3`，也不得依赖全局 `python` 命令。
