# Task 08 Group Comparison 与 Target Relationship Analysis 公共契约

## 状态

已接受。本文是 Task 08 实现前的 API 决策记录。Task 08 的实现、测试和
API 文档必须遵守本文；修改本文冻结的 public API、结果字段、表格 schema、
预算、跳过原因、统计方法、错误行为、排序或范围前，必须先同步评审本文、
`SPEC.md` 和 `IMPLEMENTATION_PLAN.md`。

## 范围

Task 名称冻结为：
**Task 08 — 分组比较与 Target Relationship Analysis**。

Task 08 只新增以下 public API：

```python
def compare_groups(
    df: pd.DataFrame,
    group_by: str,
    *,
    values: Sequence[str] | None = None,
    max_groups: int = 20,
) -> GroupComparison: ...

def analyze_target_relationships(
    df: pd.DataFrame,
    target: str,
    *,
    task: Literal["classification", "regression"],
    features: Sequence[str] | None = None,
) -> TargetAnalysis: ...
```

Task 08 只做：

1. 一个 categorical group key 对一个或多个 numeric value columns 的分组比较；
2. classification target × numeric feature 的 Kruskal-Wallis 分析；
3. classification target × categorical feature 的 Chi-square 与 Cramér's V；
4. regression target × numeric feature 的 Pearson correlation；
5. regression target × categorical feature 的 Kruskal-Wallis 分析。

Task 08 不做 visualization、feature engineering、modeling、evaluation、
report generation、workflow integration、CLI integration、Excel CLI、HTML、
dashboard、plugin system、batch processing、automatic cleaning、data mutation、
custom exceptions 或依赖变更。

Task 08 不修改 Task 05 workflow/reporting/CLI，也不修改 Task 07 的四个
non-target functions、四个 result types、DataFrame schemas、skipped reasons、
errors 或 ordering。

## 共享输入规则

两个函数遵守以下共同规则：

- `df` 必须是 pandas DataFrame；否则抛出 `ValueError`，消息包含
  `df must be a pandas DataFrame`。
- DataFrame column names 必须全部是字符串；否则抛出 `ValueError`，消息包含
  `DataFrame column names must all be strings`。
- 重复 DataFrame column names 不支持；否则抛出 `ValueError`，消息包含
  `duplicate DataFrame column names are not supported`。
- Task 08 所有需要真实数值统计的 target、value 和 feature 只接受 real numeric
  non-boolean dtype：integer、unsigned integer、floating-point、pandas nullable
  integer 和 pandas nullable floating-point。boolean、complex、datetime、
  timedelta、categorical、object、string 和其他 dtype 不视为 Task 08 real
  numeric。
- pandas object、string、category 或 boolean dtype 视为 categorical。
- datetime、timedelta 和其他 dtype 不自动强制转换为 numeric 或 categorical。
- 自动选择保持 DataFrame column order；显式选择保持 caller order。
- 不调用 `infer_schema`、`summarize_dataframe` 或 `check_data_quality`。
- 不调用任何 Task 07 public analysis function，包括
  `analyze_numeric_features`、`analyze_categorical_features`、
  `compute_correlations` 和 `detect_outliers`。
- 可以复用 `analysis.py` 中不改变 Task 07 public contract 的 private
  validation/helper functions，但不得改变 helper 的既有语义。如果复用会改变
  Task 07 API、result schema、reason codes、errors、ordering 或既有行为，必须
  新建聚焦的 private helper。
- 不修改输入 DataFrame、index、column order、dtype 或值。
- 所有输出 deterministic，不包含 timestamp、path、random ID、duration、plot、
  model 或 generated file。
- 所有统计均为探索性统计，不表示因果关系；p-values 不做 multiple-testing
  correction，也不得自动翻译为业务结论或显著性标签。

## 固定内部常量

Task 08 冻结以下内部常量：

```text
TASK08_MIN_GROUP_SIZE = 2
```

该常量不是 public API，不公开为函数参数，也不保存到 `GroupComparison` 或
`TargetAnalysis` metadata。它只适用于 classification × numeric 和
regression × categorical 的 Kruskal-Wallis group retention；不适用于
`compare_groups`、classification × categorical 或 regression × numeric。

## 公开结果类型

两个结果类型必须使用 `dataclass(frozen=True)`。容器字段和 DataFrame 本身不
保证深度不可变；调用者不应依赖 deep immutability。

### `GroupComparison`

字段顺序和类型冻结为：

```python
n_rows: int
group_by: str
requested_values: tuple[str, ...] | None
analyzed_values: tuple[str, ...]
skipped_values: tuple[str, ...]
skipped_reasons: dict[str, str]
max_groups: int
available_group_count: int
displayed_group_count: int
missing_group_count: int
truncated: bool
truncation_reason: str | None
summary: pd.DataFrame
```

规则：

- `requested_values` 在 `values is None` 时为 `None`；否则为保持 caller order
  的 tuple。
- `analyzed_values` 保持选择顺序。
- `skipped_values` 保持 request/DataFrame order。
- `available_group_count` 是 `group_by` 中 non-missing unique groups 的数量，
  在应用 `max_groups` 前计算。
- `displayed_group_count = min(available_group_count, max_groups)`。
- `missing_group_count` 是 `group_by` 为 missing 的输入行数；missing group rows
  不参与 group ranking、summary 或 value statistics。
- 当 `available_group_count > max_groups` 时，`truncated=True` 且
  `truncation_reason="exceeds_max_groups"`；否则 `truncated=False` 且
  `truncation_reason=None`。

### `TargetAnalysis`

字段顺序和类型冻结为：

```python
n_rows: int
target: str
task: str
requested_features: tuple[str, ...] | None
analyzed_features: tuple[str, ...]
skipped_features: tuple[str, ...]
skipped_reasons: dict[str, str]
max_features: int
max_categories: int
available_feature_count: int
truncated: bool
truncation_reason: str | None
numeric_details: pd.DataFrame
category_details: pd.DataFrame
statistical_tests: pd.DataFrame
limitations: tuple[str, ...]
```

规则：

- `task` 只保存规范化后的原始合法值 `"classification"` 或
  `"regression"`，不得根据 target dtype 猜测或改写。
- `requested_features` 在 `features is None` 时为 `None`；否则为保持 caller
  order 的 tuple。
- `analyzed_features` 保持分析顺序，只包含成功产生批准路径结果的 features。
- `skipped_features` 保持 request/DataFrame order。
- `available_feature_count` 是完成 feature-level dtype、all-missing、non-finite、
  constant、insufficient sample/group checks 后、应用 category budget 和
  `max_features` 前的 eligible feature 数。超过 category budget 的 feature 仍
  计入该数值；后续得到 `statistical_test_not_applicable` 的 feature 也已计入
  该数值。category budget 和 statistical outcome 均不回写该计数。
- `max_features` 固定为 `50`；`max_categories` 固定为 `20`。
- `truncated=True` 当且仅当至少一个 feature 使用
  `exceeds_max_features`，此时
  `truncation_reason="exceeds_max_features"`；否则 `truncated=False` 且
  `truncation_reason=None`。`exceeds_max_categories` 不改变 result-level
  `truncated`。
- `limitations` 使用本文冻结的封闭 vocabulary 和生成规则；实现不得自由生成
  文本。

## `GroupComparison.summary` schema

`summary` 使用 long-form，每个 analyzed value × displayed group 一行。列顺序
和 dtype 冻结为：

| Column | dtype |
|---|---|
| `value` | `object` |
| `group` | `object` |
| `group_count` | `int64` |
| `count` | `int64` |
| `missing_count` | `int64` |
| `mean` | `float64` |
| `q25` | `float64` |
| `median` | `float64` |
| `q75` | `float64` |

规则：

- `group_count` 是该 displayed group 的总行数，不依赖当前 value 是否 missing。
- `count` 是当前 group/value 的 finite non-missing count。
- `missing_count = group_count - count`。value 中的 NaN/NA 计入 missing；包含
  infinity 的 value column 整列跳过，不把 infinity 计入 missing。
- `mean`、`q25`、`median`、`q75` 只使用 finite non-missing values；quantile
  使用 pandas default interpolation。
- 如果某 displayed group 对某 value 的 `count == 0`，保留该行，四个统计值
  均为 `NaN`。
- 无 analyzed values 或无 displayed groups 时返回具有全部固定 columns 和
  dtypes 的空 DataFrame。

## `TargetAnalysis.numeric_details` schema

`numeric_details` 只承载 classification × numeric 的逐 target category 摘要。
regression × numeric 的 Pearson 结果只进入 `statistical_tests`，不在此表
重复。列顺序和 dtype 冻结为：

| Column | dtype |
|---|---|
| `feature` | `object` |
| `target_category` | `object` |
| `group_count` | `int64` |
| `count` | `int64` |
| `missing_count` | `int64` |
| `mean` | `float64` |
| `q25` | `float64` |
| `median` | `float64` |
| `q75` | `float64` |

规则：

- 缺失 target rows 在所有 target analysis 中先排除。
- classification target categories 按原 target column 的 first appearance
  排序。
- `group_count` 是当前 non-missing target category 的行数。
- `count` 是当前 feature/target category 的 finite non-missing pair count。
- `missing_count = group_count - count`。
- 统计量只使用 finite non-missing feature values。
- 每个 analyzed numeric feature × retained target category 一行。样本量小于
  `TASK08_MIN_GROUP_SIZE` 的 target categories 被排除，不出现在表中。
- classification 没有 analyzed numeric features，或 task 为 regression 时，
  返回 fixed-schema empty DataFrame。

## `TargetAnalysis.category_details` schema

`category_details` 只使用以下通用七列版本，不保留任何并行 schema：

| Column | dtype |
|---|---|
| `feature` | `object` |
| `feature_category` | `object` |
| `target_category` | `object` |
| `count` | `int64` |
| `rate` | `float64` |
| `target_mean` | `float64` |
| `target_median` | `float64` |

Classification rules：

- 输出 complete cases 中全部 retained feature category × observed target
  category 的完整笛卡尔积；target categories 是 complete cases 中实际观察到的
  categories，并按原 classification target 的 first appearance 排序。
- missing feature 或 missing target rows 不进入表。
- `count` 是该 cell 的 pair count。
- `rate` 的分母是当前 `feature_category` 下所有 non-missing target categories
  的总 count，即 target distribution within feature category。
- zero-count cells 必须保留，并固定为 `count=0`、`rate=0.0`、
  `target_mean=NaN`、`target_median=NaN`。
- target category 按 classification target 的 first appearance 排序；feature
  category 按 complete cases 中 first appearance 排序。
- `target_mean` 和 `target_median` 固定为 `NaN`。

Regression rules：

- 每个 retained categorical feature category 一行；样本量小于
  `TASK08_MIN_GROUP_SIZE` 的 categories 被排除且不出现在表中。
- `target_category` 固定为 `None`，dtype 保持 `object`。
- missing feature、missing target 或 non-finite target rows 不进入表。
- `count` 是该 feature category 的 valid target count。
- `rate = count /` 当前 feature 的全部 retained-group valid feature-target
  pairs；已删除的小组不进入分母，且分母不为零。
- `target_mean` 和 `target_median` 使用该组 finite target values。

没有 analyzed categorical features 时返回 fixed-schema empty DataFrame。

## `TargetAnalysis.statistical_tests` schema

每个成功分析的 feature 恰有一行。列顺序和 dtype 冻结为：

| Column | dtype |
|---|---|
| `feature` | `object` |
| `feature_kind` | `object` |
| `analysis` | `object` |
| `n_obs` | `int64` |
| `group_count` | `int64` |
| `statistic` | `float64` |
| `p_value` | `float64` |
| `effect_size` | `float64` |
| `effect_size_name` | `object` |
| `limitation` | `object` |

固定 vocabulary：

- `feature_kind`: `"numeric"` 或 `"categorical"`。
- `analysis`: `"kruskal_wallis"`、`"chi_square"` 或 `"pearson"`。
- `effect_size_name`: `"epsilon_squared"`、`"cramers_v"` 或
  `"absolute_pearson_r"`。
- `limitation` 只允许：
  - `"exploratory_unadjusted_p_value"`；
  - `"exploratory_unadjusted_p_value; chi_square_expected_counts_may_be_small"`。
- non-Chi-square 成功结果固定使用第一项。
- Chi-square expected frequencies 全部 `>= 5` 时使用第一项；任一 expected
  frequency `< 5` 时使用第二项。
- 不允许其他 limitation 文本，不输出 significance label。

空结果返回具有全部固定 columns 和上述 dtypes 的 empty DataFrame；不得输出
NaN statistic/p-value/effect-size row。

## `TargetAnalysis.limitations` vocabulary

只允许以下 codes，并按以下顺序：

1. `exploratory_unadjusted_p_values`
2. `chi_square_expected_counts_may_be_small`

生成规则：

- `statistical_tests` 非空时包含第一项。
- 至少一个成功输出的 Chi-square 检验存在任一 expected frequency `< 5` 时，
  再包含第二项。
- 其他情况不包含第二项。
- `statistical_tests` 为空时，`limitations=()`。
- 不允许添加其他 code 或任意文本。

## 四条 target analysis 路径

### Classification × Numeric

- 先构造 non-missing target 与 finite non-missing feature complete cases。
- target groups 按 target first-appearance order 识别。
- 删除样本量小于 `TASK08_MIN_GROUP_SIZE` 的 groups；保留其余 groups 继续
  分析。被删除 groups 不出现在 `numeric_details`。
- retained groups 少于 2 时，整个 feature 使用 `insufficient_groups`。
- `n_obs` 是 retained groups 中用于检验的 complete-case 总样本量，
  `group_count` 是 retained groups 数。
- 使用 `scipy.stats.kruskal` 的默认 tie correction 计算 Kruskal-Wallis H 和
  p-value。
- effect size 使用 epsilon squared：

  ```text
  epsilon_squared = max(0.0, (H - k + 1) / (n - k))
  ```

  其中 `H=statistic`、`k=retained group_count`、`n=retained n_obs`。`n <= k`
  时不运行检验，feature 使用 `insufficient_groups`。

### Classification × Categorical

- 使用 feature 与 target 均 non-missing 的 pairs。
- feature 和 target 均须至少两个 observed categories；contingency table 至少
  2×2，`n_obs >= 4`，且不得有 zero-marginal row/column。
- 使用 `scipy.stats.chi2_contingency` 默认参数；2×2 table 保留 SciPy 默认
  Yates correction 行为。
- `statistic` 是 Chi-square statistic，`p_value` 是对应 p-value，
  `group_count` 是 feature categories 数。
- effect size 使用未做 bias correction 的 Cramér's V：

  ```text
  cramers_v = sqrt(chi2 / (n * min(r - 1, c - 1)))
  ```

  其中 `r`、`c` 是 contingency table shape。分母为零时不运行检验并使用
  `insufficient_groups`。
- 如果 statistic、p-value 或 Cramér's V 非有限，按本文统一 SciPy 非有限结果
  规则跳过 feature。

### Regression × Numeric

- target 必须为 numeric non-boolean dtype。
- 使用 target 与 feature 均 finite non-missing 的 pairs。
- 至少三个 pairs，且 pair-filtered target 和 feature 均不得 constant。
- 使用 `scipy.stats.pearsonr` 计算 Pearson `r` 和 two-sided p-value。
- `statistic` 保存带方向的 `r`，`group_count` 固定为 `0`。
- `effect_size = abs(r)`，`effect_size_name="absolute_pearson_r"`。
- 如果 `r`、p-value 或 effect size 非有限，按本文统一 SciPy 非有限结果规则
  跳过 feature。

### Regression × Categorical

- 先构造 non-missing feature category 与 finite non-missing target complete
  cases。
- feature categories 按 complete-case first-appearance order 识别。
- 删除样本量小于 `TASK08_MIN_GROUP_SIZE` 的 groups；保留其余 groups 继续
  分析。被删除 groups 不出现在 `category_details`。
- retained groups 少于 2 时，整个 feature 使用 `insufficient_groups`。
- `n_obs` 是 retained groups 中用于检验的 complete-case 总样本量，
  `group_count` 是 retained groups 数。
- 使用 `scipy.stats.kruskal` 默认 tie correction。
- effect size 使用与 classification × numeric 相同的 epsilon squared 公式，
  其中 `n` 和 `k` 使用 retained observations 和 retained groups。

### 统一 SciPy 非有限结果规则

四条 statistical paths 在完成各自前置 checks 后，如果 SciPy 返回的
statistic 或 p-value 不是有限数，或者 effect size 计算结果不是有限数：

- 不输出 `statistical_tests` row；
- 不输出该 feature 的 detail rows；
- 整个 feature 使用 `statistical_test_not_applicable`；
- 一个 feature 只记录这一个 skipped reason。

## Column selection

### `compare_groups`

- `group_by` 只接受一个字符串列名；该列必须是 categorical dtype。
- `values=None` 自动选择除 `group_by` 外全部 real numeric non-boolean columns，
  保持 DataFrame order；complex columns 不进入自动选择。
- 显式 `values` 必须全部是 real numeric non-boolean columns。complex 或其他
  不适用列作为函数级参数错误拒绝，不静默跳过。
- `group_by` 不得同时出现在 `values`。
- 无 numeric values 是合法结果，返回 empty summary 和对应空 tuples。

### `analyze_target_relationships`

- `target` 必须由 caller 显式提供；schema target candidates 不触发本函数。
- `features=None` 自动选择除 target 外所有 numeric 或 categorical columns，
  保持 DataFrame order。其他 dtype 不进入自动选择，也不记录 skipped reason。
- 显式 features 可以包含其他 dtype；它们使用 `unsupported_dtype` 跳过。
- target 不得出现在显式 features 中。
- classification target 只允许 object、pandas string、category、bool，或
  low-cardinality numeric non-boolean dtype。datetime、timedelta、complex 和
  其他 dtype 是函数级参数错误。
- numeric classification target 仍按原始 observed values 作为 categories，
  不做 binning 或 coercion，且不得包含 positive/negative infinity。
- classification target 必须有 2 至 20 个 non-missing observed categories。
- regression target 必须是 real numeric non-boolean dtype，至少三个 finite
  non-missing values，且不得 constant。complex target 使用 regression dtype
  mismatch 的既有稳定错误。

## Budgets

### Group budget

- `max_groups` 默认 `20`，必须是非 bool 的 `int` 且 `>= 1`。
- group ranking 使用 non-missing group row frequency descending；ties 按
  group value 在原 `group_by` column 中 first appearance 打破。
- 只保留 ranking 中前 `max_groups` groups；不创建 `other` bucket。
- 被截断的 group rows 不进入 summary，截断通过 result counts、`truncated`
  和 `truncation_reason` 披露。

### Category budget

- `max_categories` 固定为每个 categorical feature 最多 20 个 complete-case
  observed categories。先删除 target missing 和 feature missing，再在剩余
  feature values 上计算 non-missing unique category count。
- classification target 最多 20 个 non-missing categories。
- 不对 categories 做 top-N 截断后再检验，因为这会改变统计总体。
- 超过 feature category budget 的 feature 整体使用
  `exceeds_max_categories` 跳过；超过 target category budget 是 target 参数
  错误。
- category budget 不创建 `other` bucket，不改变 `available_feature_count`，也
  不改变 result-level `truncated`。

### Feature budget

- `max_features` 固定为 50，不能通过 Task 08 public API 修改。
- 先完成所有 eligibility checks，再按 request/DataFrame order 保留前 50 个
  eligible features。
- 其余 eligible features 使用 `exceeds_max_features`，并披露 result-level
  truncation metadata。

## Skipped reason vocabulary

Task 08 只允许以下 skipped reason codes：

- `unsupported_dtype`
- `all_missing`
- `non_finite_values`
- `insufficient_non_missing`
- `constant`
- `insufficient_groups`
- `exceeds_max_categories`
- `statistical_test_not_applicable`
- `exceeds_max_features`

`GroupComparison` 允许：`all_missing`、`non_finite_values`、
`insufficient_non_missing`、`constant`。

`TargetAnalysis` 允许全部九个 codes。不得创建其他 skipped reason code，
也不得把稳定参数错误降级为 skipped reason。

## Skipped reason precedence

同一 value/feature 只允许一个 reason。

### Group comparison numeric values

1. `all_missing`
2. `non_finite_values`
3. `insufficient_non_missing`
4. `constant`

`insufficient_non_missing` 表示所有 displayed groups 合计的 finite non-missing
value count 少于 2。constant 在全部 displayed-group finite values 上判断。

### Classification × Numeric

1. `unsupported_dtype`
2. `all_missing`
3. `non_finite_values`
4. `insufficient_non_missing`
5. `constant`
6. `insufficient_groups`
7. `statistical_test_not_applicable`
8. `exceeds_max_features`

### Classification × Categorical

1. `unsupported_dtype`
2. `all_missing`
3. `insufficient_non_missing`
4. `constant`
5. `exceeds_max_categories`
6. `insufficient_groups`
7. `statistical_test_not_applicable`
8. `exceeds_max_features`

### Regression × Numeric

1. `unsupported_dtype`
2. `all_missing`
3. `non_finite_values`
4. `insufficient_non_missing`
5. `constant`
6. `statistical_test_not_applicable`
7. `exceeds_max_features`

### Regression × Categorical

1. `unsupported_dtype`
2. `all_missing`
3. `insufficient_non_missing`
4. `constant`
5. `exceeds_max_categories`
6. `insufficient_groups`
7. `statistical_test_not_applicable`
8. `exceeds_max_features`

Rules：

- `all_missing` 在 target 已过滤后、当前 feature 没有 non-missing observation
  时使用。
- numeric feature 出现任一 positive/negative infinity 时整列使用
  `non_finite_values`，不局部删除 infinity 后继续检验。
- `insufficient_non_missing` 按对应路径的最小 pair count 判断。
- numeric constant 在路径所使用的 finite pair-filtered feature values 上判断。
- categorical constant 在路径所使用的 non-missing pair-filtered feature
  categories 上判断。
- group/category applicability checks 在 feature budget 前执行；budget reason
  永远是最后 precedence。
- `statistical_test_not_applicable` 只在 statistical eligibility/group checks
  成功后、SciPy statistic/p-value 或 effect size 非有限时使用，并位于
  `exceeds_max_features` 之前。

## Missing、constant、infinity 与 sample behavior

- missing target rows 在 target analysis 的所有路径中排除，并通过每个表的
  counts/n_obs 体现；不把 missing target 当作 category。
- missing group rows 在 group comparison 中排除，并通过
  `missing_group_count` 披露。
- missing numeric/categorical feature values 不进入统计检验；分类 numeric
  summary 仍通过 `missing_count` 披露。
- all-missing feature/value 不产生伪造统计量。
- classification target all-missing、单类别或超过 20 类是函数级错误。
- regression target missing rows 可排除；all-missing、non-numeric、包含
  infinity、有效样本少于 3 或 constant 是函数级错误。
- numeric feature 中 infinity 是 feature/value 级 skipped reason；不得把
  infinity 当作 missing。
- 不足最小样本或 group requirements 时跳过该 feature，不返回 NaN 检验行。
- classification target validation order 固定为：supported dtype、numeric
  infinity、non-missing availability、minimum class count、maximum class count。
- regression target validation order 固定为：numeric non-boolean dtype、
  infinity、minimum finite count、constant。

## Deterministic ordering

- Explicit values/features 按 caller order；auto selection 按 DataFrame order。
- Skipped values/features 按 request/DataFrame order。
- Group ranking 按 frequency descending，ties 按 first appearance。
- `GroupComparison.summary` 按 analyzed value order，再按 displayed group
  ranking order。
- classification target categories 按 target column first appearance。
- categorical feature categories 按各 feature column first appearance。
- `numeric_details` 按 analyzed feature order，再按 target category order。
- classification `category_details` 按 analyzed feature order、feature category
  order、target category order。
- regression `category_details` 按 analyzed feature order、feature category
  order。
- `statistical_tests` 按 analyzed feature order。
- 不按 p-value、statistic 或 effect size 排序。
- 不按 column/category 字母序排序，不依赖 hash iteration order。

## Error behavior

只使用 built-in exceptions，不创建 custom exceptions。稳定错误如下：

### Shared

- non-DataFrame `df`: `ValueError`，消息包含
  `df must be a pandas DataFrame`。
- non-string DataFrame columns: `ValueError`，消息包含
  `DataFrame column names must all be strings`。
- duplicate DataFrame columns: `ValueError`，消息包含
  `duplicate DataFrame column names are not supported`。

### `compare_groups`

- non-string `group_by`: `ValueError`，消息包含
  `group_by must be a string`。
- missing group column: `ValueError`，消息包含 `group column not found`。
- non-categorical group column: `ValueError`，消息包含
  `group_by must be categorical`。
- group column 无 non-missing value: `ValueError`，消息包含
  `group_by must contain at least one non-missing value`。
- non-string values member: `ValueError`，消息包含
  `values must contain only strings`。
- duplicate values member: `ValueError`，消息包含
  `duplicate value column parameter`。
- missing value column: `ValueError`，消息包含 `value column not found`。
- explicit non-real-numeric value，包括 complex: `ValueError`，消息包含
  `values must contain only numeric columns`。
- `group_by` included in values: `ValueError`，消息包含
  `group_by must not appear in values`。
- invalid `max_groups`: `ValueError`，消息包含
  `max_groups must be an integer >= 1`。

### `analyze_target_relationships`

- non-string target: `ValueError`，消息包含 `target must be a string`。
- missing target: `ValueError`，消息包含 `target column not found`。
- invalid task: `ValueError`，消息包含
  `task must be classification or regression`。
- non-string features member: `ValueError`，消息包含
  `features must contain only strings`。
- duplicate features member: `ValueError`，消息包含
  `duplicate feature column parameter`。
- missing feature: `ValueError`，消息包含 `feature column not found`。
- target included in features: `ValueError`，消息包含
  `target must not appear in features`。
- all-missing classification target: `ValueError`，消息包含
  `classification target must contain non-missing values`。
- classification target 少于两类: `ValueError`，消息包含
  `classification target must contain at least two classes`。
- unsupported classification target dtype: `ValueError`，消息包含
  `classification target must be categorical or low-cardinality numeric`。
- numeric classification target contains infinity: `ValueError`，消息包含
  `classification target must contain only finite values`。
- classification target 超过 20 类: `ValueError`，消息包含
  `classification target must contain at most 20 classes`。
- non-numeric regression target: `ValueError`，消息包含
  `regression target must be numeric`。
- regression target contains infinity: `ValueError`，消息包含
  `regression target must contain only finite non-missing values`。
- regression target 少于三个 finite values: `ValueError`，消息包含
  `regression target must contain at least three finite values`。
- constant regression target: `ValueError`，消息包含
  `regression target must not be constant`。

## Public API 与 `__all__`

Task 08 implementation must export：

- `GroupComparison`
- `TargetAnalysis`
- `compare_groups`
- `analyze_target_relationships`

这些符号必须加入 `src/sharper/__init__.py` 和 `__all__`。不得删除或改变
Tasks 01-07 已有 public API。

## 测试合同

Task 08 implementation tests must cover：

1. 两个 public signatures；
2. 四个 public exports；
3. 两个 dataclass 的 frozen behavior、字段顺序和 type annotations；
4. 所有 output tables 的固定 columns、顺序和 dtypes，包括 empty results；
5. shared non-DataFrame、非字符串和重复 DataFrame column errors；
6. input DataFrame、index、columns、dtypes 和 values 不变；
7. group auto/explicit value selection 与 ordering；
8. invalid/missing/multi-key/non-categorical group rejection；
9. group count、missing count、mean、quartiles 与 pandas baseline；
10. missing group exclusion 与 `missing_group_count`；
11. group frequency truncation、first-appearance tie break、metadata 和空结果；
12. invalid max_groups，包括 bool、0、负数和非 int；
13. group value all-missing、infinity、insufficient 和 constant precedence；
14. classification target validation、first-appearance category ordering 和
    20-category boundary；
15. classification × numeric Kruskal-Wallis statistic/p-value 与 SciPy baseline；
16. classification × numeric epsilon squared hand calculation；
17. classification numeric detail counts、missing 与 quantiles；
18. classification × categorical contingency counts、`rate` denominator、
    Chi-square 和 SciPy default 2×2 correction；
19. Cramér's V hand calculation 与 expected-count limitation；
20. regression target validation：dtype、missing、infinity、sample 和 constant；
21. regression × numeric Pearson r/p-value 与 SciPy baseline；
22. `absolute_pearson_r` effect size；
23. regression × categorical target count/rate/mean/median；
24. regression × categorical Kruskal-Wallis 与 epsilon squared；
25. all-missing、constant、infinity、insufficient samples/groups 和 unsupported
    dtype 的每条 skipped reason；
26. 每条 skipped precedence 的重叠 case；
27. 20-category feature boundary 和 `exceeds_max_categories`；
28. 50-feature boundary、`exceeds_max_features`、result truncation metadata；
29. explicit/auto feature ordering、category ordering、detail/test row ordering；
30. classification/regression 不通过 target dtype 自动切换 task；
31. limitations 封闭 vocabulary、固定顺序且没有因果或自动显著性结论；
32. `category_details` 精确等于冻结的通用七列且没有额外列；
33. Tasks 01-07 regression suite 仍通过；
34. spy/monkeypatch 证明不调用 I/O、schema、summary、quality、workflow、
    reporting、CLI 或 modeling APIs。
35. `TASK08_MIN_GROUP_SIZE=2` 是非 public、非参数、非 result metadata 的固定
    内部常量，且只用于两条 Kruskal group-retention 路径；
36. classification × numeric 删除 size `< 2` groups 后继续，以及 retained
    groups 少于 2 时使用 `insufficient_groups`；
37. regression × categorical 删除 size `< 2` groups 后继续，以及 retained
    groups 少于 2 时使用 `insufficient_groups`；
38. 两条 Kruskal 路径的 details、`n_obs`、`group_count` 和 epsilon-squared
    只使用 retained data；
39. classification datetime、timedelta、complex 和其他 unsupported target
    dtype 使用冻结错误；
40. categorical feature budget 基于 complete cases，超过 20 整列跳过且不改变
    `available_feature_count` 或 result-level `truncated`；
41. classification categorical zero-count Cartesian cells 保留，且
    count/rate/NaN fields 精确；
42. mocked non-finite Kruskal、Chi-square、Pearson 和 effect-size results 使用
    `statistical_test_not_applicable` 且不产生 output rows；
43. result-level `limitations` vocabulary/顺序/空结果和
    `statistical_tests.limitation` 精确文本；
44. spy/monkeypatch 证明不调用四个 Task 07 public analysis functions。
45. Task 08 real-numeric predicate 接受 integer、unsigned integer、float 和 pandas
    nullable integer/float，拒绝 bool、complex 和其他非 real numeric dtype；
46. `compare_groups` 显式 complex value 使用冻结 `ValueError`，auto selection
    排除 complex；
47. classification/regression complex features 使用 `unsupported_dtype`，且不调用
    Kruskal-Wallis、Chi-square 或 Pearson；
48. classification complex target 使用 classification dtype mismatch 错误，
    regression complex target 使用 `regression target must be numeric`；
49. bool group、20/21 complete-case category boundary、前置 skip 后第 50/51 个
    eligible feature、duplicate index、mixed hashable categories 和 non-2×2
    Chi-square regression cases。

Task 08 专用 real-numeric 规则不得改变 Task 07 `_is_numeric_non_bool` 或四个
Task 07 public analysis functions 的既有 numeric dtype 行为。

数值比较必须使用明确 tolerance，不以 coverage percentage 代替统计合同证据。

## Verification commands

Task 08 implementation and review must use：

```bash
.venv/bin/python -m pytest
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
.venv/bin/python -m build
```

不得使用系统 `python3`，不得依赖 `python` command 存在。

## Documentation sync

Task 08 implementation 完成时更新 `docs/api.md`，说明两个 public functions、
两个 result types、固定表 schema、预算、统计限制、errors 和 ordering。

README 只可说明 Task 08 独立 Python API 已完成；不得声称 Task 08 已进入
`run_analysis`、Markdown report 或 CLI。完整 workflow/reporting/CLI integration
仍留给 Task 13。

## 明确推迟

以下能力推迟到后续 tasks：

- Task 08 results 的 visualization：Task 10；
- classification/regression modeling and evaluation：Tasks 11-12；
- workflow、reporting、HTML、Excel CLI 和 full CLI integration：Task 13；
- multiple group keys、pivot DSL、category automatic merging、post-hoc pairwise
  tests、multiple-testing correction、causal inference 和 automatic conclusions：
  不属于 Task 08 v0.1。
