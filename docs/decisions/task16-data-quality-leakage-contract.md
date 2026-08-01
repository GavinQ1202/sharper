# Task 16 Data Quality and Leakage Audit 精确合同

## 1. 状态、身份与权威边界

**状态：Approved — Go。**

**批准记录：** Full contract review：`No-Go`；targeted contract fixes：`complete`；bounded
contract closure：`Go`；P0：`0`；P1：`0`；P2：`0`。Contract：`Approved — Go`；
Implementation：`Implementation complete — review Go`。

**实现审查记录：** Full implementation review：`No-Go`；targeted implementation fixes：
`complete`；bounded implementation closure：`Go`；P0：`0`；P1：`0`；P2：`0`。

Task 16 合同与 implementation review 阶段已完成；不得再次进行开放式 Task 16 full contract
或 full implementation review。Task 16 尚未 commit 或发布，Task 17 尚未开始，v0.2 尚未完成
或发布；当前 package version 仍为 `0.1.0`。

**正式名称：** Task 16 — Data Quality and Leakage Audit。

**基线：** commit `4ed300695a62146832c9b45c5b666fbdfa9f0e8f`；v0.1.0 已完成，
v0.2 roadmap 为 Approved — Go，Task 15 为 Implementation complete — review Go，
Task 16 implementation 已完成且 review 为 `Go`，Tasks 17--20 implementation 均未开始，
当前 package version 仍为 `0.1.0`。

本合同服从根目录 `AGENTS.md`、`SPEC.md`、`IMPLEMENTATION_PLAN.md`、
`docs/decisions/v02-roadmap-contract.md`、Tasks 03/04 frozen schema/quality contracts 和
Task 15 frozen risk-validation contract。冲突时不得在实现中自行解释或扩张，必须先
修订并 review 治理文件和本合同。

## 2. 目标、MVP 与明确排除

Task 16 只执行：

```text
detect -> measure -> classify -> report
```

它唯一拥有：

1. opt-in data-quality、schema、special-value 和 input-profile audit，包括caller-declared
   score/fold/action/policy/cost/exposure/constraint/window/horizon provenance；
2. single-dataset missingness profiling；
3. reference/current missingness drift 与 schema drift；
4. suspected identifier、target proxy、partition/group/time/point-in-time leakage evidence；
5. Tasks 17/18 唯一共享的 private closed condition-evaluation kernel；
6. 有界、确定、无原始值的 findings、evidence、provenance、warnings 和 limitations。

Task 16 不自动 drop、impute、encode、repair、rename、clip、deduplicate、repartition、
rewrite rules 或生成修复后的 DataFrame。它不拥有或实现：

- Task 15 estimator fit、fold/OOF construction、positive-label inference、metrics、
  calibration、threshold 或 business arithmetic；
- Task 17 eligibility rule objects、priority、actions、cutoffs、constraints、policy
  simulation 或 backtest；
- Task 18 signals、alerts、episodes、warning policy 或 lifecycle metrics；
- Task 19 explanation、comparison、prediction drift 或 governance inventory；
- Task 20 workflow、report、CLI、JSON parsing 或 release integration；
- WOE、target encoding、supervised binning、feature selector/optimizer、模型训练或解释；
- 反欺诈、身份核验、设备指纹、IP、velocity、network rules 或 fraud finding；
- 通用规则引擎、public expression DSL、任意代码执行、实时服务或生产决策执行。

`suspected_identifier` 只表示结构性建模风险，不表示身份真实性、欺诈或违法行为。
association、distribution difference、high cardinality 或 high correlation 不自动等于
leakage 或 causation；每项 advisory evidence 都必须携带 limitation。

## 3. 唯一模块方案与 dependency direction

### 3.1 Proposed directory tree

```text
src/sharper/
├── data_audit.py             # Task 16 public audit owner
└── _condition_kernel.py      # Tasks 17/18 future internal shared truth owner

tests/
├── test_data_audit.py
└── test_condition_kernel.py
```

不得新建 `risk.py`、`audit_manager.py`、`rules.py`、`utils.py`、registry、plugin 或
framework 层；不得扩张 `analysis.py`。

### 3.2 Module responsibilities

| Module | Owns | May depend on | Must not do |
|---|---|---|---|
| `sharper.data_audit` | public dataclasses/function、role/config validation、schema/quality/input/missingness/leakage orchestration、reference/current comparison、result assembly | pandas、NumPy、Task 03 public `infer_schema`、Task 16 private `_condition_kernel._evaluate_atomic_condition`；只读消费 Task 15 built-in type-aware positive-label matching语义 | 修改 Task 03/04/15 behavior；fit model；构造 fold/OOF；独立实现 comparison/missing/type/timezone 语义；执行 condition tree/policy/action/alert；import workflow/reporting/CLI/visualization |
| `sharper._condition_kernel` | private condition representation validation、closed operators、tri-state truth、Boolean composition、missing/type/effective-time semantics 和 budgets | Python stdlib、pandas、NumPy | 成为 public API/DSL；理解 audit finding、rule priority、action、alert、cost、constraint；文件/network/plugin/code execution |

依赖方向冻结为：

```text
schema -> data_audit
_condition_kernel -> data_audit / future Task 17 / future Task 18
data_audit result -> future Task 19 / future Task 20
```

`_condition_kernel.py` 是 equality、ordering、membership、between、missing、type
compatibility、nonfinite、datetime/timezone 和 tri-state comparison 的唯一 truth owner。
`data_audit.py` 必须把每个 `ColumnAuditRule` comparison 委托给 private
`_evaluate_atomic_condition`；它只规范化封闭 request、为 monotonic audit 按 row position
构造相邻 pair、聚合 truth/status/reason、计算 count/rate 和组装 finding，不得复制任何
comparison、missing、type 或 timezone 逻辑。Task 17/18 后续只 import private kernel，不能
复制 evaluator。Task 19 只消费 frozen Task 16 result，不得重新扫描 raw data 计算
missingness/input profiles。Task 16 不提前创建 Task 17/18 domain objects。

该唯一ownership约束适用于audit rule和row-wise leakage/time comparisons。通过whitelist后的
dataset structural counting（duplicate rows、unique/cardinality、safe scalar `(exact type,
value)` grouping）不是condition operator，也不得扩展operator inventory或产生tri-state truth；
任何需要逐row true/false/unknown的审计仍必须调用atomic kernel。

每个 side 在DataFrame/column-label validation后、任何`infer_schema`/`duplicated`/`isna`/
comparison前，先按第12.3节以row/column position扫描index scalars、全部object/category cells和categories，
执行non-dispatching exact scalar whitelist；unsupported object立即稳定失败且不得调用其
dunder。通过后exact调用一次`infer_schema(side)`，不传target、不
修改 Task 03 id/name inference。`logical_type` 只消费其 frozen result；Task 16 的
`suspected_identifier` 另按第 8 节纯结构阈值计算，绝不把 Task 03 name token 证据升级为
leakage/fraud。Task 16 不调用 `summarize_dataframe` 或 `check_data_quality`，避免改变或
包装 v0.1 frozen result；相同基础计数必须通过 directed compatibility tests 对齐。

## 4. 唯一执行模型

Task 16 只有一个 public 入口，同时覆盖单表和可选 reference/current comparison：

```python
def audit_data_quality(
    data: pd.DataFrame,
    *,
    reference: pd.DataFrame | None = None,
    roles: DataAuditRoles | None = None,
    config: DataAuditConfig | None = None,
) -> DataAuditResult: ...
```

- `data` 始终命名为 `current` side；`reference=None` 表示 single-dataset audit；
- 不提供第二个 drift 入口、method、builder、session 或 implicit global reference；
- 两侧 row identity 都是各自从 `0` 开始的 row position；pandas index 从不用于 join、
  alignment、sampling 或 finding identity；
- duplicate index 安全且作为结构 evidence 报告，不改变 position alignment；
- 列 identity 是 exact string label；列顺序保留输入顺序；不得 strip、case-fold、rename
  或转换列名；
- duplicate column labels 和任一 non-string column label 在分析前稳定失败；
- `data`、`reference`、roles、config 和 rule containers 均不得被修改；
- 不随机 sampling。所有 row samples 是排序后的最小 row positions，并受 budget 限制；
- result 不保留任一 input/reference/current DataFrame、view、修复后 DataFrame、原始 cell
  value、index label、category/target label、special/range/membership literal、文件路径、
  动态时间戳或随机 ID；它只包含本合同明确批准的有界 result DataFrames、sanitized
  provenance、deterministic config fingerprint 和 row-position samples。Fingerprint 是批准的
  provenance 字段，不是 raw-data hash。

### 4.1 Profile column set

- `roles.features is None`：profile 所有不在 `excluded_columns` 的列；这只定义 profile
  范围，不把列推断为 feature role；
- `roles.features == ()`：显式 zero-feature dataset；仍 profile所有显式role/audit-input
  columns和dataset structure；missing-pattern按第7.7节产生empty pattern，collinearity/proxy
  calculations为空；
- 非空 `features`：missing patterns、proxy和collinearity只使用该tuple顺序；column/numeric/
  categorical profile column set是按current column order排列的`features`与所有显式
  score/cost/exposure/constraint/scalar role/availability/post-outcome columns的去重union；
- `excluded_columns` 不进入 column/numeric/categorical/missing-pattern/collinearity/
  proxy tables，但仍计入 dataset `n_columns`；
- reference side 以 exact column label 匹配。reference 缺少 current feature 或出现新增列
  是 structured schema drift，不做 pandas 隐式 alignment。

## 5. Public API 与 frozen dataclasses

Task 16 只新增以下五个顶层 public symbols，`sharper.__all__` 在现有 Task 15 exports
之后按此顺序追加：

```text
DataAuditRoles
ColumnAuditRule
DataAuditConfig
DataAuditResult
audit_data_quality
```

不导出 type alias、condition type、operator、truth enum、private helper 或 exception。

最小 public workflow 冻结为：

```python
from sharper import DataAuditRoles, audit_data_quality

result = audit_data_quality(
    current,
    reference=reference,
    roles=DataAuditRoles(
        target="outcome",
        features=("income", "balance"),
        observation_time="observed_at",
        feature_available_time_map=(("balance", "balance_available_at"),),
    ),
)
```

调用只返回 evidence；不会修改 `current`/`reference` 或生成 cleaned data。

### 5.1 `DataAuditRoles`

```python
@dataclass(frozen=True)
class DataAuditRoles:
    target: str | None = None
    features: tuple[str, ...] | None = None
    score_columns: tuple[str, ...] = ()
    excluded_columns: tuple[str, ...] = ()
    row_identifier: str | None = None
    group: str | None = None
    partition: str | None = None
    fold: str | None = None
    selection: str | None = None
    historical_action: str | None = None
    historical_policy: str | None = None
    cost_columns: tuple[str, ...] = ()
    exposure_columns: tuple[str, ...] = ()
    constraint_input_columns: tuple[str, ...] = ()
    observation_time: str | None = None
    event_time: str | None = None
    shared_feature_available_time: str | None = None
    feature_available_time_map: tuple[tuple[str, str], ...] = ()
    label_available_time: str | None = None
    outcome_end_time: str | None = None
    partition_cutoff: str | None = None
    window_start: str | None = None
    window_end: str | None = None
    horizon_end: str | None = None
    analysis_as_of: str | None = None
    post_outcome_columns: tuple[str, ...] = ()
```

`selection` 是 caller-declared historical selection/action-assignment slice，只用于 outcome
support evidence；Task 16 不解释其 values 为 approve/decline/action。`event_time` 是可选
observed event timestamp，不替代 outcome end 或 label availability。

`score_columns` 只声明 caller 已有 score provenance：审计存在性、dtype、missing、finite、
caller rule range、fold/partition provenance completeness 和 window/horizon time provenance；
不重算 score、不判断模型优劣、不计算 Task 15 metrics，也不把 ranking score 当 probability。
`fold` 只表示 caller-provided fold identity：审计 missing、identity、row/group/identifier 跨
fold overlap 及其与 partition provenance 的一致性；不构造/修改 fold、执行 Task 15
validation 或计算 OOF。`historical_action`/`historical_policy` 只审计存在、missing、
cardinality 及 partition/window availability；不解释 action、模拟 policy 或生成 rule。
`cost_columns`、`exposure_columns` 和 `constraint_input_columns` 只审计存在、dtype、missing、
finite 和 time availability；exposure 另报告 negative/nonfinite evidence，但不计算 profit、
优化 constraint 或执行策略。

`shared_feature_available_time` 与 `feature_available_time_map` 是互斥声明。前者适用于全部
audited features；后者每项是 `(feature_column, availability_time_column)`，按 caller 顺序
冻结且 feature 不重复。`window_start`、`window_end`、`horizon_end`、`analysis_as_of` 只提供
chronology/maturity metadata audit，不重做 Task 15 label maturity。没有任何列名 guessing，
不识别 `id`、`target`、`date`、`fold` 或信用领域字段。

Column profile 的 `role` 不输出 nested collection。命中多个 role 时按固定优先级
`feature,score,target,row_identifier,group,partition,fold,selection,historical_action,
historical_policy,cost,exposure,constraint_input,observation_time,event_time,
feature_available_time,label_available_time,outcome_end_time,partition_cutoff,window_start,
window_end,horizon_end,analysis_as_of,post_outcome` 以 `|` 连接；未命中
role 时为 `unassigned`。

### 5.2 `ColumnAuditRule`

```python
@dataclass(frozen=True)
class ColumnAuditRule:
    column: str
    minimum: object | None = None
    maximum: object | None = None
    minimum_inclusive: bool = True
    maximum_inclusive: bool = True
    allowed_values: tuple[object, ...] = ()
    special_values: tuple[object, ...] = ()
    not_after_columns: tuple[str, ...] = ()
    nondecreasing: bool = False
```

这是封闭的 audit declaration，不是 private condition kernel 或通用 DSL：只允许 per-
column range、allowed-values、caller-declared special-values、`column <= other_column`
和按 row position 的 nondecreasing check。空 tuple表示该membership rule未声明，不会向kernel
提交empty membership request。每个实际comparison必须委托private atomic kernel。不得接受
callable、mapping、expression、regex、function/module path、template或`$ref`。

### 5.3 `DataAuditConfig`

字段和默认值按以下顺序冻结：

```python
@dataclass(frozen=True)
class DataAuditConfig:
    positive_label: object | None = None
    missing_warning_rate: float = 0.40
    near_constant_rate: float = 0.95
    high_cardinality_count: int = 50
    high_cardinality_rate: float = 0.50
    identifier_rate: float = 0.98
    identifier_min_non_missing: int = 20
    rare_class_count: int = 20
    rare_class_rate: float = 0.05
    proxy_min_support: int = 20
    near_copy_rate: float = 0.99
    collinearity_threshold: float = 0.999
    collinearity_min_periods: int = 20
    missingness_drift_absolute_threshold: float = 0.10
    missingness_drift_relative_threshold: float = 0.50
    minimum_drift_rows: int = 30
    partition_target_rate_shift_threshold: float | None = None
    partition_target_min_support: int = 30
    max_columns: int = 500
    max_missing_patterns: int = 100
    max_finding_samples: int = 20
    max_unique_inspection_rows: int = 100_000
    max_category_levels: int = 1_000
    max_collinearity_columns: int = 50
    duplicate_scan_row_limit: int = 1_000_000
    max_column_rules: int = 100
    column_rules: tuple[ColumnAuditRule, ...] = ()
```

所有 defaults 是稳定通用 audit policy，不按数据自动优化、不按行业变化。调用者可显式
覆盖，但 result 只能保留第 7.13 节冻结的 sanitized provenance 和 fingerprint，不得保留
config/roles/rule/literal 对象。`positive_label` 只在 target 已声明时合法；先通过Task 16
exact scalar whitelist，再对批准的NumPy bool/int/float执行与Task 15一致的normalization并
使用built-in type-aware exact matching；不执行新的正类 inference。Task 16安全边界不接受
`np.str_`或其他未批准scalar，这不改变Task 15自身public behavior。Target 已声明但
`positive_label=None` 时仍生成
class profile，但 event-rate field 为 unavailable。

Config validation 唯一冻结为：所有 rate/threshold float 必须是 finite built-in/NumPy
real（bool 排除）；missing/near/high-cardinality/identifier/rare/near-copy rates 在
`[0, 1]`，其中 `near_constant_rate`、`identifier_rate`、`near_copy_rate` 必须 `>0`；
collinearity threshold 在 `(0, 1]`；drift absolute/relative threshold 在 `[0, +inf)`；
`partition_target_rate_shift_threshold` 为 `None` 或 finite real `[0, 1]`；
所有 count/minimum/budget fields 必须是 exact positive integer，但
`max_finding_samples`、`max_column_rules` 可为 0。第 14 节另列的上下限同时适用。

### 5.4 `DataAuditResult`

```python
@dataclass(frozen=True)
class DataAuditResult:
    config_fingerprint: str
    n_rows: int
    n_columns: int
    reference_n_rows: int | None
    reference_n_columns: int | None
    dataset_profile: pd.DataFrame
    column_profile: pd.DataFrame
    numeric_profile: pd.DataFrame
    categorical_profile: pd.DataFrame
    target_profile: pd.DataFrame
    slice_profile: pd.DataFrame
    missingness_patterns: pd.DataFrame
    missingness_drift: pd.DataFrame
    schema_drift: pd.DataFrame
    collinearity: pd.DataFrame
    point_in_time_profile: pd.DataFrame
    resource_usage: pd.DataFrame
    provenance: pd.DataFrame
    findings: pd.DataFrame
    warnings: tuple[str, ...]
    limitations: tuple[str, ...]
```

Dataclass 只保证 field assignment frozen；批准的 result DataFrames 本身不声明 deep
immutable。函数必须创建新结果表，不能返回 input/reference/current view 或其他 raw-data
copy。结果只可包含上述十四张 frozen evidence DataFrames；不包含 input/repaired DataFrame、
原始 `DataAuditConfig`/`DataAuditRoles`/`ColumnAuditRule`、positive label、caller literal、
estimator、Pipeline、Figure、condition tree、calibrator、private state 或修复建议对象。
Sample 只允许有界、排序后的 row positions。

## 6. Role selectors、冲突和 reference 规则

### 6.1 Selector validation

- 每个 scalar selector 必须是 non-empty exact `str`；tuple selector 必须为 tuple，成员
  必须为 non-empty exact `str`，且 tuple 内无重复；
- `data` 上每个显式 selector 必须存在；未知 selector 抛出 input `ValueError`；
- scalar identity roles `target`、`row_identifier`、`group`、`partition`、`fold`、`selection`、
  `historical_action`、`historical_policy`、`observation_time`、`event_time`、
  `shared_feature_available_time`、`label_available_time`、`outcome_end_time`、
  `partition_cutoff`、`window_start`、`window_end`、`horizon_end`、`analysis_as_of` 必须彼此
  不同；
- tuple selectors `score_columns`、`cost_columns`、`exposure_columns`、
  `constraint_input_columns` 各自无重复且彼此不重叠，也不得与 scalar identity roles 重叠；
  `features` 可以按下述规则故意重叠以产生 evidence；
- `excluded_columns` 不得与任何其他 role、feature、availability mapping、audit-input tuple
  或 `post_outcome_columns` 重叠；
- `shared_feature_available_time` 与非空 `feature_available_time_map` 不能同时声明；map 的
  feature 必须在 explicit non-empty `features` 中，shared declaration也要求explicit non-empty
  `features`；availability column必须存在且不得等于其feature；同一feature不得重复；
  多个features可以显式共享同一map value column，这不是duplicate selector；
- `features` 可以故意包含target、identifier、group、partition、fold、score、action、policy、
  cost、exposure、constraint、time、availability、window/horizon或post-outcome role column；
  这不是config exception，而是按第10.1节报告的direct/review evidence；
- `post_outcome_columns` 可以与 features 重叠以触发 finding，但不能包含 target；
- `positive_label` without target 是 config error；
- role config 不从 reference 补全。reference 缺失 target/group/partition/fold/score/action/
  policy/cost/exposure/constraint/time/identifier role 时，
  current audit 继续，comparison evidence 使用 `unavailable/role_absent_in_reference`；
  reference 缺失 feature 列进入 schema/missingness drift，不抛 exception。

### 6.2 Role meaning

| Role | Frozen meaning |
|---|---|
| `target` | caller-declared observed outcome column；Task 16 只 profile/audit |
| `features` | caller-declared candidate predictive inputs；`None` 表示未声明 feature role |
| `excluded_columns` | 完全排除于 column/value audit 的列；不删除输入 |
| `row_identifier` | caller-declared entity/row key evidence；不用于 alignment |
| `group` | caller-declared group isolation unit |
| `partition` | caller-declared train/validation/test/fold-like membership values；不猜值语义 |
| `fold` | caller-provided fold identity provenance；不构造或修改 fold |
| `selection` | historical selection/action-assignment slice；values 无 action semantic |
| `historical_action` | historical action-assignment provenance；不解释 action value |
| `historical_policy` | historical policy-version provenance；不选择或模拟 policy |
| `score_columns` | caller-computed score inputs；不重算、不评价、不升级为 probability |
| `cost_columns` | caller-declared cost inputs；只审计质量/availability |
| `exposure_columns` | caller-declared exposure inputs；另审计 negative/nonfinite evidence |
| `constraint_input_columns` | caller-declared constraint inputs；不优化或执行 constraint |
| `observation_time` | information snapshot time |
| `event_time` | optional observed event timestamp；不等于 outcome/label maturity |
| `shared_feature_available_time` | all-audited-feature shared availability timestamp evidence |
| `feature_available_time_map` | complete per-feature availability timestamp evidence |
| `label_available_time` | target availability timestamp evidence；不构造 Task 15 maturity |
| `outcome_end_time` | caller-declared outcome completion time |
| `partition_cutoff` | caller-provided per-row cutoff evidence；不生成 cutoff/fold |
| `window_start` / `window_end` | caller-declared audit window boundaries |
| `horizon_end` | caller-declared horizon completion evidence |
| `analysis_as_of` | caller-declared audit as-of timestamp；不读取当前时间 |
| `post_outcome_columns` | caller declaration that a column is known only after outcome |

所有新增 tuple/scalar selectors 均使用同一 exact-string normalization、unknown-column、
duplicate-selector 和 conflict validation。Reference/current 只按 exact column name匹配；缺失
role 只影响对应 comparison evidence，不从另一侧补全、不按 position 对齐。

## 7. Frozen result table schemas

Task 16 冻结十四张 result DataFrames。所有 text columns 使用 pandas `string` dtype；nullable
integer/float/bool 使用 `Int64`/`Float64`/`boolean`；非 nullable counts 使用 `int64`；
`sample_positions` 唯一允许 `object`，每个值是 `tuple[int, ...]`。`finding_key` 是 nullable
`string`。表为空时仍保留 exact columns/order/dtypes。任何宽表不得以一个 row-level
status/reason 覆盖多个独立指标组。

表级 status/reason closed pairs 为：

```text
available/computed
available/same
available/safe
unavailable/reference_not_provided
unavailable/role_not_declared
unavailable/role_absent_in_reference
unavailable/positive_label_not_declared
unavailable/target_not_declared
unavailable/column_absent
unavailable/current_column_added
unavailable/current_column_missing
undefined/no_rows
undefined/no_features
undefined/no_non_missing_values
undefined/no_finite_values
undefined/no_evaluable_rows
undefined/insufficient_rows
undefined/insufficient_support
undefined/zero_baseline_no_change
undefined/zero_baseline_increase
not_applicable/dtype_not_applicable
not_applicable/mode_not_applicable
not_applicable/role_not_declared
not_applicable/no_features
not_verifiable/budget_exceeded
not_verifiable/duplicate_scan_budget
not_verifiable/pattern_schema_mismatch
not_verifiable/unsupported_target_label
not_verifiable/unsupported_target_values
not_verifiable/missing_slice_value
not_verifiable/missing_partition_value
not_verifiable/missing_fold_value
not_verifiable/missing_availability_metadata
not_verifiable/partial_feature_availability_mapping
not_verifiable/timezone_mismatch
```

Available 数值不得配 missing reason。非 available 指标组只能按其 table rule保留已计算的
count/denominator，其余为 `pd.NA`，不得以 0 伪装 unavailable/undefined。全文出现的每个
status reason 必须属于上述 inventory；exception/finding-only reasons 另在第 11、13 节冻结。

### 7.1 `dataset_profile`

```text
side, n_rows, n_columns, profiled_column_count, declared_feature_count,
feature_status, feature_reason, duplicate_row_count, duplicate_row_rate,
duplicate_row_status, duplicate_row_reason, duplicate_index_count, duplicate_index_rate,
duplicate_index_status, duplicate_index_reason,
memory_usage_bytes, finding_key
```

一行 current；提供 reference 时第二行为 reference。`features is None` 时 declared count 为
`pd.NA` 且 `unavailable/role_not_declared`；explicit tuple 为 `available/computed`。Duplicate
counts 使用 `duplicated(keep=False)`，index 仅检测；超过 scan budget 时 count/rate 为
`pd.NA`、`not_verifiable/duplicate_scan_budget`，但duplicate-index group仍独立available。

### 7.2 `column_profile`

```text
side, column, column_position, role, pandas_dtype, logical_type, n_rows,
non_missing_count, missing_count, missing_rate, missing_status, missing_reason,
mixed_python_type_count,
empty_string_count, whitespace_only_count, unique_count, unique_rate, top_count,
top_rate, all_missing, constant, near_constant, high_cardinality,
suspected_identifier, value_profile_status, value_profile_reason, finding_key
```

Current rows 按 current order，随后 reference order；排除列不出现。String counts 是 exact
string matches。Unique/top/constant/cardinality/identifier 被 budget skip 时对应 values 为
`pd.NA` 且 `not_verifiable/budget_exceeded`。Zero-row column counts为0，missing rate为`pd.NA`、
`undefined/no_rows`；non-empty side的missing group为`available/computed`。

### 7.3 `numeric_profile`

```text
side, column, n_rows, non_missing_count, missing_count, finite_count,
positive_inf_count, negative_inf_count, mean, std, minimum, q25, median, q75,
maximum, count_status, count_reason, finite_status, finite_reason,
location_status, location_reason, dispersion_status, dispersion_reason,
range_status, range_reason, quantile_status, quantile_reason, finding_key
```

只含 non-bool physical numeric columns；statistics 只使用 finite non-missing float64 values；
`std` 为 `ddof=1`。Count group 总是 available；零 finite values 时 location/dispersion/range/
quantile 为 `undefined/no_finite_values`；一个 finite value 时 location/range/quantile
available，而 dispersion 为 `undefined/insufficient_rows`。Finite group 独立报告 inf counts。

### 7.4 `categorical_profile`

```text
side, column, non_missing_count, unique_count, unique_rate, top_count, top_rate,
singleton_level_count, unseen_in_current_count, unseen_in_current_rate,
count_status, count_reason, cardinality_status, cardinality_reason,
cardinality_rate_status, cardinality_rate_reason,
frequency_status, frequency_reason, concentration_status, concentration_reason,
comparison_status, comparison_reason, finding_key
```

适用于 logical categorical/boolean/text/identifier，不输出 labels。Count控制non-missing；
cardinality控制unique/singleton counts；cardinality-rate只控制unique rate；frequency只控制top
count；concentration只控制top rate；comparison只控制unseen fields。Reference row使用
`not_applicable/mode_not_applicable`，无reference的current row使用
`unavailable/reference_not_provided`，且不覆盖其他组。零non-missing values时count和
cardinality counts仍available，cardinality-rate/frequency/concentration/comparison按各自
denominator/mode为undefined或unavailable。

### 7.5 `target_profile`

```text
side, class_position, is_positive, positive_label_declared, positive_label_type,
count, rate, target_non_missing_n, class_status, class_reason,
binary_status, binary_reason, balance_status, balance_reason,
positive_class_status, positive_class_reason, finding_key
```

不输出 target/positive raw labels。Class rows 仍按 explicit positive first（如 exact match），
其余按 first row-position appearance；`positive_label_type` 只允许 closed family key
`bool/int/float/str/unsupported/not_declared`。Class count可 available而 binary/balance/
positive-class 独立 unavailable、undefined 或 not-verifiable。未声明 positive label时
`positive_label_declared=false`、type=`not_declared`，class/binary/balance 不被覆盖。

### 7.6 `slice_profile`

```text
side, slice_role, row_kind, slice_ordinal, partition_ordinal, fold_ordinal,
missing_bucket, row_count,
target_non_missing_count, target_non_missing_rate, positive_count, event_rate,
size_status, size_reason, target_rate_status, target_rate_reason,
event_status, event_reason, quality_status, quality_reason, finding_key
```

`row_kind`只允许`summary/value`。Partition和fold始终各有一条summary activation row：未声明
时size/quality为`not_applicable/role_not_declared`且counts/rates为`pd.NA`；声明时summary为
available并使用全体row/target support。Declared `group`、`partition`、`fold`、`selection`、
`historical_action`、`historical_policy`再按该role priority生成value rows；不输出raw values。
每个safe exact scalar value按首次出现的最小row position取得ordinal；missing使用dedicated
bucket、`missing_bucket=true`，不得stringify。
Size始终独立；target-rate控制target non-missing count/rate，event控制positive count/rate，
两者互不覆盖且均不覆盖size。Partition/fold missing bucket分别使用
`not_verifiable/missing_partition_value`、`not_verifiable/missing_fold_value`。Partition rows的
`partition_ordinal=slice_ordinal`、fold rows的`fold_ordinal=slice_ordinal`，其他role-specific
ordinal为`pd.NA`；这两个显式字段是后续安全provenance，不是raw value副本。

### 7.7 `missingness_patterns`

```text
pattern_key, pattern_bits, aggregated, source_pattern_count, missing_count,
row_count, row_rate, missing_cell_count, min_missing_count, max_missing_count,
sample_positions, reference_row_count, reference_row_rate, absolute_rate_change,
count_status, count_reason, rate_status, rate_reason,
reference_count_status, reference_count_reason,
reference_rate_status, reference_rate_reason,
comparison_status, comparison_reason, finding_key
```

Pattern 按 frozen audited-feature order，`1` missing、`0` non-missing，不含列名/值。普通 exact
pattern：`missing_count=count("1")`、`missing_cell_count=missing_count*row_count`、min=max=
missing_count、`source_pattern_count=1`。`pattern_key="p:" + pattern_bits`。Comparison rows使用 current row_count/rate，
reference fields按同一 exact identity填充；single mode comparison 为
`unavailable/reference_not_provided`。Current count/rate、reference count/rate和absolute change
各自使用独立status；零row side的count仍available而rate/delta undefined。

Current 有 0 rows 时唯一 synthetic row 为：`pattern_key="__NO_ROWS__"`、bits/missing/min/max
为 `pd.NA`、source_pattern_count=0、row_count=0、row_rate=`pd.NA`、missing_cell_count=0、count
`available/computed`、rate `undefined/no_rows`；如有reference，comparison及其reference fields
仍为`undefined/no_rows`/`pd.NA`，不把reference patterns伪装成current identity。有 rows 且 audited features 为零时唯一 empty
pattern：key=`"p:"`、bits=`""`、source_pattern_count=1、missing_count=0、row_count=n_rows、
row_rate=1.0，不归类为all-missing且不使用`no_features`覆盖available count/rate；
`no_features`只用于不适用的comparison/limitation evidence。
P 个 features 的 all-missing pattern 是 `"1" * p`、missing_count=p。

超 budget 时保留 combined reference/current row_count降序、bits升序的前
`max_missing_patterns-1` exact patterns，其余合并为最后一行 `__OTHER__`：bits/missing_count
为 `pd.NA`；row_count、reference_row_count和missing_cell_count求和；min/max取被合并 exact
patterns的 missing_count extrema；sample_positions取所有聚合 rows 的最小有界 positions。
Ties 按 bits；exact rows 始终先于 `__OTHER__`；不得随机采样。Reference缺任一 audited
feature 时不伪对齐，comparison 为 `not_verifiable/pattern_schema_mismatch`。
`sample_positions`只表示current row positions；reference-only pattern的tuple为空，reference
row positions不与current positions混入同一字段。
`__OTHER__`的aggregate counts/rates仍为`available/computed`；丢失的是单一exact pattern
identity，由`aggregated=true`、source count、resource usage及truncation warning表达，不把已求和
的count伪装成not-verifiable。

### 7.8 `missingness_drift`

```text
column, reference_present, current_present, reference_n, current_n,
reference_missing_count, current_missing_count, reference_missing_rate,
current_missing_rate, absolute_rate_change, relative_rate_change,
new_all_missing, recovered, count_status, count_reason, rate_status, rate_reason,
reference_count_status, reference_count_reason, current_count_status, current_count_reason,
reference_rate_status, reference_rate_reason, current_rate_status, current_rate_reason,
absolute_change_status, absolute_change_reason,
relative_change_status, relative_change_reason, finding_key
```

仅 reference provided 时生成。Relative formula 是
`(current_rate-reference_rate)/reference_rate`。Zero/zero 与 zero/increase分别为
`undefined/zero_baseline_no_change`、`undefined/zero_baseline_increase`，但不覆盖 available
absolute change。`count_status/count_reason`与`rate_status/rate_reason`是comparison-level摘要；
每侧count/rate另有独立status，因此absent或zero-row side不会覆盖另一侧已计算值。
Drift finding仍以 absolute 或 available relative threshold的 `>=` 触发。

### 7.9 `schema_drift`

```text
column, reference_position, current_position, reference_dtype, current_dtype,
reference_logical_type, current_logical_type, reference_role, current_role,
column_added, column_removed, dtype_changed, logical_type_changed, role_changed,
primary_change, status, reason, finding_key
```

Booleans 保留并发变化；`primary_change` precedence 固定：`removed > added >
dtype_and_logical_type_changed > dtype_changed > logical_type_changed > role_changed >
unchanged`。它只用于排序/摘要，不丢弃独立 flags。Added/removed rows 的不可比较字段为
`pd.NA`，分别使用`unavailable/current_column_added`和
`unavailable/current_column_missing`；unchanged使用`available/same`，其他present-both变化
使用`available/computed`。当前single-roles API不接受独立reference roles，因此两侧同名且
present的column使用同一caller role，`role_changed=false`；added/removed时为`pd.NA`。字段和
precedence仍冻结，Task 16不得据schema猜测第二套role。

### 7.10 `collinearity`

```text
left_column, right_column, valid_n, pearson_r, absolute_r, threshold,
status, reason, finding_key
```

只输出达到 Task 16 near-collinear threshold 的 pairs；pair按 feature positions，complete-case
finite rows，valid_n/min-periods和constant rules不变。它不返回完整 matrix，不替代 Task 07。

### 7.11 `point_in_time_profile`

```text
side, scope, column, evaluated_count, violation_count,
not_verifiable_count, status, reason, finding_key
```

PIT 未激活时表含唯一 dataset row，`not_applicable/role_not_declared`。激活时先按 audited
feature order输出`scope="feature"` rows，再按role priority输出score/action/policy/cost/
exposure/constraint的`scope="audit_input"` rows，再按第10.5节关系顺序输出
`scope="chronology"` rows，最后`scope="dataset"` row；chronology/dataset rows的
`column=pd.NA`。Scope inventory只允许上述四项。Feature row仅在该 feature所有有 provenance
且全部 non-missing/timezone-compatible rows可判断时为 available；有violation时使用
`available/computed`并链接finding，无violation时可`available/safe`。Partial mapping、missing
observation/time values或timezone mismatch均 not_verifiable。Dataset row只有全部 required
feature rows available、无violation且无未验证rows时才可`available/safe`；全部可验证但存在
violation时为`available/computed`；任何缺metadata不得为safe。

### 7.12 `resource_usage`

```text
side, resource, requested, available, actual, truncated, status, reason, finding_key
```

固定 resource 顺序：`columns, column_rules, duplicate_scan_rows, unique_inspection_rows,
category_levels, missing_patterns, collinearity_columns, finding_samples`。Current、reference、
comparison 顺序不变。局部 truncation/skip 保留 requested/available/actual；config policy不在
result 中重复保存。

### 7.13 `provenance`

```text
provenance_key, value_type, numeric_value, text_value, count_value,
boolean_value, status, reason
```

固定顺序：`config_schema_version`、`atomic_kernel_version`、所有 numeric thresholds/budgets按 `DataAuditConfig` field
order、`positive_label_declared/positive_label_type`、role column names按 `DataAuditRoles` field order、每条 rule按 caller order的
`rule_key/rule_column_name/operator_key/literal_declared/literal_count/literal_type_family`、`rule_count`、
`config_fingerprint`。Role column names可以保留，因为它们是 schema provenance；不得保留
role values、target label或 rule literals。`rule_column_name`同样只允许validated schema column
name，不是cell value。

`config_schema_version="task16-data-audit-config-v1"`。Rule key固定为
`column_rule:{zero_based_config_ordinal}`；operator key只允许`minimum_ge`、`minimum_gt`、
`maximum_le`、`maximum_lt`、`allowed_in`、`special_in`、`not_after_le`、
`nondecreasing_le`，按 `ColumnAuditRule` field order输出。Literal type family只允许
`none/bool/int/float/str/date/datetime/timestamp/mixed`，不含literal representation。
`atomic_kernel_version="task16-audit-v1"`，与audit direct atomic call完全一致。

`config_fingerprint` 是 lowercase 64-character SHA-256 hex。输入是 UTF-8 canonical JSON：
固定由stdlib `json.dumps(payload, sort_keys=True, ensure_ascii=True, allow_nan=False,
separators=(",", ":"))`产生，随后UTF-8 encode；内容只包含config schema version、numeric/boolean
policy、positive-label declared/type family、role/rule column names、rule/operator keys、
literal-present flag/count/type-family；不得包含
positive label或任何 allowed/special/range/membership/sentinel literal。它只证明 sanitized
provenance equality；不同 literal但相同 sanitized shape 可以产生同一 fingerprint，不是密码学
安全或原始 config identity 承诺；禁止 Python `hash()`。
Tuple role members和rules保持caller tuple order；JSON object keys才lexicographically sorted。

### 7.14 `findings`

```text
finding_key:string
category:string
scope:string
dataset_role:string
column:string
column_position:Int64
role:string
severity:string
status:string
reason:string
metric_key:string
value:Float64
threshold:Float64
count:Int64
denominator:Int64
affected_rate:Float64
sample_positions:object
detail_table:string
detail_row_ordinal:Int64
recommendation:string
limitation:string
provenance:string
```

每个会生成 finding 的 detail table 都有 nullable `finding_key`。`detail_row_ordinal` 是该表按
合同排序完成后的 0-based row position，不是 pandas index；一个 detail row最多指定一个
primary finding（最高 severity后按 finding order最前者），其他 findings可共享相同
`detail_table/detail_row_ordinal`。Finding key 是 canonical UTF-8 fields
`category,scope,dataset_role,column_position,metric_key,reason,detail_table,
detail_row_ordinal,sample_position_ordinal` 的SHA-256 lowercase hex；无sample时ordinal=`-1`，
否则取最小row position。上述字段按列出顺序放入JSON array，并使用第7.13节相同
`json.dumps`参数和UTF-8 encoding；不使用Python `hash()`、内存地址、raw value或sample
tuple字符串。

`detail_table`只允许：`dataset_profile`、`column_profile`、`numeric_profile`、
`categorical_profile`、`target_profile`、`slice_profile`、`missingness_patterns`、
`missingness_drift`、`schema_drift`、`collinearity`、`point_in_time_profile`、
`resource_usage`。`provenance`不产生finding，`findings`不自关联。

Nullable cells 使用 `pd.NA`，由逐指标 status/reason解释。`recommendation` 只允许
non-executing review text。`provenance` 只允许 `pandas_structure`、`task03_schema`、
`task15_label_semantics`、`caller_roles`、`caller_column_rule`、`condition_kernel_atomic`、
`reference_current_comparison`、`resource_budget`、`sanitized_config`，多来源按此顺序连接。
`value`只允许aggregate metric，不能是raw cell/label/literal。`threshold`只允许本合同
`DataAuditConfig`中的非敏感numeric policy；caller range/allowed/special/membership literals
触发finding时threshold必须为`pd.NA`，只以sanitized operator key、literal-declared/count/type
provenance表明规则来源。

## 8. Quality definitions、thresholds 与 policies

### 8.1 Dataset-level

- empty data：`n_rows == 0`，structured error-severity finding，不抛 exception；
- zero-feature：`features == ()`，warning finding；`features is None` 不声称 zero feature；
- duplicate columns/non-string labels：input schema exception，因为无法建立唯一 column
  identity；
- duplicate rows：全列 exact pandas `duplicated(keep=False)`；任何 count > 0 报 warning；
- duplicate index：index `duplicated(keep=False)`；任何 count > 0 报 info；index 不进入
  alignment；
- memory/large input 只作为 limitation/warning，不建立 parallel/distributed engine。

### 8.2 Column-level

- 所有 inspected object/category scalars 必须先通过第 12.3 节 non-dispatching whitelist，之后
  才可调用 pandas missing/comparison operations；
- missing：validated safe scalar 上使用 pandas `isna`；`missing_rate >= 0.40` warning，
  `missing_rate == 1.0` error；
- finite/inf：只对 non-bool physical numeric dtype；NaN 是 missing，不是 infinity；
- constant：恰有一个 non-missing unique value；all-missing 不是 constant；
- near constant：非 constant 且 `top_count / non_missing_count >= 0.95`；
- high cardinality：`unique_count > 50` **and** `unique_rate > 0.50`；count 边界是 `>`，
  rate 边界是 `>`，与 Task 04 语义一致；
- suspected identifier：non-missing >= 20 且 `unique_rate >= 0.98`；完全结构性，不读取
  column name，不调用身份/反欺诈逻辑；
- mixed types：非缺失approved Python/NumPy scalars归一为built-in type family后family count >1；
- empty string：exact `str` 且 `value == ""`；whitespace-only：exact `str`、非空且
  `value.strip() == ""`；不修改原字符串；
- datetime role consistency：显式 time role 必须是 pandas datetime dtype；object/string
  不自动 parse，产生 error finding `datetime_role_mismatch`；
- special sentinel：只检查 caller rule 的 exact type-aware literal；不自动猜测 `-999`、
  `unknown`、空白或任何行业 sentinel。

### 8.3 Column rules

- Column rules 只约束 current `data`；reference 只提供 baseline profile/drift，不用 current
  policy retroactively 判定 reference violations；
- range、allowed-values、special-values、`not_after_columns` 和 monotonic adjacent-pair 的
  equality/ordering/membership/missing/type/nonfinite/timezone truth，全部委托
  `_condition_kernel._evaluate_atomic_condition`，audit层不得自行比较；
- range 只生成对应 `ge/gt/le/lt` atomic requests；missing 不违反但按 kernel unknown聚合为
  not-verifiable，lower/upper equality由 inclusive flags选择operator；
- allowed/special values生成 `in` atomic request，bool/int separation和literal validation完全
  使用kernel；special只计数并报 evidence，不替换；
- `not_after_columns` 固定检查 left `<=` right，missing pair 不算 violation，进入
  not-verifiable count；
- `nondecreasing=True` 由audit层按 row position构造相邻pair；有group时，group相邻边界的
  `eq`与value pair的`le`都由atomic kernel执行，audit层只据truth划分/聚合。Missing/type/
  timezone mismatch沿用kernel unknown/reason；
- 同一 column 最多一个 `ColumnAuditRule`；rule 中没有任何实际 constraint 时 config error。
- minimum/maximum 并存时必须是支持的同类 scalar，bound ordering也通过one-row atomic `le`
  request验证为true，不得用Python直接比较；每个 literal
  tuple type-aware 无重复，`not_after_columns` 无重复且不得含自身；special 与 allowed
  values 可以重叠：special finding 独立生成，allowed violation 只由 allowed set 判断；
  non-empty allowed/special tuple各自长度必须`<=100`，与atomic membership hard limit一致，
  超过使用`membership_budget_exceeded`而非audit层第二套budget。

Intrinsic dataset/column/target quality rules 对 current 和存在的 reference 分别执行；
direct/proxy/partition/group leakage只审计current；time/PIT provenance按P1-7要求在current与
存在的reference各自独立审计，不做cross-side row alignment。Reference其余只进入comparison、
profile和drift evidence。

### 8.4 Target quality

- missing target、constant target、non-binary target（normalized unique count != 2）分别
  独立 finding；
- class count/rate 对所有 normalized classes计算；rare class 是
  `count < rare_class_count` **or** `rate < rare_class_rate`，边界使用 `<`；
- event rate 只在 binary target 且 explicit positive label exact match 时 available；
- declared group/partition/fold/selection/historical-action/historical-policy 的 target
  non-missing support 与 event rate 进入
  `slice_profile`。selection slices 间 target non-missing rate 的 max-min
  `>= missingness_drift_absolute_threshold` 时生成
  `selection_outcome_support_gap` warning；只报告 support difference，不推断 selection
  effect、bias、policy quality 或 causation；
- Task 16 不填补 rejected/unselected outcomes，不做 reject inference，不训练 proxy model。

### 8.5 Roadmap audit-input quality

- score/cost/exposure/constraint inputs进入普通column profile和caller rules；score、cost、
  exposure要求non-bool physical numeric dtype，否则`audit_input_dtype_mismatch`；constraint
  inputs不预设dtype，numeric时才有finite evidence。Score不计算metrics，cost不计算profit，
  constraint不优化；
- 声明score但partition与fold均未声明时产生
  `score_partition_provenance_missing` warning；任一已声明provenance列的missing rows由其
  dedicated slice evidence表达；
- exposure finite numeric rows通过atomic kernel `lt` literal 0比较，`< 0`产生
  `negative_exposure` warning；nonfinite仍使用`nonfinite_values`；
- 任一declared constraint input存在missing cell时产生`constraint_input_missing` warning，
  因完整性是其审计目标；不自动填补；
- fold/action/policy进入 slice profile，审计missing、cardinality、partition/window availability；
  不解释其 raw values；
- score/fold/action/policy/cost/exposure/constraint role在reference缺失时只影响对应comparison，
  使用 `unavailable/role_absent_in_reference`。

## 9. Missingness profiling 与 drift ownership

Task 16 是 missingness profiling/drift 的唯一 owner。Task 19 只能消费本合同表，不得
重算。单表 profile 包含 per-column count/rate/all-missing/non-missing、row-level
missing-count evidence 和有界 exact pattern frequency；comparison 包含两侧 n、absolute/relative change、new/missing
columns、dtype/logical-type change、new-all-missing、recovered 和 pattern difference。

- 两侧 column matching 是 exact string equality，不使用 pandas index alignment；
- current/reference row counts 可以不同；不做 row join；
- 两侧都少于 `minimum_drift_rows` 不阻止手算 rates，但追加
  `insufficient_drift_rows` warning/limitation；
- empty current/reference 严格使用第 7.7 节 `__NO_ROWS__` synthetic evidence，rate/delta 为
  undefined，不把 0 行的 synthetic `0.0` rate解释为真实无缺失；
- pattern column order 使用 explicit features；features is None 时使用 current profile
  columns，并在 reference 上按 exact names读取，reference-only columns 不进入 pattern；
- pattern budget requested/actual 和 aggregation reason 进入 warnings/limitations 与
  aggregated row；禁止 hash/random sampling。

## 10. Leakage finding taxonomy

`category` 的固定优先级：

```text
dataset_structure
column_quality
missingness
schema_drift
missingness_drift
target_quality
direct_target_leakage
target_proxy
partition_leakage
group_leakage
time_leakage
point_in_time
constraint_violation
resource_limitation
```

`scope` 只允许：`dataset`、`column`、`row_set`、`partition`、`fold`、`group`、`time`、
`comparison`。`dataset_role` 只允许 `current`、`reference`、`comparison`。

### 10.1 Direct target leakage

- target exact column included in features：`target_in_features`，error；
- feature/target jointly non-missing values全部 type-aware exact equal且 support >=2：
  `exact_target_copy`，error；其paired `eq` truth必须由atomic kernel提供；near-copy使用同一
  atomic result聚合，不得再比较一次；
- caller-declared post-outcome column included in features：`post_outcome_feature`，error；
- identifier/group/partition/fold/score/action/policy/cost/exposure/constraint/time/availability/
  cutoff/window/horizon role included in features：
  `role_column_in_features`，warning，并说明可能合法但必须 review。

### 10.2 Target proxy evidence

- near target copy：paired support >= `proxy_min_support` 且 exact-match rate
  `>= near_copy_rate`，warning，必须带 `association_not_causation` 和 false-positive
  limitation；exact copy 优先且不重复；
- deterministic categorical proxy：feature levels <= `max_category_levels`、support 达标、
  至少两个 target classes、每个 observed feature level 只映射一个 normalized target
  class，且每个 level support >=2；warning；
- suspected identifier 不进入 categorical proxy；continuous threshold search、mutual
  information、model fit、correlation-only proxy 与任意 deterministic transformation search
  明确排除；
- 高 association 但未命中上述 deterministic/near-copy rule 不得产生 leakage finding。

### 10.3 Partition 与 fold provenance

Partition audit仅当`roles.partition is not None`激活；否则slice/partition comparison中对应
evidence为`not_applicable/role_not_declared`，不得生成missing-partition warning。Fold同理只在
`roles.fold` 声明时激活。

- partition/fold missing使用dedicated internal bucket，不 stringify、不输出 raw value，分别
  产生 `missing_partition_value`/`missing_fold_value`；ordinal按首次出现的最小 row position；
- missing partition/fold/group/identifier values不进入cross-value overlap或mapping denominator，
  只由各自missing finding/count表达；

- same row identifier exact value in >1 partition：`identifier_partition_overlap`；
- same group exact value in >1 partition：`group_partition_overlap`；
- exact duplicate payload rows（比较时排除 partition column）跨 partition：
  `duplicate_row_partition_overlap`；
- fold另审计row identifier/group exact value跨fold overlap。Fold/partition consistency固定为：
  对joint non-missing safe scalars，每个exact fold identity只能映射到一个exact partition
  identity；任一fold映射到多个partition时产生`fold_partition_inconsistent`。不要求fold与
  partition raw labels相等，也不推断train/test语义；只消费DataFrame内caller-provided列，
  不接受外部membership table；
- target-distribution shift仅在 partition、target和可评估positive semantics齐备且
  `partition_target_rate_shift_threshold` 非None时生成finding。每个partition使用
  `abs(partition_rate-pooled_rate) >= threshold`，且partition与pooled evaluable rows均
  `>= partition_target_min_support`；support不足为 `undefined/insufficient_support`。Threshold
  为None仍可返回slice测量，但不生成shift finding；该finding仅是distribution evidence，
  不称leakage；
- Task 16 不生成或修改 fold，不重新执行 Task 15 validation/OOF。

### 10.4 Group leakage

- explicit group missing：`missing_group`；
- group cardinality 和 singleton-group rate 进入 evidence；
- group cross-partition overlap 单独归类，不从列名猜 group；
- no partition 时 group overlap 不适用，不把重复 group 当 leakage。

### 10.5 Time 与 point-in-time leakage

PIT仅在至少一个 `observation_time`、shared/map feature availability、`event_time`、
`outcome_end_time`、`label_available_time`、`partition_cutoff`、`window_start`、`window_end`、
`horizon_end` 或 `analysis_as_of` 被声明时激活；全部未声明时
`point_in_time_profile=not_applicable/role_not_declared`且不产生warning。Reference/current各侧
独立按其实际role columns检查完整性，不从另一侧补全。激活后只审计caller provenance：

```text
feature_available_time <= observation_time
observation_time <= outcome_end_time
observation_time <= event_time
event_time <= outcome_end_time
outcome_end_time <= label_available_time
observation_time < partition_cutoff
feature_available_time <= partition_cutoff
window_start <= window_end
window_end <= analysis_as_of
horizon_end <= analysis_as_of
```

- 第一条逐 feature mapping；违反为 `feature_after_observation` error；
- observation/event/outcome/label relations 只检查 chronology，不执行 Task 15 horizon、
  reporting-delay 或 maturity derivation；event_time absent 时对应 relations 为
  not_applicable，不用 event occurrence 代替 label availability；
- cutoff checks 只读取 caller-provided per-row cutoff，不构造 fold；
- exact boundary 如上：availability/event/outcome/label relations 使用 `<=`，
  observation/cutoff 使用 `<`；
- time role missing、NaT 或 incompatible timezone row 不当作 safe；计入
  not_verifiable evidence；
- explicit row identifier 或 group 与 observation time 的重复 exact key 报
  `duplicate_entity_time`；
- observation time 按 row position 逆序只报 `time_order_violation`，不排序输入；
- shared availability适用于全部 audited features；per-feature map必须完整覆盖全部 audited
  features。二者互斥；partial map不静默skip，未映射feature逐项返回
  `not_verifiable/partial_feature_availability_mapping`；
- `features is None`时没有caller-declared feature universe，不能声明shared/map availability；
  其他time chronology仍可审计，但dataset feature-safety不得为safe。`features == ()`时不生成
  feature rows，feature-safety为`not_applicable/no_features`；已声明chronology独立审计，若其
  全部可验证且无violation，chronology-only dataset summary可以`available/safe`；
- 缺 observation time时所有availability comparisons为
  `not_verifiable/missing_availability_metadata`；missing/NaT/timezone mismatch rows均计入
  per-feature not-verifiable count；
- dataset PIT只有全部audited features完整映射且每项全部可判断时才可
  `available/safe`；任何partial/missing/incompatible metadata绝不返回safe；
- window/horizon只检查上述chronology；缺任一所需配对metadata时not-verifiable，不推断
  mature，不生成cutoff；
- 声明score/cost/exposure/constraint input且同时声明window/horizon时，`observation_time`是其
  唯一time-source provenance；缺失时对应input逐列not-verifiable。Task 16只检查metadata
  completeness和上述chronology，不额外推导score生成时间、horizon maturity或label maturity；
- historical action/policy同样生成audit-input availability rows。每个audit-input row只统计
  input non-missing且所需partition/fold/window/horizon/observation metadata完整的evaluated
  count和缺失metadata count；不解释raw action/policy、不得用window推导selection效果；
- Task 15 继续唯一拥有 time validation plan、fold cutoff construction、label maturity、
  train purge、OOF 和 metrics。Task 16 不调用 Task 15 validation entry。

## 11. Finding、severity、status、reason 与 ordering

### 11.1 Status

只允许：

- `available`：evidence 已按合同计算；
- `unavailable`：所需声明/side/column 没有提供；
- `undefined`：合法输入但数学分母为零或 empty side；
- `not_applicable`：该概念对 dtype/role/mode 不适用；
- `not_verifiable`：metadata、type support 或 budget 不足，不能证明安全或违规。

上述五项是public audit table/finding status。Private `_ConditionEvaluation.status` 只允许
`available`、`not_verifiable`、`inactive`；`inactive`仅表示root effective window外，不进入
public audit table status vocabulary，也不等于false/unknown metadata。

### 11.2 Severity

只允许 `info`、`warning`、`error`。Severity 是 audit classification，不是 exception；
检测到 error finding 仍返回完整 result。Invalid config/input 才抛 exception。

Severity 按 reason 唯一冻结；同一 reason 不因 count/category变化：

- `error`：`empty_dataset`、`all_missing_column`、`datetime_role_mismatch`、
  `audit_input_dtype_mismatch`、`target_constant`、`target_non_binary`、
  `target_in_features`、`exact_target_copy`、`post_outcome_feature`、`range_violation`、
  `allowed_value_violation`、`cross_column_order_violation`、`monotonic_time_violation`、
  `feature_after_observation`、`observation_after_outcome_end`、`event_before_observation`、
  `event_after_outcome_end`、`label_before_outcome_end`、`observation_at_or_after_cutoff`、
  `feature_after_cutoff`、`window_start_after_window_end`、`window_end_after_analysis_as_of`、
  `horizon_end_after_analysis_as_of`；
- `warning`：`zero_feature_dataset`、`duplicate_rows`、`high_missing_column`、
  `nonfinite_values`、`negative_exposure`、`constant_column`、`near_constant_column`、
  `mixed_python_types`、`empty_strings`、`special_value_present`、`target_missing`、
  `rare_target_class`、`selection_outcome_support_gap`、`unsupported_target_values`、
  `role_column_in_features`、`score_partition_provenance_missing`、
  `constraint_input_missing`、`near_target_copy`、`deterministic_categorical_proxy`、
  `identifier_partition_overlap`、`group_partition_overlap`、
  `duplicate_row_partition_overlap`、`identifier_fold_overlap`、`group_fold_overlap`、
  `fold_partition_inconsistent`、`missing_group`、`missing_partition_value`、
  `missing_fold_value`、`duplicate_entity_time`、`time_order_violation`、
  `missing_availability_metadata`、`partial_feature_availability_mapping`、
  `timezone_mismatch`、
  `role_absent_in_reference`、`pattern_schema_mismatch`、`current_column_added`、
  `current_column_missing`、`dtype_changed`、`logical_type_changed`、`role_changed`、
  `missingness_rate_changed`、`missingness_pattern_changed`、`insufficient_sample`；
- `info`：`duplicate_index`、`high_cardinality_column`、`suspected_identifier`、
  `whitespace_only`、`target_distribution_shift`、`singleton_group`、`budget_exceeded`。

未检测到 finding 时不在 `findings` 生成 passed row；PIT safe/activation状态由
`point_in_time_profile`表达。Metadata不足必须生成对应 warning finding。

### 11.3 Finding keys 与 reason

Finding key的SHA-256输入和detail linkage已在第7.14节唯一冻结。固定 finding reason
vocabulary按 category/metric rank顺序为：

```text
empty_dataset, zero_feature_dataset, duplicate_rows, duplicate_index,
all_missing_column, high_missing_column, nonfinite_values, constant_column,
near_constant_column, high_cardinality_column, suspected_identifier,
mixed_python_types, empty_strings, whitespace_only, datetime_role_mismatch,
audit_input_dtype_mismatch, negative_exposure,
score_partition_provenance_missing, constraint_input_missing,
special_value_present, range_violation, allowed_value_violation,
cross_column_order_violation, monotonic_time_violation,
target_missing, target_constant, target_non_binary, rare_target_class,
unsupported_target_values,
selection_outcome_support_gap,
target_in_features, exact_target_copy, post_outcome_feature, role_column_in_features,
near_target_copy, deterministic_categorical_proxy,
identifier_partition_overlap, group_partition_overlap,
duplicate_row_partition_overlap, identifier_fold_overlap, group_fold_overlap,
fold_partition_inconsistent, missing_partition_value, missing_fold_value,
target_distribution_shift, missing_group, singleton_group, feature_after_observation,
observation_after_outcome_end, event_before_observation, event_after_outcome_end,
label_before_outcome_end,
observation_at_or_after_cutoff, feature_after_cutoff,
window_start_after_window_end, window_end_after_analysis_as_of,
horizon_end_after_analysis_as_of, duplicate_entity_time, time_order_violation,
missing_availability_metadata, partial_feature_availability_mapping,
timezone_mismatch,
role_absent_in_reference, pattern_schema_mismatch,
current_column_added, current_column_missing, dtype_changed, logical_type_changed, role_changed,
missingness_rate_changed, missingness_pattern_changed,
insufficient_sample, budget_exceeded
```

不冻结完整 prose message；冻结 reason、numeric evidence、recommendation category 和
limitation code。Recommendation 必须与事实字段分开，不能声称自动修复已发生。

### 11.4 Deterministic ordering

Findings 排序键固定为：severity rank `error > warning > info`、第10节category rank、
dataset-role rank `current < reference < comparison`、column position（无column为`-1`）、
metric-key rank、sample-position ordinal、detail-row ordinal、finding key。Metric-key rank按
对应detail table冻结的column/指标组出现顺序；不在table中的metric按上述finding reason
inventory顺序。无sample时ordinal=`-1`，多sample取最小row position；不得按sample tuple
字符串排序。Warnings/limitations按各自closed vocabulary priority去重，不依赖set/hash。

## 12. Private closed condition kernel

### 12.1 Internal representation

以下 symbols 全部 private，位于 `sharper._condition_kernel`，不得从 `sharper` export：

```python
@dataclass(frozen=True)
class _ConditionOperand:
    kind: Literal["column", "literal"]
    value: object

@dataclass(frozen=True)
class _ConditionNode:
    node_type: Literal["atomic", "and", "or", "not"]
    operator: str | None
    left: _ConditionOperand | None
    right: _ConditionOperand | None
    children: tuple["_ConditionNode", ...]
    effective_from: datetime | None
    expires_at: datetime | None
    version: str | None

@dataclass(frozen=True)
class _ConditionEvaluation:
    truth: pd.Series
    status: pd.Series
    reason: pd.Series
    root_version: str
```

两个且仅两个 internal entries：

```python
def _evaluate_atomic_condition(
    data: pd.DataFrame,
    *,
    operator: str,
    left: _ConditionOperand,
    right: _ConditionOperand | None,
    root_version: str,
) -> _ConditionEvaluation: ...

def _evaluate_condition(
    data: pd.DataFrame,
    condition: _ConditionNode,
    *,
    evaluation_time: str | datetime | None = None,
) -> _ConditionEvaluation: ...
```

`_evaluate_atomic_condition` 是唯一atomic comparison实现；`_evaluate_condition`负责whole-tree
validation、Boolean composition和effective-time orchestration，并对每个atomic node调用同一
primitive。Audit path使用固定 synthetic root version `task16-audit-v1`调用atomic entry。
两入口均private且不export。Evaluator在读取任何row values前验证可由schema/spec验证的全部
结构；随后在执行任何node前，对whole tree引用的全部object/category columns和categories逐
cell完成第12.3节non-dispatching scalar validation。Short-circuit不能绕过该pre-scan或invalid
descendant validation。返回Series按row position、RangeIndex
`0..n-1`，不继承duplicate index。

### 12.2 Closed operators

Atomic operator inventory 唯一冻结为：

```text
eq, ne, lt, le, gt, ge,
in, not_in, between,
is_missing, is_not_missing
```

- `eq/ne/lt/le/gt/ge` 支持 column/literal 或 column/column；
- `in/not_in` 的 right 必须是 non-empty exact built-in tuple；empty tuple stable invalid
  `empty_membership_collection`，不得计算为false/true/unknown；
- `between` 的 right 必须是 exact two-element literal tuple，闭区间 `[lower, upper]`；
- `is_missing/is_not_missing` 只使用 left，right 必须为 `None`；
- 不支持 contains、startswith、regex、arithmetic、aggregation、window、callable 或
  dynamic operator。Task 17/18 若需要新 operator，必须先 review 本合同变更。

每个 atomic node 的 left 必须是 `kind="column"`；right 只按 operator 允许 literal 或
column。Literal-left expression 不支持。所有 referenced columns 在 whole-tree validation
时一次性验证，unknown column 不进入 row evaluation。

### 12.3 Scalar normalization 与 type semantics

任何 `pd.isna`、equality、ordering、membership、iteration、hashing、sorting、conversion、
stringification、`repr` 或NumPy/pandas coercion之前，必须先以 `type(value) is ...` 或批准的
exact NumPy scalar class完成non-dispatching validation。Exact scalar whitelist只有：built-in
`NoneType/bool/int/float/str`、批准的NumPy boolean/integer/floating scalar、exact
`datetime.date`、exact `datetime.datetime`、exact `pandas.Timestamp`、`pd.NA` singleton和
`pd.NaT` singleton。Built-in/user subclasses、list/dict/set/array及其他object均不接受。
Membership container必须是exact built-in tuple，元素逐项先whitelist；index、object/category
列的每个cell及category categories也在missing/hash/duplicate检查前验证。Index value错误使用
其row position和column position `-1`，仍不得读取或返回index label。

批准的NumPy exact type inventory固定为：`np.bool_`；`np.int8/int16/int32/int64`；
`np.uint8/uint16/uint32/uint64`；`np.float16/float32/float64`。Aliases只有在其exact runtime
type等于上述某项时自然接受；不按`isinstance(value, np.generic)`开放其他NumPy scalar。

- 只有批准的 exact NumPy scalar可安全 `.item()`：`np.bool_ -> bool`、`np.integer -> int`、
  `np.floating -> float`；NumPy string不在kernel whitelist，必须作为
  `unsupported_scalar_type`拒绝；pandas Timestamp保持datetime semantic；
- bool 与 int 始终类型不同；string 不转 numeric/bool/datetime；category 使用其实际
  scalar；
- numeric ordering 只允许 finite int/float（bool 排除）；data 中 inf 对普通 comparison
  产生 unknown/nonfinite_operand；numeric literal inf/NaN 在 spec validation 失败；
- string 只 exact comparison，不 strip/case-fold/locale compare；
- datetime 只允许 datetime/Timestamp；naive 只能与 naive 比，aware 只能与 aware 比，
  aware values 按 UTC instant 比较但不修改输入；aware/naive 混合为 unknown；
- 两个均为whitelisted但family不兼容的per-row comparison返回unknown/type_mismatch；不在
  whitelist的data value立即抛audit input或condition evaluation
  `unsupported_scalar_type`，unsupported literal在spec validation失败；
- complex、period、interval、nested collection 或其他不在 numeric/bool/string/datetime/
  category scalar families 的 physical/observed dtype，在 evaluator 读取 rows 前以
  `condition evaluation is invalid: unsupported_dtype` 失败；支持的 object/category 列中
  偶发 supported-family mismatch 才返回 row-level unknown/type_mismatch；
- membership tuple不得为空，不得包含missing、nested collection或重复normalized literal；
  bool/int exact-type duplicate 不合并；
- missing 使用 `pd.isna` 标量语义，覆盖 `None`、NaN、NaT、`pd.NA`。除两个 missing
  operators 外，只要任一 operand missing，truth 为 unknown/missing_operand；即使两边
  都 missing，`eq` 也不是 true。

### 12.4 Tri-state truth tables

Truth 只允许 `true`、`false`、`unknown`，不得依赖 Python truthiness。

Private reason inventory固定为：`computed`、`missing_operand`、`type_mismatch`、
`timezone_mismatch`、`nonfinite_operand`、`missing_evaluation_time`、
`outside_effective_window`。Available true/false使用`computed`；unknown保留导致unknown的reason；
inactive只使用`outside_effective_window`。Boolean组合有多个unknown children时按children tuple
顺序保留第一个参与最终unknown结果的reason，NOT保留child reason。

| AND | true | false | unknown |
|---|---|---|---|
| true | true | false | unknown |
| false | false | false | false |
| unknown | unknown | false | unknown |

| OR | true | false | unknown |
|---|---|---|---|
| true | true | true | true |
| false | true | false | unknown |
| unknown | true | unknown | unknown |

| input | NOT |
|---|---|
| true | false |
| false | true |
| unknown | unknown |

AND/OR children 按 tuple 顺序；AND 遇 false、OR 遇 true 可 deterministic short-circuit，
但所有 nodes 已先 validation。NOT 必须 exact one child；AND/OR 至少 two children；atomic
无 children；Boolean node 的 operator/operands 必须 `None`。

### 12.5 Effective/expiration semantics

- 只有 root node 可设置 `effective_from`/`expires_at`；descendant 设置即 invalid spec；
- root `version` 必须non-empty ASCII token且最多64 characters；所有descendant version必须
  `None`，否则 `child_version_not_allowed`；kernel不选择最新版本、不比较版本优先级；result
  `root_version`和provenance只记录root version；
- interval固定为 `[effective_from, expires_at)`；static bounds同时存在时必须同为aware或
  同为naive、timezone可比较且 `effective_from < expires_at`。Awareness mismatch或不可比较
  bounds均为invalid config `invalid_effective_window`；
- 有 time bounds 时 `evaluation_time` 必须是 exact datetime scalar或现有 datetime-dtype
  column name；缺失/NaT row 返回 truth=unknown、status=not_verifiable、
  reason=missing_evaluation_time；
- before effective or at/after expiration：truth=unknown、status=inactive、
  reason=outside_effective_window；inactive 不等于 false；
- runtime evaluation time与合法bounds awareness不一致时逐row
  `unknown/not_verifiable/timezone_mismatch`；不抛config error、不localize、不删除timezone；
- active normal evaluation：status=available；unknown truth 的 reason 是
  missing_operand、type_mismatch、timezone_mismatch 或 nonfinite_operand；
- row datetime literal/column-column awareness mismatch同样是
  `unknown/not_verifiable/timezone_mismatch`；aware datetime按UTC instant比较但不修改输入；
- no bounds 时 evaluation_time 被忽略。

### 12.6 Kernel budgets and security

硬上限冻结为：最大 nesting depth `8`、condition nodes `128`、membership literal
length `100`、literal string length `1024`。边界值允许，超过即 stable config
`ValueError`。Rule count/priority 属于 Tasks 17/18，不属于 kernel。

Cycle validation 使用 object identity；检测 cycle 稳定失败。Kernel 不接受或访问：

```text
eval, exec, Python expression, callable, lambda, module/function path,
$ref, include, template, regex, environment expansion, arbitrary object construction,
plugin, filesystem, network, subprocess, import hook
```

被拒绝对象的错误信息只能读取 `type(value).__name__`、row position和column position；不得调用
其 `str()`/`repr()`。验证/拒绝过程不得触发 `__eq__`、`__ne__`、`__lt__`、`__le__`、
`__gt__`、`__ge__`、`__iter__`、`__contains__`、`__bool__`、`__float__`、`__int__`、
`__str__`、`__repr__`、`__hash__` 或 `__array__`。Data audit使用同一whitelist和拒绝顺序。

Condition/result/input 均不修改；不缓存跨调用状态；不读取当前日期、locale、timezone
default 或 environment。

### 12.7 Future parity contract

Task 16 implementation提供direct kernel tests、audit/atomic parity fixtures和稳定private
import path。每种range/membership/cross-column/monotonic-pair rule必须与直接
`_evaluate_atomic_condition`逐row truth/status/reason/sample positions相同。Task 17 与
Task 18 合同必须各自复用同一 fixtures，证明 atomic/Boolean/missing/type/time/effective
cases 逐 row truth/status/reason 完全相同。Parity test 不允许 monkeypatch 复制 evaluator，
也不要求 Task 16 提前实现 policy/alert objects。

## 13. Errors、structured status、warnings 与 limitations

### 13.1 Exception boundary

Task 16 不新增 public exception class。全部 invalid caller input/config 使用 `ValueError`
和稳定 prefix/key：

```text
data audit config is invalid: <key>
data audit input is invalid: <key>
condition specification is invalid: <key>
condition evaluation is invalid: <key>
```

固定 audit keys：

```text
data_not_dataframe, reference_not_dataframe, duplicate_columns,
non_string_columns, invalid_selector, duplicate_selector, unknown_selector,
conflicting_roles, invalid_feature_availability_mapping, positive_label_without_target,
invalid_positive_label, invalid_threshold, invalid_budget, duplicate_column_rule,
empty_column_rule, unsupported_rule_literal, unsupported_scalar_type,
membership_budget_exceeded, max_columns_exceeded
```

固定 kernel keys：

```text
invalid_node_type, invalid_operator, invalid_operand, unknown_column,
invalid_children, invalid_literal, invalid_membership, invalid_between,
invalid_effective_window, invalid_version, descendant_effective_window,
empty_membership_collection, child_version_not_allowed, unsupported_scalar_type,
condition_depth_exceeded, condition_nodes_exceeded, membership_budget_exceeded,
string_budget_exceeded, condition_cycle, evaluation_time_invalid,
unsupported_dtype
```

`empty_membership_collection`、`unsupported_scalar_type`、`child_version_not_allowed`和
`timezone_mismatch` 是master closed reason inventory的一部分；前三者是invalid config/schema
exception keys，runtime `timezone_mismatch`是structured not-verifiable reason。不得将invalid
spec转为unknown。

Public audit委托atomic kernel后，如caller `ColumnAuditRule`触发
`membership_budget_exceeded`/`unsupported_scalar_type`，只将prefix翻译为
`data audit config is invalid`并保留同一key及exception cause；不得在audit层重新验证或改变
kernel语义。Data cell触发`unsupported_scalar_type`仍使用`data audit input is invalid`。

### 13.2 Result boundary

- finding detected：available structured result，不是 exception；
- prerequisite declaration absent：unavailable；
- zero denominator/empty side：undefined；
- irrelevant dtype/mode：not_applicable；
- metadata/type support/budget不足：not_verifiable；
- duplicate scan/unique inspection/collinearity/pattern truncation 等可安全局部跳过的资源
  情况必须返回 warning/limitation，不静默跳过；
- `max_columns`、config budgets、kernel structural budgets 是 hard exceptions，因为超过
  后无法可靠构造 bounded schema。

Warnings 固定顺序：

```text
large_input
duplicate_scan_skipped
unique_inspection_skipped
category_levels_truncated
missing_patterns_truncated
collinearity_columns_truncated
insufficient_drift_rows
point_in_time_not_verifiable
```

Limitations 固定顺序：

```text
in_memory_single_process
structural_identifier_evidence_only
association_not_causation
target_proxy_false_positive_possible
caller_declared_time_provenance
no_automatic_leakage_repair
budget_limited_evidence
```

## 14. Determinism、immutability 与 resource boundaries

- DataFrame values/index/dtypes/column order 均不变；config/roles/rules/condition tuples 不变；
- row position 是唯一 row evidence identity；duplicate index 不影响任何 alignment；
- normalization 只发生于临时 scalars，不写回；
- no random sampling；same input/config/version produces same table values/order/dtypes、
  findings、samples、warnings、limitations；
- 不依赖 hash order、当前日期、locale、machine timezone、thread count 或 external state；
- float calculations 使用 float64；rate denominator 必须记录；
- finding samples 是最小触发 row positions 的前 `max_finding_samples`；requested/actual
  count 与 truncation reason 由 count/denominator/warning 表达；
- `max_columns` 同时约束 current columns、reference columns 和 union width；config 允许
  1--500，0-column input 仍因 observed width `0 <= max_columns` 而合法；
- `max_missing_patterns` 1--1000；`max_finding_samples` 0--100；
- `max_unique_inspection_rows` 1--1,000,000；超过时 top/category/proxy evidence局部
  not_verifiable，不使用 first-N sample伪装完整统计；
- `max_category_levels` 2--100,000；`max_collinearity_columns` 2--200；
- `duplicate_scan_row_limit` 1--5,000,000；`max_column_rules` 0--500；
- `input_n_rows > 1,000,000` 固定 warning `large_input`；仍执行缺失 count/profile，昂贵
  calculations 按各自 budget skip；
- 不新增并行、distributed、approximate sketch 或 runtime dependency。

## 15. Test strategy and matrix

### 15.1 Public API/schema

- exact signature、keyword-only boundary、五个 export order；
- four frozen dataclass field order/type hints/defaults；
- 十四张 DataFrame exact columns/order/dtypes，包含 empty tables及每个逐指标status group；
- result不引用input/reference/current，不保留config/roles/rule对象、repaired/raw DataFrame、
  estimator/Figure/private condition；批准的十四张evidence DataFrames必须存在；
- positive label和allowed/special/range/membership/sentinel literals不在任何result cell；
  sanitized fingerprint repeatable，且不同同family/count sensitive literals不明文泄露并可共享
  sanitized fingerprint；
- v0.1 `__all__` relative order 和 Task 15 five exports/signatures/fields 不变。

### 15.2 Roadmap roles、input 与 immutability

- non-DataFrame、duplicate columns、non-string columns、unknown/duplicate selector、role
  conflicts、features intentional overlap、missing reference roles；
- score/fold/action/policy/cost/exposure/constraint roles的unknown、duplicate、cross-role conflict、
  profile/provenance；score缺partition/fold provenance、constraint missing和negative exposure；
  score path不调用metrics，fold path不生成fold/OOF；
- window/horizon/as-of safe、violation、missing pair和not-verifiable；missing metadata绝不safe；
- duplicate index 两侧 position-safe；reference/current columns deliberately reordered仍按exact
  names匹配，新增/缺失列不按position错配；
- before/after deep snapshots 验证 data/reference/config/roles/rules 不变；
- no raw sample/index/category/target/sentinel values in result。

### 15.3 Quality/profile

- empty dataset、zero features、duplicate rows/index、all-missing、missing boundary 0.40、
  constant、near-constant exact 0.95、high-cardinality `>50 and >0.50` boundaries；
- suspected ID exact 0.98/minimum 20 且不读取列名；mixed NumPy/built-in types；
- NaN/positive/negative infinity；empty/whitespace strings；datetime-role mismatch；
- range inclusive/exclusive、allowed values bool/int distinction、special sentinel、
  not-after、nondecreasing with/without group；
- audit/atomic parity：每个range/membership/cross-column/monotonic pair以相同值/operator直接调用
  audit和`_evaluate_atomic_condition`，逐row比较true/false/unknown、missing、type mismatch、
  nonfinite、timezone和sample positions；audit membership tuple 100接受、101以同一kernel key
  失败；
- one-row numeric的mean available/std undefined、all-missing numeric、categorical cardinality
  available/concentration undefined、target class available/positive unavailable、slice size
  available/target rate unavailable；
- numeric/categorical/target/slice profiles 与手算结果/denominators比较；selection outcome
  support gap 只产生 association evidence。

### 15.4 Missingness/drift

- zero rows的`__NO_ROWS__`、zero features with rows的empty bits、all-missing exact pattern、
  普通pattern missing count/cells、`__OTHER__` min/max/missing-cell aggregation与sample budget；
- pattern frequency/ties/order/exact-before-other/budget boundary；
- reference/current exact name match、different row counts、duplicate indices；
- zero reference missing-rate relative semantics、absolute/relative threshold exact boundary；
- current added/missing columns、dtype/logical/role simultaneous change、`primary_change`
  precedence、new-all-missing、recovered；
- empty reference/current、insufficient sample warning、pattern difference ordering；
- reference/current pattern schema mismatch=`pattern_schema_mismatch`；
- Task 19 不存在重算路径，result self-contained evidence。

### 15.5 Leakage/point-in-time

- target in features、exact copy、near-copy exact threshold、deterministic categorical proxy；
- high association但非 frozen proxy 不产生 leakage；identifier 不误入 proxy；
- partition未声明not_applicable且无warning；missing partition value、stable ordinal；duplicate
  payload/identifier/group跨partition；
- shift threshold None不生成finding、minimum support上下界、threshold equality触发且只info；
- fold未声明not_applicable、missing fold、identifier/group跨fold、fold/partition一致性；
- missing/singleton group；missing provenance；
- PIT全部未声明not_applicable；shared availability、complete map、partial map、missing observation、
  unmapped feature不静默skip，dataset safe要求全部features可验证；
- feature availability equality safe、after observation violation；observation/event/outcome/
  label exact endpoints；observation `< cutoff` boundary；feature `<= cutoff` boundary；
- duplicate entity-time、time reverse order、safe case、not-verifiable case；
- 不产生 fraud finding，不调用 Task 15 folds/metrics。

### 15.6 Condition kernel

- 每个 closed operator、column/literal/column-column；
- complete AND/OR/NOT truth tables、unknown propagation 和 deterministic short-circuit；
- AND/OR 0或1 child、NOT 0或2 children、atomic children非空全部stable拒绝；多个unknown
  short-circuit/provenance按first contributing child reason；
- missing values、both-missing equality、is-missing operators；
- numeric/string/bool/datetime/category/approved NumPy normalization；bool/int separation；
- nonfinite、type mismatch、timezone aware/naive；
- empty `in`/`not_in`拒绝；root version返回、child version无论与root相同或不同均拒绝；
- aware/naive static bounds、static mismatch invalid、runtime mismatch和row datetime mismatch；
- `[effective_from, expires_at)` exact endpoints、missing evaluation time、inactive semantics；
- depth 8/9、nodes 128/129、membership 100/101、string 1024/1025、cycle；
- unknown operator/field、callable/mapping/set/expression-like literal rejection；
- returned RangeIndex row positions、condition/input immutability、repeatability。

### 15.7 Security adversarial fixtures

- malicious class的`__eq__`、`__ne__`、`__lt__`、`__le__`、`__gt__`、`__ge__`、
  `__iter__`、`__contains__`、`__bool__`、`__float__`、`__int__`、`__str__`、`__repr__`、
  `__hash__`、`__array__`全部抛错并计数；
- quality、missingness、atomic comparison、membership和category paths均在`pd.isna`前拒绝；
  所有dunder count必须为0且input不变；
- malicious object位于Boolean short-circuited descendant时仍在whole-tree pre-scan拒绝；
- malicious object index在`Index.duplicated`前以column position `-1`拒绝且dunder count为0；
- membership container exact tuple、elements先whitelist、built-in subclass/user subclass拒绝。

### 15.8 Closed reasons、linkage、ordering、resources和compatibility

- every stable error prefix/key；available/unavailable/undefined/not_applicable/not_verifiable；
- `role_absent_in_reference`、`pattern_schema_mismatch`、无sample finding、多sample finding；
- 每张detail table的primary `finding_key`、shared detail linkage、detail ordinal、SHA-256 key
  determinism；severity/category/dataset-role/column/metric/sample/detail/key全部tie-break；
- warning/limitation/finding deterministic ordering；sample budget 0/max/overflow；
- max columns、column rules、unique/category/pattern/collinearity/duplicate scan每项合法上限与
  超限/截断行为；
- complete existing v0.1 suite and Task 15 suite unchanged；
- public import/distribution clean-install with no new dependency；workflow/report/CLI unchanged。

| Contract or risk | Test type | Location | Success evidence |
|---|---|---|---|
| Public opt-in surface | contract/compatibility | `tests/test_public_api.py` | exact five exports appended, old exports unchanged |
| Audit math/schema | unit/hand fixture | `tests/test_data_audit.py` | exact values/dtypes/status/order |
| Leakage/time boundaries | unit/adversarial | `tests/test_data_audit.py` | safe/violation/not-verifiable separated |
| Shared atomic parity | cross-path unit | both Task 16 test files | no second comparison implementation |
| Shared truth/security | private/adversarial unit | `tests/test_condition_kernel.py` | all operators/truth tables/budgets/zero-dunder rejections |
| Distribution isolation | artifact smoke | `tests/test_distribution.py` | wheel/sdist imports, no extra dependency |
| v0.1/Task 15 compatibility | regression/full suite | existing tests | no signature/behavior weakening |

## 16. Documentation、packaging 与 examples

- 不修改 `pyproject.toml`、core/optional dependencies、entry points、Python support 或 version；
- implementation 完成后 README 只增加一个 opt-in single/reference example，`docs/api.md`
  记录 exact five symbols/status semantics；
- 不新建 example file，不接入 workflow/report/CLI；
- examples/documentation 不暴露 `_condition_kernel`，不声称自动修复、fraud detection、
  production safety 或 v0.2 已发布；
- package version 保持 `0.1.0`，直到后续 release governance 明确授权。

## 17. Ordered implementation plan

1. **Private truth foundation**：只创建 `_condition_kernel.py` 和 direct tests；验证atomic/tree
   entries、operators、truth tables、root version、time/security/budgets，无 public export。
2. **Public contracts and validation**：创建 `data_audit.py` dataclasses、十四张empty typed
   tables、sanitized provenance/fingerprint和selector/config validation；验证signatures、
   immutability、stable errors。
3. **Single-data vertical slice**：实现 dataset/column/numeric/categorical/target/missing-
   pattern profiles和 declarative column rules；以 hand fixtures 验证 thresholds/dtypes。
4. **Comparison vertical slice**：实现 exact-name schema/missingness/pattern drift；验证 empty/
   zero-baseline/new/missing/all-missing/recovered/budgets。
5. **Leakage vertical slice**：实现direct/proxy/partition/fold/roadmap-input/time/PIT evidence和
   audit/atomic parity；验证Task 15 ownership、no-fraud/no-causation boundaries。
6. **Assembly and compatibility**：冻结 finding/order/warnings/limitations，追加五个 exports，
   更新批准文档，运行 v0.1/Task 15 regression、full verification 和 distribution gates。

每个 slice 必须连同 tests 提交 review；不得先写 Task 17/18 adapters 或把 private kernel
公开化。

## 18. Future implementation allowlist

Task 16 合同取得 Approved — Go 后，implementation 只允许创建或修改：

```text
src/sharper/data_audit.py                              (new; public Task 16 owner; consumes private atomic entry)
src/sharper/_condition_kernel.py                       (new; sole private atomic/tree truth owner)
src/sharper/__init__.py                                (only five approved exports)
tests/test_data_audit.py                               (new)
tests/test_condition_kernel.py                         (new)
tests/test_public_api.py                               (only Task 16 + compatibility assertions)
tests/test_distribution.py                             (only installed exports/no-dependency smoke)
docs/decisions/task16-data-quality-leakage-contract.md (status/evidence only after approval)
SPEC.md                                                (only Task 16 status/API index sync)
IMPLEMENTATION_PLAN.md                                 (only Task 16 status/link sync)
README.md                                              (only implemented opt-in usage/status)
docs/api.md                                            (only implemented Task 16 API)
```

No existing schema/summary/quality file modification is required：Task 16 只调用 Task 03 public
`infer_schema`，且`data_audit.py`只额外消费同一allowlist内Task 16 private atomic kernel；不抽取
或更改v0.1 private helpers。默认禁止：

- `src/sharper/schema.py`、`summary.py`、`quality.py` 及其 frozen contracts/tests；
- Task 15 modules、contract、tests 或 exports order/behavior；
- `analysis.py`、features、visualization、modeling、evaluation、risk_validation；
- workflow、reporting、CLI、examples；
- `pyproject.toml`、dependencies、lock files、version；
- Tasks 17--20 modules/contracts/implementation；
- broad rewrite/weakening/removal of v0.1 or Task 15 tests；
- cleanup/transform/repair output 或任何 fraud capability。

不得使用“其他必要文件”开放兜底。若实现确实需要allowlist外文件，必须停止实现并取得用户
明确授权，按`AGENTS.md`判断是否属于新的系统性P0/基础假设变化；不能在code diff中自行扩张，
也不能以该情形为由自动重开full audit。

## 19. Completion gates

Task 16 只有全部满足后才可取得 implementation review Go：

1. 本合同经一次独立 full contract review、targeted fixes、bounded closure 后改为
   Approved — Go；
2. five public symbols、dataclass fields/signatures、十四张table schemas、sanitized provenance、
   status/reasons 已冻结；
3. quality/missingness/drift/leakage/point-in-time hand fixtures 全部通过；
4. private atomic/tree kernel、complete tri-state truth tables、root-version、effective/
   expiration、zero-dunder security 和 budgets 全部通过；
5. Task 17/18 未实现；audit/atomic parity和future parity hook只保留private import path/tests；
6. inputs/config/conditions unchanged，duplicate-index position safety 和 deterministic
   ordering 通过；
7. v0.1 与 Task 15 signatures、fields、behavior、tests 未删除/搬移/弱化；
8. 无新 dependency/version/workflow/report/CLI/Task 17--20 implementation；
9. `bash scripts/verify-uv-env.sh` 通过，且所有 Python commands 使用 `.venv/bin/python`；
10. `.venv/bin/python -m pytest`、Ruff lint/format、build、wheel/sdist clean install 和
    distribution smoke 全部通过；
11. `git diff --check`、`git diff --cached --check` 通过，diff 只在 allowlist；
12. 一次独立 implementation full review 后按 targeted/bounded closure 获得 final Go。

## 20. Contract review gate 与当前结论

唯一一次 full contract review 登记的 10 个 P1 已完成 targeted fixes，并经 bounded contract
closure 确认全部关闭；closure verdict 为 `Go`，P0/P1/P2 均为 `0`。合同阶段已终止，不得再次
进行开放式 full contract review。后续 implementation 必须严格遵守本合同及第 18 节 allowlist。

本次批准没有开始 implementation，也没有创建源码/tests/API/export。当前结论是：

```text
Contract status: Approved — Go
Task 16 implementation: Implementation complete — review Go
Tasks 17–20 contracts and implementation: Not started
Package version: 0.1.0
```
