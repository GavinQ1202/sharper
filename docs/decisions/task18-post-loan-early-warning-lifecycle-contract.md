# Task 18 贷后预警与生命周期监测精确合同

**状态：Approved — Go。**

**Contract drafting：complete。** 这是 contract-first 设计记录，不是 implementation、full
contract review、release 或 Task 19/20 授权。Task 18 implementation 为 **Not started**。

**正式名称：** Task 18 — Post-loan Early Warning and Lifecycle Monitoring。

## 1. 权威依据、目标与非目标

本合同服从根目录 `AGENTS.md`、已批准的 `docs/decisions/v02-roadmap-contract.md`、
`SPEC.md`、`IMPLEMENTATION_PLAN.md`、Task 15 frozen risk-validation contract 和 Task 16
frozen data-audit/private-condition-kernel contract。冲突时停止并先修订治理文档；不得在
implementation 中自行解释。

Task 18 的唯一目标是：对 caller 提供的单个 prepared DataFrame，在显式
`entity × observation_time` 上只使用当时已可用的信息，离线执行 caller-declared warning
scenarios，形成 raw rule hits、notifications、alert episodes、future-event backtest 与
caller-defined lifecycle state/transition evidence，并输出可审计的 portfolio/time/segment/
vintage/cohort summaries。结果只用于 observe、classify、track、measure、summarize 和 report。

Task 18 明确不：

- 读取文件、join 多表、猜测客户/账户关系、字段角色、MOB/DPD、状态或 alert level；
- 建立 streaming、scheduler、background task、notifier、service、database 或 external state；
- 发送通知、联系客户、执行催收、修改账户/额度、批准/拒绝或分配任何业务 action；
- fit estimator、calibrator、peer model、threshold 或 rule；不搜索、推荐、优化或选择 winner；
- 实现 survival、causal/action-effect、反欺诈、设备/IP/图网络能力；
- 导入/消费 Task 17 policy/action result，或提前实现 Task 19 explanation/governance、Task 20
  workflow/report/CLI/serialization；
- 把 historical association、capture、roll-rate 或 cure difference 表述为 warning 的因果效果。

## 2. Package architecture、ownership 与依赖方向

未来实现只新增一个聚焦 public owner：

```text
src/sharper/
├── _condition_kernel.py       # Task 16 private closed truth owner；只读消费
├── risk_validation.py         # Task 15 frozen optional evidence；只读消费
├── data_audit.py              # Task 16 frozen evidence；只读消费
└── lifecycle_monitoring.py    # Task 18 public owner；当前不存在
```

| 模块 | 唯一职责 | 可依赖 | 禁止 |
|---|---|---|---|
| `_condition_kernel` | atomic/Boolean/missing/date comparison 与三值 truth | pandas、NumPy | Task 18 state、window、episode、metric；public DSL |
| `lifecycle_monitoring` | point-in-time signal alignment、history features、warning persistence/cooldown/episode、event matching、lifecycle transition、summary 与 provenance | pandas、NumPy、stdlib、Task 16 private kernel、Task 15/16 frozen result types | workflow/reporting/CLI/visualization、Task 17/19/20、fit/IO/action execution、第二套 condition evaluator |

Ownership 冻结如下：

- exact scalar/type/literal safety、closed atomic/Boolean operators 和 three-valued comparison
  truth 只属于 Task 16 private kernel；
- score/ranking/probability、positive-event、OOF/fold/maturity source semantics 只属于 Task 15；
- input/missingness/leakage profiling 只属于 Task 16；Task 18 不重算；
- entity-time alignment、prior-only history signal、warning persistence/cooldown/notification/
  episode、future-event matching、lifecycle state/transition、vintage/cohort/roll/cure 和本合同
  summaries只属于 Task 18；
- Task 19 未来只能消费本合同 frozen result，不得重放 rules/episodes/metrics；Task 20 未来只编排。

Task 18 不修改 private kernel。本合同确认其 atomic/Boolean inventory 足够：Task 18 先构造
position-safe temporary signal Series，再编译到 Task 16 `_ConditionNode` 并调用唯一
`_evaluate_condition`。Persistence、history、episode 与 transition 不是 comparison truth，留在
Task 18 owner。

## 3. 唯一 public API 与 exports

Task 18 精确新增七个、且仅七个 opt-in public symbols：

```python
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

import pandas as pd


@dataclass(frozen=True)
class MonitoringCondition:
    kind: Literal["atomic", "and", "or", "not"]
    operator: Literal[
        "eq", "ne", "lt", "le", "gt", "ge",
        "in", "not_in", "between", "is_missing", "is_not_missing",
    ] | None = None
    left_kind: Literal[
        "column", "ranking_score", "event_probability",
        "prior_value", "change", "trend", "history_mean", "peer_deviation",
        "prior_state", "state_transition",
    ] | None = None
    left: str | None = None
    right_kind: Literal["literal", "column"] | None = None
    right: object | None = None
    window: Literal["recent", "history"] | None = None
    children: tuple["MonitoringCondition", ...] = ()


@dataclass(frozen=True)
class EarlyWarningRule:
    rule_key: str
    priority: int
    alert_level: str
    condition: MonitoringCondition
    persistence_observations: int = 1
    resolution_observations: int = 1
    cooldown: timedelta = timedelta(0)
    enabled: bool = True
    effective_from: datetime | None = None
    expires_at: datetime | None = None
    description_key: str | None = None


@dataclass(frozen=True)
class WarningScenario:
    scenario_key: str
    scenario_kind: Literal[
        "no_alert", "single_threshold", "rule_set", "model_score", "model_plus_rules"
    ]
    rules: tuple[EarlyWarningRule, ...]


@dataclass(frozen=True)
class LifecycleState:
    state_key: str
    state_rank: int
    priority: int
    condition: MonitoringCondition
    terminal: bool = False
    enabled: bool = True
    description_key: str | None = None


@dataclass(frozen=True)
class LifecycleMonitoringConfig:
    monitoring_key: str
    monitoring_version: str
    analysis_as_of: datetime
    entity_column: str
    observation_time_column: str
    available_time_column: str
    condition_feature_columns: tuple[str, ...]
    event_time_column: str | None
    positive_event_key: str | None
    prediction_horizon: timedelta | None
    horizon_end_inclusive: bool
    recent_window: timedelta
    history_window: timedelta
    history_start_inclusive: bool
    expected_observation_interval: timedelta | None
    period_unit: Literal["day", "week", "month", "quarter"]
    time_zone: str | None
    scenarios: tuple[WarningScenario, ...]
    reference_scenario_key: str
    alert_level_ranks: tuple[tuple[str, int], ...]
    states: tuple[LifecycleState, ...]
    default_state_key: str
    unknown_state_key: str
    allowed_transitions: tuple[tuple[str, str], ...] = ()
    adverse_state_keys: tuple[str, ...] = ()
    cure_state_keys: tuple[str, ...] = ()
    cohort_time_column: str | None = None
    cohort_column: str | None = None
    peer_group_columns: tuple[str, ...] = ()
    peer_reference_start: datetime | None = None
    peer_reference_end: datetime | None = None
    ranking_score_column: str | None = None
    ranking_score_direction: Literal["higher_risk", "lower_risk"] | None = None
    exposure_column: str | None = None
    loss_fraction: float | str | None = None
    observed_loss_column: str | None = None
    observed_loss_available_time_column: str | None = None
    observed_loss_is_mature_snapshot: bool = False
    segment_columns: tuple[str, ...] = ()
    time_frequency: Literal["day", "week", "month", "quarter"] = "month"


@dataclass(frozen=True)
class LifecycleMonitoringResult:
    monitoring_key: str
    monitoring_version: str
    monitoring_fingerprint: str
    input_n_rows: int
    entity_count: int
    evaluable_observation_count: int
    requested_scenario_count: int
    requested_rule_count: int
    active_rule_count: int
    requested_state_count: int
    observation_history: pd.DataFrame
    rule_evaluations: pd.DataFrame
    notifications: pd.DataFrame
    alert_episodes: pd.DataFrame
    event_matches: pd.DataFrame
    state_history: pd.DataFrame
    state_transitions: pd.DataFrame
    monitoring_summary: pd.DataFrame
    scenario_comparison: pd.DataFrame
    lifecycle_summary: pd.DataFrame
    provenance: pd.DataFrame
    warnings: tuple[str, ...]
    limitations: tuple[str, ...]


def monitor_lifecycle(
    data: pd.DataFrame,
    config: LifecycleMonitoringConfig,
    *,
    risk_validation: "BinaryRiskValidationResult | None" = None,
    data_audit: "DataAuditResult | None" = None,
) -> LifecycleMonitoringResult: ...
```

字段、参数、keyword-only 边界、Literal vocabulary 与默认值均 frozen。Dataclasses shallow
frozen；constructor 不验证，函数 whole-config-first 验证。不得增加 alias、method、builder、
manager、registry、`**kwargs`、public exception 或 public plot function。

`sharper.__all__` 只在 Task 17 suffix 后依次追加：

```text
MonitoringCondition
EarlyWarningRule
WarningScenario
LifecycleState
LifecycleMonitoringConfig
LifecycleMonitoringResult
monitor_lifecycle
```

Public docstrings 必须包含参数、返回、errors、副作用、missing/time strategy 与最小示例。

## 4. Public condition、signals 与 Task 16 kernel

`MonitoringCondition` 是 closed typed pure-data tree，不是通用 DSL。不得接受 dict/mapping、
callable、expression、regex、script、module path、template、`$ref`、private kernel object 或动态
operator。

Condition column role inventory封闭为：

```text
condition-eligible:
ordinary_point_in_time_feature, ranking_signal,
approved_probability_signal, approved_derived_signal

condition-forbidden reserved:
entity_identifier, observation_time, available_time,
future_event_indicator, future_event_time, observed_loss,
outcome_maturity, outcome_evaluability,
backtest_only_exposure, backtest_only_loss_fraction,
segment, cohort, vintage, source_provenance, internal_alignment
```

`condition_feature_columns`是caller显式声明的`ordinary_point_in_time_feature`完整allowlist。每个
column-valued operand独立解析role：`left_kind="column"`的`left`、`right_kind="column"`的`right`，
以及每个derived-signal source都只能引用该tuple，使用完全相同的allowlist和forbidden-role inventory。
Tuple内column不得同时承担任一reserved role，也不得与entity/time/event/loss/exposure/loss-fraction/
segment/cohort/peer/alignment role重叠。
若caller确实需要与exposure相同的业务量作为warning feature，必须在prepared input提供独立column并
显式加入`condition_feature_columns`；不得复用`exposure_column`。Ranking和Task 15 probability只
通过各自专用`left_kind`进入condition。左侧安全绝不信任右侧：`ordinary_feature > future_event_time_column`、
`future_event_indicator_column == ordinary_feature`及任何一侧的forbidden role均在compile/evaluate前
失败`lifecycle condition: forbidden_condition_role`，不区分左右错误key。

Condition operand matrix封闭如下。`derived signal`指`prior_value/change/trend/history_mean/peer_deviation`
等现有`left_kind`；当前public `right_kind`只允许`literal`或`column`，故RHS derived-signal无表示且
禁止，未列组合一律禁止：

| left kind | right kind | role/taint validation | arity 与 Task 16 ownership |
|---|---|---|---|
| `column` | `literal` | left必须condition-eligible；literal不作column role lookup | arity与exact scalar/literal safety由Task 16 |
| `column` | `column` | 两侧独立condition-eligible；无taint lineage | 仅Task 16批准的column comparison/type compatibility |
| derived signal | `literal` | source及lineage必须condition-eligible且untainted；literal不作column role lookup | signal source/window及Task 16 literal comparison |
| derived signal | `column` | derived source/lineage和RHS column独立合法且untainted | 仅Task 16批准的comparison/type compatibility |
| `column` | derived signal | 禁止：`right_kind`没有derived-signal representation | kernel compilation前拒绝 |
| derived signal | derived signal | 禁止：`right_kind`没有derived-signal representation | kernel compilation前拒绝 |
| logical tree | no operand | 无column/signal operand；递归所有child | `and/or/not` shape rules |
| unary operator | empty RHS only | 每个supplied operand仍须role-validated；RHS不可被忽略 | `right_kind=None, right=None`后才可kernel compilation |

`right_kind="literal"`不参与column-role lookup，但仍逐项继承Task 16 frozen exact-scalar与literal
safety；literal永远是值而非column name、dynamic lookup、attribute、callable、expression或object protocol，
不得建立第二套literal semantics。

- `atomic`：无 children；必须有 operator/left_kind。Missing operators必须是冻结empty RHS
  `right_kind=None, right=None`；其他 operators必须有 right。结构验证后，对每个已提供的left/right
  column或signal operand先作独立role/taint validation，再作operator-specific arity和kernel compilation；
  unary operator携带的任何RHS（包括forbidden column）绝不静默忽略。
- `and/or`：至少两个 children，其他字段为 `None`；`not` 恰好一个 child。
- literal/type/membership/between/timezone 规则逐项继承 Task 16 kernel；between 是闭区间。
- `column` 读取当前 observation row；`ranking_score`/`event_probability` 的 `left=None`，只允许
  literal RHS 和 comparison operators；不得 membership、between 或 column RHS。
- `prior_value`、`change`、`trend`、`history_mean`、`peer_deviation` 的 `left` 是 exact source
  column；`window` 必须分别为 `recent/history`，但 `prior_value` 允许 `None` 表示最近 prior。
- `prior_state/state_transition` 的 `left=None, window=None`，只允许 `eq/ne/in/not_in` 与literal
  RHS。它们分别读取上一条consecutive lifecycle state key和本次closed transition kind；entity
  first row、gap reentry或state unknown时为unknown。`LifecycleState.condition`禁止这两种source，
  避免state定义递归。
- derived numeric signals只支持 real numeric non-bool source。`change=current-prior`；trend 是
  eligible history rows上、以 elapsed seconds为x的 float64 least-squares slope；history_mean是
  arithmetic mean；peer_deviation是`current - frozen peer mean`。不足2点的trend unknown，其他
  history signal无prior支持时 unknown；不以0填充。
- `peer_deviation` 要求 `peer_group_columns` 和 `peer_reference_end`；baseline 只用
  `peer_reference_start <= observation_time <= peer_reference_end`且
  `available_time <= peer_reference_end`的rows；`peer_reference_start=None`表示无下界。每组至少
  2个finite值。不存在组或支持不足为unknown，不回退全局。对每个被评价row另强制
  `peer_reference_end < observation_time`；不满足时signal为
  `not_verifiable/peer_reference_not_prior`，不得自动缩短window或删除future candidates后继续。
  每个scenario独立构造immutable baseline；不得跨scenario共享mutable cache/state。
- ranking lower-risk 的不等式在编译时方向翻转；Task 15 normalized ranking不翻转。
- event probability必须finite `[0,1]`且代表显式 `positive_event_key`；ranking/margin永不升级为
  probability。唯一来源是通过第12节完整alignment的Task 15 evidence；DataFrame probability入口
  不存在。Probability只在Task 15 evaluable positions可用，其他positions为
  `unavailable/probability_unavailable`，不伪造0。

每个derived signal必须携带transient source-role lineage。左或右的任何已支持signal operand lineage只要
包含future event、observed loss、future maturity/evaluability、post-horizon evidence或
available-after-cutoff evidence，即为tainted，并在编译前整体失败
`ValueError("lifecycle condition: forbidden_condition_role")`。Taint递归传播，禁止
`future_event -> derived_signal -> condition`绕过reserved roles；当前RHS derived signal并无public
表示，属于上表禁止组合。

Whole-tree validation固定为：validate全部scenarios -> 全部rules/states -> 对每个condition node先作
structural validation -> 递归检查左右两侧每个已提供的column-valued或derived-signal operand及其lineage
-> 拒绝forbidden role/taint -> operator-specific arity validation -> 最后才compile/evaluate。相同
condition-role allowlist、forbidden-role inventory与taint规则对两侧对称适用。后置节点、未命中branch、
inactive rule/state和可能被short-circuit的branch同样验证，operator语义上未使用的supplied operand也
不能跳过验证。该错误不含raw column value、entity或literal。Private kernel specification与reserved-role
错误均映射第16节统一prefix；reserved role exact error为
`lifecycle condition: forbidden_condition_role`。

Public tree完整验证、derived Series构造完成后，atomic/Boolean truth 只能来自Task 16 kernel。
Kernel的`true/false/unknown`、missing/type/nonfinite/timezone reasons逐row原样映射到本合同 closed
vocabulary。Parity tests必须证明全部operators/Boolean/missing/date/timezone与Task 16一致。

逐observation计算顺序固定为：point-in-time base/history/peer signals -> lifecycle state candidates ->
本次transition kind -> warning conditions -> persistence/notification/episode。State classification
不读取warning state，因此依赖无环。

## 5. Entity identity、row identity 与 privacy

- `data`必须是DataFrame，columns是唯一exact non-empty strings；index完全忽略。
- `row_position`是输入顺序0-based built-in int；任何sort后仍保留该identity。
- `entity_column` values只用于分组；missing/non-hashable/entity值为invalid input。Exact scalar
  whitelist继承Task 16；bool、int、finite float、str、date/datetime允许，其他object拒绝。
- entity排序不依赖raw value：第一次出现在input的entity为`entity_position=0`，之后依首次出现
  顺序编号。Result、errors、warnings、limitations、provenance、repr与fingerprint均不得包含raw
  entity value、pandas index、raw event/segment/cohort/peer value。
- `(entity, observation_time)`必须唯一；duplicate立即
  `ValueError("lifecycle input schema is invalid: duplicate_entity_observation_time")`，不deduplicate。
- 同entity按`observation_time`升序、再row_position升序处理；输出全局按第14节排序。
- segment、cohort、peer raw keys不直接返回；使用按首次出现顺序分配的
  `segment_position/cohort_position/peer_position`和sanitized type-family provenance。

## 6. Point-in-time time model 与 no-lookahead

Temporal scalar只允许`type(value) is datetime.datetime`或`type(value) is pandas.Timestamp`。
Exact-type检查先于pandas/NumPy conversion、missing、timezone或`utcoffset()`调用；str、
`datetime.date`、`numpy.datetime64`、subclass、arbitrary datetime-like、`None`和`NaT`均失败。
所有temporal config/input cells必须全naive或全aware；混用整体失败。Aware value必须具有安全有效
`utcoffset()`，验证后统一转换为UTC进行比较和输出；`time_zone`必须等于caller声明的source zone
name。Naive value保持naive，不localize、不猜timezone。已构造aware timestamp的DST歧义由其明确
offset决定；Task 18不解析或构造ambiguous/nonexistent local time。禁止环境timezone和current time；
`analysis_as_of`是唯一as-of source。

对observation `t`：

```text
current row usable       iff observation_time == t
                           and observation_time <= analysis_as_of
                           and available_time <= observation_time
                           and available_time <= analysis_as_of
history row usable       iff source_observation_time < t
                           and source_available_time <= t
recent history interval  (t - recent_window, t) by default
history interval         (t - history_window, t) by default
```

当`history_start_inclusive=True`时左端改为闭；右端永远开，current row永不作为自身history。
`recent_window > 0`、`history_window >= recent_window`。Observation/available time missing整体失败；
`observation_time > analysis_as_of`、`available_time > analysis_as_of`或
`available_time > observation_time`也整体失败，不静默排除。Future-only value不得回填此前signal、
peer/history baseline、warning或state。

Unsorted input合法，内部按`entity_position, observation_time, row_position`处理；pandas index不参与。
Duplicate `(entity, observation_time)`无论available time是否不同均整体失败。Stable input keys冻结为：

```text
observation_time_missing, available_time_missing, observation_after_as_of,
available_after_as_of, available_after_observation, datetime_type_invalid,
datetime_awareness_mismatch, datetime_timezone_invalid,
duplicate_entity_observation_time
```

`expected_observation_interval=None`表示不声明规则cadence；否则相邻observation gap大于该值
标记`non_consecutive`。Gap不合成row、不自动关闭episode、不把missing period当condition false。

Event仅用于backtest。合法future match固定为：

```text
notification_time < event_time < notification_time + prediction_horizon
```

若`horizon_end_inclusive=True`，末端改为`<=`；起点永远open，同刻event不算future capture。
Mature notification要求`notification_time + prediction_horizon <= analysis_as_of`。Immature
horizon为right-censored，不进入precision/false-alert/lead-time denominator。Event time缺失表示
该row没有declared event；`event_time > analysis_as_of`不可见且不匹配。

`event_time_column/positive_event_key/prediction_horizon`必须all-or-none；缺失三者表示纯warning/
lifecycle路径，所有event metrics为not_applicable。一个entity可在不同rows声明多个distinct event
times；同entity相同event time去重为一个event并记录duplicate evidence，不重复denominator。
`event_row_position`取最小source row position，`duplicate_source_row_count`记录被折叠的额外rows；
raw payload不参与identity或tie-break。

## 7. Scenario、rule、alert level 与 precedence

Scenario keys唯一。`reference_scenario_key`必须存在。Scenario kinds语义：

| kind | frozen shape |
|---|---|
| `no_alert` | rules必须空；永不hit/notify，metrics保留undefined/not_applicable语义 |
| `single_threshold` | 恰好1条active rule，condition只使用ranking/probability source |
| `rule_set` | 至少1条rule；不得用score/probability source |
| `model_score` | 至少1条rule；每条只用score/probability source |
| `model_plus_rules` | 至少2条rule，至少一条score/probability且至少一条非score condition |

`current/challenger`不是冻结enum；reference与其他caller-key scenarios在相同input、entity、as-of、
horizon、maturity和scope上比较。Task 18不选择winner。

Rule key在scenario内唯一；priority是exact int 0..9999，tie按rule_key。Alert levels由caller
`alert_level_ranks`完整一对一映射；rank exact int 0..9999，多个level可同rank，tie再按priority/
rule_key。示例`none/watch/warning/high/critical`不冻结。

所有enabled且window-active rules均求值，允许multi-hit diagnostics。每个rule独立形成episode；
同observation的primary alert从active rules按`rank降序、priority升序、rule_key升序`选取，不会
抑制其他rule evidence。Disabled/outside-window rules不求值为truth，输出inactive。

Effective boundary为`effective_from <= observation_time`；expires boundary为
`observation_time < expires_at`。`effective_from < expires_at`。Version只属于config；不得从数据
选择版本。

## 8. Persistence、notification、cooldown 与 episode state machine

每个`scenario × entity × rule`按time顺序维护唯一离线state：

```text
clear --true streak reaches persistence_observations--> active/open + notification
active --true------------------------------------------> active
active --false streak reaches resolution_observations-> resolved
resolved --true streak reaches persistence_observations> active/reopen + notification
unknown/inactive/non-consecutive ----------------------> do not increment true/false streak
```

- True/false streak必须连续可评估observations；opposite truth重置另一streak。Unknown不关闭episode，
  但打断连续streak。Non-consecutive gap同时打断streak但不关闭active episode。
- Episode在true streak达到persistence的当前observation开启，`episode_start_time`为当前observation
  time；不回写此前row为active，也不输出此前raw timestamps。
- Active episode在false streak达到resolution的当前observation关闭，`episode_end_time`为当前time；
  此前false tail保留episode open但`condition_truth=false`。False-tail row不算warning hit、不产生
  notification，旧open episode也不能成为该row primary alert；primary只能来自当前row truth=true且
  已达到persistence的eligible rules。达到resolution threshold的当前row关闭episode。
- Reopen创建新`episode_ordinal`；`is_reopen=True`。First episode是ordinal 0。
- 首次open/reopen必发notification。Active episode后续再次满足true时，只有距离上次emitted
  notification `>= cooldown`才发repeated notification。Cooldown为0时每个active true observation
  可notification；cooldown只抑制notification，不抑制raw hit、active state或episode。
- Missing observation不是row，gap不产生notification/resolution。Analysis-as-of时仍active的episode
  `unresolved`；未来row不得改变此前point-in-time state。
- 同一observation同rule最多一个notification。Raw hit、notification、episode分别计数，名称不得
  混用。

Episode duration冻结为：resolved episode的`episode_end_time`是触发resolution threshold的
observation time，`duration_seconds=episode_end_time-episode_start_time`；open episode的
`episode_end_time=pd.NaT`，`duration_seconds=analysis_as_of-episode_start_time`。不得使用最后hit、
最后notification或最后observation time替代resolved end。

## 9. Lifecycle state、transition、roll 与 cure

Lifecycle classification独立于warning scenarios。所有enabled states求值；True候选按
`priority升序、state_key升序`选择primary。无True且至少一个Unknown使用`unknown_state_key`；
全部False使用`default_state_key`。Default/unknown keys必须在states中，二者不同；它们的conditions
仍须合法，但仅作inventory，不参与候选求值，且必须`enabled=True, terminal=False`。

`state_rank`是caller-declared ordered severity，只用于描述transition direction：

```text
same key            transition_kind=stay, direction=flat
different key       transition_kind=change, direction by rank
first observation   transition_kind=entry, direction=not_applicable
gap re-entry        transition_kind=reentry, direction=not_applicable
not allowed         transition_kind=invalid
```

Direction inventory只有`not_applicable/flat/roll_forward/roll_back`。Different keys同rank时direction
为`flat`。State names、ranks和cure mappings不硬编码。`adverse_state_keys`与`cure_state_keys`各自
唯一且不重叠；`is_cure=True`只在from属于adverse且to属于cure时成立，独立于kind/direction。

`allowed_transitions=()`表示所有非terminal transitions可描述；非空时必须列出允许的distinct
from/to pairs。Terminal state只允许self transition。First state observation生成entry row，其
`from_row_position/from_state_key/from_rank`均nullable且为missing。Gap后生成reentry。

每row先计算`candidate_state`。Allowed candidate成为`effective_state`；out-of-inventory、not-allowed
或terminal exit记录`transition_kind=invalid`和candidate diagnostics，但effective state保持last valid
state，`transition_direction=not_applicable`，后续row继续从last valid effective state计算。
Invalid transition不计入roll/cure denominator且`is_cure=False`。Configured default/unknown state keys是
普通合法candidate：first row可成为effective state，后续同样接受allowed/terminal transition检查；
Task 18不猜测合同inventory外state。无自动状态修复、插值或账户操作。

Vintage/cohort age：若`cohort_time_column`声明，按calendar elapsed完整day/week/month/quarter计算
non-negative `period_index`；observation早于cohort为row not-verifiable。若`cohort_column`声明，只
形成caller cohort buckets；两者可同时存在。Calendar month为`12*(year diff)+month diff`并在
observation day/time早于anchor day/time时减1；quarter=`month_index // 3`；day/week用elapsed
whole units。不得从列名或最早observation猜cohort/MOB。

## 10. Event matching、maturity 与 backtest denominators

Matching顺序固定：event按`event_time,row_position`；notification按
`notification_time,scenario,entity,rule`。实现必须对每个entity/scenario使用排序scan或binary
search，不得materialize notification × event Cartesian table。一个event可被同scenario多个notifications覆盖，但
capture/lead-time只归因该scenario内最早eligible emitted notification；一个notification可覆盖
多个events但precision numerator只记1 positive notification。每个scenario独立，不能跨scenario
消耗event。

- `mature event`：`event_time <= analysis_as_of`且存在至少一个同entity observation，其完整
  prediction horizon已成熟并且event落入其future horizon；
- capture/recall denominator是上述unique mature events；无可观察prior opportunity的event不进入；
- precision denominator是mature emitted notifications；positive iff horizon内至少一unique event；
- false-alert share=`false mature notifications / mature notifications`；
- false-positive rate=`alerted mature no-event observations / mature no-event observations`，其中
  observation按scenario primary notification有/无计算；二者不得混名；
- lead time=`event_time - earliest capturing notification_time`；只对captured events；
- no-alert scenario precision、false-alert share、false-positive rate、lead time均undefined而非0；
- exposure/loss只描述关联。Exposure必须finite non-negative real；observed loss必须finite
  non-negative real且成熟。Availability column模式要求`loss_available_time <= analysis_as_of`；
  mature snapshot模式仅由explicit bool声明。两模式互斥于缺失availability declaration。

Task 15 result只提供position-aligned ranking/probability和其frozen mature/evaluable provenance；
Task 18不使用其target值重建event，不重算fold、maturity、calibration或business metrics。Task 18
raw event matching仍只用本合同event time。

### 10.1 Source cardinality 与position availability

每种semantic独立规范化为`not_requested/dataframe/task15`恰好一种；不是所有semantic都支持全部
三种来源：

| Semantic | 批准来源 | 禁止/用途 |
|---|---|---|
| ranking score | DataFrame `ranking_score_column`或Task 15 ranking，二选一 | finite real non-bool；direction必需；`[0,1]`仍非probability |
| event probability | 仅aligned Task 15 probability | DataFrame probability字段不存在；只供condition/expected loss |
| future-event outcome | 仅DataFrame `event_time_column` | Task 15 target不重建event/time/lead time；backtest only |
| exposure | 仅DataFrame `exposure_column` | nonnegative finite；backtest only |
| loss fraction | 仅config finite `[0,1]` scalar或DataFrame column named by`loss_fraction` | expected-loss only；backtest only |
| observed loss | 仅DataFrame `observed_loss_column`及其maturity declaration | actual mature evidence；backtest only |

同一semantic出现两个来源时whole-config-first失败。Stable config/alignment keys至少：
`duplicate_ranking_source, duplicate_probability_source, duplicate_outcome_source,
duplicate_exposure_source, duplicate_observed_loss_source`。后三个keys也防御forged/未来扩展schema；
当前合法public config只允许表中单一来源，绝不读取Task 15 target/exposure/loss作为替代。

DataFrame ranking按row position全量扫描，missing/nonfinite失败；Task 15 ranking/probability只在其
predicted且对应semantic non-missing positions可用，probability另要求`is_evaluable=True`。Excluded、
non-predicted或non-evaluable position使用structured`unavailable/score_unavailable`或
`unavailable/probability_unavailable`；不按entity复制、不以index align、不drop row。Task 15
positive-event语义和probability provenance原样继承，raw positive label不进入result/provenance；
Task 18不校准、重解释或把ranking升级为probability。

Future event、exposure、loss fraction和observed loss只进入本合同批准backtest/summary数学，全部是
condition-forbidden reserved roles。Observed loss使用availability-column或explicit mature-snapshot
模式，二者不可同时声明；缺失component使相关metric not-verifiable，不做partial sum。

## 11. Finite metric contract matrix 与准确数学

Matrix中`M`=`monitoring_summary`、`L`=`lifecycle_summary`；scope aliases由第12节精确定义。
每row的`metric_value=numerator/denominator`仅用于rate/mean；count/sum的metric_value=numerator且
denominator为`pd.NA`；median metric的numerator与metric_value均为median、denominator记录support
count但不执行除法。`minimum_support=1`。Zero denominator固定`undefined/zero_denominator`；source未请求固定
`not_applicable/source_not_requested`；已请求但component缺失/不成熟固定
`not_verifiable/<specific reason>`。Status precedence固定：invalid config/schema/alignment exception ->
source not requested -> inactive -> censored/not mature -> incomplete evidence -> insufficient support ->
zero denominator -> available/computed。Ordering按下表row order。

| metric_key | table | allowed_scope | numerator | denominator | support_unit | required evidence / maturity | unit |
|---|---|---|---|---|---|---|---|
| warning_hit_count | M | warning scopes | current truth=true evaluations | NA | rule_evaluation | evaluable active rules | count |
| warning_observation_rate | M | warning scopes | observations with >=1 current eligible hit | evaluable observations | observation | none | fraction |
| warned_entity_count | M | warning scopes | distinct entities with >=1 current hit | NA | entity | none | count |
| warned_entity_rate | M | warning scopes | warned entities | evaluable entities | entity | none | fraction |
| persistent_warning_count | M | warning scopes | observations with current truth=true in open episode after persistence | NA | observation | episode state | count |
| persistent_warning_rate | M | warning scopes | persistent-warning observations | evaluable observations | observation | episode state | fraction |
| notification_count | M | warning scopes | emitted simulated notifications | NA | notification | none | count |
| notifications_per_entity | M | warning scopes | emitted notifications | evaluable entities | entity | none | count/entity |
| overlap_count | M | scenario/segment/time | observations with >=2 current true rules | NA | observation | none | count |
| conflict_count | M | scenario/segment/time | observations with >=2 current true rules tied at highest alert rank | NA | observation | alert ranks | count |
| episode_count | M | warning scopes | opened episodes | NA | episode | episode state | count |
| open_episode_count | M | warning scopes | episodes open at analysis_as_of | NA | episode | as-of | count |
| resolved_episode_count | M | warning scopes | episodes resolved by analysis_as_of | NA | episode | as-of | count |
| episode_duration_mean | M | warning scopes | sum duration_seconds | episodes | episode | episode state | seconds |
| episode_duration_median | M | warning scopes | median duration_seconds | episodes | episode | episode state | seconds |
| captured_event_count | M | event scopes | unique mature captured events | NA | event | event source; mature horizon | count |
| event_recall | M | event scopes | captured mature events | eligible mature events | event | event source; mature horizon | fraction |
| notification_precision | M | event scopes | mature notifications matching >=1 event | mature notifications | notification | event source; mature horizon | fraction |
| false_alert_share | M | event scopes | mature notifications matching no event | mature notifications | notification | event source; mature horizon | fraction |
| false_positive_rate | M | event scopes | mature no-event observations with primary notification | mature no-event observations | observation | event source; mature horizon | fraction |
| lead_time_mean | M | event scopes | sum earliest-capture lead seconds | captured events | event | event source; mature horizon | seconds |
| lead_time_median | M | event scopes | median earliest-capture lead seconds | captured events | event | event source; mature horizon | seconds |
| warning_to_event_rate | M | event scopes | mature warned observations with future event | mature warned observations | observation | event source; mature horizon | fraction |
| state_observation_count | L | lifecycle state scopes | effective-state observations | NA | observation | verifiable effective state | count |
| state_observation_rate | L | lifecycle state scopes | effective-state observations in state | evaluable state observations | observation | effective state | fraction |
| entity_state_count | L | lifecycle state scopes | entities ever in effective state | NA | entity | effective state | count |
| entity_state_rate | L | lifecycle state scopes | entities ever in state | evaluable entities | entity | effective state | fraction |
| transition_count | L | lifecycle transition scopes | valid consecutive stay/change rows | NA | transition | valid consecutive transition | count |
| transition_rate | L | lifecycle transition scopes | transitions for requested from/to | all valid consecutive stay/change rows | transition | valid consecutive transition | fraction |
| roll_forward_count | L | lifecycle transition scopes | valid direction=roll_forward | NA | transition | valid consecutive transition | count |
| roll_forward_rate | L | lifecycle transition scopes | roll_forward rows | valid consecutive stay/change rows | transition | valid consecutive transition | fraction |
| roll_back_count | L | lifecycle transition scopes | valid direction=roll_back | NA | transition | valid consecutive transition | count |
| roll_back_rate | L | lifecycle transition scopes | roll_back rows | valid consecutive stay/change rows | transition | valid consecutive transition | fraction |
| cure_count | L | lifecycle transition scopes | valid is_cure=true rows | NA | transition | caller cure mapping | count |
| cure_rate | L | lifecycle transition scopes | cure rows | valid transitions from adverse states | transition | caller cure mapping | fraction |
| entry_count | L | lifecycle state scopes | entry rows | NA | transition | effective state | count |
| reentry_count | L | lifecycle state scopes | reentry rows | NA | transition | cadence declaration | count |
| time_in_state_mean | L | lifecycle state scopes | sum elapsed seconds until next observation/as-of | state observations | observation | effective state | seconds |
| exposure_by_state | L | lifecycle state scopes | sum nonnegative exposure | NA | observation | exposure source | exposure_unit |
| exposure_sum | M | warning/event scopes | exposure on scoped observations | NA | observation | exposure source | exposure_unit |
| expected_loss_sum | M | scenario warning scopes | sum(probability*exposure*loss_fraction) on scoped warned rows | NA | observation | aligned Task15 probability+exposure+loss fraction | exposure_unit |
| expected_loss_rate | M | scenario warning scopes | expected_loss_sum | scoped exposure_sum | observation | same exact support as expected_loss_sum | fraction |
| observed_loss_sum | M,L | warning/lifecycle scopes | sum mature per-observation observed loss | NA | observation | observed loss maturity | exposure_unit |
| observed_loss_rate | M,L | warning/lifecycle scopes | observed_loss_sum | same-support exposure_sum | observation | mature loss+exposure | fraction |

Stay rows进入`transition_count`和transition-rate denominator；entry、reentry、invalid和unknown不进入。
Expected/observed metric任一required component在任一scoped row缺失时整项not-verifiable，不做partial
sum。Observed loss是caller-declared actual per-observation non-overlapping horizon amount；允许按该
observation的scenario/segment/time/cohort/vintage/effective-state作historical association resegmentation，
不得按future event payload或后续state重分段，也不得称为causal effect。No-alert scenario的event
precision/false-alert/lead-time仍undefined而非0。

## 12. Table-specific scopes、comparison 与 alignment

Scope key inventory和table组合封闭为：

| scope_key | `monitoring_summary` | `lifecycle_summary` | 准确维度 |
|---|---:|---:|---|
| `overall` | yes | yes | all eligible rows |
| `scenario` | yes | no | scenario |
| `scenario_rule` | yes | no | scenario × rule |
| `scenario_alert_level` | yes | no | scenario × alert level |
| `scenario_segment` | yes | no | scenario × one segment column/category |
| `scenario_time` | yes | no | scenario × time bucket |
| `scenario_cohort` | yes | no | scenario × cohort ordinal |
| `scenario_vintage` | yes | no | scenario × period_index |
| `scenario_state` | yes | no | scenario × effective state |
| `scenario_transition` | yes | no | scenario × transition kind/direction |
| `state` | no | yes | effective state |
| `transition` | no | yes | from/to effective state |
| `segment_time` | yes | yes | one segment dimension × time bucket |
| `cohort_time` | yes | yes | cohort ordinal × time bucket |
| `vintage_state` | yes | yes | period_index × effective state |

Matrix以外组合全部禁止；尤其禁止segment×segment、segment×cohort×vintage、任意三维以上cube及
scenario×全部维度Cartesian展开。第11节aliases准确为：`warning scopes`=所有批准monitoring
scopes；`event scopes`=overall/scenario/scenario_rule/scenario_segment/scenario_time/
scenario_cohort/scenario_vintage；`lifecycle state scopes`=overall/state/segment_time/cohort_time/
vintage_state；`lifecycle transition scopes`=overall/transition/segment_time/cohort_time；`scenario
warning scopes`=所有以scenario开头的批准scope；`warning/lifecycle scopes`按metric所在table取各自
批准scope。

Overall先；scenario按config；rule按scenario/priority/key；alert level按rank降序/key；segment按
config column order和category首次物理row position；time按`time_frequency`calendar bucket；state/
transition按config/rank/key。只生成实际存在且budget内的scope，不top-N、不sampling、不合并other。

`cohort_column`是caller-provided categorical bucket，只返回0-based `cohort_position`；missing最后，
其他category按首次物理row position编号。`cohort_time_column`是caller-declared entity origin time，
必须对同entity恒定；它只按第9节生成exact integer `period_index`作为vintage age，不自动按月、季度、
年份或任意calendar label再分桶。二者不是同义词。Raw cohort/vintage/segment values不返回；arbitrary
object在ordinal assignment前按exact safe scalar whitelist拒绝，pandas index不参与。

`scenario_comparison`每个comparator相对reference、每个批准monitoring metric/scope一row，必须使用
相同mature observation/entity/event support。`delta=comparator_value-reference_value`，不输出relative
delta。Support不同不抛schema exception，而输出`not_verifiable/support_not_comparable`且value/delta
missing；不通过intersection静默修复。Comparison仅为historical offline association；不称A/B、
winner、best、recommended、champion或deployed。

Task 15 source alignment完整继承其frozen result：调用Task 15 validator后，验证
prediction positions、fold validation/evaluable positions、fold id、excluded union、input counts，
以及mode-dependent maturity；non-time modes允许maturity counts为0且evaluable>0。任一失败使用
`lifecycle source alignment: <key>`，至少冻结：

```text
task15_schema_mismatch, row_scope_mismatch, prediction_scope_mismatch,
fold_membership_mismatch, evaluable_scope_mismatch, maturity_count_mismatch,
non_time_maturity_count_nonzero, time_mode_maturity_mismatch
```

Task 16 result只是optional diagnostic evidence。必须先验证owner result frozen schema、input row count、
declared column scope和config fingerprint；只记录fingerprint、finding/warning counts与sanitized status，
不读取raw values、不重算profiles。失败keys：`task16_schema_mismatch`,
`task16_row_scope_mismatch`, `task16_column_scope_mismatch`。Task 16没有raw-data fingerprint，因此即使
shape相同也不能证明same snapshot；provenance固定记录
`task16_snapshot_identity=unverified`。它不得授权feature availability、删除row/column、修改/修复
value或改变warning/state assignment。缺result不妨碍执行，因为hard dependency是kernel/code
ownership而非每次调用必须传audit evidence；provenance标记not_provided。

## 13. Frozen result table schemas

所有text列为pandas`string`；non-null counts/positions为`int64`；nullable integers/floats/bools为
`Int64/Float64/boolean`；timestamps为与input相同timezone的`datetime64[ns]`或
`datetime64[ns, tz]`。空表保留exact columns/order/dtypes。只允许以下schemas：

```text
observation_history:
row_position:int64, entity_position:int64, observation_time:datetime,
observation_status:string, observation_reason:string,
is_consecutive:boolean, period_index:Int64, cohort_position:Int64,
primary_scenario_key:string, primary_rule_key:string, primary_alert_level:string,
primary_alert_rank:Int64, active_rule_count:int64, emitted_notification_count:int64,
maturity_status:string, event_within_horizon:boolean,
effective_state_key:string, effective_state_rank:Int64, state_status:string, state_reason:string

rule_evaluations:
row_position:int64, entity_position:int64, observation_time:datetime,
scenario_key:string, scenario_order:int64, rule_key:string, rule_order:int64,
alert_level:string, alert_rank:int64, path_status:string, truth:string,
true_streak:int64, false_streak:int64, episode_status:string,
notification_status:string, status:string, reason:string, finding_key:string

notifications:
entity_position:int64, scenario_key:string, rule_key:string,
episode_ordinal:int64, notification_ordinal:int64, row_position:int64,
notification_time:datetime, alert_level:string, alert_rank:int64,
notification_kind:string, is_repeated:boolean, first_matched_event_ordinal:Int64,
matched_event_count:int64, maturity_status:string,
status:string, reason:string, finding_key:string

alert_episodes:
entity_position:int64, scenario_key:string, rule_key:string,
episode_ordinal:int64, alert_level:string, alert_rank:int64,
episode_start_time:datetime, episode_end_time:datetime, duration_seconds:Float64,
raw_hit_count:int64, notification_count:int64, suppressed_notification_count:int64,
is_reopen:boolean, is_unresolved:boolean, status:string, reason:string, finding_key:string

event_matches:
scenario_key:string, entity_position:int64, event_ordinal:int64, event_row_position:Int64,
event_time:datetime, duplicate_source_row_count:int64,
event_status:string, match_status:string, captured:boolean,
capturing_rule_key:string, capturing_episode_ordinal:Int64,
capturing_notification_ordinal:Int64, capturing_notification_row_position:Int64,
notification_time:datetime,
lead_time_seconds:Float64, candidate_notification_count:int64,
status:string, reason:string, finding_key:string

state_history:
row_position:int64, entity_position:int64, observation_time:datetime,
candidate_state_key:string, candidate_state_rank:Int64, candidate_state_priority:Int64,
effective_state_key:string, effective_state_rank:Int64,
matching_state_count:int64, status:string, reason:string, finding_key:string

state_transitions:
entity_position:int64, from_row_position:Int64, to_row_position:int64,
transition_time:datetime, from_state_key:string,
candidate_to_state_key:string, effective_to_state_key:string,
from_rank:Int64, candidate_to_rank:Int64, effective_to_rank:Int64,
transition_kind:string, transition_direction:string, is_allowed:boolean,
is_consecutive:boolean, is_cure:boolean, exposure:Float64, observed_loss:Float64,
status:string, reason:string, finding_key:string

monitoring_summary:
scope_key:string, scope_position:Int64, scenario_key:string, rule_key:string,
metric:string, metric_value:Float64, numerator:Float64, denominator:Float64,
support_n:int64, support_unit:string, mature_n:int64, censored_n:int64, unit:string,
status:string, reason:string, finding_key:string

scenario_comparison:
reference_scenario_key:string, comparator_scenario_key:string, metric:string,
reference_value:Float64, comparator_value:Float64, delta:Float64,
numerator:Float64, denominator:Float64, support_n:int64, support_unit:string,
status:string, reason:string, finding_key:string

lifecycle_summary:
scope_key:string, scope_position:Int64, from_state_key:string, to_state_key:string,
metric:string, metric_value:Float64, numerator:Float64, denominator:Float64,
support_n:int64, support_unit:string, unit:string, status:string, reason:string, finding_key:string

provenance:
provenance_key:string, provenance_value:string, status:string,
reason:string, finding_key:string
```

`primary_scenario_key`是reference scenario；不是跨scenario自动选winner。所有pandas`string`列均
nullable。`primary_rule_key/primary_alert_level`在无primary alert时、capture fields在uncaptured event
时、episode_end_time在open episode时、from fields在entry时、candidate fields在unknown state时均
missing。Nullable timestamps使用`pd.NaT`，nullable integers使用`Int64`；metric value/numerator/
denominator是`Float64`且unavailable时为`pd.NA`。Result不得保存raw DataFrame/view、config/tree/rule/state/
scenario object、Task15/16 result、estimator、Figure、raw entity/segment/cohort/peer/event values、
condition literals、file path、run timestamp/duration或random id。

## 14. Closed status/reason/key、sorting 与 determinism

Status唯一inventory：

```text
available, unavailable, undefined, not_applicable, not_verifiable, inactive, censored
```

Reason唯一inventory：

```text
computed, no_rows, no_rules, no_events, no_matching_state, default_state_applied,
unknown_condition, missing_operand, type_mismatch, nonfinite_operand, timezone_mismatch,
missing_available_time, value_not_yet_available, scenario_inactive, rule_inactive,
state_inactive, outside_effective_window, insufficient_history, insufficient_peer_support,
peer_reference_not_prior,
score_unavailable, probability_unavailable, event_source_not_declared,
prediction_horizon_not_mature, event_not_observed, event_captured,
zero_denominator, insufficient_support, cooldown_suppressed, persistence_pending,
resolution_pending, episode_opened, episode_active, episode_resolved, episode_reopened,
non_consecutive_observation, entry_observation, reentry_observation,
transition_allowed, transition_not_allowed, terminal_state_exit,
cohort_not_declared, vintage_not_evaluable, exposure_unavailable,
observed_loss_unavailable, observed_loss_not_mature, source_not_requested,
task15_evidence_not_provided, task16_evidence_not_provided,
duplicate_event, support_not_comparable
```

Closed vocabularies精确为：

```text
truth_status: true, false, unknown, not_evaluated
notification_kind: episode_open, episode_reopen, repeated
notification_status: emitted, suppressed, not_emitted
episode_status: clear, pending, active, resolved, not_evaluated
event_status: mature, censored, not_eligible
match_status: captured, not_captured, not_applicable, not_verifiable
transition_kind: entry, reentry, stay, change, invalid
transition_direction: not_applicable, flat, roll_forward, roll_back
maturity_status: mature, immature, not_applicable, not_verifiable
metric_status: available, unavailable, undefined, not_applicable,
               not_verifiable, inactive, censored
path_status: evaluated, not_evaluated
```

Status/reason legal pairing冻结为：

| status | only legal reason families |
|---|---|
| available | `computed`, event/episode/transition lifecycle facts |
| unavailable | declared source absent on position: score/probability/exposure/observed-loss unavailable |
| undefined | `zero_denominator`, `insufficient_support`, or empty/no-rule/no-event denominator |
| not_applicable | `source_not_requested`, event/cohort source not declared, evidence not provided |
| not_verifiable | unknown/type/nonfinite/history/peer/maturity/transition/support-comparability reasons |
| inactive | scenario/rule/state inactive or outside effective window |
| censored | `prediction_horizon_not_mature` only |

`cooldown_suppressed`只配rule-evaluation available evidence且notification_status=suppressed；
episode open/active/resolved/reopened与transition allowed facts只配available。Exception keys绝不降级为
structured reason。不得在实现阶段新增近义值或其他pairing。

`finding_key`只按`monitoring:<key>`、`scenario:<ordinal>`、`rule:<scenario ordinal>:<rule ordinal>`、
`entity:<position>:episode:<ordinal>`、`state:<ordinal>`、`transition:<from ordinal>:<to ordinal>`
构造，不含raw value。

稳定排序：observations按row_position；rule evidence按row_position/scenario config order/priority/key；
notifications按time/entity/scenario/rule/ordinal；episodes按entity/scenario/rule/ordinal；events按
entity/event_time/event ordinal；states按row_position；transitions按entity/transition_time/to row；summary按
scope inventory、scope ordinal、scenario、rule、metric inventory；provenance按固定key inventory。
不得依赖set/hash iteration、pandas index、locale、environment timezone、current date、random或
parallel completion。

Float只用float64与本合同公式。相同input/config/evidence repeated run的tables/dtypes/order、
warnings、limitations和fingerprint逐项一致。所有成功/失败路径只读，input cell/dtype/index、
dataclasses与tuples不变。

## 15. Privacy、provenance 与 fingerprints

Validation在构造error前sanitize。Public error只含stable key，不能插raw column之外的scalar值；
raw entity/event/cohort/segment/peer/condition literal不得进入error/repr/warning/result。

获准timestamp evidence仅限：`observation_time`（observation/state history）、
`simulated_notification_time`即schema中的`notification_time`（notifications/event matches）、
`episode_start_time/episode_end_time`（episodes）、`future_event_time`即schema中的`event_time`
（event matches）、`state_transition_time`即schema中的`transition_time`（transitions）和provenance单值
`analysis_as_of`。Aware timestamps以UTC、原nanosecond precision输出；naive保持naive。不得返回其他
datetime feature或timestamp集合，且这些表不伴随raw entity identifier。禁止raw event values是指
event label/payload，不包括上述获准future-event timestamp。Provenance只保存timezone、UTC/naive、
closure和precision policy，不保存observation/event timestamp集合。

`monitoring_fingerprint`为lowercase 64-char SHA-256 hex。Canonical JSON参数固定：
`sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(",", ":")`。Payload含schema
version、monitor/version、time/window/horizon declarations、scenario/rule/state keys与ordinals、
flags/ranks/condition shape、literal exact canonical value、scope declarations、business source
declarations及最终applicable warnings/limitations keys；raw literals只在transient payload中，result只存digest。Datetime ISO-8601，tuple
保序，批准NumPy scalar转built-in；dict/set/callable拒绝。

Provenance keys固定：`contract_version, monitoring_fingerprint, row_identity,
entity_identity, time_model, analysis_as_of, history_windows, horizon_policy, scenario_inventory,
rule_inventory, alert_level_inventory, state_inventory, transition_policy, event_source,
score_source, probability_source, task15_evidence_status, task15_evidence_fingerprint,
task16_evidence_status, task16_config_fingerprint, task16_snapshot_identity,
exposure_source, observed_loss_source,
scope_inventory, resource_usage`。Value只含sanitized versions/counts/type families/column names/
digests，不含raw values。Task15 evidence digest由aligned frozen schema/positions/numeric prediction
canonical bytes计算，不声称修改Task15 public result。

Warnings按`input,time,source,scenario,episode,event,lifecycle,resource`固定顺序。Limitations仅：

```text
offline_monitoring_not_executed, historical_comparison_not_causal,
caller_defined_states_and_alert_levels, external_score_semantics_caller_declared,
right_censored_event_horizons, peer_baseline_is_descriptive,
entity_linkage_depends_on_prepared_input
```

只输出适用项且去重。

## 16. Resource coordination、gate precedence 与 stable errors

Closed resource keys与整数上限精确为：

| stable key | maximum |
|---|---:|
| `condition_depth` | 8 |
| `condition_nodes` | 128 per tree |
| `membership_literals` | 100 per atomic node |
| `literal_string_length` | 1,024 chars |
| `input_rows` | 100,000 |
| `entities` | 50,000 |
| `observations_per_entity` | 10,000 |
| `scenarios` | 10 |
| `rules_per_scenario` | 50 |
| `all_rules` | 100 |
| `states` | 50 |
| `alert_level_mappings` | 50 |
| `allowed_transitions` | 2,500 |
| `segment_columns` | 4 |
| `peer_group_columns` | 4 |
| `categories_per_scope_column` | 100 |
| `time_buckets` | 240 |
| `cohort_buckets` | 240 |
| `vintage_buckets` | 240 |
| `derived_scopes` | 2,000 |
| `rule_evaluations` | 1,000,000 |
| `state_evaluations` | 5,000,000 |
| `notifications` | 1,000,000 |
| `episodes` | 500,000 |
| `event_match_rows` | 500,000 |
| `event_match_scan_operations` | 11,000,000 |
| `state_history_rows` | 100,000 |
| `state_transition_rows` | 100,000 |
| `monitoring_summary_rows` | 200,000 |
| `lifecycle_summary_rows` | 200,000 |
| `scenario_comparison_rows` | 200,000 |
| `key_version_description_column_length` | 64 ASCII chars |

Preflight projection公式唯一为：

```text
rule_evaluations = eligible_rows * sum(enabled rules over scenarios)
state_evaluations = eligible_rows * enabled global lifecycle states
notifications_upper = rule_evaluations
episodes_upper = sum over entity/scenario/rule ceil(eligible observations / 2)
state_history_rows = eligible_rows
state_transition_rows = eligible_rows
event_match_rows = unique declared events * scenarios
event_match_scan_operations = scenarios * (notifications_upper + unique declared events)
monitoring_summary_rows = sum(actual approved monitoring scope instances * eligible M metrics)
lifecycle_summary_rows = sum(actual approved lifecycle scope instances * eligible L metrics)
scenario_comparison_rows = comparator_count * comparable monitoring metrics *
                           approved comparison scope instances
```

`eligible_rows`在resource projection仅指通过top-level row/time cardinality检查的rows，不依赖condition
truth。Scope instances从批准第12节matrix和safe anonymous category cardinalities计算。通过一个gate
不保证整个run成功。每个gate的requested==maximum通过该gate；maximum+1在任何对应大型
materialization前首先失败。无silent truncation、sampling、top-N或partial result。

Event matching固定为per `entity × scenario` sorted two-pointer scan。Notification和event先按第14节
tie-break排序；sorting time为`O((N_notifications+N_events) log(N_notifications+N_events))`，每scenario
scan为`O(N_notifications+N_events)`，working space为`O(N_notifications+N_events)`加有界output。
禁止notification×event Cartesian materialization。一个event最多一个capture owner（earliest
eligible notification，same timestamp按scenario/rule/notification ordinal）；一个notification可匹配
多个future events，notifications表只记录first matched event和count，完整关系在event_matches。
任一output/scan上界超限在processing前整体失败。No new dependency/parallel backend。

只使用`ValueError`，public prefixes：

```text
lifecycle config is invalid: <key>
lifecycle input schema is invalid: <key>
lifecycle condition: <key>
lifecycle source alignment: <key>
lifecycle resource limit exceeded: <key>
```

至少冻结config keys：`wrong_type, invalid_key, duplicate_key, invalid_time_window,
invalid_horizon_declaration, invalid_scenario_shape, reference_scenario_missing,
invalid_alert_level_mapping, invalid_rule, invalid_state, invalid_transition,
invalid_score_declaration, invalid_probability_declaration, invalid_loss_declaration,
duplicate_ranking_source, duplicate_probability_source, duplicate_outcome_source,
duplicate_exposure_source, duplicate_observed_loss_source`；input keys另包括：
`invalid_dataframe, invalid_column_labels, duplicate_columns, missing_column, unsupported_dtype,
duplicate_entity_observation_time, missing_entity, invalid_time_dtype, timezone_mismatch,
nonfinite_value, negative_value, observation_time_missing, available_time_missing,
observation_after_as_of, available_after_as_of, available_after_observation,
datetime_type_invalid, datetime_awareness_mismatch, datetime_timezone_invalid`；resource keys精确为
上表snake_case inventory。

Validation/gate precedence冻结为：config cardinality/key lengths -> condition-tree budgets及whole-tree
forbidden-role validation -> input rows/entities/observations-per-entity -> scenarios/rules/states/allowed
transitions -> derived scopes -> rule/state evaluation projections -> notification/episode/event-match
projections -> summary-row projections -> input schema及temporal validation -> Task15/16 source alignment ->
condition evaluation -> episode/state/event processing -> table materialization。Exact scalar/type safety在
任何category/time/projection protocol dispatch前执行。任一失败不消费future value、不返回partial
result；通过前一个gate不保证后续gate或run成功。

## 17. Future implementation test strategy

实现后的直接证据必须使错误实现失败：

1. **API/schema**：exact 7 exports/signatures/field order/frozen dataclasses；11张表的空/非空
   columns/order/dtypes；无dict DSL/public kernel alias。
2. **Kernel parity**：全部atomic/Boolean/missing/date/timezone与Task16逐rowtruth/reason一致；spy证明
   无第二comparison evaluator。
3. **Point-in-time**：future-only row/value/event不改变current；availability exact boundary、window
   open/closed、prior-only history、peer reference cutoff、personal history手算。
4. **Signals**：level/change/trend/history mean/peer deviation、insufficient support、lower-risk direction、
   score/probability隔离手算；ranking margin不能产生probability metrics。
5. **Episodes**：single/continuous deterioration、persistence、false/unknown、cooldown只抑制notification、
   resolution/reopen、missing gap不关闭、first/repeated、multiple rules/levels与precedence。
6. **Time/events**：horizon start/end exact boundaries、mature/immature/right-censor、multiple alerts/events、
   duplicate events、no-alert undefined metrics、lead-time earliest attribution。
7. **Metrics**：全部finite warning metrics逐项与hand-worked fixture一致，特别证明false-alert-share与
   false-positive-rate不同；zero denominator/status/denominator fields正确。
8. **Lifecycle**：caller states、overlap/default/unknown、entry/reentry、allowed/impossible/terminal exit、
   roll-forward/back/cure、non-consecutive、MOB/vintage/cohort boundaries与pandas baseline一致。
9. **Scenarios**：no-alert/single threshold/rules/model/model+rules same support；challenger变化不改变
   reference；不选winner、不称A/B。
10. **Sources**：Task15 six modes完整position/fold/evaluable/maturity alignment，包括non-time mature=0
    且evaluable>0；Task16 schema/fingerprint only consumption；spy证明不重算Task15/16。
11. **Privacy/determinism**：secret entity/index/literal/event/segment/cohort/peer values递归扫描result/
    errors/repr/warnings均不存在；duplicate index安全；repeat result/fingerprint一致；input不变。
12. **Resources**：每项max和max+1；scope/product limits在evaluation前失败；合法max无truncate。
13. **Compatibility/distribution**：full v0.1及Tasks15--17 tests不削弱；old exports relative order不变；
    wheel/sdist含module与合同，clean install exact symbols；workflow/report/CLI不变。

以下direct fixtures是上述matrix的强制细化，不得skip/xfail、只断言不抛异常、用production helper
生成expected、monkeypatch limit或公开private budget helper：

- condition role fixtures分别覆盖：ordinary left 与RHS future-event time、observed loss、
  outcome-maturity column；future-event indicator left 与ordinary RHS；AND/OR/NOT deep RHS、inactive
  rule/state与short-circuit branch的RHS；以及`is_not_missing`携带forbidden RHS column。每项均在任何
  row evaluation前失败`lifecycle condition: forbidden_condition_role`，不得静默忽略unary RHS；
  RHS tainted derived signal若未来public representation被提议亦必须同样失败。两个ordinary
  point-in-time eligible columns在Task 16批准的operator/type compatibility下可进入kernel；
  nested后置node及tainted derived lineage同样必须在任何row evaluation前失败；
- January observation + March peer reference end、reference end等于observation均
  `not_verifiable/peer_reference_not_prior`；严格早于可算；candidate available晚于reference end排除；
  修改future peer rows不改变historical result；
- exact datetime families、malicious datetime-like、future observation、available after observation/as-of、
  awareness mismatch、UTC output、duplicate entity/time和unsorted duplicate-index input；
- duplicate ranking/probability source、DataFrame probability不存在、Task15 excluded/non-evaluable
  positions、ranking不升级probability、Task16 snapshot identity固定unverified；
- false-tail不hit/notify/primary、open episode duration到analysis_as_of、resolved duration到threshold row、
  nullable entry、invalid candidate不替换effective state、cure与roll direction独立；
- uncaptured event nullable fields、notification多eventcount、timestamp allowlist、raw label/payload/category
  不泄露、每个closed metric逐项hand-worked numerator/denominator/support/status；
- 上表每个resource gate分别requested=max通过该gate、max+1首先以stable prefix/key失败；验证所有
  projection在materialization前、无truncate/partial result；event matcher fixture使用足以使Cartesian
  实现越界的数据并证明sorted two-pointer semantics。

Implementation完成后必须先`bash scripts/verify-uv-env.sh`，再以`.venv/bin/python`运行full pytest、
Ruff check、Ruff format check、build、wheel/sdist source-free clean install、distribution smoke与diff
gates。Contract drafting轮不运行这些命令。

## 18. Future implementation allowlist 与禁止修改

Task 18合同通过唯一一次full contract review、targeted fixes和bounded closure后，implementation
准确allowlist仅：

```text
src/sharper/lifecycle_monitoring.py
src/sharper/__init__.py
tests/test_lifecycle_monitoring.py
tests/test_public_api.py
tests/test_distribution.py
docs/decisions/task18-post-loan-early-warning-lifecycle-contract.md
SPEC.md
IMPLEMENTATION_PLAN.md
README.md
docs/api.md
```

只读且默认禁止修改：

```text
src/sharper/_condition_kernel.py
src/sharper/risk_validation.py
src/sharper/data_audit.py
src/sharper/decision_strategy.py
```

禁止修改Task15--17 contracts/modules/behavior、v0.1 modules、workflow/reporting/CLI/visualization、
Task19/20、dependencies、lock、package version、build backend或CI。若实现证明private kernel不足，
必须停止并走post-approval contract amendment，不得越allowlist修补owner。

## 19. Compatibility、lifecycle 与完成定义

新API完全opt-in。v0.1 signatures/dataclass fields/defaults/errors/reports/CLI/exports和Tasks15--17
frozen contracts/behavior不变。无dependency diff；package version保持`0.1.0`；v0.2未完成/未发布。

合同生命周期当前冻结为：

```text
Task 18 contract: Approved — Go
Task 18 contract drafting: complete
Full contract review: No-Go
Original findings: P0=2, P1=6, P2=0
Targeted contract fixes: complete
Bounded contract closure attempt 1: No-Go
Residual finding: T18-CR-01 / P0 (RHS column role-isolation gap)
Residual targeted contract fix: complete
Final bounded contract closure: Go
Closure P0: 0
Closure P1: 0
Closure P2: 0
Task 18 implementation: Not started
```

Task 18唯一一次开放式full contract review已经完成且不得再次执行；final bounded contract closure为`Go`，
合同为`Approved — Go`。Task 18 implementation仍未开始；不stage、commit、push、tag或release，不开始
Task19/20。只有按第18节allowlist进入implementation并完成要求的验证后，才可进入唯一一次full
implementation review。

Task 18 implementation只有在：合同Approved — Go；exact APIs/schemas/time/no-lookahead/kernel parity/
episode/event/lifecycle/metrics/source alignment/privacy/resources/compatibility tests全部通过；full
pytest/Ruff/build/clean-install/distribution gates通过；无Task19/20越界后，才可进入唯一一次full
implementation review。当前无未决API模型；本合同选择multi-label rule diagnostics + deterministic
primary alert、per-rule episodes、caller-ranked lifecycle states、single DataFrame和single public
execution entry。
