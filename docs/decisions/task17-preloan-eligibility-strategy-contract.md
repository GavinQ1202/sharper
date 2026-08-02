# Task 17 — Pre-loan Eligibility Rules and Decision Strategy Simulation 精确合同

**合同状态：** Approved — Go
**Implementation 状态：** Implementation complete — review Go

**Review记录：** Full contract review：`No-Go`；targeted contract fixes：`complete`；bounded
contract closure：`Go`；P0：`0`；P1：`0`；P2：`0`。Contract：`Approved — Go`；
Implementation：`Implementation complete — review Go`。不得再次进行开放式Task 17
full contract review。已通过review的Task 15/16结论保持关闭。

**Post-approval compatibility amendment：** Task 15 mode-dependent maturity alignment已定向修正；
amendment bounded closure：`Go`；`T17-A1`：`closed`；P0：`0`；P1：`0`；P2：`0`。该
post-approval compatibility amendment已批准；amendment closure当时Implementation仍为
`Not started`，partial implementation files不改变该历史状态。Amendment已先形成独立合同
commit，随后恢复Task 17 implementation。不得再次进行开放式Task 17 full contract review。

**Implementation记录：** Full implementation review：`No-Go`；targeted implementation
fixes：`complete`；bounded implementation closure attempt 1：`No-Go`；residual targeted
implementation fixes：`complete`；final bounded implementation closure：`Go`；P0：`0`；P1：
`0`；P2：`0`。Implementation：`Implementation complete — review Go`。Task 17唯一一次开放式
full implementation review已经完成，后续不得再次执行。

**合同 drafting/targeted-fix/bounded-closure历史 scope：** 当时只新建本合同，并在closure时同步
`SPEC.md`和`IMPLEMENTATION_PLAN.md`状态。本次post-approval targeted amendment scope仅为本合同
内的`T17-A1/P1`；明确排除代码、tests、roadmap、Task 15/16 合同、README、API 文档、
workflow/reporting/CLI、依赖、lock、package version、commit、push、tag 和发布。阻塞条件是
本合同与已批准 roadmap、Task 15/16 frozen schema/ownership 或 v0.1 compatibility 发生无法
在本文件内消解的冲突。本次只允许定向修正mode-dependent maturity alignment；不得执行bounded
amendment closure、重开已关闭结论或扩围。

## 1. 权威依据、目标与边界

本合同服从根目录 `AGENTS.md`、`SPEC.md`、`IMPLEMENTATION_PLAN.md`、
`docs/decisions/v02-roadmap-contract.md`、Task 15 与 Task 16 已批准合同。提示中的建议与
roadmap 不一致时以 roadmap 为准；尤其是 action names 必须 caller-defined，不能冻结成仅
`approve/decline/review` 三个名字。

Task 17 唯一负责：caller-declared 贷前 eligibility/action rules；priority、stop-on-hit、
multi-hit、conflict、fallback、effective time 与 version；caller-frozen score/probability
cutoffs；离线 simulated action；rule hit/coverage/overlap/conflict；action-level business
evidence；caller-declared constraints；显式映射后的 historical-action paired comparison；
strategy/rule/decision provenance。它只做 `declare -> validate -> evaluate -> assign ->
simulate -> measure -> report`。

Task 17 输出只是假设敏感的离线模拟证据，不是贷款审批、合规结论、已实现收益或生产命令。
它不得 optimize、learn、fit、calibrate、repair、deploy 或执行真实决定，也不得：

- 重建 Task 15 folds/OOF、训练、calibration、AUC、AP、Gini、KS、Brier、log loss、ECE、
  gains/lift 或 analytical operating point；
- 重算 Task 16 quality、leakage、input profile 或 missingness drift；
- 实现 Task 18 alert/episode/post-loan lifecycle、Task 19 governance/winner selection、Task
  20 workflow/report/CLI/JSON parser/persistence；
- 自动阈值/band/rule/order/action/policy 生成、自动 strategy optimization、solver、policy
  repair、dynamic pricing、credit-limit、portfolio、collections 或 profit optimization；
- reject inference、causal/uplift、SHAP、WOE、target encoding、supervised binning、feature
  selection 或 model training；
- 研究或实现反欺诈、身份核验、设备/IP/velocity/network rules；
- 建立 public 通用 DSL、`eval`、expression、callable、lambda、plugin、script 或任意代码
  执行；
- 读写文件、网络或外部系统，建立 production service、real-time decision engine，返回
  estimator、Figure、修复后 DataFrame 或 production decision object。

## 2. Ownership 与依赖 DAG

未来实现唯一领域模块为 `src/sharper/decision_strategy.py`。它唯一拥有本合同 public
dataclasses/function、public config/condition validation、public condition 到 Task 16 private
tree 的编译、rule orchestration、action assignment、constraints、result tables 与 sanitized
provenance。

`src/sharper/_condition_kernel.py` 继续唯一拥有 equality、ordering、membership、between、
missing、type compatibility、nonfinite、timezone、effective-window 与 three-valued Boolean
truth。Task 17 必须调用 `_evaluate_condition`；不得复制 comparison 或建立第二 evaluator，
不得公开 `_Condition*` symbols。依赖冻结为：

```text
_condition_kernel -> decision_strategy
risk_validation frozen result type -> decision_strategy (optional evidence path)
data_audit frozen result type -> decision_strategy (optional evidence path)
decision_strategy result -> future Task 19 / Task 20
```

不得反向 import、循环依赖、扩张 `analysis.py`/`evaluation.py`/Task 15/16 modules，亦不得
import workflow/reporting/CLI/visualization。Task 15/16 inputs 只读消费。

## 3. 唯一执行模型

每次调用精确接受一个 raw/prepared `pd.DataFrame`、一个 strategy config、可选一个
`BinaryRiskValidationResult` 和可选一个 `DataAuditResult`。每次只模拟一个 strategy key/
version，不选择 latest，不接受或自动比较多个 strategy configs。

- DataFrame 始终必需；conditions、exposure 和 historical action 均从该 frame 按 row
  position 读取。reference/current 双表不属于 Task 17。
- Task 15 result 可选。纯规则/action-only replay 不需要它；使用其 ranking score、event
  probability、mature outcome 或相应 business evidence时才需要。缺少 rule 所必需的 score
  source 是 invalid config/schema；仅 metric/constraint 缺 evidence 则返回 structured
  `unavailable`/`not_verifiable`，绝不重算 Task 15。
- Task 16 result 可选，只能提供 audit provenance/limitations。未提供不阻止模拟；提供后
  必须通过 result schema、`n_rows`、`n_columns` 验证，否则 alignment exception。Task 17
  不从 audit findings 猜测或改写规则。
- Ranking score只能来自DataFrame column或Task 15 result二者之一；重复立即cardinality error。
  Event probability只有Task 15来源。Ranking与probability是不同kinds，可同时有。
- 所有 identity 与 join 使用零基 `row_position`；不使用 DataFrame index alignment。
  Duplicate index 安全且原样保留在 input 中。
- Task 15 evidence必须先通过其frozen result validation，再通过下述Task 17增强cross-table
  source-alignment validation；两层均在condition evaluation、decision assignment和business
  calculation之前完成。Task 17不得把第一层validator当作cross-table证明。
- 只在 predictions scope materialize score/outcome，其他 positions 为 source missing。
  `excluded_rows` 不从 Task 17 删除；不在 prediction scope 的 row 仍可由纯规则形成 action，
  但 score condition 为 `unknown`，监督/概率 evidence 不可用。
- DataFrame、Task 15/16 result、config、condition tree、rules、constraints、tuples 和 mappings
  均不得修改。Result 不持有它们的引用或 view。

历史 comparison 不是第二 strategy execution：caller 显式把历史 raw action values 映射到
本 strategy 的 action names 后，Task 17 只报告相同 row positions 上的 paired transition。
它不重放历史 policy、不推断其版本、不选 winner。该路径满足 roadmap 的 reference
(historical frozen actions)/challenger(simulated actions) common-support evidence。

### 3.1 Task 15 enhanced cross-table source alignment

使用Task 15实际frozen字段定义以下按升序排列、无重复的position tuples：

```text
prediction_positions = tuple(predictions["row_position"])
fold_validation_positions = ordered union of folds["validation_row_positions"]
fold_evaluable_positions = ordered union of
    folds["evaluable_validation_row_positions"]
prediction_evaluable_positions = tuple(
    predictions.loc[predictions["is_evaluable"], "row_position"]
)
excluded_positions = tuple(excluded_rows["row_position"])
declared_source_scope = tuple(range(input_n_rows))
```

每个`folds` row的`validation_row_positions`和`evaluable_validation_row_positions`必须是exact
tuple；后者是前者子集。每个validation position恰好属于一个fold；每个prediction position
恰好出现一次，且其`fold_id`等于包含该position的`folds.fold_id`。必须同时成立：

```text
prediction_positions == fold_validation_positions
prediction_evaluable_positions == fold_evaluable_positions
set(prediction_evaluable_positions) <= set(prediction_positions)
set(prediction_positions).isdisjoint(excluded_positions)
ordered union(prediction_positions, excluded_positions) == declared_source_scope
predicted_n_rows == len(prediction_positions)
evaluable_n_rows == len(prediction_evaluable_positions)
input_n_rows == len(data)
```

Task 15 closed `validation_mode` inventory精确为`stratified_holdout`、`stratified_kfold`、
`group_holdout`、`group_kfold`、`time_holdout`和`time_forward`；Task 17不得根据maturity count
猜mode，也不得创造新的public mode key。所有mode逐fold共同满足：

```text
validation_n == len(validation_row_positions)
evaluable_validation_n == len(evaluable_validation_row_positions)
```

`prediction_evaluable_positions == fold_evaluable_positions`继续保证prediction的`is_evaluable`
与fold evaluable membership逐position一致。上述通用position、fold membership、fold_id、union、
overlap、coverage和top-level count检查不因maturity分类而削弱。

对time-based modes `time_holdout`和`time_forward`，另须精确继承Task 15 frozen maturity语义：

```text
validation_mature_n == evaluable_validation_n
validation_excluded_n == immature_validation_n
validation_excluded_n == validation_n - evaluable_validation_n
```

Task 17只核对Task 15声明的mature/evaluable/immature scopes与counts，不读取日期重新计算label
maturity，不重新分类任何row。

对non-time modes `stratified_holdout`、`stratified_kfold`、`group_holdout`和`group_kfold`，Task 15
冻结的maturity provenance为not applicable，因此逐fold必须满足：

```text
validation_mature_n == 0
validation_excluded_n == 0
immature_validation_n == 0
```

这些0值不改变独立的evaluable membership：合法non-time fold允许
`validation_mature_n == 0`且`evaluable_validation_n > 0`，其predictions、ranking score和合法
event probability仍可消费；不得因maturity count为0把evaluable rows解释为0、把evidence降级为
unavailable或拒绝source。

`eligible_n_rows`必须等于`input_n_rows`减去
`excluded_rows.reason == "missing_target"`的position数；predicted/excluded合并后不得missing、
duplicate或extra。所有position必须是exact non-boolean built-in int、0-based且位于
`[0, input_n_rows)`；不得使用pandas index。

Task 17只验证上述Task 15声明的source scope，不重新定义其`missing_target`、`training_only`、
`before_first_validation_window`、`outside_validation_window`、immature或evaluable语义，也不
修复result。增强alignment precedence固定为：1) 调用Task 15 frozen result validation；2) 读取其
frozen `validation_mode`；3) 验证通用position/fold/evaluable不变量；4) 按上述准确mode验证对应
maturity语义；5) 验证top-level counts与source scope；6) 全部通过后才允许消费ranking score或
event probability。任一schema、count、union、fold membership、fold_id、is_evaluable、overlap、
coverage或maturity关系失败，统一在任何求值前抛
`ValueError("strategy source alignment: <key>")`；不得降级为row unknown、静默丢行、先消费
probability或修改Task 15 input/result。

## 4. Frozen public API

Task 17 精确新增六个、且仅六个 opt-in public symbols：

```python
@dataclass(frozen=True)
class StrategyCondition:
    kind: Literal["atomic", "and", "or", "not"]
    operator: Literal[
        "eq", "ne", "lt", "le", "gt", "ge",
        "in", "not_in", "between", "is_missing", "is_not_missing",
    ] | None = None
    left_kind: Literal["column", "ranking_score", "event_probability"] | None = None
    left: str | None = None
    right_kind: Literal["literal", "column"] | None = None
    right: object | None = None
    children: tuple[StrategyCondition, ...] = ()

@dataclass(frozen=True)
class DecisionRule:
    rule_key: str
    phase: Literal["eligibility", "decision"]
    priority: int
    condition: StrategyCondition
    action_name: str
    stop_on_hit: bool = True
    enabled: bool = True
    effective_from: datetime | None = None
    expires_at: datetime | None = None
    description_key: str | None = None

@dataclass(frozen=True)
class DecisionConstraint:
    constraint_key: str
    metric: Literal[
        "action_count", "action_rate", "selected_rate", "rejected_rate",
        "review_count", "review_rate", "request_information_rate",
        "selected_exposure_sum", "expected_loss_sum", "expected_loss_rate",
        "observed_loss_sum", "observed_loss_rate", "selected_event_rate",
    ]
    operator: Literal["le", "ge"]
    threshold: float
    action_name: str | None = None
    action_role: Literal[
        "selected", "rejected", "review", "request_information", "limited", "other"
    ] | None = None
    minimum_support: int = 1

@dataclass(frozen=True)
class DecisionStrategyConfig:
    strategy_key: str
    strategy_version: str
    effective_from: datetime
    expires_at: datetime | None
    evaluation_time: datetime
    rules: tuple[DecisionRule, ...]
    default_action_name: str
    unknown_action_name: str
    action_role_mapping: tuple[tuple[str, Literal[
        "selected", "rejected", "review", "request_information", "limited", "other"
    ]], ...]
    constraints: tuple[DecisionConstraint, ...] = ()
    ranking_score_column: str | None = None
    ranking_score_direction: Literal["higher_risk", "lower_risk"] | None = None
    historical_action_column: str | None = None
    historical_action_mapping: tuple[tuple[object, str], ...] = ()
    historical_policy_version: str | None = None
    exposure_column: str | None = None
    loss_fraction: float | str | None = None
    action_assumptions: tuple[tuple[str, float, float], ...] = ()
    exposure_unit: str | None = None
    segment_columns: tuple[str, ...] = ()
    time_slice_column: str | None = None

@dataclass(frozen=True)
class DecisionStrategyResult:
    strategy_key: str
    strategy_version: str
    strategy_fingerprint: str
    input_n_rows: int
    decided_n_rows: int
    unavailable_n_rows: int
    requested_rule_count: int
    active_rule_count: int
    requested_constraint_count: int
    row_decisions: pd.DataFrame
    rule_evaluations: pd.DataFrame
    rule_summary: pd.DataFrame
    action_summary: pd.DataFrame
    business_summary: pd.DataFrame
    constraint_summary: pd.DataFrame
    historical_transitions: pd.DataFrame
    provenance: pd.DataFrame
    warnings: tuple[str, ...]
    limitations: tuple[str, ...]

def simulate_decision_strategy(
    data: pd.DataFrame,
    config: DecisionStrategyConfig,
    *,
    risk_validation: BinaryRiskValidationResult | None = None,
    data_audit: DataAuditResult | None = None,
) -> DecisionStrategyResult: ...
```

所有 dataclass 仅 shallow frozen；constructor 不执行 runtime validation，validation 全部在
function 中 whole-config-first 完成。函数前两个参数 positional-or-keyword，evidence 参数
keyword-only；不得增加 `**kwargs`。Public docstrings 必须写参数、结果、errors、副作用、
missing策略和最小示例。

`sharper.__all__` 只在 Task 16 suffix 后按以下顺序追加：

```text
StrategyCondition
DecisionRule
DecisionConstraint
DecisionStrategyConfig
DecisionStrategyResult
simulate_decision_strategy
```

v0.1 prefix、Task 15 suffix、Task 16 suffix的内容和相对顺序不变。

## 5. Public condition representation 与 kernel compilation

`StrategyCondition` 是 closed、typed、pure-data、可序列化 representation，不是通用 DSL，
不得接收 mapping/dict schema、callable、expression 或 private kernel object。

- `atomic`：`children=()`；必须有 operator/left_kind。`left_kind="column"` 时 `left` 是
  非空 column name；score/probability source 时 `left=None`。Missing operators 要求
  `right_kind/right=None`。其他 operators 必须有 right。
- `and/or`：operator/operands 均为 `None`，children tuple 至少两个；`not` 恰好一个。
- `eq/ne/lt/le/gt/ge` 支持 column-to-literal、column-to-column；score/probability 只允许
  位于 left 且 right 为 literal。`in/not_in/between` right 是 exact built-in tuple；
  `between` 为闭区间。Boolean 开闭 score bands 用 `gt/ge/lt/le` 的 AND 显式表达。
- Literal families 精确继承 kernel：`None/pd.NA/pd.NaT` 不能作 comparison literal；其他
  allowed exact families是 bool/int/finite float/str/date/datetime/pd.Timestamp 和 Task 16
  allowlisted NumPy scalars。Membership 同 family或 int/float compatible，不接受 list/set/
  dict/object。
- Public condition 没有 version/effective fields。Version 只属于 strategy；rule window只
  属于 rule，避免 root/child version 冲突。
- 条件先 public validate/normalize，再编译 `_ConditionOperand/_ConditionNode`，然后且仅然后
  调用 `_evaluate_condition`。Whole tree、全部 later rules 在读取任何 row value前完成
  validation；invalid later rule不能被 first match 掩盖。
- column source直接编译。Task 15/DataFrame score source先按第8节构造 position-safe private
  temporary Series，再编译成 reserved private column；reserved name永不进入 result/error。
- DataFrame `lower_risk` ranking source 的 `gt/ge/lt/le` 在编译时分别翻转成
  `lt/le/gt/ge`，literal cutoff保持caller原值；Task 15 ranking 已是 normalized higher-event-
  risk，不翻转。`eq/ne`不变。Score source拒绝 membership/between/column RHS。
- Kernel `condition specification/evaluation is invalid` 不成为 public prefix；Task 17 按
  第16节翻译 key。不得在 Task 17 再计算 comparison truth。

Condition parity tests 必须逐 operator/Boolean/missing/type/nonfinite/timezone/effective boundary
比较 compiled public tree 与直接 private kernel result，逐 row 的 truth/status/reason完全相同。

## 6. Actions、roles 与 historical mapping

Public config中的`action_name`字段就是caller symbolic action key；summary tables使用较短列名
`action_key`表示同一值，不是第二概念。不冻结 `approve/decline/refer` 等业务文本。Closed
`action_role` inventory唯一为：

```text
selected, rejected, review, request_information, limited, other
```

每个action key必须由`action_role_mapping`恰好映射一个closed role；多个keys可共享role，
unknown/extra/duplicate/conflicting/missing mapping均invalid config。Tuple顺序是caller-declared
action order；不得按key文本猜role或自动生成action。每row final action key因此精确解析为一个
互斥role。Role集合语义唯一为：

| Role | Selected population | Rejected population | Review capacity | 专用语义 |
|---|---:|---:|---:|---|
| `selected` | yes | no | no | 普通caller-declared selection |
| `limited` | yes | no | no | caller-declared constrained selection；不计算、推荐或优化额度 |
| `rejected` | no | yes | no | simulated rejection |
| `review` | no | no | yes | substantive manual review |
| `request_information` | no | no | yes | additional-information queue；与review互斥但同进capacity |
| `other` | no | no | no | 只进all-action和action-key metrics，不进role专用metrics |

```text
selected population = role in {"selected", "limited"}
rejected population = role == "rejected"
review capacity = role in {"review", "request_information"}
```

Eligibility rules只允许发出`rejected/review/request_information`；不得发出
`selected/limited/other`。Decision rules可发任意closed role。Default action可映射任意role；
unknown action只允许`rejected/review/request_information`，不得映射
`selected/limited/other`。Default不得覆盖unknown path。Unknown/default action keys和所有rule
actions都必须在完整mapping中；unknown-action rows有final action，属于decided scope。

Action summary排序固定为closed role rank
`selected, limited, rejected, review, request_information, other`，再按mapping tuple ordinal，
最后按action key；不得依赖dict或字母顺序。

Historical action是可选 input column。启用时必须同时提供非空 exact mapping；每个 raw
literal通过 Task 16 scalar whitelist验证，并精确映射到已声明 action name。Task 17不猜测
raw语义。Unmapped raw/missing值只累计为 sanitized
`not_verifiable/historical_action_unmapped`，不得进入result/error/repr。Transition table只含
mapped action names与模拟action names；historical policy version仅作provenance，不重放policy，
不把历史action当ground truth或因果assignment。

## 7. Two-phase rule semantics

所有rules按 `phase`，再按ascending `priority`，最后以`rule_key`只作稳定展示 tie-break；
但同一phase priority必须唯一，duplicate priority invalid。Rule key跨两phase全局唯一；
priority是exact int（bool拒绝），范围0..9999。Enabled false为inactive且不参与precedence。
每个rule继承strategy version，可有自己的 `[effective_from, expires_at)`；rule窗口必须完全
位于strategy窗口内。Inactive与unknown不同。

### 7.1 Eligibility phase

Condition true精确表示“eligibility rule matched”，不是“资格通过”。按priority处理：

1. false继续；unknown立即停止action path并使用caller的`unknown_action_name`，不得当false；
2. 第一个true成为applied/base rule/action；`stop_on_hit=True`立即结束；
3. `stop_on_hit=False`继续收集multi-hit；后续true与base action相同是累积hit，不改变action；
   与base action不同是conflict，final使用`unknown_action_name`并记录conflict；
4. eligibility有任何matched rule后不进入decision phase；没有match且没有unknown才继续。

Eligibility rules可发caller action keys；Task 17不按key猜“hard/soft/refer”。其完整mapping
role必须是`rejected/review/request_information`，其他role为invalid config。

### 7.2 Decision phase

进入后使用相同ascending priority、first-match/multi-hit/conflict语义。Unknown立即使用
`unknown_action_name`。至少一个match形成base action；无match时使用显式
`default_action_name`并记录`default_action_applied`。No decision rules是合法的，全部rows走
default；全strategy `rules=()`也合法并产生`no_rules` limitation。

### 7.3 Diagnostic evaluation、overlap 与 reachability

所有active conditions为diagnostics完整求值，以发现overlap；precedence path另标记
`evaluated`或`not_evaluated`。Stop/conflict/eligibility terminal之后的rule即使diagnostic truth
为true也不能改变action，只记overlap且path_status=`not_evaluated`。`applied rule`、
`matched rules`、`overlapping rules`、`unknown rules`、`inactive rules`严格分开。
Unreachable rule是`evaluated_n=0`且active的summary evidence，不是exception。Overlap不改变
final action；conflict只按上述reached multi-hit产生。

Strategy在evaluation_time不属于`[effective_from, expires_at)`时完全inactive：不分配action，
每row final action为missing，status/reason=`inactive/strategy_inactive`，constraints为
not_applicable，conditions不读取row values。Static config/tree仍先完整validation。

## 8. Score、probability 与 cutoff sources

Task 17不定义自动band对象；caller用closed conditions显式声明cutoff与open/closed bands。
所有cutoff caller-declared，ties由operator精确决定：`ge`包含等号、`gt`不包含；lower bound
和upper bound同理。Gap落到后续rule/default，overlap按第7节记录。

- `ranking_score` 可为任意finite real，只用于rule/cutoff、distribution和observed replay；
  `[0,1]` ranking绝不升级为probability。
- `event_probability`只能来自已通过第3.1节完整alignment的
  `BinaryRiskValidationResult.predictions["event_probability"]`。它必须finite、位于`[0,1]`，
  并完全继承Task 15的`positive_label`、`probability_provenance`和higher-positive-event-risk
  语义；可用于rule/cutoff与expected loss。Task 17不接收raw positive-label literal，不重算或
  重新解释positive event。
- DataFrame ranking source要求column real numeric non-bool、全部non-missing finite，并显式
  `ranking_score_direction`。Column absent与direction absent/conflict均invalid schema/config。
- Task 15 source来自frozen `predictions`的`ranking_score`/`event_probability`列；probability
  absent仍可ranking-only。`score_direction`必须是frozen
  `higher_positive_event_risk`。Wrong result schema、row scope、positive-label/probability
  provenance mismatch立即alignment exception。
- Ranking source cardinality独立计算：config声明DataFrame ranking column且Task 15 result也
  提供ranking时，报`duplicate_score_source`，不得比较后择一。Event probability不存在
  DataFrame来源或source precedence。
- Task 15 analytical operating point、threshold_analysis、gains和calibration不被读取为policy。

Every DataFrame row有纯rule decision opportunity；仅Task 15 predicted positions有Task 15
score/outcome。Temporary sources以RangeIndex构造并按position赋值，绝不使用原index。Immature
`is_evaluable=False`的target不可进入event/observed denominator。

未提供Task 15 result时，probability summary、expected loss/payoff及probability-dependent
constraints为`not_applicable/source_not_requested`；提供Task 15但只有ranking score时为
`not_verifiable/probability_unavailable`。`[0,1]` ranking仍只按ranking处理。Result provenance
只记录`probability_source="task15"`、availability、positive event是否已声明、positive-label
type family、Task 15 evidence fingerprint/version和probability provenance，不保存raw label。

## 9. Business evidence 与 replay semantics

三种claim边界固定：

1. `action_only`：总能报告actions/rule evidence；不声称outcome。
2. `observed_replay`：只用Task 15 predictions中`is_evaluable=True`的mature target，按其
   positive_label形成observed event indicator。Ranking score可有可无。它是历史关联，不是
   action causal effect。
3. `model_based_expectation`：只用合法event probability；明确是expectation，不是observed。

Exposure可选，必须是DataFrame real numeric non-bool、finite、nonnegative；任一metric要求的
row缺失/非法时不得partial sum。`loss_fraction`可为finite `[0,1]` scalar或同样合法column。
三类loss严格分开：

1. **Expected loss**逐row为`event_probability * exposure * loss_fraction`。Probability只来自
   aligned Task 15；ranking绝不使用。任一component未声明为not_applicable，已声明但缺失为
   not_verifiable。
2. **Assumption-based observed-event loss**逐mature/evaluable row为
   `positive_event_indicator * exposure * loss_fraction`，metric key唯一为
   `assumption_based_observed_event_loss`。它只是caller assumptions施加于observed event的模拟
   evidence，不得称actual/realized/Task 15 observed/historical booked loss。
3. **Actual observed loss**只消费Task 15 `business_metrics`中available、
   `segment_kind="all"`的aggregate `observed_loss_sum`及兼容`exposure_sum`、support和unit。
   Task 17不得按new action、role、segment或time slice重新分配；任何non-overall request为
   `not_verifiable/observed_loss_not_resegmentable`。缺合法aggregate evidence为
   `not_verifiable/observed_loss_not_mature`。

`action_assumptions`是唯一value/cost来源，每项exact tuple
`(action_key, assumed_value, assumed_cost)`；value为finite real，cost为finite nonnegative real，
均为per-row常量。若声明，必须恰好覆盖action inventory各一次，无extra/missing/duplicate key；
不得使用DataFrame columns作为第二来源。`exposure_unit`是caller opaque unit key，所有exposure、
loss、value、cost和payoff使用同一unit；不做currency conversion或真实收益声明。未声明
assumptions时value/cost/payoff全部`not_applicable/action_assumption_not_declared`，不得按0。
声明assumptions或任何exposure/loss metric时，`exposure_unit`必须是non-empty safe ASCII unit key；
否则invalid config。

唯一payoff metric为：

```text
assumption_based_payoff
= assumed_action_value - assumed_action_cost - expected_loss
```

只有完整assumptions、probability、exposure和loss fraction都available时计算。Precedence为：
assumptions absent -> not_applicable；probability -> exposure -> loss fraction缺口依次
not_verifiable；否则available。它不是actual profit/revenue，不进入threshold search、rule reorder、
winner recommendation或strategy optimization。

最小evidence固定包含：pooled row/action counts/rates；action-level count/rate；rule hit、
applied、sole-hit、overlap、unknown、conflict、not-evaluated；ranking/probability mean/min/max
（仅合法support）；mature target event count/rate；exposure count/sum；expected loss、
assumption-based observed-event loss、overall actual observed loss、assumption-based payoff；
historical transitions；constraints。Single-class event rate仍defined；zero
denominator为`undefined/zero_denominator`；missing evidence按是否声明为`not_applicable`或
`not_verifiable`，不得伪造0。Task 17不输出discrimination/calibration metrics、pricing、limit
recommendation、profit optimization或causal结论。Rule summary还必须给出fixed-order
incremental action count、leave-one-out changed-action count和mature target capture；这些只在
同一frozen rows上重放caller规则，不改变策略、不搜索order，support不足时structured
unavailable。

## 10. Caller-declared constraints

`DecisionConstraint.metric`保持恰好13个keys，不接受arbitrary expression。统一row sets为：

```text
input scope = all DataFrame row positions
active scope = input scope when strategy active, otherwise empty
decided scope = active rows whose final_action_name is non-null
metric-evaluable scope = decided rows with every source required by that metric
```

Unknown/default rows属于decided scope；final action为空的inactive/unavailable rows不属于。
13项唯一语义矩阵如下；`D=decided_n_rows`，`S=selected population`，`R=rejected population`，
`Q=review capacity`，`M`为Task 15 mature/evaluable membership：

| metric_key | scope | numerator / value | denominator | unit | required evidence | support_n | zero/missing behavior |
|---|---|---|---|---|---|---|---|
| `action_count` | action key，必须填action_name | `count(action==key)` | none | rows | decided actions | `D` | 无source缺口；support不足undefined |
| `action_rate` | action key，必须填action_name | `count(action==key)` | `D` | fraction | decided actions | `D` | `D=0` zero_denominator |
| `selected_rate` | closed role；action_role必须为`selected` | `count(role in {selected,limited})` | `D` | fraction | complete role mapping | `D` | `D=0` zero_denominator |
| `rejected_rate` | closed role；action_role必须为`rejected` | `count(role==rejected)` | `D` | fraction | complete role mapping | `D` | `D=0` zero_denominator |
| `review_count` | closed role；action_role必须为`review` | `count(role in {review,request_information})` | none | rows | complete role mapping | `D` | support不足undefined |
| `review_rate` | closed role；action_role必须为`review` | `count(Q)` | `D` | fraction | complete role mapping | `D` | `D=0` zero_denominator |
| `request_information_rate` | closed role；action_role必须为`request_information` | `count(role==request_information)` | `D` | fraction | complete role mapping | `D` | `D=0` zero_denominator |
| `selected_exposure_sum` | closed selected population | `sum(exposure on S)` | none | exposure unit | finite nonnegative exposure for every S row | `count(S)` | any missing invalidates; no S gives value 0 then insufficient_support when below minimum |
| `expected_loss_sum` | global | `sum(p*exposure*loss_fraction)` on all decided rows | none | exposure unit | aligned Task 15 probability plus complete exposure/loss fraction | metric-evaluable count, which must equal `D` | undeclared source not_applicable; incomplete source not_verifiable |
| `expected_loss_rate` | global | same expected-loss sum | `sum(exposure)` on the identical complete scope | fraction | same as expected_loss_sum | metric-evaluable count, must equal `D` | exposure denominator 0 undefined; never divides by row count |
| `observed_loss_sum` | global overall only | Task 15 `business_metrics` overall `observed_loss_sum` | none | Task 15 unit | available Task 15 aggregate actual observed-loss evidence | `n_observed_loss_mature_rows` | no Task 15: not_applicable; Task 15 supplied but absent/not mature: not_verifiable; no scoped use |
| `observed_loss_rate` | global overall only | same Task 15 actual observed-loss sum | Task 15 overall `exposure_sum` on compatible evidence | fraction | both Task 15 aggregate rows available with same support/unit | mature support | no Task 15: not_applicable; incomplete: not_verifiable; zero exposure undefined; never resegments |
| `selected_event_rate` | closed selected population | mature positive-event rows in `S & M` | `count(S & M)` | fraction | aligned Task 15 target/positive-event evidence | `count(S & M)` | denominator 0 undefined; immature rows excluded |

All 13 metrics allow both`le`and`ge`; equality passes exactly (`le: value <= threshold`,
`ge: value >= threshold`).Count/sum thresholds arefinite nonnegative。Action/role/event proportion
rates及expected_loss_rate thresholds在`[0,1]`；observed_loss_rate threshold只要求finite
nonnegative，因为caller actual loss可能大于exposure。`minimum_support`是exact non-boolean
int>=1。Any action_name/action_role combination not exactly allowed by the matrix is invalid strategy
config；actual observed-loss constraints are only global and any scoped attempt is invalid config。

Status precedence fixed to：1) invalid config -> exception；2) inapplicable scope -> exception；
3) required source not declared -> `not_applicable/source_not_requested`；4) source declared but incomplete ->
`not_verifiable` with specific source reason；5) denominator zero -> `undefined/zero_denominator`；
6) `support_n < minimum_support` -> `undefined/insufficient_support`；7) available value compare
threshold。Pass is`available/constraint_satisfied`；fail is`available/constraint_failed`。
Gap is`ge: actual-threshold`or`le: threshold-actual`，violation magnitude is`max(0,-gap)`。
Constraint failure is result not exception。

Constraints只测量已冻结simulated actions，绝不反写row action、搜索cutoff、调整rule order、
target approval rate、repair policy或选择alternate strategy。改变constraints必须保持
`row_decisions`逐cell相同。

## 11. Effective time 与 version

Strategy key/version为non-empty ASCII `[A-Za-z0-9._-]`，最长64。`effective_from`、
`evaluation_time`必须exact datetime；`expires_at`可None；窗口为
`[effective_from, expires_at)`且start<end。三者timezone-awareness必须一致；不读取当前日期、
环境timezone或locale。

Rule不拥有version，只继承strategy version；rule bounds均可None，但两端类型、awareness、
静态顺序及包含于strategy window均需validation。Enabled false与window外均为inactive；
inactive不等于unknown。Rule窗口编译为private root effective window，root version使用strategy
version；public child不得有version/time。不选择latest，不根据历史policy version执行。

### 11.1 Bounded segment/time stability

`segment_columns`和`time_slice_column`只接受caller显式声明的existing unique string columns。
Time slice必须是caller预先分桶的values；Task 17不按日/周/月/季度自动分桶、不猜列名。Selector
values在任何missing/grouping/sorting前通过Task 16 exact scalar whitelist；不返回raw values。
多个segment columns分别处理，不生成segment×segment，只生成每个segment与time的交叉。
Segment columns不得重复，time slice不得同时出现在segments；两者均不得与ranking score、
historical action、exposure或column-valued loss fraction使用同一column，冲突为invalid config。

Long-form summary的`scope_type`唯一为`overall/segment/time_slice/segment_time`：overall覆盖相关
rows；segment表示一个column的匿名category；time_slice表示匿名time bucket；segment_time表示
一个segment category与time bucket交集。`scope_column`可保存批准column name；segment_time使用
只含两个validated column names的canonical JSON array string，不含values或delimiter歧义。
Category/time bucket按首次最小row position
分配stable 0-based ordinal；missing为内部专用bucket并排在所有non-missing ordinals后。不调用
value的`str()`/`repr()`，只使用Task 16批准exact scalar的normalized `(type family, value)`
structural key，不使用index；空交集不生成row。未声明scope只输出overall；0-row
DataFrame保持typed schemas；all-missing time生成一个missing bucket。

Rule stability至少输出hit/unknown/overlap count/rate及rate overall baseline/absolute delta；action
stability由action_summary输出per-key count/rate，由business_summary的role scopes输出selected/
rejected/review-capacity/unknown-action rate及baseline/delta；unknown-action精确指final action key等于
config unknown_action_name（包括unknown/conflict fallback）的rows。Business stability在evidence
可用时输出event-probability mean、selected-event rate、expected-
loss rate、exposure sum、assumption-based payoff及baseline/delta。Delta精确为scope rate/value减
overall value；overall自身不输出baseline/delta。每项status独立。Stability只描述变化，不修改
actions、不生成threshold、不作fairness结论、不选winner或进入Task 19 governance score。

Hard limits固定为segment columns 4、每segment categories 100、time slices 100、total derived
non-overall scopes 1,000、三张summary合计scope-metric rows 100,000。Max合法，max+1在summary
assembly前以`decision strategy resource limit exceeded: <key>`失败，不静默截断。Metric
denominator 0为undefined；声明source但evidence不完整为not_verifiable。

## 12. Frozen result schemas

所有表都是新建deep-copied DataFrame、`RangeIndex`、固定列/dtype；nullable strings用
`string`、nullable numerics用`Int64/Float64`、booleans用`boolean`。空表保持完整schema。
精确八张表如下。

三张long-form summaries统一scope顺序为overall、segment（按config column order/ordinal）、
time_slice（ordinal）、segment_time（segment config order/segment ordinal/time ordinal），再按各表
domain key和固定metric order。Overall的scope_column/ordinals为`pd.NA`；segment只有
scope_ordinal，time_slice只有time_slice_ordinal，segment_time两者均有。不得按raw bucket value
排序。

### 12.1 `row_decisions`

```text
row_position:int64, decision_status:string, decision_reason:string,
base_action_name:string, final_action_name:string, applied_rule_key:string,
matched_rule_count:int64, unknown_rule_count:int64, overlap_rule_count:int64,
conflict_rule_count:int64, override_applied:boolean,
historical_mapping_status:string
```

按row_position升序，恰好input_n_rows行；不返回raw feature、identifier/group、historical raw
value、score或probability。Strategy inactive时action/rule cells为`pd.NA`。`override_applied`
只表示base与final action不同（unknown/conflict fallback），不是production override。

顶层counts精确为：`input_n_rows=len(data)`；`decided_n_rows`是本表
`final_action_name.notna()`的行数；`unavailable_n_rows`是`final_action_name.isna()`的行数。
二者互斥且`decided_n_rows + unavailable_n_rows == input_n_rows`。Unknown/default rows均decided；
strategy inactive时decided为0、unavailable为input_n_rows。Active valid strategy不得产生final
action为空且没有structured reason的row。

### 12.2 `rule_evaluations`

```text
row_position:int64, rule_key:string, phase:string, priority:int64,
rule_order:int64, path_status:string, truth:string, status:string, reason:string,
is_applied:boolean, is_overlap:boolean, is_conflict:boolean
```

按row_position、phase(eligibility, decision)、priority、rule_key。只含enabled且strategy active
rules；每row-rule一行。`path_status`仅`evaluated/not_evaluated`。Truth仅
`true/false/unknown`。Inactive rules不展开row detail，进入summary。

### 12.3 `rule_summary`

```text
scope_type:string, scope_column:string, scope_ordinal:Int64,
time_slice_ordinal:Int64, phase:string, priority:int64, rule_key:string,
action_key:string, action_role:string, metric_key:string,
metric_value:Float64, numerator:Float64, denominator:Float64,
support_n_rows:int64, unit:string, status:string, reason:string,
finding_key:string
```

一行一个`scope x rule x metric`。Metric固定顺序为`evaluated_count, hit_count, hit_rate,
unknown_count, unknown_rate, not_evaluated_count, applied_count, sole_hit_count, overlap_count,
overlap_rate, conflict_count, incremental_action_count, leave_one_out_changed_action_count,
captured_event_count, target_capture_rate`；非overall scopes的rate metrics随后各输出
`<metric>_overall_baseline`和`<metric>_absolute_delta`。Count的denominator为`pd.NA`；rate保存
精确numerator/denominator。Incremental/leave-one-out语义沿第9节；target capture denominator是
all mature positive events。Disabled/window inactive仍为每metric
`inactive/rule_inactive`，不得用一个status覆盖其他metrics。

### 12.4 `action_summary`

```text
scope_type:string, scope_column:string, scope_ordinal:Int64,
time_slice_ordinal:Int64, action_key:string, action_role:string,
metric_key:string, metric_value:Float64, numerator:Float64,
denominator:Float64, support_n_rows:int64, unit:string,
status:string, reason:string, finding_key:string
```

一行一个`scope x action key x metric`。Metric固定顺序为`action_count, action_rate,
evaluable_event_count, event_count, event_rate, exposure_sum, expected_loss_sum,
assumption_based_observed_event_loss_sum, assumed_action_value_sum, assumed_action_cost_sum,
assumption_based_payoff_sum`；非overall rate metrics随后输出overall baseline和absolute delta。
每个metric独立status/reason。Action排序按第6节role rank/mapping ordinal/key。Inactive strategy
返回inventory metric rows但status inactive，不伪造available zero rate。

### 12.5 `business_summary`

```text
scope_type:string, scope_column:string, scope_ordinal:Int64,
time_slice_ordinal:Int64, action_key:string, action_role:string,
metric_key:string, metric_value:Float64, numerator:Float64,
denominator:Float64, support_n_rows:int64, unit:string,
status:string, reason:string, finding_key:string
```

一行一个`scope x optional action/role x business metric`；overall metric的action cells为nullable。
Metric固定顺序：

```text
row_count, decided_rate, ranking_score_mean, ranking_score_min,
ranking_score_max, event_probability_mean, event_probability_min,
event_probability_max, observed_event_count, observed_event_rate,
selected_rate, rejected_rate, review_capacity_rate, unknown_action_rate,
exposure_sum, expected_loss_sum, expected_loss_rate,
assumption_based_observed_event_loss_sum,
actual_observed_loss_sum, actual_observed_loss_rate,
assumed_action_value_sum, assumed_action_cost_sum,
assumption_based_payoff_sum, selected_event_rate, historical_mapped_rate
```

Non-overall stability metrics additionally emit`<metric>_overall_baseline`and
`<metric>_absolute_delta`。Actual observed loss只在overall/action-null/role-null rows合法；scoped
rows为`not_verifiable/observed_loss_not_resegmentable`。每个metric拥有独立status/reason。

### 12.6 `constraint_summary`

```text
constraint_key:string, metric:string, operator:string, threshold:float64,
action_name:string, action_role:string, actual_value:Float64,
status:string, reason:string, support_n:int64, gap:Float64,
violation_magnitude:Float64, finding_key:string
```

按config tuple order；threshold是caller明确允许公开的business threshold，不是condition
literal。

### 12.7 `historical_transitions`

```text
historical_action_name:string, simulated_action_name:string,
row_count:int64, row_rate:Float64, status:string, reason:string,
finding_key:string
```

先按historical mapped action的closed role rank/mapping ordinal/key，再按simulated action同一顺序；
最后至多一行unmapped
aggregate，其historical/simulated names均`pd.NA`、reason
`historical_action_unmapped`。Rate denominator是全部input rows。未启用historical source为空表。

### 12.8 `provenance`

```text
provenance_key:string, provenance_value:string, status:string,
reason:string, finding_key:string
```

固定key顺序：`strategy_schema_version, condition_kernel_version, strategy_key,
strategy_version, strategy_effective_from, strategy_expires_at, evaluation_time,
strategy_fingerprint, row_identity, score_source, score_direction,
probability_provenance, prediction_scope, positive_label_type_family,
task15_evidence_status, task15_evidence_version, task15_evidence_fingerprint,
task16_evidence_status, task16_config_fingerprint,
historical_policy_version, historical_mapping_count, action_mapping_count,
action_assumption_count, rule_count, constraint_count, segment_column_count,
time_slice_declared, exposure_unit`。Missing项使用structured status，不插字符串
`None`。不得记录raw positive label、condition literal或historical raw value。

`task15_evidence_version`固定为合同批准的`task15-binary-risk-validation-v1`。
`task15_evidence_fingerprint`由Task 17对aligned frozen result的sanitized canonical payload计算，
payload含validation mode、source/probability provenance、positive-label type family、fold/row
positions、availability/counts及prediction数值的canonical bytes，但不含raw positive label；result
只保存SHA-256 digest。它不修改或声称Task 15新增public fingerprint字段。

`DecisionStrategyResult`不得保存input DataFrame/view、config/condition/rule/constraint/mapping
object、private kernel object、Task 15/16 result、estimator、Figure、raw literal或raw historical
value。

## 13. Closed status、reason、finding 与 warning vocabulary

Public status唯一inventory：

```text
available, unavailable, undefined, not_applicable, not_verifiable, inactive
```

Closed reason inventory（available evidence的正常计算使用`computed`）：

```text
computed, no_rows, no_rules, no_matching_rule, default_action_applied,
unknown_condition, missing_operand, type_mismatch, nonfinite_operand,
timezone_mismatch, missing_evaluation_time, strategy_inactive, rule_inactive,
outside_effective_window, required_action_role_unmapped,
score_unavailable, probability_unavailable,
exposure_unavailable, loss_fraction_unavailable, label_not_evaluable,
observed_loss_not_mature, observed_loss_not_resegmentable,
action_assumption_not_declared, insufficient_support, zero_denominator,
constraint_satisfied, constraint_failed, constraint_not_verifiable,
historical_action_unmapped, historical_action_unavailable,
provenance_mismatch, row_scope_mismatch, source_not_requested,
not_evaluated_by_precedence, rule_overlap, rule_conflict
```

不得临时增加近义reason。Missing required column、invalid config/schema/alignment是exception，
不降级为table reason；evidence availability缺口才structured。Simulated rejected action不是
error。Constraint failed不是severity finding。Task 17不公开severity字段；`finding_key`是
稳定linkage，按`strategy:<key>`、`rule:<rule_key>`、`constraint:<constraint_key>`、
`action:<ordinal>`、`historical:transition`构造，不含raw values。

Reason/status/table pairing固定：`action_assumption_not_declared`只配`not_applicable`并只出现在
action/business summaries；`observed_loss_not_resegmentable`只配`not_verifiable`并出现在
business/constraint summaries；`source_not_requested`配`not_applicable`；source unavailable/
not mature reasons配`not_verifiable`；`zero_denominator`和`insufficient_support`只配`undefined`；
constraint satisfied/failed只配`available`；`strategy_inactive`在row/rule/action/business中配
`inactive`、在constraint summary中配`not_applicable`；`rule_inactive`只配`inactive`。Resource scope
超限是exception key而非result reason。所有这些都是non-severity evidence，不新增finding severity。

Warnings唯一顺序：strategy、source、mapping、outcome、constraint、resource。Limitations唯一
顺序：`simulated_actions_not_executed, historical_comparison_not_causal,
model_expectation_not_observed, outcome_support_limited,
custom_score_provenance_caller_declared`，只输出适用项且去重稳定。

## 14. Privacy、sanitized provenance 与 fingerprint

Validation必须在error构造前sanitise；error/repr/warning不得插入caller literal、raw historical
action、identifier/group值或raw policy值。Samples只允许row position；本Task不返回raw-value
samples。Description只接受`description_key`，不冻结或返回任意全文。

`strategy_fingerprint`是lowercase 64-character SHA-256 hex，不用Python `hash()`。输入为UTF-8
canonical JSON：`sort_keys=True, ensure_ascii=True, allow_nan=False,
separators=(",", ":")`。Canonical payload含schema version、strategy key/version/time、rule
keys/phase/priority/action/flags/window、condition node/operator/source shape、literal exact
canonical value、mapping、constraints、source declarations和business assumptions；只在内存
transient使用，result只保留digest。Datetime统一ISO 8601，tuple保持有序array，NumPy scalar先
转批准built-in scalar；dict/set/callable拒绝。

Provenance只保存批准keys、counts、type families、public business thresholds、versions和digest。
Condition literal只能通过`literal_present/literal_count/literal_type_family`的condition-shape
计数参与sanitized diagnostics，不明文进入result；condition cutoff也视为literal，不进入
provenance。Constraint threshold可进入其专用表。Fingerprint用于确定性/provenance equality，
不是密码存储或抗猜测安全承诺。

## 15. Determinism 与 input immutability

稳定规则如下：row position升序；phase固定eligibility后decision；priority升序；rules tie-break
key；actions按第12.4节；constraints按caller tuple；transitions按mapped/action inventory；statuses/
reasons/warnings/limitations按本合同inventory；finding/provenance keys按固定表序。Float运算使用
float64与固定公式；不得依赖set/hash iteration、pandas index alignment、dict order、当前日期、
locale、环境timezone、随机数或parallel completion。

Validation/evaluation/result assembly必须只读。包括失败路径在内，所有input snapshots逐cell/
dtype/index相同；tuples/mappings/tree identity和内容不变。Repeated run的tables/dtypes/order、
warnings、limitations和fingerprint完全相同。

## 16. Resource boundaries 与 stable public errors

Condition budgets直接继承Task 16：depth 8、nodes 128、membership literals 100、literal string
1024。Task 17不得创建不同condition上限。其他hard limits：

```text
input rows                         100_000
all rules                         100
eligibility rules                  50
decision rules                     50
constraints                        50
action-role mappings               50
historical raw-action mappings     50
action assumptions                 50
distinct simulated actions         50
rule-evaluation result rows  1_000_000
segment columns                      4
categories per segment column      100
time slices                        100
total derived scopes             1_000
scope-summary rows              100_000
key/version/action/description      64 ASCII chars
```

Minimum 0适用于rows/rules/constraints/mappings；condition至少1 node。超过hard limit在任何row
evaluation前exception，不做silent truncation。由于row/rule evidence是决策可审计性的一部分，
本Task不截断row_decisions或rule_evaluations；caller需缩小input/rules后重试。Resource limit
及requested/allowed只进入sanitized error key/count，不含raw内容。无新core/optional dependency，
不建立parallel/realtime engine。

只使用`ValueError`，稳定public prefixes为：

```text
decision strategy config is invalid: <key>
decision strategy input schema is invalid: <key>
decision condition is invalid: <key>
strategy source alignment: <key>
decision strategy resource limit exceeded: <key>
```

Config包括wrong dataclass exact type、key/priority/action/mapping/constraint/time/source cardinality；
input schema包括DataFrame type、labels、duplicate columns、missing/unsupported columns/dtypes；
alignment包括Task 15/16 schema/provenance/positions/scope；resource包括所有fixed budgets。
Task 15 maturity alignment的closed stable keys至少精确区分：

```text
maturity_count_mismatch
non_time_maturity_count_nonzero
time_mode_maturity_mismatch
```

`maturity_count_mismatch`只用于不属于mode-specific三元关系的通用maturity/count schema不一致；
non-time任一冻结maturity count非0使用`non_time_maturity_count_nonzero`；time-based mature、
immature、excluded或evaluable关系不一致使用`time_mode_maturity_mismatch`。Keys固定且不拼接fold
id或raw值；public prefix仍为`strategy source alignment:`。
Task 16 kernel `condition specification is invalid`中operator/operand/children/version/window keys映射
到`decision condition is invalid`；depth/nodes/membership/string budget keys映射resource prefix；
kernel `condition evaluation is invalid`中的unsupported dtype映射input schema，其他值域语义保留
为row-level unknown。Public error不得泄露private module path。

Validation precedence固定：exact top-level types -> resource-independent config shape -> key/time/
priority uniqueness -> complete public condition trees -> fixed resource limits -> input DataFrame/
column labels -> declared source schema -> Task 15 frozen validator -> Task 15 validation mode -> common
position/fold/evaluable alignment -> mode-dependent maturity alignment -> top-level count/source-scope
alignment -> Task 16 alignment -> all value family/finite/range scans -> evaluation。任何validation
error前不得返回partial result或消费Task 15 score/probability。

## 17. Future implementation test matrix

Tests必须是可使错误实现失败的直接证据，而非主题清单。

### 17.1 Public API/schema

- Exact six symbols、signatures、keyword-only boundary、frozen field order/default/type hints、
  eight table schemas/dtypes/empty schemas及三张long-form逐metric status；fixture直接证明count
  available同时probability unavailable、hit available同时capture undefined，增加字段、换表、
  shared status、裸dict或mutable dataclass均失败。
- Hand rows断言unknown/default均计入decided、inactive计入unavailable、两counts互斥且完整覆盖；
  active row final action为空且无reason的错误实现失败。
- `__all__`断言v0.1 prefix -> exact Task 15 suffix -> exact Task 16 suffix -> exact Task 17 suffix；
  recursive import断言private kernel symbols不可见。

### 17.2 Condition compilation parity

- 每个atomic operator、AND/OR/NOT完整truth、nested column/literal/column-to-column fixtures逐cell
  比较public compiled与private direct evaluation；若Task 17自行比较、把unknown当false、使用
  pandas alignment，会在missing/type/nonfinite/timezone/duplicate-index fixtures失败。
- Invalid operator、list/set/dict/callable/object/expression、child version/time、cycle、whole-tree
  invalid later rule、depth/node/membership 7/8/9及127/128/129、99/100/101边界；证明不能
  short-circuit validation或扩大DSL。
- Lower-risk score在exact cutoff的`gt/ge/lt/le` fixture证明operator翻转且equality closure正确；
  `[0,1]` ranking fixture证明不生成probability evidence。

### 17.3 Rule/action semantics

- Hand-worked hard/review/request-information symbolic names覆盖eligibility true=no-entry-to-
  decision、no-match继续、decision first match、default、unknown terminal、same-action multi-hit、
  different-action conflict、stop-on-hit、diagnostic overlap与path not-evaluated；错误的last-match、
  unknown-as-false或overlap改action会得到不同row decisions。
- Duplicate key/priority、same priority、disabled/window-inactive、unreachable、empty rules、
  eligibility发selected/limited/other失败；断言inactive/unknown/not-evaluated不混淆。
- Arbitrary names、two names-one role、complete/missing/extra/duplicate mapping；手算证明limited
  进入selected、request_information进入review capacity、other不进专用metrics，unknown映射
  selected/limited/other失败；固定文本猜role的实现会失败。

### 17.4 Score/probability/alignment

- DataFrame ranking-only、Task 15 probability、Task 15 ranking-only、ranking+probability、同kind
  ranking双source conflict、finite/range/direction、threshold equality、higher/lower risk；无Task 15
  时probability metrics not_applicable，Task 15 ranking-only expected loss not_verifiable，`[0,1]`
  ranking不升级且raw positive label不进result。
- 除Task 15现有validator外，直接构造prediction positions与fold validation union不同、正确position
  但错误fold_id、错误is_evaluable、fold validation重复、prediction重复、missing/extra position、
  predicted/excluded overlap、declared source scope未覆盖、wrong counts；均必须被Task 17增强alignment
  检查拒绝。合法多fold与duplicate pandas index按positions通过。
- 合法`stratified_kfold` direct fixture逐fold使用手工冻结值
  `validation_n=4, evaluable_validation_n=4, validation_mature_n=0,
  validation_excluded_n=0, immature_validation_n=0`；alignment必须通过，ranking及合法Task 15
  event probability均可消费，证明non-time maturity 0不表示evaluable 0或evidence unavailable。
- 合法`time_forward` direct fixture按Task 15 frozen fold tuples/counts证明
  `validation_mature_n == evaluable_validation_n`且
  `validation_excluded_n == immature_validation_n == validation_n - evaluable_validation_n`；alignment
  必须通过，spy/未来日期sentinel证明Task 17未重算或重分类maturity。
- Non-time fixture只把`validation_mature_n`改为正数，必须以
  `strategy source alignment: non_time_maturity_count_nonzero`失败；time-based fixture分别破坏
  mature/evaluable及excluded/immature关系，必须以
  `strategy source alignment: time_mode_maturity_mismatch`失败。另有通用count-schema mismatch
  fixture命中`maturity_count_mismatch`。这些expected均为手工常量或Task 15批准公式，不调用Task
  17 production helper生成。
- Mode-dependent fixtures与原有union/fold_id/is_evaluable/duplicate/missing/extra/overlap/coverage
  fixtures同时保留；测试必须使“只删除mature=evaluable equality却不验证non-time三项0值和time
  三元关系”的错误实现失败。
- Immature/excluded rows仍可pure-rule action但score condition unknown，target不进入denominator；
  以future-only outcome改变fixture证明不改变actions/cutoffs。

### 17.5 Constraints/business math

- 13个constraint keys逐项各有pass/fail/equality、allowed scope、required-evidence unavailable、
  denominator/support边界；独立hand constants证明action/selected/rejected/review denominators均D、
  expected-loss rate除以同scope exposure而非row count、selected-event仅mature selected、actual
  observed loss只overall。Expected为literal常量/独立公式，不调用production helper生成。
- 手算`p*E*L`expected、`y*E*L`assumption-based observed-event loss和assumed value-cost-expected
  loss payoff；schema/metric-key断言使把event-derived loss命名为actual observed loss的实现失败；
  ranking不能expected，actual observed loss不能按action重分组，assumptions未声明不按0，complete
  mapping及missing/extra/duplicate/nonfinite assumptions失败，component unavailable保留。
- Constraint tuple A/B仅改threshold，逐cell断言row_decisions完全相同；自动threshold search/
  action rewrite实现会失败。
- Single class、no evaluable rows、ranking+mature outcome、probability without outcome分别证明
  observed与expected正交；ranking不得生成expected loss。

### 17.6 Historical comparison/privacy

- Exact explicit mapping、unmapped/missing raw values、partial mapping、stable transition order与
  common denominator；historical policy column/value改变但mapping/actions相同不得产生winner/
  causal字段。
- Recursive inspection所有result/error/repr/warnings，使用unique secrets作为feature、literal、
  identifier、raw historical action、positive label、raw assumption payload及segment/time values，
  断言均不存在；只允许positions、mapped action keys、type families、ordinals、counts和fingerprint。
- Same exact config重复SHA-256相同，literal变化digest变化但明文不出现；不允许Python hash。

### 17.7 Effective time、determinism、immutability、resources

- Strategy/rule start included、expiry excluded、invalid bounds、naive/aware mismatch、missing/
  wrong evaluation time、inactive strategy、rule inheritance；系统clock变化不影响结果。
- Duplicate index、repeated runs、input/result/config/tree/mapping snapshots与failure-before-mutation；
  shuffleddict/set不可作为input。
- Segment-only、time-only、segment×time、两个segments分别处理、不生成segment×segment、missing
  bucket、first-position ordinal、duplicate index、rate/baseline/delta hand math、probability unavailable
  but counts available、repeated ordering；stability不得修改actions。
- 每个resource minimum/max/max+1，包括condition/rule-evaluation、category/time/scope/summary-row
  caps；max+1必须pre-evaluation stable error，合法max不截断，scope budget不得静默截断。

### 17.8 Compatibility/distribution

- Full v0.1、Task 15、Task 16 regression保持结果/errors/signatures/exports相对顺序；spy证明不调用
  Task 15 metrics/folds或Task 16 audit重算，condition只进private kernel。
- No dependency diff；wheel/sdist包含`decision_strategy.py`和合同，source-free clean install可
  import exact six symbols，private kernel仍不public；旧workflow/report/CLI不变。

Implementation完成后必须先`bash scripts/verify-uv-env.sh`，再用`.venv/bin/python`运行full
pytest、Ruff check、Ruff format check、build、wheel/sdist source-free clean install、distribution
smoke和diff checks。不得用system Python。

## 18. Future implementation allowlist

Task 17 implementation的准确allowlist仅为：

```text
src/sharper/decision_strategy.py
src/sharper/__init__.py
tests/test_decision_strategy.py
tests/test_public_api.py
tests/test_distribution.py
docs/decisions/task17-preloan-eligibility-strategy-contract.md
SPEC.md
IMPLEMENTATION_PLAN.md
README.md
docs/api.md
```

下列只读消费且默认禁止修改：

```text
src/sharper/_condition_kernel.py
src/sharper/risk_validation.py
src/sharper/data_audit.py
```

本合同确认现有kernel足够，不批准修改private kernel。禁止修改Task 15/16 public schema/
behavior、workflow/reporting/CLI、Tasks 18--20 modules、dependencies/lock/version，禁止optimizer、
public generic DSL和自动修复。

## 19. Compatibility、完成定义与未决问题

新API完全opt-in；v0.1 signatures/dataclass fields/defaults/errors/reports/CLI/exports、Task 15与
Task 16 frozen contracts均不变。Package version在Task 17 implementation也保持`0.1.0`，除非
后续独立release合同批准。

Task 17 implementation只有在：本合同唯一一次full review已完成、targeted fixes完成且bounded
closure为Go；所有public
schema/condition parity/Task 15 cross-table alignment/precedence/unknown safety/score-probability
isolation/13项constraint non-optimization/三类loss与assumptions/bounded stability/historical
privacy/time/version/determinism/immutability/resources/
compatibility tests通过；full pytest/Ruff/build/clean-install/distribution/diff gates通过；无Task
18--20越界后，方可获得implementation review Go。

**Closure状态：** `T17-C1--C7`均已关闭；唯一一次full contract review为`No-Go`，targeted contract
fixes已完成，bounded contract closure为`Go`，P0/P1/P2均为`0`。Roadmap的caller-defined action
names通过完整closed role mapping实现；single-strategy execution与reference/challenger要求通过
historical mapped reference actions和本次simulated challenger actions的same-row paired
transitions同时满足，不需要第二strategy输入或historical policy重放。

Task 17合同阶段已完成，不得再次进行开放式Task 17 full contract review。合同当前为
**Approved — Go**；Implementation为 **Implementation complete — review Go**。
`T17-A1` post-approval targeted compatibility amendment已批准，bounded amendment closure为
`Go`且P0/P1/P2均为`0`。Task 17唯一一次开放式full implementation review已经完成，final
bounded implementation closure为`Go`，Implementation为 **Implementation complete — review
Go**，后续不得再次执行开放式Task 17 full implementation review。Tasks
18--20尚未开始；v0.2整体尚未完成或发布；当前package version仍为`0.1.0`。本轮不提交、push、
tag或发布，也不开始Task 18。
