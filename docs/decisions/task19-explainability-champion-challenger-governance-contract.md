# Task 19 — Explainability, Champion/Challenger and Governance 精确合同

**Contract 状态：Approved — Go（amended v2）。**

**Implementation 状态：In progress。**

本记录已完成唯一一次 full contract review，verdict为`No-Go`；第一次bounded contract-review closure
同样为`No-Go`，其中`T19-CR-01/02/03/05/06/09/10/11/12/15/16`已`Closed`，残余
`T19-CR-04/07/08/13/14`经targeted repair后在bounded contract-review re-closure中取得`Go`。
`T19-CR-01..16`在原approved checkpoint均为`Closed`，Residual P0/P1/P2为`0/0/0`。
随后post-approval implementation-blocker adjudication确认`T19-CR-13`中两个error keys在approved semantics下不可达；
本次targeted amendment已应用，bounded amendment closure为`Go`并已`Closed`。`T19-CR-13`的历史closure保持`Closed`，
其post-approval blocker与amendment现均已`Closed`；`T19-CR-15`保持`Closed`，仅直接acceptance wording随本amendment机械同步。Task 18 保持
`Approved — Go`、`Implementation complete — review Go`；本合同不重开 Tasks 15--18。

Historical note: the immutable `442fdc0...` approved checkpoint froze the original v1/78-key inventory; the current
amended v2 contract intentionally removes the two unreachable keys and freezes 76 executable keys.该历史checkpoint
保持immutable，后续implementation以新的v2 checkpoint为baseline。

**正式名称：** Task 19 — Explainability, Champion/Challenger and Governance。

## 1. 权威依据、目标与边界

本合同服从 `AGENTS.md`、`SPEC.md`、`IMPLEMENTATION_PLAN.md`、
`docs/decisions/v02-roadmap-contract.md` 和 Tasks 15--18 frozen contracts。冲突时必须先
修订并 review 治理文件，不得在实现中猜测上游语义。

Task 19 唯一执行：

```text
consume -> validate -> trace -> compare -> summarize -> record governance evidence
```

它读取 Task 15 risk-validation、Task 16 data-audit、Task 17 decision-strategy 和 Task 18
lifecycle-monitoring 的 frozen results，并接受本合同冻结的sanitized structured model-attribution、
prediction-profile、performance-slice和governance-metadata declarations，形成privacy-safe
explanation、candidate comparison、prediction drift、performance stability、criterion evidence、
simulated recommendation、governance metadata与audit trail。

明确不做：

- 重算 condition kernel、eligibility、rules、actions、alerts、episodes、states、transitions、
  backtests、scenario/lifecycle/monitoring metrics 或 missingness drift；
- 训练、fit、calibrate、调参、threshold search、策略/预警生成、candidate optimizer，或读取/调用/
  introspect estimator、Pipeline、predict method、feature matrix、target vector；
- callable、lambda、表达式、脚本、插件、任意 Python、动态 import、registry 或服务；
- causal inference、treatment effect、uplift、A/B claim、adverse-action notice、法律公平/监管认证；
- SHAP、LIME、反欺诈、AML、KYC、设备/IP/velocity/network fraud、generic ML/MLOps；
- 外部写入、通知、部署、自动 promotion、Task 17/18 config mutation 或 Task 20 workflow/report/CLI；
- 修改 v0.1、Tasks 15--18 API/schema/ownership、mandatory dependencies、lock 或 package version。

Explainability 只表示 descriptive attribution、reason trace、source lineage 或 metric evidence；
不是 causal proof。禁止宣称 `feature X caused rejection`、`alert Y caused loss` 或
`challenger caused improvement`。

## 2. DAG、模块和 ownership

```text
Task 15 ─┐
Task 16 ─┼─> Task 17 / Task 18 ─> Task 19 ─> Task 20
         └───────────────────────────────────┘
```

实现只允许新增 `src/sharper/model_governance.py`、`tests/test_model_governance.py`，并为批准的
result-only plot精确修改`src/sharper/visualization.py`、`tests/test_visualization.py`、
`src/sharper/__init__.py`、`tests/test_public_api.py`、`tests/test_distribution.py`、`README.md`和
`docs/api.md`（后两者仅在合同获批后的implementation阶段）。`model_governance`只读消费
四个owner result类型、下文四个structured declarations及pandas/numpy/stdlib；不得import workflow、
reporting、CLI、visualization、Task 20或`analysis.py`。`visualization`只可import
`GovernanceResult`并读取其frozen tables，不得反向调用`evaluate_governance`。Task 19 是
单向 consumer：Task 15 拥有 risk/OOF/maturity/business，Task 16 拥有 quality/leakage/
missingness/kernel，Task 17 拥有 eligibility/actions/constraints/transitions，Task 18 拥有
warning/episode/state/transition/monitoring/lifecycle/scenario；Task 19 不重算这些 facts。
因此Task19对Task15/16存在直接、只读public-result dependency；这不建立任何Task15/16反向依赖，
predecessor modules不得import或调用Task19，DAG仍严格单向向前。

模块职责和最小目录严格为：

| module/file | owns | may depend on | must not do |
|---|---|---|---|
| `src/sharper/model_governance.py` | structured evidence validation、TVD drift、thresholded stability、trace/comparison/governance tables | Tasks 15--18 public result types、pandas、NumPy、scikit-learn、stdlib | estimator/raw-data execution、owner recomputation、plotting、workflow/report/CLI |
| `src/sharper/visualization.py` | Task 19 result-only figures | `GovernanceResult`、matplotlib | recompute importance/drift/stability/recommendations；read owner results/raw data |
| `tests/test_model_governance.py` | Task 19 domain/contract tests | public Task 19 API | implementation helpers as public surface |
| `tests/test_visualization.py` | Task 19 result-only plot tests | public result/plot API | owner/model execution |

实现顺序冻结为：先实现四个structured declaration与exact validation；再物化model attribution、
prediction TVD、performance stability和metadata vertical slices；随后接入既有trace/comparison/
recommendation surface；最后实现五个result-only plots、exports、distribution与compatibility gates。
每一步都必须先通过对应direct contract tests才可进入下一步，不得在合同获批前开始。

## 3. 唯一 public API（十二个 symbols）

按以下顺序追加到 `sharper.__all__`，且仅此十二个：

1. `GovernanceEvidenceRef`
2. `GovernanceCandidate`
3. `GovernanceCriterion`
4. `GovernanceExplanation`
5. `GovernanceAttributionEvidence`
6. `GovernancePredictionProfile`
7. `GovernancePerformanceEvidence`
8. `GovernanceMetadata`
9. `GovernancePolicy`
10. `GovernanceResult`
11. `evaluate_governance`
12. `plot_model_governance`

不得新增exception、第二个plot入口、manager、registry、builder、alias或`**kwargs`。

```python
@dataclass(frozen=True)
class GovernanceEvidenceRef:
    source_task: Literal["task15", "task16", "task17", "task18"]
    source_result_position: int
    source_table: str
    source_use: Literal[
        "comparison_criterion", "diagnostic", "explanation",
        "attribution_context", "drift_context", "stability_context",
    ]
    candidate_key: str | None
    expected_source_fingerprint: str | None
    field_key: str | None = None
    metric_key: str | None = None
    side_key: str | None = None
    column_key: str | None = None
    scope_key: str | None = None
    scope_column: str | None = None
    scope_position: int | None = None
    time_position: int | None = None
    fold_id: int | None = None
    row_position: int | None = None
    entity_position: int | None = None
    from_row_position: int | None = None
    to_row_position: int | None = None
    scenario_key: str | None = None
    reference_scenario_key: str | None = None
    comparator_scenario_key: str | None = None
    rule_key: str | None = None
    phase_key: str | None = None
    action_key: str | None = None
    action_role: str | None = None
    constraint_key: str | None = None
    slice_role: str | None = None
    row_kind: str | None = None
    state_key: str | None = None
    from_state_key: str | None = None
    to_state_key: str | None = None
    statistic_key: str | None = None
    category_key: str | None = None
    category_position: int | None = None
    pattern_key: str | None = None
    episode_ordinal: int | None = None
    notification_ordinal: int | None = None
    event_ordinal: int | None = None
    numeric_value: float | None = None
    finding_key: str | None = None
    provenance_key: str | None = None

@dataclass(frozen=True)
class GovernanceCandidate:
    candidate_key: str
    candidate_family: Literal["model", "strategy", "warning_scenario"]
    source_task: Literal["task15", "task17", "task18"]
    source_result_position: int
    source_candidate_key: str | None
    expected_source_fingerprint: str | None
    version: str | None
    declared_role: Literal["champion", "challenger"]
    declared_state: Literal["candidate", "under_review", "approved", "rejected", "retired"]
    evidence_refs: tuple[GovernanceEvidenceRef, ...]

@dataclass(frozen=True)
class GovernanceCriterion:
    criterion_key: str
    candidate_family: Literal["model", "strategy", "warning_scenario"]
    source_task: Literal["task15", "task16", "task17", "task18"]
    source_table: str
    metric_key: str
    scope_key: str
    scope_position: int | None
    rule_key: str | None
    criterion_role: Literal["decision", "diagnostic"]
    required_for_promotion: bool
    direction: Literal["higher_is_better", "lower_is_better", "target_range", "not_directional"]
    target_low: float | None = None
    target_high: float | None = None
    minimum_support: int = 1
    required_support_unit: str | None = None
    priority: int = 0

@dataclass(frozen=True)
class GovernanceExplanation:
    explanation_key: str
    candidate_key: str
    method: Literal[
        "rule_trace", "reason_trace", "source_lineage", "metric_evidence",
        "scenario_delta_trace", "state_transition_trace",
        "coefficient_direction", "native_importance", "permutation_importance",
    ]
    source_ref: GovernanceEvidenceRef
    feature_key: str | None
    relation: Literal["positive", "negative", "neutral", "not_directional"] | None
    priority: int
    status: Literal["available", "unavailable", "undefined", "not_applicable", "not_verifiable"]
    reason: str | None

@dataclass(frozen=True)
class GovernanceAttributionEvidence:
    candidate_key: str
    method: Literal["coefficient_direction", "native_importance", "permutation_importance"]
    feature_key: str
    metric_key: str | None
    value: float
    relation: Literal["positive", "negative", "neutral", "not_directional"]
    evaluation_scope: Literal["not_applicable", "holdout", "oof"]
    support_n: int
    uncertainty_std: float | None
    permutation_repeats: int | None
    random_state: int | None
    evidence_as_of: datetime
    source_ref: GovernanceEvidenceRef

@dataclass(frozen=True)
class GovernancePredictionProfile:
    candidate_key: str
    snapshot_key: str
    snapshot_role: Literal["reference", "current"]
    analysis_as_of: datetime
    prediction_kind: Literal["ranking_score", "event_probability"]
    scope_key: str
    scope_position: int | None
    reference_state_fingerprint: str
    bin_boundaries: tuple[float, ...]
    bin_counts: tuple[int, ...]
    support_n: int
    missing_n: int
    bootstrap_repeats: int
    random_state: int
    source_ref: GovernanceEvidenceRef

@dataclass(frozen=True)
class GovernancePerformanceEvidence:
    candidate_key: str
    snapshot_key: str
    snapshot_role: Literal["reference", "current"]
    window_start: datetime
    window_end: datetime
    evidence_as_of: datetime
    evaluation_scope: Literal["holdout", "oof"]
    scope_key: str
    scope_position: int | None
    target_values: tuple[bool, ...]
    ranking_scores: tuple[float, ...] | None
    event_probabilities: tuple[float, ...] | None
    assignment_mechanism: Literal["randomized", "non_randomized", "unknown"]
    common_support: Literal["verified", "unverified"]
    bootstrap_repeats: int
    random_state: int
    source_ref: GovernanceEvidenceRef

@dataclass(frozen=True)
class GovernanceMetadata:
    metadata_key: str
    metadata_scope: Literal["governance", "candidate"]
    candidate_key: str | None
    purpose_key: str
    owner_key: str
    materiality: Literal["low", "medium", "high"]
    assumption_keys: tuple[str, ...] = ()
    limitation_keys: tuple[str, ...] = ()
    monitoring_thresholds: tuple[tuple[str, float], ...] = ()
    issue_status: Literal["none", "open", "monitoring", "resolved"] = "none"
    remediation_status: Literal["not_required", "planned", "in_progress", "complete"] = "not_required"

@dataclass(frozen=True)
class GovernancePolicy:
    governance_key: str
    governance_version: str
    analysis_as_of: datetime
    candidates: tuple[GovernanceCandidate, ...]
    comparison_pairs: tuple[tuple[str, str], ...]
    criteria: tuple[GovernanceCriterion, ...]
    metadata: tuple[GovernanceMetadata, ...]
    minimum_comparable_criteria: int = 1
    human_review_mode: Literal["promotion_only", "all_recommendations"] = "promotion_only"
    evidence_refs: tuple[GovernanceEvidenceRef, ...] = ()
    explanations: tuple[GovernanceExplanation, ...] = ()
    entity_alignment: Literal["not_requested", "owner_verified"] = "not_requested"

@dataclass(frozen=True)
class GovernanceResult:
    governance_key: str
    governance_version: str
    governance_fingerprint: str
    analysis_as_of: datetime
    candidate_count: int
    comparison_pair_count: int
    criterion_count: int
    explanation_count: int
    source_snapshot_status: Literal["verified", "unverified", "not_applicable"]
    entity_alignment_status: Literal["verified", "unverified", "not_applicable"]
    evidence_time_status: Literal["verified", "unverified", "not_applicable"]
    explanations: pd.DataFrame
    model_attributions: pd.DataFrame
    prediction_drift: pd.DataFrame
    performance_stability: pd.DataFrame
    candidate_comparisons: pd.DataFrame
    governance_evaluations: pd.DataFrame
    recommendations: pd.DataFrame
    governance_summary: pd.DataFrame
    governance_metadata: pd.DataFrame
    provenance: pd.DataFrame
    warnings: tuple[str, ...]
    limitations: tuple[str, ...]

def evaluate_governance(
    policy: GovernancePolicy,
    *,
    risk_validations: tuple[BinaryRiskValidationResult, ...] = (),
    data_audits: tuple[DataAuditResult, ...] = (),
    decision_strategies: tuple[DecisionStrategyResult, ...] = (),
    lifecycle_monitorings: tuple[LifecycleMonitoringResult, ...] = (),
    model_attributions: tuple[GovernanceAttributionEvidence, ...] = (),
    prediction_profiles: tuple[GovernancePredictionProfile, ...] = (),
    performance_evidence: tuple[GovernancePerformanceEvidence, ...] = (),
) -> GovernanceResult: ...

def plot_model_governance(
    result: GovernanceResult,
    *,
    kind: Literal[
        "importance", "candidate_comparison", "prediction_drift",
        "performance_stability", "governance_summary",
    ],
) -> matplotlib.figure.Figure: ...
```

字段顺序、defaults、keyword-only 边界、Literal vocabulary 和 shallow-frozen 语义均冻结。
whole-policy-first validation 在任何 source read 前执行；结果 tables 独立物化，不保留输入、
policy、上游 DataFrame、estimator、feature value、condition、path 或 external handle。

## 4. Candidate、pair 和 comparison identity

Candidate key 必须是 non-empty safe ASCII identifier，不从名称、顺序、position 或 metric
value 推断，且在整个调用中全局唯一。`candidate_key`只是在Task 19内使用的privacy-safe governance
identifier，不是owner source identity。恰好一个 candidate 必须声明 `champion`；zero or more 可声明 `challenger`；
multiple champions invalid。Candidate inventory必须非空且恰好一个caller-declared champion；zero
candidates、zero champion或multiple champions均为invalid config，绝不first-candidate-wins。Champion
state只能是`approved`；`candidate/under_review/rejected/retired` champion均invalid。一个approved champion、
零challenger合法：仍可物化其attribution、drift、stability、metadata与summary，comparison/evaluation/
recommendation tables typed-empty。

`comparison_pairs` 是显式有序 `(champion_candidate_key, challenger_candidate_key)` inventory；
两 key 必须存在、不同、同 family、角色分别正确，并且必须exact覆盖每个declared challenger一次且仅一次，
每一pair第一项都是唯一champion、第二项是一个distinct challenger。Zero challenger要求empty pairs；
missing/duplicate/extra/reversed/challenger-challenger pair及challenger==champion均invalid。禁止implicit
all-pairs、global-best或key guessing；multiple challengers独立评估。Candidate family 封闭为 `model`（Task 15）、
`strategy`（Task 17）、`warning_scenario`（Task 18）；跨 family 禁止比较，Task 16 仅为
audit evidence source。

Pair-level semantic validity is a separate closed predicate from pair coverage: when the pair container,
candidate inventory, champion, pair uniqueness, and exact challenger coverage are all valid, a pair whose two
candidate families differ is `invalid_pair`. Missing, extra, reversed, self, duplicate, or otherwise incomplete
coverage remains `invalid_pair_coverage` or `duplicate_pair` according to the earlier checks; no pair-level
catch-all is permitted. A normative representative is one valid model champion, one valid strategy challenger,
and exactly one complete pair: the first failure is `model governance: invalid_pair`.

Challenger state matrix封闭为：`candidate/under_review/approved`均可进入evidence evaluation；其中
`approved`只表示caller-declared current governance state，不使其成为champion，也不自动promotion。
`rejected/retired`是合法声明但构成closed hard veto：完成全部基础config/type/source/privacy validation后，
该pair recommendation固定`reject_challenger`。Task19从不修改role/state；任何recommendation都只是offline
label。除challenger `rejected/retired`外没有metadata、reason-string、priority或implementation-derived hard
veto；以后若需扩展必须先修订合同。

### 4.1 Owner result collections与candidate binding

Source architecture严格分为两类且不得互换：A类是下列Tasks15--18 typed **owner result sources**，
只允许按closed registry读取既有owner facts；B类是Task19 typed **structured declaration sources**
`GovernanceAttributionEvidence/GovernancePredictionProfile/GovernancePerformanceEvidence/
GovernanceMetadata`，只允许按各自closed schema验证并物化Task19-owned evidence。B类不是owner result、
不能占用`source_result_position`，A类也不能伪装成generic evidence object或替代B类数值声明。

四个keyword-only owner collections分别是exact tuples：`risk_validations`、`data_audits`、
`decision_strategies`、`lifecycle_monitorings`；每项必须是对应exact public result type。Tuple order是本次
调用内0-based source declaration order，`source_result_position`必须是exact non-bool built-in int、
non-negative且在对应tuple内；它不是persistent/global ID。每个tuple最多16项；同一object在同一tuple
重复出现是duplicate declaration并fail closed（仅用于调用内重复检查，不成为source identity），而
public-value-identical deep copy是独立position declaration且不自动失败。Candidate semantic source仍不得
重复声明。
所有tuple elements、candidate bindings、refs和structured declarations即使未被pair/plot选中也必须完成
基础type/scalar/privacy/source validation；不得以inactive/unselected为由绕过wrong type、locator或fingerprint。

| candidate family | authoritative task/type | candidate locator | same-result multiple candidates |
|---|---|---|---|
| `model` | Task15 `BinaryRiskValidationResult` | `source_result_position`选择整个single validation subject；`source_candidate_key=None` | no；一个Task15 result最多绑定一个candidate |
| `strategy` | Task17 `DecisionStrategyResult` | position选择整个result且`source_candidate_key == result.strategy_key` | no；一个Task17 result最多绑定一个candidate |
| `warning_scenario` | Task18 `LifecycleMonitoringResult` | position选择result且`source_candidate_key`必须是`monitoring_summary.scenario_key`中存在的exact scenario key | yes；同一result可绑定多个不同scenario keys |

Authoritative candidate source identity分别为`(task15, position)`、
`(task17, position, strategy_key)`、`(task18, position, scenario_key)`。同一semantic identity不得用两个
candidate keys创建alias；model/strategy共享同一position必定重复，warning scenarios只有locator不同才
合法。Expected fingerprint必须与第7.2节owner-authoritative digest一致；没有authoritative digest的
Task15使用`None`，不得由caller字符串制造identity。任何out-of-range position、wrong result type、
missing/duplicate locator、fingerprint mismatch或same-source alias均在读取criterion前fail closed。

每个`GovernanceCriterion.candidate_family`必须与pair两侧family相同；registry entry还必须允许该family和
`comparison_criterion` use。Task16所有entries及registry标记diagnostic/explanation-only的owner rows绝不
成为directional comparison criterion。Task17 pair不得使用Task18 source，Task18 pair不得使用Task17
source，model pair不得使用Tasks17/18 metric；不降级为not-applicable。
每个criterion必须分别从champion与challenger各自绑定的owner result/locator解析唯一row；两侧source
position、candidate locator与fingerprint独立验证。禁止用champion row同时填充challenger value、反向复用、
或因metric/scope文字相同而跨result取值。
具体地，每侧candidate的`evidence_refs`必须恰有一条`source_use="comparison_criterion"` ref匹配
criterion的task/table/metric/scope/rule identity；两条ref各自携带完整table-specific locator并分别解析。
零条或多条均是invalid source declaration，不得从DataFrame猜row。

Comparison identity 精确为：

```text
(governance_key, champion_candidate_key, challenger_candidate_key,
 criterion_key, source_task, source_table, metric_key, scope_key,
 scope_position, rule_key, support_unit)
```

finding_key、raw value、repr、str、Python hash、pandas index 和 memory address 不参与 identity。

Owner result或其public DataFrames被deep-copy且values/dtypes/column order相同时，candidate/ref resolution
必须落到同一semantic source/row。Pandas index、`iloc`、object identity和memory address均禁止作为
owner-table locator。

## 5. Explainability inventory

每条caller-declared trace explanation只有一个candidate、source reference和method；它是
`GovernancePolicy.explanations`输入并逐条产生`explanations`输出row，不是model-attribution数值容器。
method closed inventory：

| method | owner evidence |
|---|---|
| `rule_trace` | Task 17 rule evaluation/summary |
| `reason_trace` | Task 17/18 frozen reason/status |
| `source_lineage` | approved owner provenance |
| `metric_evidence` | Task 15--18 frozen metric row |
| `scenario_delta_trace` | Task 18 scenario comparison |
| `state_transition_trace` | Task 18 state/transition |
| `coefficient_direction` | `GovernanceAttributionEvidence(method="coefficient_direction")` |
| `native_importance` | `GovernanceAttributionEvidence(method="native_importance")` |
| `permutation_importance` | `GovernanceAttributionEvidence(method="permutation_importance")` |

SHAP/LIME 不是 method。status 仅为 `available`、`unavailable`、`undefined`、
`not_applicable`、`not_verifiable`；非available必须有 reason，missing evidence 绝不代表
zero contribution。`explanations` public table只按caller declaration position严格升序；candidate、method、
priority、source task/table/key和explanation key均不得重排public rows。`GovernanceExplanation.priority`
只是保留的semantic/display metadata，不参与governance decision，也不改变normalized table order。
没有对应structured attribution时不合成零或猜测model attribute；对应trace为
不产row；若caller显式声明trace，其status/reason必须按第11节从合法source映射。

### 5.1 Model attribution exact ownership

Task 19拥有**结构化model attribution evidence的验证、规范化、排序、trace与输出**，不拥有训练或
任意model执行。Task 15 result不保存estimator、feature matrix、coefficients或importance，因此三种
model evidence必须由caller通过`model_attributions`显式提供，并由`source_ref`关联对应candidate的
Task 15 frozen validation provenance。Task 19不得接受estimator、Pipeline、callable、raw feature
matrix/values、target values、intercept、class labels或multioutput payload，也不得使用`getattr`/
`hasattr`/model protocol。

- `coefficient_direction`：仅单一binary positive-event输出；`value`是finite signed coefficient，
  `relation`必须按value与0的exact comparison分别为positive/negative/neutral；不接收intercept。
- `native_importance`：`value`是caller已从单一binary fitted model提取的finite non-negative native
  importance；`relation="not_directional"`、`evaluation_scope="not_applicable"`，不猜
  `feature_importances_`或其他attribute。
- `permutation_importance`：caller在Task 15 frozen holdout/OOF evaluable row set上预先计算；
  `metric_key`必须是Task 15批准且可用的model metric，`value=baseline_metric-permuted_metric_mean`
  对lower-is-better metric改为`permuted_metric_mean-baseline_metric`，因此正值统一表示性能下降；
  `uncertainty_std`是repeats的population standard deviation（`ddof=0`），`permutation_repeats`
  是exact positive int且`random_state`是exact non-negative int。Task 19不执行permutation、prediction
  或metric计算。

三种evidence均一row一candidate×method×feature，feature key为validated non-empty safe identifier；
不得含raw feature value。缺证据是unavailable，不代表0。排序按candidate declaration、method固定顺序
`coefficient_direction,native_importance,permutation_importance`、feature evidence声明顺序。

## 6. Prediction drift、performance stability 与 governance metadata

### 6.1 Prediction drift

Task 19拥有基于caller-provided bounded prediction histograms的prediction drift；它不重算Task 16
missingness/feature drift。每个`GovernancePredictionProfile`只携带一个candidate/snapshot/scope的
聚合计数，不含row、entity、target或raw score。每个comparison unit必须恰有一个reference和一个
current profile，且candidate、prediction_kind、scope、`reference_state_fingerprint`与
`bin_boundaries`完全相同；current `analysis_as_of`不得早于reference且二者不得晚于governance
`analysis_as_of`。

`bin_boundaries`必须是9个严格递增finite float，从而冻结10 bins；event probability必须使用
`0.1,...,0.9`，ranking score的boundaries由reference snapshot caller声明并通过相同reference-state
fingerprint固定，current不得重估。`bin_counts`恰为10个non-negative exact ints且sum等于
`support_n`；`missing_n`为独立non-negative exact int，不进入分布。任一侧`support_n=0`为
`undefined/insufficient_support`。

唯一drift metric为total-variation distance：

```text
p_i = reference_count_i / reference_support_n
q_i = current_count_i / current_support_n
prediction_tvd = 0.5 * sum_i(abs(q_i - p_i))
```

值域`[0,1]`；不输出PSI、KS、Jensen-Shannon或Wasserstein，不推断root cause。输出一row一matched
profile pair，保留两侧support/missing counts、source fingerprints与as-of。Direction固定为
`lower_is_better`，但drift本身只作diagnostic，除非后续CR-02/03批准的criterion registry显式纳入。
不确定性使用相同exact non-negative `random_state`和相同`bootstrap_repeats`：以
`numpy.random.default_rng(random_state)`依次对reference、current按各自10-bin probabilities和
support执行multinomial resample，每repeat计算一次TVD，输出`uncertainty_std=std(ddof=0)`；repeats
必须在`[2,1000]`。该过程只读取aggregate counts，不生成/返回synthetic rows。

### 6.2 Performance stability

Task 19自己只计算roadmap批准的model prediction-performance stability，并且只从
`GovernancePerformanceEvidence`内显式、bounded、position-free的frozen window vectors计算；不从
Task15 tables、Task17/18 owner facts或raw DataFrame重分段。每个row代表一个candidate×snapshot×
time/group anonymous scope。Reference/current unit必须candidate、evaluation_scope、scope identity和
source fingerprint相同，windows不重叠、`window_end <= governance analysis_as_of`，且
`len(target_values)`等于所提供prediction vector长度并在资源门内。`target_values`只接受built-in
bool；scores必须finite；probability必须在`[0,1]`。

metric inventory只允许Task 15现有数学：ranking source为`roc_auc`；probability source为
`brier_score`。ROC-AUC按Task 15/sklearn semantics，single-class为`undefined/single_class`；Brier为
`mean((p-y)^2)`。每个metric输出reference/current value、`delta=current-reference`、两侧support；
direction分别固定为`higher_is_better`、`lower_is_better`。不确定性对每个window独立使用其exact
non-negative`random_state`与`bootstrap_repeats in [2,1000]`，以
`numpy.random.default_rng(random_state)`按tuple position有放回抽取n个positions，逐repeat用同一
冻结metric公式计算，输出available replicate values的population standard deviation（`ddof=0`）；
ROC bootstrap single-class replicate跳过，少于2个available replicates时整个对应window metric为
`undefined/insufficient_bootstrap_support`。每个metric输出reference/current value、uncertainty、
`delta=current-reference`、两侧support；
这是stability evidence，不自动把delta解释为better/worse。`assignment_mechanism`必须原样输出，
`non_randomized`或`unknown`不得产生A/B或causal claim。group scopes只使用anonymous
`scope_position`，不得返回raw group label。Reference/current comparison只有两侧
`common_support="verified"`且source identity一致时可比较；否则输出
`not_verifiable/support_not_comparable`，不做intersection repair。

### 6.3 Governance metadata

`GovernanceMetadata`是bounded caller declarations，不是model registry：每个policy恰好一个
`metadata_scope="governance",candidate_key=None` row，每个candidate允许零或一个
`metadata_scope="candidate",candidate_key=<existing key>` row；scope/candidate identity不得重复。
其exact inventory为purpose/owner/materiality、ordered assumption/limitation keys、ordered
`(threshold_key, finite threshold)`、issue status和remediation status。Keys必须是safe identifiers；
threshold keys唯一。禁止deployment URI、endpoint、model binary、credential、free-text、legal approval
或合规认证。`governance_metadata`按上述field order一row一个field/item规范化输出。

### 6.4 Roadmap traceability（normative）

Approved Task 19 roadmap拆为以下16项且每项恰有一个归宿：

| roadmap capability | status/owner | input | public API/result surface | resource gate | direct test obligation | out-of-scope note |
|---|---|---|---|---|---|---|
| linear coefficients | IN SCOPE — OWNED BY TASK19 validation/trace | structured attribution | `GovernanceAttributionEvidence` → `model_attributions` | attribution rows | signed/relation/missing/privacy | no estimator/intercept/multiclass |
| native importance | IN SCOPE — OWNED BY TASK19 validation/trace | structured attribution | same | attribution rows | nonnegative/not-directional | no open model attribute protocol |
| holdout/OOF permutation importance | IN SCOPE — OWNED BY TASK19 validation/trace | caller precomputed evidence | same | attribution rows/repeats | scope/seed/n/std/formula provenance | no permutation execution |
| source-feature provenance | IN SCOPE — SOURCE-BACKED OWNER EVIDENCE | attribution `source_ref` | `model_attributions`/`explanations` | evidence refs | wrong/missing ref | no raw feature values |
| model champion/challenger comparison | IN SCOPE — OWNED BY TASK19 | Task15 candidates/results | existing candidate/evaluation/recommendation tables | pair×criteria | frozen fold/support comparison | no cross-family comparison |
| Task17 policy comparison inventory | IN SCOPE — SOURCE-BACKED OWNER EVIDENCE | `DecisionStrategyResult` | explanations/comparisons/summary | owner evidence rows | no-recompute spy | no Task17 math |
| Task18 warning comparison inventory | IN SCOPE — SOURCE-BACKED OWNER EVIDENCE | `LifecycleMonitoringResult` | explanations/comparisons/summary | owner evidence rows | AM-04/support preservation | no Task18 delta recompute |
| reason/override/mapping/fallback audit | IN SCOPE — SOURCE-BACKED OWNER EVIDENCE | Tasks17/18 traces | explanations/governance summary | explanation/source rows | source/reason/coverage trace | no rule/alert execution |
| evaluated/hit/order/base-final/episode/override facts | IN SCOPE — SOURCE-BACKED OWNER EVIDENCE | Tasks17/18 frozen result rows | explanations/governance summary | source rows | exact trace/no-recompute | no raw conditions or owner execution |
| external assignment/time/segment/common-support provenance | IN SCOPE — OWNED BY TASK19 recording | performance declarations | `performance_stability` assignment/window/scope/support fields | performance rows | randomized/non-randomized/unknown and verified/unverified support | no experiment routing/causal lift |
| prediction drift | IN SCOPE — OWNED BY TASK19 math | prediction profiles | `prediction_drift` | profile/bin rows | TVD hand calculation/reference reuse | no feature/missingness drift |
| performance-by-time/group | IN SCOPE — OWNED BY TASK19 math | performance evidence | `performance_stability` | vector/row limits | AUC/Brier/window/scope cases | no raw group labels |
| Task16 missingness/feature-profile governance summary | IN SCOPE — SOURCE-BACKED OWNER EVIDENCE | `DataAuditResult` frozen tables | metric explanations + governance summary status/counts | Task16 source rows | n/budgets/warnings preserved; no-recompute spy | no new profile/drift or snapshot authorization |
| model/rule/policy stability | IN SCOPE — mixed: model owned; policy source-backed | performance evidence + Tasks17/18 frozen stability | stability/comparison/summary tables | performance/source rows | model math + owner no-recompute | no upstream stability recompute |
| governance metadata | IN SCOPE — OWNED BY TASK19 recording | `GovernanceMetadata` | `governance_metadata` | metadata rows | exact inventory/privacy | no registry/service/legal approval |
| result-only plots | IN SCOPE — OWNED BY TASK19 presentation | `GovernanceResult` only | `plot_model_governance` | one figure/call | five kinds/no-recompute/Figure lifecycle | no dashboard/report/CLI/save |

No approved roadmap capability is deferred. Explicit roadmap exclusions remain causal inference、SHAP/LIME、
adverse-action generation、fairness/regulatory certification、automatic approval/deployment、Task20
workflow/report/CLI/release与anti-fraud。

## 7. Closed owner-source registry、EvidenceRef与snapshot proof

### 7.1 EvidenceRef exact semantics

`GovernanceEvidenceRef`只定位一条approved owner evidence；其`candidate_key`仅校验该ref绑定的既存
Task19 candidate，绝不创建/识别candidate，也不声明formula或recommendation。
所有非适用locator fields必须`None`，适用fields必须exact typed。Resolution顺序固定为：validate all
scalars → task → result position/type → table registry → locator shape → authoritative fingerprint → unique
row → owner schema/dtype/status → normalized evidence。Wrong task/table/shape、out-of-range、fingerprint
mismatch、zero-row match或multiple-row match均是invalid ref并fail closed；不得first-row-wins或转成
unavailable。只有ref合法且唯一row本身是owner unavailable/undefined/not-verifiable时才保留其owner
status/reason进入evidence。

### 7.2 Exact owner-source registry（38 entries）

Allowed-use vocabulary：`C=model/strategy/warning_scenario comparison criterion`，`D=diagnostic`，
`E=explanation`，`X=structured attribution/drift/stability context`。`—`表示field不存在且Task19不得
合成。Registry以外task/table/use/metric组合全部unsupported；不存在“other frozen metrics”fallback。

| # | task/type + table | kind/use/family | exact row locator | metric/value | status/reason | support/scope/time |
|---:|---|---|---|---|---|---|
| 1 | T15 `BinaryRiskValidationResult.metrics` | metric C/E/X model | `scope,fold_id,metric,statistic` | metric=`roc_auc,average_precision,normalized_gini,ks_statistic,brier_score,log_loss,expected_calibration_error`; value=`value` | `status,reason` | `n_rows,n_positive,n_negative`; fold/cutoff via `folds` |
| 2 | T15 `.gains` | metric C/E model | `scope,fold_id,requested_fraction` | `event_rate,capture,lift` select same-named value/status/reason group | per-value status/reason | `selected_n,total_positive_n,actual_fraction`; fold |
| 3 | T15 `.threshold_analysis` | metric C/E model | `scope,fold_id,threshold_kind,threshold` | `sensitivity,specificity,precision,negative_predictive_value,f1,accuracy,predicted_positive_rate` | selected metric `_status/_reason` | `tp,fp,tn,fn`; fold/threshold |
| 4 | T15 `.business_metrics` | metric C/E model | `segment_kind,segment_value,metric` | `event_rate,predicted_positive_rate,exposure_sum,observed_loss_sum,expected_loss_sum`; `value` | `status,reason` | `n_rows,n_evaluable_rows,n_observed_loss_mature_rows,unit`; result observed-loss as-of |
| 5 | T15 `.folds` | fact D/E/X model | `fold_id` | explicit fold/window/maturity cells only | — | validation/train/evaluable position tuples and cutoff/window/as-of |
| 6 | T16 `DataAuditResult.dataset_profile` | fact D/E only | `side`; `current/reference` | exact Task16 §7.1 count/rate/feature fields | selected feature/duplicate-row/duplicate-index status-reason group | side; no source time |
| 7 | T16 `.column_profile` | fact D/E only | `side,column` | exact Task16 §7.2 missing/count/rate/profile/flag fields | selected missing/value-profile status-reason group | column position/n rows/counts; no source time |
| 8 | T16 `.numeric_profile` | fact D/E only | `side,column` | exact Task16 §7.3 count/finite/location/dispersion/range/quantile fields | selected count/finite/location/dispersion/range/quantile status-reason group | n/counts; no source time |
| 9 | T16 `.categorical_profile` | fact D/E only | `side,column` | exact Task16 §7.4 count/cardinality/rate/frequency/concentration/comparison fields | selected matching status-reason group | non-missing counts; no source time |
| 10 | T16 `.missingness_drift` | metric D/E only | `column` | `reference_missing_rate,current_missing_rate,absolute_rate_change,relative_rate_change` | selected matching status/reason group | reference/current n/counts; no source time |
| 11 | T16 `.point_in_time_profile` | metric D/E only | `side,scope,column`; only feature/audit_input rows with non-null column and the unique dataset row are supported | `evaluated_count,violation_count,not_verifiable_count` | `status,reason` | chronology rows unsupported because owner exposes no collision-free public locator; no source time |
| 12 | T16 `.slice_profile` | metric D/E only | `side,slice_role,row_kind,slice_ordinal` | `row_count,target_non_missing_count,target_non_missing_rate,positive_count,event_rate` | selected size/target/event/quality status-reason group | partition/fold ordinals; no source time |
| 13 | T16 `.resource_usage` | fact D/E only | `side,resource` | `requested,available,actual,truncated` | `status,reason` | — |
| 14 | T16 `.findings` | finding D/E only | exact non-null `finding_key` | exact finding fields `value,threshold,count,denominator,affected_rate` | `status,reason` | owner canonical SHA-256 identity; no source time |
| 15 | T16 `.provenance` | provenance D/E/X only | `provenance_key` | `field_key` selects exactly one of `numeric_value,text_value,count_value,boolean_value` compatible with `value_type` | `status,reason` | — |
| 16 | T17 `DecisionStrategyResult.row_decisions` | fact E strategy | `row_position` | decision/base/final/applied-rule/count/override/mapping fields | `decision_status,decision_reason` | row-local only; evaluation_time |
| 17 | T17 `.rule_evaluations` | rule E strategy | `row_position,rule_key` | phase/priority/order/path/truth/applied/overlap/conflict | `status,reason` | row-local only; evaluation_time |
| 18 | T17 `.rule_summary` | metric C/E strategy | `scope_type,scope_column,scope_ordinal,time_slice_ordinal,phase,rule_key,metric_key` | exact §12.3 inventory; `metric_value` | `status,reason` | `numerator,denominator,support_n_rows,unit`; provenance evaluation_time |
| 19 | T17 `.action_summary` | metric C/E strategy | `scope_type,scope_column,scope_ordinal,time_slice_ordinal,action_key,metric_key` | exact §12.4 inventory; `metric_value` | `status,reason` | `numerator,denominator,support_n_rows,unit`; evaluation_time |
| 20 | T17 `.business_summary` | metric C/E strategy | `scope_type,scope_column,scope_ordinal,time_slice_ordinal,action_key,action_role,metric_key` | exact §12.5 inventory; `metric_value` | `status,reason` | `numerator,denominator,support_n_rows,unit`; evaluation_time |
| 21 | T17 `.constraint_summary` | fact D/E strategy | `constraint_key,metric` | `actual_value,gap,violation_magnitude` | `status,reason` | `support_n`; evaluation_time |
| 22 | T17 `.historical_transitions` | transition D/E strategy | `historical_action_name,simulated_action_name` | `row_count,row_rate` | `status,reason` | input rows denominator; evaluation_time |
| 23 | T17 `.provenance` | provenance D/E/X strategy | `provenance_key` | `field_key="provenance_value"`; exact cell | `status,reason` | `evaluation_time` row is source time |
| 24 | T18 `LifecycleMonitoringResult.monitoring_summary` | metric C/E warning_scenario | `scenario_key,scope_key,scope_position,rule_key,metric` | exact §11 M metric inventory; `metric_value` | `status,reason` | `numerator,denominator,support_n,support_unit,mature_n,censored_n,unit`; provenance analysis_as_of |
| 25 | T18 `.scenario_comparison` | comparison D/E warning_scenario | `reference_scenario_key,comparator_scenario_key,metric,scope_key,scope_position,rule_key` | exact §11 M metric subset; `reference_value,comparator_value,delta` | `status,reason` | `numerator,denominator,support_n,support_unit`; analysis_as_of |
| 26 | T18 `.lifecycle_summary` | metric D/E only | `scope_key,scope_position,from_state_key,to_state_key,metric` | exact §11 L metric inventory; `metric_value` | `status,reason` | `numerator,denominator,support_n,support_unit,unit`; analysis_as_of |
| 27 | T18 `.rule_evaluations` | rule E warning_scenario | `row_position,scenario_key,rule_key` | path/truth/streak/episode/notification facts | `status,reason` | observation time/episode/maturity fields |
| 28 | T18 `.alert_episodes` | lifecycle fact E warning_scenario | `entity_position,scenario_key,rule_key,episode_ordinal` | alert/duration/count/reopen/unresolved facts | `status,reason` | episode timestamps/ordinals; never cross-owner join |
| 29 | T18 `.state_history` | lifecycle fact E only | `row_position` | candidate/effective state/rank/count facts | `status,reason` | observation time; never cross-owner join |
| 30 | T18 `.state_transitions` | lifecycle fact E only | `from_row_position,to_row_position` | state/rank/kind/direction/allowed/consecutive/cure/loss facts | `status,reason` | transition time; never cross-owner join |
| 31 | T18 `.provenance` | provenance D/E/X warning_scenario | `provenance_key` | `field_key="provenance_value"`; exact cell | `status,reason` | `analysis_as_of` row is source time |
| 32 | T16 `.target_profile` | fact D/E only | `side,class_position` | exact Task16 §7.5 class/count/rate/declaration cells only | selected class/binary/balance/positive-class status-reason group | `target_non_missing_n`; no raw label/source time |
| 33 | T16 `.missingness_patterns` | metric D/E only | exact `pattern_key` | exact pattern count/rate/missing-cell/reference/change cells | selected count/rate/reference/comparison status-reason group | `row_count,reference_row_count`; no source time |
| 34 | T16 `.schema_drift` | fact D/E only | exact `column` | exact Task16 §7.9 positions/dtypes/logical types/roles/change flags/`primary_change` | `status,reason` | column-local; no source time |
| 35 | T16 `.collinearity` | metric D/E only | `left_column,right_column` | `valid_n,pearson_r,absolute_r,threshold` | `status,reason` | complete-case `valid_n`; no source time |
| 36 | T18 `.observation_history` | lifecycle fact E warning_scenario | `row_position` | consecutive/period/cohort/primary alert/count/maturity/event/effective-state facts | observation and state status/reason groups | observation time; row-local only |
| 37 | T18 `.notifications` | lifecycle fact E warning_scenario | `entity_position,scenario_key,rule_key,episode_ordinal,notification_ordinal` | notification kind/repeated/matched-event/count/maturity facts | `status,reason` | notification time/row position; never cross-owner join |
| 38 | T18 `.event_matches` | lifecycle fact E warning_scenario | `scenario_key,entity_position,event_ordinal` | event/match/capture/notification/lead-time/count facts | `status,reason` | event and notification times; never cross-owner join |

For Task17 §§12.3--12.5 and Task18 §11, “exact inventory” is a normative closed reference to the explicitly
enumerated metric keys in those approved sections and admits no later/private/unknown values. For Task16 entries
6--15 and 32--35, the selectable cells/status groups are the exact columns enumerated in Task16 §§7.1--7.14;
for all fact/transition/rule/comparison entries, `field_key` must select exactly one non-locator public cell from
that entry's named approved schema, and no private/later/derived cell is admitted. Task17 historical
transitions is atransition fact, not metric-key source. Task16不存在且已从Task19删除
`finding_count,available_evidence_count,missing_rate_delta,missing_rate_relative_delta`；真实字段为上表所列。

Task15没有public owner fingerprint，authoritative fingerprint状态为`unavailable`且expected必须为`None`；
该owner只以result position绑定，绝不由Task19伪造digest。Task16 authoritative result digest是
`DataAuditResult.config_fingerprint`，但只证明sanitized config shape；Task17是
`DecisionStrategyResult.strategy_fingerprint`；Task18是
`LifecycleMonitoringResult.monitoring_fingerprint`。Task17 candidate identity额外包含`strategy_key`，
Task18 candidate identity额外包含`scenario_key`；这些typed identity只进入source inventory payload，
不得字符串拼接后冒充owner fingerprint。Expected fingerprint必须从owner result读取并exact比较；caller
字符串不能使source verified。所有structured declarations按第10节产生semantic fingerprint。

### 7.3 Tagged locator与finding-key规则

Ref locator严格使用上表columns；`field_key/metric_key/scope_key/scope_column/scope_position/time_position/
row_position/entity_position/scenario_key/rule_key/state_key/from_state_key/to_state_key/statistic_key/
category_key/category_position/pattern_key/episode_ordinal/notification_ordinal/event_ordinal/numeric_value/
finding_key/provenance_key`只在对应registry
row允许时非null。Tagged映射也完全冻结：owner
non-metric selected cell映射`field_key`，owner
`metric`映射`metric_key`，`scope/scope_type/segment_kind`映射`scope_key`，`scope_ordinal/slice_ordinal`
映射`scope_position`，`time_slice_ordinal`映射`time_position`，`statistic`映射`statistic_key`，
`class_position`映射`category_position`，`pattern_key`原样映射，`resource/threshold_kind/
historical_action_name`映射`category_key`，`simulated_action_name`映射`action_key`，而owner
`requested_fraction/threshold/finite segment_value`映射`numeric_value`。Task16 `column`映射
`column_key`；`left_column/right_column`分别映射`column_key/scope_column`。未列出的重载禁止。
对于gains/threshold及Task16 wide tables中没有long-form `metric` column的closed value groups，
`metric_key`选择上表明确列出的exact value/status/reason group；它不是row locator，且不得选择任意numeric
column。`field_key`同样只选择该registry entry明确批准的单一fact cell。
Task16 `findings`唯一允许把exact non-null owner canonical SHA-256 `finding_key`作为sole locator；该
table内必须唯一，否则owner table invalid。其他detail-table `finding_key`可能nullable/shared，不作sole
locator。Tasks17/18 finding keys是privacy-safe linkage但可能按
domain unit共享，因此registry使用closed compound locator，绝不假设finding_key全表唯一。Task15完全
使用compound locator。Float locator只用于owner-approved threshold/requested-fraction exact finite value，
按owner canonical float equality；不接受近似匹配。
`GovernancePolicy.evidence_refs`只承载governance-wide owner refs：exact `source_use="diagnostic"`，或
governance-wide lineage的exact `source_use="explanation"`，且`candidate_key=None`；candidate comparison refs只在
对应`GovernanceCandidate.evidence_refs`，其
`candidate_key`必须等于container candidate。Structured declaration-owned refs和explanation refs分别由其
具名declaration唯一承载，`candidate_key`必须等于该declaration candidate；它们只链接candidate已经绑定的
authoritative owner result/context，不要求也不得在`GovernanceCandidate.evidence_refs`复制同一个ref。
Legal structured-to-candidate linkage不是duplicate declaration。`duplicate_evidence_ref`仅适用于standalone ref
carriers `GovernancePolicy.evidence_refs`/`GovernanceCandidate.evidence_refs`内的exact或semantic duplicate；
embedded refs不得按ref fields或resolved owner row做global dedupe，其duplicate由enclosing declaration identity判定，
分别映射`invalid_explanation/invalid_attribution/invalid_prediction_profile/invalid_performance_evidence`。不同合法
enclosing identities解析同一owner row/context是合法共享。
Task16 ref在
governance-wide diagnostic时`candidate_key=None`；作为某candidate的contextual explanation时
必须匹配existing candidate key，但这种关联不证明candidate与audit来自same snapshot。

Carrier ownership matrix是normative且closed：

| semantic use | unique carrier | candidate binding | duplicate rule |
|---|---|---|---|
| governance-wide `diagnostic`；或governance-wide lineage with `source_use="explanation"` | `GovernancePolicy.evidence_refs` | `candidate_key=None` | 同一policy tuple中exact/semantic ref重复为`duplicate_evidence_ref` |
| candidate `comparison_criterion` | `GovernanceCandidate.evidence_refs` | ref candidate必须等于container candidate并解析到该candidate authoritative source | 每侧criterion semantic identity恰一ref；零条或多条按§4.1 fail closed |
| candidate `diagnostic` context | `GovernanceCandidate.evidence_refs` | ref candidate必须等于container candidate | 同一candidate tuple中exact ref fields重复为`duplicate_evidence_ref` |
| explanation trace `explanation` | `GovernanceExplanation.source_ref` | ref candidate必须等于explanation candidate | 每个explanation declaration只承载其一个ref；不同合法explanations可解析同一owner row且不是duplicate |
| `attribution_context` | `GovernanceAttributionEvidence.source_ref` | candidate、source result position与ref三者必须解析到该model candidate绑定的Task15 result/context | attribution declaration是唯一carrier；不得在candidate refs复制；重复attribution identity为`invalid_attribution` |
| `drift_context` | `GovernancePredictionProfile.source_ref` | candidate与ref必须解析到该model candidate绑定的Task15 result/context | profile declaration是唯一carrier；reference/current共享candidate owner context合法；重复/grouping invalid为`invalid_prediction_profile` |
| `stability_context` | `GovernancePerformanceEvidence.source_ref` | candidate与ref必须解析到该model candidate绑定的Task15 result/context | performance declaration是唯一carrier；reference/current共享candidate owner context合法；重复/grouping invalid为`invalid_performance_evidence` |

上表不新增`source_use` vocabulary。Candidate refs不得承载`explanation/attribution_context/drift_context/
stability_context`，structured/explanation refs不得复制到candidate refs。Embedded ref occurrence是否duplicate由
enclosing semantic declaration identity判定，而不是由resolved owner row是否相同判定；candidate source binding
本身是legal cross-link而非第二个ref carrier，每个合法occurrence仍取得独立`source_ref_position`。Carrier-required
`source_use`不匹配使用`invalid_evidence_ref`；ref use合法但task/table/use不在38-entry registry使用
`unsupported_source`；wrong candidate、wrong result position或跨candidate source
binding使用`invalid_source_binding`，均不物化unavailable row。`source_key`、pandas index和object identity仍不存在
且不得作为carrier/link identity。

The normative duplicate representative is two separately declared, structurally valid, semantic-identical
Task16 governance-wide diagnostic refs in `GovernancePolicy.evidence_refs`, with all earlier policy, owner,
locator, and fingerprint checks valid. The first failure is `model governance: duplicate_evidence_ref`.

### 7.3.1 Exhaustive owner-validation error partition

Owner validation uses this closed partition; no generic `invalid_owner_result`, unexpected-owner-result, or other
catch-all exists:

| first invalid predicate | exact error key |
|---|---|
| wrong owner collection/container type or wrong exact owner tuple element result type | `invalid_owner_result_container` |
| required owner table/schema structure or column set/order mismatch | `invalid_owner_schema` |
| schema is correct but a frozen owner column dtype is wrong | `invalid_owner_dtype` |
| owner row status token is outside that owner table's closed vocabulary | `invalid_owner_status` |
| owner reason token, or status/reason combination, violates that owner contract matrix | `invalid_owner_reason` |
| owner cell/value domain violates the selected owner table's frozen value rule | `invalid_owner_value` |
| duplicate semantic owner source declaration | `duplicate_owner_source` |
| candidate/source binding, locator position, or candidate identity does not match | `invalid_source_binding` |
| fingerprint field is not the approved lowercase 64-character SHA-256 hexadecimal form | `invalid_source_fingerprint` |
| fingerprint has the approved form but differs from the authoritative digest | `source_fingerprint_mismatch` |

These predicates are checked in the listed owner-phase order after structural validation and before time,
resource, math, or materialization phases. The six currently missing production branches that remain required
after this amendment are `invalid_pair`, `duplicate_evidence_ref`, `invalid_owner_dtype`, `invalid_owner_status`,
`invalid_owner_reason`, and `invalid_source_fingerprint`; each requires a direct executable representative.
For the dtype representative, Task17 `rule_summary.metric_value` is the frozen `Float64` field: with the
approved column set/order intact but that field materialized with an incompatible dtype, the first error is
`invalid_owner_dtype`, not `invalid_owner_schema` or `invalid_owner_value`. For status/reason representatives,
an otherwise valid owner row with a token outside its closed vocabulary (for example, test value `bogus`) is
`invalid_owner_status`, while a legal status with an unsupported reason or forbidden status/reason combination is
`invalid_owner_reason`; `bogus` is a representative test value, not vocabulary.

### 7.4 Explanation/structured-source binding matrix

| method/source | allowed registry/source | binding/value/relation/privacy |
|---|---|---|
| `rule_trace` | T17 rule_evaluations/rule_summary or T18 rule_evaluations | candidate-bound rule facts/metric；no condition literal/raw row value |
| `reason_trace` | T17/T18 entries with status/reason | owner status/reason unchanged；no free text |
| `source_lineage` | provenance entries 15/23/31 or T15 fold/result facts | candidate-bound provenance value；no raw data identity |
| `metric_evidence` | metric entries 1--4,10--12,18--20,24,26,32--35 | exact registry value/support/status |
| `scenario_delta_trace` | entry25 only | owner reference/comparator/delta unchanged |
| `state_transition_trace` | entry30 only | approved ordinal/state-key facts；no entity join |
| `reason_trace` lifecycle extension | T18 entries 36--38 | owner observation/notification/event status/reason and privacy-safe facts only |
| three model-attribution methods | 每个`GovernanceExplanation.source_ref`是该trace自身、use=`explanation`的唯一carrier，并按candidate/method/feature唯一链接matching attribution；matching `GovernanceAttributionEvidence.source_ref`是structured evidence自身、use=`attribution_context`的唯一carrier并指向该candidate的T15 entry1或5；candidate refs不复制 | 两者是不同semantic declarations，不要求ref fields相同、不互相复制；即使解析同一owner row也不是duplicate；exact numeric value/relation/feature schema key/support；not owner-verified attribution |

`GovernanceExplanation` is exclusively a caller trace declaration in`GovernancePolicy.explanations`; each valid
declaration produces one normalized `explanations` row. It never doubles as an output result object or computes
attribution. Every explanation candidate/ref/method combination must match this matrix.

Attribution identity is`(candidate_key,method,feature_key,evaluation_scope,metric_key)` and must be unique.
`feature_key` is a validated public schema identifier, not a cell value. Coefficient/native refs establish the
specific candidate's T15 context but do not prove model-attribute origin; permutation ref must point to that
candidate's T15 baseline metric/fold context and preserve metric/scope/support/repeats/seed, yet remains
`source_snapshot_status=unverified` because Task15 cannot prove caller permutation used the same raw matrix.
Sanitized attribution allowlist is exactly candidate/family/method/feature key/metric/value/relation/scope/support/
uncertainty/repeats/seed/source task/table/locator/fingerprint/status/reason/finding key；rawfeature value、row、
target、model/attribute object、training record、repr和class/intercept payload禁止。

Prediction profiles and performance evidence bind to exactly onemodel candidate and its T15
`source_result_position`; their`candidate_key` and`source_ref.candidate_key` must match, and their embedded source refs
are respectively the unique `drift_context`/`stability_context` carriers，不在candidate refs复制。No profile/evidence
can be reused across candidate keys. GovernanceMetadata
is explicitly governance-wide when`metadata_scope="governance",candidate_key=None` and candidate-specific when
`metadata_scope="candidate"` with an existing candidate key; metadata keys are unique within scope/candidate.

### 7.5 Snapshot and entity-alignment proof

Vocabulary remains`verified,unverified,not_applicable`:

- `verified` is reachable only for relationships between rows/tables resolved from the same owner tuple and same
  `source_result_position`; the owner result object itself proves one invocation/result boundary. For Task18
  scenario rows within oneresult, same monitoring fingerprint plus same position is intra-result verified.
- Different source positions—even equal public values/deep copies—are different declarations and snapshot
  `unverified`. Cross-task/cross-owner relationships are always snapshot`unverified`; current Tasks15--18 expose
  no shared raw-snapshot token. Equal as-of, row count, shape, candidate key, scope/entity position, or digest does
  not upgrade this status.
- `not_applicable` is only for a single-source/aggregate operation that does not claim two sources share a raw
  snapshot. A requested cross-source comparison without proof is`unverified`, never not-applicable.

Task16 `config_fingerprint` is not a raw-data hash and cannot prove snapshot or alignment. Task17/18 positions are
local ordinals; equal`entity_position`/`row_position`across results or owners never proves same entity. Task19
forbids all cross-owner entity-level joins and accepts only owner aggregate/common-support evidence. Entity
alignment is`verified` only within oneowner result where the owner contract itself defines aligned rows;
`not_applicable` for aggregate operations needing noentity match；otherwise`unverified`. Raw entity/customer/
account identifiers and pandas index are never accepted as repair mechanisms.

Public result/provenance records source task、result position、table、privacy-safe locator、authoritative
fingerprint availability/value and proof basis`same_owner_result/cross_result_unverified/alignment_not_required`.
`source_ref_position`是所有refs按policy-global → candidate declaration/evidence order → explanations declaration order →
model attribution → prediction profile → performance evidence flatten后的0-based audit ordinal；它只链接
该result的provenance locator rows，不是owner row locator或persistent ID。
FIX-C第8节冻结recommendation consequences与normalized support identity；FIX-D第9/9A/11节已冻结
source-time gate、canonical encoder/fingerprints/provenance和exact error/status/reason/finding identities。

## 8. Authoritative direction、normalized support 与 recommendation matrix

### 8.1 Governance metric direction registry（92 exact metric keys）

`GovernanceCriterion.direction`保留为caller assertion以最小化public API变化，但必须exact等于下表
authoritative direction；不匹配即invalid criterion。`criterion_role="diagnostic"`必须
`direction="not_directional"`、`required_for_promotion=false`；它可进入comparison/summary但outcome固定
`not_directional`，不计入任何recommendation count。`criterion_role="decision"`只允许registry中
directional rows。Attribution、permutation importance、Task16 facts/drift、Tasks17/18 non-metric facts、
Task18 `scenario_comparison`本身与governance metadata全部diagnostic/explanation only，绝不因数值大小
成为promotion evidence。

| task/table/family | higher_is_better | lower_is_better | target_range |
|---|---|---|---|
| T15 `metrics` / model | `roc_auc,average_precision,normalized_gini,ks_statistic` | `brier_score,log_loss,expected_calibration_error` | none |
| T15 `gains` / model | `event_rate,capture,lift` | none | none |
| T15 `threshold_analysis` / model | `sensitivity,specificity,precision,negative_predictive_value,f1,accuracy` | none | `predicted_positive_rate` only |
| T15 `business_metrics` / model | `event_rate` | `observed_loss_sum,expected_loss_sum` | `predicted_positive_rate,exposure_sum` only; `segment_kind="all"` only |
| T17 `rule_summary` / strategy | `captured_event_count,target_capture_rate` | `unknown_count,unknown_rate,not_evaluated_count,overlap_count,overlap_rate,conflict_count` | `hit_count,hit_rate,applied_count,sole_hit_count,incremental_action_count,leave_one_out_changed_action_count` only; all `scope_type="overall"` |
| T17 `action_summary` / strategy | `assumed_action_value_sum,assumption_based_payoff_sum` | `event_count,event_rate,expected_loss_sum,assumption_based_observed_event_loss_sum,assumed_action_cost_sum` | `action_count,action_rate,exposure_sum` only; all `scope_type="overall"` |
| T17 `business_summary` / strategy | `decided_rate,assumed_action_value_sum,assumption_based_payoff_sum,historical_mapped_rate` | `observed_event_count,observed_event_rate,unknown_action_rate,expected_loss_sum,expected_loss_rate,assumption_based_observed_event_loss_sum,actual_observed_loss_sum,actual_observed_loss_rate,assumed_action_cost_sum,selected_event_rate` | `selected_rate,rejected_rate,review_capacity_rate,exposure_sum` only; all `scope_type="overall"` |
| T18 `monitoring_summary` / warning_scenario | `resolved_episode_count,captured_event_count,event_recall,notification_precision,lead_time_mean,lead_time_median,warning_to_event_rate` | `warning_hit_count,warning_observation_rate,warned_entity_count,warned_entity_rate,persistent_warning_count,persistent_warning_rate,notification_count,notifications_per_entity,overlap_count,conflict_count,episode_count,open_episode_count,episode_duration_mean,episode_duration_median,false_alert_share,false_positive_rate,expected_loss_sum,expected_loss_rate,observed_loss_sum,observed_loss_rate` | `exposure_sum` only |

Exact count为92：T15 22、T17 42、T18 28。Task17 `evaluated_count/evaluable_event_count/row_count`、
ranking-score/event-probability location metrics及non-overall generated
`<rate_metric>_overall_baseline/<rate_metric>_absolute_delta`、Task17 non-overall scopes、Task18 lifecycle-only metrics及任何registry外
key均diagnostic-only，不能伪装成decision criterion。`target_range`只允许上表17个明确列出的metric；
这些metric不得声明higher/lower。Target bounds必须是exact finite non-bool real scalars，且
`target_low <= target_high`；区间是closed `[low,high]`。Non-target-range bounds必须均为`None`。

### 8.2 Exact numeric comparison

Comparison outcome vocabulary严格为：`challenger_better,champion_better,tie,not_comparable,
not_directional`。所有available numeric source values必须finite，否则是invalid owner source；NA、
unavailable、undefined、not-verifiable不进入numeric math。统一raw
`delta = challenger_value - champion_value`，不随direction反转。

- higher：challenger `>` champion为`challenger_better`，`<`为`champion_better`，`==`为`tie`；
- lower：challenger `<` champion为`challenger_better`，`>`为`champion_better`，`==`为`tie`；
- target range：`d(x)=0` when `low<=x<=high`，`low-x` when `x<low`，`x-high` when `x>high`；
  challenger distance更小/更大/相等分别为challenger_better/champion_better/tie。两侧均在range为tie；
  一侧在range则该侧better；两侧低于或高于range时距相应nearest boundary更近者better；一低一高按各自
  nearest-boundary distance，等distance为tie。没有center preference，也不新增distance output。

Tie只接受exact numeric equality或exact distance equality；禁止epsilon、`isclose`、rounding或display
precision tolerance。Support/status proof在numeric equality之前；相同value但support不匹配仍是
`not_comparable`，绝不是tie。

### 8.3 Normalized support identity

每侧先构造`(source task/table/family, metric identity, normalized scope identity, support unit,
support population identity, mode/fold, time/window, maturity/evaluability basis, candidate owner context)`；
只有两侧owner status允许比较、identity逐项相等、各自support达到criterion `minimum_support`且下列
owner-specific proof成立才comparable。`minimum_support`是exact non-bool int `[1,200000]`。禁止
intersection repair、resegmentation、partial sum或用equal as-of替代support proof。

- **Task15**：identity含result `validation_mode,prediction_scope,score_source`，table metric/statistic，
  `scope,fold_id`，gains requested fraction或threshold kind/value或business segment kind/value，
  `n_rows/n_positive/n_negative`或table exact denominator counts，及
  `observed_loss_maturity_mode,observed_loss_analysis_as_of,n_observed_loss_mature_rows`（仅loss metrics）。
  本节全部Task15 decision metrics都是aggregate comparisons，不声明same raw snapshot/entity relationship，
  因而snapshot/alignment固定not_applicable；support identity证明的是相同evaluation definition/count context，
  不是相同raw rows。Fold、mode、evaluation population、
  threshold/segment、denominator或maturity不同即not_comparable。
- **Task17**：identity含table、metric、`scope_type,scope_column,scope_ordinal,time_slice_ordinal`，以及
  rule的`phase,rule_key`、action/business的`action_key,action_role`，`unit,numerator,denominator,
  support_n_rows`和provenance `evaluation_time`。Only exact same anonymous scope/config ordinal within
  independently bound strategies is comparable。Overall rows是aggregate、snapshot/alignment not_applicable；
  segment/time/segment_time anonymous ordinals在不同strategy results间没有shared identity token，因此即使
  ordinal相同也not_comparable。Non-metric facts不进入numeric comparison。
- **Task18**：Task19不建立第二套scenario alignment；优先消费owner `scenario_comparison`的R/C pair，
  exact `(metric,scope_key,scope_position,rule_key)`、reference/comparator values、numerator、denominator、
  `support_n,support_unit,status,reason`。也可从同一monitoring result内两scenario source refs重建相同lookup，
  但必须逐字遵守AM-04九scope normalization：scenario/scenario_rule丢弃raw scenario ordinal，其余七scope
  保留且要求相同subordinate ordinal；四个ineligible scopes不得比较。Owner
  `not_verifiable/support_not_comparable`或support_n不同直接not_comparable，不重算delta。同一Task18 result
  内AM-04 pair proof为verified；不同Task18 result没有shared scenario/scope token，固定not_comparable。

Maturity/evaluability不是散文附注：它是上述identity的required component；owner缺少可证明字段、status
不允许或basis不同均`not_comparable`。Segment/cohort/vintage/state/transition/time只在owner明确冻结跨两侧
normalized identity时可比；局部position相等本身不够。需要same snapshot/entity alignment的criterion若
其status为unverified，outcome必须not_comparable；aggregate registry row若无需raw matching，可使用
not_applicable且不因此阻塞。Semantic reasons严格使用第11.3节15-key registry；本路径只允许
`support_not_comparable,source_unavailable,source_not_verifiable,source_undefined,snapshot_unverified,
alignment_unverified,time_unverified,maturity_not_comparable`。

Comparison semantic precedence固定：valid ref → owner status → family → metric/direction registry → authoritative
time → required snapshot/alignment proof → normalized support → finite values → raw delta → direction/range → outcome。

### 8.4 Criterion roles and policy gates

`criterion_role`只允许`decision/diagnostic`；`required_for_promotion`必须是exact built-in bool。
Diagnostic永不计数。有challenger/pair时decision criteria数量必须至少1，且
`GovernancePolicy.minimum_comparable_criteria`是exact non-bool int、
`1 <= value <= declared decision criterion count`。Zero challenger时criteria可empty/diagnostic-only，
`minimum_comparable_criteria`仍必须exact 1但不应用于任何pair。只有outcome为
challenger_better/champion_better/tie的decision rows计comparable；not_comparable/not_directional不计。

Required decision criterion的not-comparable/unavailable/undefined/not-verifiable/support/maturity/snapshot/
alignment failure一律使pair`insufficient_evidence`。Non-required decision criterion若不可比，不单独触发
insufficient且不计数；但其可比`champion_better`仍阻止promotion并参与mixed/retain，其可比
`challenger_better`同样参与。Priority只用于ordering/display，不覆盖其他criterion、不形成veto/weight。

### 8.5 Complete recommendation decision matrix

Recommendation vocabulary保持exact五项：`promote_challenger,retain_champion,continue_evaluation,
insufficient_evidence,reject_challenger`。每pair恰一row；无criterion-level recommendation。

| precedence / decision evidence | recommendation | semantic reason |
|---|---|---|
| challenger state `rejected/retired` | `reject_challenger` | closed hard state veto |
| any required decision criterion not comparable, or comparable count below minimum | `insufficient_evidence` | required evidence/support/snapshot/alignment/maturity incomplete |
| at least one challenger_better and at least one champion_better | `continue_evaluation` | mixed evidence |
| all comparable outcomes tie | `continue_evaluation` | criteria tied |
| at least one challenger_better, zero champion_better, remainder tie; all required complete and minimum met | `promote_challenger` | challenger favorable |
| at least one champion_better, zero challenger_better, remainder tie; all required complete and minimum met | `retain_champion` | champion favorable |

Optional unavailable evidence does not change a row already resolved by comparable decision evidence; with no
directional better outcome and insufficient comparable count it still maps toinsufficient. There is no majority
vote: any better+worse is mixed regardless of counts. All five recommendations are reachable respectively by
better(+ties)、worse(+ties)、mixed or all-ties、required/minimum failure、rejected/retired challenger。

Promotion requires all: valid pair；challenger state candidate/under_review/approved；no closed hard veto；
minimum comparable count met；every required criterion comparable；zero champion_better；at least one
challenger_better；all remaining comparable decision outcomes tie；`human_review_required=true`。Partial required
unavailable/support mismatch/snapshot or alignment unverified/maturity mismatch always blocks promotion。

`human_review_mode`只允许`promotion_only/all_recommendations`。Recommendations新增non-null boolean
`human_review_required`：promotion在两种mode下始终true；promotion_only下其他四项false；
all_recommendations下五项全true。它只表示必须交给外部人工治理流程复核，不表示review已执行/完成、
approval、deployment或state mutation。Task19不接收review callback/result，不自动promote/approve/deploy。
Metadata仅记录purpose/owner/materiality/assumptions/limitations/issue/remediation；不转numeric score、不触发
hard veto。No weights、composite score、optimizer或majority vote。

Recommendation semantic precedence固定：pair/config → challenger hard state veto → collect decision outcomes →
required completeness → minimum count → better/worse/tie pattern → exact recommendation → human-review boolean →
explanation/materialization。Exact public error/reason precedence见第11节。

## 9. Authoritative evidence time、snapshot、alignment 与 privacy

### 9.1 Governance clock and exact datetime domain

`GovernancePolicy.analysis_as_of`、`GovernanceAttributionEvidence.evidence_as_of`、
`GovernancePredictionProfile.analysis_as_of`、`GovernancePerformanceEvidence.window_start/window_end/
evidence_as_of`只接受`type(value) is datetime.datetime`或`type(value) is pandas.Timestamp`。拒绝
`datetime.date`、string、NumPy datetime、`NaT`、datetime/Timestamp subclass、missing和任何自动解析。
同一次调用的governance clock、所有owner authoritative times及structured times必须全naive或全aware；
混合即`datetime_awareness_mismatch`。Naive保持naive并按原wall-clock比较；aware在比较、canonicalization
和输出前转UTC，不读取本机timezone。相等合法；所有source/evidence time必须`<= analysis_as_of`。

### 9.2 Closed authoritative source-time registry（38 entries）

本表逐条覆盖第7.2节38-entry source registry，不存在caller-declared time fallback：

| registry entries | authoritative time | verification and consequence |
|---|---|---|
| T15 1--3 | time mode时，`.folds.analysis_as_of`在所有referenced fold/overall contributing folds中必须exact、non-null、唯一相同；non-time mode无absolute evidence time | time mode为`verified`；non-time decision criterion为`not_verifiable/time_unverified`，diagnostic可保留且time=`unverified` |
| T15 4 | `observed_loss_sum`使用result `observed_loss_analysis_as_of`；其余business metric在time mode使用folds as-of，否则无absolute time | required cell缺失/不一致invalid；非-time非observed-loss decision evidence为`not_verifiable/time_unverified` |
| T15 5 | time mode使用该fold row `analysis_as_of`；non-time fold fact无time claim | time mode`verified`；non-time diagnostic `not_applicable` |
| T16 6--15,32--35 | none；`DataAuditResult.config_fingerprint`和sanitized provenance不是result-level absolute evidence time | 永不升级；只允许diagnostic/explanation，time=`unverified`，不得成为decision evidence |
| T17 16--23 | `DecisionStrategyResult.provenance`唯一`provenance_key="evaluation_time"`的`provenance_value`；只接受Task17 exact ISO-8601 datetime serialization并按本节awareness规则解析 | result-wide `verified`；missing/duplicate/noncanonical invalid |
| T18 24--31,36--38 | `LifecycleMonitoringResult.provenance`唯一`provenance_key="analysis_as_of"`的`provenance_value`；只接受Task18 canonical-JSON datetime fragment并按本节awareness规则解析 | result-wide `verified`；missing/duplicate/noncanonical invalid；referenced row-local observation/event/notification/transition time还必须不晚于该owner as-of |

Task15 cutoff、validation window或row position不能替代absolute as-of；Task16 role-level string metadata不能替代
result time。Task17/18 result-wide future time使整个对应owner result invalid，不能过滤后继续。Whole-input safety
validation也检查未被ref选择的owner/structured declarations：Task15所有fold as-of/observed-loss as-of、Task17
evaluation_time、Task18所有获准timestamp columns与analysis_as_of均须自洽且不晚于governance clock。向owner
result加入governance as-of之后的evidence不得改变既有较早governance result，而是使该调用在materialization前
fail closed。

### 9.3 Structured evidence time

- Attribution自己的`evidence_as_of`必须不晚于governance as-of；其Task15 context time另外按上表判断。
- Prediction profile pair必须`reference.analysis_as_of <= current.analysis_as_of <= governance analysis_as_of`；
  两侧awareness一致且source refs分别通过registry gate。
- Performance pair必须各自`window_start < window_end <= evidence_as_of <= governance analysis_as_of`，且
  `reference.evidence_as_of <= current.evidence_as_of`；既有non-overlap window规则继续适用。
- GovernanceMetadata是声明性governance fact，time固定`not_applicable`，不得新增时间字段或推断当前时间。
- GovernanceExplanation继承其唯一source ref的time status；不接受第二个caller time。

`evidence_time_status`使用独立vocabulary `verified/unverified/not_applicable`，与
`source_snapshot_status`和`entity_alignment_status`正交；三者不得相互升级。Decision/comparison evidence的
time为unverified时，result row固定`not_verifiable/time_unverified`且不参与delta、direction、minimum count或
promotion。Diagnostic/explanation可保留原source semantic status，并单独记录time unverified；不得表述为
point-in-time verified。未来或required-time missing是invalid input并抛ValueError，不物化status row。
Result-level `evidence_time_status`聚合规则固定：至少一个retained owner/structured evidence为unverified则
`unverified`；否则至少一个为verified则`verified`；完全没有time-bearing evidence才`not_applicable`。

Future-evidence gate遵守第11.2节唯一global validation phase order：它位于valid exact scalar/container和owner
schema/fingerprint之后、snapshot/alignment与resource phase之前；equal as-of passes，future fails closed且没有
partial result。本节不定义第二套phase order。

### 9.4 Snapshot、alignment and privacy

source snapshot只有第7.5节的same-owner-result proof可为`verified`；cross-result/cross-owner即使owner
provenance、as-of、row count或fingerprint相等仍为`unverified`，单source且不要求matching才为
`not_applicable`。相同entity position、row count、pandas index不证明同一entity/snapshot。
owner_verified alignment没有owner proof时为`unverified`。禁止raw entity/index joins、segment/cohort/
vintage labels、raw feature values和arbitrary object repr。

candidate/source/metric/scope selectors在repr/str/hash/float/int/bool/comparison/iteration/pandas conversion前
验证；malicious scalars必须稳定ValueError。结果tables独立物化，输入、policy、上游tables不变。

### 9.5 FIX-D time direct-test obligations

Direct tests冻结：两种exact datetime通过；date/string/NumPy datetime、NaT和两个subclass拒绝；naive/aware
混合拒绝；aware UTC normalization；past/equal/future source；required time missing；Task15 time与non-time、
Task16 diagnostic-only、Task17 evaluation_time、Task18 analysis_as_of；attribution/profile/performance future；
metadata not-applicable；time/snapshot/alignment三维独立；future-evidence invariance及第11.2节single global-order
precedence。本节只引用该authority，不另列phase order。

## 9A. Canonical encoder、fingerprints 与 exact provenance

### 9A.1 Canonical semantic encoder

Encoder version固定`task19-canonical-json-v1`。Accepted domain仅为：exact `None`、exact built-in `bool`、
exact built-in `int`、finite exact built-in `float`、exact built-in `str`、上节两个exact datetime types、
exact `tuple`、exact `dict`（仅Task19 private payload）、以及exact九个Task19 input dataclass types
`GovernanceEvidenceRef/GovernanceCandidate/GovernanceCriterion/GovernanceExplanation/
GovernanceAttributionEvidence/GovernancePredictionProfile/GovernancePerformanceEvidence/GovernanceMetadata/
GovernancePolicy`。拒绝subclass、NumPy scalar、Decimal/Fraction、nonfinite、bytes、set/frozenset、list、任意
Mapping/Sequence、DataFrame/Series/array和其他object；没有repr/str/hash/bool/int/float/iter/eq/array fallback。

Canonical node是tagged JSON object：None=`{"t":"none"}`；bool=`{"t":"bool","v":true|false}`；
int value是无`+`/leading-zero的base-10 ASCII string；float value是exact lowercase `float.hex()` string（保留
signed zero）；str value是原Unicode scalar，不strip/casefold/Unicode-normalize；tuple=
`{"t":"tuple","v":[nodes...]}`并保留顺序；dict=
`{"t":"map","v":[[key,node]...]}`，key必须exact str、唯一，并按Unicode code-point升序。Caller mapping和
list不因内容合法而接受。Dataclass=`{"t":"dataclass","n":"<exact class name>","v":[[field,node]...]}`，
fields按frozen declaration order且全部observable fields进入；不得`asdict`后丢type。

Exact datetime node name区分`datetime`与`timestamp`。Naive datetime固定
`YYYY-MM-DDTHH:MM:SS.ffffff`，aware datetime先UTC后`YYYY-MM-DDTHH:MM:SS.ffffffZ`；Timestamp分别使用
九位fraction `YYYY-MM-DDTHH:MM:SS.fffffffff[Z]`，保留nanoseconds。Fold=0不省略fraction。JSON exact
options为`sort_keys=True,ensure_ascii=True,allow_nan=False,separators=(",", ":")`（实际separator无空格：
comma与colon），无newline，UTF-8 bytes。Fingerprint为`hashlib.sha256(bytes).hexdigest()`，必须lowercase
64-char hex。类型validation完成后只使用上述明确scalar algorithms；“unsupported”永不回退。

### 9A.2 Exact fingerprint inventory（11）

1. `policy_fingerprint`：exact GovernancePolicy dataclass全部字段及nested declarations；
2. `candidate_inventory_fingerprint`：candidate declaration order、每个candidate全部字段；
3. `criterion_inventory_fingerprint`：criterion declaration order、每个criterion全部字段；
4. `source_inventory_fingerprint`：四个owner tuple的exact type/position、flattened source_ref order/全部字段、
   owner authoritative fingerprint availability/value和candidate typed identity；
5. `explanation_inventory_fingerprint`：explanation declaration order/全部字段和resolved ref ordinal；
6. `attribution_inventory_fingerprint`：attribution order/全部字段和resolved ref ordinal；
7. `prediction_profile_inventory_fingerprint`：profile order/全部fields；
8. `performance_evidence_inventory_fingerprint`：performance declaration order/全部fields；
9. `metadata_inventory_fingerprint`：metadata declaration order/全部fields；
10. `structured_evidence_fingerprint`：fingerprints 5--9的ordered tuple；
11. `governance_fingerprint`：contract identity、package version、governance UTC/naive as-of、以下三个registry
    identities、fingerprints 1--10及四个ordered owner inventory identities。

Registry identities固定为`task19-source-registry-38-v1`、`task19-direction-registry-92-v1`、
`task19-recommendation-matrix-5-v1`。Task15 authoritative fingerprint永远`unavailable/None`；Task16只记录
owner `config_fingerprint`且不得声称raw snapshot；Task17记录owner `strategy_fingerprint`；Task18记录
owner `monitoring_fingerprint`。因此Task15 owner content变化不能由authoritative digest证明，Task16--18
digests也只证明各owner合同声明的范围；position/type/ref identities仍进入source inventory，不夸大保证。

### 9A.3 Exact provenance inventory（35 fixed rows）

`provenance_value`是上述privacy-safe canonical JSON node的完整JSON fragment，不是`str(value)`；不得输出
raw canonical policy/source payload。Rows严格按下列ordinal materialize，typed-empty仍保留schema：

```text
0 contract_version                 18 structured_evidence_fingerprint
1 package_version                  19 governance_fingerprint
2 canonical_encoder_version        20 source_snapshot_status
3 source_registry_identity         21 entity_alignment_status
4 direction_registry_identity      22 evidence_time_status
5 recommendation_policy_identity   23 candidate_count
6 analysis_as_of                   24 comparison_pair_count
7 time_model                       25 criterion_count
8 policy_fingerprint               26 risk_validation_result_count
9 candidate_inventory_fingerprint  27 data_audit_result_count
10 criterion_inventory_fingerprint 28 decision_strategy_result_count
11 source_inventory_fingerprint    29 lifecycle_monitoring_result_count
12 source_binding_inventory        30 task15_owner_fingerprint_statuses
13 explanation_inventory_fingerprint 31 task16_owner_fingerprints
14 attribution_inventory_fingerprint 32 task17_owner_fingerprints
15 prediction_profile_inventory_fingerprint 33 task18_owner_fingerprints
16 performance_evidence_inventory_fingerprint 34 evidence_time_statuses
17 metadata_inventory_fingerprint
```

Row 12是每个flattened source ref的canonical privacy-safe tuple：source ref ordinal、task、result position、table、
source use、candidate position（不是raw key）、registry entry ordinal、applicable tagged locator fields、authoritative
fingerprint availability/value；它证明binding但不输出policy/full payload。Rows 30--34 contain canonical tuples
ordered Task15→16→17→18 and then owner tuple position；没有dynamic provenance rows。Task15 tuple只含
`unavailable`，其他fingerprint tuples只含validated lowercase digests；time tuple含每个resolved ref及structured
declaration的`verified/unverified/not_applicable`，按flattened ref、attributions、profiles、performance order。
Provenance不得包含raw candidate label、entity、segment/cohort/vintage label、
feature value、target/vector、owner finding value、credential、path、repr或完整payload。Package version固定
记录`0.1.0`，contract version记录`task19-contract-targeted-fixed-v2`；不声称v0.2 released。Current clock、generated
at、environment和plot不参与任何fingerprint/provenance；plot只读existing result。

### 9A.4 FIX-D canonical/provenance direct-test obligations

Direct tests冻结：repeat/copy equality、semantic/order change sensitivity、finite hex float、nonfinite rejection、
bool/int distinction、aware/naive/Timestamp golden values、unsupported malicious callbacks zero、mapping key order、
nine dataclass field order、11 fingerprints、64 lowercase hex、35-row exact inventory/order、Task15 unavailable
limitation、owner tuple element order、no current clock和all provenance privacy sentinels。
Malicious callback matrix exact包含`__repr__/__str__/__hash__/__bool__/__int__/__float__/__iter__/
__lt__/__eq__/__array__`，type rejection前各count必须为0。

## 10. Exact result tables and schemas

### 10.1 Global registry rules

`GovernanceResult`的十个`pandas.DataFrame` fields按且仅按：`explanations, model_attributions,
prediction_drift, performance_stability, candidate_comparisons, governance_evaluations,
recommendations, governance_summary, governance_metadata, provenance`。不能省略、optionalize或改成mapping。
每表即使零row也由explicit dtype map构造exact columns/order/dtypes；禁止裸`DataFrame(columns=...)`。

本节schema notation中`!`=每个materialized row non-null，`?`=nullable且只可在下述条件null。Text=`string`，
positions/counts=`Int64`，real=`Float64`，flags=`boolean`；没有public `object` dtype。所有datetime columns在
naive invocation是`datetime64[ns]`，aware invocation是`datetime64[ns, UTC]`，保留nanoseconds且同表/同调用
不混用。每row `status!`、`finding_key!`；`reason?` iff status=`available`时必须null，其他status必须是第11.4
节该表允许reason。Duplicate unique identity在materialization前fail closed，使用对应invalid declaration/
source error，不first-wins/deduplicate。所有keys/positions来自validated declarations，不输出raw entity/group。

### 10.2 Public result schema registry（normative）

1. **`explanations`** — one row per `GovernanceExplanation` declaration，不随source展开。Columns：
`explanation_position:Int64!, explanation_key:string!, candidate_position:Int64!, candidate_family:string!,
method:string!, source_ref_position:Int64!, source_task:string!, source_result_position:Int64!, source_table:string!,
source_registry_position:Int64!, source_fingerprint:string?, feature_key:string?, relation:string?, priority:Int64!,
evidence_time_status:string!, source_status:string!, source_reason:string?, status:string!, reason:string?,
finding_key:string!`。Fingerprint只在owner有authoritative digest时non-null；feature/relation仅attribution methods
适用；source reason服从owner；Task19 reason服从第11节。`explanation_position`是
`GovernancePolicy.explanations`的0-based declaration ordinal，exact为`0..N-1`。Unique=
`explanation_position`，public rows严格按该position升序且没有secondary sort；key=
`governance:explanation:<explanation_position>`。

2. **`model_attributions`** — one row per valid `GovernanceAttributionEvidence` declaration。Columns：
`candidate_position:Int64!, attribution_position:Int64!, candidate_family:string!, method:string!, feature_key:string!,
metric_key:string?, value:Float64!, relation:string!, evaluation_scope:string!, support_n:Int64!,
uncertainty_std:Float64?, permutation_repeats:Int64?, random_state:Int64?, evidence_as_of:<datetime>!,
evidence_time_status:string!, source_task:string!, source_result_position:Int64!, source_table:string!,
source_ref_position:Int64!, source_fingerprint:string?, source_status:string!, source_reason:string?, status:string!,
reason:string?, finding_key:string!`。Coefficient：metric/uncertainty/repeats/seed null，scope not_applicable；native：
same nulls、nonnegative value、relation not_directional；permutation：metric、holdout/oof scope、uncertainty、
positive repeats、seed均non-null。Unique=`(candidate_position,attribution_position)`；order candidate position，
method fixed order，attribution declaration position；finding grammar同第11.5节。

3. **`prediction_drift`** — declarations按`(candidate_key,prediction_kind,scope_key,scope_position,
reference_state_fingerprint)`分组，每组必须exact one reference+one current；没有implicit all-pairs，one pair one
row。Columns：`candidate_position:Int64!, reference_profile_position:Int64!, current_profile_position:Int64!,
prediction_kind:string!, scope_key:string!, scope_position:Int64?, reference_snapshot_key:string!,
current_snapshot_key:string!, reference_analysis_as_of:<datetime>!, current_analysis_as_of:<datetime>!,
reference_time_status:string!, current_time_status:string!, reference_support_n:Int64!, current_support_n:Int64!,
reference_missing_n:Int64!, current_missing_n:Int64!, bin_count:Int64!, reference_state_fingerprint:string!,
reference_source_fingerprint:string?, current_source_fingerprint:string?, metric:string!, prediction_tvd:Float64?,
direction:string!, uncertainty_std:Float64?, bootstrap_repeats:Int64!, random_state:Int64!, status:string!,
reason:string?, finding_key:string!`。bin_count固定10；vectors不输出。TVD/std仅available时non-null；undefined/
not-verifiable时均null。Unique=三个positions；order candidate then reference profile declaration；finding grammar固定。

4. **`performance_stability`** — declarations按`(candidate_key,evaluation_scope,scope_key,scope_position,
metric-kind,source fingerprint)`分组，每组exact one reference+one current；ranking_scores→roc_auc，
event_probabilities→brier_score，若一个declaration同时提供二者则是两个closed metric units且必须用两组独立
declarations，禁止一对扩成两row。Columns：`candidate_position:Int64!, reference_evidence_position:Int64!,
current_evidence_position:Int64!, evaluation_scope:string!, scope_key:string!, scope_position:Int64?,
reference_snapshot_key:string!, current_snapshot_key:string!, reference_window_start:<datetime>!,
reference_window_end:<datetime>!, current_window_start:<datetime>!, current_window_end:<datetime>!,
reference_evidence_as_of:<datetime>!, current_evidence_as_of:<datetime>!, reference_time_status:string!,
current_time_status:string!, metric:string!, reference_value:Float64?, current_value:Float64?, delta:Float64?,
direction:string!, reference_uncertainty_std:Float64?, current_uncertainty_std:Float64?,
reference_bootstrap_repeats:Int64!, current_bootstrap_repeats:Int64!, reference_random_state:Int64!,
current_random_state:Int64!, reference_support_n:Int64!, current_support_n:Int64!,
reference_assignment_mechanism:string!, current_assignment_mechanism:string!, reference_common_support:string!,
current_common_support:string!, reference_source_fingerprint:string?, current_source_fingerprint:string?,
status:string!, reason:string?, finding_key:string!`。Values/std/delta仅available时non-null；否则全部null。
Unique=三个positions；order candidate/reference declaration/metric inventory；finding grammar固定。

5. **`candidate_comparisons`** — exactly one row per declared pair×every declared criterion，包括decision与
diagnostic。Columns：`pair_position:Int64!, champion_candidate_position:Int64!, challenger_candidate_position:Int64!,
candidate_family:string!, criterion_position:Int64!, criterion_role:string!, source_task:string!, source_table:string!,
metric_key:string!, scope_key:string!, scope_position:Int64?, rule_key:string?, champion_source_result_position:Int64!,
challenger_source_result_position:Int64!, champion_source_ref_position:Int64!, challenger_source_ref_position:Int64!,
champion_source_fingerprint:string?, challenger_source_fingerprint:string?, champion_time_status:string!,
challenger_time_status:string!, champion_source_status:string!, champion_source_reason:string?,
challenger_source_status:string!, challenger_source_reason:string?, source_snapshot_status:string!,
entity_alignment_status:string!, normalized_support_identity:string?, champion_value:Float64?,
challenger_value:Float64?, delta:Float64?,
champion_support_n:Int64?, challenger_support_n:Int64?, support_unit:string?, direction:string!,
target_low:Float64?, target_high:Float64?, comparison_outcome:string!, support_comparable:boolean!, status:string!,
reason:string?, finding_key:string!`。Target bounds non-null iff target_range。Available directional/tie row要求两value、
delta、supports/unit non-null；diagnostic not_directional在两value available时仍保留raw delta且outcome
not_directional，status=`not_applicable`,reason=`diagnostic_only`，但不贡献recommendation；
not_comparable/unavailable/undefined/not-verifiable的delta null。Normalized support identity是privacy-safe
canonical JSON string，仅两侧support definition可验证时non-null，不含raw rows/groups。
Unique=`(pair_position,criterion_position)`；pair then criterion order；finding grammar固定。

6. **`governance_evaluations`** — exactly one row per pair×every criterion。Columns：
`pair_position:Int64!, champion_candidate_position:Int64!, challenger_candidate_position:Int64!,
criterion_position:Int64!, criterion_role:string!, required_for_promotion:boolean!, priority:Int64!,
comparison_outcome:string!, comparable:boolean!, counts_toward_minimum:boolean!, blocks_promotion:boolean!,
directional_contribution:string!, evidence_time_status:string!, status:string!, reason:string?, finding_key:string!`。
Contribution exact `challenger_better/champion_better/tie/not_comparable/not_directional`；counts true only available
decision better/worse/tie；blocks true exactly hard required-incomplete proof underFIX-C。Unique pair+criterion；order同上。

7. **`recommendations`** — exactly one row per pair；zero challenger typed-empty。Columns：
`pair_position:Int64!, champion_candidate_position:Int64!, challenger_candidate_position:Int64!,
candidate_family:string!, recommendation:string!, recommendation_basis:string!, hard_veto:boolean!,
human_review_mode:string!, human_review_required:boolean!, minimum_comparable_criteria:Int64!,
criteria_available_n:Int64!, criteria_unavailable_n:Int64!, criteria_better_n:Int64!, criteria_worse_n:Int64!,
criteria_tied_n:Int64!, required_incomplete_n:Int64!, support_comparable:boolean!, status:string!, reason:string?,
finding_key:string!`。每个合法pair recommendation均`status=available,reason=pd.NA`，包括
insufficient_evidence；basis而非reason说明决策路径。Unique/order=`pair_position`；finding grammar固定。

8. **`governance_summary`** — exactly one row per candidate，没有governance-wide extra row。Columns：
`candidate_position:Int64!, candidate_family:string!, declared_role:string!, declared_state:string!, source_task:string!,
source_result_position:Int64!, source_candidate_position:Int64?, source_snapshot_status:string!,
entity_alignment_status:string!, evidence_time_status:string!, criterion_count:Int64!, available_criterion_count:Int64!,
unavailable_criterion_count:Int64!, not_verifiable_criterion_count:Int64!, attribution_count:Int64!,
prediction_drift_count:Int64!, performance_stability_count:Int64!, recommendation_count:Int64!,
human_review_required_count:Int64!, status:string!, reason:string?, finding_key:string!`。source candidate position仅
strategy/scenario有subcandidate时non-null。Summary始终available/reason null；counts exact nonnegative。
Unique/order=`candidate_position`；finding grammar固定。

9. **`governance_metadata`** — one declaration expands deterministically intofield rows：purpose、owner、materiality、
每个assumption、每个limitation、每个threshold、issue status、remediation status；minimum 5 rows/declaration。
Columns：`metadata_position:Int64!, metadata_scope:string!, candidate_position:Int64?, field_position:Int64!,
field_key:string!, item_position:Int64?, text_value:string?, numeric_value:Float64?, evidence_time_status:string!,
status:string!, reason:string?, finding_key:string!`。Candidate position non-null iff candidate scope；item position
non-null only repeated assumption/limitation/threshold；exactly one of text/numeric non-null（threshold numeric，其他
text）；time not_applicable，status available，reason null。Unique=`(metadata_position,field_position)`；order
metadata declaration then fixed field order/items declaration order；finding key uses scope branch and positions。

10. **`provenance`** — always exactly35 rows from§9A.3。Columns：`provenance_position:Int64!,
provenance_key:string!, provenance_value:string!, status:string!, reason:string?, finding_key:string!`。Value是
canonical JSON string；status always available/reason null；positions exact0..34，keys exact fixed inventory。
Unique/order=`provenance_position`（key also unique）；finding grammar fixed。

### 10.3 Typed-empty, duplicate and schema acceptance

All ten tables have the identical registered schema/dtypes when empty and non-empty；empty datetime dtype follows the
invocation's validated clock mode。Materialization validates every non-null/conditional invariant before returning；
no row drop/coercion/object fallback。Schema tests must independently cover empty and populated form for each table。

## 11. Status、errors、finding keys、precedence

### 11.1 Exact public error contract（76 keys）

所有public validation failure仅抛built-in `ValueError`，`str(error)` exact为
`model governance: <error_key>`；没有prefix variant、动态value、repr、raw key或exception chaining text。
Closed 76-key inventory如下；其排列只登记identity/count，不定义跨phase precedence；唯一precedence见§11.2：

```text
invalid_policy_type, invalid_governance_key, invalid_governance_version,
invalid_analysis_as_of, datetime_awareness_mismatch, invalid_candidate_container,
invalid_pair_container, invalid_criterion_container, invalid_metadata_container,
invalid_evidence_ref_container, invalid_explanation_container,
invalid_attribution_container, invalid_prediction_profile_container,
invalid_performance_evidence_container, invalid_owner_result_container,
invalid_human_review_mode, invalid_entity_alignment, invalid_candidate,
duplicate_candidate, invalid_champion, invalid_pair, invalid_pair_coverage,
duplicate_pair, invalid_criterion, duplicate_criterion, unsupported_criterion,
invalid_metadata, invalid_evidence_ref, duplicate_evidence_ref,
invalid_explanation, invalid_attribution, invalid_prediction_profile,
invalid_performance_evidence, duplicate_owner_source,
invalid_source_binding, unsupported_source, invalid_source_locator,
source_not_found, source_not_unique, invalid_owner_schema, invalid_owner_dtype,
invalid_owner_status, invalid_owner_reason, invalid_owner_value,
invalid_source_fingerprint, source_fingerprint_mismatch,
authoritative_time_missing, authoritative_time_mismatch, future_evidence_time,
invalid_canonical_value,
resource_candidates, resource_comparison_pairs, resource_criteria,
resource_explanations, resource_model_attribution_rows,
resource_attribution_permutation_repeats, resource_prediction_profile_rows,
resource_prediction_profile_bin_count, resource_performance_evidence_rows,
resource_performance_vector_values, resource_drift_bootstrap_draws,
resource_performance_bootstrap_draws, resource_prediction_drift_rows,
resource_performance_stability_rows, resource_governance_metadata_rows,
resource_source_evidence_rows, resource_candidate_comparison_rows,
resource_governance_evaluation_rows, resource_recommendation_rows,
resource_governance_summary_rows, resource_provenance_rows,
resource_risk_validation_results, resource_data_audit_results,
resource_decision_strategy_results, resource_lifecycle_monitoring_results,
resource_evidence_refs
```

上表为76 keys（50 semantic + 26 resource）。
每条failure只能映射到该closed inventory；具体字段问题归入对应aggregate semantic key，禁止新suffix。
Invalid ref/owner/schema/fingerprint/time/value永远raise，不物化成unavailable row。

Owner validation is exhaustive: `invalid_owner_result` is intentionally not a public error key. Privacy safety is
also not a generic caller-time classifier: `privacy_unsafe_value` is intentionally not a public error key. Invalid
candidate, metadata, evidence-ref, criterion, explanation, attribution, prediction-profile, or performance values
must use their field-specific closed error; raw/private values remain prohibited from output tables, errors, keys,
provenance, and plots.

### 11.2 Single normative global validation phase order and whole-input safety

在读取任一owner cell、计算fingerprint、resource projection或materialize前，先对policy、四个owner tuples和
四类structured collections的每个元素做exact type/container/scalar/privacy validation；inactive、diagnostic、
unpaired、unreferenced元素也不能逃过。恶意protocol callbacks在type rejection前调用次数必须为0。

本节是全合同唯一normative **global validation phase order**与cross-phase public-error precedence authority；
§9和§12只能引用该顺序并定义各自local semantics/internal order，不得重述或覆盖：

1. policy exact type、governance key/version、所有datetime exact scalar type与所有container exact tuple type；
2. 所有element scalar/privacy safety（只验证类型/shape，不执行任何`resource_*` length gate）；
3. candidate identity/duplicate/role/state、champion、pair structural/family validation；
4. criterion、metadata、evidence-ref locator shape、explanation及structured declarations structural validation；
5. owner tuple element exact type、duplicate semantic source和candidate/source binding；
6. lightweight owner inspection与ref resolution：table presence、schema/dtype/status/reason/value、row locator、
   authoritative fingerprint/expected match及bounded private identity indexes；
7. authoritative owner/row/structured evidence-time extraction与required-time presence检查；缺失先报
   `authoritative_time_missing`；
8. invocation-wide datetime awareness compatibility与authoritative-time consistency检查；不一致分别使用
   `datetime_awareness_mismatch`或`authoritative_time_mismatch`；
9. future-evidence gate；
10. snapshot/alignment proof classification；
11. caller-variable resource preflight：第12.2节17 gates按其registry order执行，随后第12.3节fixed invariants；
12. normalized support/comparison eligibility；
13. numeric comparison与governance evaluation；
14. recommendation aggregation；
15. ten-table public result materialization。

此顺序同时是76-key error registry的唯一multi-invalid precedence authority；其他章节只能引用，不得复制或
改写第二套phase list。Tuple/container length可在structural safety中读取并记录overage，但exact-tuple elements只做
一次linear type/privacy pass；不得分配Cartesian/projection/public rows，其observable `resource_*` error仍只在phase 11
发生，不能伪装成phase 1/2 structural error。Phase 6--8只读取authoritative result/row time与必要owner
metadata，不生成pair×criterion rows、不展开metadata public rows、不计算bootstrap。更早structural/source
validity全部通过后，future evidence与任一resource overage共存时首错必须是`future_evidence_time`；required
authoritative time missing或不一致与resource overage共存时首错分别是`authoritative_time_missing`、
`authoritative_time_mismatch`。Resource preflight始终先于Cartesian allocation和任何public materialization。
Canonical encoder only
runs after its whole payload domain passes; any impossible post-validation encoder failure maps`invalid_canonical_value`。
失败不返回partial result、不修改输入、不调用plot/model/condition/owner computations。

### 11.3 Task19 status/reason semantics（5 statuses, 15 reasons）

Task19 status vocabulary exact：`available,unavailable,undefined,not_applicable,not_verifiable`。Task19 reason
vocabulary exact：

```text
source_unavailable, source_undefined, source_not_verifiable,
support_not_comparable, insufficient_support, maturity_not_comparable,
snapshot_unverified, alignment_unverified, time_unverified,
common_support_unverified, insufficient_bootstrap_support,
zero_denominator, single_class,
operation_not_applicable, diagnostic_only
```

`available`必须`reason=pd.NA`，不用`computed/available`字符串。`unavailable`表示合法且唯一的owner row明确
unavailable，reason只能`source_unavailable`；`undefined`表示合法domain内数学无定义，reason只能
`source_undefined,insufficient_support,insufficient_bootstrap_support,zero_denominator,single_class`；
`not_verifiable`表示evidence存在但proof不足，使用
`source_not_verifiable,support_not_comparable,maturity_not_comparable,snapshot_unverified,alignment_unverified,
time_unverified,common_support_unverified`；`not_applicable`只用`operation_not_applicable/diagnostic_only`。
owner原始closed status/reason原样仅进入`source_status/source_reason`，再映射到Task19语义；它们不扩展上述
15-key registry。Optional attribution/profile/performance/explanation未声明时不产row；required declaration/ref
missing是error，故完全删除`not_provided`、`source_missing`、`source_table_missing`及`*_not_provided`冲突。

Recommendation对每个valid pair总能物化一个`available/reason=pd.NA` row；`recommendation_basis`exact为
`hard_state_veto,required_evidence_incomplete,minimum_not_met,mixed_evidence,all_ties,
challenger_favorable,champion_favorable`，与五值recommendation分离，不滥用reason。

### 11.4 Ten-table status/reason matrix

| table | allowed Task19 statuses | exact reason subset/additional rule |
|---|---|---|
| explanations | available,unavailable,undefined,not_verifiable | source_*、insufficient_support、snapshot_unverified、alignment_unverified、time_unverified；caller/source mapping exact |
| model_attributions | available,unavailable,undefined,not_verifiable | source_*, insufficient_support, snapshot_unverified, time_unverified |
| prediction_drift | available,unavailable,undefined,not_verifiable | source_*, zero_denominator, insufficient_support, insufficient_bootstrap_support, snapshot_unverified, time_unverified |
| performance_stability | available,unavailable,undefined,not_verifiable | source_*, single_class, insufficient_support, insufficient_bootstrap_support, common_support_unverified, snapshot_unverified, time_unverified |
| candidate_comparisons | all five | source_*, support_not_comparable, insufficient_support, maturity_not_comparable, snapshot_unverified, alignment_unverified, time_unverified, diagnostic_only |
| governance_evaluations | all five | exact same mapped reason as its pair×criterion comparison row |
| recommendations | available only | reason always null；basis承载recommendation semantics |
| governance_summary | available only | reason always null；三种proof status在独立columns |
| governance_metadata | available only | reason null；evidence_time_status=`not_applicable` |
| provenance | available only | reason null；value为9A.3 canonical safe fragment |

### 11.5 Exact finding-key families（10）

每个materialized row都必须有non-null unique Task19 finding key，即使status available；copy-equivalent inputs产生
相同keys。Exact zero-based grammars：

```text
governance:explanation:<explanation_position>
governance:attribution:<candidate_position>:<attribution_position>
governance:drift:<candidate_position>:<reference_profile_position>:<current_profile_position>
governance:stability:<candidate_position>:<reference_evidence_position>:<current_evidence_position>
governance:comparison:<pair_position>:<criterion_position>
governance:evaluation:<pair_position>:<criterion_position>
governance:recommendation:<pair_position>
governance:summary:<candidate_position>
governance:metadata:governance:<metadata_position>:<field_position>
governance:metadata:candidate:<candidate_position>:<metadata_position>:<field_position>
governance:provenance:<provenance_position>
```

Positions来自validated declaration/paired materialization order，绝不用pandas index、Python hash或value-derived
ordinal。Drift/stability matched-pair identity以两侧declaration positions绑定；scope/time/group raw label不进入。
Comparison/evaluation family隔离且每个pair×criterion唯一；recommendation每pair唯一；metadata按scope分支，
candidate scope必须含candidate position，field_position是第6.3节closed field/item materialization ordinal；
provenance精确0..34。Owner input finding_key仅为source locator，
namespace不变且绝不复制成Task19 output identity。Keys不含raw candidate label、feature key/value、metric display、
entity/group/segment/cohort/vintage/time value、target、repr或malicious string。

`warnings`按十表顺序收集status为unavailable/undefined/not_verifiable rows的Task19 finding keys；
`limitations`按首次出现顺序去重相应15-key reasons。二者只含closed safe tokens。

### 11.6 FIX-D error/status/finding direct-test obligations

Direct tests要求76 errors逐项映射且无unknown key（resource key的reachability见第12节）；
multi-invalid precedence明确绑定本节唯一global order，并直接覆盖future+resource、authoritative-time-missing+
resource、authoritative-time-mismatch+resource三个首错sentinel；raw-secret absence；15 reasons
在有semantic path处reachable且无unknown；十表status/reason/null scan；invalid source raises；十family exact regex、
uniqueness、copy determinism、pair×criterion collision resistance、owner namespace isolation；errors/reasons/keys/
provenance统一恶意sentinel privacy scan。

## 12. Ordering、resource gates、no truncation

### 12.1 Phases and complexity

Global phase placement and all cross-phase precedence follow the sole normative authority in §11.2；本节不重述或
覆盖。只有该global order进入resource-preflight phase后，§12.2才定义17 caller-variable gates的内部相对顺序，
随后检查§12.3九项fixed invariants。早期type/container/element safety即使读取length并记录overage，也不得提前
发出`resource_*`；所有resource errors留在global resource phase。

§11.2 lightweight owner inspection只可验证owner result types、table presence、exact schema/dtype、row counts、
required provenance/fingerprints并为每个approved owner table建立一个bounded private identity index；不得生成
pair×criterion rows、explanations、drift/stability、summary或任何public result table。Refs用这些indexes以
`O(total inspected owner rows + evidence_refs log N)`或equivalent bounded lookup解析；禁止per-ref repeated full-table
scan。所有sum/product使用bounded Python integers，并在Cartesian/public materialization前检查。

### 12.2 Caller-variable resource gate registry（17 exact gates）

After every earlier §11.2 phase has passed and the global resource-preflight phase has begun, the following17 gates run
in the listed order。Every maximum is inclusive and reachable by structurally valid public input；for every gate, a
fixture at max passes that gate and a fixture at max+1 is the first failure **within that resource phase** with the listed
exact error；an earlier semantic/source/time failure retains §11.2 precedence。Primitive resource projections precede
bounded pairing/semantic checks that depend on them, without overriding earlier global phases。

| order | gate / exact count formula | max | resource subphase | exact error key |
|---:|---|---:|---|---|
| 1 | `risk_validation_results=len(risk_validations)` | 16 | container | `resource_risk_validation_results` |
| 2 | `data_audit_results=len(data_audits)` | 16 | container | `resource_data_audit_results` |
| 3 | `decision_strategy_results=len(decision_strategies)` | 16 | container | `resource_decision_strategy_results` |
| 4 | `lifecycle_monitoring_results=len(lifecycle_monitorings)` | 16 | container | `resource_lifecycle_monitoring_results` |
| 5 | `candidates=len(policy.candidates)` | 16 | declaration | `resource_candidates` |
| 6 | `comparison_pairs=len(policy.comparison_pairs)` | 15 | declaration | `resource_comparison_pairs` |
| 7 | `criteria=len(policy.criteria)` | 64 | declaration | `resource_criteria` |
| 8 | `explanations=len(policy.explanations)` | 4096 | declaration | `resource_explanations` |
| 9 | `model_attribution_rows=len(model_attributions)` | 4096 | declaration | `resource_model_attribution_rows` |
| 10 | `attribution_permutation_repeats=max(permutation_repeats,0)` | 100 | declaration | `resource_attribution_permutation_repeats` |
| 11 | `prediction_profile_rows=len(prediction_profiles)` | 4096 | declaration | `resource_prediction_profile_rows` |
| 12 | `performance_evidence_rows=len(performance_evidence)` | 4096 | declaration | `resource_performance_evidence_rows` |
| 13 | `governance_metadata_rows=sum(5+len(assumptions)+len(limitations)+len(thresholds))` | 256 | declaration projection | `resource_governance_metadata_rows` |
| 14 | `evidence_refs=len(flattened ref occurrences in §7.5 order)` | 8192 | declaration projection | `resource_evidence_refs` |
| 15 | `performance_vector_values=sum(len(target_values))` | 200000 | post-schema primitive | `resource_performance_vector_values` |
| 16 | `drift_bootstrap_draws=sum(bootstrap_repeats per matched profile pair)` | 2000000 | post-pair projection | `resource_drift_bootstrap_draws` |
| 17 | `performance_bootstrap_draws=sum(reference_support_n*reference_repeats + current_support_n*current_repeats)` | 2000000 | post-pair projection | `resource_performance_bootstrap_draws` |

Prediction/performance bootstrap repeats are each exact int `[2,1000]`; attribution permutation repeats exact `[1,100]`。
Performance declaration must provide exactly one ofranking scores/event probabilities and its length equals targets；
counting targets therefore counts each evaluated position once。Max/max+1 fixtures vary only the named dimension while
keeping earlier dimensions within bounds；gate precedence sentinels assert the first error。Projection maxima are
arithmetically reachable without violating earlier gates；for example performance max+1 uses four side terms
`1000*1000 + 999*1000 + 1*2 + 1*999 = 2,000,001` with vector total1002。Drift multinomial runtime is
`O(drift_bootstrap_draws*10)`；performance resampling runtime is`O(performance_bootstrap_draws)`，both before metric
constant factors and without materializing synthetic rows。

### 12.3 Fixed contract cardinality/projection invariants（9）

These are consequences of already validated public structure, not independent caller gates and therefore have exact
cardinality tests rather than fake max+1 tests. Their retained `resource_*` error keys are defensive invariant failures。

| invariant | exact formula / bound | defensive key |
|---|---|---|
| profile shape | each profile exactly9 boundaries+10 counts, `bin_count=10` | `resource_prediction_profile_bin_count` |
| source evidence rows | exactly one resolved owner row per unique valid ref occurrence; equals`evidence_refs`, max8192 | `resource_source_evidence_rows` |
| prediction drift rows | `prediction_profile_count / 2`, exact paired even count, max2048 | `resource_prediction_drift_rows` |
| performance stability rows | `performance_evidence_count / 2`, exact paired even count, max2048 | `resource_performance_stability_rows` |
| candidate comparison rows | `pair_count * criterion_count`, includes diagnostic, max`15*64=960` | `resource_candidate_comparison_rows` |
| governance evaluation rows | `pair_count * criterion_count`, max960 | `resource_governance_evaluation_rows` |
| recommendation rows | `pair_count`, max15 | `resource_recommendation_rows` |
| governance summary rows | `candidate_count`, max16 | `resource_governance_summary_rows` |
| provenance rows | exactly35 in every valid result | `resource_provenance_rows` |

With any pair, criteria contain at least one decision criterion；zero challenger permits empty criteria and all pair-derived
tables aretyped-empty。Metadata declarations remain exactly one governance-wide plus zero/one per candidate, while their
field expansion is bounded by resource gate13。No truncation, slice, head, sample, top-k, partial result, repeat shrinking, profile
dropping or implicit all-pairs is permitted。Same input/order yields identical tables/keys/fingerprints；no set/hash/locale/
pandas-index/parallel-completion/current-time ordering。

### 12.4 Resource acceptance obligations

Implementation acceptance requires 17 max-pass/max+1-first-fail parameterized cases, precedence multi-overage sentinels,
future+resource、authoritative-time-missing+resource与authoritative-time-mismatch+resource的§11.2首错sentinels，
and nine exact invariant cases。Pair semantic validation additionally proves aftergate6 that
`challenger_count=candidate_count-1`and`pair_count=challenger_count`。It also proves resource preflight occurs before
pair×criterion allocation and public table
materialization, and uses an owner lookup spy/counter to reject`O(refs*rows)` repeated scans。

## 13. Result-only visualization

`plot_model_governance`位于`visualization.py`并严格复用Task15单kind裸
`matplotlib.figure.Figure` precedent。它只接受exact `GovernanceResult`及closed kind：

| kind | sole source | plotted unit/order |
|---|---|---|
| `importance` | `model_attributions` | candidate declaration、method、feature evidence order；x=value、y=feature key |
| `candidate_comparison` | `candidate_comparisons` | pair、criterion order；champion/challenger values |
| `prediction_drift` | `prediction_drift` | candidate/scope order；TVD value |
| `performance_stability` | `performance_stability` | candidate/scope/window order；reference/current metric values |
| `governance_summary` | `recommendations` + `governance_summary` | pair/candidate order；recommendation/status counts |

Plot只schema-validate并呈现已物化rows，不读取owner results、structured inputs、model或raw data，不调用
`evaluate_governance`，不计算importance、TVD、metrics、delta、comparison、recommendation或summary。
malformed/empty/no-available source使用built-in `ValueError`且不得创建partial Figure。每次调用创建一个
caller-owned Figure；库不调用`show/savefig/close`，不改backend/rcParams/global style，不保存文件。
Feature key和anonymous scope ordinal仅在本合同允许时显示；不得显示raw entity、segment/cohort/vintage
label或feature value。Axis category/order完全继承source table order；缺失/not-verifiable row不画data
artist且不重排。只使用现有matplotlib，不新增seaborn/Plotly/SHAP等依赖。

## 14. Packaging、compatibility、tests

获批后只可按第2节allowlist新增/修改Task19 owner、result-only plot、tests、exports及明确
批准的API docs；不得改 pyproject、lock、mandatory dependencies、package version 或
v0.1 behavior。module 必须进入 wheel/sdist、source-free import，version 保持 `0.1.0`。

### 14.1 Normative implementation acceptance matrix

The word **requires** below is mandatory; representative smoke tests do not substitute for a named category。

| category | mandatory direct evidence |
|---|---|
| Public API/package | exact module import、12 top-level exports、10 frozen dataclass field order/defaults、two exact signatures、version0.1.0、no new mandatory dependencies |
| Ten schemas / CR-08 | each table independently: exact columns/order/dtypes, typed-empty and populated, conditional nullability, unique identity, duplicate rejection, deterministic order, exact finding family；explanations declaration-order adversary使candidate/method/priority/source/key lexical order与声明顺序相反，rows仍为`explanation_position=0..N-1`且priority不得重排 |
| CR-01 binding | two model/twoTask15 results, illegal same-result reuse, Task18 same result/different scenarios, strategy binding, wrong family, out-of-range, missing/duplicate locator, pandas index ignored, deep-copy identity |
| CR-02 direction | parameterized all92 registry entries, reversed assertion rejection, higher/lower, diagnostic non-directional, approved target-range subset/full truth table, no epsilon/isclose |
| CR-03 recommendation | all five values direct; required unavailable, support/snapshot/alignment failure, optional unavailable, minimum count, all ties, mixed, better+ties, worse+ties, no majority, promotion always human review |
| CR-04 time | exact/rejected datetime types/subclasses/NaT, awareness/UTC, past/equal/future, Tasks15--18, all structured evidence, future invariance and precedence |
| CR-05 roadmap | one direct category for each of§6.4's16 rows; coefficient/native/permutation, source provenance, model/Task17/Task18 comparison, traces, assignment, TVD, AUC/Brier, Task16 summary, stability, metadata, five plots |
| CR-06 registry | parameterized allow for all38 entries and closed deny task/table/use matrix; assert no invented Task16 fields |
| CR-07 refs | tagged compound locators, wrong table/missing/duplicate/fingerprint, copy stability, attribution surface/candidate binding/feature-value privacy/method-source matrix；attribution/drift/stability single-carrier、无需Candidate ref副本、合法owner-context sharing非duplicate、wrong use/source binding拒绝、duplicate structured identity按enclosing declaration error拒绝 |
| CR-09 math/support | delta orientation, higher/lower/range, Task15/17 normalized support, Task18 AM-04 reuse, fold/scope/maturity/support mismatch, equal value mismatch not tie |
| CR-10 state | zero candidate/champion/challenger, multiple champion, pair coverage/duplicate/reversed/extra/self, every state, rejected/retired veto, approved challenger, two review modes, no mutation/auto promotion |
| CR-11 proof | intra-result verified, cross-result unverified, equal as-of and Task16 digest not proof, position-only/raw-entity alignment forbidden, not-applicable vs unverified |
| CR-12 fingerprint | all11 fingerprints, 35 provenance rows, scalar/datetime golden values, mapping order, nonfinite/malicious rejection, copy determinism/sensitivity, registry identities, owner limitations/privacy |
| CR-13 vocabulary | all76 approved keys mechanically mapped with direct representative executable branches, all15 semantically reachable reasons, ten-table status/reason scan, ten finding regex/uniqueness/determinism/privacy/namespace separation |
| CR-14 resources | all17 variable gates exact max/max+1 first-fail plus precedence sentinels; all9 fixed invariants; no preflight Cartesian and bounded lookup counter |
| Security/privacy | scan all ten tables, errors, reasons, keys, provenance and plot text for raw entity/segment/cohort/vintage/feature value/target/model repr/malicious secret; this validates non-disclosure invariants, not a `privacy_unsafe_value` error branch |
| Malicious scalar | `repr,str,hash,bool,int,float,iter,eq,lt,array` objects in selectors, sources and structured declarations; callbacks remain zero before rejection |
| Immutability/determinism | policy/nested tuples/owner DataFrames/structured tuples unchanged; repeated/deep-copy tables, order, keys, fingerprints, recommendations equal across clock/env/hash seed |
| Result-only plots | all five kinds consume result only, exact Figure, no evaluator/owner/recompute, closed empty behavior, safe labels/order/lifecycle, no dependency/global style/save |
| Owner regression | run the real Task15,16,17,18 public/regression test files existing at implementation time; no predecessor semantic edits |
| Distribution | reuse Task18 controlled offline harness; wheel contains model_governance and plot surface, sdist source, source-free clean import and12 symbols/plot, no deps/version change |
| Boundary claims | static/behavior checks forbid causal lift/“caused by”/legal fairness certification/adverse-action notice/automatic deployment or promotion mutation |

### 14.2 Roadmap and release-critical traceability

The 16§6.4 capability rows are themselves the exact traceability registry and each maps to a named test category above；
none may be satisfied only by a general smoke test。Packaging/build/distribution are implementation acceptance gates because
Task19 adds public API and module surface；they do not publish, tag or release。Task19 implementation commands must use the
project uv environment and ultimately run full pytest、Ruff check/format-check、build and controlled clean-install evidence
required byAGENTS.md；this contract-only residual targeted repair runs none of them。

## 15. Approved contract status and review history

Task 19唯一一次full contract review为`No-Go`；第一次bounded contract-review closure同样为`No-Go`。
该closure关闭11项，只留下`T19-CR-04/07/08/13/14`；随后完成residual targeted repair，且bounded
contract-review re-closure为`Go`。原approved checkpoint的16项finding均保持其历史状态，未执行第二次
full contract review。其后post-approval implementation-blocker adjudication确认`T19-CR-13`中
`invalid_owner_result`与`privacy_unsafe_value`不可达；本次targeted amendment删除这两个key并将当前
executable vocabulary收紧为76项；bounded approved-contract amendment closure随后取得`Go`，因此当前amended
contract正式为`Approved — Go`。

FIX-A关闭`T19-CR-05`；FIX-B关闭`T19-CR-01/06/07/11`；FIX-C关闭`T19-CR-02/03/09/10`；
FIX-D关闭`T19-CR-04/12/13`；FIX-E关闭`T19-CR-08/14/15/16`。`T19-CR-13`历史closure保持`Closed`；
Post-Approval T19-CR-13 Blocker已确认并关闭，Targeted Approved-Contract Amendment为`Applied`；
Bounded Approved-Contract Amendment Closure为`Go`并已`Closed`。`T19-CR-15`保持`Closed`，仅其direct
acceptance wording按本amendment改为76/76。

```text
Task 19 original approved contract checkpoint: Approved — Go v1 / 78 keys (442fdc0 immutable)
Task 19 amended contract: Approved — Go v2 / 76 keys
Task 19 implementation: In progress
Unique Full Contract Review: Completed once — No-Go
First Bounded Contract-Review Closure: No-Go
Residual Targeted Contract Repair: Completed for T19-CR-04/07/08/13/14
Bounded Contract-Review Re-Closure: Go (original vocabulary)
Post-Approval T19-CR-13 Blocker: Confirmed — Closed
Targeted Approved-Contract Amendment: Applied — Closed
Bounded Approved-Contract Amendment Closure: Go — Closed
T19-CR-01..16: Closed
Residual P0/P1/P2: 0/0/0
Task 19: Ready to resume Implementation Stage from amended v2 checkpoint
Task 18: unchanged and closed
Task 20 contract: Not started
Task 20 implementation: Not started
```

implementation尚未完成；下一阶段唯一为 **TASK19 IMPLEMENTATION — FINAL ACCEPTANCE RESIDUAL CLOSURE FROM V2 CONTRACT BASELINE**。
不得第二次full contract review、寻找`T19-CR-17`、扩大scope或提前开始Task 20。
