# Task 20 — v0.2 Integration and Release Readiness 精确合同

**Contract 状态：Approved — Go as amended by Governance Amendments A1 + A2 + A3。**
**Implementation 状态：REPAIRED AND CLOSURE-VALIDATED。**
**Full Contract Review：NO-GO — Permanently Closed。**
**Full Implementation Review：NO-GO — CONSUMED — quota 0。**

本记录是 Task 20 的冻结合同、C1–C4 targeted fix history、post-closure governance
status history 与 Governance Amendment A4。合同治理现为 **Complete**，Task20
implementation 已为 **REPAIRED AND CLOSURE-VALIDATED**；A4 只修正 final post-closure
user-document truth-sync 的程序性 scope，不修改 semantic contract。历史 exact
three-document contract checkpoint 继续保留；A4 自身仍只修改三份 governance 文件，
并授权下一阶段 final sync 使用 exact7。本记录不创建源码、测试、CLI、examples、exports、
依赖或版本变化，也不授权任何 release action。

## 1. Authority、baseline 与治理

本合同服从以下权威，冲突时必须先修订治理文件，不得在实现中猜测：

1. `AGENTS.md`；
2. `SPEC.md`；
3. `IMPLEMENTATION_PLAN.md`；
4. `docs/decisions/v02-roadmap-contract.md`；
5. `docs/decisions/task13-full-workflow-static-html-cli-contract.md`；
6. `docs/decisions/task15-binary-risk-validation-contract.md`；
7. `docs/decisions/task16-data-quality-leakage-contract.md`；
8. `docs/decisions/task17-preloan-eligibility-strategy-contract.md`；
9. `docs/decisions/task18-post-loan-early-warning-lifecycle-contract.md`；
10. `docs/decisions/task19-explainability-champion-challenger-governance-contract.md`。

本合同的 implementation baseline 为：

```text
135cee1dd1fb45d1c7d38ab91249a689940b0caf
```

该 baseline 上 Task 19 已完成、Task 20 尚未开始是历史 contract-baseline 状态，package
version 当时为 `0.1.0`。本合同只冻结 Task 20，不改写 Task 19 的历史治理语义、76-key error registry、38-entry
owner registry、92-key direction registry、result schemas 或 review history。

Task 20 的唯一一次开放式 Full Contract Review 已完成，历史 verdict 为
`NO-GO — Permanently Closed`；该 review 永久关闭，第二次 Full Contract Review
被禁止。C1–C4 targeted fixes 已完成，bounded contract closure 已对精确集合
`T20-CR-01..15` 判定 `PASS`，15 个 finding 均已 `CLOSED`，0 个保持开放。本状态同步
不重开历史 review、不创建新的 finding、不改变任何 normative contract semantics；
Task20 implementation 仍须从本 approved contract checkpoint 之后开始。

### 1.1 Historical Full Contract Review and C1/C2/C3/C4 finding ledger

The historical review remains an immutable record: `P0=0`, `P1=15`, `P2=0`, with
findings `T20-CR-01..15`. The current ledger is:

| Finding | Severity | Title | Current status | Targeted wave |
|---|---|---|---|---|
| `T20-CR-01` | P1 | CLI score-validation Path A absent | `CLOSED` | C1 |
| `T20-CR-02` | P1 | Governance dependency matrix incomplete | `CLOSED` | C1 |
| `T20-CR-03` | P1 | Raw DataFrame carrier boundary contradictory | `CLOSED` | C1 |
| `T20-CR-04` | P1 | `positive_label` is an open type escape hatch | `CLOSED` | C1 |
| `T20-CR-05` | P1 | Warning/limitation token semantics under-specified | `CLOSED` | C1 |
| `T20-CR-06` | P1 | Policy JSON tuple mapping incomplete | `CLOSED` | C2 |
| `T20-CR-07` | P1 | Temporal and duration lexical grammar incomplete | `CLOSED` | C2 |
| `T20-CR-08` | P1 | Resource counting and PNG gates incomplete | `CLOSED` | C2 |
| `T20-CR-09` | P1 | API-specific validation precedence incomplete | `CLOSED` | C3 |
| `T20-CR-10` | P1 | Report section provenance incomplete | `CLOSED` | C3 |
| `T20-CR-11` | P1 | Plot acquisition matrix incomplete | `CLOSED` | C3 |
| `T20-CR-12` | P1 | Report title/output path semantics incomplete | `CLOSED` | C3 |
| `T20-CR-13` | P1 | Version source allowlist incomplete | `CLOSED` | C4 |
| `T20-CR-14` | P1 | Hatchling artifact inclusion incomplete | `CLOSED` | C4 |
| `T20-CR-15` | P1 | CHANGELOG allowlist incomplete | `CLOSED` | C4 |

The targeted fix waves are complete: C1 covers `T20-CR-01..05`, C2 covers
`T20-CR-06..08`, C3 covers `T20-CR-09..12`, and C4 covers `T20-CR-13..15`.
Bounded Contract Review Closure reviewed the exact set `T20-CR-01..15` and recorded
`15 CLOSED / 0 NOT CLOSED / 0 OPEN` with verdict `PASS`. A second Full Contract Review
remains forbidden. The next stage is **TASK20 — PHASED IMPLEMENTATION KICKOFF** after
this approved contract checkpoint is committed.

## 2. Exact title、目标与边界

正式名称固定为：

```text
Task 20 — v0.2 Integration and Release Readiness
```

Task 20 只负责四类工作：

- **orchestration**：按本合同调用 Tasks 15–19 已批准 public APIs；
- **adaptation**：验证封闭 JSON carrier、构造既有 Task 17/18 config；
- **static presentation**：消费 frozen result 和 approved result-only Figures，生成
  Markdown/HTML 静态报告；
- **release-readiness integration**：兼容性、文档、examples、CI、distribution 和
  clean-install 证据。

Task 20 不拥有 score algorithm、validation metric、condition truth、policy/action、
warning/alert/lifecycle、governance comparison、drift/stability、promotion 或任何
业务决策算法。Task 20 不执行真实审批、账户操作、客户联系、催收、通知、部署、外部
写入或发布。

下列能力明确排除：通用 DSL、`eval`/`exec`、callable/script/plugin、YAML/TOML、
通用配置框架、dashboard/interactive app、server/scheduler、数据库、持久化 run
manager、多表关系引擎、分布式 backend、新 mandatory dependency、自动模型训练、
自动策略优化、自动 winner/promotion、因果结论、反欺诈、实际 PyPI/GitHub release。

## 3. Upstream DAG、路径和 exact-call matrix

### 3.1 DAG

Task 20 的唯一上游拓扑为：

```text
Task 15 validate_binary_risk ───────────────┐
Task 16 audit_data_quality ─────────────────┼─> Task 17 / Task 18 ─> Task 19
                                            └───────────────────────┘      │
                                                                           v
                                                                  Task 20 integration
```

Task 17 和 Task 18 不互相依赖；Task 18 只在其 config 使用模型分数时可选消费 Task 15。
Task 20 不调用 private `_condition_kernel`，不复制 Task 16 comparison/truth logic，
不把 Task 16 改造成 policy engine。Task 16 仍是 diagnostic/audit owner；Task 20 只
把一次已批准的 `DataAuditResult` 传给需要它的 owner。

### 3.2 Three primary paths

| Path | Key | Enabled by | Owner call | Independent boundary |
|---|---|---|---|---|
| A | `score_validation` | `V02WorkflowRequest.score_validation is not None` | `validate_binary_risk` exactly once | 需要自己的 target/config/prediction source；不要求 policy、warning 或 governance |
| B | `preloan` | `V02WorkflowRequest.preloan is not None` | `simulate_decision_strategy` exactly once | 可无 Path A；只消费 caller DataFrame score 或已传入的 Task 15 result；不生成 warning |
| C | `postloan` | `V02WorkflowRequest.postloan is not None` | `monitor_lifecycle` exactly once | 可无 Path A/B；不消费 Task 17 action；不执行 notification |

`audit` 是一个独立、可选的 diagnostic handoff，不计入三条 primary path。它由
`V02WorkflowRequest.audit is not None` 显式启用，调用 `audit_data_quality` 一次；其
result 可只读传给 Path B/C，并可供 Task 19 diagnostic evidence 使用。Task 20 不因
启用 B/C 自动运行 audit。

`governance` 是独立的 optional final step，由 `V02WorkflowRequest.governance is not
None` 显式启用；只在上游已完成后调用 `evaluate_governance` 一次。它不自动运行，
不因存在 Task 15–18 result、candidate 或 policy 而隐式启用。

For the C1 CLI adapter, the exact Path A enable predicate is:

```text
score_cli_enabled :=
    exactly_one(--external-ranking-score-column, --external-event-probability-column)
    and --target is present
    and --positive-label-type is present
    and --positive-label-value is present
    and --score-validation-mask-column is present
```

The Python API predicate remains the request-carrier predicate in the table above; the
CLI predicate is a closed adapter subset and does not change the Python signature. The
CLI never scans the input to discover a target, score, mask, label, direction, provenance,
features or exclusions. If the predicate is false while either score-column option is
present, the CLI raises the existing `sharper task20: cli_argument` error and makes no
owner call. If neither score-column option is present, Path A is disabled and the existing
Path B/Path C/audit selection rules remain in force.

When governance is enabled, the Task 19 dependency matrix is frozen as follows. “Required”
means the workflow must produce and pass the typed owner result before the sole governance
call; “optional” means an absent diagnostic reference is legal.

| Governance consumer | Source path | Exact owner result / locator | Required? | Legal empty behavior |
|---|---|---|---|---|
| candidate family `model` | A `score_validation` | `BinaryRiskValidationResult`, whole result, `source_candidate_key=None` | required when the policy declares a model candidate | no model candidate may be bound to an empty tuple |
| candidate family `strategy` | B `preloan` | `DecisionStrategyResult`, `source_candidate_key=strategy_key` | required when the policy declares a strategy candidate | no strategy candidate may be bound to an empty tuple |
| candidate family `warning_scenario` | C `postloan` | `LifecycleMonitoringResult`, exact `scenario_key` in `monitoring_summary` | required when the policy declares a warning candidate | no warning candidate may be bound to an empty tuple |
| diagnostic evidence ref `task15` | A `score_validation` | `BinaryRiskValidationResult`, whole result | optional | absent reference is omitted from the Task 19 tuple |
| diagnostic evidence ref `task16` | `audit` | `DataAuditResult`, diagnostic only; never a directional candidate | optional | absent reference is omitted from the Task 19 tuple |
| diagnostic evidence ref `task17` | B `preloan` | `DecisionStrategyResult`, result-owned trace/summary locator | optional | absent reference is omitted from the Task 19 tuple |
| diagnostic evidence ref `task18` | C `postloan` | `LifecycleMonitoringResult`, result-owned episode/scenario locator | optional | absent reference is omitted from the Task 19 tuple |

The candidate rows are interpreted only through the Task 19 approved family/source
mapping; Task 16 remains diagnostic-only. A governance policy must still contain the
Task 19-required champion candidate, so a governance-only request with no A/B/C owner
result is structurally invalid even if it carries structured declarations. When a required
owner result is absent, Task 20 raises `sharper task20: governance_dependency_missing`
before any owner call. When the matrix is satisfied, `evaluate_governance` is called exactly
once with the actual non-empty owner tuples and never with placeholders. This is a local
governance gate; the four API-specific precedence tables are frozen in §9.5.

### 3.3 Exact call order and carriers

预检通过后，workflow 的 domain-call 顺序固定为：

1. `audit_data_quality(data, reference=..., roles=..., config=...)`，仅在 audit enabled；
2. `validate_binary_risk(data, target, positive_label=..., config=..., estimator=...,
   external_predictions=..., features=..., exclude_columns=...)`，仅在 Path A enabled；
3. `simulate_decision_strategy(data, config, risk_validation=score_result,
   data_audit=audit_result)`，仅在 Path B enabled；
4. `monitor_lifecycle(data, config, risk_validation=score_result,
   data_audit=audit_result)`，仅在 Path C enabled；
5. `evaluate_governance(policy, risk_validations=(score_result,) or (),
   data_audits=(audit_result,) or (), decision_strategies=(preloan_result,) or (),
   lifecycle_monitorings=(postloan_result,) or (), model_attributions=...,
   prediction_profiles=..., performance_evidence=...)`，仅在 governance enabled。

The exact call trace is recorded as a tuple of these function names in `V02WorkflowResult`.
No domain function is called by reporting or CLI. Every enabled call is made once per
successful workflow invocation; there is no retry. If a call raises, the already-called
prefix is not retried, later calls are not attempted, no partial result is returned, and
the original owner error remains the failure cause.

Allowed direct consumers are: `V02WorkflowResult`, `generate_v02_report`, the Task 20 CLI
adapter, and tests. No other module may call an upstream owner through a Task 20 private
shortcut.

## 4. Frozen public API

Task 20 adds exactly **9** public symbols, in this order, after the current v0.1/Tasks
15–19 `__all__` suffix:

```text
V02ScoreValidationRequest
V02AuditRequest
V02PreLoanRequest
V02PostLoanRequest
V02GovernanceRequest
V02WorkflowRequest
V02WorkflowResult
run_v02_workflow
generate_v02_report
```

The public dataclass count is exactly **7**: the six request/config carriers and
`V02WorkflowResult`. The public function count is exactly **2**: `run_v02_workflow` and
`generate_v02_report`.

The count is 9 because the first 7 names are dataclasses and the last 2 names are the
workflow runner and report generator. They are defined as follows:

```python
# src/sharper/v02_workflow.py
@dataclass(frozen=True)
class V02ScoreValidationRequest:
    target: str
    config: BinaryRiskValidationConfig
    positive_label: str | int | bool | np.generic | None = None
    estimator: ClassifierMixin | None = None
    external_predictions: ExternalRiskPredictions | None = None
    features: tuple[str, ...] | None = None
    exclude_columns: tuple[str, ...] = ()

@dataclass(frozen=True)
class V02AuditRequest:
    reference: pd.DataFrame | None = None
    roles: DataAuditRoles | None = None
    config: DataAuditConfig | None = None

@dataclass(frozen=True)
class V02PreLoanRequest:
    config: DecisionStrategyConfig

@dataclass(frozen=True)
class V02PostLoanRequest:
    config: LifecycleMonitoringConfig

@dataclass(frozen=True)
class V02GovernanceRequest:
    policy: GovernancePolicy
    model_attributions: tuple[GovernanceAttributionEvidence, ...] = ()
    prediction_profiles: tuple[GovernancePredictionProfile, ...] = ()
    performance_evidence: tuple[GovernancePerformanceEvidence, ...] = ()

@dataclass(frozen=True)
class V02WorkflowRequest:
    data: pd.DataFrame
    score_validation: V02ScoreValidationRequest | None = None
    audit: V02AuditRequest | None = None
    preloan: V02PreLoanRequest | None = None
    postloan: V02PostLoanRequest | None = None
    governance: V02GovernanceRequest | None = None

@dataclass(frozen=True)
class V02WorkflowResult:
    contract_version: Literal["task20-integration-v1"]
    enabled_paths: tuple[str, ...]
    path_status: pd.DataFrame
    call_trace: tuple[str, ...]
    score_validation: BinaryRiskValidationResult | None
    data_audit: DataAuditResult | None
    preloan: DecisionStrategyResult | None
    postloan: LifecycleMonitoringResult | None
    governance: GovernanceResult | None
    warnings: tuple[str, ...]
    limitations: tuple[str, ...]

def run_v02_workflow(request: V02WorkflowRequest) -> V02WorkflowResult: ...

# src/sharper/v02_reporting.py
def generate_v02_report(
    result: V02WorkflowResult,
    output_path: str | Path,
    *,
    title: str = "Sharper v0.2 Integration Report",
    format: Literal["markdown", "html"] = "markdown",
    overwrite: bool = True,
) -> ReportArtifact: ...
```

Dataclass field order, defaults, keyword-only boundaries and return types are frozen. The
request dataclasses are shallow-frozen input carriers; constructors do not execute owner
validation. `V02WorkflowRequest.data` is the primary raw DataFrame carrier for the current
workflow input. `V02AuditRequest.reference` is the sole secondary, audit-only reference
DataFrame carrier; it is passed only as `reference` to `audit_data_quality` and never to
score validation, pre-loan, post-loan, governance or reporting. A reference can be supplied
only with an enabled `V02AuditRequest`; there is no separate reference path. The same object
is legal as both `data` and `reference`: it is borrowed twice read-only, and no identity
claim is emitted. The caller retains ownership of both frames. The result is newly allocated
and does not retain `request`, either raw frame, a raw model, estimator, callable, credential,
open file handle, manager, private kernel object or mutable cache. It may hold only the frozen
owner result objects listed above; those objects remain governed by their own contracts.
`ReportArtifact` is the existing Task 13 result type and is not a new symbol.

`positive_label` is a closed Task 15 input domain, not an open escape hatch. Its exact public
annotation is `str | int | bool | np.generic | None`; `np.generic` is accepted only as the
Task 15 normalization carrier and is not exported as a Task 20 alias. After one
`np.generic.item()` normalization, the only accepted runtime values are exact built-in
`str`, exact built-in `bool`, or exact built-in `int` excluding bool. Python subclasses of
these built-ins are rejected; a NumPy scalar is accepted only when normalization yields one
of those exact built-ins. Floats, bytes, datetimes, tuples, containers, arbitrary instances
and callables are rejected. Type-aware equality applies, so `True` is not `1`. `None` uses
Task 15's canonical-label rule only: canonical boolean `{False, True}` resolves to `True`,
canonical integer `{0, 1}` resolves to `1`, and every other two-class label set requires
an explicit label. Task 20 does not coerce labels or infer them from data.

The workflow requires at least one of A/B/C/audit/governance. A disabled primary path is
represented by `None` in the request and cannot carry path-specific fields. A path request
must not be supplied while its path is disabled. A workflow result is either complete or
raises; it never returns a skipped partial run.

## 5. Task 20 result schema and path states

`V02WorkflowResult` has exactly 11 fields shown above. `path_status` is a newly allocated
`RangeIndex` DataFrame with these exact columns and dtypes:

```text
path_key:string, enabled:boolean, status:string, reason:string
```

It has exactly five rows in this order:

```text
score_validation, audit, preloan, postloan, governance
```

Task 20 path status vocabulary has exactly two values: `completed`, `not_requested`.
`enabled=True` pairs only with `completed`; `enabled=False` pairs only with
`not_requested`. The path reason vocabulary has exactly two values: `completed` and
`path_not_requested`, with the same pairing. Missing evidence never becomes a hidden
`skipped` path: owner result tables retain their owner-specific unavailable/undefined/
not-applicable/not-verifiable status. Owner failure raises and produces no workflow result.

`enabled_paths` is the fixed path-key tuple in the five-row order, excluding
`not_requested` paths. `call_trace` uses the exact function names in the order actually
called. `warnings` and `limitations` use the closed token contract in §5.1; Task 20 does
not invent, translate or forward domain warning text. The result retains no raw
row/entity/value payload and no unbounded nested object.

### 5.1 Closed warning and limitation token contract

Task 20 constructs only these exact token forms:

```text
task20.warning.<owner>.<source>
task20.limitation.<owner>.<source>
```

`<owner>` is exactly one of `task15`, `task16`, `task17`, `task18`, `task19`. For Tasks
15–18, `<source>` must be one of the following closed inventories, in the owner-defined
order shown; an owner result tuple is never converted to arbitrary text:

| Owner | Warning source inventory | Limitation source inventory |
|---|---|---|
| Task 15 | `duplicate_index`, `duplicate_rows`, `missing_target_rows_excluded`, `external_fit_not_verifiable`, `duplicate_thresholds_removed`, `duplicate_gain_fractions_removed`, `large_input` | `random_or_group_validation_not_time_safe`, `entity_isolation_not_checked`, `time_validation_not_general_feature_audit`, `external_fit_not_verifiable`, `ranking_probability_order_may_differ`, `probability_metrics_unavailable`, `calibration_diagnostic_only`, `partial_validation_maturity`, `single_class_validation_fold`, `observed_association_not_causal` |
| Task 16 | `large_input`, `duplicate_scan_skipped`, `unique_inspection_skipped`, `category_levels_truncated`, `missing_patterns_truncated`, `collinearity_columns_truncated`, `insufficient_drift_rows`, `point_in_time_not_verifiable` | `in_memory_single_process`, `structural_identifier_evidence_only`, `association_not_causation`, `target_proxy_false_positive_possible`, `caller_declared_time_provenance`, `no_automatic_leakage_repair`, `budget_limited_evidence` |
| Task 17 | `strategy`, `source`, `mapping`, `outcome`, `constraint`, `resource` | `simulated_actions_not_executed`, `historical_comparison_not_causal`, `model_expectation_not_observed`, `outcome_support_limited`, `custom_score_provenance_caller_declared` |
| Task 18 | `input`, `time`, `source`, `scenario`, `episode`, `event`, `lifecycle`, `resource` | `offline_monitoring_not_executed`, `historical_comparison_not_causal`, `caller_defined_states_and_alert_levels`, `external_score_semantics_caller_declared`, `right_censored_event_horizons`, `peer_baseline_is_descriptive`, `entity_linkage_depends_on_prepared_input` |

Task 19 warning sources are exactly its approved ten-table `finding_key` families:

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

Task 19 limitation sources are exactly:
`source_unavailable`, `source_undefined`, `source_not_verifiable`,
`support_not_comparable`, `insufficient_support`, `maturity_not_comparable`,
`snapshot_unverified`, `alignment_unverified`, `time_unverified`,
`common_support_unverified`, `insufficient_bootstrap_support`, `zero_denominator`,
`single_class`, `operation_not_applicable`, and `diagnostic_only`.

The source inventories are matched after owner validation; missing owner evidence does not
create a Task 20 token. The final path order is `audit` (Task 16), `score_validation`
(Task 15), `preloan` (Task 17), `postloan` (Task 18), `governance` (Task 19). Within each
owner, the owner tuple order is preserved. Duplicate tokens are removed by stable
first-occurrence order separately within `warnings` and within `limitations`; a token in
one channel does not deduplicate the same source in the other channel. No whitespace,
uppercase, raw arbitrary message, exception message, `repr`, or unlisted source is legal.
An empty channel is exactly `()`. Python, CLI and report output use these same final tuples;
reports render result tuples only and never recollect owner messages.

## 6. JSON policy/warning adapter

JSON adaptation is private and lives in `src/sharper/v02_json.py`. It is not a public DSL,
does not add a public symbol, and does not execute rules or warnings. The adapter converts
JSON bytes/text/file content into the two frozen owner config types only:

```text
policy schema:  task20.policy.v1  -> V02PreLoanRequest(DecisionStrategyConfig)
warning schema: task20.warning.v1 -> V02PostLoanRequest(LifecycleMonitoringConfig)
```

The JSON schema count is exactly **2**. Each top-level document is an object with required
`schema_version` and the exact owner fields below. JSON object field order is not semantic;
array order is semantic and is preserved when constructing owner tuples.

### 6.1 Policy JSON

The policy top-level closed field set is exactly:

```text
schema_version, strategy_key, strategy_version, effective_from, expires_at,
evaluation_time, rules, default_action_name, unknown_action_name,
action_role_mapping, constraints, ranking_score_column, ranking_score_direction,
historical_action_column, historical_action_mapping, historical_policy_version,
exposure_column, loss_fraction, action_assumptions, exposure_unit,
segment_columns, time_slice_column
```

The value fields map one-to-one to the same-named `DecisionStrategyConfig` fields after
the exact array-to-tuple conversions in the C2 mapping matrix below. No JSON object is
accepted as an alternative representation for a tuple field. No new operator, role,
phase, metric or action meaning is introduced. Required owner fields are required in JSON;
nullable owner fields accept JSON `null`; omitted fields are allowed only where the owner
dataclass has the corresponding frozen default.

### 6.1.1 Policy field mapping matrix

The policy has exactly 21 owner fields plus `schema_version`. The table is the complete
carrier inventory; “required” refers to JSON key presence, while “owner empty” records
whether the mapped Task 17 tuple may be empty after parsing.

| JSON field | Exact Task 17 type | Canonical JSON shape | Position semantics | Required / owner empty | Order / duplicate rule |
|---|---|---|---|---|---|
| `strategy_key` | `str` | string | scalar | required / not applicable | scalar |
| `strategy_version` | `str` | string | scalar | required / not applicable | scalar |
| `effective_from` | `datetime` | C2 datetime string | strategy start | required / not nullable | scalar |
| `expires_at` | `datetime \| None` | C2 datetime string or `null` | exclusive strategy end | required / `null` allowed | scalar |
| `evaluation_time` | `datetime` | C2 datetime string | strategy evaluation instant | required / not nullable | scalar |
| `rules` | `tuple[DecisionRule, ...]` | array of exact rule objects | array index is caller rule order | required / `[]` maps to `()` but owner accepts empty rules | preserve; no dedupe |
| `default_action_name` | `str` | string | scalar | required / not applicable | scalar |
| `unknown_action_name` | `str` | string | scalar | required / not applicable | scalar |
| `action_role_mapping` | `tuple[tuple[str, role], ...]` | array of arrays, inner arity **2**: `[action_name:str, action_role:closed role]` | index 0 action key, index 1 role | required / `[]` maps to `()` but owner rejects an unmapped default/unknown action | preserve caller action order; duplicate action keys forbidden |
| `constraints` | `tuple[DecisionConstraint, ...]` | array of exact constraint objects | array index is caller constraint order | optional default / `[]` allowed | preserve; duplicate `constraint_key` forbidden by owner |
| `ranking_score_column` | `str \| None` | string or `null` | scalar column name | optional default / `null` allowed | scalar |
| `ranking_score_direction` | `Literal["higher_risk", "lower_risk"] \| None` | closed string or `null` | scalar | optional default / `null` allowed | scalar; owner requires column pairing |
| `historical_action_column` | `str \| None` | string or `null` | scalar column name | optional default / `null` allowed | scalar |
| `historical_action_mapping` | `tuple[tuple[object, str], ...]` | array of arrays, inner arity **2**: `[raw_scalar, action_name:str]` | index 0 raw Task16-allowlisted scalar, index 1 mapped action key | optional default / `[]` allowed only when no historical column is declared | preserve; duplicate raw scalar identities forbidden |
| `historical_policy_version` | `str \| None` | string or `null` | scalar | optional default / `null` allowed | scalar |
| `exposure_column` | `str \| None` | string or `null` | scalar column name | optional default / `null` allowed | scalar |
| `loss_fraction` | `float \| str \| None` | finite JSON number, string column name, or `null` | scalar; number is constant, string is column | optional default / `null` allowed | scalar; bool is not a number |
| `action_assumptions` | `tuple[tuple[str, float, float], ...]` | array of arrays, inner arity **3**: `[action_name:str, value:finite number, cost:finite non-negative number]` | indexes 0/1/2 are action, value, cost | optional default / `[]` allowed when no assumptions are declared | preserve; duplicate action keys forbidden and a non-empty set must cover the owner mapping |
| `exposure_unit` | `str \| None` | string or `null` | scalar | optional default / `null` allowed | scalar |
| `segment_columns` | `tuple[str, ...]` | array of strings | array index is caller segment order | optional default / `[]` allowed | preserve; duplicate column names forbidden |
| `time_slice_column` | `str \| None` | string or `null` | scalar column name | optional default / `null` allowed | scalar |

The affected tuple-of-tuples field count is exactly **3**: `action_role_mapping`,
`historical_action_mapping`, and `action_assumptions`. Each has its own fixed inner
arity and position semantics in the matrix; no object-field names are inferred.

`role` in the table is exactly `selected`, `rejected`, `review`, `request_information`,
`limited`, or `other`. Every inner array is required to have the stated arity; an object,
short array, long array, `null` member, wrong member type, or JSON boolean in a numeric
position is rejected before owner construction. `action_role_mapping` has no legal empty
accepted policy because the required default and unknown actions must map; the parser may
construct the empty tuple only as an intermediate shape that the owner rejects.

Nested policy carriers are fixed as follows:

| Nested field | Exact owner type | Canonical JSON shape | Empty / order rule |
|---|---|---|---|
| `rules[*]` | `DecisionRule` | object with exactly `rule_key`, `phase`, `priority`, `condition`, `action_name`, `stop_on_hit`, `enabled`, `effective_from`, `expires_at`, `description_key` | all fields present; rule array order preserved |
| `rules[*].condition` | `StrategyCondition` | object with exactly `kind`, `operator`, `left_kind`, `left`, `right_kind`, `right`, `children` | exact owner shape; no omitted child inventory |
| `rules[*].condition.children` | `tuple[StrategyCondition, ...]` | array of condition objects | array order preserved; owner arity rules apply (`and/or` at least 2, `not` exactly 1, atomic empty) |
| `rules[*].condition.right` for `in`/`not_in` | exact built-in tuple of Task16-allowed literals | array of literals | array order preserved; member duplicate policy is the owner condition contract, with no Task20 dedupe |
| `rules[*].condition.right` for `between` | exact built-in tuple of two Task16-allowed literals | array of exactly 2 literals | lower/upper order preserved and interpreted as the owner closed interval |
| `constraints[*]` | `DecisionConstraint` | exact constraint object | constraint array order preserved |

Thus `action_role_mapping`, `historical_action_mapping`, and `action_assumptions` each
have one canonical array-of-arrays shape. They never accept an object form. All other
tuple/closed-sequence policy fields have the single array form shown above; JSON arrays are
converted to Python tuples without sorting, deduplication, or dictionary conversion.

### 6.1.2 Policy round-trip and duplicate semantics

For every valid policy, parsing produces one and only one `DecisionStrategyConfig` value
in the owner type domain. A valid JSON fixture is compared field by field against a
hand-authored `DecisionStrategyConfig`, including tuple nesting, scalar types, `null`,
condition child order, mapping order, and duplicate-sensitive owner behavior. Python and
CLI carriers using the same semantic fields produce equal owner config values; JSON text
formatting and object member order are not part of equality.

Duplicate JSON object keys remain `json_duplicate_key`. Duplicate entries in arrays are
not parser-level object duplicates: `action_role_mapping`, `historical_action_mapping`,
`action_assumptions`, `segment_columns`, `rules`, and `constraints` retain owner semantics.
The Task 17 owner rejects duplicate action keys, duplicate historical raw-scalar identity,
duplicate assumption action keys, duplicate segment columns, duplicate rule keys/priorities,
and duplicate constraint keys. Task 20 performs no sort or deduplication. Unknown fields
remain `json_unknown_field`.

### 6.2 Warning JSON

The warning top-level closed field set is exactly:

```text
schema_version, monitoring_key, monitoring_version, analysis_as_of,
entity_column, observation_time_column, available_time_column,
condition_feature_columns, event_time_column, positive_event_key,
prediction_horizon, horizon_end_inclusive, recent_window, history_window,
history_start_inclusive, expected_observation_interval, period_unit, time_zone,
scenarios, reference_scenario_key, alert_level_ranks, states, default_state_key,
unknown_state_key, allowed_transitions, adverse_state_keys, cure_state_keys,
cohort_time_column, cohort_column, peer_group_columns, peer_reference_start,
peer_reference_end, ranking_score_column, ranking_score_direction,
exposure_column, loss_fraction, observed_loss_column,
observed_loss_available_time_column, observed_loss_is_mature_snapshot,
segment_columns, time_frequency
```

The value fields map one-to-one to the same-named `LifecycleMonitoringConfig` fields.
`scenarios` uses exact `WarningScenario` fields; nested `rules` use exact
`EarlyWarningRule` fields; nested conditions use exact `MonitoringCondition` fields;
`states` use exact `LifecycleState` fields. Durations use the integer-microsecond carrier
in the C2 temporal matrix and are converted to exact `timedelta` values. Datetimes use
the exact six-fraction-digit strings defined below. No automatic timezone, current date,
locale or period conversion is performed.

### 6.2.1 Task20 JSON temporal mapping matrix

The matrix covers every named datetime-like and duration-like field carried by the two
JSON schemas and has exactly **15 rows**. Task 17 JSON datetime values construct built-in `datetime.datetime`; Task 18
JSON datetime values also construct built-in `datetime.datetime`, which is within its
approved `datetime | pandas.Timestamp` target union. Aware JSON values are restricted to
UTC so the standard-library mapping is deterministic.

| JSON schema | JSON field | Owner field / target type | Nullability | JSON representation |
|---|---|---|---|---|
| policy | `effective_from` | Task17 `DecisionStrategyConfig.effective_from: datetime` | required, non-null | C2 datetime grammar, naive or UTC-aware |
| policy | `expires_at` | Task17 `DecisionStrategyConfig.expires_at: datetime \| None` | required, nullable | C2 datetime grammar or `null` |
| policy | `evaluation_time` | Task17 `DecisionStrategyConfig.evaluation_time: datetime` | required, non-null | C2 datetime grammar, same naive/aware family as strategy window |
| policy | `rules[*].effective_from` | Task17 `DecisionRule.effective_from: datetime \| None` | required nested key, nullable | C2 datetime grammar or `null` |
| policy | `rules[*].expires_at` | Task17 `DecisionRule.expires_at: datetime \| None` | required nested key, nullable | C2 datetime grammar or `null` |
| warning | `analysis_as_of` | Task18 `LifecycleMonitoringConfig.analysis_as_of: datetime \| pandas.Timestamp` | required, non-null | C2 datetime grammar, naive or UTC-aware |
| warning | `prediction_horizon` | Task18 `prediction_horizon: timedelta \| None` | required, nullable | integer microseconds or `null`, strictly positive when present |
| warning | `recent_window` | Task18 `recent_window: timedelta` | required, non-null | integer microseconds, strictly positive |
| warning | `history_window` | Task18 `history_window: timedelta` | required, non-null | integer microseconds, at least `recent_window` |
| warning | `expected_observation_interval` | Task18 `expected_observation_interval: timedelta \| None` | required, nullable | integer microseconds or `null`, strictly positive when present |
| warning | `peer_reference_start` | Task18 `peer_reference_start: datetime \| None` | optional default, nullable | C2 datetime grammar or `null` |
| warning | `peer_reference_end` | Task18 `peer_reference_end: datetime \| None` | optional default, nullable | C2 datetime grammar or `null` |
| warning | `scenarios[*].rules[*].cooldown` | Task18 `EarlyWarningRule.cooldown: timedelta` | required nested key, non-null | integer microseconds, non-negative |
| warning | `scenarios[*].rules[*].effective_from` | Task18 `EarlyWarningRule.effective_from: datetime \| None` | required nested key, nullable | C2 datetime grammar or `null` |
| warning | `scenarios[*].rules[*].expires_at` | Task18 `EarlyWarningRule.expires_at: datetime \| None` | required nested key, nullable | C2 datetime grammar or `null` |

Condition `right` values are not an additional temporal carrier. JSON strings in condition
objects remain exact built-in strings; Task20 JSON does not encode Task17/18 date or
datetime condition literals because the public condition fields have no type tag. Python
requests may use the owner-approved literal families directly. This restriction prevents
a condition string from having two possible owner types.

### 6.2.2 Exact datetime grammar and semantics

The only accepted datetime lexical grammar is:

```text
naive_datetime := YYYY-MM-DD "T" HH ":" MM ":" SS "." microsecond6
utc_datetime   := naive_datetime "Z"
datetime      := naive_datetime | utc_datetime
```

The exact regular-language widths are year 4 digits, month/day/hour/minute/second each
2 digits, and fractional seconds exactly 6 digits. `T` and `Z` are uppercase ASCII. No
space separator, lowercase `t`/`z`, offset (`+00:00`, `-00:00`, or other), timezone name,
date-only form, omitted fraction, or leading/trailing string whitespace is accepted.
The grammar is matched before construction; construction then rejects month 00/13,
invalid day-of-month and leap-day combinations, hour outside 00–23, minute outside
00–59, or second outside 00–59. A six-digit fraction is mapped directly to Python
microseconds; no nanosecond value is accepted.

No-suffix values construct naive `datetime`; `Z` values construct `datetime` with
`timezone.utc`. Within one Task17 config, strategy and rule temporal values must remain
all naive or all UTC-aware, matching Task17's awareness check. Within one Task18 config,
all config boundaries follow the same awareness as `analysis_as_of`; a UTC-aware config
must set `time_zone` to the exact string `"UTC"`, while a naive config must set
`time_zone` to JSON `null`. Rule boundaries follow the same rule. No local timezone,
offset normalization, DST lookup, or silent localization is performed.

### 6.2.3 Exact duration grammar and semantics

The only duration carrier is a JSON integer representing **microseconds**. JSON booleans,
JSON floating-point numbers, strings, `null` where a duration is required, non-finite
values, and negative values are rejected. The carrier is lossless for Python
`timedelta` because the owner type has microsecond resolution. `prediction_horizon`,
`recent_window`, `history_window`, and `expected_observation_interval` use the field
constraints in the matrix; `cooldown` permits zero. For `history_window`, the constructed
value must be greater than or equal to `recent_window`. No seconds, milliseconds, ISO
duration, clock-time, signed text, or fractional-number representation is accepted.

### 6.3 Closed parser semantics

The accepted JSON domain is standard JSON object/array/string/boolean/null and finite
number, plus the one exact datetime grammar above and integer-microsecond durations. The adapter
rejects YAML, TOML, Python literals, comments, bytes/object tags, NaN, Infinity,
`-Infinity`, callable, expression, script, template, URL, `$ref`, include chaining,
environment expansion, tilde expansion, glob expansion and path interpolation.

Duplicate object keys are rejected during parsing; last-one-wins is forbidden. Unknown
schema versions raise `sharper task20: json_schema_version`. Unknown fields raise
`sharper task20: json_unknown_field`. Unknown operators raise
`sharper task20: json_unknown_operator`. Missing required fields, wrong container types,
invalid null placement or non-finite/scalar values raise `json_structure` or
`json_scalar` according to the error registry below. A non-object root raises
`json_not_object`; malformed JSON raises `json_decode`; invalid UTF-8 raises
`json_encoding`.

For C2 temporal validation, an object/array where a scalar is required, or a missing/null
value where the matrix requires a non-null scalar, is `json_structure`. A scalar with the
wrong JSON type, boolean in an integer/number position, wrong lexical form, invalid calendar
range, forbidden timezone suffix, awareness mismatch, non-integer duration, invalid duration
sign, or cross-field duration relation is `json_scalar`. Temporal checks occur in this order:
exact string grammar, calendar construction, timezone-family rule, then owner field
relationship. All these scalar failures intentionally use the existing stable
`sharper task20: json_scalar` prefix; no new error key is required. The C2 error inputs are
registered for and ordered by the C3 API-specific precedence tables in §9.5.

When reading a JSON file, the path is a literal filesystem path. The adapter performs no
expansion or network access. File open/read failures remain `OSError`; content failures
are `ValueError` with the stable Task 20 prefix. No Task 19 canonical encoder or
fingerprint is reused: semantic JSON equality is required for Python/CLI parity, not byte
identity.

## 7. Error registry and validation precedence

Task 20-owned validation errors have the exact prefix `sharper task20: <error_key>` and
the following **28-key** closed registry:

| Key | Reachable predicate |
|---|---|
| `invalid_request_type` | public workflow/report request is not the exact approved carrier type |
| `request_requires_primary_path` | all five path carriers are disabled |
| `request_path_input_conflict` | a path-specific carrier is present in a disabled or contradictory path position |
| `request_raw_carrier` | request/result carrier contains an open handle, callable, credential or prohibited arbitrary object |
| `json_not_object` | JSON root is not an object |
| `json_decode` | JSON text is syntactically malformed |
| `json_encoding` | accepted bytes cannot be decoded as UTF-8 |
| `json_duplicate_key` | one JSON object contains a duplicate key |
| `json_schema_version` | version is absent, unknown or not the exact schema value |
| `json_unknown_field` | object contains a field outside its closed schema |
| `json_unknown_operator` | condition operator is outside the owner-approved set |
| `json_structure` | required field, object/array shape or nullable placement is invalid |
| `json_scalar` | scalar is non-finite, wrong JSON type or invalid temporal/duration scalar |
| `json_budget` | any parser budget is exceeded before owner config construction |
| `policy_mapping` | policy JSON cannot be mapped one-to-one to `DecisionStrategyConfig` |
| `warning_mapping` | warning JSON cannot be mapped one-to-one to `LifecycleMonitoringConfig` |
| `owner_call_contract` | Task 20 would call an owner with a wrong carrier or forbidden cross-path result |
| `result_contract` | returned owner result type or assembled Task 20 result violates the frozen schema |
| `report_format` | report format is outside `markdown`/`html` |
| `report_result` | report input is not a complete `V02WorkflowResult` |
| `report_asset_budget` | report figure/asset projection exceeds the frozen report budget |
| `report_title` | report title is not the exact accepted string input |
| `report_path` | report output path is not an accepted file path or is an existing directory |
| `report_overwrite` | report overwrite argument is not the exact accepted boolean input |
| `cli_argument` | CLI command has an invalid or mutually exclusive argument combination |
| `cli_spec_required` | CLI path selection requires a policy/warning JSON carrier that was not supplied |
| `cli_output` | CLI output destination or output mode violates the frozen command contract |
| `governance_dependency_missing` | enabled governance policy requires a Task 15/16/17/18 owner result that no enabled path can produce |

Task 20 does not wrap or rename a domain owner’s stable `ValueError`; owner errors retain
their own exact prefix and cause. Task 20 errors are only for integration, adaptation,
report, and CLI predicates listed above. File reads, input data reads, report writes and
rollback failures are `OSError` (with existing `FileExistsError` preserved for
`overwrite=False`), never `ValueError`. Unexpected internal exceptions are not converted
to caller errors.

### 7.1 Shared error and ownership boundary

The four public surfaces have separate precedence tables in §9.5. The only shared
sub-order is: a Task20-owned validation failure stops the current surface; a literal file
read failure remains the `OSError` boundary; an enabled owner call is never retried; an
owner exception retains its owner prefix and cause; and no later report/CLI step runs
after an earlier failure. Disabled path carriers are absent in Python workflow requests;
the CLI rule for present path-specific options is frozen in the CLI table. No partial
workflow result, partial report, silently repaired carrier or duplicate owner call is
permitted.

The local governance dependency gate remains a workflow-owned predicate: after the
workflow request/path checks and before any enabled owner call, it checks the §3.2 matrix,
raises `governance_dependency_missing` when a required result cannot be produced, and
permits the single governance owner call only when the actual non-empty typed tuples are
available. Its position relative to JSON, report and CLI predicates is defined only by
the corresponding surface table.

## 8. JSON and report resource gates

The resource registry has exactly **12** rows. Every maximum is inclusive. The value at
the maximum passes; the first value above it raises the row's error key. No gate truncates,
samples, drops, skips, retries, or silently omits input. The expected max and max+1 values
in future tests are literal test constants, not values read from implementation helpers.

| Resource key | Maximum and unit | Scope and counting algorithm | Phase | Error key | Max/max+1 proof |
|---|---|---|---|---|---|
| `json_bytes_per_document` | 5,000,000 raw UTF-8 bytes | each policy or warning document independently; a string is UTF-8 encoded, bytes are counted as supplied, file input is counted before decode | JSON parser preflight | `json_budget` | exact 5,000,000 passes; 5,000,001 first fails |
| `json_nesting_depth` | 16 container levels | root object/array is depth 1; entering each nested object/array adds 1; scalar members add 0 | JSON parser preflight | `json_budget` | depth 16 passes; depth 17 first fails |
| `object_members_per_object` | 256 members | count members of each individual JSON object; no document aggregate; duplicate keys are rejected before this budget | JSON parser preflight | `json_budget` | one object with 256 unique keys passes; 257 first fails |
| `array_items_per_array` | 4,096 items | count items in each individual JSON array; no document aggregate | JSON parser preflight | `json_budget` | one array with 4,096 items passes; 4,097 first fails |
| `condition_nodes_per_document` | 128 condition nodes | count every `StrategyCondition` or `MonitoringCondition` object, including each atomic/and/or/not node, across the document's condition trees; child arrays and literal-member arrays are not extra nodes | JSON parser preflight | `json_budget` | 128 nodes passes; 129 first fails |
| `policy_rules` | 100 rule entries | count only outer `policy.rules` entries; one `DecisionRule` object is one rule; nested condition nodes are counted only by the condition gate | JSON parser preflight | `json_budget` | 100 passes; 101 first fails |
| `policy_constraints` | 50 constraint entries | count only outer `policy.constraints` entries; one `DecisionConstraint` object is one constraint | JSON parser preflight | `json_budget` | 50 passes; 51 first fails |
| `warning_scenarios` | 10 scenario entries | count outer `warning.scenarios` entries; one `WarningScenario` object is one scenario | JSON parser preflight | `json_budget` | 10 passes; 11 first fails |
| `warning_rules_per_scenario` | 50 rules | count `scenario.rules` independently for each scenario; it is not an aggregate across scenarios | JSON parser preflight | `json_budget` | any scenario with 50 passes; 51 first fails |
| `warning_states` | 50 state entries | count outer `warning.states` entries; one `LifecycleState` object is one state | JSON parser preflight | `json_budget` | 50 passes; 51 first fails |
| `report_figure_asset_entries` | 9 entries | count one Task20-generated plot entry per acquired Figure plus its associated PNG asset path; Markdown, HTML and non-plot files do not count; only entries selected by the later frozen acquisition matrix and enabled result paths count | report plot pre-acquisition | `report_asset_budget` | 9 planned entries pass; 10 first fails before Figure creation |
| `report_png_bytes` | 64,000,000 bytes | aggregate encoded byte size of all Task20-generated PNG files in one report transaction; count exact staged file bytes, not source Figure memory, Markdown, HTML or other assets | staged output materialization | `report_asset_budget` | aggregate 64,000,000 passes; first staged total above it fails |

The nesting examples are exact: `{"x": 0}` has depth 1; a chain of 16 nested object
containers has depth 16 and passes; the same chain with 17 containers fails. The rule is
identical for arrays and mixed object/array paths. Object duplicate-key rejection remains
`json_duplicate_key` under the existing parser precedence; member and array budgets apply
to otherwise structurally valid JSON.

The first-fail order for the ten JSON parser gates is the table order above. The figure
entry gate runs after the enabled result-only plot acquisition plan is known and before
any Figure is created. The PNG-byte gate runs after each PNG is written into Task13-style
temporary staging: it adds the exact encoded file size to the running report total, and if
the new total exceeds 64,000,000 it closes acquired Figures, removes staging, restores
the prior output transaction state, and raises `sharper task20: report_asset_budget`.
No final report or asset replacement occurs before this staged gate succeeds.

Task20 transport limits do not replace owner limits. The 100 policy rules and 50 policy
constraints equal the Task 17 maxima; the 10 warning scenarios, 50 rules per scenario,
and 50 states equal the corresponding Task 18 maxima; the 128 condition-node limit is
the Task 16/17/18 node limit. JSON nesting, object-member and array-item limits are
transport budgets and do not change the owner condition depth, tuple cardinality, or
semantic validation. After mapping, Task 17/18 whole-config validation remains mandatory;
owner errors are not converted into a Task20 budget pass. Action-role mappings,
historical mappings, assumptions, transitions and other owner collections retain their
owner hard limits even though they have no separate row in this 12-row Task20 registry.

The ten parser gates use `json_budget`; the figure-entry and staged-PNG gates use
`report_asset_budget`. The registry deliberately does not create a new error key for each
resource. Resource keys remain internal stable identifiers attached to the direct error
predicate. This C2 resource ordering is a local input inventory; its cross-API placement
is the JSON sub-order in §9.5 and does not alter the resource meanings.

Future direct acceptance must construct max and max+1 fixtures for all 12 rows, prove the
depth root rule, condition-node definition, per-object/per-array scope, per-scenario rule
scope, figure-entry pre-acquisition failure, staged aggregate PNG failure, exact owner
limit relationship, and absence of truncation or partial final output.

## 9. Static Markdown/HTML report

`generate_v02_report` supports exactly **2** formats: `markdown` and `html`. It never
accepts PDF, dashboard, JavaScript application or interactive backend. It consumes only a
complete `V02WorkflowResult` and approved result-only owner plots; it never accepts raw
DataFrame/model input and never calls `validate_binary_risk`, `audit_data_quality`,
`simulate_decision_strategy`, `monitor_lifecycle` or `evaluate_governance`.

The report has exactly **12** sections, always in this order and with the same semantic
presence in both formats:

1. `Run Context`
2. `Path Status`
3. `Score Validation`
4. `Data Audit and Leakage`
5. `Pre-loan Eligibility`
6. `Post-loan Warning`
7. `Governance`
8. `Cross-path Comparison`
9. `Reason and Override Trace`
10. `Stability and Business Evidence`
11. `Warnings and Limitations`
12. `Provenance and Release Readiness`

Disabled paths render an explicit `not_requested` marker with no fabricated metric,
claim or figure; sections are never omitted in one format and retained in the other.
The Markdown and HTML tables, status tokens, section names, ordering, numeric values,
asset links and disabled markers are semantically identical. HTML escapes every
caller-controlled text value. Markdown uses a safe renderer: no raw HTML, no template
execution, no link/image URL from caller text, and code spans/table cells escape pipe,
backtick and control characters. No raw entity, row payload, credentials, environment
value, private expanded path, model repr or arbitrary repr is rendered.

The report must preserve `observed`, `expected`, `simulated`, `association`, `drift`,
`unavailable`, `not-verifiable` and `limitation` meanings. It must not claim approved loan,
compliance, safety, certification, production readiness, causal effect, deployed model,
executed notification, executed collection or released package.

### 9.1 Figure and asset ownership

Task 20 reuses Task 13 ownership rules:

- report preflight obtains Figures from result-only plot APIs immediately before the first
  staging side effect;
- the report generator owns every acquired Figure and closes each exactly once in success
  and failure paths after acquisition;
- the workflow and CLI do not close a Figure they did not acquire;
- report generation follows the exact nine-slot registry in §9.3: a true slot predicate
  calls its named result-only public plot function exactly once, and a false predicate
  calls it zero times;
- plot calls are presentation calls over frozen results, never domain recomputation; no
  owner result is mutated and no global matplotlib state is changed.

The output asset directory is the sibling `<output-stem>_assets`; each slot has a fixed
filename `plot-001.png` through `plot-009.png` bound to its slot ordinal. A slot that is
disabled or unavailable produces no file and leaves its ordinal gap; acquired later slots
retain their fixed names. No UUID, timestamp or hash-randomized filename is allowed.
`overwrite=False` fails before staging if report or asset directory exists.
`overwrite=True` uses Task 13’s staging/backup/rollback protocol: stage report and assets,
backup existing targets, atomically replace, remove backups only after both replacements
succeed, restore backups and clean staging on any failure. A failed transaction leaves no
new partial report/assets.

### 9.2 TASK20_REPORT_SECTION_PROVENANCE_MATRIX

The report provenance matrix has exactly 12 rows. Section ordinal, title, presence
condition, source fields, source columns, row order, empty behavior, plot eligibility and
warning/limitation ownership are frozen below. Every row has recomputation=NO.
The disabled marker is not_requested. The enabled-empty marker is
empty_result:<section_key>; an enabled empty table is never rendered as a fabricated zero.
Only section 11 renders V02WorkflowResult.warnings and limitations.

| Ordinal / title | Enabled condition | Exact source fields and columns | Order | Enabled empty / plot |
|---|---|---|---|---|
| 1 Run Context | always | V02WorkflowResult.contract_version, enabled_paths, call_trace; no owner table | listed field order; tuple order preserved | no empty source; no plot |
| 2 Path Status | always | V02WorkflowResult.path_status: path_key, enabled, status, reason | exactly score_validation, audit, preloan, postloan, governance | zero rows is report_result; no plot |
| 3 Score Validation | score_validation is not None | scalar fields validation_mode, prediction_scope, score_source, score_direction, probability_provenance, input_n_rows, eligible_n_rows, predicted_n_rows, evaluable_n_rows, requested_threshold_count, actual_threshold_count, observed_loss_maturity_mode, observed_loss_mature_n, observed_loss_excluded_n; Task15 metrics columns scope, fold_id, metric, statistic, value, at_threshold, status, reason, n_rows, n_positive, n_negative; business_metrics columns segment_kind, segment_value, metric, value, status, reason, n_rows, n_evaluable_rows, n_observed_loss_mature_rows, unit | scalar order then owner table order | empty_result:score_validation; plot slots 1–4 |
| 4 Data Audit and Leakage | data_audit is not None | scalar fields n_rows, n_columns, reference_n_rows, reference_n_columns; Task16 dataset_profile columns side, n_rows, n_columns, profiled_column_count, declared_feature_count, feature_status, feature_reason, duplicate_row_count, duplicate_row_rate, duplicate_row_status, duplicate_row_reason, duplicate_index_count, duplicate_index_rate, duplicate_index_status, duplicate_index_reason, memory_usage_bytes, finding_key; point_in_time_profile columns side, scope, column, evaluated_count, violation_count, not_verifiable_count, status, reason, finding_key; missingness_drift columns column, reference_present, current_present, reference_n, current_n, reference_missing_count, current_missing_count, reference_missing_rate, current_missing_rate, absolute_rate_change, relative_rate_change, new_all_missing, recovered, count_status, count_reason, rate_status, rate_reason, reference_count_status, reference_count_reason, current_count_status, current_count_reason, reference_rate_status, reference_rate_reason, current_rate_status, current_rate_reason, absolute_change_status, absolute_change_reason, relative_change_status, relative_change_reason, finding_key | scalar order then owner table order | empty_result:data_audit; no plot |
| 5 Pre-loan Eligibility | preloan is not None | scalar fields strategy_key, strategy_version, input_n_rows, decided_n_rows, unavailable_n_rows, requested_rule_count, active_rule_count, requested_constraint_count; Task17 row_decisions columns row_position, decision_status, decision_reason, base_action_name, final_action_name, applied_rule_key, matched_rule_count, unknown_rule_count, overlap_rule_count, conflict_rule_count, override_applied, historical_mapping_status; business_summary columns scope_type, scope_column, scope_ordinal, time_slice_ordinal, action_key, action_role, metric_key, metric_value, numerator, denominator, support_n_rows, unit, status, reason, finding_key; constraint_summary columns constraint_key, metric, operator, threshold, action_name, action_role, actual_value, status, reason, support_n, gap, violation_magnitude, finding_key | scalar order then owner table order | empty_result:preloan; no plot |
| 6 Post-loan Warning | postloan is not None | scalar fields monitoring_key, monitoring_version, input_n_rows, entity_count, evaluable_observation_count, requested_scenario_count, requested_rule_count, active_rule_count, requested_state_count; Task18 monitoring_summary columns scope_key, scope_position, scenario_key, rule_key, metric, metric_value, numerator, denominator, support_n, support_unit, mature_n, censored_n, unit, status, reason, finding_key; lifecycle_summary columns scope_key, scope_position, from_state_key, to_state_key, metric, metric_value, numerator, denominator, support_n, support_unit, unit, status, reason, finding_key; observation_history columns row_position, entity_position, observation_time, observation_status, observation_reason, primary_scenario_key, primary_rule_key, primary_alert_level, primary_alert_rank, active_rule_count, emitted_notification_count, maturity_status, effective_state_key, effective_state_rank, state_status, state_reason | scalar order then owner table order | empty_result:postloan; no plot |
| 7 Governance | governance is not None | scalar fields governance_key, governance_version, analysis_as_of, candidate_count, comparison_pair_count, criterion_count, explanation_count, source_snapshot_status, entity_alignment_status, evidence_time_status; Task19 governance_summary columns candidate_position, candidate_family, declared_role, declared_state, source_task, source_result_position, source_candidate_position, source_snapshot_status, entity_alignment_status, evidence_time_status, criterion_count, available_criterion_count, unavailable_criterion_count, not_verifiable_criterion_count, attribution_count, prediction_drift_count, performance_stability_count, recommendation_count, human_review_required_count, status, reason, finding_key; recommendations columns pair_position, champion_candidate_position, challenger_candidate_position, candidate_family, recommendation, recommendation_basis, hard_veto, human_review_mode, human_review_required, minimum_comparable_criteria, criteria_available_n, criteria_unavailable_n, criteria_better_n, criteria_worse_n, criteria_tied_n, required_incomplete_n, support_comparable, status, reason, finding_key | scalar order then candidate position then pair position | empty_result:governance; plot slots 5–9 |
| 8 Cross-path Comparison | governance is not None | candidate_comparisons columns pair_position, champion_candidate_position, challenger_candidate_position, candidate_family, criterion_position, source_task, source_table, metric_key, scope_key, scope_position, champion_value, challenger_value, delta, comparison_outcome, support_comparable, status, reason, finding_key; governance_evaluations columns pair_position, criterion_position, criterion_role, required_for_promotion, priority, comparison_outcome, comparable, counts_toward_minimum, blocks_promotion, directional_contribution, evidence_time_status, status, reason, finding_key; recommendations columns listed in row 7 | pair position then criterion position | empty_result:cross_path_comparison; no plot |
| 9 Reason and Override Trace | preloan is not None or postloan is not None or governance is not None | Task17 row_decisions columns row_position, decision_status, decision_reason, base_action_name, final_action_name, applied_rule_key, matched_rule_count, unknown_rule_count, overlap_rule_count, conflict_rule_count, override_applied, historical_mapping_status; Task17 rule_evaluations columns row_position, rule_key, phase, priority, rule_order, path_status, truth, status, reason, is_applied, is_overlap, is_conflict; Task18 rule_evaluations columns row_position, entity_position, observation_time, scenario_key, scenario_order, rule_key, rule_order, alert_level, alert_rank, path_status, truth, true_streak, false_streak, episode_status, notification_status, status, reason, finding_key; Task18 state_history columns row_position, entity_position, observation_time, candidate_state_key, candidate_state_rank, candidate_state_priority, effective_state_key, effective_state_rank, matching_state_count, status, reason, finding_key; Task19 explanations columns explanation_position, explanation_key, candidate_position, candidate_family, method, source_ref_position, source_task, source_result_position, source_table, source_registry_position, source_fingerprint, feature_key, relation, priority, evidence_time_status, source_status, source_reason, status, reason, finding_key | owner table order, then owner row/declaration order | empty_result:reason_override_trace; no plot |
| 10 Stability and Business Evidence | score_validation or preloan or postloan or governance is not None | Task15 business_metrics columns listed in row 3; Task19 prediction_drift columns candidate_position, reference_profile_position, current_profile_position, prediction_kind, scope_key, scope_position, metric, prediction_tvd, direction, status, reason, finding_key; performance_stability columns candidate_position, reference_evidence_position, current_evidence_position, evaluation_scope, scope_key, scope_position, metric, reference_value, current_value, delta, direction, reference_support_n, current_support_n, reference_assignment_mechanism, current_assignment_mechanism, reference_common_support, current_common_support, status, reason, finding_key | owner table order | empty_result:stability_business; no plot |
| 11 Warnings and Limitations | always | V02WorkflowResult.warnings and limitations | warnings tuple then limitations tuple | empty_result:warnings_limitations when both empty; no plot |
| 12 Provenance and Release Readiness | always | Task20 contract_version and call_trace; Task15 score_source, score_direction, probability_provenance; Task16 provenance columns provenance_key, value_type, numeric_value, text_value, count_value, boolean_value, status, reason; Task17/18 provenance columns provenance_key, provenance_value, status, reason, finding_key; Task19 provenance columns provenance_position, provenance_key, provenance_value, status, reason, finding_key | Task20 fields then owner order Task15→16→17→18→19 | optional empty_result:provenance; no plot |

Rows 3–10 with a false enabled condition render not_requested and read no source table.
Every true condition validates the named owner table schema before rendering. Markdown and
HTML use the same source fields, values, markers and order; their only difference is output
serialization. Cross-path Comparison reads only Task19 comparison/evaluation/recommendation
facts; it never computes a new delta, compares Task17 with Task18, selects a winner or
creates a business score. Reason and Override Trace reads stored owner trace rows and does
not interpret free-form text or create a new reason vocabulary. Stability and Business
Evidence displays stored Task15 business metrics and Task19 drift/stability rows; it never
calculates a trend, aggregates a metric or compares windows.

### 9.3 TASK20_REPORT_PLOT_SLOT_REGISTRY

The result-only plot inventory has exactly PLOT_SLOT_COUNT=9 slots, equal to and not above
the C2 report_figure_asset_entries limit. Tasks 16, 17 and 18 provide no approved public
result-only plot function and contribute no slot. A true predicate calls its named public
plot function exactly once; a false predicate calls it zero times and renders
plot_unavailable:<slot_key>. No placeholder Figure is created.

| Slot | Owner function | Kind | Required table and exact predicate | Path | Filename | Section |
|---|---|---|---|---|---|---|
| 1 | Task15 plot_binary_risk_validation | gains | score_validation.gains has an overall row with missing fold_id, finite actual_fraction, capture_status=available, missing capture_reason, finite capture | score_validation | plot-001.png | Score Validation |
| 2 | Task15 plot_binary_risk_validation | lift | score_validation.gains has an overall row with missing fold_id, finite actual_fraction, lift_status=available, missing lift_reason, finite lift | score_validation | plot-002.png | Score Validation |
| 3 | Task15 plot_binary_risk_validation | calibration | score_validation.calibration has a non-empty-bin overall row with status=available, missing reason, finite mean_predicted_probability and observed_event_rate | score_validation | plot-003.png | Score Validation |
| 4 | Task15 plot_binary_risk_validation | threshold | score_validation.threshold_analysis has an overall row with missing fold_id and predicted_positive_rate, sensitivity, precision, specificity all available, reason-missing and finite | score_validation | plot-004.png | Score Validation |
| 5 | Task19 plot_model_governance | importance | governance.model_attributions has status=available, non-missing feature_key and finite value | governance | plot-005.png | Governance |
| 6 | Task19 plot_model_governance | candidate_comparison | governance.candidate_comparisons has status=available, non-missing criterion_position and finite delta | governance | plot-006.png | Governance |
| 7 | Task19 plot_model_governance | prediction_drift | governance.prediction_drift has status=available, non-missing scope_key and finite prediction_tvd | governance | plot-007.png | Governance |
| 8 | Task19 plot_model_governance | performance_stability | governance.performance_stability has status=available, non-missing scope_key and finite delta | governance | plot-008.png | Governance |
| 9 | Task19 plot_model_governance | governance_summary | governance.governance_summary has status=available, non-missing candidate_position and non-missing available_criterion_count | governance | plot-009.png | Governance |

Slot order is numeric order above and does not depend on owner result arrival, dictionary
iteration or filename sorting. The report generator owns and closes each acquired Figure
exactly once after the Task13 acquisition point on success, plot failure, staged PNG write
failure, backup/commit/cleanup failure and rollback failure. The CLI never closes a Figure
owned by the report generator. Disabled or unavailable slots leave their fixed ordinal gap;
only acquired slots create PNG files and links. The report figure count is the number of
acquired slot entries, and the aggregate staged PNG byte gate remains the C2 rule.

### 9.4 Report signature, title, path and transaction contract

The exact Task20 signature is:

    def generate_v02_report(
        result: V02WorkflowResult,
        output_path: str | Path,
        *,
        title: str = "Sharper v0.2 Integration Report",
        format: Literal["markdown", "html"] = "markdown",
        overwrite: bool = True,
    ) -> ReportArtifact: ...

output_path accepts only str or pathlib.Path under the Task13 runtime rule; arbitrary
os.PathLike objects are rejected. An output path naming an existing directory raises
sharper task20: report_path. A missing parent is created only after preflight. title accepts
only str; cleaning is exactly Task13: replace every str.splitlines boundary with one ASCII
space, trim leading/trailing whitespace, and use the frozen default when empty. Other title
characters remain content; HTML escapes them and Markdown uses the cleaned value as the H1
content, exactly as in Task13. format accepts only markdown or html; it controls serialization
and the literal caller suffix is preserved without suffix inference or rewriting. overwrite
accepts only bool.

ReportArtifact is the existing Task13 type with path=Path(output_path), format equal to the
requested literal, and title equal to the cleaned title. overwrite=False and an existing
report target or sibling asset directory raise the existing FileExistsError. overwrite=True
moves an existing report file, asset directory, asset file or partial asset directory to
the exact Task13 backup path before replacement; no existing path is silently deleted.
Any pre-existing staging or backup path raises the existing FileExistsError.

The exact transaction is: validate result → title → format → output path → overwrite type
and final-target policy → residual staging/backup policy → validate section sources →
preflight nine slots and the C2 figure-entry budget → render in memory → create the parent
directory → acquire eligible Figures immediately before the first staging side effect →
create staging paths → write fixed-slot PNGs and report staging → check aggregate staged PNG
bytes → backup existing targets → commit assets then report → remove backups in Task13 order
→ return ReportArtifact. Any failure after Figure acquisition closes every acquired Figure
exactly once and applies the Task13 staging, backup and rollback rules. No final replacement
occurs before the staged PNG gate succeeds. No failure returns ReportArtifact. Staging,
backup and commit failures leave no half-written final bundle; cleanup or compensation
failure follows Task13's explicitly retained recoverable backup/path state and never silently
deletes it.

### 9.5 API-specific precedence contracts

CR-09 is frozen as four surface-specific tables, not one universal chain. Within a table,
the first matching stage is the first observable error; equal-stage checks use the listed
left-to-right order. These tables consume C1/C2 predicates and the preceding C3 matrices
without changing owner semantics.

#### A. run_v02_workflow — 9 stages

| Stage | First check | Error/result |
|---|---|---|
| 1 | exact V02WorkflowRequest type and raw-carrier safety | invalid_request_type, then request_raw_carrier |
| 2 | primary-path requirement, path presence and cross-path conflicts | request_requires_primary_path, then request_path_input_conflict |
| 3 | score request structure, positive-label domain and score-source pairing | owner_call_contract; owner calls=0 |
| 4 | governance dependency matrix | governance_dependency_missing; owner calls=0 |
| 5 | enabled request-carrier and call-plan structure | owner_call_contract; owner calls=0 |
| 6 | enabled owner calls in §3.3 order, exactly once | owner error retains owner prefix and cause |
| 7 | returned owner result exact types and required schemas | result_contract |
| 8 | assembled result, path rows and token tuples | result_contract |
| 9 | successful return | complete result only |

Workflow does not parse JSON, inspect report title/path, parse CLI syntax or acquire report
assets.

#### B. JSON policy/warning adapter — 13 stages

| Stage | First check | Error |
|---|---|---|
| 1 | literal file acquisition | OSError |
| 2 | raw UTF-8 byte count for the individual document | json_budget |
| 3 | UTF-8 decoding | json_encoding |
| 4 | JSON syntax | json_decode |
| 5 | duplicate object keys | json_duplicate_key |
| 6 | root object | json_not_object |
| 7 | exact schema_version | json_schema_version |
| 8 | closed fields and required field presence | json_unknown_field, then json_structure |
| 9 | object/array/null shape | json_structure |
| 10 | scalar JSON type and finite-number domain | json_scalar |
| 11 | C2 depth/member/array/condition/rule/scenario/state gates in registry order | json_budget |
| 12 | datetime lexical/calendar/timezone, duration sign/unit and owner relationships | json_scalar |
| 13 | operators, one-to-one mapping and typed config construction | json_unknown_operator, policy_mapping or warning_mapping |

For valid JSON, stage 8 precedes stage 11, so an unknown field wins over an oversized
object; stage 11 precedes typed owner construction. After the raw-byte gate, the ten JSON
resource sub-gates retain C2 order: depth, object members, array items, condition nodes,
policy rules, policy constraints, warning scenarios, per-scenario warning rules, warning
states. Policy validation completes before warning validation when both documents are
supplied.

#### C. generate_v02_report — 14 stages

| Stage | First check | Error/result |
|---|---|---|
| 1 | exact V02WorkflowResult type and 11-field schema | report_result |
| 2 | title type and Task13 cleaning input | report_title |
| 3 | format literal | report_format |
| 4 | output path type and existing-directory check | report_path |
| 5 | overwrite type | report_overwrite |
| 6 | final-target conflict with overwrite=False | FileExistsError |
| 7 | residual staging/backup paths | FileExistsError |
| 8 | section source result types, columns and order | report_result |
| 9 | fixed-slot predicates and nine-entry projection | report_asset_budget |
| 10 | in-memory section/Markdown/HTML rendering | report_result |
| 11 | parent/staging creation and Figure acquisition | OSError or report_result |
| 12 | staged PNG writes and aggregate byte gate | OSError or report_asset_budget |
| 13 | backup, asset-first commit, report commit and cleanup | OSError with cause |
| 14 | exact ReportArtifact construction | success only |

Invalid result plus invalid title returns report_result; invalid title plus invalid format
returns report_title; invalid format plus invalid output path returns report_format; invalid
output path plus overwrite conflict returns report_path. A report failure never reruns
workflow.

#### D. sharper v02-run CLI — 13 stages

| Stage | First check | Error/exit |
|---|---|---|
| 1 | Typer usage, missing option, unknown option or primitive option type | Typer usage error, exit 2 |
| 2 | literal INPUT path and csv/xlsx suffix | cli_argument, exit 2 |
| 3 | INPUT data file read | OSError, exit 3 |
| 4 | literal reference-input read | OSError, exit 3 |
| 5 | policy JSON read then warning JSON read | OSError, exit 3 |
| 6 | policy JSON adapter then warning JSON adapter | Task20 JSON error, exit 2 |
| 7 | output option and closed CLI combinations | cli_output or cli_argument, exit 2 |
| 8 | score metadata, typed positive label and Path A predicate | cli_argument, exit 2 |
| 9 | policy/warning/audit/path consistency | cli_spec_required, cli_argument or request_path_input_conflict, exit 2 |
| 10 | typed request construction and workflow call | Task20/owner error, exit 2; report calls=0 on failure |
| 11 | workflow exactly once and complete result | owner/Task20 error, exit 2 |
| 12 | report exactly once after workflow success | report ValueError=2, filesystem OSError=3 |
| 13 | success/error translation | success=0, unexpected internal=70 |

CLI stage 1 is the only syntax parser stage. Raw JSON parser exceptions are never exposed.
Task20 and owner validation errors map to exit 2; filesystem errors including FileExistsError
map to exit 3; unexpected internal exceptions map to the fixed internal error line and exit
70. A present score-source option without the complete C1 Path A predicate is a conflict,
not implicit enablement. If workflow fails, report call count is zero; if report fails,
workflow is not rerun.

The complete 28-key reachability matrix is:

| Error key | Surface/stage | Direct predicate |
|---|---|---|
| invalid_request_type | workflow/1 | request is not exact approved type |
| request_requires_primary_path | workflow/2; CLI/9 | all primary/audit/governance carriers absent |
| request_path_input_conflict | workflow/2; CLI/9 | disabled path carries path-specific input |
| request_raw_carrier | workflow/1 | prohibited handle/callable/credential/object |
| json_not_object | JSON/6 | decoded root is not object |
| json_decode | JSON/4 | syntax malformed |
| json_encoding | JSON/3 | bytes are not UTF-8 |
| json_duplicate_key | JSON/5 | object key repeats |
| json_schema_version | JSON/7 | schema version absent/unknown/wrong |
| json_unknown_field | JSON/8 | field outside closed schema |
| json_unknown_operator | JSON/13 | operator outside owner set |
| json_structure | JSON/8–9 | missing, wrong container, arity or null placement |
| json_scalar | JSON/10–12 | wrong scalar, non-finite number, temporal or duration scalar |
| json_budget | JSON/2,11 | C2 byte or parser limit exceeded |
| policy_mapping | JSON/13 | policy cannot construct Task17 config |
| warning_mapping | JSON/13 | warning cannot construct Task18 config |
| owner_call_contract | workflow/3,5 | wrong enabled owner carrier or forbidden cross-path result |
| result_contract | workflow/7–8 | owner/result assembly schema invalid |
| report_format | report/3 | format outside markdown/html |
| report_result | report/1,8,10–11 | result or frozen source structure invalid |
| report_asset_budget | report/9,12 | nine-entry or staged PNG budget exceeded |
| report_title | report/2 | title is not str |
| report_path | report/4 | output path is not str or Path, or names a directory |
| report_overwrite | report/5 | overwrite is not bool |
| cli_argument | CLI/2,7–8 | closed CLI value or combination invalid |
| cli_spec_required | CLI/9 | selected policy/warning carrier missing |
| cli_output | CLI/7 | output option or mode violates CLI contract |
| governance_dependency_missing | workflow/4 | governance-required owner result cannot be produced |

Every row has one named future direct test. The same key on multiple surfaces denotes the
same semantic predicate family and uses the surface-specific stage above.

## 10. CLI contract

The CLI command count is exactly **1** new command:

```text
sharper v02-run INPUT --output OUTPUT
```

It is a new opt-in subcommand. Existing `sharper analyze` and root `--version` semantics
are unchanged. The exact options are:

```text
INPUT                         required literal .csv or .xlsx path
--output / -o PATH            required literal report path
--format [markdown|html]      default markdown
--policy-json PATH            optional; enables Path B
--warning-json PATH           optional; enables Path C
--audit / --no-audit          default false; explicitly enables Task 16 audit
--reference-input PATH        optional literal .csv/.xlsx path; valid only with --audit
--target COLUMN               required when Path A score CLI is enabled
--external-ranking-score-column COLUMN
                              mutually exclusive with the probability-column option
--external-event-probability-column COLUMN
                              mutually exclusive with the ranking-column option
--ranking-direction [higher_risk|lower_risk]
                              required only with the ranking-column option
--probability-provenance [predict_proba|fold_safe_calibrated|external_declared]
                              required only with the probability-column option
--score-validation-mask-column COLUMN
                              required when Path A score CLI is enabled; strict boolean mask
--positive-label-type [str|int|bool]
                              required when Path A score CLI is enabled
--positive-label-value VALUE  required with --positive-label-type; no type inference
--score-test-size FLOAT       Path A fixed stratified-holdout test size; default 0.20
--score-random-state INT      Path A random state; default 42
--overwrite / --no-overwrite  default true
```

The CLI now enables the closed external-prediction subset of Path A. The exact Path A
option inventory is the four score-source options (`--external-ranking-score-column` or
`--external-event-probability-column`, `--ranking-direction` or
`--probability-provenance`), `--target`, `--score-validation-mask-column`, the two typed
positive-label options, and the two score split options. The score path is fixed to
`stratified_holdout`; it does not expose estimator construction, group/time modes,
fold-cutoff/maturity metadata, feature selection or exclusion lists. Governance remains
Python-only because Task 19 structured declaration carriers are not represented by the
closed pure-data CLI carriers.

The complete Path A caller-input mapping is:

| CLI input | Task 15 / Task 20 mapping |
|---|---|
| `INPUT` | one `pd.DataFrame`, passed as `V02WorkflowRequest.data` and Task 15 `df` |
| `--target` | Task 15 `target` exactly; no target scan |
| ranking column | physical input-column values at true mask positions become `ranking_scores` |
| probability column | physical input-column values at true mask positions become `event_probabilities` |
| `--score-validation-mask-column` | strict built-in boolean column; true physical positions become `row_positions`, false positions become the one holdout fit set |
| `--ranking-direction` | Task 15 `ranking_direction`, required only for ranking scores |
| `--probability-provenance` | Task 15 `probability_provenance`, required only for probabilities |
| typed positive label | Task 15 `positive_label` and, on probability path, `probability_positive_label` |
| `--score-test-size` / `--score-random-state` | `BinaryRiskValidationConfig.test_size` / `.random_state` |
| fixed adapter defaults | `validation_mode="stratified_holdout"`, no estimator, `features=None`, `exclude_columns=()`, no group/time/maturity/business metadata, and one fixed external fold id |

The adapter constructs `ExternalRiskPredictions` from explicit physical positions and
values; it does not infer positions from non-null scores, infer a label from classes,
infer direction or probability semantics, or auto-detect target/score/mask columns. The
score and mask columns remain ordinary input data and are not removed or mutated. An
invalid score kind, direction, provenance, mask dtype/value/NA, label encoding or option
combination raises the exact `sharper task20: cli_argument` error. Owner-level row/split/
label validity errors retain their Task 15 error and cause.

CLI label encoding is closed: `--positive-label-type str` uses the exact UTF-8 option text
without trimming or case conversion; `int` accepts only the lexical form
`-?(0|[1-9][0-9]*)` and constructs a built-in `int`; `bool` accepts only lowercase `true`
or `false` and constructs the corresponding built-in `bool`. The type and value options
are jointly required, cannot be repeated, and do not encode `None`. Missing, extra,
malformed or cross-path label options raise `sharper task20: cli_argument`; no label
auto-inference or arbitrary-object escape hatch exists.

The CLI requires at least one of the exact Path A predicate, `--policy-json`,
`--warning-json` or `--audit`; it never silently runs an empty workflow. It loads data
through existing `load_csv`/`load_excel`, constructs the typed request, calls
`run_v02_workflow` exactly once and calls `generate_v02_report` exactly once. It never
calls an owner directly. Python Path A requests and CLI Path A requests must produce equal
constructed Task 15 external-prediction semantics; the CLI is only a closed adapter and
does not broaden the Python API.

CLI accepts only standard JSON policy/warning files and literal local filesystem paths.
It rejects inline Python, YAML/TOML, URL, environment config, templates, callable,
expression, include and path expansion. Default output has no traceback. Exit codes are
closed: `0` success, `2` caller/spec/owner validation or CLI argument error, `3` filesystem
or report I/O error, and `70` unexpected internal error. The CLI prints the stable error
message for code 2/3 and a fixed `internal error` line for code 70 without repr or traceback.

## 11. Permanent v0.1 compatibility and current release surface

The permanent compatibility manifest is the exact baseline `sharper.__all__` sequence,
including `__version__`, `load_csv`, `load_excel`, all v0.1 symbols, and the frozen Task
15–19 suffix through `plot_model_governance`. Its full sequence is recorded here so the
implementation test can compare names and relative order mechanically:

```text
__version__, load_csv, load_excel, ColumnSchema, TargetCandidate, SchemaReport,
infer_schema, DataFrameSummary, summarize_dataframe, QualityIssue, QualityReport,
check_data_quality, AnalysisRun, run_analysis, ReportArtifact,
generate_analysis_report, NumericAnalysis, CategoricalAnalysis, CorrelationAnalysis,
OutlierAnalysis, analyze_numeric_features, analyze_categorical_features,
compute_correlations, detect_outliers, GroupComparison, TargetAnalysis, compare_groups,
analyze_target_relationships, FeatureSuggestion, FeatureSuggestionReport,
FeatureDerivationResult, suggest_feature_derivations, derive_features, TrainingResult,
train_classifier, RegressionTrainingResult, train_regressor, ClassificationEvaluation,
evaluate_classifier, RegressionEvaluation, evaluate_regressor, evaluate_model, PlotResult,
PlotCollection, plot_distributions, plot_missingness, plot_correlations, plot_outliers,
plot_group_comparison, plot_target_relationships, plot_classification_evaluation,
plot_regression_evaluation, BinaryRiskValidationConfig, ExternalRiskPredictions,
BinaryRiskValidationResult, validate_binary_risk, plot_binary_risk_validation,
DataAuditRoles, ColumnAuditRule, DataAuditConfig, DataAuditResult, audit_data_quality,
StrategyCondition, DecisionRule, DecisionConstraint, DecisionStrategyConfig,
DecisionStrategyResult, simulate_decision_strategy, MonitoringCondition,
EarlyWarningRule, WarningScenario, LifecycleState, LifecycleMonitoringConfig,
LifecycleMonitoringResult, monitor_lifecycle, GovernanceEvidenceRef,
GovernanceCandidate, GovernanceCriterion, GovernanceExplanation,
GovernanceAttributionEvidence, GovernancePredictionProfile,
GovernancePerformanceEvidence, GovernanceMetadata, GovernancePolicy, GovernanceResult,
evaluate_governance, plot_model_governance
```

Task 20 uses an append-only migration: the above prefix and relative order are never
deleted, renamed or reordered. After implementation, the current release surface is that
exact prefix followed by the nine Task 20 symbols in §4. Permanent v0.1 compatibility
tests and current release-surface tests are separate suites; current-surface tests cannot
replace the permanent suite.

The following remain unchanged: `AnalysisRun`, `run_analysis`,
`generate_analysis_report`, all v0.1 defaults/signatures/errors/reports/Figure ownership,
`sharper analyze`, root `--version`, current CLI output semantics, and all prior exports.

## 12. Version and release-readiness gate

Current package version is exactly `0.1.0` throughout this Contract Definition, Full
Contract Review and contract checkpoint creation. The future metadata target is `0.2.0`.
Even after all 15 findings have targeted drafts and an approved Task20 contract checkpoint
is created, that checkpoint remains `0.1.0`. Implementation may prepare `0.2.0` metadata
only after the unique Full Contract Review remains permanently closed, all 15 original
findings pass bounded contract closure, Task20 governance status is `Approved — Go`, and
the approved Task20 contract checkpoint exists. The only authorized source edit is the
single `__version__` literal in `src/sharper/__init__.py`; no draft-stage command may change
version or exports.

Task 20 may prepare changelog/release notes, build artifacts, CI evidence, wheel/sdist,
clean-install and final release-readiness evidence after implementation gates pass. Actual
release actions remain out of scope: no commit, tag, push, upload, PyPI publication,
GitHub release, hosted docs publication or external announcement. The terminal state is
exactly:

```text
Release Ready — Not Released
```

Passing tests never changes that terminal state.

## 13. Historical C1–C3 future implementation list (superseded)

The following C1–C3 list is retained as historical evidence only. It is superseded by the
canonical C4 `TASK20_IMPLEMENTATION_TRACKED_FILE_ALLOWLIST` in §13.1 and is not an
implementation authorization by itself.

```text
src/sharper/v02_workflow.py       new independent workflow/request/result owner
src/sharper/v02_json.py          new private policy/warning JSON adapter
src/sharper/v02_reporting.py     new static report generator
src/sharper/v02_cli.py           new private CLI adapter helpers
src/sharper/cli.py               additive v02-run registration only; analyze untouched
src/sharper/__init__.py          append exactly nine approved exports only
tests/test_v02_workflow.py       Task20 request/result/call/ownership tests
tests/test_v02_json.py           closed JSON/error/budget/parity tests
tests/test_v02_reporting.py      Markdown/HTML/figure/asset/rollback tests
tests/test_v02_cli.py            command/exit/I-O/traceback tests
tests/test_public_api.py         append Task20 and permanent/current surface assertions
tests/test_distribution.py       append Task20 wheel/sdist/clean-install assertions
README.md                        implemented opt-in usage and stale Task19 status correction
docs/quickstart.md               implemented v0.2 quickstart additions
docs/analysis-guide.md           implemented integration guide additions
docs/leakage.md                  implemented v0.2 ownership/leakage notes
docs/api.md                      implemented nine-symbol API reference
docs/v02-integration-guide.md    new focused Task20 guide
docs/release-readiness.md        new focused release-readiness notes
examples/v02_score_validation.py synthetic Python Path A example
examples/v02_preloan.py          synthetic Python Path B example
examples/v02_postloan.py         synthetic Python Path C example
examples/v02_combined_report.py  synthetic combined Python/report example
examples/v02_cli_json.py         synthetic literal JSON/CLI example
pyproject.toml                   metadata/entry-point/version changes authorized here only
.github/workflows/ci.yml         existing CI matrix/gate additions only
SPEC.md                          Task20 approved/implementation/release status sync only
IMPLEMENTATION_PLAN.md           Task20 phase/status/evidence sync only
docs/decisions/task20-v02-integration-release-readiness-contract.md status/evidence only
```

That historical list has no “other necessary files” allowance. Existing Task 15–19
modules/contracts/tests, Task 13 workflow/reporting behavior, `analysis.py`, private kernel,
dependencies, lock files, generated reports, caches, `.DS_Store`, local data and unrelated
changes remain read-only/default forbidden.

The historical documentation allowlist is not the C4 canonical allowlist. There is no new
documentation framework, and the known stale README statement that Task 19–20 have not
started remains unchanged in this contract stage.

### 13.1 TASK20_IMPLEMENTATION_TRACKED_FILE_ALLOWLIST

After the approved Task20 contract checkpoint and Governance Amendment A1 checkpoint,
implementation may modify exactly the 32 tracked paths in the table below. `existing` means
the path exists in the current contract workspace; `new` means implementation may create that
exact path. There is no wildcard,
“other necessary files”, “supporting files”, “as needed” or `etc.` escape hatch.

| # | Exact path | State | Future implementation scope |
|---:|---|---|---|
| 1 | `SPEC.md` | existing | Task20 approved/implementation/release status sync only |
| 2 | `IMPLEMENTATION_PLAN.md` | existing | Task20 phase, status and evidence sync only |
| 3 | `docs/decisions/task20-v02-integration-release-readiness-contract.md` | existing | status, evidence and review-history sync only |
| 4 | `pyproject.toml` | existing | exact Hatchling sdist include/exclude keys, existing entry-point registration if required, and already-frozen dynamic-version linkage only |
| 5 | `src/sharper/v02_workflow.py` | new | independent request/result workflow owner |
| 6 | `src/sharper/v02_json.py` | new | closed policy/warning JSON adapter |
| 7 | `src/sharper/v02_reporting.py` | new | static Markdown/HTML report integration |
| 8 | `src/sharper/v02_cli.py` | new | private `v02-run` adapter helpers |
| 9 | `src/sharper/cli.py` | existing | additive `v02-run` registration only; `analyze` untouched |
| 10 | `src/sharper/__init__.py` | existing | update the single `__version__` literal after the gate and append exactly nine exports |
| 11 | `tests/test_v02_workflow.py` | new | request/result, call order, ownership and no-recomputation tests |
| 12 | `tests/test_v02_json.py` | new | closed JSON schema, mapping, budget and parity tests |
| 13 | `tests/test_v02_reporting.py` | new | 12 sections, nine plots, Markdown/HTML, assets and rollback tests |
| 14 | `tests/test_v02_cli.py` | new | command, precedence, exit, I/O and traceback tests |
| 15 | `tests/test_v02_compatibility.py` | new | permanent v0.1 compatibility manifest independent of current version |
| 16 | `tests/test_cli.py` | existing | update only root `--version` expected literals from `0.1.0` to `0.2.0`; preserve all other CLI regressions |
| 17 | `tests/test_public_api.py` | existing | append current nine-symbol surface assertions and version parity |
| 18 | `tests/test_distribution.py` | existing | exact wheel/sdist content, metadata, clean-install and smoke assertions |
| 19 | `README.md` | existing | stale Task19/20 state correction and implemented opt-in usage |
| 20 | `docs/quickstart.md` | existing | implemented v0.2 quickstart additions |
| 21 | `docs/analysis-guide.md` | existing | implemented integration guide additions |
| 22 | `docs/leakage.md` | existing | implemented v0.2 ownership/leakage notes |
| 23 | `docs/api.md` | existing | exact nine-symbol API reference |
| 24 | `docs/v02-integration-guide.md` | new | focused Task20 integration guide |
| 25 | `docs/release-readiness.md` | new | focused release-readiness notes |
| 26 | `examples/v02_score_validation.py` | new | synthetic Python Path A example |
| 27 | `examples/v02_preloan.py` | new | synthetic Python Path B example |
| 28 | `examples/v02_postloan.py` | new | synthetic Python Path C example |
| 29 | `examples/v02_combined_report.py` | new | synthetic combined Python/report example |
| 30 | `examples/v02_cli_json.py` | new | synthetic literal JSON/CLI example |
| 31 | `.github/workflows/ci.yml` | existing | existing matrix/gate additions required for Task20 release evidence |
| 32 | `CHANGELOG.md` | existing | exact Task20 user-facing changelog entry |

The existing v0.1 examples `examples/basic_analysis.py` and
`examples/baseline_modeling.py` are not modified by Task20 and are not duplicated in this
allowlist. Existing Task 15–19 modules/contracts/tests, Task 13 workflow/reporting behavior,
`analysis.py`, the private kernel, dependencies, lock files, generated reports, caches,
`.DS_Store`, local data and unrelated changes remain read-only/default forbidden. The five
new examples use synthetic data, public APIs, no PII, no Kaggle data and create no tracked
report artifact.

### 13.2 Roadmap requirement to exact future files

| Roadmap requirement | Approved future file(s) or no-change decision |
|---|---|
| workflow orchestration | `src/sharper/v02_workflow.py`, `tests/test_v02_workflow.py` |
| closed JSON policy/warning adapter | `src/sharper/v02_json.py`, `tests/test_v02_json.py` |
| static report | `src/sharper/v02_reporting.py`, `tests/test_v02_reporting.py` |
| opt-in CLI | `src/sharper/v02_cli.py`, `src/sharper/cli.py`, `tests/test_v02_cli.py` |
| nine exports | `src/sharper/__init__.py`, `tests/test_public_api.py` |
| version transition | `src/sharper/__init__.py`, `tests/test_v02_compatibility.py`, `tests/test_distribution.py`, `tests/test_cli.py` |
| permanent/current compatibility | `tests/test_v02_compatibility.py`, `tests/test_public_api.py`, `tests/test_distribution.py`, `tests/test_cli.py` |
| user-facing documentation | `README.md`, `docs/quickstart.md`, `docs/analysis-guide.md`, `docs/leakage.md`, `docs/api.md`, `docs/v02-integration-guide.md`, `docs/release-readiness.md` |
| five Task20 examples | `examples/v02_score_validation.py`, `examples/v02_preloan.py`, `examples/v02_postloan.py`, `examples/v02_combined_report.py`, `examples/v02_cli_json.py` |
| CHANGELOG | `CHANGELOG.md` |
| distribution content and clean install | `pyproject.toml`, `tests/test_distribution.py` |
| CI release gates | `.github/workflows/ci.yml` |
| contract/status governance | `SPEC.md`, `IMPLEMENTATION_PLAN.md`, `docs/decisions/task20-v02-integration-release-readiness-contract.md` |

Every allowlisted path appears in at least one row above. No roadmap requirement maps to a
path outside the 32-entry list, and no allowlisted path is approved merely because it might
be useful.

### 13.3 Version source and transition gate

The authoritative version source is the existing `src/sharper/__init__.py` field
`__version__`, currently the single literal `"0.1.0"`. `pyproject.toml` remains
`dynamic = ["version"]` and `[tool.hatch.version].path = "src/sharper/__init__.py"`; it
must not gain a second `project.version` literal or another version source. The approved
Task20 contract checkpoint remains package version `0.1.0`, even after all 15 findings have
targeted drafts and even after the checkpoint is created.

The only transition is `0.1.0 → 0.2.0`, and it is permitted only after: the unique Full
Contract Review remains permanently closed; all `T20-CR-01..15` have targeted fixes; the
bounded contract closure passes all 15 original deterministic conditions; Task20 governance
status is `Approved — Go`; and the approved Task20 contract checkpoint exists. Only then may
implementation update `src/sharper/__init__.py::__version__` from `"0.1.0"` to `"0.2.0"`.
That metadata step is an early Task20 implementation step after the checkpoint and before
release-readiness validation, not a draft-stage or release-publication action.

The root `sharper --version`, installed `sharper.__version__`, wheel metadata, sdist
`PKG-INFO`, and source-free installed package must all report `0.2.0` from this same source.
The v0.1 compatibility manifest protects API and behavior, not a permanent
`__version__ == "0.1.0"` assertion. Version `0.2.0` does not imply released, published,
tagged, pushed or uploaded; the terminal wording remains `Release Ready — Not Released`.

### 13.4 CHANGELOG contract

`CHANGELOG.md` is the existing exact tracked path authorized for the Task20 implementation.
It currently uses a top-level `# Changelog`, an introductory sentence, undated `## X.Y.Z`
headings and short hyphen bullets; it has no `Unreleased` heading. Task20 must therefore add
an undated `## 0.2.0` section above `## 0.1.0` during the post-checkpoint metadata step,
using the existing bullet style. The entry must cover the new opt-in v0.2 workflow, score /
pre-loan / post-loan integration, static Markdown/HTML report, closed JSON policy/warning
adapter, `v02-run`, governance/report integration, v0.1 compatibility preservation and
release-readiness state. It may mention metadata target `0.2.0`; it must not mention CR IDs
unless the existing user-facing convention changes.

The entry must not claim PyPI release, GitHub release, publication, upload, deployment or
production readiness. `Release Ready — Not Released` remains the only release terminal
state.

### 13.5 Allowlist change and generated-artifact gate

If implementation discovers that a tracked file outside the 32-entry allowlist is required,
it must stop and report `TASK20 IMPLEMENTATION ALLOWLIST BLOCKER`. It may not edit the file;
the only next action is a bounded governance amendment that adds the exact path and purpose
before implementation resumes. Build outputs, generated reports, PNGs, staging paths,
backups and caches are not tracked implementation files; they must be created outside the
repository or in a controlled ignored/untracked location and must not be committed.

### 13.6 Governance Amendment A1 — I6 compatibility scope correction

The I6 start gate confirmed a deterministic compatibility contradiction: the authoritative
version transition is `0.1.0 → 0.2.0`, while the existing root CLI regression tests in
`tests/test_cli.py` contain two expected `sharper 0.1.0` literals. Full pytest is required at
the I6 gate, and `src/sharper/cli.py` reads the same `sharper.__version__` source for root
`--version` output. The existing test path therefore requires a narrow scope correction
before I6 can start.

Governance Amendment A1 is a bounded implementation-allowlist amendment only. It authorizes
`tests/test_cli.py` as the sole additional tracked path and permits only the two root
`--version` expected-literal updates from `0.1.0` to `0.2.0`. Help, exit, `analyze`, option,
error, output-format and all other CLI regression assertions remain unchanged. This is not a
runtime semantic change, does not create or reopen a finding, does not reopen the unique Full
Contract Review, and does not consume the Full Implementation Review quota.

The original approved contract checkpoint remains preserved at
`4a6fec677fab5b152efc4dbf0a15c805da469bb1`. The A1 amendment changes the implementation
allowlist from 31 to 32 exact tracked paths and the I6 scope from 12 to 13 exact tracked
files. The A1-era amended I6 scope was exactly:

1. `src/sharper/__init__.py`
2. `tests/test_v02_compatibility.py`
3. `tests/test_public_api.py`
4. `tests/test_distribution.py`
5. `tests/test_cli.py`
6. `examples/v02_score_validation.py`
7. `examples/v02_preloan.py`
8. `examples/v02_postloan.py`
9. `examples/v02_combined_report.py`
10. `examples/v02_cli_json.py`
11. `.github/workflows/ci.yml`
12. `CHANGELOG.md`
13. `pyproject.toml`

Before the A1 checkpoint, Task20 remains package version `0.1.0`, I1–I5 remain complete,
I6 remains not started, release remains not released, and no implementation or test file is
modified by this amendment. After the A1 checkpoint, the next and only implementation stage
is `TASK20 IMPLEMENTATION — WAVE I6 PUBLIC SURFACE, VERSION, DISTRIBUTION AND RELEASE
READINESS`.

### 13.7 Governance Amendment A2 — I6 post-transition documentation truth scope

The I6 preflight established a bounded documentation truth conflict. Once the approved I6
transition changes package metadata from `0.1.0` to `0.2.0`, activates the nine Task20 root
exports, creates the five Task20 examples, and completes distribution and CI readiness
evidence, six I5 documents contain unconditional current-state claims that would become
false. Those documents are already inside the 32-entry global Task20 implementation
allowlist, but they were outside the A1-era 13-file I6 scope.

Governance Amendment A2 is an I6 wave-scope correction only. It does not expand the global
implementation allowlist, which remains exactly 32 paths. It changes the I6 exact scope from
13 to 19 by adding exactly these six existing allowlisted paths:

1. `README.md`
2. `docs/quickstart.md`
3. `docs/analysis-guide.md`
4. `docs/api.md`
5. `docs/v02-integration-guide.md`
6. `docs/release-readiness.md`

The six documents receive only post-transition truth-maintenance edits: current version,
root-export activation, Task20 completion, example availability, distribution/CI readiness
and release-state facts. A2 does not authorize tutorial rewrites, new API or CLI options,
JSON grammar changes, report-semantic changes, new examples, or unrelated v0.1 edits.
`docs/leakage.md` is explicitly excluded because the preflight found no unconditional
current-state claim that the I6 version/export/readiness transition would invalidate.

A2 preserves the A1 narrow authorization for `tests/test_cli.py`: only the two root version
expected literals may change from `sharper 0.1.0` to `sharper 0.2.0`. A2 changes no runtime,
public API, error, resource, schema, report or CLI contract; creates no finding; reopens no
review; and consumes no Full Implementation Review quota. The I5 checkpoint
`7af8f212312c28617f7d357a77b8eb9fb0f7a1a9` remains historically complete and accurate for
the state at which it was created. I5 is not failed, incomplete or reopened. I1–I5 remain
COMPLETE and I6 remains NOT STARTED until the A2 checkpoint exists.

The final I6 exact scope after A2 is:

1. `src/sharper/__init__.py`
2. `tests/test_v02_compatibility.py`
3. `tests/test_public_api.py`
4. `tests/test_distribution.py`
5. `tests/test_cli.py`
6. `examples/v02_score_validation.py`
7. `examples/v02_preloan.py`
8. `examples/v02_postloan.py`
9. `examples/v02_combined_report.py`
10. `examples/v02_cli_json.py`
11. `.github/workflows/ci.yml`
12. `CHANGELOG.md`
13. `pyproject.toml`
14. `README.md`
15. `docs/quickstart.md`
16. `docs/analysis-guide.md`
17. `docs/api.md`
18. `docs/v02-integration-guide.md`
19. `docs/release-readiness.md`

The A2-era exact-19 list above is historical. Governance Amendment A3 is a bounded
pre-Full-Implementation-Review interoperability repair authorization. Its mechanical root-cause
gate confirmed that the approved `path_status` contract freezes a `boolean` dtype and boolean
semantics, not built-in scalar identity. The real workflow's schema, dtype, row order, status and
reason values are contract-valid; pandas `boolean` storage returns `numpy.bool_` scalars. The I3
reporting validator instead requires `type(value) is bool`, so a real workflow result is rejected
with `sharper task20: report_result` by both report generation and the real `v02-run` path. This
is an implementation interoperability defect owned by `src/sharper/v02_reporting.py`, not a
contract defect.

A3 keeps the global implementation allowlist at exactly 32 and changes the current I6 scope from
19 to exactly 21 by adding only these existing allowlisted paths:

20. `src/sharper/v02_reporting.py` — accept the exact contract-valid boolean scalar family
    produced by the approved dtype, while rejecting non-boolean scalar domains without
    truthiness coercion.
21. `tests/test_v02_reporting.py` — add the direct real workflow→report regression and the
    invalid-scalar negative regression.

The A3 runtime authorization is narrow: it does not change `src/sharper/v02_workflow.py`, the
`path_status` semantic dtype/schema, report signatures, twelve sections, nine plot slots, report
errors, precedence, transactions, Figure ownership, CLI implementation, or any other runtime
contract. The accepted domain is exact built-in `bool` plus the existing pandas/NumPy boolean
scalar family required by the approved dtype; integers, floats, strings, `None`, `pd.NA` and
arbitrary truthy/falsy coercions remain invalid. A3 adds no dependency, no global path, no finding,
no second Full Contract Review and no Full Implementation Review consumption. A1 and A2 remain
active, `docs/leakage.md` remains immutable, and release remains not released.

I6 sequencing is frozen as: first apply the A3 reporting interoperability repair and its direct
regression, then run the real workflow→report and real `v02-run` smoke; next complete the version,
root-surface, examples, distribution and CI implementation; obtain the corresponding evidence;
perform the six authorized post-transition truth-sync edits; run the post-I6 documentation truth
scan; then execute the final tests, build, examples, distribution and A–X gates. If I6 fails
before completion, the six documents must not be written as if all readiness gates passed.

## 14. CI and distribution readiness matrix

### 14.1 TASK20_DISTRIBUTION_CONTENT_MATRIX

Wheel and sdist are different artifacts with different consumers. They must not be treated
as having identical contents.

| Artifact | Required paths/classes | Forbidden paths/classes | Smoke consumer and reason |
|---|---|---|---|
| wheel | `sharper/**/*.py` runtime package files, including all future Task20 production modules in the allowlist; generated `.dist-info/METADATA`, `WHEEL`, `RECORD`, and `entry_points.txt`; backend-generated `.dist-info/licenses/LICENSE`; no external report template/data file | root `README.md`, `CHANGELOG.md`, `pyproject.toml`; `docs/`; `examples/`; `tests/`; `.github/`; `.agents/`; `.codex/`; `AGENTS.md`; `SPEC.md`; `IMPLEMENTATION_PLAN.md`; `artifacts/`; `dist/`; `notebooks/`; research files; `__pycache__/`; `*.pyc`; generated reports/PNGs; data/models/credentials/caches/private machine paths | source-free wheel install, `import sharper`, nine exports, `sharper --version`, `sharper v02-run`, and report smoke; wheel is runtime-only |
| sdist | `pyproject.toml`, `README.md`, `LICENSE`, `CHANGELOG.md`; `src/sharper/**/*.py`; `tests/**/*.py`; `docs/analysis-guide.md`, `docs/api.md`, `docs/leakage.md`, `docs/quickstart.md`, `docs/v02-integration-guide.md`, `docs/release-readiness.md`; all approved `docs/decisions/*.md`; the two existing v0.1 examples and five exact Task20 examples | `.github/`; `.agents/`; `.codex/`; `AGENTS.md`; `SPEC.md`; `IMPLEMENTATION_PLAN.md`; `docs/research/`; `scripts/`; `artifacts/`; `dist/`; `notebooks/`; `__pycache__/`; `*.pyc`; generated reports/PNGs; data/models/credentials/caches/private machine paths | source-free sdist build/install, source-tree build from extracted sdist, documentation/decision inspection and seven-example smoke; sdist is the reproducible source/build artifact |

The sdist example set is exactly `examples/basic_analysis.py`,
`examples/baseline_modeling.py`, `examples/v02_score_validation.py`,
`examples/v02_preloan.py`, `examples/v02_postloan.py`,
`examples/v02_combined_report.py` and `examples/v02_cli_json.py`. The wheel contains no
example file and therefore no wheel smoke may execute an example by reaching back into the
source tree. Neither artifact contains publishing tokens/config, local absolute paths,
Kaggle data, trained models or generated output.

The wheel does not carry a root README or root LICENSE file: the README text is represented
only in generated project metadata, and the license is represented only by Hatchling's
`.dist-info/licenses/LICENSE` metadata file. This is not a missing artifact; it is the
current backend's metadata contract.

### 14.2 Exact Hatchling authorization

Current `pyproject.toml` has `build-backend = "hatchling.build"`,
`[tool.hatch.version].path = "src/sharper/__init__.py"` and
`[tool.hatch.build.targets.wheel].packages = ["src/sharper"]`. The wheel package setting
remains unchanged; no wheel inclusion change is required because Task20 report text uses
Python-embedded strings and owns no external templates or static data files.

The only future build-content change authorized for CR-14 is adding the exact table
`[tool.hatch.build.targets.sdist]` with the following `include` and `exclude` patterns:

```toml
[tool.hatch.build.targets.sdist]
include = [
  "/CHANGELOG.md",
  "/LICENSE",
  "/README.md",
  "/docs/analysis-guide.md",
  "/docs/api.md",
  "/docs/decisions/*.md",
  "/docs/leakage.md",
  "/docs/quickstart.md",
  "/docs/release-readiness.md",
  "/docs/v02-integration-guide.md",
  "/examples/baseline_modeling.py",
  "/examples/basic_analysis.py",
  "/examples/v02_cli_json.py",
  "/examples/v02_combined_report.py",
  "/examples/v02_postloan.py",
  "/examples/v02_preloan.py",
  "/examples/v02_score_validation.py",
  "/pyproject.toml",
  "/src/sharper/**/*.py",
  "/tests/**/*.py",
]
exclude = [
  "/.DS_Store",
  "/**/.DS_Store",
  "/**/__pycache__/**",
  "/**/*.pyc",
  "/.agents/**",
  "/.codex/**",
  "/.github/**",
  "/AGENTS.md",
  "/IMPLEMENTATION_PLAN.md",
  "/SPEC.md",
  "/artifacts/**",
  "/dist/**",
  "/docs/research/**",
  "/notebooks/**",
  "/scripts/**",
]
```

The exact inclusion change is a build-content change, not a version-source change. No
second build backend, publishing plugin, lock file, network build step, mandatory runtime
dependency, or credential configuration is authorized. If the installed Hatchling version
requires an equivalent syntax for the same exact table, implementation must stop and file
a governance amendment rather than silently broaden the allowlist.

### 14.3 Future distribution smoke matrix

The future direct acceptance matrix is: (A) build wheel; (B) build sdist; (C) inspect wheel
required/forbidden entries and metadata; (D) inspect sdist required/forbidden entries and
metadata; (E) source-free wheel install; (F) source-free sdist install and wheel build from
extracted sdist; (G) source-free `import sharper`; (H) installed nine Task20 exports and
permanent v0.1 surface; (I) installed `sharper v02-run`; (J) installed root `--version`;
(K) installed report generation; (L) seven examples from the appropriate source-tree or
extracted-sdist environment; (M) no network. Wheel and sdist expected-entry tests use
test-side frozen sets and never derive expectations from the active Hatchling include
configuration.

The future CI readiness target is the existing supported Python matrix `3.10`, `3.11`,
`3.12`, `3.13`, with `pytest`, Ruff check, Ruff format check, and the existing approved
format-governance handling. Python 3.12 additionally runs `build --no-isolation`, controlled
CLI smoke, five example smokes, wheel/sdist checks and source-free clean-install smoke.
No network-dependent test, credential, publishing step or new mandatory dependency is
allowed.

Distribution acceptance is the matrix in §14.3. The wheel is runtime-only; the sdist is the
buildable source/documentation/example artifact. The mandatory dependency change allowance
is exactly **No**; existing runtime and dev dependency ranges remain unchanged.

## 15. Determinism, parity, privacy and immutability

For the same Python request or the same semantic JSON document, Task 20 must produce the
same enabled path tuple, call order, owner result field values, path-status rows, warning/
limitation ordering, report section ordering and asset names. Filesystem mtimes are not
part of equality. JSON object key order is non-semantic; array order and scalar values are
semantic. No current time, environment timezone, random ID, hash order, parallel completion
or filesystem glob order may affect results.

Python and CLI parity is defined at the constructed owner-config boundary: a CLI policy or
warning JSON carrier must produce the same `DecisionStrategyConfig` or
`LifecycleMonitoringConfig` fields, tuple order, temporal values, condition tree and
defaults as the equivalent Python request. Owner result tables, Task 20 path-status rows,
call trace, warnings and limitations must then be equal under the owner contracts. JSON
bytes need not be equal.

Inputs, nested dataclasses, tuples, mappings and owner results are read-only. The workflow
does not mutate or cache into the input DataFrame, and the result contains no raw full
DataFrame. Reports write only their requested output transaction. Result/report/CLI output
never exposes raw entity/row payload, target labels, credentials, environment values,
expanded paths, model/estimator repr, arbitrary object repr or private kernel content.
No generic PII classifier is introduced; closed schemas, owner privacy contracts and
non-retention are the safety boundary.

## 16. Normative acceptance matrix A–X

There are exactly **24** acceptance domains. Each domain requires a named direct executable
test; “covered by full pytest” is not sufficient:

| Domain | Required direct test |
|---|---|
| A Public API/types | `test_v02_public_symbols_signatures_and_fields` |
| B Three path independence | `test_v02_paths_enable_disable_independently` |
| C Upstream exactly-once | `test_v02_enabled_owner_calls_once_in_frozen_order` |
| D No recomputation | `test_v02_report_and_cli_do_not_call_domain_owners` |
| E Task15–19 handoffs | `test_v02_owner_result_carriers_and_governance_handoff` |
| F No raw result DataFrame | `test_v02_result_has_no_raw_frame_or_runtime_handle` |
| G Closed JSON schema | `test_v02_policy_warning_schema_field_mapping` |
| H Duplicate keys | `test_v02_json_duplicate_keys_rejected` |
| I Unknown version/field/operator | `test_v02_json_unknown_version_field_operator` |
| J JSON budgets | `test_v02_json_budget_max_and_max_plus_one` |
| K Python/CLI parity | `test_v02_python_cli_semantic_config_parity` |
| L Markdown/HTML parity | `test_v02_report_sections_and_semantic_parity` |
| M Report no recomputation | `test_v02_report_consumes_result_only` |
| N Figure/assets ownership | `test_v02_figure_close_asset_order_and_rollback` |
| O CLI exit/error behavior | `test_v02_cli_exit_codes_and_no_default_traceback` |
| P Permanent v0.1 compatibility | `test_v01_compatibility_manifest_unchanged` |
| Q Current release surface | `test_current_release_surface_appends_task20` |
| R Version metadata | `test_v02_version_transition_gate` |
| S Docs/examples | `test_v02_examples_and_documentation_smoke` |
| T Distribution/clean install | `test_v02_distribution_source_free_smoke` |
| U CI readiness | `test_ci_matrix_contains_v02_gates` |
| V Determinism | `test_v02_repeatability_and_ordering` |
| W Privacy/immutability | `test_v02_privacy_and_input_immutability` |
| X No release actions | `test_v02_release_terminal_state_is_not_released` |

Every one of the 28 Task 20 error keys has at least one direct public Python or CLI test;
no private `_raise` helper is accepted as sole evidence. Every one of the 12 resource
gates has a max-pass and max+1-first-fail direct test. The acceptance run must also execute
the canonical regression suites for Tasks 15, 16, 17, 18 and 19 plus the permanent v0.1
compatibility suite; current release-surface tests are additive.

C1 implementation tests must additionally prove the exact Path A CLI predicate and its
closed option/mask/label mapping, the governance dependency matrix and exactly-once gate,
the two raw-frame carrier roles including same-object behavior, the Task 15 label domain,
and stable warning/limitation tuple construction, Python/CLI parity and result-only report
consumption. These are future implementation obligations; this C1 document wave adds no
test file and reports no implementation evidence.

C2 implementation tests must additionally prove the complete policy-field inventory and
one-to-one JSON construction for all three tuple-of-tuples fields, exact inner arities,
outer-array ordering, duplicate/unknown/null/type rejection, owner duplicate semantics,
and direct Python/JSON config-field equality. They must exercise the full temporal-field
inventory with the exact six-digit datetime grammar, lexical/calendar/time-zone checks,
integer-microsecond duration boundaries, boolean/non-finite number rejection, and
Python/JSON temporal parity. They must also exercise all 12 resource rows at max and
max+1, including root/depth, per-object/per-array, condition-node, per-scenario, figure
entry pre-acquisition and aggregate staged-PNG gates; a failing gate must leave no partial
final output, perform no silent truncation, and preserve the owner-limit boundary. These
are future implementation obligations; this C2 document wave adds no test file and
reports no implementation evidence.

C3 implementation tests must additionally prove the exact 12-row provenance matrix, the
disabled versus enabled-empty markers, the nine fixed plot slots and their true/false call
predicates, fixed ordinal filenames and unavailable behavior, Figure close ownership, the
Task13 title/path/overwrite transaction, the ReportArtifact field values, all four
surface-specific precedence tables, multi-invalid first-error fixtures, the 28-key
reachability matrix, JSON/parser exception containment, workflow-failure report call count,
report-failure workflow non-rerun, and preservation of every C1/C2 boundary. These are
future implementation obligations; this C3 document wave adds no test file and reports no
implementation evidence.

C4 implementation tests must additionally prove the single authoritative version source and
the pre-checkpoint `0.1.0` gate; post-checkpoint `0.2.0` parity across import, root CLI,
wheel metadata and sdist `PKG-INFO`; absence of a second version literal; the exact wheel
and sdist required/forbidden content sets; exact Hatchling include/exclude behavior; source-
free wheel/sdist/example environments; offline behavior; the existing CHANGELOG heading and
bullet convention; the exact 32-entry tracked allowlist; roadmap-to-file coverage; reverse
allowlist justification; and rejection of any unallowlisted tracked change. These are future
implementation obligations; this C4 document wave adds no test file and reports no
implementation evidence.

## 17. Risk register history and C4 impact

The kickoff risk register and its prior resolution claim are historical discovery evidence,
not a substitute for the Full Contract Review. The Full Contract Review adjudicated the
contract against the 12 discovery risks and produced the separate frozen `T20-CR-01..15`
finding ledger. C4 updates only the directly impacted version/distribution/changelog/
allowlist risk wording below; C1/C2/C3 wording is historical evidence and it does not
declare the whole register resolved.

| ID | Discovery ambiguity | Contract decision / section | Scope result | Remaining ambiguity |
|---|---|---|---|---|
| RISK-T20-01 | New workflow could collide with `AnalysisRun` | independent `v02_workflow.py` API; §4 | no v0.1 expansion | No |
| RISK-T20-02 | Owner calls could be repeated by report/CLI | exact call matrix, one-call governance gate and trace; §3, §9 | C1 targeted fix confirmed by bounded closure | No |
| RISK-T20-03 | Three paths could become one mandatory chain | independent nullable request carriers plus explicit closed Path A CLI predicate; §3.2, §4, §10 | C1 targeted fix confirmed by bounded closure | No |
| RISK-T20-04 | Audit could become a hidden policy engine | diagnostic-only `V02AuditRequest` and sole secondary reference carrier; §3.2, §4, §15 | C1 targeted fix confirmed by bounded closure | No |
| RISK-T20-05 | Governance could run implicitly | explicit final `governance` carrier plus exact Task 19 dependency matrix; §3.2 | C1 targeted fix confirmed by bounded closure | No |
| RISK-T20-06 | JSON could become a generic rules DSL | one-to-one Task17/18 mapping plus one canonical tuple carrier; §6 | C2 targeted fix confirmed by bounded closure | No |
| RISK-T20-07 | Duplicate/unknown JSON could fail open | parser-level rejection, owner duplicate semantics and fixed resource gates; §6.3, §8 | C2 targeted fix confirmed by bounded closure | No |
| RISK-T20-08 | Report could recompute or own domain state | exact 12-row provenance matrix, result-tuple consumption and no-recompute boundary; §9.2 | C3 targeted fix confirmed by bounded closure | No |
| RISK-T20-09 | Figures/assets could leak or double-close | exact nine-slot registry, fixed filenames, Figure ownership and staged PNG gate; §8, §9.3 | C3 targeted fix confirmed by bounded closure | No |
| RISK-T20-10 | CLI could change `analyze` semantics | separate `v02-run`, closed score Path A options and surface-specific CLI precedence; §9.5, §10, §11 | C3 targeted fix confirmed by bounded closure | No |
| RISK-T20-11 | v0.1 exports/version could migrate silently | permanent manifest, append-only rule, single `__version__` source and post-checkpoint 0.2.0 gate; §11, §12, §13.2 | C4 targeted fix confirmed by bounded closure | No |
| RISK-T20-12 | readiness could be mistaken for release or artifact scope could be ambiguous | terminal `Release Ready — Not Released`, wheel/sdist matrix, exact Hatchling authorization and tracked allowlist; §13–14 | C4 targeted fix confirmed by bounded closure | No |

The kickoff record's historical `12/12` resolution claim is preserved as history only. It
is not repeated as the current closure verdict. C1's five, C2's three, C3's four and C4's
three entries are now closed by the bounded closure recorded for the exact set
`T20-CR-01..15`. C4 updates only the directly impacted version/distribution/changelog/
allowlist wording above; all other risk entries retain their prior wording and are not
re-adjudicated beyond the bounded closure's finding-specific checks.

## 18. Ambiguity register history and C4 impact

The 18 kickoff ambiguities and their prior claimed resolutions remain historical context.
The Full Contract Review findings are the current adjudication. C4 updates only the
directly impacted version/distribution/changelog/allowlist entries below; all other entries
remain untouched.

| ID | Ambiguity | Frozen resolution | Section |
|---|---|---|---|
| AMB-T20-01 | workflow module/name | `v02_workflow.py`, `run_v02_workflow` | §4 |
| AMB-T20-02 | public symbol count | exactly 9 append-only symbols | §4 |
| AMB-T20-03 | audit placement | optional diagnostic handoff, not a primary path | §3.2 |
| AMB-T20-04 | governance timing | explicit optional final step, dependency matrix, exactly-once gate and workflow precedence stage 4; bounded closure PASS | §3.2, §9.5 |
| AMB-T20-05 | disabled path carrier behavior | absent carrier; no hidden validation | §4, §7 |
| AMB-T20-06 | path status vocabulary | two statuses and two reasons only | §5 |
| AMB-T20-07 | raw DataFrame retention | primary workflow carrier plus audit-only secondary reference; never in result/report; same-object behavior closed in C1 | §4–5 |
| AMB-T20-08 | domain call order | audit, score, preloan, postloan, governance, with C1 governance gate before owner calls | §3.3 |
| AMB-T20-09 | JSON version names | `task20.policy.v1` and `task20.warning.v1` | §6 |
| AMB-T20-10 | JSON field order/canonicalization | object order ignored; every tuple/sequence uses one array shape, array order preserved, no byte identity; C2 duplicate semantics are owner-defined | §6.1, §6.3, §15 |
| AMB-T20-11 | JSON temporal encoding | exact six-fraction-digit naive/UTC grammar and integer-microsecond duration carrier; C2 bounded closure PASS | §6.2 |
| AMB-T20-12 | owner field mapping | complete same-name Task17/18 mapping matrix with fixed tuple arity and one JSON shape; C2 bounded closure PASS | §6.1–6.2 |
| AMB-T20-13 | CLI path coverage | closed external-prediction Path A plus B/C/audit; Governance remains Python-only; bounded closure PASS | §3.2, §10 |
| AMB-T20-14 | disabled report sections | exact 12-row provenance matrix with `not_requested` versus enabled-empty markers | §9.2 |
| AMB-T20-15 | plot materialization | exact nine result-only slots, true predicate exactly one call, false predicate zero calls, fixed ordinal filenames and gaps | §8, §9.3 |
| AMB-T20-16 | write failure behavior | exact Task13 title/path/overwrite stages, staging/backup/rollback and staged aggregate PNG-byte gate | §8, §9.4 |
| AMB-T20-17 | compatibility surface | baseline manifest plus append-only suffix | §11 |
| AMB-T20-18 | version/release transition | authoritative `src/sharper/__init__.py::__version__`; checkpoint remains 0.1.0; 0.2.0 only after 15/15 bounded closure, Approved — Go and approved checkpoint; no release | §12, §13.2 |

The kickoff record's historical `18/18` resolution claim is preserved as history only; it
is not a current closure verdict. C1's five, C2's three, C3's four and C4's three findings
are closed by the bounded closure. C4 adds the version-source, artifact-content,
Hatchling, CHANGELOG and exact-allowlist rules above; it creates no new finding.

## 19. Self-consistency and current gate

The C4 and A3 amendments have zero `TBD`/`TODO` placeholders in their targeted normative
sections. A3 leaves the C1–C4 contract semantics and all frozen runtime/API counts unchanged.
Its ownership is explicit: `__version__` remains the sole version source, Hatchling owns
only the exact sdist content table, CHANGELOG uses the existing heading/bullet convention,
and the 32-entry tracked allowlist owns future implementation scope. C1/C2/C3 normative
semantics remain unchanged; the fifteen findings across C1/C2/C3/C4 are closed by the
bounded closure, with no open findings. Public symbols remain 9, dataclasses 7, result
fields 11, Task20 errors 28, resource gates 12, report formats 2, report sections 12 and
plot slots 9;
C4 and A3 change none of these runtime/API counts. They approve no mandatory dependency change.

This contract-definition stage has completed its own documentation/inventory checks only;
it does not claim Python tests, Ruff, build, distribution, CLI smoke or implementation
evidence. Those are implementation and release-readiness gates after contract approval.

### Governance Amendment A4 — Final post-closure user-document truth-sync scope correction

A4 is a bounded procedural governance amendment. It records that the historical Full
Implementation Review is now `NO-GO — CONSUMED`, `T20-IR-01..04` are `CLOSED`, targeted
repair is complete, and bounded implementation closure is `PASS`. Those post-I6 lifecycle
events made four A2-era current-state statements stale in the user-facing documentation.

The mechanical post-closure evidence set is exactly:

| Path | Current statement | Historical/current | Stale after review + repair + closure | Requires final truth sync |
|---|---|---|---|---|
| `README.md` | Task20 implementation complete but Full Implementation Review pending | current | yes | yes |
| `docs/analysis-guide.md` | final implementation review still pending | current | yes | yes |
| `docs/v02-integration-guide.md` | final implementation review is pending | current | yes | yes |
| `docs/release-readiness.md` | current terminal claim says Full Implementation Review Pending | current | yes | yes |
| `docs/quickstart.md` | v0.2 is not released; no review-pending claim | current | no | no |
| `docs/api.md` | post-review closure/current package facts; no review-pending claim | current | no | no |
| `docs/leakage.md` | Task20 audit handoff and v0.2 non-release boundary; no review-pending claim | current | no | no |

The evidence set is exact: `STALE_SET = 4`. All four paths are already members of the
32-entry global Task20 implementation allowlist, so the global allowlist remains exactly
32 and global path additions are zero. A4 does not amend semantic contract authority:
A3 `1697dcf14b0e7d1c4fce23b1c1b186d7090f1dd1` remains the sole authoritative semantic
contract baseline. A4 itself modifies only this decision record, `SPEC.md`, and
`IMPLEMENTATION_PLAN.md`; it does not modify any user document.

The A4 sufficiency proof is complete: all four stale paths are in global32; exact4 is
sufficient for final truth restoration; quickstart, API, and leakage require no change;
no runtime/test/CI/example change is required; and no semantic contract change is required.
The frozen non-scope inventory remains unchanged: 9 public symbols, 7 dataclasses, 11
result fields, 28 Task20 errors, 12 resource gates, 2 report formats, 12 report sections,
9 plot slots, 2 JSON schemas, 5 Task20 examples, 7 sdist examples, global32, and package
version `0.2.0`.

After a successful A4 checkpoint, the only authorized next stage is
**TASK20 — POST-REPAIR FINAL STATUS AND USER-DOCUMENT TRUTH SYNC**. Its exact final scope
is seven paths: the three governance files plus `README.md`, `docs/analysis-guide.md`,
`docs/v02-integration-guide.md`, and `docs/release-readiness.md`. The latter four have
narrow authorization to correct only post-review status, repair/closure state, final
Task20 readiness, `A–X = 24/24 PASS`, and `Release Ready — Not Released` as applicable.
`docs/quickstart.md`, `docs/api.md`, and `docs/leakage.md` are immutable in that stage.
No runtime, test, CI, example, `pyproject.toml`, CHANGELOG, version, export, dependency,
review, finding, release, push, tag, upload, publish, or deploy action is authorized by A4.

Current state is frozen as:

```text
Task20 title:                 Task 20 — v0.2 Integration and Release Readiness
Contract:                     Approved — Go as amended by A1 + A2 + A3
Contract governance:          Complete
Implementation:               REPAIRED AND CLOSURE-VALIDATED
I1–I6:                        Complete
Targeted repair:              Complete
Bounded implementation closure: PASS
Full Contract Review:         NO-GO — Permanently Closed
Full Implementation Review:   NO-GO — CONSUMED (quota 0)
T20-IR-01..04:                CLOSED (4 CLOSED / 0 OPEN)
Affected D/G/J/N/T/U/X:       PASS
A–X:                          24/24 PASS
Task20:                       A4 procedural amendment in progress
Package version:              0.2.0
Future implementation target: 0.2.0
Release state:                NOT RELEASED
Full Contract Review findings: T20-CR-01..15 (P1=15; P0=0; P2=0)
C1 finding status:            T20-CR-01..05 CLOSED
C2 finding status:            T20-CR-06..08 CLOSED
C3 finding status:            T20-CR-09..12 CLOSED
C4 finding status:            T20-CR-13..15 CLOSED
Bounded Contract Closure:     PASS (15 CLOSED / 0 OPEN)
Approved checkpoint:          Original three-document commit preserved
Amended A1 checkpoint:         d0ced4a11257423bc11c442462cd3fff8d000656
Amended A2 checkpoint:         This exact Governance Amendment A2 commit
A3 checkpoint:                 This exact Governance Amendment A3 commit
A4 classification:             Bounded procedural governance amendment only
A3 semantic authority:         1697dcf14b0e7d1c4fce23b1c1b186d7090f1dd1
Global implementation allowlist: 32
I6 exact scope:                21 (complete)
Final sync scope after A4:     exact7
A4 own scope:                  exact3 governance files (count 3)
Added final-stage paths:       README.md; docs/analysis-guide.md; docs/v02-integration-guide.md; docs/release-readiness.md
Second Full Contract Review:  Forbidden
Next stage:                   TASK20 — POST-REPAIR FINAL STATUS AND USER-DOCUMENT TRUTH SYNC (EXACT7 FINAL CHECKPOINT)
```

No implementation file, test, CLI implementation, example, CI workflow, user document,
`pyproject.toml`, version, export, dependency, push, tag or release is changed by A4.
The exact three-document A4 commit is a procedural governance checkpoint only; A3 remains
the semantic contract authority, and the authorized exact7 final truth-sync scope begins
only in the next stage.
