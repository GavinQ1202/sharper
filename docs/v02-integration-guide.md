# v0.2 integration guide

Task 20 is Sharper's opt-in integration surface for the frozen Tasks 15–19
results. It adds typed orchestration, closed JSON carriers for Task 17/18
specifications, static reports, and the `v02-run` command while preserving the
v0.1 workflow and CLI.

The approved v0.2 surface is documented here. The current package remains
version `0.1.0`; the v0.2 target is `0.2.0`, and v0.2 is not released.

## Architecture

Task 20 has three independent primary paths:

1. score validation — explicit Task 15 score kind, direction, positive label,
   and prediction provenance;
2. pre-loan — caller-frozen Task 17 eligibility/decision rules and simulated
   actions;
3. post-loan — caller-frozen Task 18 point-in-time warning and lifecycle
   monitoring.

Task 16 audit is an optional diagnostic path. Task 19 governance is an
optional final step that consumes the enabled upstream owner results. They are
not additional primary business paths.

Only enabled paths are called, in this fixed order:

```text
audit → score validation → pre-loan → post-loan → governance
```

Each enabled owner is called exactly once. Task 20 does not retry a failed
owner, copy domain algorithms into the workflow, or let reporting recompute
statistics. The workflow result contains typed owner results and integration
metadata, but not the primary raw DataFrame or the optional audit reference.

## Python API

The approved Task 20 public symbols are exactly:

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

The seven dataclasses are the five path requests, the complete workflow
request, and the workflow result. `run_v02_workflow(request)` returns a
`V02WorkflowResult`; `generate_v02_report` accepts the exact `title`, `format`,
and `overwrite` keyword parameters and returns the existing `ReportArtifact`
type. Exact fields, annotations, defaults, and ordering are in the
[API reference](api.md).

The final v0.2 usage is shaped as follows:

```python
from sharper import (
    V02ScoreValidationRequest,
    V02WorkflowRequest,
    run_v02_workflow,
)

request = V02WorkflowRequest(
    data=frame,
    score_validation=V02ScoreValidationRequest(
        target="outcome",
        config=score_config,
        external_predictions=external_predictions,
    ),
)
result = run_v02_workflow(request)
```

The Task20 names become active root exports only at the final public-surface
gate. This documentation describes the approved final surface and does not
claim that the current `0.1.0` root package already exposes these nine names.
The internal JSON adapters are carriers for the CLI, not additional public
symbols.

### Score validation

The caller must declare the target, Task 15 config, and exactly one estimator
or external prediction source. The positive label, ranking/probability kind,
risk direction, and provenance are explicit. Sharper does not infer a target,
score source, positive class, or risk direction. Ranking scores remain ranking
scores; only an aligned `[0, 1]` event probability for the explicit positive
class can support probability-only calibration or expected-loss arithmetic.

### Pre-loan

`V02PreLoanRequest` carries a frozen `DecisionStrategyConfig`. The strategy
owns the caller's closed conditions, rule order, actions, action-role mapping,
constraints, and evaluation time. It produces offline simulated action and
rule evidence only; it does not approve, decline, contact, collect, optimize,
or deploy a policy.

### Post-loan

`V02PostLoanRequest` carries a frozen `LifecycleMonitoringConfig`. It produces
point-in-time warning, notification, episode, event-match, state, transition,
and lifecycle evidence. It does not send notifications, modify accounts, or
execute collection actions. Future events are used only for the owner's
backtest semantics and never to construct an earlier signal.

### Audit and governance

`V02AuditRequest` passes the primary frame and optional audit-only reference to
Task 16. It does not turn the reference into a policy or score comparison
source. `V02GovernanceRequest` is the final optional step and consumes frozen
owner results plus approved structured governance declarations; it does not
re-run rules, alerts, metrics, missingness drift, or lifecycle calculations.

## Closed JSON carriers

The CLI accepts exactly two versioned pure-data JSON schemas:

```text
task20.policy.v1
task20.warning.v1
```

The policy carrier maps one-to-one to `DecisionStrategyConfig`. Its top-level
fields are:

```text
schema_version, strategy_key, strategy_version, effective_from, expires_at,
evaluation_time, rules, default_action_name, unknown_action_name,
action_role_mapping, constraints, ranking_score_column, ranking_score_direction,
historical_action_column, historical_action_mapping, historical_policy_version,
exposure_column, loss_fraction, action_assumptions, exposure_unit,
segment_columns, time_slice_column
```

The warning carrier maps one-to-one to `LifecycleMonitoringConfig`. Its
top-level fields are:

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

Nested objects use the exact approved Task 17/18 dataclass fields. Arrays map
to tuples without sorting, deduplication, or conversion to object mappings.
The only accepted datetime grammar is:

```text
naive_datetime := YYYY-MM-DD "T" HH ":" MM ":" SS "." microsecond6
utc_datetime   := naive_datetime "Z"
datetime      := naive_datetime | utc_datetime
```

The fraction is exactly six digits. A `Z` value is UTC-aware; a value without
`Z` is naive. No offset, date-only value, whitespace, lowercase marker,
automatic localization, current-date lookup, or DST conversion is accepted.
Durations are JSON integers representing microseconds; `cooldown` may be zero,
while the other required positive durations follow their owner constraints.

The carriers are closed and non-executing. They reject YAML, TOML, Python
configuration, callables, expressions, scripts, templates, `include`, URLs,
`$ref`, environment expansion, tilde/glob/path expansion, comments, unknown
fields, unknown operators, duplicate object keys, non-finite numbers, and
unsupported schema versions. File acquisition is literal; file read failures
remain `OSError`.

## Static reports

`generate_v02_report` supports exactly two formats:

```text
markdown
html
```

The report has exactly twelve deterministic sections, in this order:

1. Run Context
2. Path Status
3. Score Validation
4. Data Audit and Leakage
5. Pre-loan Eligibility
6. Post-loan Warning
7. Governance
8. Cross-path Comparison
9. Reason and Override Trace
10. Stability and Business Evidence
11. Warnings and Limitations
12. Provenance and Release Readiness

The report consumes stored result tables and caller-owned Figures. It does not
split, fit, predict, evaluate, execute rules, simulate actions, reconstruct
alerts, recompute backtests, or recalculate drift/importance. It writes a
static report and a sibling PNG asset directory. There are nine possible
deterministic plot slots:

```text
score_validation_gains
score_validation_lift
score_validation_calibration
score_validation_threshold
governance_importance
governance_candidate_comparison
governance_prediction_drift
governance_performance_stability
governance_summary
```

Only slots with valid available result evidence are acquired; the fixed asset
budget is nine entries. Markdown and HTML use the same section and asset
semantics.

## CLI

The opt-in command is:

```bash
sharper v02-run INPUT --output OUTPUT
```

Approved options are:

```text
--output, -o
--format {markdown,html}
--policy-json PATH
--warning-json PATH
--audit / --no-audit
--reference-input PATH
--target TEXT
--external-ranking-score-column TEXT
--external-event-probability-column TEXT
--ranking-direction TEXT
--probability-provenance TEXT
--score-validation-mask-column TEXT
--positive-label-type TEXT       (repeatable; str, int, or bool)
--positive-label-value TEXT      (repeatable; one value)
--score-test-size FLOAT          (default 0.20)
--score-random-state INTEGER     (default 42)
--overwrite / --no-overwrite    (default overwrite)
```

`--policy-json` and `--warning-json` select the pre-loan and post-loan paths.
Score validation requires its explicit target, exactly one external score
column, validation mask, positive-label type/value, and the matching direction
or probability provenance. Audit may be attached with `--audit`; a reference
frame is supplied with `--reference-input`. There is no governance CLI option,
title option, or alternate configuration format.

Examples:

```bash
sharper v02-run data.csv --output policy-report.md --policy-json policy.json
sharper v02-run data.csv --output warning-report.html \
  --warning-json warning.json --format html
sharper v02-run data.csv --output score-report.md \
  --target outcome \
  --external-ranking-score-column risk_score \
  --ranking-direction higher_risk \
  --score-validation-mask-column is_validation \
  --positive-label-type int --positive-label-value 1
```

Exit behavior is stable: `0` for success, `2` for caller/validation/owner
contract errors, `3` for filesystem I/O errors, and `70` for unexpected
internal errors. The default path emits an error without a traceback.

## Errors, determinism, privacy, and limitations

Task20-owned errors use the exact prefix `sharper task20: <error_key>`. The
closed 28-key registry is:

```text
invalid_request_type, request_requires_primary_path, request_path_input_conflict,
request_raw_carrier, json_not_object, json_decode, json_encoding,
json_duplicate_key, json_schema_version, json_unknown_field,
json_unknown_operator, json_structure, json_scalar, json_budget,
policy_mapping, warning_mapping, owner_call_contract, result_contract,
report_format, report_result, report_asset_budget, report_title, report_path,
report_overwrite, cli_argument, cli_spec_required, cli_output,
governance_dependency_missing
```

Domain-owner errors retain their own stable prefix and cause. Task20 does not
rename them. Invalid input is rejected before an enabled owner is called;
enabled owner calls are not retried. Fixed ordering, typed tables, explicit
status/reason fields, source provenance, requested/actual budgets, warnings,
limitations, and the absence of random sampling provide deterministic output
for the same input and typed request.

The integration layer validates raw carriers and does not retain the primary
or audit reference DataFrames in its result. Reports and errors avoid raw rows,
credentials, arbitrary object representations, private expanded paths, and
unsupported configuration payloads. This is a Task20 boundary; it is not a
claim that every upstream result type contains no user data.

The result remains offline evidence. It does not prove causality, production
safety, regulatory compliance, approval, deployment, realized revenue, or
operational alert execution. Inspect each Task 15–19 owner's warnings,
limitations, maturity/censoring status, effective sample support, and
reference/current semantics before using the result.

For current version, public-surface, distribution, examples, CI, and release
gates, see [release readiness](release-readiness.md).
