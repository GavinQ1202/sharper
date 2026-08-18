# Analysis guide

`run_analysis` combines the existing schema, summary, quality, non-target
analysis, feature-suggestion, visualization, and optional modeling APIs into
one `AnalysisRun`. It records skipped work, warnings, limitations, and existing
budget metadata rather than recomputing results while reporting.

Target analysis requires an explicit `target` and `task`. It is exploratory:
relationship statistics are not causal claims. Setting `include_model=False`
provides target analysis without fitting a model. Setting it to `True` runs the
matching split-first classification or regression baseline and holdout-only
evaluation.

`generate_analysis_report` writes deterministic Markdown or static HTML plus a
sibling directory of PNG assets. Reporting consumes the stored results and
Figures; it does not retrain, predict, or calculate new statistics. Sharper
does not provide a dashboard, server, or interactive HTML report in v0.1.

The workflow cannot eliminate entity/group leakage or make random holdout safe
for temporal data. Review the recorded warnings and limitations, and see
[Leakage safeguards](leakage.md) before interpreting a baseline model.

## Opt-in v0.2 integration

Task 20 adds a separate typed integration path. It does not change the default
`run_analysis` result, the v0.1 report, or the `sharper analyze` command.

The three primary paths are independently enabled:

- score validation, using the explicit Task 15 request and positive-label/
  score provenance;
- pre-loan eligibility, using the caller-frozen Task 17 strategy config;
- post-loan warning/lifecycle, using the caller-frozen Task 18 monitoring config.

Task 16 audit is an optional diagnostic path attached to the run. Task 19
governance is an optional final step that consumes the enabled upstream owner
results; neither is a fourth or fifth primary business path.

For enabled paths, the fixed owner order is:

```text
audit → score validation → pre-loan → post-loan → governance
```

Each enabled owner is called exactly once. Task 20 does not retry an owner
failure, and the reporting layer consumes stored result tables and Figures; it
does not refit, predict, evaluate, execute rules, reconstruct alerts, or
recompute domain statistics.

`V02WorkflowResult` stores the typed owner results, fixed path status, call
trace, warnings, and limitations. It does not retain the primary raw DataFrame
or the optional audit reference. The v0.2 workflow is opt-in and the current
package metadata is `0.2.0`; the Full Implementation Review is
`NO-GO — CONSUMED`, and T20-IR-01..04 are closed after targeted repair and
bounded implementation closure. The current implementation is
`REPAIRED AND CLOSURE-VALIDATED`, and v0.2 has not been released.
