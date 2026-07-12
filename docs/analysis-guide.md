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
