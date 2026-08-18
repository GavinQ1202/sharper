# Quickstart

Sharper analyzes one local tabular DataFrame at a time. The public workflow is
deterministic for a fixed `random_state` and returns structured results before a
report is written.

## Analysis without a model

```python
from sharper import generate_analysis_report, load_csv, run_analysis

frame = load_csv("customers.csv")
run = run_analysis(
    frame,
    target="churned",
    task="classification",
    include_model=False,
)
artifact = generate_analysis_report(run, "customer-analysis.md", format="markdown")
```

`include_model=False` still runs the requested target relationship analysis, but
`run.training` and `run.evaluation` are `None`. The Markdown report and its
sibling `customer-analysis_assets/` directory form one PNG-assets bundle.

## Classification or regression baseline

```python
from sharper import generate_analysis_report, run_analysis

run = run_analysis(
    frame,
    target="churned",
    task="classification",
    include_model=True,
    random_state=42,
)
generate_analysis_report(run, "customer-analysis.html", format="html")
```

For regression, supply a finite numeric target and use `task="regression"`.
The HTML report is static and uses a sibling PNG-assets directory; Sharper does
not start a server or provide an interactive dashboard.

## CLI

```bash
sharper analyze customers.csv --output customer-analysis.html
sharper analyze customers.csv --target churned --task classification --model --output customer-analysis.html
```

The CLI accepts local CSV and single-sheet XLSX input. Install the optional
`excel` extra before reading XLSX files.

## Opt-in v0.2 integration

Task 20 keeps the v0.1 workflow independent and adds three explicit paths:
score validation, pre-loan eligibility, and post-loan warning/lifecycle. Audit
is an optional diagnostic attachment; governance is an optional final step.

The approved final Python surface uses the following typed carriers:

```python
from sharper import (
    V02ScoreValidationRequest,
    V02WorkflowRequest,
    run_v02_workflow,
)

request = V02WorkflowRequest(
    data=frame,
    score_validation=V02ScoreValidationRequest(
        target="churned",
        config=score_config,
        external_predictions=external_predictions,
    ),
)
result = run_v02_workflow(request)
```

`score_config` and `external_predictions` above are caller-created approved
Task 15 objects. The same request carrier can opt into `audit`, `preloan`,
`postloan`, and `governance` with their corresponding typed requests. These
The nine Task 20 names are active root exports in package version `0.2.0`.
Task 20 remains opt-in, and the package has not been released or deployed.

The CLI path is explicit and uses only the frozen options:

```bash
sharper v02-run customers.csv --output v02-report.md --policy-json policy.json
sharper v02-run customers.csv --output v02-report.html --warning-json warning.json --format html
```

Score validation can be enabled with the approved external score options, and
audit can be added with `--audit` and, when needed, `--reference-input`.
`v02-run` has no governance-specific CLI option, title option, YAML/TOML
carrier, or arbitrary-code configuration path. See the
[v0.2 integration guide](v02-integration-guide.md) for the complete path and
JSON boundaries. v0.2 is not released.
