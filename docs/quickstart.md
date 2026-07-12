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
