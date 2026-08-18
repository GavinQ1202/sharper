"""Write a deterministic combined Task 20 Markdown report bundle."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from sharper import (
    DecisionStrategyConfig,
    EarlyWarningRule,
    LifecycleMonitoringConfig,
    LifecycleState,
    MonitoringCondition,
    V02PostLoanRequest,
    V02PreLoanRequest,
    V02WorkflowRequest,
    WarningScenario,
    generate_v02_report,
    run_v02_workflow,
)


def _run(output_dir: Path) -> Path:
    frame = pd.DataFrame(
        {
            "entity": ["entity-a", "entity-b"],
            "income": [40_000, 80_000],
            "observed": [datetime(2025, 1, 1), datetime(2025, 1, 2)],
            "available": [datetime(2025, 1, 1), datetime(2025, 1, 2)],
            "balance": [100, 0],
        }
    )
    strategy = DecisionStrategyConfig(
        "example-combined-preloan",
        "v1",
        datetime(2025, 1, 1),
        None,
        datetime(2025, 1, 2),
        (),
        "review",
        "review",
        (("review", "review"),),
    )
    condition = MonitoringCondition("atomic", "gt", "column", "balance", "literal", 0)
    monitoring = LifecycleMonitoringConfig(
        "example-combined-postloan",
        "v1",
        datetime(2025, 1, 5),
        "entity",
        "observed",
        "available",
        ("balance",),
        None,
        None,
        None,
        False,
        timedelta(days=1),
        timedelta(days=2),
        False,
        None,
        "day",
        None,
        (
            WarningScenario(
                "reference",
                "rule_set",
                (EarlyWarningRule("balance-positive", 0, "high", condition),),
            ),
        ),
        "reference",
        (("high", 1),),
        (
            LifecycleState("current", 0, 0, condition),
            LifecycleState("unknown", 0, 1, condition),
        ),
        "current",
        "unknown",
    )
    result = run_v02_workflow(
        V02WorkflowRequest(
            data=frame,
            preloan=V02PreLoanRequest(strategy),
            postloan=V02PostLoanRequest(monitoring),
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "v02-combined.md"
    artifact = generate_v02_report(result, report_path)
    assert artifact.path.is_file()
    assert report_path.with_name("v02-combined_assets").is_dir()
    return artifact.path


def main() -> None:
    """Write the report to a caller directory or an external temporary folder."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.output_dir is not None:
        report_path = _run(args.output_dir)
        print(f"combined report completed: {report_path}")
        return
    with TemporaryDirectory(prefix="sharper-v02-combined-") as directory:
        report_path = _run(Path(directory))
        print(f"combined report completed: {report_path}")


if __name__ == "__main__":
    main()
