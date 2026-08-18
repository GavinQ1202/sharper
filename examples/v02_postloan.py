"""Run a deterministic offline Task 20 post-loan warning example."""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from sharper import (
    EarlyWarningRule,
    LifecycleMonitoringConfig,
    LifecycleState,
    MonitoringCondition,
    V02PostLoanRequest,
    V02WorkflowRequest,
    WarningScenario,
    run_v02_workflow,
)


def main() -> None:
    """Evaluate a caller-frozen warning scenario on offline observations."""
    frame = pd.DataFrame(
        {
            "entity": ["entity-a", "entity-b"],
            "observed": [datetime(2025, 1, 1), datetime(2025, 1, 2)],
            "available": [datetime(2025, 1, 1), datetime(2025, 1, 2)],
            "balance": [100, 0],
        }
    )
    condition = MonitoringCondition("atomic", "gt", "column", "balance", "literal", 0)
    config = LifecycleMonitoringConfig(
        "example-postloan",
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
            postloan=V02PostLoanRequest(config),
        )
    )
    assert result.postloan is not None
    print(f"post-loan monitoring completed; entities={result.postloan.entity_count}")


if __name__ == "__main__":
    main()
