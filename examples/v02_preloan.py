"""Run a deterministic offline Task 20 pre-loan strategy example."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from sharper import (
    DecisionStrategyConfig,
    V02PreLoanRequest,
    V02WorkflowRequest,
    run_v02_workflow,
)


def main() -> None:
    """Simulate a caller-frozen strategy without executing any action."""
    frame = pd.DataFrame({"income": [40_000, 80_000], "balance": [1_000, 2_000]})
    config = DecisionStrategyConfig(
        "example-preloan",
        "v1",
        datetime(2025, 1, 1),
        None,
        datetime(2025, 1, 2),
        (),
        "review",
        "review",
        (("review", "review"),),
    )
    result = run_v02_workflow(
        V02WorkflowRequest(
            data=frame,
            preloan=V02PreLoanRequest(config),
        )
    )
    assert result.preloan is not None
    print(
        f"pre-loan simulation completed; decided rows={result.preloan.decided_n_rows}"
    )


if __name__ == "__main__":
    main()
