"""Run a deterministic Task 20 score-validation example with external scores."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sharper import (
    BinaryRiskValidationConfig,
    ExternalRiskPredictions,
    V02ScoreValidationRequest,
    V02WorkflowRequest,
    run_v02_workflow,
)


def main() -> None:
    """Validate fixed external ranking scores without fitting a model."""
    frame = pd.DataFrame(
        {
            "feature": list(range(12)),
            "target": [0, 1] * 6,
        }
    )
    scores = tuple(float(value) for value in np.linspace(0.05, 0.95, 12))
    predictions = ExternalRiskPredictions(
        row_positions=tuple(range(12)),
        fold_ids=(0, 1, 0, 0, 2, 0, 1, 1, 2, 2, 1, 2),
        fold_fit_row_positions=(
            (0, (1, 4, 6, 7, 8, 9, 10, 11)),
            (1, (0, 2, 3, 4, 5, 8, 9, 11)),
            (2, (0, 1, 2, 3, 5, 6, 7, 10)),
        ),
        ranking_scores=scores,
        ranking_direction="higher_risk",
        event_probabilities=None,
        probability_positive_label=None,
        probability_provenance=None,
    )
    config = BinaryRiskValidationConfig(
        validation_mode="stratified_kfold",
        n_splits=3,
        thresholds=(0.5,),
        threshold_kind="ranking_score",
    )
    result = run_v02_workflow(
        V02WorkflowRequest(
            data=frame,
            score_validation=V02ScoreValidationRequest(
                target="target",
                config=config,
                positive_label=1,
                external_predictions=predictions,
            ),
        )
    )
    assert result.score_validation is not None
    print(
        "score validation completed; "
        f"evaluable rows={result.score_validation.evaluable_n_rows}"
    )


if __name__ == "__main__":
    main()
