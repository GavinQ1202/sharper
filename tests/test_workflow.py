"""Contract tests for the minimal Task 05 workflow."""

from dataclasses import FrozenInstanceError, fields
from typing import get_type_hints

import pandas as pd
import pytest

from sharper import AnalysisRun, run_analysis


def test_analysis_run_frozen_fields_and_types() -> None:
    assert [field.name for field in fields(AnalysisRun)] == [
        "schema",
        "summary",
        "quality",
        "target",
        "task",
        "include_model",
        "id_columns",
        "exclude_columns",
        "random_state",
        "skipped",
        "warnings",
    ]
    assert get_type_hints(AnalysisRun)["id_columns"] == tuple[str, ...]
    run = run_analysis(pd.DataFrame({"value": [1, 2]}))
    with pytest.raises(FrozenInstanceError):
        run.target = "value"  # type: ignore[misc]


def test_run_analysis_normal_case_composes_results() -> None:
    run = run_analysis(pd.DataFrame({"value": [1, 2], "group": ["a", "b"]}))
    assert run.schema.n_rows == run.summary.n_rows == run.quality.n_rows == 2
    assert run.target is None
    assert run.task is None
    assert run.include_model is False
    assert run.skipped == ("modeling", "visualization", "feature_engineering")
    assert run.warnings == ()


def test_run_analysis_records_target_task_and_column_options() -> None:
    run = run_analysis(
        pd.DataFrame({"id": [1, 2], "feature": [3, 4], "target": [0, 1]}),
        target="target",
        task="classification",
        id_columns=["id"],
        exclude_columns=["feature"],
        random_state=7,
    )
    assert run.target == "target"
    assert run.task == "classification"
    assert run.id_columns == ("id",)
    assert run.exclude_columns == ("feature",)
    assert run.random_state == 7
    assert run.schema.target_candidates[0].name == "target"
    assert run.warnings == (
        "target recorded but target analysis is not available in Task 05",
        "task recorded but modeling is not available in Task 05",
        "id_columns recorded but not applied in Task 05",
        "exclude_columns recorded but not applied in Task 05",
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"target": "missing"}, "target column not found"),
        ({"target": "target", "task": "clustering"}, "task must be"),
        ({"task": "classification"}, "task requires target"),
        ({"include_model": True}, "modeling is not available"),
        ({"id_columns": ["missing"]}, "column not found"),
        ({"exclude_columns": ["missing"]}, "column not found"),
        ({"id_columns": ["id", "id"]}, "duplicate column parameter"),
        ({"exclude_columns": ["id", "id"]}, "duplicate column parameter"),
        (
            {"id_columns": ["id"], "exclude_columns": ["id"]},
            "id_columns and exclude_columns must not overlap",
        ),
        ({"random_state": 1.5}, "random_state must be an integer"),
    ],
)
def test_run_analysis_rejects_invalid_options(
    kwargs: dict[str, object],
    message: str,
) -> None:
    frame = pd.DataFrame({"id": [1, 2], "target": [0, 1]})
    with pytest.raises(ValueError, match=message):
        run_analysis(frame, **kwargs)  # type: ignore[arg-type]


def test_run_analysis_does_not_mutate_input() -> None:
    frame = pd.DataFrame({"id": [1, 2], "value": [3.0, None]})
    original = frame.copy(deep=True)
    run_analysis(frame, id_columns=["id"])
    pd.testing.assert_frame_equal(frame, original)


def test_run_analysis_rejects_non_dataframe_and_non_string_columns() -> None:
    with pytest.raises(ValueError, match="df must be a pandas DataFrame"):
        run_analysis([1, 2])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="column names must all be strings"):
        run_analysis(pd.DataFrame({1: [1]}))
