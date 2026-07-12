# ruff: noqa: E501

from dataclasses import FrozenInstanceError, fields
from functools import wraps

import pandas as pd
import pytest

from sharper import AnalysisRun, run_analysis, workflow


def test_analysis_run_task13_field_order_and_frozen() -> None:
    names = [field.name for field in fields(AnalysisRun)]
    assert names[:15] == [
        "schema",
        "summary",
        "quality",
        "target",
        "task",
        "include_model",
        "id_columns",
        "exclude_columns",
        "features",
        "time_column",
        "group_by",
        "reference_date",
        "max_suggestions",
        "test_size",
        "random_state",
    ]
    run = run_analysis(pd.DataFrame({"x": [1, 2], "g": ["a", "b"]}))
    with pytest.raises(FrozenInstanceError):
        run.target = "x"  # type: ignore[misc]


def test_analysis_only_workflow_composes_all_non_target_results() -> None:
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "g": ["a", "a", "b", "b"]})
    run = run_analysis(frame, group_by="g")
    assert run.schema.n_rows == 4
    assert run.numeric_analysis is not None and run.feature_suggestions is not None
    assert run.group_comparison is not None and run.target_analysis is None
    assert run.training is run.evaluation is None
    assert "modeling_not_requested" in run.skipped
    pd.testing.assert_frame_equal(
        frame, pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "g": ["a", "a", "b", "b"]})
    )


def test_target_analysis_without_model() -> None:
    run = run_analysis(
        pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [0, 0, 1, 1]}),
        target="y",
        task="classification",
    )
    assert run.target_analysis is not None
    assert run.training is None


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"task": "bad"}, "task must be"),
        ({"include_model": True}, "modeling requires"),
        ({"features": ("x",)}, "features require"),
        ({"id_columns": ("missing",)}, "column not found"),
    ],
)
def test_workflow_validation(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        run_analysis(pd.DataFrame({"x": [1, 2]}), **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("task", "trainer", "other_trainer", "plotter"),
    [
        (
            "classification",
            "train_classifier",
            "train_regressor",
            "plot_classification_evaluation",
        ),
        (
            "regression",
            "train_regressor",
            "train_classifier",
            "plot_regression_evaluation",
        ),
    ],
)
def test_workflow_calls_each_public_api_once_in_contract_order(
    monkeypatch: pytest.MonkeyPatch,
    task: str,
    trainer: str,
    other_trainer: str,
    plotter: str,
) -> None:
    calls: list[str] = []
    names = [
        "infer_schema",
        "summarize_dataframe",
        "check_data_quality",
        "analyze_numeric_features",
        "analyze_categorical_features",
        "compute_correlations",
        "detect_outliers",
        "compare_groups",
        "analyze_target_relationships",
        "suggest_feature_derivations",
        trainer,
        "evaluate_model",
        "plot_distributions",
        "plot_missingness",
        "plot_correlations",
        "plot_outliers",
        "plot_group_comparison",
        "plot_target_relationships",
        plotter,
    ]
    for name in names:
        original = getattr(workflow, name)

        @wraps(original)
        def wrapped(
            *args: object,
            __name: str = name,
            __original: object = original,
            **kwargs: object,
        ):
            calls.append(__name)
            return __original(*args, **kwargs)  # type: ignore[operator]

        monkeypatch.setattr(workflow, name, wrapped)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("wrong task trainer called")

    monkeypatch.setattr(workflow, other_trainer, forbidden)
    frame = pd.DataFrame(
        {
            "x": range(30),
            "g": ["a", "b"] * 15,
            "y": [i % 2 for i in range(30)]
            if task == "classification"
            else [float(i) for i in range(30)],
        }
    )
    original = frame.copy(deep=True)
    run = run_analysis(frame, target="y", task=task, include_model=True, group_by="g")  # type: ignore[arg-type]
    assert calls == names
    pd.testing.assert_frame_equal(frame, original)
    assert not any(value is frame for value in vars(run).values())


def test_workflow_wraps_selected_step_error_with_original_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fault = RuntimeError("numeric fault")

    def fail(*args: object, **kwargs: object) -> None:
        raise fault

    monkeypatch.setattr(workflow, "analyze_numeric_features", fail)
    with pytest.raises(
        ValueError,
        match=r"^workflow step failed: analyze_numeric_features: numeric fault$",
    ) as caught:
        run_analysis(pd.DataFrame({"x": [1.0, 2.0]}))
    assert caught.value.__cause__ is fault
