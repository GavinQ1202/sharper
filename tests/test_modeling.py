"""Task 11 split-first classification training contracts."""
# ruff: noqa: E501

from dataclasses import fields, replace

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from sharper import (
    RegressionTrainingResult,
    TrainingResult,
    evaluate_regressor,
    modeling,
    train_classifier,
    train_regressor,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "number": [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
            "segment": ["a", "b"] * 6,
            "future_score": list(range(100, 112)),
            "target": [0] * 6 + [1] * 6,
        },
        index=[5, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    )


def test_train_classifier_returns_train_only_fitted_snapshot() -> None:
    frame = _frame()
    result = train_classifier(
        frame, "target", exclude_columns=["future_score"], test_size=0.25
    )

    assert [field.name for field in fields(TrainingResult)] == [
        "task",
        "target",
        "feature_columns",
        "excluded_columns",
        "time_column",
        "schema",
        "pipeline",
        "estimator",
        "classes",
        "train_row_positions",
        "test_row_positions",
        "X_test",
        "y_test",
        "test_size",
        "random_state",
        "warnings",
        "limitations",
    ]
    assert result.task == "classification"
    assert result.feature_columns == ("number", "segment")
    assert tuple(result.X_test.columns) == result.feature_columns
    assert set(result.train_row_positions).isdisjoint(result.test_row_positions)
    assert sorted((*result.train_row_positions, *result.test_row_positions)) == list(
        range(len(frame))
    )
    assert result.estimator is result.pipeline.named_steps["estimator"]
    assert "duplicate_index" in result.warnings
    pd.testing.assert_frame_equal(frame, _frame())


def test_exclusions_and_time_risk_are_checked_before_training() -> None:
    frame = _frame()
    with pytest.raises(ValueError, match="target must not appear in exclude_columns"):
        train_classifier(frame, "target", exclude_columns=["target"])
    with pytest.raises(
        ValueError, match="time-ordered classification is not supported"
    ):
        train_classifier(frame, "target", time_column="number")
    dated = frame.assign(observed_at=pd.date_range("2024-01-01", periods=len(frame)))
    with pytest.raises(
        ValueError, match="time-ordered classification is not supported"
    ):
        train_classifier(dated, "target")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"features": "number"}, "target column not found: 'missing'"),
        ({"exclude_columns": "number"}, "target column not found: 'missing'"),
        ({"time_column": "number"}, "target column not found: 'missing'"),
    ],
)
def test_target_existence_precedes_nested_column_arguments(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=f"^{message}$"):
        train_classifier(_frame(), "missing", **kwargs)  # type: ignore[arg-type]


def test_scalar_and_dataframe_validation_precedence() -> None:
    with pytest.raises(
        ValueError,
        match="^test_size must permit a stratified split strictly between 0 and 1$",
    ):
        train_classifier(_frame(), "missing", test_size=True)
    duplicate = pd.DataFrame([[0, 0, 1, 1]], columns=["x", "x", "target", "z"])
    with pytest.raises(
        ValueError, match="^DataFrame column names must be unique strings$"
    ):
        train_classifier(duplicate, "missing")
    with pytest.raises(ValueError, match="^df must be a pandas DataFrame$"):
        train_classifier([], 1)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "features",
    ["number", ["number", "number"], ["missing"]],
)
def test_time_risk_precedes_feature_validation(features: object) -> None:
    frame = _frame().assign(time=range(len(_frame())))
    with pytest.raises(
        ValueError, match="^time-ordered classification is not supported$"
    ):
        train_classifier(frame, "target", time_column="time", features=features)  # type: ignore[arg-type]


def test_datetime_risk_precedes_feature_validation() -> None:
    frame = _frame().assign(observed_at=pd.date_range("2024-01-01", periods=12))
    with pytest.raises(
        ValueError, match="^time-ordered classification is not supported$"
    ):
        train_classifier(frame, "target", features="number")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("target", "message"),
    [
        (
            [0] * 12,
            "classification target must contain at least two classes with two rows "
            "each",
        ),
        (
            [0, 0, 0, 0, 0, None, 1, 1, 1, 1, 1, 1],
            "classification target labels must be complete homogeneous scalar values",
        ),
        (
            [0, 0, 0, 0, 0, 0, "x", "x", "x", "x", "x", "x"],
            "classification target labels must be complete homogeneous scalar values",
        ),
    ],
)
def test_target_label_boundaries_have_stable_errors(
    target: list[object], message: str
) -> None:
    frame = _frame().assign(target=target)
    with pytest.raises(ValueError, match=f"^{message}$"):
        train_classifier(frame, "target")


def test_infer_schema_receives_only_one_training_partition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _frame().reset_index(drop=True)
    received: list[pd.DataFrame] = []
    original = modeling.infer_schema

    def spy(df: pd.DataFrame) -> object:
        received.append(df.copy(deep=True))
        return original(df)

    monkeypatch.setattr(modeling, "infer_schema", spy)
    result = train_classifier(
        frame, "target", exclude_columns=["future_score"], test_size=0.25
    )

    assert len(received) == 1
    assert set(received[0].index) == set(result.train_row_positions)
    assert set(received[0].index).isdisjoint(result.test_row_positions)
    assert tuple(received[0].columns) == ("number", "segment")


def test_holdout_only_changes_do_not_make_train_constant_feature_eligible() -> None:
    frame = _frame().assign(train_constant=1)
    first = train_classifier(
        frame, "target", exclude_columns=["future_score"], test_size=0.25
    )
    changed = frame.copy(deep=True)
    changed.iloc[
        list(first.test_row_positions), changed.columns.get_loc("train_constant")
    ] = list(range(len(first.test_row_positions)))
    second = train_classifier(
        changed, "target", exclude_columns=["future_score"], test_size=0.25
    )

    assert first.test_row_positions == second.test_row_positions
    assert "train_constant" not in first.feature_columns
    assert "train_constant" not in second.feature_columns


class _NoScoreClassifier(ClassifierMixin, BaseEstimator):
    _estimator_type = "classifier"

    def fit(self, X: object, y: list[int]) -> "_NoScoreClassifier":
        self.classes_ = [0, 1]
        return self

    def predict(self, X: object) -> list[int]:
        return [0] * len(X)  # type: ignore[arg-type]


class _FailFitClassifier(_NoScoreClassifier):
    def fit(self, X: object, y: list[int]) -> "_FailFitClassifier":
        raise RuntimeError("fit boom")


class _RecordingClassifier(_NoScoreClassifier):
    fit_calls: list[tuple[object, tuple[int, ...]]] = []

    def fit(self, X: object, y: list[int]) -> "_RecordingClassifier":
        type(self).fit_calls.append((X.copy(), tuple(y)))  # type: ignore[union-attr]
        self.classes_ = [0, 1]
        self.fit_state_ = (X.copy(), tuple(y))  # type: ignore[union-attr]
        return self


def test_custom_estimator_is_cloned_and_randomness_is_disclosed() -> None:
    estimator = _NoScoreClassifier()
    result = train_classifier(_frame(), "target", estimator=estimator, test_size=0.25)

    assert not hasattr(estimator, "classes_")
    assert result.estimator is not estimator
    assert result.warnings[-1] == "custom_estimator_random_state_not_managed"
    assert result.limitations[-1] == "custom_estimator_determinism_not_guaranteed"


def test_fit_failure_is_wrapped_and_does_not_fit_caller_estimator() -> None:
    estimator = _FailFitClassifier()
    with pytest.raises(ValueError, match="^classifier estimator fit failed$") as error:
        train_classifier(_frame(), "target", estimator=estimator, test_size=0.25)
    assert isinstance(error.value.__cause__, RuntimeError)
    assert not hasattr(estimator, "classes_")


def test_holdout_x_does_not_change_estimator_fit_input() -> None:
    frame = _frame().reset_index(drop=True)
    first_estimator = _RecordingClassifier()
    _RecordingClassifier.fit_calls = []
    first = train_classifier(
        frame,
        "target",
        estimator=first_estimator,
        exclude_columns=["future_score"],
        test_size=0.25,
    )
    changed = frame.copy(deep=True)
    changed.iloc[list(first.test_row_positions), changed.columns.get_loc("number")] = (
        999
    )
    second = train_classifier(
        changed,
        "target",
        estimator=_RecordingClassifier(),
        exclude_columns=["future_score"],
        test_size=0.25,
    )

    first_x, first_y = _RecordingClassifier.fit_calls[0]
    second_x, second_y = _RecordingClassifier.fit_calls[1]
    assert first.train_row_positions == second.train_row_positions
    assert first.feature_columns == second.feature_columns
    assert first_y == second_y
    np.testing.assert_allclose(first_x, second_x)
    np.testing.assert_allclose(
        first.estimator.fit_state_[0], second.estimator.fit_state_[0]
    )


def test_fixed_split_holdout_y_does_not_change_estimator_fit_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _frame().reset_index(drop=True)
    train_positions = np.array([0, 1, 2, 3, 6, 7, 8, 9])
    test_positions = np.array([4, 5, 10, 11])
    monkeypatch.setattr(
        modeling,
        "train_test_split",
        lambda *args, **kwargs: (train_positions, test_positions),
    )
    _RecordingClassifier.fit_calls = []
    first = train_classifier(frame, "target", estimator=_RecordingClassifier())
    changed = frame.copy(deep=True)
    changed.iloc[test_positions, changed.columns.get_loc("target")] = [1, 1, 0, 0]
    second = train_classifier(changed, "target", estimator=_RecordingClassifier())

    first_x, first_y = _RecordingClassifier.fit_calls[0]
    second_x, second_y = _RecordingClassifier.fit_calls[1]
    assert (
        first.train_row_positions
        == second.train_row_positions
        == tuple(train_positions)
    )
    assert first_y == second_y
    np.testing.assert_allclose(first_x, second_x)
    assert first.estimator.fit_state_[1] == second.estimator.fit_state_[1]


def test_training_snapshot_and_fitted_clone_are_input_independent() -> None:
    frame = _frame().reset_index(drop=True)
    estimator = _NoScoreClassifier()
    result = train_classifier(frame, "target", estimator=estimator, test_size=0.25)
    snapshot = result.X_test.copy(deep=True)
    frame.loc[:, "number"] = -1
    estimator.changed_after_training = True

    pd.testing.assert_frame_equal(result.X_test, snapshot)
    assert not hasattr(result.estimator, "changed_after_training")


def _regression_frame() -> pd.DataFrame:
    values = np.tile(np.arange(4, dtype=float), 6)
    return pd.DataFrame(
        {
            "x": values,
            "segment": np.tile(["a", "b"], 12),
            "future_score": np.arange(24, dtype=float),
            "target": values * 2.0 + np.tile([0.0, 0.5], 12),
        },
        index=[3, 3, *range(4, 26)],
    )


def test_train_regressor_creates_independent_train_only_snapshot() -> None:
    frame = _regression_frame()
    result = train_regressor(
        frame, "target", exclude_columns=["future_score"], test_size=0.25
    )

    assert isinstance(result, RegressionTrainingResult)
    assert result.task == "regression"
    assert result.feature_columns == ("x", "segment")
    assert tuple(result.X_test.columns) == result.feature_columns
    assert set(result.train_row_positions).isdisjoint(result.test_row_positions)
    assert sorted((*result.train_row_positions, *result.test_row_positions)) == list(
        range(len(frame))
    )
    assert result.estimator is result.pipeline.named_steps["estimator"]
    assert "duplicate_index" in result.warnings
    pd.testing.assert_frame_equal(frame, _regression_frame())


@pytest.mark.parametrize(
    ("target", "message"),
    [
        (
            [1.0] * 24,
            "regression target must contain at least two distinct values",
        ),
        (
            [1.0] * 23 + [np.nan],
            "regression target must be complete finite real numeric values",
        ),
        (
            [1.0] * 23 + [np.inf],
            "regression target must be complete finite real numeric values",
        ),
        (
            [True, False] * 12,
            "regression target must be complete finite real numeric values",
        ),
        (
            ["1.0"] * 24,
            "regression target must be complete finite real numeric values",
        ),
    ],
)
def test_regression_target_contract(target: list[object], message: str) -> None:
    frame = _regression_frame().assign(target=target)
    with pytest.raises(ValueError, match=f"^{message}$"):
        train_regressor(frame, "target")


def test_regression_infer_schema_uses_only_train_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _regression_frame().reset_index(drop=True)
    received: list[pd.DataFrame] = []
    original = modeling.infer_schema

    def spy(df: pd.DataFrame) -> object:
        received.append(df.copy(deep=True))
        return original(df)

    monkeypatch.setattr(modeling, "infer_schema", spy)
    result = train_regressor(
        frame, "target", exclude_columns=["future_score"], test_size=0.25
    )

    assert len(received) == 1
    assert set(received[0].index) == set(result.train_row_positions)
    assert tuple(received[0].columns) == ("x", "segment")


class _MeanRegressor(RegressorMixin, BaseEstimator):
    _estimator_type = "regressor"

    def fit(self, X: object, y: list[float]) -> "_MeanRegressor":
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X: object) -> np.ndarray:
        return np.full(len(X), self.mean_)  # type: ignore[arg-type]


class _TaggedPlainRegressor(BaseEstimator):
    """Contract-valid regressor without sklearn's newer estimator tags."""

    _estimator_type = "regressor"

    def fit(self, X: object, y: list[float]) -> "_TaggedPlainRegressor":
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X: object) -> np.ndarray:
        return np.full(len(X), self.mean_)  # type: ignore[arg-type]


class _FailFitRegressor(_MeanRegressor):
    def fit(self, X: object, y: list[float]) -> "_FailFitRegressor":
        raise RuntimeError("fit boom")


class _RecordingRegressor(_MeanRegressor):
    fit_calls: list[tuple[np.ndarray, tuple[float, ...]]] = []

    def fit(self, X: object, y: list[float]) -> "_RecordingRegressor":
        values = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        type(self).fit_calls.append((np.asarray(values).copy(), tuple(y)))
        self.fit_state_ = (np.asarray(values).copy(), tuple(y))
        return super().fit(X, y)


def test_regression_custom_estimator_is_cloned_and_fit_failures_keep_cause() -> None:
    estimator = _MeanRegressor()
    result = train_regressor(_regression_frame(), "target", estimator=estimator)
    assert result.estimator is not estimator
    assert not hasattr(estimator, "mean_")
    assert result.warnings[-1] == "custom_estimator_random_state_not_managed"
    assert result.limitations[-1] == "custom_estimator_determinism_not_guaranteed"

    with pytest.raises(ValueError, match="^regressor estimator fit failed$") as error:
        train_regressor(_regression_frame(), "target", estimator=_FailFitRegressor())
    assert isinstance(error.value.__cause__, RuntimeError)


def test_tagged_cloneable_regressor_does_not_require_new_sklearn_tags() -> None:
    result = train_regressor(
        _regression_frame(), "target", estimator=_TaggedPlainRegressor()
    )
    assert result.estimator._estimator_type == "regressor"


def test_regression_time_and_split_validation_are_stable() -> None:
    frame = _regression_frame()
    with pytest.raises(ValueError, match="^time-ordered regression is not supported$"):
        train_regressor(frame, "target", time_column="x")
    with pytest.raises(
        ValueError, match="^test_size must produce at least two train and holdout rows$"
    ):
        train_regressor(frame.iloc[:4], "target", test_size=0.2)


def _fixed_regression_split(
    monkeypatch: pytest.MonkeyPatch, train: list[int], holdout: list[int]
) -> None:
    monkeypatch.setattr(
        modeling,
        "train_test_split",
        lambda *args, **kwargs: (np.asarray(train), np.asarray(holdout)),
    )


def test_regression_holdout_id_like_and_constant_changes_do_not_leak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train, holdout = list(range(8)), list(range(8, 12))
    _fixed_regression_split(monkeypatch, train, holdout)
    frame = pd.DataFrame(
        {
            "stable": [0, 1] * 6,
            "candidate": [0, 1] * 4 + [0, 1, 0, 1],
            "train_constant": [1] * 8 + [1, 2, 3, 4],
            "target": [float(i) for i in range(12)],
        }
    )
    first = train_regressor(frame, "target", estimator=_RecordingRegressor())
    changed = frame.copy(deep=True)
    changed.loc[holdout, "candidate"] = [100, 101, 102, 103]
    changed.loc[holdout, "train_constant"] = [99, 98, 97, 96]
    second = train_regressor(changed, "target", estimator=_RecordingRegressor())

    assert first.train_row_positions == second.train_row_positions == tuple(train)
    assert first.test_row_positions == second.test_row_positions == tuple(holdout)
    assert first.feature_columns == second.feature_columns
    assert "train_constant" not in first.feature_columns
    np.testing.assert_allclose(
        first.estimator.fit_state_[0], second.estimator.fit_state_[0]
    )
    assert first.estimator.fit_state_[1] == second.estimator.fit_state_[1]


def test_regression_holdout_x_y_and_index_do_not_change_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train, holdout = list(range(8)), list(range(8, 12))
    _fixed_regression_split(monkeypatch, train, holdout)
    frame = pd.DataFrame(
        {
            "x": [0, 1] * 6,
            "segment": ["a", "b"] * 6,
            "target": list(map(float, range(12))),
        },
        index=[1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
    )
    first = train_regressor(frame, "target", estimator=_RecordingRegressor())
    changed = frame.copy(deep=True)
    changed.loc[changed.index[holdout], "x"] = 999
    changed.iloc[holdout, changed.columns.get_loc("target")] = [20.0, 21.0, 22.0, 23.0]
    changed.index = list(reversed(range(len(changed))))
    second = train_regressor(changed, "target", estimator=_RecordingRegressor())

    assert first.train_row_positions == second.train_row_positions == tuple(train)
    assert first.feature_columns == second.feature_columns
    np.testing.assert_allclose(
        first.estimator.fit_state_[0], second.estimator.fit_state_[0]
    )
    assert first.estimator.fit_state_[1] == second.estimator.fit_state_[1]
    assert first.y_test != second.y_test
    assert first.X_test.iloc[:, 0].tolist() != second.X_test.iloc[:, 0].tolist()


@pytest.mark.parametrize(
    ("frame", "kwargs", "message"),
    [
        (
            _regression_frame(),
            {"test_size": True},
            "test_size must be strictly between 0 and 1",
        ),
        (
            pd.DataFrame([[1, 2]], columns=["x", "x"]),
            {},
            "DataFrame column names must be unique strings",
        ),
        (
            _regression_frame(),
            {"features": ["missing"]},
            "feature column not found: 'missing'",
        ),
        (
            _regression_frame(),
            {"time_column": "x", "features": ["missing"]},
            "time-ordered regression is not supported",
        ),
        (
            _regression_frame().assign(at=pd.date_range("2024-01-01", periods=24)),
            {"features": ["missing"]},
            "time-ordered regression is not supported",
        ),
        (
            _regression_frame().assign(target=[1.0] * 24),
            {"exclude_columns": "x"},
            "exclude_columns must be a sequence of unique column names",
        ),
        (
            _regression_frame().assign(target=["bad"] * 24),
            {"estimator": object()},
            "regression target must be complete finite real numeric values",
        ),
        (
            _regression_frame().assign(target=[1.0] * 24),
            {"estimator": object()},
            "regression target must contain at least two distinct values",
        ),
    ],
)
def test_regression_validation_precedence(
    frame: pd.DataFrame, kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=f"^{message}$"):
        train_regressor(
            frame, "missing" if "test_size" in kwargs else "target", **kwargs
        )  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "target",
    [
        pd.Series([1, 2] * 12, dtype="Int64"),
        pd.Series([1.0, 2.0] * 12, dtype="Float64"),
    ],
)
def test_nullable_regression_targets_are_accepted(target: pd.Series) -> None:
    frame = _regression_frame().reset_index(drop=True).assign(target=target)
    assert train_regressor(frame, "target").task == "regression"


@pytest.mark.parametrize(
    "target",
    [
        pd.Series([1.0] * 23 + [pd.NA], dtype="Float64"),
        np.arange(24, dtype=complex),
        pd.Series([float(i) for i in range(24)], dtype="object"),
        [1, "x"] * 12,
        [True, 1] * 12,
    ],
)
def test_regression_target_additional_rejections(target: object) -> None:
    with pytest.raises(
        ValueError,
        match="^regression target must be complete finite real numeric values$",
    ):
        train_regressor(_regression_frame().assign(target=target), "target")


def test_regression_training_result_tampering_has_one_stable_error() -> None:
    result = train_regressor(_regression_frame(), "target")
    replacements = [
        {"task": "classification"},
        {"feature_columns": ("x", "x")},
        {"train_row_positions": (0, 0, *result.train_row_positions[2:])},
        {
            "test_row_positions": (result.test_row_positions[0],)
            * len(result.test_row_positions)
        },
        {"X_test": result.X_test.iloc[:, ::-1]},
        {"X_test": result.X_test.iloc[:-1]},
        {"y_test": result.y_test[:-1]},
        {"y_test": (float("nan"), *result.y_test[1:])},
        {"excluded_columns": (*result.excluded_columns, result.feature_columns[0])},
        {"random_state": -1},
        {"warnings": ("unknown",)},
    ]
    for replacement in replacements:
        with pytest.raises(
            ValueError, match="^regression training result has invalid schema$"
        ):
            evaluate_regressor(replace(result, **replacement))
