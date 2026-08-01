"""Task 15 binary-risk validation contract tests."""

from dataclasses import FrozenInstanceError, fields, replace
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)

from sharper import (
    BinaryRiskValidationConfig,
    BinaryRiskValidationResult,
    ExternalRiskPredictions,
    validate_binary_risk,
)


def _binary_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": [0, 1, 0, 1, 2, 1, 2, 0, 2, 1, 0, 2],
            "target": [0, 1] * 6,
        },
        index=[4, 4, 3, 3, 2, 2, 1, 1, 0, 0, -1, -1],
    )


def _kfold_external(
    target: tuple[object, ...] | list[object],
    *,
    scores: tuple[float, ...] | None = None,
    probabilities: tuple[float, ...] | None = None,
    positive_label: object = 1,
    n_splits: int = 3,
    random_state: int = 42,
) -> ExternalRiskPredictions:
    labels = np.asarray(target)
    positions = np.arange(len(labels))
    fold_by_position: dict[int, int] = {}
    fit_rows = []
    splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    for fold_id, (train, validation) in enumerate(splitter.split(positions, labels)):
        train_positions = tuple(sorted(int(value) for value in train))
        fit_rows.append((fold_id, train_positions))
        for value in validation:
            fold_by_position[int(value)] = fold_id
    row_positions = tuple(range(len(labels)))
    if scores is None and probabilities is not None:
        ranking_direction = None
    else:
        ranking_direction = "higher_risk"
    return ExternalRiskPredictions(
        row_positions=row_positions,
        fold_ids=tuple(fold_by_position[position] for position in row_positions),
        fold_fit_row_positions=tuple(fit_rows),
        ranking_scores=scores,
        ranking_direction=ranking_direction,
        event_probabilities=probabilities,
        probability_positive_label=(
            positive_label if probabilities is not None else None
        ),
        probability_provenance=(
            "external_declared" if probabilities is not None else None
        ),
    )


def _external_result(
    frame: pd.DataFrame | None = None,
    *,
    config: BinaryRiskValidationConfig | None = None,
    probabilities: tuple[float, ...] | None = None,
) -> BinaryRiskValidationResult:
    frame = _binary_frame() if frame is None else frame
    target = tuple(frame["target"].tolist())
    values = tuple(np.linspace(0.02, 0.98, len(frame)))
    probabilities = values if probabilities is None else probabilities
    external = _kfold_external(
        target,
        scores=values,
        probabilities=probabilities,
    )
    return validate_binary_risk(
        frame,
        "target",
        config=config
        or BinaryRiskValidationConfig(
            validation_mode="stratified_kfold",
            n_splits=3,
            thresholds=(0.25, 0.5),
            threshold_kind="event_probability",
            operating_metric="f1",
            calibration_bins=4,
        ),
        external_predictions=external,
    )


def test_source_cardinality_is_exact_and_has_no_default_estimator() -> None:
    frame = _binary_frame()
    config = BinaryRiskValidationConfig(validation_mode="stratified_kfold", n_splits=3)
    external = _kfold_external(
        tuple(frame["target"]), scores=tuple(np.linspace(0.0, 1.0, len(frame)))
    )
    message = "exactly one of estimator and external_predictions must be provided"
    with pytest.raises(ValueError, match=f"^{message}$"):
        validate_binary_risk(frame, "target", config=config)
    with pytest.raises(ValueError, match=f"^{message}$"):
        validate_binary_risk(
            frame,
            "target",
            config=config,
            estimator=LogisticRegression(),
            external_predictions=external,
        )


def test_external_path_is_position_aligned_feature_free_and_duplicate_index_safe() -> (
    None
):
    frame = _binary_frame().loc[:, ["target"]]
    before = frame.copy(deep=True)
    scores = tuple(float(value) for value in range(len(frame)))
    external = _kfold_external(tuple(frame["target"]), scores=scores)
    result = validate_binary_risk(
        frame,
        "target",
        config=BinaryRiskValidationConfig(
            validation_mode="stratified_kfold", n_splits=3
        ),
        external_predictions=external,
    )

    assert tuple(result.predictions["row_position"]) == tuple(range(len(frame)))
    assert tuple(result.predictions["ranking_score"]) == scores
    assert result.folds["feature_columns"].tolist() == [(), (), ()]
    assert result.warnings[:2] == ("duplicate_index", "duplicate_rows")
    pd.testing.assert_frame_equal(frame, before)

    with pytest.raises(
        ValueError,
        match="^binary risk validation config has invalid schema$",
    ):
        validate_binary_risk(
            frame,
            "target",
            config=BinaryRiskValidationConfig(
                validation_mode="stratified_kfold", n_splits=3
            ),
            external_predictions=external,
            features=("missing",),
        )


def test_external_lower_risk_direction_is_negated_before_hand_auc() -> None:
    frame = _binary_frame().reset_index(drop=True)
    raw_scores = tuple(float(value) for value in range(len(frame)))
    external = replace(
        _kfold_external(tuple(frame["target"]), scores=raw_scores),
        ranking_direction="lower_risk",
    )
    result = validate_binary_risk(
        frame,
        "target",
        config=BinaryRiskValidationConfig(
            validation_mode="stratified_kfold", n_splits=3
        ),
        external_predictions=external,
    )

    assert result.predictions["ranking_score"].tolist() == [
        -value for value in raw_scores
    ]
    auc = result.metrics.loc[
        (result.metrics["scope"] == "overall") & (result.metrics["metric"] == "roc_auc")
    ].iloc[0]
    # Fifteen of the 36 positive/negative pairs are correctly ordered.
    assert auc["value"] == pytest.approx(15.0 / 36.0)


@pytest.mark.parametrize(
    "replacement",
    [
        {"row_positions": tuple(range(11))},
        {"row_positions": (*range(12), 12)},
        {"row_positions": (0, 0, *range(2, 12))},
        {"fold_ids": (0,) * 12},
        {"fold_fit_row_positions": ()},
    ],
)
def test_external_missing_duplicate_extra_and_fold_provenance_fail(
    replacement: dict[str, object],
) -> None:
    from dataclasses import replace

    frame = _binary_frame()
    external = _kfold_external(
        tuple(frame["target"]), scores=tuple(np.linspace(0.0, 1.0, len(frame)))
    )
    with pytest.raises(
        ValueError,
        match="^external prediction provenance does not match validation plan$",
    ):
        validate_binary_risk(
            frame,
            "target",
            config=BinaryRiskValidationConfig(
                validation_mode="stratified_kfold", n_splits=3
            ),
            external_predictions=replace(external, **replacement),
        )


@pytest.mark.parametrize(
    ("labels", "positive", "expected"),
    [
        ([np.int64(0), np.int64(1)] * 6, np.int64(1), 1),
        ([np.bool_(False), np.bool_(True)] * 6, np.bool_(True), True),
        ([np.str_("no"), np.str_("yes")] * 6, np.str_("yes"), "yes"),
    ],
)
def test_numpy_label_scalars_normalize_before_exact_matching(
    labels: list[object], positive: object, expected: object
) -> None:
    frame = pd.DataFrame({"target": pd.Series(labels, dtype="object")})
    probabilities = tuple(np.linspace(0.05, 0.95, len(frame)))
    result = validate_binary_risk(
        frame,
        "target",
        positive_label=positive,
        config=BinaryRiskValidationConfig(
            validation_mode="stratified_kfold", n_splits=3
        ),
        external_predictions=_kfold_external(
            labels,
            probabilities=probabilities,
            positive_label=positive,
        ),
    )
    assert type(result.positive_label) is type(expected)
    assert result.positive_label == expected


def test_float_labels_string_inference_and_bool_int_mixing_are_rejected() -> None:
    config = BinaryRiskValidationConfig(validation_mode="stratified_kfold", n_splits=2)
    cases = [
        ([0.0, 1.0] * 4, None, "homogeneous string, integer, or boolean"),
        (["0", "1"] * 4, None, "positive_label must be provided"),
        ([False, 1] * 4, 1, "homogeneous string, integer, or boolean"),
    ]
    for labels, positive, message in cases:
        frame = pd.DataFrame({"target": pd.Series(labels, dtype="object")})
        external = _kfold_external(
            labels, scores=tuple(np.linspace(0.0, 1.0, len(labels))), n_splits=2
        )
        with pytest.raises(ValueError, match=message):
            validate_binary_risk(
                frame,
                "target",
                positive_label=positive,
                config=config,
                external_predictions=external,
            )


class _RecordingClassifier(ClassifierMixin, BaseEstimator):
    fit_sizes: list[int] = []

    def fit(self, X: object, y: object) -> "_RecordingClassifier":
        type(self).fit_sizes.append(len(X))  # type: ignore[arg-type]
        self.fit_token_ = object()
        self.fit_n_ = len(X)  # type: ignore[arg-type]
        self.classes_ = np.asarray([0, 1])
        self.fitted_ = True
        return self

    def predict_proba(self, X: object) -> np.ndarray:
        n_rows = len(X)  # type: ignore[arg-type]
        return np.tile(np.asarray([[0.4, 0.6]]), (n_rows, 1))


class _ReverseProbabilityAndDecisionClassifier(ClassifierMixin, BaseEstimator):
    decision_calls = 0

    def fit(self, X: object, y: object) -> "_ReverseProbabilityAndDecisionClassifier":
        self.classes_ = np.asarray([1, 0])
        return self

    def predict_proba(self, X: object) -> np.ndarray:
        return np.tile(np.asarray([[0.8, 0.2]]), (len(X), 1))  # type: ignore[arg-type]

    def decision_function(self, X: object) -> np.ndarray:
        type(self).decision_calls += 1
        return np.full(len(X), 99.0)  # type: ignore[arg-type]


class _ReverseDecisionClassifier(ClassifierMixin, BaseEstimator):
    def fit(self, X: object, y: object) -> "_ReverseDecisionClassifier":
        self.classes_ = np.asarray([1, 0])
        return self

    def decision_function(self, X: object) -> np.ndarray:
        return np.arange(len(X), dtype=float) + 0.25  # type: ignore[arg-type]


class _TransformedRowRecorder(ClassifierMixin, BaseEstimator):
    fit_values: list[tuple[float, ...]] = []
    fit_tokens: list[object] = []

    def fit(self, X: object, y: object) -> "_TransformedRowRecorder":
        values = np.asarray(X, dtype=float)
        type(self).fit_values.append(tuple(float(value) for value in values[:, 0]))
        token = object()
        type(self).fit_tokens.append(token)
        self.fit_token_ = token
        self.classes_ = np.asarray([0, 1])
        return self

    def predict_proba(self, X: object) -> np.ndarray:
        return np.tile(np.asarray([[0.4, 0.6]]), (len(X), 1))  # type: ignore[arg-type]


def test_estimator_is_cloned_and_fit_once_per_fold_without_mutating_caller() -> None:
    _RecordingClassifier.fit_sizes.clear()
    estimator = _RecordingClassifier()
    result = validate_binary_risk(
        _binary_frame(),
        "target",
        config=BinaryRiskValidationConfig(
            validation_mode="stratified_kfold", n_splits=3
        ),
        estimator=estimator,
        features=("feature",),
    )
    assert _RecordingClassifier.fit_sizes == [8, 8, 8]
    assert not hasattr(estimator, "fitted_")
    assert result.score_source == "estimator_predict_proba"
    assert result.folds["feature_columns"].tolist() == [
        ("feature",),
        ("feature",),
        ("feature",),
    ]

    ordered = _binary_frame().assign(second=[2, 0, 1, 2] * 3)
    reordered = validate_binary_risk(
        ordered,
        "target",
        config=BinaryRiskValidationConfig(
            validation_mode="stratified_kfold", n_splits=3
        ),
        estimator=_RecordingClassifier(),
        features=("second", "feature"),
    )
    assert reordered.folds["feature_columns"].tolist() == [
        ("second", "feature"),
        ("second", "feature"),
        ("second", "feature"),
    ]


def test_reversed_estimator_classes_select_positive_probability_and_prefer_proba() -> (
    None
):
    _ReverseProbabilityAndDecisionClassifier.decision_calls = 0
    result = validate_binary_risk(
        _binary_frame(),
        "target",
        config=BinaryRiskValidationConfig(
            validation_mode="stratified_kfold", n_splits=3
        ),
        estimator=_ReverseProbabilityAndDecisionClassifier(),
        features=("feature",),
    )

    assert result.score_source == "estimator_predict_proba"
    assert result.probability_provenance == "predict_proba"
    assert result.predictions["event_probability"].tolist() == pytest.approx([0.8] * 12)
    assert result.predictions["ranking_score"].tolist() == pytest.approx([0.8] * 12)
    assert _ReverseProbabilityAndDecisionClassifier.decision_calls == 0


def test_reversed_decision_classes_negate_margin_without_creating_probability() -> None:
    result = validate_binary_risk(
        _binary_frame(),
        "target",
        config=BinaryRiskValidationConfig(
            validation_mode="stratified_kfold", n_splits=3
        ),
        estimator=_ReverseDecisionClassifier(),
        features=("feature",),
    )

    assert result.score_source == "estimator_decision_function"
    assert result.probability_provenance is None
    assert result.predictions["event_probability"].isna().all()
    for fold in result.folds.to_dict("records"):
        positions = fold["validation_row_positions"]
        actual = result.predictions.loc[
            result.predictions["row_position"].isin(positions), "ranking_score"
        ].tolist()
        expected = [-(index + 0.25) for index in range(len(positions))]
        assert actual == pytest.approx(expected)


@pytest.mark.parametrize(
    ("mode", "expected_scope"),
    [
        ("stratified_holdout", "validation"),
        ("group_holdout", "validation"),
        ("group_kfold", "oof"),
    ],
)
def test_random_and_group_validation_modes_rebuild_exact_external_plan(
    mode: str, expected_scope: str
) -> None:
    target = np.asarray([0, 1] * 6)
    positions = np.arange(len(target))
    groups = np.repeat(np.arange(6), 2)
    if mode == "stratified_holdout":
        train, validation = train_test_split(
            positions, stratify=target, test_size=0.25, random_state=42
        )
        splits = [(train, validation)]
        config = BinaryRiskValidationConfig(
            validation_mode="stratified_holdout", test_size=0.25
        )
        frame = pd.DataFrame({"target": target})
    elif mode == "group_holdout":
        splits = list(
            GroupShuffleSplit(n_splits=1, test_size=0.34, random_state=42).split(
                positions, target, groups
            )
        )
        config = BinaryRiskValidationConfig(
            validation_mode="group_holdout",
            test_size=0.34,
            group_column="group",
        )
        frame = pd.DataFrame({"target": target, "group": groups})
    else:
        splits = list(
            StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42).split(
                positions, target, groups
            )
        )
        config = BinaryRiskValidationConfig(
            validation_mode="group_kfold", n_splits=3, group_column="group"
        )
        frame = pd.DataFrame({"target": target, "group": groups})
    fold_by_position: dict[int, int] = {}
    fit_rows = []
    predicted: list[int] = []
    for fold_id, (train, validation) in enumerate(splits):
        train_tuple = tuple(sorted(int(value) for value in train))
        validation_tuple = tuple(sorted(int(value) for value in validation))
        fit_rows.append((fold_id, train_tuple))
        predicted.extend(validation_tuple)
        for position in validation_tuple:
            fold_by_position[position] = fold_id
    row_positions = tuple(sorted(predicted))
    result = validate_binary_risk(
        frame,
        "target",
        config=config,
        external_predictions=ExternalRiskPredictions(
            row_positions=row_positions,
            fold_ids=tuple(fold_by_position[value] for value in row_positions),
            fold_fit_row_positions=tuple(fit_rows),
            ranking_scores=tuple(float(value) for value in row_positions),
            ranking_direction="higher_risk",
            event_probabilities=None,
            probability_positive_label=None,
            probability_provenance=None,
        ),
    )
    assert result.validation_mode == mode
    assert result.prediction_scope == expected_scope
    assert tuple(result.predictions["row_position"]) == row_positions
    assert result.folds["feature_columns"].tolist() == [()] * len(splits)


def _time_result() -> BinaryRiskValidationResult:
    observation = pd.date_range("2026-01-01", periods=8, freq="D")
    frame = pd.DataFrame(
        {
            "target": [0, 1] * 4,
            "observation": observation,
            "outcome_end": observation + pd.Timedelta(days=2),
        },
        index=[0, 0, 1, 1, 2, 2, 3, 3],
    )
    config = BinaryRiskValidationConfig(
        validation_mode="time_forward",
        observation_time_column="observation",
        outcome_end_time_column="outcome_end",
        maturity_source="outcome_end",
        reporting_delay=timedelta(days=1),
        fold_cutoffs=(datetime(2026, 1, 5), datetime(2026, 1, 7)),
        validation_end=datetime(2026, 1, 9),
        analysis_as_of=datetime(2026, 1, 9),
        calibration_bins=2,
    )
    external = ExternalRiskPredictions(
        row_positions=(4, 5, 6, 7),
        fold_ids=(0, 0, 1, 1),
        fold_fit_row_positions=((0, (0, 1)), (1, (0, 1, 2, 3))),
        ranking_scores=(0.1, 0.2, 0.8, 0.9),
        ranking_direction="higher_risk",
        event_probabilities=(0.1, 0.2, 0.8, 0.9),
        probability_positive_label=1,
        probability_provenance="external_declared",
    )
    return validate_binary_risk(
        frame, "target", config=config, external_predictions=external
    )


def test_time_outcome_definition_train_maturity_and_unevaluable_predictions() -> None:
    result = _time_result()
    assert result.folds["train_row_positions"].tolist() == [(0, 1), (0, 1, 2, 3)]
    assert result.folds["evaluable_validation_row_positions"].tolist() == [
        (4, 5),
        (),
    ]
    assert result.folds["outcome_end_source"].tolist() == ["column", "column"]
    assert result.folds["immature_validation_n"].tolist() == [0, 2]
    immature = result.predictions.loc[~result.predictions["is_evaluable"]]
    assert immature["target_value"].isna().all()
    assert set(immature["unevaluable_reason"]) == {"immature_validation_outcome"}
    empty_fold = result.metrics.loc[
        (result.metrics["scope"] == "fold") & (result.metrics["fold_id"] == 1)
    ]
    assert set(empty_fold["reason"]) == {"no_evaluable_rows"}
    assert len(result.calibration.loc[result.calibration["fold_id"] == 1]) == 2


def test_time_validation_requires_outcome_definition_and_exact_consistency() -> None:
    observation = pd.date_range("2026-01-01", periods=6, freq="D")
    frame = pd.DataFrame({"target": [0, 1] * 3, "observation": observation})
    base = dict(
        validation_mode="time_holdout",
        observation_time_column="observation",
        maturity_source="observation_horizon",
        fold_cutoffs=(datetime(2026, 1, 5),),
        validation_end=datetime(2026, 1, 7),
        analysis_as_of=datetime(2026, 1, 7),
    )
    external = ExternalRiskPredictions(
        (4, 5),
        (0, 0),
        ((0, (0, 1, 2, 3)),),
        (0.1, 0.9),
        "higher_risk",
        None,
        None,
        None,
    )
    with pytest.raises(
        ValueError, match="^label maturity metadata is missing or inconsistent$"
    ):
        validate_binary_risk(
            frame,
            "target",
            config=BinaryRiskValidationConfig(**base),
            external_predictions=external,
        )

    frame["outcome"] = observation + pd.Timedelta(days=3)
    with pytest.raises(
        ValueError, match="^label maturity metadata is missing or inconsistent$"
    ):
        validate_binary_risk(
            frame,
            "target",
            config=BinaryRiskValidationConfig(
                **base,
                outcome_end_time_column="outcome",
                prediction_horizon=timedelta(days=2),
            ),
            external_predictions=external,
        )


def test_observed_loss_uses_independent_maturity_and_skips_immature_sentinel() -> None:
    frame = _binary_frame()
    frame["exposure"] = 100.0
    frame["observed_loss"] = [10.0] * 6 + [np.inf] * 6
    frame["loss_available"] = pd.to_datetime(["2026-01-01"] * 6 + ["2026-02-01"] * 6)
    config = BinaryRiskValidationConfig(
        validation_mode="stratified_kfold",
        n_splits=3,
        analysis_as_of=datetime(2026, 1, 15),
        exposure_column="exposure",
        observed_loss_column="observed_loss",
        observed_loss_available_time_column="loss_available",
        loss_fraction=0.5,
        exposure_unit="USD",
    )
    result = validate_binary_risk(
        frame,
        "target",
        config=config,
        external_predictions=_kfold_external(
            tuple(frame["target"]), probabilities=tuple([0.2] * len(frame))
        ),
    )
    assert result.observed_loss_maturity_mode == "availability_column"
    assert (result.observed_loss_mature_n, result.observed_loss_excluded_n) == (6, 6)
    all_rows = result.business_metrics.loc[
        result.business_metrics["segment_kind"] == "all"
    ].set_index("metric")
    assert all_rows.loc["observed_loss_sum", "value"] == pytest.approx(60.0)
    assert all_rows.loc["expected_loss_sum", "value"] == pytest.approx(120.0)


def test_ranking_only_keeps_ranking_metrics_and_structures_probability_absence() -> (
    None
):
    frame = _binary_frame().reset_index(drop=True)
    frame["exposure"] = np.arange(100.0, 1300.0, 100.0)
    scores = tuple(float(value) for value in np.linspace(-2.0, 2.0, len(frame)))
    result = validate_binary_risk(
        frame,
        "target",
        config=BinaryRiskValidationConfig(
            validation_mode="stratified_kfold",
            n_splits=3,
            exposure_column="exposure",
            loss_fraction=0.5,
            exposure_unit="USD",
        ),
        external_predictions=_kfold_external(tuple(frame["target"]), scores=scores),
    )

    metrics = result.metrics.loc[result.metrics["scope"] == "overall"].set_index(
        "metric"
    )
    assert metrics.loc["roc_auc", "status"] == "available"
    for metric in ("brier_score", "log_loss", "expected_calibration_error"):
        assert (metrics.loc[metric, "status"], metrics.loc[metric, "reason"]) == (
            "unavailable",
            "probability_absent",
        )
    assert result.calibration.empty
    expected_loss = result.business_metrics.loc[
        (result.business_metrics["segment_kind"] == "all")
        & (result.business_metrics["metric"] == "expected_loss_sum")
    ].iloc[0]
    assert pd.isna(expected_loss["value"])
    assert (expected_loss["status"], expected_loss["reason"]) == (
        "unavailable",
        "probability_absent",
    )


def test_mature_snapshot_observed_loss_has_exact_provenance_and_sum() -> None:
    frame = _binary_frame().reset_index(drop=True)
    frame["observed_loss"] = np.arange(1.0, 13.0)
    result = validate_binary_risk(
        frame,
        "target",
        config=BinaryRiskValidationConfig(
            validation_mode="stratified_kfold",
            n_splits=3,
            analysis_as_of=datetime(2026, 1, 31),
            observed_loss_column="observed_loss",
            observed_loss_is_mature_snapshot=True,
            exposure_unit="USD",
        ),
        external_predictions=_kfold_external(
            tuple(frame["target"]), scores=tuple(np.linspace(0.0, 1.0, len(frame)))
        ),
    )

    assert result.observed_loss_maturity_mode == "mature_snapshot"
    assert result.observed_loss_analysis_as_of == datetime(2026, 1, 31)
    assert result.observed_loss_mature_n == len(frame)
    assert result.observed_loss_excluded_n == 0
    observed = result.business_metrics.loc[
        (result.business_metrics["segment_kind"] == "all")
        & (result.business_metrics["metric"] == "observed_loss_sum")
    ].iloc[0]
    assert observed["value"] == pytest.approx(78.0)
    assert observed["n_observed_loss_mature_rows"] == len(frame)
    assert (observed["status"], pd.isna(observed["reason"])) == ("available", True)


def test_loss_fraction_column_and_scalar_expected_loss_are_hand_calculated() -> None:
    frame = _binary_frame().reset_index(drop=True)
    frame["exposure"] = np.arange(10.0, 130.0, 10.0)
    frame["loss_fraction"] = np.linspace(0.1, 0.6, len(frame))
    probabilities = tuple(float(value) for value in np.linspace(0.05, 0.95, len(frame)))
    external = _kfold_external(tuple(frame["target"]), probabilities=probabilities)
    column_result = validate_binary_risk(
        frame,
        "target",
        config=BinaryRiskValidationConfig(
            validation_mode="stratified_kfold",
            n_splits=3,
            exposure_column="exposure",
            loss_fraction="loss_fraction",
            exposure_unit="USD",
        ),
        external_predictions=external,
    )
    scalar_result = validate_binary_risk(
        frame,
        "target",
        config=BinaryRiskValidationConfig(
            validation_mode="stratified_kfold",
            n_splits=3,
            exposure_column="exposure",
            loss_fraction=0.5,
            exposure_unit="USD",
        ),
        external_predictions=external,
    )

    expected_column = float(
        sum(
            probability * exposure * fraction
            for probability, exposure, fraction in zip(
                probabilities,
                frame["exposure"],
                frame["loss_fraction"],
                strict=True,
            )
        )
    )
    expected_scalar = float(
        sum(
            probability * exposure * 0.5
            for probability, exposure in zip(
                probabilities, frame["exposure"], strict=True
            )
        )
    )

    def all_expected_loss(result: BinaryRiskValidationResult) -> float:
        row = result.business_metrics.loc[
            (result.business_metrics["segment_kind"] == "all")
            & (result.business_metrics["metric"] == "expected_loss_sum")
        ].iloc[0]
        assert row["status"] == "available"
        return float(row["value"])

    assert all_expected_loss(column_result) == pytest.approx(expected_column)
    assert all_expected_loss(scalar_result) == pytest.approx(expected_scalar)
    assert expected_column != pytest.approx(expected_scalar)


@pytest.mark.parametrize("invalid", [-0.1, np.inf, 1.1])
def test_loss_fraction_column_invalid_values_fail_stably(invalid: float) -> None:
    frame = _binary_frame().reset_index(drop=True)
    frame["exposure"] = 100.0
    frame["loss_fraction"] = 0.5
    frame.loc[0, "loss_fraction"] = invalid
    with pytest.raises(ValueError, match="^binary risk business inputs are invalid$"):
        validate_binary_risk(
            frame,
            "target",
            config=BinaryRiskValidationConfig(
                validation_mode="stratified_kfold",
                n_splits=3,
                exposure_column="exposure",
                loss_fraction="loss_fraction",
                exposure_unit="USD",
            ),
            external_predictions=_kfold_external(
                tuple(frame["target"]),
                probabilities=tuple(np.linspace(0.05, 0.95, len(frame))),
            ),
        )


def test_loss_fraction_missing_column_fails_before_prediction() -> None:
    frame = _binary_frame().reset_index(drop=True)
    with pytest.raises(
        ValueError, match="^loss_fraction column not found: 'missing_fraction'$"
    ):
        validate_binary_risk(
            frame,
            "target",
            config=BinaryRiskValidationConfig(
                validation_mode="stratified_kfold",
                n_splits=3,
                loss_fraction="missing_fraction",
                exposure_unit="USD",
            ),
            external_predictions=_kfold_external(
                tuple(frame["target"]), scores=tuple(np.linspace(0.0, 1.0, len(frame)))
            ),
        )


@pytest.mark.parametrize(
    ("available_column", "snapshot"), [(None, False), ("loss_available", True)]
)
def test_observed_loss_requires_exactly_one_maturity_provenance(
    available_column: str | None, snapshot: bool
) -> None:
    frame = _binary_frame()
    frame["observed_loss"] = 0.0
    frame["loss_available"] = pd.Timestamp("2026-01-01")
    with pytest.raises(ValueError, match="^binary risk business inputs are invalid$"):
        validate_binary_risk(
            frame,
            "target",
            config=BinaryRiskValidationConfig(
                validation_mode="stratified_kfold",
                n_splits=3,
                analysis_as_of=datetime(2026, 1, 2),
                observed_loss_column="observed_loss",
                observed_loss_available_time_column=available_column,
                observed_loss_is_mature_snapshot=snapshot,
                exposure_unit="USD",
            ),
            external_predictions=_kfold_external(
                tuple(frame["target"]), scores=tuple(np.linspace(0, 1, len(frame)))
            ),
        )


def test_single_class_scope_keeps_probability_metrics_and_independent_statuses() -> (
    None
):
    result = _time_result()
    fold = result.metrics.loc[
        (result.metrics["scope"] == "fold") & (result.metrics["fold_id"] == 0)
    ].set_index("metric")
    assert fold.loc["roc_auc", "status"] == "available"

    # A time holdout whose only mature validation rows are non-events.
    observation = pd.date_range("2026-01-01", periods=6, freq="D")
    frame = pd.DataFrame(
        {
            "target": [0, 1, 0, 1, 0, 0],
            "observation": observation,
            "outcome": observation + pd.Timedelta(days=1),
        }
    )
    config = BinaryRiskValidationConfig(
        validation_mode="time_holdout",
        observation_time_column="observation",
        outcome_end_time_column="outcome",
        maturity_source="outcome_end",
        fold_cutoffs=(datetime(2026, 1, 5),),
        validation_end=datetime(2026, 1, 7),
        analysis_as_of=datetime(2026, 1, 7),
        thresholds=(0.5,),
        threshold_kind="event_probability",
        calibration_bins=2,
    )
    external = ExternalRiskPredictions(
        (4, 5),
        (0, 0),
        ((0, (0, 1, 2, 3)),),
        None,
        None,
        (0.2, 0.7),
        1,
        "external_declared",
    )
    single = validate_binary_risk(
        frame, "target", config=config, external_predictions=external
    )
    metrics = single.metrics.loc[single.metrics["scope"] == "overall"].set_index(
        "metric"
    )
    for metric in ("roc_auc", "average_precision", "normalized_gini", "ks_statistic"):
        assert (metrics.loc[metric, "status"], metrics.loc[metric, "reason"]) == (
            "undefined",
            "single_class",
        )
    for metric in ("brier_score", "log_loss", "expected_calibration_error"):
        assert metrics.loc[metric, "status"] == "available"
    gains = single.gains.loc[single.gains["scope"] == "overall"]
    assert set(gains["event_rate_status"]) == {"available"}
    assert set(gains["capture_status"]) == {"undefined"}
    assert set(gains["lift_status"]) == {"undefined"}
    threshold = single.threshold_analysis.loc[
        single.threshold_analysis["scope"] == "overall"
    ].iloc[0]
    assert threshold["predicted_positive_rate_status"] == "available"
    assert threshold["sensitivity_status"] == "undefined"


def test_operating_point_tie_uses_higher_threshold_without_policy_mutation() -> None:
    frame = _binary_frame().reset_index(drop=True)
    probabilities = tuple(0.9 if value == 1 else 0.1 for value in frame["target"])
    config = BinaryRiskValidationConfig(
        validation_mode="stratified_kfold",
        n_splits=3,
        thresholds=(0.2, 0.8),
        threshold_kind="event_probability",
        operating_metric="sensitivity",
        calibration_bins=2,
    )
    result = _external_result(frame, config=config, probabilities=probabilities)

    overall = result.threshold_analysis.loc[
        result.threshold_analysis["scope"] == "overall"
    ]
    assert overall["sensitivity"].tolist() == pytest.approx([1.0, 1.0])
    operating = result.operating_point.iloc[0]
    assert operating["threshold"] == pytest.approx(0.8)
    assert operating["metric_value"] == pytest.approx(1.0)
    assert operating["candidate_count"] == 2
    assert (operating["status"], pd.isna(operating["reason"])) == (
        "available",
        True,
    )
    assert config.thresholds == (0.2, 0.8)
    assert result.config.thresholds == (0.2, 0.8)
    assert "guardrail" not in {field.name for field in fields(type(result.config))}


def test_operating_point_all_undefined_has_no_feasible_candidate() -> None:
    observation = pd.date_range("2026-01-01", periods=6, freq="D")
    frame = pd.DataFrame(
        {
            "target": [0, 1, 0, 1, 1, 1],
            "observation": observation,
            "outcome": observation + pd.Timedelta(days=1),
        }
    )
    config = BinaryRiskValidationConfig(
        validation_mode="time_holdout",
        observation_time_column="observation",
        outcome_end_time_column="outcome",
        maturity_source="outcome_end",
        fold_cutoffs=(datetime(2026, 1, 5),),
        validation_end=datetime(2026, 1, 7),
        analysis_as_of=datetime(2026, 1, 7),
        thresholds=(0.2, 0.8),
        threshold_kind="event_probability",
        operating_metric="specificity",
        calibration_bins=2,
    )
    result = validate_binary_risk(
        frame,
        "target",
        config=config,
        external_predictions=ExternalRiskPredictions(
            (4, 5),
            (0, 0),
            ((0, (0, 1, 2, 3)),),
            None,
            None,
            (0.9, 0.95),
            1,
            "external_declared",
        ),
    )

    overall = result.threshold_analysis.loc[
        result.threshold_analysis["scope"] == "overall"
    ]
    assert set(overall["specificity_status"]) == {"undefined"}
    assert set(overall["specificity_reason"]) == {"zero_denominator"}
    operating = result.operating_point.iloc[0]
    assert pd.isna(operating["threshold"])
    assert pd.isna(operating["metric_value"])
    assert (operating["status"], operating["reason"]) == (
        "undefined",
        "objective_undefined",
    )
    assert result.config.thresholds == (0.2, 0.8)


def test_no_thresholds_generate_no_candidates_or_operating_point() -> None:
    config = BinaryRiskValidationConfig(
        validation_mode="stratified_kfold", n_splits=3, calibration_bins=2
    )
    result = _external_result(config=config)
    assert result.requested_threshold_count == result.actual_threshold_count == 0
    assert result.threshold_analysis.empty
    assert result.operating_point.empty
    assert result.config.thresholds == ()


def test_fixed_epsilon_log_loss_accepts_zero_and_one_and_rejects_out_of_range() -> None:
    frame = _binary_frame()
    probabilities = tuple(float(value) for value in frame["target"])
    result = _external_result(frame, probabilities=probabilities)
    row = result.metrics.loc[
        (result.metrics["scope"] == "overall")
        & (result.metrics["metric"] == "log_loss")
    ].iloc[0]
    epsilon = 1e-15
    expected = float(
        np.mean(
            -(
                np.asarray(frame["target"]) * np.log(1.0 - epsilon)
                + (1 - np.asarray(frame["target"])) * np.log(1.0 - epsilon)
            )
        )
    )
    assert row["value"] == pytest.approx(expected)

    bad = list(probabilities)
    bad[0] = -0.01
    with pytest.raises(
        ValueError, match=r"^event probabilities must be finite values in \[0, 1\]$"
    ):
        _external_result(frame, probabilities=tuple(bad))


def test_resource_bounds_sorted_positions_frozen_config_and_table_dtypes() -> None:
    result = _external_result()
    assert result.config is not BinaryRiskValidationConfig(
        validation_mode="stratified_kfold", n_splits=3
    )
    assert tuple(result.folds["fold_id"]) == (0, 1, 2)
    for column in (
        "train_row_positions",
        "validation_row_positions",
        "evaluable_validation_row_positions",
    ):
        assert all(tuple(sorted(value)) == value for value in result.folds[column])
    assert str(result.metrics["value"].dtype) == "Float64"
    assert str(result.threshold_analysis["tp"].dtype) == "Int64"
    assert str(result.predictions["is_evaluable"].dtype) == "boolean"
    assert result.predictions["target_value"].dtype == object
    with pytest.raises(FrozenInstanceError):
        result.config.n_splits = 4  # type: ignore[misc]
    with pytest.raises(
        ValueError, match="^binary risk validation config has invalid schema$"
    ):
        validate_binary_risk(
            _binary_frame(),
            "target",
            config=BinaryRiskValidationConfig(
                validation_mode="stratified_kfold",
                n_splits=3,
                calibration_bins=51,
            ),
            external_predictions=_kfold_external(
                tuple(_binary_frame()["target"]),
                scores=tuple(np.linspace(0, 1, 12)),
            ),
        )


def test_result_fields_exclude_estimators_pipelines_figures_and_private_state() -> None:
    assert [field.name for field in fields(BinaryRiskValidationResult)] == [
        "target",
        "positive_label",
        "validation_mode",
        "config",
        "prediction_scope",
        "score_source",
        "score_direction",
        "probability_provenance",
        "input_n_rows",
        "eligible_n_rows",
        "predicted_n_rows",
        "evaluable_n_rows",
        "requested_threshold_count",
        "actual_threshold_count",
        "observed_loss_maturity_mode",
        "observed_loss_analysis_as_of",
        "observed_loss_mature_n",
        "observed_loss_excluded_n",
        "folds",
        "predictions",
        "excluded_rows",
        "metrics",
        "gains",
        "calibration",
        "threshold_analysis",
        "operating_point",
        "business_metrics",
        "warnings",
        "limitations",
    ]
    assert not any(
        token in field.name
        for field in fields(BinaryRiskValidationResult)
        for token in ("estimator", "pipeline", "figure", "calibrator")
    )


@pytest.mark.parametrize("argument", ["features", "exclude_columns"])
def test_external_estimator_only_config_precedes_missing_target(argument: str) -> None:
    kwargs = {argument: ("missing_feature",)}
    with pytest.raises(
        ValueError, match="^binary risk validation config has invalid schema$"
    ):
        validate_binary_risk(
            _binary_frame(),
            "missing_target",
            config=BinaryRiskValidationConfig(
                validation_mode="stratified_kfold", n_splits=3
            ),
            external_predictions=ExternalRiskPredictions(
                (), (), (), (), "higher_risk", None, None, None
            ),
            **kwargs,
        )


def test_legal_external_config_and_estimator_source_keep_error_precedence() -> None:
    with pytest.raises(ValueError, match="^target column not found: 'missing_target'$"):
        validate_binary_risk(
            _binary_frame(),
            "missing_target",
            config=BinaryRiskValidationConfig(
                validation_mode="stratified_kfold", n_splits=3
            ),
            external_predictions=ExternalRiskPredictions(
                (), (), (), (), "higher_risk", None, None, None
            ),
        )
    with pytest.raises(ValueError, match="^target column not found: 'missing_target'$"):
        validate_binary_risk(
            _binary_frame(),
            "missing_target",
            config=BinaryRiskValidationConfig(
                validation_mode="stratified_kfold", n_splits=3
            ),
            estimator=_RecordingClassifier(),
            features=("missing_feature",),
        )


def test_each_fold_has_distinct_estimator_and_exact_train_fitted_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sharper import risk_validation

    frame = _binary_frame().reset_index(drop=True)
    frame["feature"] = np.arange(len(frame), dtype=float)
    snapshots = []
    original = risk_validation._fit_classifier_fold

    def spy(*args: object, **kwargs: object) -> object:
        snapshot = original(*args, **kwargs)
        snapshots.append(snapshot)
        return snapshot

    monkeypatch.setattr(risk_validation, "_fit_classifier_fold", spy)
    caller = _RecordingClassifier()
    result = validate_binary_risk(
        frame,
        "target",
        config=BinaryRiskValidationConfig(
            validation_mode="stratified_kfold", n_splits=3
        ),
        estimator=caller,
        features=("feature",),
    )

    assert len(snapshots) == 3
    assert len({id(snapshot.estimator) for snapshot in snapshots}) == 3
    assert len({id(snapshot.estimator.fit_token_) for snapshot in snapshots}) == 3
    assert not hasattr(caller, "fitted_")
    for snapshot, fold in zip(snapshots, result.folds.to_dict("records"), strict=True):
        train_positions = fold["train_row_positions"]
        assert snapshot.estimator.fit_n_ == len(train_positions)
        numeric = snapshot.pipeline.named_steps["preprocessor"].named_transformers_[
            "numeric"
        ]
        expected_mean = float(frame.iloc[list(train_positions)]["feature"].mean())
        assert numeric.named_steps["scaler"].mean_[0] == pytest.approx(expected_mean)


def test_each_fold_estimator_receives_exact_manual_transformed_train_rows() -> None:
    _TransformedRowRecorder.fit_values.clear()
    _TransformedRowRecorder.fit_tokens.clear()
    frame = _binary_frame().reset_index(drop=True)
    frame["row_value"] = np.arange(10.0, 10.0 + len(frame), dtype=float)
    caller = _TransformedRowRecorder()
    result = validate_binary_risk(
        frame,
        "target",
        config=BinaryRiskValidationConfig(
            validation_mode="stratified_kfold", n_splits=3
        ),
        estimator=caller,
        features=("row_value",),
    )

    assert len(_TransformedRowRecorder.fit_values) == len(result.folds) == 3
    assert len({id(token) for token in _TransformedRowRecorder.fit_tokens}) == 3
    assert not hasattr(caller, "fit_token_")
    for actual, fold in zip(
        _TransformedRowRecorder.fit_values,
        result.folds.to_dict("records"),
        strict=True,
    ):
        train_positions = fold["train_row_positions"]
        validation_positions = fold["validation_row_positions"]
        train_raw = frame.iloc[list(train_positions)]["row_value"].to_numpy()
        mean = float(train_raw.mean())
        scale = float(np.sqrt(np.mean((train_raw - mean) ** 2)))
        expected = tuple(float((value - mean) / scale) for value in train_raw)
        validation_raw = frame.iloc[list(validation_positions)]["row_value"].to_numpy()
        validation_transformed = tuple(
            float((value - mean) / scale) for value in validation_raw
        )
        assert actual == pytest.approx(expected)
        assert len(actual) == len(train_positions)
        assert not {round(value, 12) for value in actual} & {
            round(value, 12) for value in validation_transformed
        }


def test_time_fold_fit_values_exclude_maturity_purged_row() -> None:
    _TransformedRowRecorder.fit_values.clear()
    _TransformedRowRecorder.fit_tokens.clear()
    observation = pd.date_range("2026-01-01", periods=8, freq="D")
    frame = pd.DataFrame(
        {
            "row_value": np.arange(20.0, 28.0),
            "target": [0, 1] * 4,
            "observation": observation,
            "outcome": observation + pd.Timedelta(days=1),
            "available": pd.to_datetime(
                [
                    "2026-01-02",
                    "2026-01-03",
                    "2026-01-20",
                    "2026-01-05",
                    "2026-01-06",
                    "2026-01-07",
                    "2026-01-08",
                    "2026-01-09",
                ]
            ),
        }
    )
    result = validate_binary_risk(
        frame,
        "target",
        config=BinaryRiskValidationConfig(
            validation_mode="time_holdout",
            observation_time_column="observation",
            outcome_end_time_column="outcome",
            label_available_time_column="available",
            maturity_source="label_available_time",
            fold_cutoffs=(datetime(2026, 1, 6),),
            validation_end=datetime(2026, 1, 9),
            analysis_as_of=datetime(2026, 1, 9),
            calibration_bins=2,
        ),
        estimator=_TransformedRowRecorder(),
        features=("row_value",),
    )

    train_positions = result.folds.iloc[0]["train_row_positions"]
    assert train_positions == (0, 1, 3, 4)
    assert result.folds.iloc[0]["purged_train_n"] == 1
    raw = frame.iloc[list(train_positions)]["row_value"].to_numpy()
    mean = float(raw.mean())
    scale = float(np.sqrt(np.mean((raw - mean) ** 2)))
    expected = tuple(float((value - mean) / scale) for value in raw)
    assert _TransformedRowRecorder.fit_values == [pytest.approx(expected)]
    purged_value = float((frame.iloc[2]["row_value"] - mean) / scale)
    assert all(
        not np.isclose(value, purged_value)
        for value in _TransformedRowRecorder.fit_values[0]
    )


def test_holdout_validation_only_values_do_not_change_preprocessing_or_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sharper import modeling, risk_validation

    n_rows = 16
    target = np.asarray([0, 1] * 8)
    positions = np.arange(n_rows)
    train, validation = train_test_split(
        positions, stratify=target, test_size=0.25, random_state=42
    )
    numeric = np.arange(n_rows, dtype=float)
    numeric[int(train[0])] = np.nan
    numeric[validation] = 999.0
    category = np.asarray(["a", "b"] * 8, dtype=object)
    category[validation] = "validation_only"
    pattern = np.ones(n_rows, dtype=float)
    pattern[validation] = np.arange(10.0, 10.0 + len(validation))
    frame = pd.DataFrame(
        {
            "numeric": numeric,
            "category": category,
            "validation_pattern": pattern,
            "target": target,
        }
    )
    schema_inputs: list[pd.DataFrame] = []
    snapshots = []
    original_schema = modeling.infer_schema
    original_fold = risk_validation._fit_classifier_fold

    def schema_spy(value: pd.DataFrame) -> object:
        schema_inputs.append(value.copy(deep=True))
        return original_schema(value)

    def fold_spy(*args: object, **kwargs: object) -> object:
        snapshot = original_fold(*args, **kwargs)
        snapshots.append(snapshot)
        return snapshot

    monkeypatch.setattr(modeling, "infer_schema", schema_spy)
    monkeypatch.setattr(risk_validation, "_fit_classifier_fold", fold_spy)
    result = validate_binary_risk(
        frame,
        "target",
        config=BinaryRiskValidationConfig(
            validation_mode="stratified_holdout", test_size=0.25
        ),
        estimator=_RecordingClassifier(),
        features=("numeric", "category", "validation_pattern"),
    )

    assert len(snapshots) == len(schema_inputs) == 1
    assert tuple(sorted(schema_inputs[0].index)) == tuple(sorted(int(v) for v in train))
    assert set(schema_inputs[0].index).isdisjoint(validation)
    assert result.folds.iloc[0]["feature_columns"] == ("numeric", "category")
    preprocessor = snapshots[0].pipeline.named_steps["preprocessor"]
    numeric_pipe = preprocessor.named_transformers_["numeric"]
    train_numeric = frame.iloc[list(train)]["numeric"]
    expected_median = float(train_numeric.median())
    expected_imputed = train_numeric.fillna(expected_median)
    assert numeric_pipe.named_steps["imputer"].statistics_[0] == pytest.approx(
        expected_median
    )
    assert numeric_pipe.named_steps["scaler"].mean_[0] == pytest.approx(
        float(expected_imputed.mean())
    )
    encoder = preprocessor.named_transformers_["categorical"].named_steps["encoder"]
    assert set(encoder.categories_[0]) == {"a", "b"}
    assert "validation_only" not in encoder.categories_[0]


def test_time_cutoff_label_maturity_and_reporting_delay_boundaries() -> None:
    observation = pd.to_datetime(
        [
            "2026-01-01",
            "2026-01-02",
            "2026-01-03",
            "2026-01-05",
            "2026-01-06",
            "2026-01-07",
        ]
    )
    frame = pd.DataFrame(
        {
            "target": [0, 1, 0, 1, 0, 1],
            "observation": observation,
            "outcome": observation + pd.Timedelta(days=1),
            "available": pd.to_datetime(
                [
                    "2026-01-03",
                    "2026-01-05",
                    "2026-01-06",
                    "2026-01-07",
                    "2026-01-20",
                    "2026-01-09",
                ]
            ),
        }
    )
    result = validate_binary_risk(
        frame,
        "target",
        config=BinaryRiskValidationConfig(
            validation_mode="time_holdout",
            observation_time_column="observation",
            outcome_end_time_column="outcome",
            label_available_time_column="available",
            maturity_source="label_available_time",
            reporting_delay=timedelta(days=1),
            fold_cutoffs=(datetime(2026, 1, 5),),
            validation_end=datetime(2026, 1, 8),
            analysis_as_of=datetime(2026, 1, 9),
            calibration_bins=2,
        ),
        external_predictions=ExternalRiskPredictions(
            (3, 4, 5),
            (0, 0, 0),
            ((0, (0, 1)),),
            (0.8, 0.2, 0.9),
            "higher_risk",
            (0.8, 0.2, 0.9),
            1,
            "external_declared",
        ),
    )
    fold = result.folds.iloc[0]
    assert fold["train_candidate_n"] == 3
    assert fold["train_row_positions"] == (0, 1)
    assert fold["purged_train_n"] == fold["immature_train_n"] == 1
    assert fold["validation_row_positions"] == (3, 4, 5)
    assert fold["evaluable_validation_row_positions"] == (3, 5)
    assert result.predictions["row_position"].tolist() == [3, 4, 5]
    assert result.predictions["is_evaluable"].tolist() == [True, False, True]
    assert result.evaluable_n_rows == 2


def test_time_forward_row_horizon_controls_later_fold_training() -> None:
    observation = pd.date_range("2026-01-01", periods=8, freq="D")
    horizon = pd.to_timedelta([1, 2, 4, 1, 4, 1, 1, 1], unit="D")
    frame = pd.DataFrame(
        {"target": [0, 1] * 4, "observation": observation, "horizon": horizon}
    )
    result = validate_binary_risk(
        frame,
        "target",
        config=BinaryRiskValidationConfig(
            validation_mode="time_forward",
            observation_time_column="observation",
            maturity_source="observation_horizon",
            prediction_horizon_column="horizon",
            fold_cutoffs=(datetime(2026, 1, 4), datetime(2026, 1, 6)),
            validation_end=datetime(2026, 1, 9),
            analysis_as_of=datetime(2026, 1, 9),
            calibration_bins=2,
        ),
        external_predictions=ExternalRiskPredictions(
            (3, 4, 5, 6, 7),
            (0, 0, 1, 1, 1),
            ((0, (0, 1)), (1, (0, 1, 3))),
            (0.1, 0.2, 0.7, 0.8, 0.9),
            "higher_risk",
            None,
            None,
            None,
        ),
    )
    assert result.folds["train_row_positions"].tolist() == [(0, 1), (0, 1, 3)]
    assert 3 in result.folds.iloc[1]["train_row_positions"]
    assert 4 not in result.folds.iloc[1]["train_row_positions"]
    assert result.folds["prediction_horizon_source"].tolist() == ["column", "column"]


def test_time_horizon_outcome_exact_match_and_mismatch() -> None:
    observation = pd.date_range("2026-01-01", periods=6, freq="D")
    frame = pd.DataFrame(
        {
            "target": [0, 1] * 3,
            "observation": observation,
            "outcome": observation + pd.Timedelta(days=1),
        }
    )
    config = BinaryRiskValidationConfig(
        validation_mode="time_holdout",
        observation_time_column="observation",
        outcome_end_time_column="outcome",
        maturity_source="observation_horizon",
        prediction_horizon=timedelta(days=1),
        fold_cutoffs=(datetime(2026, 1, 5),),
        validation_end=datetime(2026, 1, 7),
        analysis_as_of=datetime(2026, 1, 7),
        calibration_bins=2,
    )
    external = ExternalRiskPredictions(
        (4, 5),
        (0, 0),
        ((0, (0, 1, 2, 3)),),
        (0.2, 0.8),
        "higher_risk",
        None,
        None,
        None,
    )
    result = validate_binary_risk(
        frame, "target", config=config, external_predictions=external
    )
    assert result.folds.iloc[0]["outcome_end_source"] == "column"
    changed = frame.copy(deep=True)
    changed.loc[2, "outcome"] += pd.Timedelta(days=1)
    with pytest.raises(
        ValueError, match="^label maturity metadata is missing or inconsistent$"
    ):
        validate_binary_risk(
            changed, "target", config=config, external_predictions=external
        )


@pytest.mark.parametrize("case", ["timezone", "mixed_timezone", "missing", "overflow"])
def test_time_metadata_timezone_missing_and_overflow_fail(case: str) -> None:
    if case == "overflow":
        observation = pd.Series(
            [
                pd.Timestamp.max - pd.Timedelta(days=1, microseconds=value)
                for value in range(6)
            ]
        )
        cutoff = datetime(2262, 4, 11)
        end = datetime(2262, 4, 12)
        as_of = end
        horizon = timedelta(days=2)
    else:
        observation = pd.Series(pd.date_range("2026-01-01", periods=6, freq="D"))
        if case == "timezone":
            observation = observation.dt.tz_localize("UTC")
        elif case == "mixed_timezone":
            observation = observation.astype(object)
            observation.iloc[0] = pd.Timestamp("2026-01-01", tz="UTC")
        else:
            observation.iloc[2] = pd.NaT
        cutoff = datetime(2026, 1, 5)
        end = datetime(2026, 1, 7)
        as_of = end
        horizon = timedelta(days=1)
    frame = pd.DataFrame({"target": [0, 1] * 3, "observation": observation})
    with pytest.raises(ValueError, match="^time validation metadata is invalid$"):
        validate_binary_risk(
            frame,
            "target",
            config=BinaryRiskValidationConfig(
                validation_mode="time_holdout",
                observation_time_column="observation",
                maturity_source="observation_horizon",
                prediction_horizon=horizon,
                fold_cutoffs=(cutoff,),
                validation_end=end,
                analysis_as_of=as_of,
                calibration_bins=2,
            ),
            external_predictions=ExternalRiskPredictions(
                (), (), (), (), "higher_risk", None, None, None
            ),
        )


def test_maturity_purge_can_make_estimator_train_single_class() -> None:
    observation = pd.date_range("2026-01-01", periods=6, freq="D")
    frame = pd.DataFrame(
        {
            "feature": np.arange(6, dtype=float),
            "target": [0, 1, 0, 1, 0, 1],
            "observation": observation,
            "outcome": observation + pd.Timedelta(days=1),
            "available": pd.to_datetime(
                [
                    "2026-01-03",
                    "2026-01-20",
                    "2026-01-05",
                    "2026-01-20",
                    "2026-01-07",
                    "2026-01-08",
                ]
            ),
        }
    )
    with pytest.raises(
        ValueError,
        match="^validation fold 0 training target must contain both classes$",
    ):
        validate_binary_risk(
            frame,
            "target",
            config=BinaryRiskValidationConfig(
                validation_mode="time_holdout",
                observation_time_column="observation",
                outcome_end_time_column="outcome",
                label_available_time_column="available",
                maturity_source="label_available_time",
                fold_cutoffs=(datetime(2026, 1, 5),),
                validation_end=datetime(2026, 1, 7),
                analysis_as_of=datetime(2026, 1, 8),
                calibration_bins=2,
            ),
            estimator=_RecordingClassifier(),
            features=("feature",),
        )


@pytest.mark.parametrize("n_splits", [2, 20])
def test_fold_resource_boundaries_accept_minimum_and_maximum(n_splits: int) -> None:
    frame = pd.DataFrame({"target": [0, 1] * 20})
    result = validate_binary_risk(
        frame,
        "target",
        config=BinaryRiskValidationConfig(
            validation_mode="stratified_kfold", n_splits=n_splits, calibration_bins=2
        ),
        external_predictions=_kfold_external(
            tuple(frame["target"]),
            scores=tuple(np.linspace(0.0, 1.0, len(frame))),
            n_splits=n_splits,
        ),
    )
    assert len(result.folds) == n_splits


def test_fold_resource_boundary_rejects_twenty_one() -> None:
    with pytest.raises(
        ValueError, match="^binary risk validation config has invalid schema$"
    ):
        validate_binary_risk(
            _binary_frame(),
            "target",
            config=BinaryRiskValidationConfig(
                validation_mode="stratified_kfold", n_splits=21
            ),
            external_predictions=ExternalRiskPredictions(
                (), (), (), (), "higher_risk", None, None, None
            ),
        )


@pytest.mark.parametrize("count", [1, 100])
def test_threshold_resource_boundaries_accept_minimum_and_maximum(count: int) -> None:
    thresholds = tuple(float(value) for value in np.linspace(-1.0, 1.0, count))
    result = _external_result(
        config=BinaryRiskValidationConfig(
            validation_mode="stratified_kfold",
            n_splits=3,
            thresholds=thresholds,
            threshold_kind="ranking_score",
            calibration_bins=2,
        )
    )
    assert result.actual_threshold_count == count


def test_threshold_resource_boundary_rejects_one_hundred_one() -> None:
    with pytest.raises(
        ValueError, match="^binary risk validation config has invalid schema$"
    ):
        _external_result(
            config=BinaryRiskValidationConfig(
                validation_mode="stratified_kfold",
                n_splits=3,
                thresholds=tuple(float(value) for value in range(101)),
                threshold_kind="ranking_score",
            )
        )


@pytest.mark.parametrize("bins", [2, 50])
def test_calibration_bin_resource_boundaries_accept_minimum_and_maximum(
    bins: int,
) -> None:
    result = _external_result(
        config=BinaryRiskValidationConfig(
            validation_mode="stratified_kfold", n_splits=3, calibration_bins=bins
        )
    )
    assert len(result.calibration.loc[result.calibration["scope"] == "overall"]) == bins


def test_calibration_bin_resource_boundary_rejects_fifty_one() -> None:
    with pytest.raises(
        ValueError, match="^binary risk validation config has invalid schema$"
    ):
        _external_result(
            config=BinaryRiskValidationConfig(
                validation_mode="stratified_kfold", n_splits=3, calibration_bins=51
            )
        )


@pytest.mark.parametrize("count", [1, 100])
def test_gain_fraction_resource_boundaries_accept_minimum_and_maximum(
    count: int,
) -> None:
    fractions = (
        (1.0,) if count == 1 else tuple(float(value) / 100.0 for value in range(1, 101))
    )
    result = _external_result(
        config=BinaryRiskValidationConfig(
            validation_mode="stratified_kfold",
            n_splits=3,
            gain_fractions=fractions,
            calibration_bins=2,
        )
    )
    assert len(result.gains.loc[result.gains["scope"] == "overall"]) == count


def test_gain_fraction_resource_boundary_rejects_one_hundred_one() -> None:
    fractions = tuple(float(value) / 101.0 for value in range(1, 102))
    with pytest.raises(
        ValueError, match="^binary risk validation config has invalid schema$"
    ):
        _external_result(
            config=BinaryRiskValidationConfig(
                validation_mode="stratified_kfold",
                n_splits=3,
                gain_fractions=fractions,
            )
        )


def test_zero_mature_observed_loss_counts_and_reason_are_frozen() -> None:
    frame = _binary_frame()
    frame["observed_loss"] = np.inf
    frame["loss_available"] = pd.Timestamp("2026-02-01")
    result = validate_binary_risk(
        frame,
        "target",
        config=BinaryRiskValidationConfig(
            validation_mode="stratified_kfold",
            n_splits=3,
            analysis_as_of=datetime(2026, 1, 15),
            observed_loss_column="observed_loss",
            observed_loss_available_time_column="loss_available",
            exposure_unit="USD",
            calibration_bins=2,
        ),
        external_predictions=_kfold_external(
            tuple(frame["target"]), scores=tuple(np.linspace(0.0, 1.0, len(frame)))
        ),
    )
    assert result.observed_loss_mature_n == 0
    assert result.observed_loss_excluded_n == len(frame)
    observed = result.business_metrics.loc[
        (result.business_metrics["segment_kind"] == "all")
        & (result.business_metrics["metric"] == "observed_loss_sum")
    ].iloc[0]
    assert observed["n_observed_loss_mature_rows"] == 0
    assert pd.isna(observed["value"])
    assert (observed["status"], observed["reason"]) == (
        "undefined",
        "no_evaluable_rows",
    )
