"""Task 10 contracts for static analytical visualization."""

import inspect
from dataclasses import fields, replace

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

matplotlib.use("Agg")

from matplotlib.figure import Figure

from sharper import (
    PlotCollection,
    PlotResult,
    TargetAnalysis,
    analyze_target_relationships,
    compare_groups,
    compute_correlations,
    detect_outliers,
    evaluate_classifier,
    evaluate_regressor,
    plot_classification_evaluation,
    plot_correlations,
    plot_distributions,
    plot_group_comparison,
    plot_missingness,
    plot_outliers,
    plot_regression_evaluation,
    plot_target_relationships,
    train_classifier,
    train_regressor,
)
from sharper import evaluation as evaluation_module


def _close(collection: PlotCollection) -> None:
    for plot in collection.plots:
        plt.close(plot.figure)


def test_classification_evaluation_figures_use_frozen_detail() -> None:
    frame = pd.DataFrame({"x": [0, 1] * 10, "target": [0] * 10 + [1] * 10})
    evaluation = evaluate_classifier(train_classifier(frame, "target", test_size=0.3))
    result = plot_classification_evaluation(evaluation)

    assert (result.requested_count, result.available_count, result.actual_count) == (
        2,
        2,
        2,
    )
    assert result.truncated is False
    assert result.truncation_reason is None
    assert [plot.chart_type for plot in result.plots] == [
        "classification_confusion_matrix",
        "classification_roc_curve",
    ]
    assert result.plots[0].metadata == (
        ("target", "target"),
        ("classes", '["0","1"]'),
        ("n_test", "6"),
        ("metric", "count"),
    )
    _close(result)


class _NoScoreClassifier(ClassifierMixin, BaseEstimator):
    _estimator_type = "classifier"

    def fit(self, X: object, y: list[int]) -> "_NoScoreClassifier":
        self.classes_ = [0, 1]
        return self

    def predict(self, X: object) -> np.ndarray:
        return np.zeros(len(X), dtype=int)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("frame", "estimator", "expected_count"),
    [
        (
            pd.DataFrame({"x": [0, 1] * 8, "target": [0] * 8 + [1] * 8}),
            _NoScoreClassifier(),
            1,
        ),
        (
            pd.DataFrame(
                {"x": [0, 1, 2] * 6, "target": ["a"] * 6 + ["b"] * 6 + ["c"] * 6}
            ),
            None,
            1,
        ),
    ],
)
def test_classification_plot_unavailable_roc_paths(
    frame: pd.DataFrame, estimator: ClassifierMixin | None, expected_count: int
) -> None:
    training = train_classifier(frame, "target", estimator=estimator, test_size=0.25)
    evaluation = evaluate_classifier(training)
    result = plot_classification_evaluation(evaluation)

    assert (result.requested_count, result.available_count, result.actual_count) == (
        2,
        expected_count,
        expected_count,
    )
    assert result.truncated is False
    assert result.truncation_reason is None
    assert [plot.chart_type for plot in result.plots] == [
        "classification_confusion_matrix"
    ]
    assert all(plot.source == "classification_evaluation" for plot in result.plots)
    _close(result)


def test_classification_plot_invalid_result_creates_no_figure() -> None:
    frame = pd.DataFrame({"x": [0, 1] * 8, "target": [0] * 8 + [1] * 8})
    evaluation = evaluate_classifier(train_classifier(frame, "target", test_size=0.25))
    malformed = replace(evaluation, metrics=(1, 1, 1))
    before = set(plt.get_fignums())
    with pytest.raises(
        ValueError, match="^classification evaluation result has invalid schema$"
    ):
        plot_classification_evaluation(malformed)
    assert set(plt.get_fignums()) == before


def test_classification_plot_does_not_recompute_or_modify_global_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame({"x": [0, 1] * 8, "target": [0] * 8 + [1] * 8})
    training = train_classifier(frame, "target", test_size=0.25)
    evaluation = evaluate_classifier(training)
    original_close = plt.close
    backend = matplotlib.get_backend()
    rc_params = matplotlib.rcParams.copy()
    seaborn_style = matplotlib.rcParams["axes.facecolor"]

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("unexpected recomputation or lifecycle call")

    monkeypatch.setattr(evaluation_module, "evaluate_classifier", forbidden)
    monkeypatch.setattr(training.pipeline, "fit", forbidden)
    monkeypatch.setattr(training.pipeline, "predict", forbidden)
    monkeypatch.setattr(plt, "show", forbidden)
    monkeypatch.setattr(plt, "close", forbidden)
    plots = plot_classification_evaluation(evaluation)

    assert matplotlib.get_backend() == backend
    assert matplotlib.rcParams == rc_params
    assert matplotlib.rcParams["axes.facecolor"] == seaborn_style
    assert [plot.chart_type for plot in plots.plots] == [
        "classification_confusion_matrix",
        "classification_roc_curve",
    ]
    for plot in plots.plots:
        original_close(plot.figure)


def _regression_evaluation() -> object:
    x = np.tile(np.arange(4, dtype=float), 6)
    frame = pd.DataFrame(
        {
            "x": x,
            "category": np.tile(["a", "b"], 12),
            "target": x * 2.0 + np.tile([0.0, 0.5], 12),
        }
    )
    return evaluate_regressor(train_regressor(frame, "target", test_size=0.25))


def test_regression_evaluation_figures_use_only_frozen_detail() -> None:
    evaluation = _regression_evaluation()
    result = plot_regression_evaluation(evaluation)
    assert (result.requested_count, result.available_count, result.actual_count) == (
        2,
        2,
        2,
    )
    assert result.truncated is False
    assert result.truncation_reason is None
    assert [plot.chart_type for plot in result.plots] == [
        "regression_predicted_vs_actual",
        "regression_residuals",
    ]
    assert [plot.source for plot in result.plots] == [
        "regression_evaluation",
        "regression_evaluation",
    ]
    assert result.plots[0].metadata == (
        ("target", "target"),
        ("n_test", "6"),
        ("metric", "prediction"),
    )
    assert result.plots[1].metadata[-1] == ("metric", "residual")
    _close(result)


def test_regression_plot_validates_before_figure_and_never_recomputes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluation = _regression_evaluation()
    malformed = replace(evaluation, metrics=(1, 1, 1))
    before = set(plt.get_fignums())
    with pytest.raises(
        ValueError, match="^regression evaluation result has invalid schema$"
    ):
        plot_regression_evaluation(malformed)
    assert set(plt.get_fignums()) == before

    original_close = plt.close

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("unexpected recomputation or lifecycle call")

    monkeypatch.setattr(evaluation_module, "evaluate_regressor", forbidden)
    monkeypatch.setattr(plt, "show", forbidden)
    monkeypatch.setattr(plt, "close", forbidden)
    plots = plot_regression_evaluation(evaluation)
    for plot in plots.plots:
        original_close(plot.figure)


def test_plot_regression_evaluation_preserves_global_state_and_no_recomputation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = np.tile(np.arange(4, dtype=float), 6)
    frame = pd.DataFrame(
        {
            "x": x,
            "category": np.tile(["a", "b"], 12),
            "target": x * 2.0 + np.tile([0.0, 0.5], 12),
        }
    )
    training = train_regressor(frame, "target", test_size=0.25)
    result = evaluate_regressor(training)
    backend = matplotlib.get_backend()
    rc_params = matplotlib.rcParams.copy()
    seaborn_facecolor = matplotlib.rcParams["axes.facecolor"]
    original_close = plt.close

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("unexpected regression plot recomputation")

    monkeypatch.setattr(training.pipeline, "predict", forbidden)
    monkeypatch.setattr(training.estimator, "predict", forbidden)
    monkeypatch.setattr(evaluation_module, "evaluate_regressor", forbidden)
    monkeypatch.setattr(evaluation_module, "evaluate_model", forbidden)
    monkeypatch.setattr(plt, "show", forbidden)
    monkeypatch.setattr(plt, "close", forbidden)
    plots = plot_regression_evaluation(result)

    assert matplotlib.get_backend() == backend
    assert matplotlib.rcParams == rc_params
    assert matplotlib.rcParams["axes.facecolor"] == seaborn_facecolor
    assert len({id(plot.figure) for plot in plots.plots}) == 2
    for plot in plots.plots:
        original_close(plot.figure)


@pytest.mark.parametrize(
    "replacement",
    [
        {"metrics": (1, 1, 1)},
        {"holdout_positions": (0, 0, 1, 2, 3, 4)},
        {"predictions": lambda result: result.predictions.iloc[:, ::-1]},
        {"predictions": lambda result: result.predictions.assign(predicted=np.nan)},
        {"predictions": lambda result: result.predictions.assign(residual=999.0)},
        {"metrics": (("mae", 0.0), ("rmse", 0.0), ("r2", 0.0))},
    ],
)
def test_regression_plot_malformed_results_create_zero_figures(
    replacement: dict[str, object],
) -> None:
    result = _regression_evaluation()
    values = {
        key: value(result) if callable(value) else value
        for key, value in replacement.items()
    }
    malformed = replace(result, **values)
    before = set(plt.get_fignums())
    with pytest.raises(
        ValueError, match="^regression evaluation result has invalid schema$"
    ):
        plot_regression_evaluation(malformed)
    assert set(plt.get_fignums()) == before


def test_plot_does_not_mutate_evaluation() -> None:
    frame = pd.DataFrame({"x": [0, 1] * 8, "target": [0] * 8 + [1] * 8})
    evaluation = evaluate_classifier(train_classifier(frame, "target", test_size=0.25))
    before = replace(evaluation)
    plots = plot_classification_evaluation(evaluation)

    assert evaluation == before
    _close(plots)


def test_result_contract_and_distribution_metadata() -> None:
    frame = pd.DataFrame(
        {"number": [1.0, 2.0, np.nan, np.inf], "kind": ["b", "a", "b", None]}
    )
    result = plot_distributions(frame, max_plots=2, sample_size=1)
    assert [field.name for field in fields(PlotResult)] == [
        "figure",
        "chart_type",
        "title",
        "source",
        "item",
        "metadata",
    ]
    assert [field.name for field in fields(PlotCollection)] == [
        "requested_count",
        "available_count",
        "actual_count",
        "truncated",
        "truncation_reason",
        "plots",
    ]
    assert result.actual_count == 2
    assert all(isinstance(plot.figure, Figure) for plot in result.plots)
    assert result.plots[0].chart_type == "distribution_histogram"
    assert result.plots[1].chart_type == "distribution_categories"
    _close(result)


def test_missingness_and_correlation_empty_and_validation() -> None:
    missing = plot_missingness(pd.DataFrame(), max_columns=1)
    assert isinstance(missing.figure, Figure)
    assert missing.metadata == (
        ("n_rows", "0"),
        ("requested_columns", "1"),
        ("available_columns", "0"),
        ("analyzed_columns", "[]"),
        ("truncated_columns", "false"),
        ("truncation_reason", "none"),
    )
    correlations = compute_correlations(pd.DataFrame({"a": [1, 2], "b": [2, 3]}))
    heatmap = plot_correlations(correlations)
    assert heatmap.chart_type == "correlation_heatmap"
    plt.close(heatmap.figure)
    plt.close(missing.figure)
    with pytest.raises(ValueError, match="max_columns must be an integer from 1 to 50"):
        plot_missingness(pd.DataFrame({"a": [1]}), max_columns=True)


def test_outlier_budget_and_empty_collection() -> None:
    analysis = detect_outliers(pd.DataFrame({"a": [1, 2, 100], "b": [1, 2, 200]}))
    result = plot_outliers(analysis, max_plots=1)
    assert result.truncated is True
    assert result.truncation_reason == "max_plots"
    _close(result)


def _empty_numeric_details() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": pd.Series(dtype="object"),
            "target_category": pd.Series(dtype="object"),
            "group_count": pd.Series(dtype="int64"),
            "count": pd.Series(dtype="int64"),
            "missing_count": pd.Series(dtype="int64"),
            "mean": pd.Series(dtype="float64"),
            "q25": pd.Series(dtype="float64"),
            "median": pd.Series(dtype="float64"),
            "q75": pd.Series(dtype="float64"),
        }
    )


def _empty_tests() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": pd.Series(dtype="object"),
            "feature_kind": pd.Series(dtype="object"),
            "analysis": pd.Series(dtype="object"),
            "n_obs": pd.Series(dtype="int64"),
            "group_count": pd.Series(dtype="int64"),
            "statistic": pd.Series(dtype="float64"),
            "p_value": pd.Series(dtype="float64"),
            "effect_size": pd.Series(dtype="float64"),
            "effect_size_name": pd.Series(dtype="object"),
            "limitation": pd.Series(dtype="object"),
        }
    )


def test_target_budget_creates_only_returned_figures() -> None:
    features = tuple(f"f{index}" for index in range(21))
    categories = pd.DataFrame(
        {
            "feature": list(features),
            "feature_category": ["x"] * 21,
            "target_category": [None] * 21,
            "count": [2] * 21,
            "rate": [1.0] * 21,
            "target_mean": [2.0] * 21,
            "target_median": [2.0] * 21,
        }
    ).astype(
        {
            "feature": "object",
            "feature_category": "object",
            "target_category": "object",
            "count": "int64",
            "rate": "float64",
            "target_mean": "float64",
            "target_median": "float64",
        }
    )
    result = TargetAnalysis(
        2,
        "y",
        "regression",
        None,
        features,
        (),
        {},
        50,
        20,
        21,
        False,
        None,
        _empty_numeric_details(),
        categories,
        pd.DataFrame(
            {
                "feature": list(features),
                "feature_kind": ["categorical"] * 21,
                "analysis": ["kruskal_wallis"] * 21,
                "n_obs": [2] * 21,
                "group_count": [1] * 21,
                "statistic": [1.0] * 21,
                "p_value": [0.5] * 21,
                "effect_size": [0.1] * 21,
                "effect_size_name": ["epsilon_squared"] * 21,
                "limitation": ["exploratory_unadjusted_p_value"] * 21,
            }
        ).astype(
            {
                "feature": "object",
                "feature_kind": "object",
                "analysis": "object",
                "n_obs": "int64",
                "group_count": "int64",
                "statistic": "float64",
                "p_value": "float64",
                "effect_size": "float64",
                "effect_size_name": "object",
                "limitation": "object",
            }
        ),
        (),
    )
    before = set(plt.get_fignums())
    plots = plot_target_relationships(result)
    created = set(plt.get_fignums()) - before
    assert (plots.available_count, plots.actual_count, plots.truncated) == (
        21,
        20,
        True,
    )
    assert len(plots.plots) == len(created) == 20
    _close(plots)


@pytest.mark.parametrize(
    ("scenario", "frame", "task", "features", "missing_block"),
    [
        (
            "classification numeric",
            pd.DataFrame(
                {"x": [1, 2, 3, 4, 5, 6], "target": ["a", "a", "a", "b", "b", "b"]}
            ),
            "classification",
            ["x"],
            "numeric_details",
        ),
        (
            "classification categorical",
            pd.DataFrame(
                {
                    "g": ["u", "v", "u", "v", "u", "v"],
                    "target": ["a", "a", "a", "b", "b", "b"],
                }
            ),
            "classification",
            ["g"],
            "category_details",
        ),
        (
            "regression categorical",
            pd.DataFrame(
                {"g": ["u", "v", "u", "v", "u", "v"], "target": [1, 2, 3, 4, 5, 6]}
            ),
            "regression",
            ["g"],
            "category_details",
        ),
        (
            "one of two required blocks",
            pd.DataFrame(
                {
                    "x": [1, 2, 3, 4, 5, 6],
                    "g": ["u", "v", "u", "v", "u", "v"],
                    "target": ["a", "a", "a", "b", "b", "b"],
                }
            ),
            "classification",
            ["x", "g"],
            "category_details",
        ),
        (
            "regression numeric without plot detail",
            pd.DataFrame(
                {
                    "x": [1, 2, 3, 4, 5, 6],
                    "target": [2, 4, 5, 8, 9, 12],
                }
            ),
            "regression",
            ["x"],
            None,
        ),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_target_relationships_requires_detail_blocks(
    scenario: str,
    frame: pd.DataFrame,
    task: str,
    features: list[str],
    missing_block: str | None,
) -> None:
    del scenario
    result = analyze_target_relationships(frame, "target", task=task, features=features)
    if missing_block is None:
        plots = plot_target_relationships(result)
        assert (plots.actual_count, plots.plots) == (0, ())
        _close(plots)
        return
    broken = result.__class__(
        **{**result.__dict__, missing_block: getattr(result, missing_block).iloc[0:0]}
    )
    with pytest.raises(
        ValueError,
        match="^target result has invalid schema$",
    ):
        plot_target_relationships(broken)


def _classification_categorical_result(features: list[str]) -> object:
    frame = pd.DataFrame(
        {
            "g": ["u"] * 4 + ["v"] * 4,
            "h": ["m"] * 4 + ["n"] * 4,
            "target": ["a"] * 4 + ["b"] * 4,
        }
    )
    return analyze_target_relationships(
        frame, "target", task="classification", features=features
    )


@pytest.mark.parametrize(
    "scenario",
    [
        "missing cell",
        "duplicate cell",
        "extra feature category",
        "extra target category",
        "wrong combination with correct row total",
        "missing zero count cell",
        "second feature incomplete",
    ],
)
def test_target_classification_categorical_requires_complete_cartesian_block(
    scenario: str,
) -> None:
    features = ["g", "h"] if scenario == "second feature incomplete" else ["g"]
    result = _classification_categorical_result(features)
    details = result.category_details.copy()
    feature = (
        result.analyzed_features[-1] if scenario == "second feature incomplete" else "g"
    )
    block = details[details["feature"] == feature]
    if scenario == "missing cell":
        details = details.drop(block.index[0])
    elif scenario == "duplicate cell":
        details = pd.concat([details, block.iloc[[0]]], ignore_index=True)
    elif scenario == "extra feature category":
        extra = block.copy()
        extra.loc[:, "feature_category"] = "extra"
        extra.loc[:, "count"] = 0
        extra.loc[:, "rate"] = 0.0
        details = pd.concat([details, extra], ignore_index=True)
    elif scenario == "extra target category":
        extra = block.drop_duplicates("feature_category").copy()
        extra.loc[:, "target_category"] = "extra"
        extra.loc[:, "count"] = 0
        extra.loc[:, "rate"] = 0.0
        details = pd.concat([details, extra], ignore_index=True)
    elif scenario == "wrong combination with correct row total":
        details.loc[block.index[-1], "target_category"] = "extra"
    elif scenario == "missing zero count cell":
        zero_index = block.index[block["count"] == 0][0]
        details = details.drop(zero_index)
    else:
        details = details.drop(block.index[0])
    broken = result.__class__(**{**result.__dict__, "category_details": details})
    before = set(plt.get_fignums())
    with pytest.raises(
        ValueError,
        match="^target result has invalid schema$",
    ):
        plot_target_relationships(broken)
    assert set(plt.get_fignums()) == before


def test_target_classification_categorical_complete_block_control() -> None:
    result = _classification_categorical_result(["g"])
    plots = plot_target_relationships(result)
    try:
        plot = plots.plots[0]
        axes = plot.figure.axes[0]
        assert [tick.get_text() for tick in axes.get_xticklabels()] == ["u", "v"]
        assert [text.get_text() for text in axes.get_legend().get_texts()] == ["a", "b"]
        assert len(axes.patches) == 4
        assert any(patch.get_height() == 0.0 for patch in axes.patches)
    finally:
        _close(plots)


def _target_result_for_statistical_test(path: str) -> object:
    if path == "classification numeric":
        frame = pd.DataFrame({"x": list(range(1, 9)), "target": ["a"] * 4 + ["b"] * 4})
        return analyze_target_relationships(
            frame, "target", task="classification", features=["x"]
        )
    if path == "classification categorical":
        return _classification_categorical_result(["g"])
    if path == "regression categorical":
        frame = pd.DataFrame({"g": ["u"] * 4 + ["v"] * 4, "target": list(range(1, 9))})
        return analyze_target_relationships(
            frame, "target", task="regression", features=["g"]
        )
    if path == "regression numeric":
        frame = pd.DataFrame({"x": list(range(1, 9)), "target": list(range(2, 10))})
        return analyze_target_relationships(
            frame, "target", task="regression", features=["x"]
        )
    frame = pd.DataFrame(
        {
            "x": list(range(1, 9)),
            "g": ["u"] * 4 + ["v"] * 4,
            "target": ["a"] * 4 + ["b"] * 4,
        }
    )
    return analyze_target_relationships(
        frame, "target", task="classification", features=["x", "g"]
    )


@pytest.mark.parametrize(
    ("scenario", "path"),
    [
        ("classification numeric missing", "classification numeric"),
        ("classification categorical missing", "classification categorical"),
        ("regression categorical missing", "regression categorical"),
        ("regression numeric missing", "regression numeric"),
        ("second analyzed feature missing", "two features"),
        ("duplicate test row", "classification numeric"),
        ("unknown test feature", "classification numeric"),
        ("feature kind/detail path mismatch", "classification numeric"),
        ("analysis/task path mismatch", "classification categorical"),
        ("test metadata vocabulary mismatch", "classification numeric"),
    ],
)
def test_target_relationships_requires_exactly_one_statistical_test_row(
    scenario: str, path: str
) -> None:
    result = _target_result_for_statistical_test(path)
    tests = result.statistical_tests.copy()
    feature = result.analyzed_features[-1]
    if "missing" in scenario:
        tests = tests[tests["feature"] != feature].copy()
    elif scenario == "duplicate test row":
        tests = pd.concat(
            [tests, tests[tests["feature"] == feature]], ignore_index=True
        )
    elif scenario == "unknown test feature":
        unknown = tests.iloc[[0]].copy()
        unknown.loc[:, "feature"] = "unknown"
        tests = pd.concat([tests, unknown], ignore_index=True)
    elif scenario == "feature kind/detail path mismatch":
        tests.loc[tests["feature"] == feature, "feature_kind"] = "categorical"
    elif scenario == "test metadata vocabulary mismatch":
        tests.loc[tests["feature"] == feature, "effect_size_name"] = "wrong"
    else:
        tests.loc[tests["feature"] == feature, "analysis"] = "pearson"
    broken = result.__class__(**{**result.__dict__, "statistical_tests": tests})
    before = set(plt.get_fignums())
    with pytest.raises(
        ValueError,
        match="^target result has invalid schema$",
    ):
        plot_target_relationships(broken)
    assert set(plt.get_fignums()) == before


@pytest.mark.parametrize(
    ("path", "expected_count"),
    [
        ("classification numeric", 1),
        ("classification categorical", 1),
        ("regression categorical", 1),
        ("regression numeric", 0),
    ],
)
def test_target_relationships_accepts_valid_statistical_test_rows(
    path: str, expected_count: int
) -> None:
    plots = plot_target_relationships(_target_result_for_statistical_test(path))
    try:
        assert plots.actual_count == expected_count
        assert len(plots.plots) == expected_count
    finally:
        _close(plots)


def test_malformed_correlation_and_target_are_stable_errors() -> None:
    result = compute_correlations(pd.DataFrame({"a": [1, 2], "b": [2, 3]}))
    broken = result.__class__(
        **{
            **result.__dict__,
            "correlations": result.correlations.astype({"correlation": "object"}),
        }
    )
    with pytest.raises(ValueError, match="correlation result has invalid schema"):
        plot_correlations(broken)
    bad_target = TargetAnalysis(
        0,
        "y",
        "regression",
        None,
        ("f",),
        (),
        {},
        50,
        20,
        1,
        False,
        None,
        _empty_numeric_details().assign(
            feature=["f"],
            target_category=["x"],
            group_count=[1],
            count=[1],
            missing_count=[0],
            mean=[1.0],
            q25=[1.0],
            median=[1.0],
            q75=[1.0],
        ),
        pd.DataFrame(
            {
                "feature": pd.Series(dtype="object"),
                "feature_category": pd.Series(dtype="object"),
                "target_category": pd.Series(dtype="object"),
                "count": pd.Series(dtype="int64"),
                "rate": pd.Series(dtype="float64"),
                "target_mean": pd.Series(dtype="float64"),
                "target_median": pd.Series(dtype="float64"),
            }
        ),
        _empty_tests(),
        (),
    )
    with pytest.raises(ValueError, match="target result has invalid schema"):
        plot_target_relationships(bad_target)


def test_public_signatures_and_fresh_figures() -> None:
    contracts = {
        plot_distributions: ["df", "max_plots", "sample_size"],
        plot_missingness: ["df", "max_columns"],
        plot_correlations: ["result"],
        plot_outliers: ["result", "max_plots"],
        plot_target_relationships: ["result"],
    }
    for function, names in contracts.items():
        assert list(inspect.signature(function).parameters) == names
    frame = pd.DataFrame({"x": [1, 2, 3]})
    first = plot_distributions(frame)
    second = plot_distributions(frame)
    assert first.plots[0].figure is not second.plots[0].figure
    _close(first)
    _close(second)


def test_distribution_selection_order_budgets_and_immutability() -> None:
    frame = pd.DataFrame(
        {
            "cat": ["b", "a", "b", None],
            "num": pd.Series([1, 2, np.inf, pd.NA], dtype="Float64"),
            "flag": [True, False, True, None],
            "when": pd.to_datetime(["2020-01-01"] * 4),
            "delta": pd.to_timedelta([1, 2, 3, 4], unit="D"),
        }
    )
    original = frame.copy(deep=True)
    result = plot_distributions(frame, max_plots=3, sample_size=1)
    assert [plot.item for plot in result.plots] == ["cat", "num", "flag"]
    assert result.truncated is False
    pd.testing.assert_frame_equal(frame, original)
    _close(result)
    with pytest.raises(
        ValueError, match="sample_size must be an integer from 1 to 10000"
    ):
        plot_distributions(frame, sample_size=True)


def test_missingness_metadata_boundary_and_inputs_unchanged() -> None:
    frame = pd.DataFrame({f"c{i}": [np.nan, i] for i in range(51)})
    original = frame.copy(deep=True)
    result = plot_missingness(frame, max_columns=50)
    assert result.metadata == (
        ("n_rows", "2"),
        ("requested_columns", "50"),
        ("available_columns", "51"),
        (
            "analyzed_columns",
            (
                '["c0","c1","c2","c3","c4","c5","c6","c7","c8","c9",'
                '"c10","c11","c12","c13","c14","c15","c16","c17","c18",'
                '"c19","c20","c21","c22","c23","c24","c25","c26","c27",'
                '"c28","c29","c30","c31","c32","c33","c34","c35","c36",'
                '"c37","c38","c39","c40","c41","c42","c43","c44","c45",'
                '"c46","c47","c48","c49"]'
            ),
        ),
        ("truncated_columns", "true"),
        ("truncation_reason", "max_columns"),
    )
    assert result.figure.axes[0].get_ylim() == (0.0, 1.0)
    pd.testing.assert_frame_equal(frame, original)
    plt.close(result.figure)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda frame: frame.assign(column_a="a"),
        lambda frame: frame.assign(method="spearman"),
        lambda frame: pd.concat([frame, frame.iloc[[0]]], ignore_index=True),
    ],
)
def test_correlation_malformed_pairs_are_rejected(mutator: object) -> None:
    result = compute_correlations(pd.DataFrame({"a": [1, 2, 3], "b": [2, 3, 4]}))
    broken = result.__class__(
        **{**result.__dict__, "correlations": mutator(result.correlations.copy())}
    )
    with pytest.raises(ValueError, match="correlation result has invalid schema"):
        plot_correlations(broken)


def _truncated_correlation_result() -> object:
    frame = pd.DataFrame(
        {f"c{index}": np.arange(4, dtype=float) + index for index in range(51)}
    )
    return compute_correlations(frame, max_columns=50)


@pytest.mark.parametrize(
    "scenario",
    [
        "duplicate analyzed column",
        "non-string analyzed column",
        "boolean max columns",
        "zero max columns",
        "negative max columns",
        "non-boolean truncated",
        "untruncated excess reason",
        "truncated missing reason",
        "truncated wrong reason",
        "truncated column count mismatch",
        "pair feature outside metadata",
    ],
)
def test_correlations_rejects_invalid_result_metadata(scenario: str) -> None:
    single = compute_correlations(pd.DataFrame({"a": [1.0, 2.0, 3.0]}))
    pair = compute_correlations(
        pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [2.0, 3.0, 4.0]})
    )
    truncated = _truncated_correlation_result()
    if scenario == "duplicate analyzed column":
        broken = single.__class__(**{**single.__dict__, "analyzed_columns": ("a", "a")})
    elif scenario == "non-string analyzed column":
        broken = single.__class__(**{**single.__dict__, "analyzed_columns": ("a", 1)})
    elif scenario == "boolean max columns":
        broken = single.__class__(**{**single.__dict__, "max_columns": True})
    elif scenario == "zero max columns":
        broken = single.__class__(**{**single.__dict__, "max_columns": 0})
    elif scenario == "negative max columns":
        broken = single.__class__(**{**single.__dict__, "max_columns": -2})
    elif scenario == "non-boolean truncated":
        broken = single.__class__(**{**single.__dict__, "truncated": "false"})
    elif scenario == "untruncated excess reason":
        broken = truncated.__class__(**{**truncated.__dict__, "truncated": False})
    elif scenario == "truncated missing reason":
        broken = truncated.__class__(
            **{**truncated.__dict__, "skipped_columns": (), "skipped_reasons": {}}
        )
    elif scenario == "truncated wrong reason":
        skipped = dict(truncated.skipped_reasons)
        skipped[next(iter(skipped))] = "constant"
        broken = truncated.__class__(
            **{**truncated.__dict__, "skipped_reasons": skipped}
        )
    elif scenario == "truncated column count mismatch":
        columns = truncated.analyzed_columns[:-1]
        pairs = truncated.correlations[
            truncated.correlations["column_a"].isin(columns)
            & truncated.correlations["column_b"].isin(columns)
        ].copy()
        broken = truncated.__class__(
            **{**truncated.__dict__, "analyzed_columns": columns, "correlations": pairs}
        )
    else:
        correlations = pair.correlations.copy()
        correlations.loc[correlations.index[0], "column_a"] = "outside"
        broken = pair.__class__(**{**pair.__dict__, "correlations": correlations})
    with pytest.raises(
        ValueError,
        match="^correlation result has invalid schema$",
    ):
        plot_correlations(broken)


def test_correlations_accept_valid_single_and_truncated_metadata() -> None:
    single = compute_correlations(pd.DataFrame({"a": [1.0, 2.0, 3.0]}))
    truncated = _truncated_correlation_result()
    single_plot = plot_correlations(single)
    truncated_plot = plot_correlations(truncated)
    try:
        assert single_plot.chart_type == "correlation_heatmap"
        assert truncated_plot.chart_type == "correlation_heatmap"
    finally:
        plt.close(single_plot.figure)
        plt.close(truncated_plot.figure)


def test_library_does_not_mutate_global_plot_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = matplotlib.get_backend()
    rc = matplotlib.rcParams.copy()

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("library must not call pyplot lifecycle functions")

    with monkeypatch.context() as patch:
        patch.setattr(plt, "show", forbidden)
        patch.setattr(plt, "close", forbidden)
        result = plot_distributions(pd.DataFrame({"x": [1, 2]}))
    assert matplotlib.get_backend() == backend
    assert matplotlib.rcParams == rc
    _close(result)


def test_outlier_and_group_contract_smoke_and_errors() -> None:
    analysis = detect_outliers(pd.DataFrame({"a": [1, 2, 3, 4], "b": [1, 2, 3, 4]}))
    plots = plot_outliers(analysis)
    assert [plot.item for plot in plots.plots] == list(analysis.analyzed_columns)
    assert all(plot.chart_type == "outlier_rate" for plot in plots.plots)
    _close(plots)
    broken = analysis.__class__(
        **{**analysis.__dict__, "summary": analysis.summary.iloc[:, :-1]}
    )
    with pytest.raises(ValueError, match="outlier result has invalid schema"):
        plot_outliers(broken)


@pytest.mark.parametrize(
    ("scenario", "change"),
    [
        ("detail lower bound mismatch", "lower_bound"),
        ("detail upper bound mismatch", "upper_bound"),
        ("summary count exceeds detail rows", "count_too_high"),
        ("summary count is below detail rows", "count_too_low"),
        ("zero count has detail rows", "zero_count"),
        ("nonzero count has no detail rows", "missing_details"),
        ("detail has unknown feature", "unknown_feature"),
        ("boolean outlier count", "boolean_count"),
        ("negative outlier count", "negative_count"),
        ("valid control", None),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_outliers_rejects_detail_result_mismatch(
    scenario: str,
    change: str | None,
) -> None:
    del scenario
    base = detect_outliers(pd.DataFrame({"x": [0.0] * 8 + [10.0, 20.0]}))
    if change is None:
        plots = plot_outliers(base)
        assert plots.actual_count == 1
        _close(plots)
        return

    attributes = dict(base.__dict__)
    if change in {"lower_bound", "upper_bound", "unknown_feature"}:
        details = base.outliers.copy()
        if change == "lower_bound":
            details.loc[details.index[0], "lower_bound"] = 1.0
        elif change == "upper_bound":
            details.loc[details.index[0], "upper_bound"] = 1.0
        else:
            details.loc[details.index[0], "column"] = "unknown"
        attributes["outliers"] = details
    elif change == "count_too_high":
        summary = base.summary.copy()
        summary.loc[summary.index[0], "outlier_count"] = 3
        attributes["summary"] = summary
    elif change == "count_too_low":
        summary = base.summary.copy()
        summary.loc[summary.index[0], "outlier_count"] = 1
        attributes["summary"] = summary
    elif change == "zero_count":
        summary = base.summary.copy()
        summary.loc[summary.index[0], "outlier_count"] = 0
        attributes["summary"] = summary
    elif change == "missing_details":
        attributes["outliers"] = base.outliers.iloc[0:0].copy()
    elif change == "boolean_count":
        summary = base.summary.astype({"outlier_count": "object"})
        summary.loc[summary.index[0], "outlier_count"] = True
        attributes["summary"] = summary
    else:
        summary = base.summary.copy()
        summary.loc[summary.index[0], "outlier_count"] = -1
        attributes["summary"] = summary
    broken = base.__class__(**attributes)
    with pytest.raises(
        ValueError,
        match="^outlier result has invalid schema$",
    ):
        plot_outliers(broken)


def _twenty_one_feature_outlier_result() -> object:
    frame = pd.DataFrame({f"c{index}": [0.0] * 8 + [10.0, 20.0] for index in range(21)})
    return detect_outliers(frame)


@pytest.mark.parametrize(
    "scenario",
    [
        "method mismatch",
        "threshold mismatch",
        "missing summary row",
        "duplicate summary row",
        "outlier count mismatch",
        "detail bounds mismatch",
    ],
)
def test_outliers_validates_features_beyond_plot_budget(scenario: str) -> None:
    result = _twenty_one_feature_outlier_result()
    attributes = dict(result.__dict__)
    last_feature = result.analyzed_columns[-1]
    if scenario == "method mismatch":
        summary = result.summary.copy()
        summary.loc[summary["column"] == last_feature, "method"] = "wrong"
        attributes["summary"] = summary
    elif scenario == "threshold mismatch":
        summary = result.summary.copy()
        summary.loc[summary["column"] == last_feature, "threshold"] = 2.0
        attributes["summary"] = summary
    elif scenario == "missing summary row":
        attributes["summary"] = result.summary.iloc[:-1].copy()
    elif scenario == "duplicate summary row":
        attributes["summary"] = pd.concat(
            [result.summary, result.summary.iloc[[-1]]], ignore_index=True
        )
    elif scenario == "outlier count mismatch":
        summary = result.summary.copy()
        summary.loc[summary["column"] == last_feature, "outlier_count"] = 0
        attributes["summary"] = summary
    else:
        details = result.outliers.copy()
        index = details.index[details["column"] == last_feature][0]
        details.loc[index, "lower_bound"] = 1.0
        attributes["outliers"] = details
    broken = result.__class__(**attributes)
    before = set(plt.get_fignums())
    with pytest.raises(
        ValueError,
        match="^outlier result has invalid schema$",
    ):
        plot_outliers(broken, max_plots=20)
    assert set(plt.get_fignums()) == before


def test_outliers_create_only_budgeted_figures_after_full_validation() -> None:
    result = _twenty_one_feature_outlier_result()
    before = set(plt.get_fignums())
    plots = plot_outliers(result, max_plots=20)
    created = set(plt.get_fignums()) - before
    try:
        assert (plots.available_count, plots.actual_count, plots.truncated) == (
            21,
            20,
            True,
        )
        assert len(plots.plots) == len(created) == 20
    finally:
        _close(plots)


def test_result_only_plots_do_not_recompute(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = pd.DataFrame(
        {
            "g": ["a", "a", "b", "b"],
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": ["n", "n", "p", "p"],
        }
    )
    corr = compute_correlations(frame[["x"]])
    outlier = detect_outliers(frame[["x"]])
    group = compare_groups(frame, "g", values=["x"])
    target = analyze_target_relationships(
        frame, "y", task="classification", features=["x", "g"]
    )
    import sharper.analysis as analysis
    import sharper.features as features

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("recomputed")

    for module, name in (
        (analysis, "compute_correlations"),
        (analysis, "detect_outliers"),
        (analysis, "compare_groups"),
        (analysis, "analyze_target_relationships"),
        (features, "suggest_feature_derivations"),
        (features, "derive_features"),
    ):
        monkeypatch.setattr(module, name, forbidden)
    heat = plot_correlations(corr)
    outs = plot_outliers(outlier)
    groups = plot_group_comparison(group)
    targets = plot_target_relationships(target)
    plt.close(heat.figure)
    _close(outs)
    _close(groups)
    _close(targets)


def test_group_and_target_paths_metadata_labels_and_immutability() -> None:
    frame = pd.DataFrame(
        {
            "g": [1, "1", 1, "1"],
            "x": [1.0, 2.0, 3.0, 4.0],
            "cls": ["a", "a", "b", "b"],
            "reg": [1.0, 2.0, 3.0, 4.0],
        }
    )
    group = compare_groups(frame, "g", values=["x"])
    before = group.summary.copy(deep=True)
    groups = plot_group_comparison(group)
    assert groups.plots[0].chart_type == "group_median"
    assert [
        tick.get_text() for tick in groups.plots[0].figure.axes[0].get_xticklabels()
    ] == ["1", "1"]
    assert [key for key, _ in groups.plots[0].metadata] == [
        "value",
        "displayed_groups",
        "finite_medians",
        "metric",
        "error_bars",
    ]
    pd.testing.assert_frame_equal(group.summary, before)
    _close(groups)
    classification = analyze_target_relationships(
        frame, "cls", task="classification", features=["x", "g"]
    )
    regression = analyze_target_relationships(
        frame, "reg", task="regression", features=["g", "x"]
    )
    cplots = plot_target_relationships(classification)
    rplots = plot_target_relationships(regression)
    assert {plot.chart_type for plot in cplots.plots} == {
        "target_classification_numeric",
        "target_classification_categorical",
    }
    assert [plot.chart_type for plot in rplots.plots] == [
        "target_regression_categorical"
    ]
    assert all(plot.metadata for plot in cplots.plots + rplots.plots)
    _close(cplots)
    _close(rplots)


@pytest.mark.parametrize(
    ("scenario", "is_truncated", "change"),
    [
        ("available group count incorrect", False, "available"),
        ("displayed group count incorrect", True, "displayed"),
        ("displayed exceeds available", False, "displayed_exceeds_available"),
        ("untruncated displayed below available", False, "displayed_below_available"),
        ("truncated displayed equals available", True, "displayed_equals_available"),
        ("truncated reason missing", True, "missing_reason"),
        ("untruncated reason supplied", False, "unexpected_reason"),
        ("summary rows mismatch available count", False, "short_summary"),
        ("boolean count", False, "boolean_available"),
        ("negative count", False, "negative_available"),
        ("valid control", False, None),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_group_comparison_rejects_metadata_result_mismatch(
    scenario: str,
    is_truncated: bool,
    change: str | None,
) -> None:
    del scenario
    if is_truncated:
        frame = pd.DataFrame(
            {
                "g": ["a", "a", "b", "b", "c", "c"],
                "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            }
        )
    else:
        frame = pd.DataFrame({"g": ["a", "a", "b", "b"], "x": [1.0, 2.0, 3.0, 4.0]})
    base = compare_groups(frame, "g", values=["x"], max_groups=2)
    if change is None:
        plots = plot_group_comparison(base)
        assert plots.actual_count == 1
        _close(plots)
        return

    attributes = dict(base.__dict__)
    if change == "available":
        attributes["available_group_count"] = 3
    elif change == "displayed":
        attributes["displayed_group_count"] = 1
    elif change == "displayed_exceeds_available":
        attributes["displayed_group_count"] = 3
    elif change == "displayed_below_available":
        attributes["displayed_group_count"] = 1
    elif change == "displayed_equals_available":
        attributes["displayed_group_count"] = 3
    elif change == "missing_reason":
        attributes["truncation_reason"] = None
    elif change == "unexpected_reason":
        attributes["truncation_reason"] = "exceeds_max_groups"
    elif change == "short_summary":
        attributes["summary"] = base.summary.iloc[:1].copy()
    elif change == "boolean_available":
        attributes["available_group_count"] = True
    else:
        attributes["available_group_count"] = -1
    broken = base.__class__(**attributes)
    with pytest.raises(
        ValueError,
        match="^group result has invalid schema$",
    ):
        plot_group_comparison(broken)


@pytest.mark.parametrize("kind", ["missing_block", "unknown", "duplicate"])
def test_group_malformed_cases(kind: str) -> None:
    base = compare_groups(
        pd.DataFrame({"g": ["a", "a", "b", "b"], "x": [1.0, 2.0, 3.0, 4.0]}),
        "g",
        values=["x"],
    )
    summary = base.summary.copy()
    if kind == "missing_block":
        summary = summary.iloc[0:0]
    elif kind == "unknown":
        summary.loc[:, "value"] = "unknown"
    else:
        summary = pd.concat([summary, summary.iloc[[0]]], ignore_index=True)
    broken = base.__class__(**{**base.__dict__, "summary": summary})
    with pytest.raises(ValueError, match="group result has invalid schema"):
        plot_group_comparison(broken)


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r.iloc[:, ::-1],
        lambda r: r.astype({"correlation": "object"}),
        lambda r: r.assign(column_b="a"),
        lambda r: r.assign(column_a="unknown"),
        lambda r: r.assign(correlation=np.inf),
    ],
)
def test_correlation_schema_matrix(change: object) -> None:
    result = compute_correlations(pd.DataFrame({"a": [1, 2, 3], "b": [3, 2, 1]}))
    broken = result.__class__(
        **{**result.__dict__, "correlations": change(result.correlations.copy())}
    )
    with pytest.raises(ValueError, match="correlation result has invalid schema"):
        plot_correlations(broken)


@pytest.mark.parametrize("argument", [0, True, 21])
@pytest.mark.parametrize("sample", [0, True, 10_001])
@pytest.mark.parametrize("columns", [0, True, 51])
def test_plot_budgets_reject_invalid_values(
    argument: int, sample: int, columns: int
) -> None:
    frame = pd.DataFrame({"x": [1, 2]})
    with pytest.raises(ValueError, match="max_plots must be an integer from 1 to 20"):
        plot_distributions(frame, max_plots=argument)
    with pytest.raises(
        ValueError, match="sample_size must be an integer from 1 to 10000"
    ):
        plot_distributions(frame, sample_size=sample)
    with pytest.raises(ValueError, match="max_columns must be an integer from 1 to 50"):
        plot_missingness(frame, max_columns=columns)


def test_distribution_and_missingness_metadata_tuples() -> None:
    frame = pd.DataFrame({"n": [1.0, 2.0, np.nan, np.inf], "c": ["b", "a", "b", None]})
    plots = plot_distributions(frame)
    missing = plot_missingness(pd.DataFrame({f"c{i}": [np.nan] for i in range(51)}))
    try:
        assert plots.plots[0].chart_type == "distribution_histogram"
        assert plots.plots[0].metadata == (
            ("column", "n"),
            ("dtype", "float64"),
            ("finite_count", "2"),
            ("missing_count", "1"),
            ("non_finite_count", "1"),
            ("sample_size_requested", "10000"),
            ("sample_size_actual", "2"),
            ("bins", "2"),
        )
        assert plots.plots[1].chart_type == "distribution_categories"
        assert plots.plots[1].metadata == (
            ("column", "c"),
            ("dtype", "str"),
            ("non_missing_count", "3"),
            ("missing_count", "1"),
            ("available_categories", "2"),
            ("displayed_categories", '["b","a"]'),
            ("truncated_categories", "false"),
            ("category_limit", "20"),
        )
        assert missing.chart_type == "missingness_rate"
        assert missing.metadata == (
            ("n_rows", "1"),
            ("requested_columns", "50"),
            ("available_columns", "51"),
            (
                "analyzed_columns",
                (
                    '["c0","c1","c2","c3","c4","c5","c6","c7","c8",'
                    '"c9","c10","c11","c12","c13","c14","c15","c16",'
                    '"c17","c18","c19","c20","c21","c22","c23","c24",'
                    '"c25","c26","c27","c28","c29","c30","c31","c32",'
                    '"c33","c34","c35","c36","c37","c38","c39","c40",'
                    '"c41","c42","c43","c44","c45","c46","c47","c48",'
                    '"c49"]'
                ),
            ),
            ("truncated_columns", "true"),
            ("truncation_reason", "max_columns"),
        )
    finally:
        _close(plots)
        plt.close(missing.figure)


def test_result_based_plot_metadata_tuples() -> None:
    distributions = plot_distributions(
        pd.DataFrame({"n": [1.0, 2.0, np.nan, np.inf], "c": ["b", "a", "b", None]})
    )
    missing = plot_missingness(pd.DataFrame({f"c{i}": [np.nan] for i in range(51)}))
    frame = pd.DataFrame(
        {
            "g": ["u", "u", "v", "v"],
            "x": [1.0, 2.0, 3.0, 4.0],
            "cls": ["a", "a", "b", "b"],
            "reg": [1.0, 2.0, 3.0, 4.0],
        }
    )
    corr = plot_correlations(compute_correlations(frame[["x", "reg"]]))
    out = plot_outliers(detect_outliers(frame[["x"]]))
    group = plot_group_comparison(compare_groups(frame, "g", values=["x"]))
    cls = plot_target_relationships(
        analyze_target_relationships(
            frame, "cls", task="classification", features=["x", "g"]
        )
    )
    reg = plot_target_relationships(
        analyze_target_relationships(
            frame, "reg", task="regression", features=["g", "x"]
        )
    )
    try:
        assert corr.chart_type == "correlation_heatmap"
        assert corr.metadata == (
            ("method", "pearson"),
            ("analyzed_columns", '["x","reg"]'),
            ("pair_rows", "1"),
            ("missing_pairs", "[]"),
            ("input_max_columns", "50"),
            ("input_truncated", "false"),
            ("annotation_format", ".2f"),
        )
        assert out.plots[0].chart_type == "outlier_rate"
        assert out.plots[0].metadata == (
            ("displayed_features", '["x"]'),
            ("truncated_features", "false"),
            ("outlier_count", "0"),
            ("outlier_rate", "0"),
            ("lower_bound", "-0.5"),
            ("upper_bound", "5.5"),
            ("threshold", "1.5"),
        )
        assert group.plots[0].chart_type == "group_median"
        assert group.plots[0].metadata == (
            ("value", "x"),
            ("displayed_groups", '["u","v"]'),
            ("finite_medians", "2"),
            ("metric", "median"),
            ("error_bars", "q25_q75"),
        )
        numeric, categorical = cls.plots
        assert (
            distributions.plots[0].chart_type,
            distributions.plots[1].chart_type,
            missing.chart_type,
            corr.chart_type,
            out.plots[0].chart_type,
            group.plots[0].chart_type,
            numeric.chart_type,
            categorical.chart_type,
            reg.plots[0].chart_type,
        ) == (
            "distribution_histogram",
            "distribution_categories",
            "missingness_rate",
            "correlation_heatmap",
            "outlier_rate",
            "group_median",
            "target_classification_numeric",
            "target_classification_categorical",
            "target_regression_categorical",
        )
        assert numeric.chart_type == "target_classification_numeric"
        assert numeric.metadata == (
            ("feature", "x"),
            ("analysis_type", "classification_numeric"),
            ("target_categories", '["a","b"]'),
            ("metric", "median"),
            ("error_bars", "q25_q75"),
        )
        assert categorical.chart_type == "target_classification_categorical"
        assert categorical.metadata == (
            ("feature", "g"),
            ("analysis_type", "classification_categorical"),
            ("feature_categories", '["u","v"]'),
            ("target_categories", '["a","b"]'),
            ("metric", "count"),
        )
        assert reg.plots[0].chart_type == "target_regression_categorical"
        assert reg.plots[0].metadata == (
            ("feature", "g"),
            ("analysis_type", "regression_categorical"),
            ("feature_categories", '["u","v"]'),
            ("target_categories", "[]"),
            ("metric", "target_median"),
        )
    finally:
        _close(distributions)
        plt.close(missing.figure)
        plt.close(corr.figure)
        _close(out)
        _close(group)
        _close(cls)
        _close(reg)
