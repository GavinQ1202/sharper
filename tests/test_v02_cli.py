"""Task 20 Wave I4 opt-in CLI contract tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

import sharper.data_audit as data_audit
import sharper.decision_strategy as decision_strategy
import sharper.lifecycle_monitoring as lifecycle_monitoring
import sharper.model_governance as model_governance
import sharper.risk_validation as risk_validation
import sharper.v02_cli as v02_cli
import sharper.v02_reporting as reporting
from sharper import cli
from sharper.reporting import ReportArtifact
from sharper.v02_workflow import (
    V02PostLoanRequest,
    V02PreLoanRequest,
    V02WorkflowRequest,
    V02WorkflowResult,
)

runner = CliRunner()


def _path_status(enabled: tuple[str, ...] = ()) -> pd.DataFrame:
    paths = ("score_validation", "audit", "preloan", "postloan", "governance")
    return pd.DataFrame(
        {
            "path_key": paths,
            "enabled": [path in enabled for path in paths],
            "status": [
                "completed" if path in enabled else "not_requested" for path in paths
            ],
            "reason": [
                "completed" if path in enabled else "path_not_requested"
                for path in paths
            ],
        }
    )


def _workflow_result() -> V02WorkflowResult:
    return V02WorkflowResult(
        "task20-integration-v1",
        (),
        _path_status(),
        (),
        None,
        None,
        None,
        None,
        None,
        (),
        (),
    )


def _score_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target": [0, 1, 0, 1, 0, 1, 0, 1],
            "score": [0.05, 0.90, 0.20, 0.80, 0.15, 0.75, 0.10, 0.65],
            "mask": [True, True, True, True, False, False, False, False],
            "entity": ["a", "b", "c", "d", "e", "f", "g", "h"],
        }
    )


def _score_args(
    output: Path, *, label_type: str = "int", label_value: str = "1"
) -> list[str]:
    return [
        "v02-run",
        "input.csv",
        "--output",
        str(output),
        "--target",
        "target",
        "--external-ranking-score-column",
        "score",
        "--ranking-direction",
        "higher_risk",
        "--score-validation-mask-column",
        "mask",
        "--positive-label-type",
        label_type,
        "--positive-label-value",
        label_value,
    ]


def test_v02_report_and_cli_do_not_call_domain_owners(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner_calls: list[str] = []

    def owner_spy(name: str):
        def call(*args, **kwargs):
            owner_calls.append(name)
            raise AssertionError(f"direct Task15-19 owner call: {name}")

        return call

    monkeypatch.setattr(risk_validation, "validate_binary_risk", owner_spy("Task15"))
    monkeypatch.setattr(data_audit, "audit_data_quality", owner_spy("Task16"))
    monkeypatch.setattr(
        decision_strategy,
        "simulate_decision_strategy",
        owner_spy("Task17"),
    )
    monkeypatch.setattr(lifecycle_monitoring, "monitor_lifecycle", owner_spy("Task18"))
    monkeypatch.setattr(model_governance, "evaluate_governance", owner_spy("Task19"))

    reporting.generate_v02_report(_workflow_result(), tmp_path / "report-boundary.md")
    assert owner_calls == []

    monkeypatch.setattr(v02_cli, "load_csv", lambda path: _score_frame())
    workflow_calls: list[V02WorkflowRequest] = []
    monkeypatch.setattr(
        v02_cli,
        "run_v02_workflow",
        lambda request: workflow_calls.append(request) or _workflow_result(),
    )
    report_calls: list[Path] = []
    monkeypatch.setattr(
        v02_cli,
        "generate_v02_report",
        lambda result, output_path, **kwargs: (
            report_calls.append(Path(output_path))
            or ReportArtifact(Path(output_path), kwargs["format"], "title")
        ),
    )
    result = runner.invoke(
        cli.app,
        [
            "v02-run",
            "input.csv",
            "-o",
            str(tmp_path / "cli-boundary.md"),
            "--audit",
        ],
    )

    assert result.exit_code == 0
    assert len(workflow_calls) == 1
    assert report_calls == [tmp_path / "cli-boundary.md"]
    assert owner_calls == []


def test_v02_python_cli_semantic_config_parity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = _score_frame()
    original = frame.copy(deep=True)
    captured: list[V02WorkflowRequest] = []

    monkeypatch.setattr(v02_cli, "load_csv", lambda path: frame)

    def workflow(request: V02WorkflowRequest) -> V02WorkflowResult:
        captured.append(request)
        return _workflow_result()

    monkeypatch.setattr(v02_cli, "run_v02_workflow", workflow)
    monkeypatch.setattr(
        v02_cli,
        "generate_v02_report",
        lambda result, output_path, **kwargs: ReportArtifact(
            Path(output_path), kwargs["format"], "Sharper v0.2 Integration Report"
        ),
    )

    result = runner.invoke(cli.app, _score_args(tmp_path / "report.md"))

    assert result.exit_code == 0
    assert len(captured) == 1
    score = captured[0].score_validation
    assert score is not None
    assert score.target == "target"
    assert score.positive_label == 1
    assert score.config.validation_mode == "stratified_holdout"
    assert score.config.test_size == 0.20
    assert score.config.random_state == 42
    assert score.external_predictions is not None
    assert score.external_predictions.row_positions == (0, 1, 2, 3)
    assert score.external_predictions.fold_ids == (0, 0, 0, 0)
    assert score.external_predictions.fold_fit_row_positions == ((0, (4, 5, 6, 7)),)
    assert score.external_predictions.ranking_scores == (0.05, 0.9, 0.2, 0.8)
    assert score.external_predictions.ranking_direction == "higher_risk"
    assert score.external_predictions.event_probabilities is None
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize(
    ("label_type", "label_value", "expected"),
    [
        ("str", "1", "1"),
        ("str", " 1 ", " 1 "),
        ("int", "1", 1),
        ("bool", "true", True),
        ("bool", "false", False),
    ],
)
def test_v02_typed_positive_label_mapping(
    label_type: str, label_value: str, expected: object
) -> None:
    request = v02_cli._score_request(
        _score_frame(),
        target="target",
        ranking_column="score",
        probability_column=None,
        ranking_direction="higher_risk",
        probability_provenance=None,
        mask_column="mask",
        positive_label_type=label_type,
        positive_label_value=label_value,
        test_size=0.20,
        random_state=42,
    )
    assert request is not None
    assert type(request.positive_label) is type(expected)
    assert request.positive_label == expected


def test_v02_cli_score_probability_mode_and_mutual_exclusion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = _score_frame()
    captured: list[V02WorkflowRequest] = []
    monkeypatch.setattr(v02_cli, "load_csv", lambda path: frame)
    monkeypatch.setattr(
        v02_cli,
        "run_v02_workflow",
        lambda request: captured.append(request) or _workflow_result(),
    )
    monkeypatch.setattr(
        v02_cli,
        "generate_v02_report",
        lambda result, output_path, **kwargs: ReportArtifact(
            Path(output_path), kwargs["format"], "title"
        ),
    )
    result = runner.invoke(
        cli.app,
        [
            "v02-run",
            "input.csv",
            "-o",
            str(tmp_path / "prob.md"),
            "--target",
            "target",
            "--external-event-probability-column",
            "score",
            "--probability-provenance",
            "external_declared",
            "--score-validation-mask-column",
            "mask",
            "--positive-label-type",
            "bool",
            "--positive-label-value",
            "true",
        ],
    )
    assert result.exit_code == 0
    external = captured[0].score_validation.external_predictions
    assert external is not None
    assert external.ranking_scores is None
    assert external.event_probabilities == (0.05, 0.9, 0.2, 0.8)
    assert external.probability_positive_label is True
    assert external.probability_provenance == "external_declared"

    conflict = runner.invoke(
        cli.app,
        _score_args(tmp_path / "conflict.md")
        + ["--external-event-probability-column", "score"],
    )
    assert conflict.exit_code == 2
    assert conflict.stderr.strip() == "sharper task20: cli_argument"


@pytest.mark.parametrize(
    ("label_type", "label_value"),
    [
        ("float", "1"),
        ("int", "+1"),
        ("int", "01"),
        ("int", "1.0"),
        ("bool", "TRUE"),
        ("bool", "1"),
    ],
)
def test_v02_cli_rejects_invalid_typed_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    label_type: str,
    label_value: str,
) -> None:
    monkeypatch.setattr(v02_cli, "load_csv", lambda path: _score_frame())
    workflow_calls: list[V02WorkflowRequest] = []
    monkeypatch.setattr(
        v02_cli,
        "run_v02_workflow",
        lambda request: workflow_calls.append(request) or _workflow_result(),
    )
    result = runner.invoke(
        cli.app,
        _score_args(
            tmp_path / f"invalid-{label_type}-{label_value}.md",
            label_type=label_type,
            label_value=label_value,
        ),
    )
    assert result.exit_code == 2
    assert result.stderr.strip() == "sharper task20: cli_argument"
    assert workflow_calls == []


def test_v02_cli_exit_codes_and_no_default_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(v02_cli, "load_csv", lambda path: _score_frame())
    monkeypatch.setattr(v02_cli, "run_v02_workflow", lambda request: _workflow_result())
    monkeypatch.setattr(
        v02_cli,
        "generate_v02_report",
        lambda result, output_path, **kwargs: ReportArtifact(
            Path(output_path), kwargs["format"], "title"
        ),
    )
    success = runner.invoke(cli.app, _score_args(tmp_path / "ok.md"))
    assert success.exit_code == 0

    argument = runner.invoke(
        cli.app,
        [
            "v02-run",
            "input.csv",
            "-o",
            str(tmp_path / "bad.md"),
            "--reference-input",
            "ref.csv",
        ],
    )
    assert argument.exit_code == 2
    assert "Traceback" not in argument.output

    output = runner.invoke(
        cli.app, ["v02-run", "input.csv", "-o", str(tmp_path), "--audit"]
    )
    assert output.exit_code == 2
    assert output.stderr.strip() == "sharper task20: cli_output"
    assert "Traceback" not in output.output

    monkeypatch.setattr(
        v02_cli, "load_csv", lambda path: (_ for _ in ()).throw(OSError("disk"))
    )
    filesystem = runner.invoke(
        cli.app, ["v02-run", "input.csv", "-o", str(tmp_path / "io.md"), "--audit"]
    )
    assert filesystem.exit_code == 3
    assert "Traceback" not in filesystem.output

    monkeypatch.setattr(
        v02_cli,
        "load_csv",
        lambda path: (_ for _ in ()).throw(RuntimeError("secret-token")),
    )
    unexpected = runner.invoke(
        cli.app,
        ["v02-run", "input.csv", "-o", str(tmp_path / "internal.md"), "--audit"],
    )
    assert unexpected.exit_code == 70
    assert unexpected.output.strip() == "internal error"
    assert "secret-token" not in unexpected.output
    assert "Traceback" not in unexpected.output


def test_v02_cli_audit_reference_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = _score_frame()
    reference = pd.DataFrame({"reference_value": [1, 2]})
    original = frame.copy(deep=True)
    loaded: list[str] = []
    captured: list[V02WorkflowRequest] = []

    def load(path: Path) -> pd.DataFrame:
        loaded.append(str(path))
        return reference if Path(path).name == "ref.csv" else frame

    monkeypatch.setattr(v02_cli, "load_csv", load)
    monkeypatch.setattr(
        v02_cli,
        "run_v02_workflow",
        lambda request: captured.append(request) or _workflow_result(),
    )
    monkeypatch.setattr(
        v02_cli,
        "generate_v02_report",
        lambda result, output_path, **kwargs: ReportArtifact(
            Path(output_path), kwargs["format"], "title"
        ),
    )

    result = runner.invoke(
        cli.app,
        [
            "v02-run",
            "input.csv",
            "-o",
            str(tmp_path / "audit.md"),
            "--audit",
            "--reference-input",
            "ref.csv",
        ],
    )

    assert result.exit_code == 0
    assert loaded == ["input.csv", "ref.csv"]
    assert len(captured) == 1
    assert captured[0].audit is not None
    pd.testing.assert_frame_equal(captured[0].audit.reference, reference)
    pd.testing.assert_frame_equal(frame, original)


def test_v02_cli_error_keys_and_spec_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(v02_cli, "load_csv", lambda path: _score_frame())
    argument = runner.invoke(
        cli.app,
        [
            "v02-run",
            "input.csv",
            "-o",
            str(tmp_path / "bad.md"),
            "--external-ranking-score-column",
            "score",
        ],
    )
    assert argument.exit_code == 2
    assert argument.stderr.strip() == "sharper task20: cli_argument"

    monkeypatch.setattr(v02_cli, "load_v02_json", lambda policy, warning: (None, None))
    spec = runner.invoke(
        cli.app,
        [
            "v02-run",
            "input.csv",
            "-o",
            str(tmp_path / "spec.md"),
            "--policy-json",
            "policy.json",
        ],
    )
    assert spec.exit_code == 2
    assert spec.stderr.strip() == "sharper task20: cli_spec_required"

    output = runner.invoke(
        cli.app,
        ["v02-run", "input.csv", "-o", str(tmp_path), "--format", "pdf"],
    )
    assert output.exit_code == 2
    assert output.stderr.strip() == "sharper task20: cli_output"


def test_v02_cli_typer_usage_and_path_rules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert runner.invoke(cli.app, ["v02-run"]).exit_code == 2
    assert runner.invoke(cli.app, ["v02-run", "input.csv"]).exit_code == 2
    assert (
        runner.invoke(
            cli.app, ["v02-run", "input.csv", "-o", "x.md", "--unknown"]
        ).exit_code
        == 2
    )
    primitive = runner.invoke(
        cli.app,
        [
            "v02-run",
            "input.csv",
            "-o",
            str(tmp_path / "x.md"),
            "--score-test-size",
            "bad",
        ],
    )
    assert primitive.exit_code == 2
    assert "Traceback" not in primitive.output

    invalid_suffix = runner.invoke(
        cli.app, ["v02-run", "input.parquet", "-o", str(tmp_path / "x.md"), "--audit"]
    )
    assert invalid_suffix.exit_code == 2
    assert invalid_suffix.stderr.strip() == "sharper task20: cli_argument"

    monkeypatch.setattr(v02_cli, "load_csv", lambda path: _score_frame())
    repeated_label = runner.invoke(
        cli.app,
        _score_args(tmp_path / "repeated.md") + ["--positive-label-type", "str"],
    )
    assert repeated_label.exit_code == 2
    assert repeated_label.stderr.strip() == "sharper task20: cli_argument"


def test_v02_cli_policy_warning_reuse_and_call_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    policy = V02PreLoanRequest(config=None)  # type: ignore[arg-type]
    warning = V02PostLoanRequest(config=None)  # type: ignore[arg-type]
    monkeypatch.setattr(v02_cli, "load_csv", lambda path: _score_frame())

    def load_specs(policy_path, warning_path):
        calls.extend([str(policy_path), str(warning_path)])
        return policy, warning

    monkeypatch.setattr(v02_cli, "load_v02_json", load_specs)
    requests: list[V02WorkflowRequest] = []
    monkeypatch.setattr(
        v02_cli,
        "run_v02_workflow",
        lambda request: requests.append(request) or _workflow_result(),
    )
    reports: list[object] = []
    monkeypatch.setattr(
        v02_cli,
        "generate_v02_report",
        lambda result, output_path, **kwargs: (
            reports.append(kwargs)
            or ReportArtifact(Path(output_path), kwargs["format"], "title")
        ),
    )
    policy_path = tmp_path / "policy.json"
    warning_path = tmp_path / "warning.json"
    result = runner.invoke(
        cli.app,
        [
            "v02-run",
            "input.csv",
            "-o",
            str(tmp_path / "both.md"),
            "--policy-json",
            str(policy_path),
            "--warning-json",
            str(warning_path),
        ],
    )
    assert result.exit_code == 0
    assert calls == [str(policy_path), str(warning_path)]
    assert len(requests) == 1
    assert requests[0].preloan is policy
    assert requests[0].postloan is warning
    assert len(reports) == 1


def test_v02_cli_workflow_and_report_failure_call_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(v02_cli, "load_csv", lambda path: _score_frame())
    workflow_calls: list[int] = []
    report_calls: list[int] = []

    def workflow_failure(request):
        workflow_calls.append(1)
        raise ValueError("sharper task20: request_path_input_conflict")

    monkeypatch.setattr(v02_cli, "run_v02_workflow", workflow_failure)
    monkeypatch.setattr(
        v02_cli,
        "generate_v02_report",
        lambda *args, **kwargs: report_calls.append(1),
    )
    failed_workflow = runner.invoke(cli.app, _score_args(tmp_path / "workflow.md"))
    assert failed_workflow.exit_code == 2
    assert len(workflow_calls) == 1
    assert report_calls == []

    def report_failure(*args, **kwargs):
        report_calls.append(1)
        raise OSError("failed report")

    monkeypatch.setattr(v02_cli, "run_v02_workflow", lambda request: _workflow_result())
    monkeypatch.setattr(v02_cli, "generate_v02_report", report_failure)
    failed_report = runner.invoke(cli.app, _score_args(tmp_path / "report.md"))
    assert failed_report.exit_code == 3
    assert len(report_calls) == 1
