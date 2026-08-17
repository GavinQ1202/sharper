"""Private opt-in CLI adapter for the Sharper v0.2 workflow."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer

from sharper.io import load_csv, load_excel
from sharper.risk_validation import (
    BinaryRiskValidationConfig,
    ExternalRiskPredictions,
)
from sharper.v02_json import load_v02_json
from sharper.v02_reporting import generate_v02_report
from sharper.v02_workflow import (
    V02AuditRequest,
    V02ScoreValidationRequest,
    V02WorkflowRequest,
    run_v02_workflow,
)

_SUPPORTED_INPUT_SUFFIXES = frozenset({".csv", ".xlsx"})
_SCORE_DIRECTIONS = frozenset({"higher_risk", "lower_risk"})
_PROBABILITY_PROVENANCES = frozenset(
    {"predict_proba", "fold_safe_calibrated", "external_declared"}
)
_POSITIVE_LABEL_TYPES = frozenset({"str", "int", "bool"})
_INTEGER_LABEL = re.compile(r"-?(0|[1-9][0-9]*)\Z")


def v02_run(
    input_path: Annotated[Path, typer.Argument(metavar="INPUT")],
    output_path: Annotated[Path, typer.Option("--output", "-o")],
    format: Annotated[str, typer.Option("--format")] = "markdown",
    policy_json: Annotated[Path | None, typer.Option("--policy-json")] = None,
    warning_json: Annotated[Path | None, typer.Option("--warning-json")] = None,
    audit: Annotated[bool, typer.Option("--audit/--no-audit")] = False,
    reference_input: Annotated[Path | None, typer.Option("--reference-input")] = None,
    target: Annotated[str | None, typer.Option("--target")] = None,
    external_ranking_score_column: Annotated[
        str | None, typer.Option("--external-ranking-score-column")
    ] = None,
    external_event_probability_column: Annotated[
        str | None, typer.Option("--external-event-probability-column")
    ] = None,
    ranking_direction: Annotated[
        str | None, typer.Option("--ranking-direction")
    ] = None,
    probability_provenance: Annotated[
        str | None, typer.Option("--probability-provenance")
    ] = None,
    score_validation_mask_column: Annotated[
        str | None, typer.Option("--score-validation-mask-column")
    ] = None,
    positive_label_type: Annotated[
        list[str] | None, typer.Option("--positive-label-type")
    ] = None,
    positive_label_value: Annotated[
        list[str] | None, typer.Option("--positive-label-value")
    ] = None,
    score_test_size: Annotated[float, typer.Option("--score-test-size")] = 0.20,
    score_random_state: Annotated[int, typer.Option("--score-random-state")] = 42,
    overwrite: Annotated[bool, typer.Option("--overwrite/--no-overwrite")] = True,
) -> None:
    """Run the closed, opt-in v0.2 workflow and write one static report."""
    try:
        artifact = _execute(
            input_path=input_path,
            output_path=output_path,
            format=format,
            policy_json=policy_json,
            warning_json=warning_json,
            audit=audit,
            reference_input=reference_input,
            target=target,
            external_ranking_score_column=external_ranking_score_column,
            external_event_probability_column=external_event_probability_column,
            ranking_direction=ranking_direction,
            probability_provenance=probability_provenance,
            score_validation_mask_column=score_validation_mask_column,
            positive_label_type=positive_label_type,
            positive_label_value=positive_label_value,
            score_test_size=score_test_size,
            score_random_state=score_random_state,
            overwrite=overwrite,
        )
    except (ValueError, TypeError) as error:
        _emit_failure(str(error), 2)
    except OSError as error:
        _emit_failure(str(error), 3)
    except Exception:
        _emit_failure("internal error", 70)
    typer.echo(f"Report written to: {artifact.path}")


def _execute(
    *,
    input_path: Path,
    output_path: Path,
    format: str,
    policy_json: Path | None,
    warning_json: Path | None,
    audit: bool,
    reference_input: Path | None,
    target: str | None,
    external_ranking_score_column: str | None,
    external_event_probability_column: str | None,
    ranking_direction: str | None,
    probability_provenance: str | None,
    score_validation_mask_column: str | None,
    positive_label_type: list[str] | None,
    positive_label_value: list[str] | None,
    score_test_size: float,
    score_random_state: int,
    overwrite: bool,
):
    input_frame = _load_table(input_path)
    reference_frame = None if reference_input is None else _load_table(reference_input)
    policy_request, warning_request = _load_specs(policy_json, warning_json)

    _validate_output(output_path, format)
    resolved_label_type = _single_option(positive_label_type)
    resolved_label_value = _single_option(positive_label_value)
    score_request = _score_request(
        input_frame,
        target=target,
        ranking_column=external_ranking_score_column,
        probability_column=external_event_probability_column,
        ranking_direction=ranking_direction,
        probability_provenance=probability_provenance,
        mask_column=score_validation_mask_column,
        positive_label_type=resolved_label_type,
        positive_label_value=resolved_label_value,
        test_size=score_test_size,
        random_state=score_random_state,
    )
    if policy_json is not None and policy_request is None:
        _task20_error("cli_spec_required")
    if warning_json is not None and warning_request is None:
        _task20_error("cli_spec_required")
    if reference_frame is not None and not audit:
        _task20_error("request_path_input_conflict")

    audit_request = None
    if audit:
        audit_request = V02AuditRequest(reference=reference_frame)
    if (
        score_request is None
        and policy_request is None
        and warning_request is None
        and audit_request is None
    ):
        _task20_error("request_requires_primary_path")

    request = V02WorkflowRequest(
        data=input_frame,
        score_validation=score_request,
        audit=audit_request,
        preloan=policy_request,
        postloan=warning_request,
    )
    result = run_v02_workflow(request)
    return generate_v02_report(
        result,
        output_path,
        format=format,
        overwrite=overwrite,
    )


def _load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix not in _SUPPORTED_INPUT_SUFFIXES:
        _task20_error("cli_argument")
    if suffix == ".csv":
        return load_csv(path)
    return load_excel(path)


def _load_specs(policy_json: Path | None, warning_json: Path | None):
    if policy_json is None and warning_json is None:
        return None, None
    return load_v02_json(policy_json, warning_json)


def _single_option(value: list[str] | None) -> str | None:
    if value is None:
        return None
    if len(value) != 1:
        _task20_error("cli_argument")
    return value[0]


def _validate_output(output_path: Path, format: str) -> None:
    if format not in {"markdown", "html"}:
        _task20_error("cli_output")
    try:
        if output_path.exists() and output_path.is_dir():
            _task20_error("cli_output")
    except (OSError, ValueError):
        _task20_error("cli_output")


def _score_request(
    frame: pd.DataFrame,
    *,
    target: str | None,
    ranking_column: str | None,
    probability_column: str | None,
    ranking_direction: str | None,
    probability_provenance: str | None,
    mask_column: str | None,
    positive_label_type: str | None,
    positive_label_value: str | None,
    test_size: float,
    random_state: int,
) -> V02ScoreValidationRequest | None:
    source_count = int(ranking_column is not None) + int(probability_column is not None)
    score_metadata_present = (
        any(
            value is not None
            for value in (
                target,
                ranking_direction,
                probability_provenance,
                mask_column,
                positive_label_type,
                positive_label_value,
            )
        )
        or test_size != 0.20
        or random_state != 42
    )
    if source_count == 0:
        if score_metadata_present:
            _task20_error("cli_argument")
        return None
    if source_count != 1:
        _task20_error("cli_argument")
    if (
        target is None
        or mask_column is None
        or positive_label_type is None
        or positive_label_value is None
    ):
        _task20_error("cli_argument")
    if (
        type(test_size) is not float
        or not math.isfinite(test_size)
        or not 0.0 < test_size < 1.0
    ):
        _task20_error("cli_argument")
    if (
        type(random_state) is not int
        or isinstance(random_state, bool)
        or random_state < 0
    ):
        _task20_error("cli_argument")
    if positive_label_type not in _POSITIVE_LABEL_TYPES:
        _task20_error("cli_argument")
    positive_label = _decode_label(positive_label_type, positive_label_value)

    if ranking_column is not None:
        if (
            ranking_direction not in _SCORE_DIRECTIONS
            or probability_provenance is not None
        ):
            _task20_error("cli_argument")
        score_series = _column(frame, ranking_column)
        ranking_scores = _masked_values(score_series, mask_column, frame)
        event_probabilities = None
        resolved_direction = ranking_direction
        resolved_provenance = None
    else:
        if (
            probability_provenance not in _PROBABILITY_PROVENANCES
            or ranking_direction is not None
        ):
            _task20_error("cli_argument")
        score_series = _column(frame, probability_column)
        ranking_scores = None
        event_probabilities = _masked_values(score_series, mask_column, frame)
        resolved_direction = None
        resolved_provenance = probability_provenance

    mask_series = _mask_column(frame, mask_column)
    row_positions = tuple(
        int(position)
        for position, value in enumerate(mask_series.to_numpy(copy=False))
        if bool(value)
    )
    fit_positions = tuple(
        int(position)
        for position, value in enumerate(mask_series.to_numpy(copy=False))
        if not bool(value)
    )
    external = ExternalRiskPredictions(
        row_positions=row_positions,
        fold_ids=(0,) * len(row_positions),
        fold_fit_row_positions=((0, fit_positions),),
        ranking_scores=ranking_scores,
        ranking_direction=resolved_direction,
        event_probabilities=event_probabilities,
        probability_positive_label=(
            positive_label if event_probabilities is not None else None
        ),
        probability_provenance=resolved_provenance,
    )
    config = BinaryRiskValidationConfig(
        validation_mode="stratified_holdout",
        test_size=test_size,
        random_state=random_state,
    )
    return V02ScoreValidationRequest(
        target=target,
        config=config,
        positive_label=positive_label,
        external_predictions=external,
    )


def _column(frame: pd.DataFrame, name: str | None) -> pd.Series:
    if type(name) is not str or name not in frame.columns:
        _task20_error("cli_argument")
    value = frame[name]
    if type(value) is not pd.Series:
        _task20_error("cli_argument")
    return value


def _mask_column(frame: pd.DataFrame, name: str | None) -> pd.Series:
    value = _column(frame, name)
    if value.dtype != np.dtype("bool") or value.isna().any():
        _task20_error("cli_argument")
    return value


def _masked_values(
    series: pd.Series, mask_name: str | None, frame: pd.DataFrame
) -> tuple[object, ...]:
    mask = _mask_column(frame, mask_name)
    values = series.iloc[np.flatnonzero(mask.to_numpy(copy=False))]
    return tuple(values.tolist())


def _decode_label(label_type: str, value: str) -> str | int | bool:
    if label_type == "str":
        return value
    if label_type == "int":
        if _INTEGER_LABEL.fullmatch(value) is None:
            _task20_error("cli_argument")
        try:
            return int(value)
        except ValueError:
            _task20_error("cli_argument")
    if label_type == "bool":
        if value == "true":
            return True
        if value == "false":
            return False
    _task20_error("cli_argument")


def _task20_error(key: str) -> None:
    raise ValueError(f"sharper task20: {key}")


def _emit_failure(message: str, code: int) -> None:
    typer.echo(message or "internal error", err=True)
    raise typer.Exit(code)
