"""Private, closed condition evaluation kernel for Task 16."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from math import isfinite
from typing import Literal

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_complex_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
)

_OPERATORS = frozenset(
    {
        "eq",
        "ne",
        "lt",
        "le",
        "gt",
        "ge",
        "in",
        "not_in",
        "between",
        "is_missing",
        "is_not_missing",
    }
)
_NUMPY_TYPES = (
    np.bool_,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.float16,
    np.float32,
    np.float64,
)
_MAX_DEPTH = 8
_MAX_NODES = 128
_MAX_MEMBERSHIP = 100
_MAX_STRING = 1024


@dataclass(frozen=True)
class _ConditionOperand:
    kind: Literal["column", "literal"]
    value: object


@dataclass(frozen=True)
class _ConditionNode:
    node_type: Literal["atomic", "and", "or", "not"]
    operator: str | None
    left: _ConditionOperand | None
    right: _ConditionOperand | None
    children: tuple[_ConditionNode, ...]
    effective_from: datetime | None
    expires_at: datetime | None
    version: str | None


@dataclass(frozen=True)
class _ConditionEvaluation:
    truth: pd.Series
    status: pd.Series
    reason: pd.Series
    root_version: str


def _spec_error(key: str) -> ValueError:
    return ValueError(f"condition specification is invalid: {key}")


def _eval_error(key: str) -> ValueError:
    return ValueError(f"condition evaluation is invalid: {key}")


def _is_allowed_scalar(value: object) -> bool:
    value_type = type(value)
    return (
        value is pd.NA
        or value is pd.NaT
        or value_type
        in (type(None), bool, int, float, str, date, datetime, pd.Timestamp)
        or value_type in _NUMPY_TYPES
    )


def _normalize(value: object, *, specification: bool) -> object:
    if not _is_allowed_scalar(value):
        raise (
            _spec_error("unsupported_scalar_type")
            if specification
            else _eval_error("unsupported_scalar_type")
        )
    if type(value) in _NUMPY_TYPES:
        return value.item()  # exact approved NumPy scalars only
    return value


def _missing(value: object) -> bool:
    value = _normalize(value, specification=False)
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    return type(value) is float and np.isnan(value)


def _family(value: object) -> str:
    value = _normalize(value, specification=False)
    if (
        value is None
        or value is pd.NA
        or value is pd.NaT
        or (type(value) is float and np.isnan(value))
    ):
        return "missing"
    if type(value) is bool:
        return "bool"
    if type(value) is int:
        return "int"
    if type(value) is float:
        return "float"
    if type(value) is str:
        return "str"
    if type(value) is date:
        return "date"
    if type(value) in (datetime, pd.Timestamp):
        return "datetime"
    raise _eval_error("unsupported_scalar_type")


def _aware(value: datetime | pd.Timestamp) -> bool:
    try:
        return value.tzinfo is not None and value.utcoffset() is not None
    except (TypeError, ValueError):
        return False


def _literal(value: object) -> object:
    normalized = _normalize(value, specification=True)
    if type(normalized) is str and len(normalized) > _MAX_STRING:
        raise _spec_error("string_budget_exceeded")
    if _family(normalized) == "missing":
        raise _spec_error("invalid_literal")
    if type(normalized) is float and not isfinite(normalized):
        raise _spec_error("invalid_literal")
    return normalized


def _literal_tuple(value: object, *, between: bool = False) -> tuple[object, ...]:
    if type(value) is not tuple:
        raise _spec_error("invalid_between" if between else "invalid_membership")
    if not value:
        raise _spec_error("empty_membership_collection")
    if between and len(value) != 2:
        raise _spec_error("invalid_between")
    if len(value) > _MAX_MEMBERSHIP:
        raise _spec_error("membership_budget_exceeded")
    values = tuple(_literal(item) for item in value)
    accepted: list[object] = []
    for item in values:
        if any(_compare(item, prior, "eq")[0] == "true" for prior in accepted):
            raise _spec_error("invalid_membership")
        accepted.append(item)
    families = {_family(item) for item in values}
    compatible = len(families) == 1 or families <= {"int", "float"}
    if not compatible:
        raise _spec_error("invalid_membership" if not between else "invalid_between")
    if between:
        truth, _, _ = _compare(values[0], values[1], "le")
        if truth != "true":
            raise _spec_error("invalid_between")
    return values


def _validate_frame(data: pd.DataFrame, columns: tuple[str, ...]) -> None:
    if type(data) is not pd.DataFrame:
        raise _eval_error("unsupported_dtype")
    for column in columns:
        series = data[column]
        if is_complex_dtype(series.dtype):
            raise _eval_error("unsupported_dtype")
        if not (
            is_bool_dtype(series.dtype)
            or is_numeric_dtype(series.dtype)
            or is_datetime64_any_dtype(series.dtype)
            or series.dtype == object
            or isinstance(series.dtype, pd.CategoricalDtype)
            or isinstance(series.dtype, pd.StringDtype)
        ):
            raise _eval_error("unsupported_dtype")
        if series.dtype == object or isinstance(
            series.dtype, (pd.CategoricalDtype, pd.StringDtype)
        ):
            if isinstance(series.dtype, pd.CategoricalDtype):
                for category in series.cat.categories.tolist():
                    _normalize(category, specification=False)
            for position in range(len(series)):
                _normalize(series.iat[position], specification=False)


def _compare(left: object, right: object, operator: str) -> tuple[str, str, str]:
    left = _normalize(left, specification=False)
    right = _normalize(right, specification=False)
    if _missing(left) or _missing(right):
        return "unknown", "not_verifiable", "missing_operand"
    left_family, right_family = _family(left), _family(right)
    numeric = left_family in {"int", "float"} and right_family in {"int", "float"}
    compatible = left_family == right_family or numeric
    if not compatible:
        return "unknown", "not_verifiable", "type_mismatch"
    if numeric and (
        (type(left) is float and not isfinite(left))
        or (type(right) is float and not isfinite(right))
    ):
        return "unknown", "not_verifiable", "nonfinite_operand"
    if left_family == "datetime" and _aware(left) != _aware(right):
        return "unknown", "not_verifiable", "timezone_mismatch"
    if left_family == "datetime" and _aware(left):
        left = left.astimezone(timezone.utc)
        right = right.astimezone(timezone.utc)
    if operator == "eq":
        result = left == right
    elif operator == "ne":
        result = left != right
    elif operator == "lt":
        result = left < right
    elif operator == "le":
        result = left <= right
    elif operator == "gt":
        result = left > right
    else:
        result = left >= right
    return ("true" if result else "false"), "available", "computed"


def _validate_version(version: object) -> str:
    if (
        type(version) is not str
        or not version
        or len(version) > 64
        or not version.isascii()
        or any(not (char.isalnum() or char in "._-") for char in version)
    ):
        raise _spec_error("invalid_version")
    return version


def _operand_columns(operand: _ConditionOperand | None) -> tuple[str, ...]:
    if operand is not None and operand.kind == "column":
        return (operand.value,)  # validated exact str
    return ()


def _validate_atomic(
    data: pd.DataFrame,
    operator: str,
    left: _ConditionOperand,
    right: _ConditionOperand | None,
) -> tuple[str, ...]:
    if type(operator) is not str or operator not in _OPERATORS:
        raise _spec_error("invalid_operator")
    if (
        type(left) is not _ConditionOperand
        or left.kind != "column"
        or type(left.value) is not str
        or not left.value
    ):
        raise _spec_error("invalid_operand")
    if left.value not in data.columns:
        raise _spec_error("unknown_column")
    columns = [left.value]
    if operator in {"is_missing", "is_not_missing"}:
        if right is not None:
            raise _spec_error("invalid_operand")
        return tuple(columns)
    if type(right) is not _ConditionOperand or right.kind not in {"column", "literal"}:
        raise _spec_error("invalid_operand")
    if right.kind == "column":
        if (
            operator in {"in", "not_in", "between"}
            or type(right.value) is not str
            or not right.value
        ):
            raise _spec_error("invalid_operand")
        if right.value not in data.columns:
            raise _spec_error("unknown_column")
        columns.append(right.value)
    elif operator in {"in", "not_in"}:
        _literal_tuple(right.value)
    elif operator == "between":
        _literal_tuple(right.value, between=True)
    else:
        _literal(right.value)
    return tuple(columns)


def _evaluate_atomic_condition(
    data: pd.DataFrame,
    *,
    operator: str,
    left: _ConditionOperand,
    right: _ConditionOperand | None,
    root_version: str,
) -> _ConditionEvaluation:
    """Evaluate one validated closed atomic condition by row position."""
    version = _validate_version(root_version)
    columns = _validate_atomic(data, operator, left, right)
    _validate_frame(data, columns)
    right_value = None
    if right is not None and right.kind == "literal":
        right_value = (
            _literal_tuple(right.value, between=operator == "between")
            if operator in {"in", "not_in", "between"}
            else _literal(right.value)
        )
    truths: list[str] = []
    statuses: list[str] = []
    reasons: list[str] = []
    for position in range(len(data)):
        left_value = data[left.value].iat[position]
        if operator in {"is_missing", "is_not_missing"}:
            is_missing = _missing(left_value)
            truth, status, reason = (
                ("true" if is_missing == (operator == "is_missing") else "false"),
                "available",
                "computed",
            )
        elif operator in {"in", "not_in"}:
            if _missing(left_value):
                truth, status, reason = "unknown", "not_verifiable", "missing_operand"
            else:
                matches = [_compare(left_value, item, "eq") for item in right_value]
                if any(item[0] == "true" for item in matches):
                    truth, status, reason = "true", "available", "computed"
                elif any(item[0] == "unknown" for item in matches):
                    first = next(item for item in matches if item[0] == "unknown")
                    truth, status, reason = first
                else:
                    truth, status, reason = "false", "available", "computed"
                if operator == "not_in" and truth != "unknown":
                    truth = "false" if truth == "true" else "true"
        elif operator == "between":
            low = _compare(left_value, right_value[0], "ge")
            high = _compare(left_value, right_value[1], "le")
            truth, status, reason = _combine_pair(low, high, "and")
        else:
            rv = (
                data[right.value].iat[position]
                if right.kind == "column"
                else right_value
            )
            truth, status, reason = _compare(left_value, rv, operator)
        truths.append(truth)
        statuses.append(status)
        reasons.append(reason)
    index = pd.RangeIndex(len(data))
    return _ConditionEvaluation(
        pd.Series(truths, index=index, dtype="string"),
        pd.Series(statuses, index=index, dtype="string"),
        pd.Series(reasons, index=index, dtype="string"),
        version,
    )


def _combine_pair(
    left: tuple[str, str, str], right: tuple[str, str, str], kind: str
) -> tuple[str, str, str]:
    lt, _, lr = left
    rt, _, rr = right
    if kind == "and":
        truth = (
            "false"
            if "false" in (lt, rt)
            else ("unknown" if "unknown" in (lt, rt) else "true")
        )
    else:
        truth = (
            "true"
            if "true" in (lt, rt)
            else ("unknown" if "unknown" in (lt, rt) else "false")
        )
    if truth == "unknown":
        return truth, "not_verifiable", lr if lt == "unknown" else rr
    return truth, "available", "computed"


def _evaluate_condition(
    data: pd.DataFrame,
    condition: _ConditionNode,
    *,
    evaluation_time: str | datetime | None = None,
) -> _ConditionEvaluation:
    """Validate and evaluate a closed Boolean condition tree."""
    if type(data) is not pd.DataFrame:
        raise _eval_error("unsupported_dtype")
    nodes: list[_ConditionNode] = []
    referenced: list[str] = []
    active: set[int] = set()

    def visit(node: object, depth: int, root: bool) -> None:
        if type(node) is not _ConditionNode:
            raise _spec_error("invalid_node_type")
        if id(node) in active:
            raise _spec_error("condition_cycle")
        if depth > _MAX_DEPTH:
            raise _spec_error("condition_depth_exceeded")
        nodes.append(node)
        if len(nodes) > _MAX_NODES:
            raise _spec_error("condition_nodes_exceeded")
        active.add(id(node))
        if root:
            _validate_version(node.version)
        elif node.version is not None:
            raise _spec_error("child_version_not_allowed")
        if not root and (
            node.effective_from is not None or node.expires_at is not None
        ):
            raise _spec_error("descendant_effective_window")
        if node.node_type == "atomic":
            if node.children or node.operator is None or node.left is None:
                raise _spec_error("invalid_children")
            referenced.extend(
                _validate_atomic(data, node.operator, node.left, node.right)
            )
        elif node.node_type in {"and", "or", "not"}:
            if (
                node.operator is not None
                or node.left is not None
                or node.right is not None
                or type(node.children) is not tuple
            ):
                raise _spec_error("invalid_children")
            if (node.node_type in {"and", "or"} and len(node.children) < 2) or (
                node.node_type == "not" and len(node.children) != 1
            ):
                raise _spec_error("invalid_children")
            for child in node.children:
                visit(child, depth + 1, False)
        else:
            raise _spec_error("invalid_node_type")
        active.remove(id(node))

    visit(condition, 1, True)
    _validate_effective_window(condition.effective_from, condition.expires_at)
    if evaluation_time is not None and type(evaluation_time) not in (str, datetime):
        raise _spec_error("evaluation_time_invalid")
    if type(evaluation_time) is str and evaluation_time not in data.columns:
        raise _spec_error("unknown_column")
    if type(evaluation_time) is str:
        referenced.append(evaluation_time)
    _validate_frame(data, tuple(dict.fromkeys(referenced)))

    def evaluate(node: _ConditionNode) -> _ConditionEvaluation:
        if node.node_type == "atomic":
            return _evaluate_atomic_condition(
                data,
                operator=node.operator,
                left=node.left,
                right=node.right,
                root_version=condition.version,
            )
        children = [evaluate(child) for child in node.children]
        truth: list[str] = []
        status: list[str] = []
        reason: list[str] = []
        for pos in range(len(data)):
            values = [
                (child.truth.iat[pos], child.status.iat[pos], child.reason.iat[pos])
                for child in children
            ]
            if node.node_type == "not":
                item = values[0]
                t = (
                    "unknown"
                    if item[0] == "unknown"
                    else ("false" if item[0] == "true" else "true")
                )
                result = (
                    t,
                    item[1] if t == "unknown" else "available",
                    item[2] if t == "unknown" else "computed",
                )
            else:
                result = values[0]
                for item in values[1:]:
                    result = _combine_pair(result, item, node.node_type)
            truth.append(result[0])
            status.append(result[1])
            reason.append(result[2])
        idx = pd.RangeIndex(len(data))
        return _ConditionEvaluation(
            pd.Series(truth, index=idx, dtype="string"),
            pd.Series(status, index=idx, dtype="string"),
            pd.Series(reason, index=idx, dtype="string"),
            condition.version,
        )

    result = evaluate(condition)
    if condition.effective_from is None and condition.expires_at is None:
        return result
    times = (
        [evaluation_time] * len(data)
        if type(evaluation_time) is datetime
        else (
            [data[evaluation_time].iat[i] for i in range(len(data))]
            if type(evaluation_time) is str
            else [None] * len(data)
        )
    )
    truth = result.truth.copy()
    status = result.status.copy()
    reason = result.reason.copy()
    for pos, raw in enumerate(times):
        if raw is None or raw is pd.NaT:
            truth.iat[pos], status.iat[pos], reason.iat[pos] = (
                "unknown",
                "not_verifiable",
                "missing_evaluation_time",
            )
            continue
        raw = _normalize(raw, specification=False)
        if _family(raw) != "datetime":
            truth.iat[pos], status.iat[pos], reason.iat[pos] = (
                "unknown",
                "not_verifiable",
                "type_mismatch",
            )
            continue
        bound = condition.effective_from or condition.expires_at
        if _aware(raw) != _aware(bound):
            truth.iat[pos], status.iat[pos], reason.iat[pos] = (
                "unknown",
                "not_verifiable",
                "timezone_mismatch",
            )
            continue
        before = (
            condition.effective_from is not None
            and _compare(raw, condition.effective_from, "lt")[0] == "true"
        )
        expired = (
            condition.expires_at is not None
            and _compare(raw, condition.expires_at, "ge")[0] == "true"
        )
        if before or expired:
            truth.iat[pos], status.iat[pos], reason.iat[pos] = (
                "unknown",
                "inactive",
                "outside_effective_window",
            )
    return _ConditionEvaluation(truth, status, reason, condition.version)


def _validate_effective_window(start: datetime | None, end: datetime | None) -> None:
    for value in (start, end):
        if value is not None and type(value) is not datetime:
            raise _spec_error("invalid_effective_window")
    if start is not None and end is not None:
        if _aware(start) != _aware(end) or _compare(start, end, "lt")[0] != "true":
            raise _spec_error("invalid_effective_window")
