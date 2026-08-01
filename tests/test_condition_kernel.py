"""Task 16 private condition-kernel contract tests."""

from dataclasses import fields
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from sharper._condition_kernel import (
    _ConditionEvaluation,
    _ConditionNode,
    _ConditionOperand,
    _evaluate_atomic_condition,
    _evaluate_condition,
)


def _operand(kind: str, value: object) -> _ConditionOperand:
    return _ConditionOperand(kind, value)  # type: ignore[arg-type]


def _atomic(operator: str, right: _ConditionOperand | None) -> _ConditionNode:
    return _ConditionNode(
        "atomic", operator, _operand("column", "x"), right, (), None, None, None
    )


def test_private_dataclass_fields_are_frozen() -> None:
    assert [field.name for field in fields(_ConditionOperand)] == ["kind", "value"]
    assert [field.name for field in fields(_ConditionNode)] == [
        "node_type",
        "operator",
        "left",
        "right",
        "children",
        "effective_from",
        "expires_at",
        "version",
    ]
    assert [field.name for field in fields(_ConditionEvaluation)] == [
        "truth",
        "status",
        "reason",
        "root_version",
    ]


@pytest.mark.parametrize(
    ("operator", "literal", "truth"),
    [
        ("eq", 2, ["false", "true", "unknown"]),
        ("ne", 2, ["true", "false", "unknown"]),
        ("lt", 2, ["true", "false", "unknown"]),
        ("le", 2, ["true", "true", "unknown"]),
        ("gt", 2, ["false", "false", "unknown"]),
        ("ge", 2, ["false", "true", "unknown"]),
        ("in", (1, 3), ["true", "false", "unknown"]),
        ("not_in", (1, 3), ["false", "true", "unknown"]),
        ("between", (1, 2), ["true", "true", "unknown"]),
    ],
)
def test_closed_atomic_operators(
    operator: str, literal: object, truth: list[str]
) -> None:
    result = _evaluate_atomic_condition(
        pd.DataFrame({"x": [1, 2, None]}),
        operator=operator,
        left=_operand("column", "x"),
        right=_operand("literal", literal),
        root_version="v1",
    )
    assert result.truth.tolist() == truth
    assert result.truth.index.equals(pd.RangeIndex(3))


def test_missing_operators_and_bool_int_separation() -> None:
    frame = pd.DataFrame({"x": pd.Series([True, None], dtype=object)})
    missing = _evaluate_atomic_condition(
        frame,
        operator="is_missing",
        left=_operand("column", "x"),
        right=None,
        root_version="v1",
    )
    exact = _evaluate_atomic_condition(
        frame,
        operator="eq",
        left=_operand("column", "x"),
        right=_operand("literal", 1),
        root_version="v1",
    )
    assert missing.truth.tolist() == ["false", "true"]
    assert exact.reason.tolist() == ["type_mismatch", "missing_operand"]


def test_boolean_truth_tables() -> None:
    frame = pd.DataFrame({"x": [1, 2, None]})
    true_unknown = _ConditionNode(
        "or",
        None,
        None,
        None,
        (_atomic("eq", _operand("literal", 1)), _atomic("eq", _operand("literal", 2))),
        None,
        None,
        "v1",
    )
    result = _evaluate_condition(frame, true_unknown)
    assert result.truth.tolist() == ["true", "true", "unknown"]
    negated = _ConditionNode("not", None, None, None, (true_unknown,), None, None, "v1")
    with pytest.raises(ValueError, match="child_version_not_allowed"):
        _evaluate_condition(frame, negated)


def test_root_version_effective_window_and_timezone() -> None:
    node = _ConditionNode(
        "atomic",
        "ge",
        _operand("column", "x"),
        _operand("literal", 1),
        (),
        datetime(2025, 1, 1),
        datetime(2025, 2, 1),
        "v1",
    )
    result = _evaluate_condition(
        pd.DataFrame({"x": [1, 2]}), node, evaluation_time=datetime(2025, 1, 1)
    )
    assert result.status.tolist() == ["available", "available"]
    mismatch = _evaluate_condition(
        pd.DataFrame({"x": [1]}),
        node,
        evaluation_time=datetime(2025, 1, 2, tzinfo=timezone.utc),
    )
    assert mismatch.reason.tolist() == ["timezone_mismatch"]


def test_membership_and_string_budgets() -> None:
    frame = pd.DataFrame({"x": [1]})
    for literal, key in (
        ((), "empty_membership_collection"),
        (tuple(range(101)), "membership_budget_exceeded"),
    ):
        with pytest.raises(ValueError, match=key):
            _evaluate_atomic_condition(
                frame,
                operator="in",
                left=_operand("column", "x"),
                right=_operand("literal", literal),
                root_version="v1",
            )
    with pytest.raises(ValueError, match="string_budget_exceeded"):
        _evaluate_atomic_condition(
            pd.DataFrame({"x": ["a"]}),
            operator="eq",
            left=_operand("column", "x"),
            right=_operand("literal", "x" * 1025),
            root_version="v1",
        )


def test_numeric_compatible_membership_and_not_in() -> None:
    literals = (np.int64(1), 2.0)
    frame = pd.DataFrame({"x": [1, 2.0, 3]})
    included = _evaluate_atomic_condition(
        frame,
        operator="in",
        left=_operand("column", "x"),
        right=_operand("literal", literals),
        root_version="v1",
    )
    excluded = _evaluate_atomic_condition(
        frame,
        operator="not_in",
        left=_operand("column", "x"),
        right=_operand("literal", literals),
        root_version="v1",
    )
    assert included.truth.tolist() == ["true", "true", "false"]
    assert excluded.truth.tolist() == ["false", "false", "true"]
    assert literals == (np.int64(1), 2.0)


@pytest.mark.parametrize(
    "literals",
    [(1, 1.0), (True, 1), (1, "1"), (1, np.inf), (1, pd.NA)],
)
def test_invalid_or_duplicate_membership_literals(literals: tuple[object, ...]) -> None:
    with pytest.raises(ValueError, match="invalid_membership|invalid_literal"):
        _evaluate_atomic_condition(
            pd.DataFrame({"x": [1]}),
            operator="in",
            left=_operand("column", "x"),
            right=_operand("literal", literals),
            root_version="v1",
        )


def test_membership_maximum_boundary() -> None:
    result = _evaluate_atomic_condition(
        pd.DataFrame({"x": [99, 100]}),
        operator="in",
        left=_operand("column", "x"),
        right=_operand("literal", tuple(range(100))),
        root_version="v1",
    )
    assert result.truth.tolist() == ["true", "false"]


class _Hostile:
    calls = 0

    def __getattribute__(self, name: str) -> object:
        if name.startswith("__") and name != "__class__":
            type(self).calls += 1
            raise AssertionError(name)
        return object.__getattribute__(self, name)

    def __repr__(self) -> str:
        type(self).calls += 1
        raise AssertionError("repr")


def test_unsupported_object_rejected_without_dunder_dispatch() -> None:
    hostile = _Hostile()
    with pytest.raises(ValueError, match="unsupported_scalar_type"):
        _evaluate_atomic_condition(
            pd.DataFrame({"x": [hostile]}),
            operator="eq",
            left=_operand("column", "x"),
            right=_operand("literal", np.int64(1)),
            root_version="v1",
        )
    assert _Hostile.calls == 0
