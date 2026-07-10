"""Tests for the CSV input boundary."""

from pathlib import Path

import pandas as pd
import pytest

from sharper import load_csv


@pytest.fixture
def ordinary_csv(tmp_path: Path) -> Path:
    """Create a representative UTF-8 CSV."""
    path = tmp_path / "ordinary.csv"
    path.write_text("name,score\n甲,1\n乙,2\n", encoding="utf-8")
    return path


def test_load_csv_accepts_string_path_and_returns_dataframe(
    ordinary_csv: Path,
) -> None:
    """String paths produce an unmodified DataFrame."""
    result = load_csv(str(ordinary_csv))

    assert isinstance(result, pd.DataFrame)
    pd.testing.assert_frame_equal(
        result,
        pd.DataFrame({"name": ["甲", "乙"], "score": [1, 2]}),
    )


def test_load_csv_accepts_path_object(ordinary_csv: Path) -> None:
    """Path objects are accepted."""
    result = load_csv(ordinary_csv)

    assert result["name"].tolist() == ["甲", "乙"]


def test_load_csv_supports_separator_and_dtype(tmp_path: Path) -> None:
    """Documented pandas options are forwarded."""
    path = tmp_path / "semicolon.csv"
    path.write_text("code;value\n001;2\n", encoding="utf-8")

    result = load_csv(path, sep=";", dtype={"code": "string"})

    assert result.loc[0, "code"] == "001"
    assert str(result["code"].dtype) == "string"


def test_load_csv_preserves_missing_values(tmp_path: Path) -> None:
    """Pandas default missing-value parsing remains intact."""
    path = tmp_path / "missing.csv"
    path.write_text("name,value\nalpha,1\nbeta,\n", encoding="utf-8")

    result = load_csv(path)

    assert pd.isna(result.loc[1, "value"])


def test_load_csv_does_not_clean_column_names_or_values(tmp_path: Path) -> None:
    """Whitespace in names and values is not implicitly cleaned."""
    path = tmp_path / "spaces.csv"
    path.write_text(" name ,value\n  alpha  ,1\n", encoding="utf-8")

    result = load_csv(path)

    assert list(result.columns) == [" name ", "value"]
    assert result.loc[0, " name "] == "  alpha  "


def test_load_csv_missing_file_preserves_os_error_cause(tmp_path: Path) -> None:
    """Missing files produce an actionable OSError with its cause."""
    path = tmp_path / "missing.csv"

    with pytest.raises(OSError, match="Could not read CSV file") as caught:
        load_csv(path)

    assert isinstance(caught.value.__cause__, FileNotFoundError)


def test_load_csv_directory_preserves_os_error_cause(tmp_path: Path) -> None:
    """Directory paths are rejected as file read failures."""
    with pytest.raises(OSError, match="Could not read CSV file") as caught:
        load_csv(tmp_path)

    assert isinstance(caught.value.__cause__, IsADirectoryError)


def test_load_csv_empty_file_preserves_parse_error_cause(tmp_path: Path) -> None:
    """Empty files produce an actionable ValueError."""
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="Could not parse CSV file") as caught:
        load_csv(path)

    assert isinstance(caught.value.__cause__, pd.errors.EmptyDataError)


def test_load_csv_bad_format_preserves_parser_error_cause(tmp_path: Path) -> None:
    """Malformed row widths produce an actionable ValueError."""
    path = tmp_path / "bad.csv"
    path.write_text("a,b\n1,2\n3,4,5\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Could not parse CSV file") as caught:
        load_csv(path)

    assert isinstance(caught.value.__cause__, pd.errors.ParserError)


def test_load_csv_rejects_unsupported_options(ordinary_csv: Path) -> None:
    """Options outside the stable contract are rejected clearly."""
    with pytest.raises(ValueError, match="Unsupported CSV read option.*header"):
        load_csv(ordinary_csv, header=None)


def test_load_csv_rejects_invalid_path_type() -> None:
    """Non-path inputs are invalid parameters."""
    with pytest.raises(ValueError, match="path must be"):
        load_csv(123)  # type: ignore[arg-type]
