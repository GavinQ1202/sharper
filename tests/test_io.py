"""Tests for the file input boundary."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from sharper import load_csv, load_excel


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


def _openpyxl_available() -> bool:
    return importlib.util.find_spec("openpyxl") is not None


@pytest.fixture
def excel_workbook(tmp_path: Path) -> Path:
    """Create a representative .xlsx workbook when openpyxl is available."""
    if not _openpyxl_available():
        pytest.skip("openpyxl is not installed; install sharper[excel]")

    path = tmp_path / "ordinary.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame({" name ": ["  alpha  ", "beta"], "score": [1, 2]}).to_excel(
            writer,
            index=False,
            sheet_name="Scores",
        )
        pd.DataFrame({"other": [10]}).to_excel(
            writer,
            index=False,
            sheet_name="Other",
        )
    return path


def test_load_excel_accepts_string_path_and_returns_dataframe(
    excel_workbook: Path,
) -> None:
    """String paths read the default sheet and return one DataFrame."""
    result = load_excel(str(excel_workbook))

    assert isinstance(result, pd.DataFrame)
    pd.testing.assert_frame_equal(
        result,
        pd.DataFrame({" name ": ["  alpha  ", "beta"], "score": [1, 2]}),
    )


def test_load_excel_accepts_path_object(excel_workbook: Path) -> None:
    """Path objects are accepted."""
    result = load_excel(excel_workbook)

    assert result["score"].tolist() == [1, 2]


def test_load_excel_selects_sheet_by_index(excel_workbook: Path) -> None:
    """Integer sheet names select a single sheet by index."""
    result = load_excel(excel_workbook, sheet_name=1)

    pd.testing.assert_frame_equal(result, pd.DataFrame({"other": [10]}))


def test_load_excel_selects_sheet_by_name(excel_workbook: Path) -> None:
    """String sheet names select a single sheet by name."""
    result = load_excel(excel_workbook, sheet_name="Other")

    assert result.loc[0, "other"] == 10


def test_load_excel_forwards_supported_options(excel_workbook: Path) -> None:
    """Documented pandas Excel options are forwarded."""
    result = load_excel(
        excel_workbook,
        sheet_name="Scores",
        usecols=["score"],
        dtype={"score": "string"},
        nrows=1,
    )

    assert list(result.columns) == ["score"]
    assert result.loc[0, "score"] == "1"
    assert str(result["score"].dtype) == "string"


@pytest.mark.parametrize("sheet_name", [None, ["Scores"], ("Scores",), {"Scores"}])
def test_load_excel_rejects_invalid_sheet_name(sheet_name: object) -> None:
    """Values that can trigger multi-sheet behavior are rejected."""
    with pytest.raises(ValueError, match="sheet_name must be a string or integer"):
        load_excel("data.xlsx", sheet_name=sheet_name)  # type: ignore[arg-type]


def test_load_excel_rejects_non_xlsx_suffix(tmp_path: Path) -> None:
    """Only .xlsx is in the Task 06 file contract."""
    path = tmp_path / "data.xls"
    path.write_text("not excel", encoding="utf-8")

    with pytest.raises(ValueError, match="only \\.xlsx files are supported in Task 06"):
        load_excel(path)


def test_load_excel_rejects_invalid_path_type() -> None:
    """Non-path inputs are invalid parameters."""
    with pytest.raises(ValueError, match="path must be a string or Path"):
        load_excel(123)  # type: ignore[arg-type]


def test_load_excel_missing_file_raises_os_error(tmp_path: Path) -> None:
    """Missing Excel files produce the stable OSError message."""
    with pytest.raises(OSError, match="Excel file not found"):
        load_excel(tmp_path / "missing.xlsx")


def test_load_excel_directory_raises_os_error(tmp_path: Path) -> None:
    """Directory paths are rejected as file read failures."""
    with pytest.raises(OSError, match="Excel path is a directory"):
        load_excel(tmp_path)


def test_load_excel_rejects_unsupported_read_option(tmp_path: Path) -> None:
    """Options outside the stable Excel contract are rejected clearly."""
    path = tmp_path / "data.xlsx"
    path.write_bytes(b"")

    with pytest.raises(ValueError, match="unsupported Excel read option"):
        load_excel(path, parse_dates=["date"])


def test_load_excel_rejects_engine_read_option(tmp_path: Path) -> None:
    """The Excel engine is fixed to openpyxl."""
    path = tmp_path / "data.xlsx"
    path.write_bytes(b"")

    with pytest.raises(ValueError, match="engine cannot be overridden in Task 06"):
        load_excel(path, engine="calamine")


def test_load_excel_missing_sheet_raises_value_error(excel_workbook: Path) -> None:
    """Missing sheets use the stable Task 06 message."""
    with pytest.raises(ValueError, match="sheet not found"):
        load_excel(excel_workbook, sheet_name="Missing")


def test_load_excel_bad_excel_file_raises_value_error(tmp_path: Path) -> None:
    """Pandas parser failures are wrapped as ValueError."""
    if not _openpyxl_available():
        pytest.skip("openpyxl is not installed; install sharper[excel]")

    path = tmp_path / "bad.xlsx"
    path.write_text("not an excel archive", encoding="utf-8")

    with pytest.raises(ValueError, match="failed to read Excel file") as caught:
        load_excel(path)

    assert caught.value.__cause__ is not None


def test_load_excel_missing_openpyxl_raises_import_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Core installs without the Excel extra produce a clear ImportError."""
    path = tmp_path / "data.xlsx"
    path.write_bytes(b"placeholder")
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)

    with pytest.raises(ImportError, match=r"Install sharper\[excel\]"):
        load_excel(path)


def test_load_excel_does_not_call_analysis_helpers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Excel I/O remains a pure read boundary without analysis side effects."""
    path = tmp_path / "data.xlsx"
    path.write_bytes(b"placeholder")

    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: SimpleNamespace() if name == "openpyxl" else None,
    )

    def fake_read_excel(*args: Any, **kwargs: Any) -> pd.DataFrame:
        assert kwargs["sheet_name"] == 0
        assert kwargs["engine"] == "openpyxl"
        return pd.DataFrame({"raw": [1]})

    monkeypatch.setattr(pd, "read_excel", fake_read_excel)

    result = load_excel(path)

    pd.testing.assert_frame_equal(result, pd.DataFrame({"raw": [1]}))
