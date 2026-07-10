"""Local file input for Sharper."""

import importlib.util
from pathlib import Path
from typing import Any

import pandas as pd

_SUPPORTED_READ_OPTIONS = frozenset({"dtype", "encoding", "sep"})
_SUPPORTED_EXCEL_READ_OPTIONS = frozenset(
    {
        "dtype",
        "header",
        "keep_default_na",
        "na_values",
        "names",
        "nrows",
        "skiprows",
        "usecols",
    }
)


def load_csv(path: str | Path, **read_options: Any) -> pd.DataFrame:
    """Read a local CSV file into a pandas DataFrame.

    Parameters
    ----------
    path
        Local CSV path as a string or :class:`pathlib.Path`.
    **read_options
        Supported pandas CSV options are ``encoding``, ``sep``, and ``dtype``.

    Returns
    -------
    pandas.DataFrame
        The values and column names parsed by pandas without additional cleaning,
        schema inference, or type coercion.

    Raises
    ------
    ValueError
        If ``path`` is not a string or ``Path``, an unsupported option is supplied,
        or pandas cannot parse the CSV (including an empty file).
    OSError
        If the local path cannot be read, for example because it does not exist or
        refers to a directory.

    Notes
    -----
    Missing values follow pandas defaults unless affected by a supported ``dtype``
    option. The function only reads the file and does not modify other external
    state.

    Examples
    --------
    >>> from sharper import load_csv
    >>> frame = load_csv("data.csv", encoding="utf-8", sep=",")
    """
    if not isinstance(path, (str, Path)):
        raise ValueError("path must be a string or pathlib.Path")

    unsupported = sorted(set(read_options) - _SUPPORTED_READ_OPTIONS)
    if unsupported:
        names = ", ".join(repr(name) for name in unsupported)
        supported = ", ".join(sorted(_SUPPORTED_READ_OPTIONS))
        raise ValueError(
            f"Unsupported CSV read option(s): {names}. Supported options: {supported}."
        )

    try:
        return pd.read_csv(path, **read_options)
    except OSError as error:
        raise OSError(f"Could not read CSV file {str(path)!r}: {error}") from error
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeError) as error:
        raise ValueError(f"Could not parse CSV file {str(path)!r}: {error}") from error


def load_excel(
    path: str | Path,
    *,
    sheet_name: str | int = 0,
    **read_options: Any,
) -> pd.DataFrame:
    """Read one sheet from a local ``.xlsx`` file into a pandas DataFrame.

    Parameters
    ----------
    path
        Local Excel path as a string or :class:`pathlib.Path`. Task 06 supports
        only files whose suffix is ``.xlsx``.
    sheet_name
        Sheet name or zero-based sheet index to read. ``None`` and collection
        values are rejected so this function always returns one DataFrame.
    **read_options
        Supported pandas Excel options are ``header``, ``names``, ``usecols``,
        ``dtype``, ``na_values``, ``keep_default_na``, ``skiprows``, and
        ``nrows``. The Excel engine is fixed to ``openpyxl`` and cannot be
        overridden.

    Returns
    -------
    pandas.DataFrame
        The sheet values and column names parsed by pandas without additional
        cleaning, schema inference, summary, quality checks, or type coercion.

    Raises
    ------
    ImportError
        If ``openpyxl`` is not installed. Install ``sharper[excel]`` to enable
        Excel reading.
    ValueError
        If ``path``, ``sheet_name``, the file suffix, read options, sheet
        selection, or pandas parsing are invalid.
    OSError
        If the local path is missing or refers to a directory.

    Notes
    -----
    Missing values follow pandas defaults unless affected by supported read
    options. The function only reads the file and does not modify files, clean
    data, infer schema, summarize data, or run quality checks.

    Examples
    --------
    >>> from sharper import load_excel
    >>> frame = load_excel("data.xlsx", sheet_name=0)
    """
    if not isinstance(path, (str, Path)):
        raise ValueError("path must be a string or Path")
    if not isinstance(sheet_name, (str, int)) or isinstance(sheet_name, bool):
        raise ValueError("sheet_name must be a string or integer")

    path_obj = Path(path)
    if path_obj.exists() and path_obj.is_dir():
        raise OSError(f"Excel path is a directory: {str(path_obj)!r}")
    if path_obj.suffix.lower() != ".xlsx":
        raise ValueError("only .xlsx files are supported in Task 06")
    if not path_obj.exists():
        raise OSError(f"Excel file not found: {str(path_obj)!r}")

    if "engine" in read_options:
        raise ValueError("engine cannot be overridden in Task 06")

    unsupported = sorted(set(read_options) - _SUPPORTED_EXCEL_READ_OPTIONS)
    if unsupported:
        names = ", ".join(repr(name) for name in unsupported)
        supported = ", ".join(sorted(_SUPPORTED_EXCEL_READ_OPTIONS))
        raise ValueError(
            f"unsupported Excel read option(s): {names}. Supported options: "
            f"{supported}."
        )

    if importlib.util.find_spec("openpyxl") is None:
        raise ImportError("Install sharper[excel] to read Excel files")

    try:
        result = pd.read_excel(
            path_obj,
            sheet_name=sheet_name,
            engine="openpyxl",
            **read_options,
        )
    except ValueError as error:
        message = str(error).lower()
        if "worksheet" in message or "sheet" in message:
            raise ValueError(f"sheet not found: {error}") from error
        raise ValueError(f"failed to read Excel file: {error}") from error
    except ImportError as error:
        raise ImportError("Install sharper[excel] to read Excel files") from error
    except Exception as error:
        raise ValueError(f"failed to read Excel file: {error}") from error

    if not isinstance(result, pd.DataFrame):
        raise ValueError("sheet_name must be a string or integer")
    return result
