"""Local CSV input for Sharper."""

from pathlib import Path
from typing import Any

import pandas as pd

_SUPPORTED_READ_OPTIONS = frozenset({"dtype", "encoding", "sep"})


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
