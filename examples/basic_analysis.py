"""Create a deterministic, analysis-only Markdown report bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from sharper import generate_analysis_report, run_analysis


def _frame() -> pd.DataFrame:
    """Return a small deterministic classification dataset."""
    rows = list(range(24))
    return pd.DataFrame(
        {
            "age": [24 + index for index in rows],
            "spend": [40.0 + (index % 6) * 9.5 for index in rows],
            "segment": ["north" if index % 2 == 0 else "south" for index in rows],
            "outcome": ["yes" if index % 2 == 0 else "no" for index in rows],
        }
    )


def main() -> None:
    """Write an analysis-only report to a caller-provided directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "basic-analysis.md"
    run = run_analysis(
        _frame(),
        target="outcome",
        task="classification",
        include_model=False,
        random_state=42,
    )
    if (
        run.target_analysis is None
        or run.training is not None
        or run.evaluation is not None
    ):
        raise RuntimeError(
            "analysis-only workflow did not preserve the expected results"
        )
    generate_analysis_report(run, report_path, format="markdown")
    print("basic analysis completed")


if __name__ == "__main__":
    main()
