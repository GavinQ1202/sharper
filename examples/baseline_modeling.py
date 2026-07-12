"""Create a deterministic classification HTML report bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from sharper import generate_analysis_report, run_analysis


def _frame() -> pd.DataFrame:
    """Return a balanced deterministic dataset suitable for a small holdout."""
    rows = list(range(40))
    return pd.DataFrame(
        {
            "age": [22 + index for index in rows],
            "spend": [55.0 + (index % 8) * 7.25 for index in rows],
            "segment": ["north" if index % 2 == 0 else "south" for index in rows],
            "outcome": ["yes" if index % 2 == 0 else "no" for index in rows],
        }
    )


def main() -> None:
    """Write a baseline-model report to a caller-provided directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "baseline-modeling.html"
    run = run_analysis(
        _frame(),
        target="outcome",
        task="classification",
        include_model=True,
        random_state=42,
    )
    if run.training is None or run.evaluation is None:
        raise RuntimeError("baseline workflow did not produce training and evaluation")
    generate_analysis_report(run, report_path, format="html")
    print("baseline modeling completed")


if __name__ == "__main__":
    main()
