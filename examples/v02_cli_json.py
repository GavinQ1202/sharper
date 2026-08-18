"""Run the closed Task 20 policy JSON carrier through the opt-in CLI."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


def main() -> None:
    """Create only synthetic temporary inputs and invoke ``sharper v02-run``."""
    with TemporaryDirectory(prefix="sharper-v02-cli-") as directory:
        root = Path(directory)
        input_path = root / "input.csv"
        policy_path = root / "policy.json"
        output_path = root / "v02-cli.md"
        input_path.write_text("entity,amount\nentity-a,100\nentity-b,250\n")
        policy_path.write_text(
            json.dumps(
                {
                    "schema_version": "task20.policy.v1",
                    "strategy_key": "example-cli-policy",
                    "strategy_version": "v1",
                    "effective_from": "2025-01-01T00:00:00.000000",
                    "expires_at": None,
                    "evaluation_time": "2025-01-02T00:00:00.000000",
                    "rules": [],
                    "default_action_name": "review",
                    "unknown_action_name": "review",
                    "action_role_mapping": [["review", "review"]],
                }
            )
        )
        environment = os.environ.copy()
        environment["MPLCONFIGDIR"] = str(root / "mpl")
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "sharper.cli",
                "v02-run",
                str(input_path),
                "--output",
                str(output_path),
                "--policy-json",
                str(policy_path),
            ],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr or completed.stdout)
        assert completed.stdout == f"Report written to: {output_path}\n"
        assert output_path.is_file()
        print("closed JSON CLI example completed")


if __name__ == "__main__":
    main()
