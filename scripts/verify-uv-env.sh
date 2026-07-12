#!/usr/bin/env bash
set -euo pipefail

repo_root="$(
  cd "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1
  pwd -P
)"

venv_python="${repo_root}/.venv/bin/python"

if [[ ! -x "${venv_python}" ]]; then
  echo "ERROR: uv project interpreter not found: ${venv_python}" >&2
  exit 1
fi

"${venv_python}" - "${repo_root}" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(sys.argv[1]).resolve()
expected = (repo_root / ".venv" / "bin" / "python").resolve()
actual = Path(sys.executable).resolve()
venv_root = (repo_root / ".venv").resolve()
prefix = Path(sys.prefix).resolve()

if actual != expected:
    raise SystemExit(
        f"ERROR: wrong interpreter: expected {expected}, got {actual}"
    )

if sys.prefix == sys.base_prefix:
    raise SystemExit("ERROR: interpreter is not running inside a virtual environment")

try:
    prefix.relative_to(venv_root)
except ValueError as exc:
    raise SystemExit(
        f"ERROR: sys.prefix is outside project .venv: {prefix}"
    ) from exc

print(f"uv interpreter: {actual}")
print(f"uv prefix: {prefix}")
print(f"python version: {sys.version.split()[0]}")
PY
