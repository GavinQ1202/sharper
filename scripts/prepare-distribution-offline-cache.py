#!/usr/bin/env python3
"""Prepare and prove a controlled offline dependency cache for distribution tests.

The bootstrap phase is deliberately explicit and may resolve package metadata
from the configured index.  The probe phase is a separate, fresh venv and is
always installed with ``uv pip install --offline`` using only the prepared
cache.  No project environment or ambient uv cache is used as a runtime source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import venv
import zipfile
from pathlib import Path


def _run(
    command: list[str], *, env: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, env=env, text=True, capture_output=True)


def _python(venv_root: Path) -> Path:
    relative = Path("Scripts/python.exe") if os.name == "nt" else Path("bin/python")
    return venv_root / relative


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata(wheel: Path) -> tuple[str, list[str]]:
    with zipfile.ZipFile(wheel) as archive:
        metadata_names = [
            name
            for name in archive.namelist()
            if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_names) != 1:
            raise RuntimeError("wheel must contain exactly one dist-info METADATA")
        lines = archive.read(metadata_names[0]).decode("utf-8").splitlines()
    name = next(line.split(": ", 1)[1] for line in lines if line.startswith("Name: "))
    version = next(
        line.split(": ", 1)[1] for line in lines if line.startswith("Version: ")
    )
    requires = sorted(
        line.split(": ", 1)[1]
        for line in lines
        if line.startswith("Requires-Dist: ")
    )
    return f"{name}=={version}", requires


def _environment(cache_root: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment["UV_CACHE_DIR"] = str(cache_root)
    environment.pop("PYTHONPATH", None)
    return environment


def _assert_inside(path: Path, root: Path, label: str) -> None:
    try:
        path.relative_to(root)
    except ValueError:
        return
    raise RuntimeError(f"{label} must be outside the project root")


def _probe(probe_python: Path, cache_root: Path, wheel: Path) -> dict[str, str]:
    probe = r'''
import json
from pathlib import Path

import matplotlib
import sharper
from sharper.lifecycle_monitoring import (
    EarlyWarningRule,
    LifecycleMonitoringConfig,
    LifecycleMonitoringResult,
    LifecycleState,
    MonitoringCondition,
    WarningScenario,
    monitor_lifecycle,
)

symbols = {
    "MonitoringCondition": MonitoringCondition,
    "EarlyWarningRule": EarlyWarningRule,
    "WarningScenario": WarningScenario,
    "LifecycleState": LifecycleState,
    "LifecycleMonitoringConfig": LifecycleMonitoringConfig,
    "LifecycleMonitoringResult": LifecycleMonitoringResult,
    "monitor_lifecycle": monitor_lifecycle,
}
print(json.dumps({
    "sharper": str(Path(sharper.__file__).resolve()),
    "matplotlib": str(Path(matplotlib.__file__).resolve()),
    "symbols": sorted(symbols),
}))
'''
    completed = _run(
        [str(probe_python), "-c", probe],
        env=_environment(cache_root),
    )
    result = json.loads(completed.stdout)
    probe_root = probe_python.parent.parent.resolve()
    for label in ("sharper", "matplotlib"):
        source = Path(result[label]).resolve()
        try:
            source.relative_to(probe_root)
        except ValueError as exc:
            raise RuntimeError(f"{label} did not originate from probe venv") from exc
    expected = {
        "EarlyWarningRule",
        "LifecycleMonitoringConfig",
        "LifecycleMonitoringResult",
        "LifecycleState",
        "MonitoringCondition",
        "WarningScenario",
        "monitor_lifecycle",
    }
    if set(result["symbols"]) != expected:
        raise RuntimeError(
            "probe symbol smoke did not expose the seven Task 18 symbols"
        )
    return {label: result[label] for label in ("sharper", "matplotlib")}


def prepare(project_root: Path, output_root: Path) -> Path:
    project_root = project_root.resolve()
    output_root = output_root.resolve()
    _assert_inside(output_root, project_root, "output root")
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"output root is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    cache_root = output_root / "uv-cache"
    artifact_root = output_root / "artifacts"
    bootstrap_root = output_root / "bootstrap-venv"
    probe_root = output_root / "probe-venv"
    cache_root.mkdir()
    artifact_root.mkdir()
    environment = _environment(cache_root)
    project_python = project_root / ".venv" / "bin" / "python"
    if not project_python.is_file():
        raise RuntimeError(f"project uv interpreter is missing: {project_python}")
    _run(
        [
            str(project_python),
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(artifact_root),
        ],
        env=environment,
    )
    wheels = sorted(artifact_root.glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError("bootstrap must produce exactly one wheel")
    wheel = wheels[0]
    requirement, requires_dist = _metadata(wheel)
    venv.EnvBuilder(with_pip=False, system_site_packages=False).create(bootstrap_root)
    _run(
        ["uv", "pip", "install", "--python", str(_python(bootstrap_root)), str(wheel)],
        env=environment,
    )
    venv.EnvBuilder(with_pip=False, system_site_packages=False).create(probe_root)
    _run(
        [
            "uv",
            "pip",
            "install",
            "--offline",
            "--python",
            str(_python(probe_root)),
            str(wheel),
        ],
        env=environment,
    )
    origins = _probe(_python(probe_root), cache_root, wheel)
    uv_version = subprocess.run(
        ["uv", "--version"], check=True, text=True, capture_output=True
    ).stdout.strip()
    requires_hash = hashlib.sha256(
        "\n".join(requires_dist).encode("utf-8")
    ).hexdigest()
    manifest = {
        "format_version": 1,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "implementation": platform.python_implementation(),
        },
        "artifact_filename": wheel.name,
        "artifact_sha256": _sha256(wheel),
        "package_requirement": requirement,
        "package_version": requirement.rsplit("==", 1)[1],
        "cache_identifier": "uv-cache",
        "requires_dist": requires_dist,
        "requires_dist_sha256": requires_hash,
        "bootstrap_network_used": True,
        "uv_version": uv_version,
        "pyproject_sha256": _sha256(project_root / "pyproject.toml"),
        "probe_origins": origins,
        "ready": True,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        manifest = prepare(args.project_root, args.output_root)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"bootstrap failed: {exc}", file=sys.stderr)
        return 1
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
