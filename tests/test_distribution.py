"""Release-readiness distribution tests for built Sharper artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import zipfile
from email import policy
from email.parser import BytesParser
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_VENV = (PROJECT_ROOT / ".venv").resolve()
PROJECT_PYTHON = PROJECT_VENV / (
    "Scripts/python.exe" if os.name == "nt" else "bin/python"
)
VERSION = "0.2.0"
OFFLINE_CACHE_ENV = "SHARPER_DISTRIBUTION_OFFLINE_CACHE_ROOT"
OFFLINE_CACHE_FORMAT = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepared_cache() -> tuple[Path, dict[str, object]]:
    configured = os.environ.get(OFFLINE_CACHE_ENV)
    if not configured:
        raise AssertionError(
            "prepared offline distribution cache required; run "
            "scripts/prepare-distribution-offline-cache.py"
        )
    root = Path(configured).expanduser().absolute().resolve()
    if PROJECT_ROOT in root.parents or root == PROJECT_ROOT:
        raise AssertionError("prepared offline distribution cache must be outside repo")
    if root == Path.home() / ".cache" / "uv":
        raise AssertionError(
            "prepared offline distribution cache cannot be global uv cache"
        )
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise AssertionError("prepared offline cache manifest is missing")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AssertionError("prepared offline cache manifest is invalid") from exc
    if (
        manifest.get("ready") is not True
        or manifest.get("format_version") != OFFLINE_CACHE_FORMAT
        or manifest.get("package_version") != VERSION
        or manifest.get("cache_identifier") != "uv-cache"
    ):
        raise AssertionError("prepared offline cache is stale/incompatible")
    expected_platform = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "implementation": platform.python_implementation(),
    }
    if manifest.get("python") != f"{sys.version_info.major}.{sys.version_info.minor}":
        raise AssertionError("prepared offline cache is stale/incompatible")
    if manifest.get("platform") != expected_platform:
        raise AssertionError("prepared offline cache is stale/incompatible")
    cache = root / str(manifest["cache_identifier"])
    if not cache.is_dir() or cache == Path.home() / ".cache" / "uv":
        raise AssertionError("prepared offline cache directory is invalid")
    artifact = root / "artifacts" / str(manifest.get("artifact_filename", ""))
    if not artifact.is_file() or _sha256(artifact) != manifest.get("artifact_sha256"):
        raise AssertionError("prepared offline cache artifact is stale/incompatible")
    pyproject_hash = _sha256(PROJECT_ROOT / "pyproject.toml")
    if manifest.get("pyproject_sha256") != pyproject_hash:
        raise AssertionError("prepared offline cache is stale/incompatible")
    return cache, manifest


def _build_python(configured: str | None = None) -> Path:
    """Return only the verified project-UV Python used for artifact builds."""
    configured = (
        configured if configured is not None else os.environ.get("SHARPER_BUILD_PYTHON")
    )
    python = (
        Path(os.path.normpath(str(Path(configured).expanduser().absolute())))
        if configured
        else PROJECT_PYTHON
    )
    if (
        python != PROJECT_PYTHON
        or python.resolve() != PROJECT_PYTHON.resolve()
        or not python.is_file()
    ):
        raise AssertionError(
            "SHARPER_BUILD_PYTHON must equal the project .venv/bin/python"
        )
    probe = subprocess.run(
        [
            str(python),
            "-c",
            "import build, hatchling, json; "
            "print(json.dumps([build.__file__, hatchling.__file__]))",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if probe.returncode:
        raise AssertionError(
            "project .venv must provide local build and hatchling for offline builds"
        )
    for module_path in json.loads(probe.stdout):
        try:
            Path(module_path).resolve().relative_to(PROJECT_VENV)
        except ValueError as exc:
            raise AssertionError(
                "build and hatchling must be imported from the project .venv"
            ) from exc
    return python


def _venv_paths(venv_dir: Path) -> tuple[Path, Path]:
    if os.name == "nt":
        scripts = venv_dir / "Scripts"
        return scripts / "python.exe", scripts / "sharper.exe"
    scripts = venv_dir / "bin"
    return scripts / "python", scripts / "sharper"


def _environment(
    venv_dir: Path,
    *,
    uv_cache_dir: Path | None = None,
) -> dict[str, str]:
    python, _ = _venv_paths(venv_dir)
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PATH"] = f"{python.parent}{os.pathsep}{os.defpath}"
    environment["NO_COLOR"] = "1"
    environment["TERM"] = "dumb"
    if uv_cache_dir is not None:
        environment["UV_CACHE_DIR"] = str(uv_cache_dir)
    return environment


def _runtime_environment(venv_dir: Path, *, uv_cache_dir: Path) -> dict[str, str]:
    """Build an isolated runtime environment for a prepared cache install."""
    python, _ = _venv_paths(venv_dir)
    matplotlib_config = venv_dir / ".matplotlib"
    matplotlib_config.mkdir(exist_ok=True)
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PATH"] = f"{python.parent}{os.pathsep}{os.defpath}"
    environment["UV_CACHE_DIR"] = str(uv_cache_dir)
    environment["MPLCONFIGDIR"] = str(matplotlib_config)
    environment["NO_COLOR"] = "1"
    environment["TERM"] = "dumb"
    return environment


def _uv() -> Path:
    """Locate uv, which supplies only offline temporary-environment tooling."""
    executable = shutil.which("uv")
    if executable is None:
        raise AssertionError("uv is required for offline clean-install verification")
    return Path(executable)


def _run(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    expect_empty_stderr: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    if expect_empty_stderr:
        assert result.stderr == ""
    assert "Traceback" not in result.stdout
    assert "Traceback" not in result.stderr
    return result


def _metadata(payload: bytes) -> object:
    return BytesParser(policy=policy.default).parsebytes(payload)


def _requires_dist(message: object) -> list[str]:
    return sorted(message.get_all("Requires-Dist") or [])  # type: ignore[union-attr]


def _requires_dist_hash(requirements: list[str]) -> str:
    return hashlib.sha256("\n".join(requirements).encode("utf-8")).hexdigest()


def _assert_metadata(message: object) -> None:
    assert message["Name"] == "sharper"  # type: ignore[index]
    assert message["Version"] == VERSION  # type: ignore[index]
    assert message["License-Expression"] == "MIT"  # type: ignore[index]
    assert message["Requires-Python"] == ">=3.10"  # type: ignore[index]
    assert message["Description-Content-Type"] == "text/markdown"  # type: ignore[index]
    requirements = _requires_dist(message)
    runtime = {value for value in requirements if "extra ==" not in value}
    assert runtime == {
        "matplotlib",
        "numpy",
        "pandas",
        "scikit-learn",
        "seaborn",
        "scipy",
        "typer",
    }
    assert any(value.startswith("openpyxl; extra == 'excel'") for value in requirements)
    assert not any(value.startswith("openpyxl") for value in runtime)


def _venv_snapshot(
    python: Path, *, cwd: Path, environment: dict[str, str]
) -> dict[str, object]:
    result = _run(
        [
            str(python),
            "-c",
            "import json, site, sys; "
            "print(json.dumps({'prefix': sys.prefix, 'base_prefix': sys.base_prefix, "
            "'executable': sys.executable, 'path': sys.path, "
            "'site_packages': site.getsitepackages()}))",
        ],
        cwd=cwd,
        environment=environment,
    )
    return json.loads(result.stdout)


def _assert_clean_venv(venv_dir: Path, *, cwd: Path) -> tuple[Path, Path]:
    """Assert artifact isolation from project and system site-packages."""
    python, console = _venv_paths(venv_dir)
    assert python.exists()
    assert console.parent == python.parent
    assert venv_dir.absolute() not in {PROJECT_ROOT.absolute(), PROJECT_VENV}

    config = (venv_dir / "pyvenv.cfg").read_text(encoding="utf-8").lower()
    assert "include-system-site-packages = false" in config

    snapshot = _venv_snapshot(python, cwd=cwd, environment=_environment(venv_dir))
    assert snapshot["prefix"] != snapshot["base_prefix"]
    assert Path(str(snapshot["prefix"])).resolve() == venv_dir.resolve()
    assert Path(str(snapshot["executable"])).absolute() == python.absolute()

    site_packages = [Path(path).resolve() for path in snapshot["site_packages"]]
    assert len(site_packages) == 1
    target_site_packages = site_packages[0]
    assert target_site_packages.is_relative_to(venv_dir.resolve())
    assert not list(target_site_packages.glob("*.pth"))

    for entry in snapshot["path"]:
        path = Path(str(entry) if entry else cwd).resolve()
        assert PROJECT_ROOT.resolve() != path
        assert PROJECT_ROOT.resolve() not in path.parents
        assert PROJECT_VENV != path
        assert PROJECT_VENV not in path.parents
        if "site-packages" in path.parts:
            assert path == target_site_packages or target_site_packages in path.parents
    return python, console


def _create_venv(venv_dir: Path, *, cwd: Path) -> tuple[Path, Path]:
    _run(
        [str(PROJECT_PYTHON), "-m", "venv", str(venv_dir)],
        cwd=cwd,
        environment=_environment(venv_dir),
    )
    return _assert_clean_venv(venv_dir, cwd=cwd)


def _assert_installed_source(
    python: Path,
    *,
    runtime_venv: Path,
    cwd: Path,
    environment: dict[str, str],
) -> None:
    result = _run(
        [
            str(python),
            "-c",
            "import sharper; import sharper.lifecycle_monitoring as lifecycle; "
            "print(sharper.__file__); print(lifecycle.__file__)",
        ],
        cwd=cwd,
        environment=environment,
    )
    sources = [Path(value).resolve() for value in result.stdout.splitlines()]
    assert len(sources) == 2
    for source in sources:
        assert source.is_relative_to(runtime_venv.resolve())
        assert PROJECT_ROOT.resolve() not in source.parents
        assert PROJECT_ROOT.resolve() != source
        assert PROJECT_VENV.resolve() not in source.parents


def _smoke_artifact(
    *,
    python: Path,
    console: Path,
    runtime_venv: Path,
    uv_cache_dir: Path,
    cwd: Path,
    csv_path: Path,
    report_path: Path,
) -> None:
    environment = _runtime_environment(runtime_venv, uv_cache_dir=uv_cache_dir)
    assert python.exists()
    assert console.exists()
    assert python.parent == console.parent
    _assert_installed_source(
        python,
        runtime_venv=runtime_venv,
        cwd=cwd,
        environment=environment,
    )

    public_smoke = _run(
        [
            str(python),
            "-c",
            "import sharper; "
            "assert sharper.__version__ == '0.2.0'; "
            "assert set(sharper.__all__) <= set(vars(sharper)); "
            "assert sharper.__all__[-39:-34] == ['DataAuditRoles', "
            "'ColumnAuditRule', 'DataAuditConfig', 'DataAuditResult', "
            "'audit_data_quality']; "
            "assert sharper.__all__[-34:-28] == ['StrategyCondition', "
            "'DecisionRule', 'DecisionConstraint', 'DecisionStrategyConfig', "
            "'DecisionStrategyResult', 'simulate_decision_strategy']; "
            "assert sharper.__all__[-28:-21] == ['MonitoringCondition', "
            "'EarlyWarningRule', 'WarningScenario', 'LifecycleState', "
            "'LifecycleMonitoringConfig', 'LifecycleMonitoringResult', "
            "'monitor_lifecycle']; "
            "assert sharper.__all__[-21:-9] == ['GovernanceEvidenceRef', "
            "'GovernanceCandidate', 'GovernanceCriterion', "
            "'GovernanceExplanation', 'GovernanceAttributionEvidence', "
            "'GovernancePredictionProfile', 'GovernancePerformanceEvidence', "
            "'GovernanceMetadata', 'GovernancePolicy', 'GovernanceResult', "
            "'evaluate_governance', 'plot_model_governance']; "
            "assert sharper.__all__[-9:] == ['V02ScoreValidationRequest', "
            "'V02AuditRequest', 'V02PreLoanRequest', 'V02PostLoanRequest', "
            "'V02GovernanceRequest', 'V02WorkflowRequest', 'V02WorkflowResult', "
            "'run_v02_workflow', 'generate_v02_report']; "
            "import sharper.model_governance as governance; "
            "import sharper.lifecycle_monitoring as lifecycle; "
            "import matplotlib; "
            "from sharper import MonitoringCondition, EarlyWarningRule, "
            "WarningScenario, "
            "LifecycleState, LifecycleMonitoringConfig, LifecycleMonitoringResult, "
            "monitor_lifecycle; "
            "assert lifecycle.__file__.startswith('" + str(runtime_venv) + "'); "
            "assert governance.__file__.startswith('" + str(runtime_venv) + "'); "
            "assert matplotlib.__file__.startswith('" + str(runtime_venv) + "'); "
            "assert all(hasattr(sharper, name) for name in "
            "('MonitoringCondition', 'EarlyWarningRule', 'WarningScenario', "
            "'LifecycleState', 'LifecycleMonitoringConfig', "
            "'LifecycleMonitoringResult', 'monitor_lifecycle')); "
            "assert not hasattr(sharper, '_ConditionNode'); "
            "assert not hasattr(sharper, '_compile_condition')",
        ],
        cwd=cwd,
        environment=environment,
    )
    assert public_smoke.stdout == ""

    task18_smoke = _run(
        [
            str(python),
            "-c",
            "from datetime import datetime; import pandas as pd; "
            "from sharper import (EarlyWarningRule, LifecycleMonitoringConfig, "
            "LifecycleState, MonitoringCondition, WarningScenario, monitor_lifecycle); "
            "condition=MonitoringCondition('atomic','gt','column','feature',"
            "'literal',0); "
            "rule=EarlyWarningRule('rule',0,'high',condition); "
            "state=LifecycleState('current',0,0,condition); "
            "unknown=LifecycleState('unknown',0,1,condition); "
            "config=LifecycleMonitoringConfig('m','v1',datetime(2025,1,2),'entity',"
            "'observed','available',('feature',),None,None,None,False,"
            "__import__('datetime').timedelta(days=1),__import__('datetime').timedelta(days=2),"
            "False,None,'day',None,(WarningScenario('reference','rule_set',(rule,)),),"
            "'reference',(('high',1),),(state,unknown),'current','unknown'); "
            "result=monitor_lifecycle(pd.DataFrame({'entity':['e'],'observed':[datetime(2025,1,1)],"
            "'available':[datetime(2025,1,1)],'feature':[1]}),config); "
            "assert isinstance(result.monitoring_summary,pd.DataFrame)",
        ],
        cwd=cwd,
        environment=environment,
    )
    assert task18_smoke.stdout == ""

    task16_smoke = _run(
        [
            str(python),
            "-c",
            "import pandas as pd; from sharper import audit_data_quality; "
            "r=audit_data_quality(pd.DataFrame({'x':[1.0,None]})); "
            "assert r.n_rows == 2 and len(r.column_profile) == 1",
        ],
        cwd=cwd,
        environment=environment,
    )
    assert task16_smoke.stdout == ""

    task17_smoke = _run(
        [
            str(python),
            "-c",
            "from datetime import datetime; import pandas as pd; "
            "from sharper import DecisionStrategyConfig, simulate_decision_strategy; "
            "c=DecisionStrategyConfig('s','v1',datetime(2025,1,1),None,"
            "datetime(2025,1,2),(),'select','review',"
            "(('select','selected'),('review','review'))); "
            "r=simulate_decision_strategy(pd.DataFrame({'x':[1,2]}),c); "
            "assert r.decided_n_rows == 2 and r.unavailable_n_rows == 0",
        ],
        cwd=cwd,
        environment=environment,
    )
    assert task17_smoke.stdout == ""

    for command in (
        [str(console), "--help"],
        [str(console), "analyze", "--help"],
        [str(python), "-m", "sharper.cli", "--help"],
    ):
        assert command[0] in {str(console), str(python)}
        _run(command, cwd=cwd, environment=environment)

    for command in (
        [str(console), "--version"],
        [str(python), "-m", "sharper.cli", "--version"],
    ):
        assert command[0] in {str(console), str(python)}
        result = _run(command, cwd=cwd, environment=environment)
        assert result.stdout == "sharper 0.2.0\n"

    analyze = [str(console), "analyze", str(csv_path), "--output", str(report_path)]
    assert analyze[0] == str(console)
    _run(analyze, cwd=cwd, environment=environment)
    assert report_path.exists()
    assert report_path.with_name(f"{report_path.stem}_assets").is_dir()
    v02_report_path = report_path.with_name("v02-cli.md")
    _run(
        [
            str(console),
            "v02-run",
            str(csv_path),
            "--output",
            str(v02_report_path),
            "--audit",
        ],
        cwd=cwd,
        environment=environment,
    )
    assert v02_report_path.exists()
    assert v02_report_path.with_name("v02-cli_assets").is_dir()


def _wheel_api_smoke(
    python: Path,
    *,
    runtime_venv: Path,
    uv_cache_dir: Path,
    cwd: Path,
    output_dir: Path,
) -> None:
    output_dir.mkdir()
    script = """
from pathlib import Path
import pandas as pd
from sharper import generate_analysis_report, run_analysis

rows = list(range(24))
frame = pd.DataFrame({
    "age": [24 + row for row in rows],
    "segment": ["north" if row % 2 == 0 else "south" for row in rows],
    "outcome": ["yes" if row % 2 == 0 else "no" for row in rows],
})
analysis = run_analysis(
    frame,
    target="outcome",
    task="classification",
    include_model=False,
)
assert analysis.target_analysis is not None
assert analysis.training is None and analysis.evaluation is None
model = run_analysis(frame, target="outcome", task="classification", include_model=True)
assert model.training is not None and model.evaluation is not None
output = Path("wheel-api.html")
generate_analysis_report(model, output, format="html")
assert output.exists()
assert output.with_name("wheel-api_assets").is_dir()
assert list(output.with_name("wheel-api_assets").glob("*.png"))
    """
    command = [str(python), "-c", script]
    assert command[0] == str(python)
    _run(
        command,
        cwd=output_dir,
        environment=_runtime_environment(runtime_venv, uv_cache_dir=uv_cache_dir),
    )


def _build_sdist_wheel(
    sdist: Path,
    *,
    python: Path,
    cwd: Path,
    output_dir: Path,
    uv_cache_dir: Path,
) -> Path:
    """Build a wheel from extracted sdist source without isolation or network."""
    extracted = cwd / "sdist-wheel-source"
    with tarfile.open(sdist) as archive:
        if sys.version_info >= (3, 12):
            archive.extractall(extracted, filter="data")
        else:
            archive.extractall(extracted)
    source_root = next(extracted.iterdir())
    output_dir.mkdir()
    _run(
        [
            str(python),
            "-m",
            "build",
            "--no-isolation",
            "--wheel",
            "--outdir",
            str(output_dir),
            str(source_root),
        ],
        cwd=cwd,
        environment=_environment(PROJECT_VENV, uv_cache_dir=uv_cache_dir),
        expect_empty_stderr=False,
    )
    wheels = sorted(output_dir.glob("*.whl"))
    assert len(wheels) == 1
    return wheels[0]


def _run_sdist_examples(
    sdist: Path,
    *,
    python: Path,
    runtime_venv: Path,
    uv_cache_dir: Path,
    cwd: Path,
    output_dir: Path,
) -> None:
    extracted = cwd / "sdist-source"
    with tarfile.open(sdist) as archive:
        if sys.version_info >= (3, 12):
            archive.extractall(extracted, filter="data")
        else:
            archive.extractall(extracted)
    source_root = next(extracted.iterdir())
    runner = cwd / "example-runner"
    runner.mkdir()
    output_dir.mkdir()
    example_names = (
        "basic_analysis.py",
        "baseline_modeling.py",
        "v02_score_validation.py",
        "v02_preloan.py",
        "v02_postloan.py",
        "v02_combined_report.py",
        "v02_cli_json.py",
    )
    for name in example_names:
        shutil.copy2(source_root / "examples" / name, runner / name)

    for script, report in (
        ("basic_analysis.py", "basic-analysis.md"),
        ("baseline_modeling.py", "baseline-modeling.html"),
    ):
        destination = output_dir / script.removesuffix(".py")
        destination.mkdir()
        command = [str(python), str(runner / script), "--output-dir", str(destination)]
        assert command[0] == str(python)
        _run(
            command,
            cwd=cwd,
            environment=_runtime_environment(runtime_venv, uv_cache_dir=uv_cache_dir),
        )
        assert (destination / report).exists()
        assets = destination / f"{Path(report).stem}_assets"
        assert assets.is_dir()
        assert list(assets.glob("*.png"))
        assert not list(destination.glob(".*.sharper-staging"))
        assert not list(destination.glob(".*.sharper-backup"))

    for script in (
        "v02_score_validation.py",
        "v02_preloan.py",
        "v02_postloan.py",
        "v02_cli_json.py",
    ):
        _run(
            [str(python), str(runner / script)],
            cwd=cwd,
            environment=_runtime_environment(runtime_venv, uv_cache_dir=uv_cache_dir),
        )
    combined_output = output_dir / "v02-combined"
    _run(
        [
            str(python),
            str(runner / "v02_combined_report.py"),
            "--output-dir",
            str(combined_output),
        ],
        cwd=cwd,
        environment=_runtime_environment(runtime_venv, uv_cache_dir=uv_cache_dir),
    )
    assert (combined_output / "v02-combined.md").is_file()
    assert (combined_output / "v02-combined_assets").is_dir()


def test_v02_distribution_source_free_smoke(tmp_path: Path) -> None:
    """Build artifacts and smoke isolated installs without ambient dependencies."""
    uv_cache_dir, prepared_manifest = _prepared_cache()
    output_dir = tmp_path / "artifacts"
    output_dir.mkdir()
    build_python = _build_python()
    build = _run(
        [
            str(build_python),
            "-m",
            "build",
            "--no-isolation",
            "--outdir",
            str(output_dir),
        ],
        cwd=PROJECT_ROOT,
        environment=_environment(build_python.parent.parent),
        expect_empty_stderr=False,
    )
    assert build.stdout or build.stderr == ""
    artifacts = sorted(output_dir.iterdir())
    assert len(artifacts) == 2
    wheel = next(path for path in artifacts if path.suffix == ".whl")
    sdist = next(path for path in artifacts if path.suffixes[-2:] == [".tar", ".gz"])
    assert wheel.name.startswith(f"sharper-{VERSION}-")
    assert sdist.name == f"sharper-{VERSION}.tar.gz"
    assert wheel.stat().st_size > 0
    assert sdist.stat().st_size > 0

    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        assert "sharper/data_audit.py" in names
        assert "sharper/_condition_kernel.py" in names
        assert "sharper/decision_strategy.py" in names
        assert "sharper/lifecycle_monitoring.py" in names
        assert "sharper/model_governance.py" in names
        metadata_name = next(
            name for name in names if name.endswith(".dist-info/METADATA")
        )
        wheel_metadata = _metadata(archive.read(metadata_name))
        _assert_metadata(wheel_metadata)
        assert (
            _requires_dist_hash(_requires_dist(wheel_metadata))
            == prepared_manifest["requires_dist_sha256"]
        )
        assert any(name.endswith(".dist-info/licenses/LICENSE") for name in names)
        entry_points = next(
            name for name in names if name.endswith(".dist-info/entry_points.txt")
        )
        assert "sharper = sharper.cli:app" in archive.read(entry_points).decode()

    with tarfile.open(sdist) as archive:
        names = archive.getnames()
        assert any(name.endswith("/src/sharper/data_audit.py") for name in names)
        assert any(name.endswith("/src/sharper/_condition_kernel.py") for name in names)
        assert any(name.endswith("/src/sharper/decision_strategy.py") for name in names)
        assert any(
            name.endswith("/src/sharper/lifecycle_monitoring.py") for name in names
        )
        assert any(name.endswith("/src/sharper/model_governance.py") for name in names)
        assert any(
            name.endswith(
                "/docs/decisions/task17-preloan-eligibility-strategy-contract.md"
            )
            for name in names
        )
        assert any(name.endswith("/LICENSE") for name in names)
        assert any(name.endswith("/examples/basic_analysis.py") for name in names)
        assert any(name.endswith("/examples/baseline_modeling.py") for name in names)
        package_info = next(name for name in names if name.endswith("/PKG-INFO"))
        sdist_metadata = _metadata(archive.extractfile(package_info).read())  # type: ignore[union-attr]
        _assert_metadata(sdist_metadata)
        assert (
            _requires_dist_hash(_requires_dist(sdist_metadata))
            == prepared_manifest["requires_dist_sha256"]
        )
        pyproject = next(name for name in names if name.endswith("/pyproject.toml"))
        assert (
            'sharper = "sharper.cli:app"'
            in archive.extractfile(pyproject).read().decode()
        )  # type: ignore[union-attr]

    outside = tmp_path / "outside"
    outside.mkdir()
    csv_path = outside / "input.csv"
    csv_path.write_text(
        "number,category\n1,one\n2,two\n3,one\n4,two\n", encoding="utf-8"
    )
    assert uv_cache_dir != Path.home() / ".cache" / "uv"
    sdist_wheel = _build_sdist_wheel(
        sdist,
        python=build_python,
        cwd=outside,
        output_dir=tmp_path / "sdist-wheel",
        uv_cache_dir=uv_cache_dir,
    )
    assert sdist_wheel.name.startswith(f"sharper-{VERSION}-")
    with zipfile.ZipFile(sdist_wheel) as archive:
        assert "sharper/lifecycle_monitoring.py" in archive.namelist()
        metadata_name = next(
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        )
        derived_metadata = _metadata(archive.read(metadata_name))
        _assert_metadata(derived_metadata)
        assert (
            _requires_dist_hash(_requires_dist(derived_metadata))
            == prepared_manifest["requires_dist_sha256"]
        )

    wheel_runtime = tmp_path / "wheel-runtime-venv"
    sdist_runtime = tmp_path / "sdist-runtime-venv"
    wheel_python, wheel_console = _create_venv(wheel_runtime, cwd=outside)
    sdist_python, sdist_console = _create_venv(sdist_runtime, cwd=outside)
    assert wheel_python != sdist_python
    for artifact, python, runtime in (
        (wheel, wheel_python, wheel_runtime),
        (sdist_wheel, sdist_python, sdist_runtime),
    ):
        install = [
            str(_uv()),
            "pip",
            "install",
            "--offline",
            "--python",
            str(python),
            str(artifact),
        ]
        _run(
            install,
            cwd=outside,
            environment=_runtime_environment(runtime, uv_cache_dir=uv_cache_dir),
            expect_empty_stderr=False,
        )

    _smoke_artifact(
        python=wheel_python,
        console=wheel_console,
        runtime_venv=wheel_runtime,
        uv_cache_dir=uv_cache_dir,
        cwd=outside,
        csv_path=csv_path,
        report_path=outside / "wheel-cli.md",
    )
    _smoke_artifact(
        python=sdist_python,
        console=sdist_console,
        runtime_venv=sdist_runtime,
        uv_cache_dir=uv_cache_dir,
        cwd=outside,
        csv_path=csv_path,
        report_path=outside / "sdist-cli.md",
    )
    _wheel_api_smoke(
        wheel_python,
        runtime_venv=wheel_runtime,
        uv_cache_dir=uv_cache_dir,
        cwd=outside,
        output_dir=outside / "wheel-api",
    )
    _run_sdist_examples(
        sdist,
        python=sdist_python,
        runtime_venv=sdist_runtime,
        uv_cache_dir=uv_cache_dir,
        cwd=outside,
        output_dir=outside / "sdist-examples",
    )


def test_build_python_rejects_external_interpreters() -> None:
    """Only the project UV environment may control offline artifact builds."""
    assert _build_python(str(PROJECT_PYTHON)) == PROJECT_PYTHON
    external = Path(sys.base_prefix) / (
        "python.exe" if os.name == "nt" else "bin/python"
    )
    assert external.is_file()
    assert external.absolute() != PROJECT_PYTHON
    with pytest.raises(AssertionError, match="must equal the project .venv/bin/python"):
        _build_python(str(external))
