"""Release-readiness distribution tests for built Sharper artifacts."""

from __future__ import annotations

import json
import os
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
VERSION = "0.1.0"


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


def _environment(venv_dir: Path) -> dict[str, str]:
    python, _ = _venv_paths(venv_dir)
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PATH"] = f"{python.parent}{os.pathsep}{os.defpath}"
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


def _assert_metadata(message: object) -> None:
    assert message["Name"] == "sharper"  # type: ignore[index]
    assert message["Version"] == VERSION  # type: ignore[index]
    assert message["License-Expression"] == "MIT"  # type: ignore[index]
    assert message["Requires-Python"] == ">=3.10"  # type: ignore[index]
    assert message["Description-Content-Type"] == "text/markdown"  # type: ignore[index]
    requirements = message.get_all("Requires-Dist") or []  # type: ignore[union-attr]
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


def _runtime_requirements(message: object) -> list[str]:
    """Read the core dependency requirements from built artifact metadata."""
    requirements = message.get_all("Requires-Dist") or []  # type: ignore[union-attr]
    return [value for value in requirements if "extra ==" not in value]


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


def _install_runtime_dependencies(
    python: Path,
    requirements: list[str],
    *,
    venv_dir: Path,
    cwd: Path,
) -> None:
    """Install metadata-declared core dependencies into one venv from uv cache."""
    _run(
        [
            str(_uv()),
            "pip",
            "install",
            "--offline",
            "--python",
            str(python),
            *requirements,
        ],
        cwd=cwd,
        environment=_environment(venv_dir),
        expect_empty_stderr=False,
    )
    _assert_clean_venv(venv_dir, cwd=cwd)


def _assert_installed_source(
    python: Path, *, venv_dir: Path, cwd: Path, environment: dict[str, str]
) -> None:
    result = _run(
        [str(python), "-c", "import sharper; print(sharper.__file__)"],
        cwd=cwd,
        environment=environment,
    )
    source = Path(result.stdout.strip()).resolve()
    assert "site-packages" in source.parts
    assert venv_dir.resolve() in source.parents
    assert PROJECT_ROOT.resolve() not in source.parents
    assert PROJECT_ROOT.resolve() != source
    assert PROJECT_VENV not in source.parents


def _smoke_artifact(
    *,
    python: Path,
    console: Path,
    venv_dir: Path,
    cwd: Path,
    csv_path: Path,
    report_path: Path,
) -> None:
    environment = _environment(venv_dir)
    assert python.exists()
    assert console.exists()
    assert python.parent == console.parent
    _assert_installed_source(
        python, venv_dir=venv_dir, cwd=cwd, environment=environment
    )

    public_smoke = _run(
        [
            str(python),
            "-c",
            "import sharper; "
            "assert sharper.__version__ == '0.1.0'; "
            "assert set(sharper.__all__) <= set(vars(sharper))",
        ],
        cwd=cwd,
        environment=environment,
    )
    assert public_smoke.stdout == ""

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
        assert result.stdout == "sharper 0.1.0\n"

    analyze = [str(console), "analyze", str(csv_path), "--output", str(report_path)]
    assert analyze[0] == str(console)
    _run(analyze, cwd=cwd, environment=environment)
    assert report_path.exists()
    assert report_path.with_name(f"{report_path.stem}_assets").is_dir()


def _wheel_api_smoke(
    python: Path, *, cwd: Path, output_dir: Path, venv_dir: Path
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
    _run(command, cwd=output_dir, environment=_environment(venv_dir))


def _run_sdist_examples(
    sdist: Path, *, python: Path, venv_dir: Path, cwd: Path, output_dir: Path
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
    for name in ("basic_analysis.py", "baseline_modeling.py"):
        shutil.copy2(source_root / "examples" / name, runner / name)

    for script, report in (
        ("basic_analysis.py", "basic-analysis.md"),
        ("baseline_modeling.py", "baseline-modeling.html"),
    ):
        destination = output_dir / script.removesuffix(".py")
        destination.mkdir()
        command = [str(python), str(runner / script), "--output-dir", str(destination)]
        assert command[0] == str(python)
        _run(command, cwd=cwd, environment=_environment(venv_dir))
        assert (destination / report).exists()
        assets = destination / f"{Path(report).stem}_assets"
        assert assets.is_dir()
        assert list(assets.glob("*.png"))
        assert not list(destination.glob(".*.sharper-staging"))
        assert not list(destination.glob(".*.sharper-backup"))


def test_built_wheel_and_sdist_are_offline_installable(tmp_path: Path) -> None:
    """Build both artifacts and smoke them from separate, source-free venvs."""
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
        metadata_name = next(
            name for name in names if name.endswith(".dist-info/METADATA")
        )
        wheel_metadata = _metadata(archive.read(metadata_name))
        _assert_metadata(wheel_metadata)
        runtime_requirements = _runtime_requirements(wheel_metadata)
        assert any(name.endswith(".dist-info/licenses/LICENSE") for name in names)
        entry_points = next(
            name for name in names if name.endswith(".dist-info/entry_points.txt")
        )
        assert "sharper = sharper.cli:app" in archive.read(entry_points).decode()

    with tarfile.open(sdist) as archive:
        names = archive.getnames()
        assert any(name.endswith("/LICENSE") for name in names)
        assert any(name.endswith("/examples/basic_analysis.py") for name in names)
        assert any(name.endswith("/examples/baseline_modeling.py") for name in names)
        package_info = next(name for name in names if name.endswith("/PKG-INFO"))
        _assert_metadata(_metadata(archive.extractfile(package_info).read()))  # type: ignore[union-attr]
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
    wheel_venv = tmp_path / "wheel-venv"
    sdist_venv = tmp_path / "sdist-venv"
    wheel_python, wheel_console = _create_venv(wheel_venv, cwd=outside)
    sdist_python, sdist_console = _create_venv(sdist_venv, cwd=outside)
    assert wheel_venv != sdist_venv
    assert wheel_python != sdist_python
    assert wheel_console != sdist_console
    _install_runtime_dependencies(
        wheel_python,
        runtime_requirements,
        venv_dir=wheel_venv,
        cwd=outside,
    )
    _install_runtime_dependencies(
        sdist_python,
        runtime_requirements,
        venv_dir=sdist_venv,
        cwd=outside,
    )

    for artifact, python, venv_dir in (
        (wheel, wheel_python, wheel_venv),
        (sdist, sdist_python, sdist_venv),
    ):
        install = [
            str(_uv()),
            "pip",
            "install",
            "--offline",
            "--python",
            str(python),
            "--no-deps",
        ]
        install.append(str(artifact))
        assert install[5] == str(python)
        _run(
            install,
            cwd=outside,
            environment=_environment(venv_dir),
            expect_empty_stderr=False,
        )
        _assert_clean_venv(venv_dir, cwd=outside)

    _smoke_artifact(
        python=wheel_python,
        console=wheel_console,
        venv_dir=wheel_venv,
        cwd=outside,
        csv_path=csv_path,
        report_path=outside / "wheel-cli.md",
    )
    _smoke_artifact(
        python=sdist_python,
        console=sdist_console,
        venv_dir=sdist_venv,
        cwd=outside,
        csv_path=csv_path,
        report_path=outside / "sdist-cli.md",
    )
    _wheel_api_smoke(
        wheel_python,
        cwd=outside,
        output_dir=outside / "wheel-api",
        venv_dir=wheel_venv,
    )
    _run_sdist_examples(
        sdist,
        python=sdist_python,
        venv_dir=sdist_venv,
        cwd=outside,
        output_dir=outside / "sdist-examples",
    )

    excel_path = outside / "input.xlsx"
    excel_path.write_bytes(b"base-wheel optional-dependency smoke")
    result = _run(
        [
            str(wheel_python),
            "-c",
            "from pathlib import Path\n"
            "from sharper import load_excel\n"
            "try:\n"
            "    load_excel(Path('input.xlsx'))\n"
            "except ImportError as error:\n"
            "    assert str(error) == "
            "'Install sharper[excel] to read Excel files'\n"
            "else:\n"
            "    raise AssertionError(\n"
            "        'base wheel unexpectedly provides Excel support'\n"
            "    )",
        ],
        cwd=outside,
        environment=_environment(wheel_venv),
    )
    assert result.stdout == ""


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
