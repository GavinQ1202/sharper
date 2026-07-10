"""Distribution metadata smoke tests."""

from importlib.metadata import metadata, version


def test_distribution_identity() -> None:
    """Installed metadata matches the frozen distribution contract."""
    distribution = metadata("sharper")

    assert distribution["Name"] == "sharper"
    assert version("sharper") == "0.1.0"
    assert distribution["License-Expression"] == "MIT"
    assert distribution["Requires-Python"] == ">=3.10"


def test_dependency_groups() -> None:
    """Runtime, Excel, and development dependencies remain separated."""
    requirements = metadata("sharper").get_all("Requires-Dist") or []
    runtime = {item for item in requirements if "extra ==" not in item}

    assert runtime == {
        "matplotlib",
        "numpy",
        "pandas",
        "scikit-learn",
        "seaborn",
        "scipy",
        "typer",
    }
    assert any(item.startswith("openpyxl; extra == 'excel'") for item in requirements)
    assert not any(item.startswith("openpyxl") for item in runtime)
    assert any(item.startswith("pytest; extra == 'dev'") for item in requirements)
