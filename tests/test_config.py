from pathlib import Path

import pytest
from typer.testing import CliRunner

from iceberg_sar.cli import app
from iceberg_sar.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "config.yaml"


def test_repo_config_loads() -> None:
    cfg = load_config(CONFIG)
    assert cfg.section("chips")["size_px"] == 75
    assert cfg.path("aoi").is_file()
    assert cfg.path("raw") == (REPO_ROOT / "data" / "raw").resolve()


def test_missing_section_raises(tmp_path: Path) -> None:
    bad = tmp_path / "config.yaml"
    bad.write_text("paths: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing sections"):
        load_config(bad)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_cli_help_and_info() -> None:
    runner = CliRunner()
    assert runner.invoke(app, ["--help"]).exit_code == 0
    result = runner.invoke(app, ["info", "--config", str(CONFIG)])
    assert result.exit_code == 0, result.output
    assert "aoi:" in result.output
