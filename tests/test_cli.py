from typer.testing import CliRunner

from mnemonic.cli import app

runner = CliRunner()


def test_help_exits_zero() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0


def test_version_output() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "mnemonic 0.1.0" in result.output


def test_list_adapters() -> None:
    result = runner.invoke(app, ["list-adapters"])
    assert result.exit_code == 0
    assert "engram" in result.output


def test_run_requires_options() -> None:
    result = runner.invoke(app, ["run"])
    assert result.exit_code != 0
