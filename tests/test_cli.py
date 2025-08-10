"""Tests for the CLI."""

import subprocess
import sys
from unittest.mock import patch

from polargini.cli import main


def test_cli_help():
    """Test the CLI help command."""
    result = subprocess.run(
        [sys.executable, "-m", "polargini.cli", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "Polar Gini" in result.stdout or "Polar Gini" in result.stderr


def test_cli_run(tmp_path):
    """Test running the CLI with a CSV file."""
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("x,y,cluster\n0,0,0\n1,0,0\n0,1,1\n1,1,1\n")
    result = subprocess.run(
        [sys.executable, "-m", "polargini.cli", "--csv", str(csv_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "Curve" in result.stdout


def test_cli_main_function(tmp_path):
    """Test the CLI main function directly for better coverage."""
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("x,y,cluster\n0,0,0\n1,0,0\n0,1,1\n1,1,1\n")

    with patch("sys.argv", ["polargini.cli", "--csv", str(csv_path)]):
        try:
            main()
        except SystemExit as e:
            assert e.code == 0


def test_cli_main_with_plot(tmp_path):
    """Test the CLI main function with plot option."""
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("x,y,cluster\n0,0,0\n1,0,0\n0,1,1\n1,1,1\n")

    with patch("sys.argv", ["polargini.cli", "--csv", str(csv_path), "--plot"]):
        try:
            main()
        except SystemExit as e:
            assert e.code == 0
