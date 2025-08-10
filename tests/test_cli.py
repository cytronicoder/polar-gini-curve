"""Tests for the CLI."""

import subprocess
import sys


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
    csv_path.write_text("x,y,label\n0,0,0\n1,0,0\n0,1,1\n1,1,1\n")
    result = subprocess.run(
        [sys.executable, "-m", "polargini.cli", "--csv", str(csv_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "Curve" in result.stdout
