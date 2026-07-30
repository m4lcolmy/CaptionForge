"""Integration tests for module execution."""

import subprocess
import sys


def test_module_entrypoint_displays_help() -> None:
    """Running the package without arguments should display CLI help."""
    result = subprocess.run(
        [sys.executable, "-m", "app"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "CaptionForge" in result.stdout
