"""run.sh: the one-command entry point for running any ticket file unattended."""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "run.sh"
pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def run(*args, **env):
    return subprocess.run(["bash", str(SCRIPT), *args], capture_output=True, text=True, timeout=60,
                          env={**os.environ, **env}, stdin=subprocess.DEVNULL)


def test_script_is_valid_and_executable():
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0
    assert os.access(SCRIPT, os.X_OK)


def test_without_arguments_it_explains_itself_and_fails():
    r = run()
    assert r.returncode == 1 and "./run.sh <tickets file> [output folder]" in r.stdout


def test_a_missing_ticket_file_fails_before_any_setup(tmp_path):
    r = run(str(tmp_path / "missing.jsonl"), VENV=str(tmp_path / "no-venv"))
    assert r.returncode == 1 and "no such ticket file" in r.stderr
    assert not (tmp_path / "no-venv").exists()
