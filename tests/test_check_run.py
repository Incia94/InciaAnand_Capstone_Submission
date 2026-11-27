"""evaluation.check_run: the run passes or fails on its own checks; targets are reported separately."""
import json
import shutil
from pathlib import Path

from evaluation.check_run import main

RESULTS = Path(__file__).resolve().parents[1] / "evaluation" / "results"


def a_run(tmp_path):
    src = sorted(RESULTS.glob("*/metrics.json"))[-1]
    dst = tmp_path / "run1"
    dst.mkdir()
    shutil.copy(src, dst / "metrics.json")
    return dst


def test_a_reconciled_run_passes_even_with_missed_targets(tmp_path, capsys):
    run = a_run(tmp_path)
    assert main([str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "Run: PASSED" in out and "NOT met" in out and str(run / "metrics.md") in out


def test_a_run_that_does_not_reconcile_fails(tmp_path, capsys):
    run = a_run(tmp_path)
    m = json.loads((run / "metrics.json").read_text())
    m["reconcile"]["reconciled"] = False
    m["reconcile"]["missing_outcome"] = ["T-1"]
    (run / "metrics.json").write_text(json.dumps(m))
    assert main([str(run)]) == 1
    assert "Run: FAILED" in capsys.readouterr().out


def test_no_run_found(tmp_path):
    assert main([str(tmp_path / "nothing")]) == 1
