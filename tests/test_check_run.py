"""evaluation.check_run: the run passes or fails on its own checks; targets are reported separately.

Each test writes its own small metrics.json, so the result never depends on what runs happen to be
in evaluation/results/ on a given machine."""
import json

from evaluation.check_run import main


def write_run(folder, reconciled=True, processed=80, missing=()):
    folder.mkdir(parents=True)
    metrics = {
        "run": {"run_id": folder.name, "input": "tickets.json", "tickets_in_file": 80,
                "tickets_processed": processed, "interrupted": processed < 80,
                "started_at": "2026-11-21T13:07:22+00:00", "duration_s": 39.7},
        "reconcile": {"tickets_expected": 80, "tickets_with_outcome": 80 - len(missing), "decisions": 551,
                      "outcomes": {"sent": 2, "pending_review": 61, "escalated": 17 - len(missing)},
                      "missing_outcome": list(missing),
                      "duplicate_outcome": [], "write_failures": 0, "reconciled": reconciled},
        "metrics": {"targets": [
            {"target": "Business: first contact resolution 60% or better", "value": 0.79, "goal": ">= 0.6",
             "met": True},
            {"target": "Technical: citation accuracy 95% or better", "value": 0.69, "goal": ">= 0.95",
             "met": False}]},
    }
    (folder / "metrics.json").write_text(json.dumps(metrics))
    return folder


def test_a_reconciled_run_passes_even_with_missed_targets(tmp_path, capsys):
    run = write_run(tmp_path / "eval-1")
    assert main([str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "Run: PASSED" in out and "Targets: 1 of 2 met" in out and "NOT met" in out
    assert "61 answered, reply held for an agent to check" in out
    assert str(run / "metrics.md") in out


def test_a_run_that_does_not_reconcile_fails(tmp_path, capsys):
    run = write_run(tmp_path / "eval-1", reconciled=False, missing=["T-1"])
    assert main([str(run)]) == 1
    assert "Run: FAILED" in capsys.readouterr().out


def test_an_interrupted_run_fails(tmp_path, capsys):
    run = write_run(tmp_path / "eval-1", processed=40)
    assert main([str(run)]) == 1
    assert "FAIL  every ticket in the file was processed  (40 of 80)" in capsys.readouterr().out


def test_no_run_found(tmp_path):
    assert main([str(tmp_path / "nothing")]) == 1
