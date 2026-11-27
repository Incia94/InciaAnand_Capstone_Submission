"""Did the evaluation run pass? Reads a finished run's metrics.json and answers in two parts.

    python -m evaluation.check_run                       # the latest run in evaluation/results/
    python -m evaluation.check_run evaluation/results/hidden
    python -m evaluation.check_run evaluation/results/eval-20261121T130722Z-86c8a8

1. The run itself (pass or fail): the file was read, every ticket was processed with nobody
   intervening, every ticket ended sent or escalated exactly once, and the decision log reconciles
   with the tickets. These are what make the unattended run count at all.
2. The targets (met or not met): the Evaluation Framework's business, technical and governance
   targets. A run can pass while missing targets; the report says which and why.

Exit code: 0 when the run passed (whatever the targets), 1 when it did not or no run was found.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional, Sequence


def find_run(path: Path) -> Optional[Path]:
    """`path` itself if it holds metrics.json, else the newest run folder inside it."""
    if (path / "metrics.json").is_file():
        return path
    runs = sorted((p for p in path.rglob("metrics.json")), key=lambda p: p.stat().st_mtime)
    return runs[-1].parent if runs else None


def run_checks(m: dict) -> list[tuple[str, bool, str]]:
    run, rec = m.get("run", {}), m.get("reconcile", {})
    outcomes = rec.get("outcomes") or {}
    ended = sum(outcomes.get(k, 0) for k in ("sent", "escalated"))
    expected = rec.get("tickets_expected", run.get("tickets_in_file"))
    return [
        ("every ticket in the file was processed",
         run.get("tickets_processed") == run.get("tickets_in_file") and not run.get("interrupted"),
         f"{run.get('tickets_processed')} of {run.get('tickets_in_file')}"),
        ("every ticket ended sent or escalated", ended == expected and not rec.get("missing_outcome"),
         f"{outcomes.get('sent', 0)} sent, {outcomes.get('escalated', 0)} escalated"),
        ("no ticket has two outcomes", not rec.get("duplicate_outcome"), f"{len(rec.get('duplicate_outcome') or [])}"),
        ("every decision was logged (decision log reconciles)", bool(rec.get("reconciled")),
         f"{rec.get('decisions')} decisions, {rec.get('write_failures', 0)} write failures"),
    ]


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    where = Path(args[0]) if args else Path("evaluation/results")
    folder = find_run(where) if where.exists() else None
    if folder is None:
        print(f"No finished run found in {where}.")
        return 1
    m = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    run = m.get("run", {})
    print(f"Run {run.get('run_id')}  ({run.get('input')}, {run.get('tickets_in_file')} tickets, "
          f"{run.get('started_at', '')[:16].replace('T', ' ')} UTC, {run.get('duration_s')} s)\n")
    checks = run_checks(m)
    print("1. The run")
    for name, ok, detail in checks:
        print(f"   {'PASS' if ok else 'FAIL'}  {name}  ({detail})")
    targets = m.get("metrics", {}).get("targets") or []
    print("\n2. The targets")
    for t in targets:
        mark = "met    " if t.get("met") else "NOT met"
        print(f"   {mark}  {t['target']}  (value {t.get('value')}, goal {t.get('goal')})")
    passed = all(ok for _, ok, _ in checks)
    met = sum(1 for t in targets if t.get("met"))
    print(f"\nRun: {'PASSED' if passed else 'FAILED'}.  Targets: {met} of {len(targets)} met.  "
          f"Full report: {folder / 'metrics.md'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
