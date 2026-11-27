#!/usr/bin/env bash
# Cirrus AI: run any ticket file through the whole system, unattended, with one command.
#
#   ./run.sh <tickets file> [output folder]
#
#   ./run.sh data/validation_tickets.json
#   ./run.sh /path/to/hidden_set.jsonl results/hidden
#
# What it does, with no questions asked:
#   1. creates the virtual environment (.venv) if there is none, and installs requirements.txt
#      (again only when requirements.txt changes);
#   2. builds whatever is missing, from the development tickets only: the documentation index, the
#      classifier, the retrieval and routing thresholds (the first run takes a few minutes);
#   3. runs every ticket through the pipeline and writes, in a dated folder inside the output folder:
#        outcomes.jsonl  one line per ticket: sent or escalated, reply or handoff, tier, reasons
#        metrics.json    every figure, machine-readable
#        metrics.md      the report, with the Brief's targets marked met / not met
#        decisions.db    the decision log for the run
#        decisions.jsonl every decision in the governance record format
#
# The ticket file can have any name: a JSON list, {"tickets": [...]}, or JSON Lines. Labels are
# optional: without them the label-based figures say n/a. Settings come from .env if it exists
# (no API key: replies are written from the articles, without a model, and the report says so).
#
# Exit codes: 0 done and every decision logged; 2 done but the decision log does not reconcile;
#             1 the input could not be read, or setup failed.
# Extra options after the output folder go to the harness, e.g. --limit 10, --judge.
set -euo pipefail
cd "$(dirname "$0")"

if [ $# -lt 1 ]; then
    sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi
INPUT="$1"
OUTPUT="${2:-evaluation/results}"
shift $(( $# >= 2 ? 2 : 1 ))

if [ ! -f "$INPUT" ]; then
    echo "run.sh: no such ticket file: $INPUT" >&2
    exit 1
fi

VENV="${VENV:-.venv}"
if [ ! -x "$VENV/bin/python" ]; then
    PY="${PYTHON:-}"
    if [ -z "$PY" ]; then
        for candidate in python3.11 python3.12 python3.13 python3; do
            if command -v "$candidate" >/dev/null 2>&1; then PY="$candidate"; break; fi
        done
    fi
    echo "run.sh: creating $VENV with $PY"
    "$PY" -m venv "$VENV"
fi

STAMP="$VENV/.requirements.sha256"
WANT="$( (shasum -a 256 requirements.txt 2>/dev/null || sha256sum requirements.txt) | cut -d' ' -f1)"
if [ ! -f "$STAMP" ] || [ "$(cat "$STAMP")" != "$WANT" ]; then
    echo "run.sh: installing requirements (first run only)"
    "$VENV/bin/python" -m pip --version >/dev/null 2>&1 || "$VENV/bin/python" -m ensurepip --upgrade >/dev/null
    "$VENV/bin/python" -m pip install --quiet --upgrade pip
    "$VENV/bin/python" -m pip install --quiet -r requirements.txt
    echo "$WANT" > "$STAMP"
fi

exec "$VENV/bin/python" -m evaluation.harness --input "$INPUT" --output "$OUTPUT" "$@"
