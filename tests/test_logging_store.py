"""Tests for src/logging_store.py: the decision log (acceptance criterion A8)."""
import sqlite3

import pytest

from src import logging_store as L


@pytest.fixture
def store(tmp_path):
    s = L.DecisionStore(f"sqlite:///{tmp_path / 'decisions.db'}")
    yield s
    s.close()


def test_schema_contains_every_setup_guide_column(store):
    cols = [r[1] for r in sqlite3.connect(store.path).execute("PRAGMA table_info(decisions)")]
    for c in ("decision_id", "created_at", "ticket_id", "stage", "prediction", "confidence", "threshold",
              "action_taken", "reason", "sources_used", "guardrails", "prompt_version", "requirement_ids"):
        assert c in cols
    for c in ("run_id", "input_digest", "input_text", "model_version", "details", "latency_ms"):
        assert c in cols


def test_record_and_trail_round_trip(store):
    did = store.record("T1", "route", "escalate", "No help article matched.", prediction="billing_query",
                       confidence=0.42, threshold=0.8, sources=[{"chunk_id": "DOC-BILL-001::full"}],
                       input_text="My invoice is wrong", run_id="R1", details={"decided_by": "documentation"})
    (row,) = store.trail("T1")
    assert row["decision_id"] == did and row["confidence"] == 0.42 and row["threshold"] == 0.8
    assert row["sources_used"] == [{"chunk_id": "DOC-BILL-001::full"}]
    assert row["details"]["decided_by"] == "documentation"
    assert row["requirement_ids"] == ["FR-04", "FR-07", "FR-08", "FR-10", "FR-11", "A5"]
    assert row["input_digest"] and row["created_at"] and row["input_summary"] == "My invoice is wrong"
    assert row["model_name"].startswith("router")


def test_input_is_logged_with_personal_data_masked(store):
    store.record("T1", "ingest", "normalised", "ok",
                 input_text="I am a.rao@example.com, account CUST-1042, phone +44 20 7946 0958, "
                            "key sk-live-ABCDEFGH12345678xyz")
    row = store.trail("T1")[0]
    for secret in ("a.rao@example.com", "CUST-1042", "7946 0958", "sk-live-ABCDEFGH12345678xyz"):
        assert secret not in row["input_text"]
    assert "I am" in row["input_text"]                       # the rest stays readable


def answered(store, ticket_id, run_id):
    """The smallest complete trail for an answered ticket: ingest, routing, outcome."""
    store.record(ticket_id, "ingest", "normalised", "ok", run_id=run_id)
    store.record(ticket_id, "route", "auto_respond", "ok", run_id=run_id)
    store.log_outcome(ticket_id, "sent", "ok", run_id=run_id)


def test_reconcile_counts_outcomes_against_tickets(store):
    for t in ("A", "B", "C"):
        store.record(t, "ingest", "normalised", "ok", run_id="R")
    store.record("A", "route", "auto_respond", "ok", run_id="R")
    store.log_outcome("A", "sent", "ok", run_id="R")
    store.log_outcome("B", "escalated", "ok", run_id="R")
    rep = store.reconcile("R", ["A", "B", "C"])
    assert rep["missing_outcome"] == ["C"] and not rep["reconciled"]
    store.log_outcome("C", "escalated", "ok", run_id="R")
    rep = store.reconcile("R", ["A", "B", "C"])
    assert rep["reconciled"] and rep["outcomes"] == {"sent": 1, "escalated": 2}
    store.log_outcome("C", "escalated", "again", run_id="R")
    assert store.reconcile("R", ["A", "B", "C"])["duplicate_outcome"] == ["C"]


def test_runs_are_kept_apart(store):
    answered(store, "A", "R1")
    answered(store, "A", "R2")
    assert store.reconcile("R1", ["A"])["reconciled"] and store.reconcile("R2", ["A"])["decisions"] == 3


def test_unwritable_database_falls_back_to_file_and_still_reconciles(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    s = L.DecisionStore(f"sqlite:///{tmp_path / 'ok' / 'decisions.db'}")
    s._conn.close()
    s._conn = None                                            # simulate the database going away
    answered(s, "A", "R")
    assert s.write_failures == 3 and s.fallback_path.is_file()
    assert s.reconcile("R", ["A"])["reconciled"]


def test_nowhere_to_write_never_raises(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    s = L.DecisionStore(f"sqlite:///{blocker / 'sub' / 'decisions.db'}")
    assert not s.healthy
    assert s.record("A", "outcome", "sent", "ok")             # returns an id, does not raise
    assert s.last_write_ok is False


def test_only_sqlite_urls_are_accepted(tmp_path):
    with pytest.raises(ValueError):
        L.DecisionStore("postgresql://localhost/db")


def test_cli_trail_and_reconcile(store, capsys):
    answered(store, "A", "R")
    url = f"sqlite:///{store.path}"
    assert L.main(["--db", url, "trail", "A"]) == 0 and '"outcome"' in capsys.readouterr().out
    assert L.main(["--db", url, "reconcile", "--run-id", "R"]) == 0 and '"reconciled": true' in capsys.readouterr().out


def test_reconcile_flags_an_answer_with_no_ingest_or_routing_record(store):
    store.log_outcome("A", "sent", "ok", run_id="R")
    rep = store.reconcile("R", ["A"])
    assert not rep["reconciled"] and {g["gap"] for g in rep["path_gaps"]} == {
        "no ingest record", "answered without a routing record"}
