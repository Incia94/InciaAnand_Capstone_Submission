"""The Streamlit workspace (streamlit_app.py), driven headlessly: demo sign-in by username, the
supervisor assigns sample tickets and sees the reports, and the agent sends Cirrus's cited reply."""
import sqlite3
from pathlib import Path

import pytest

from src.staff import StaffDirectory

APP = str(Path(__file__).resolve().parents[1] / "streamlit_app.py")
AppTest = pytest.importorskip("streamlit.testing.v1").AppTest


@pytest.fixture
def env(tmp_path, monkeypatch):
    for k, v in dict(RETRIEVAL_BACKEND="lexical", GENERATOR="extractive", STAFF_FILE=tmp_path / "staff.json",
                     DATABASE_URL=f"sqlite:///{tmp_path / 'decisions.db'}", REVIEW_DB=tmp_path / "reviews.db",
                     TICKETS_DB=tmp_path / "tickets.db", KILL_SWITCH_FILE=tmp_path / "KS", REPLY_MODE="review",
                     OPENROUTER_API_KEY="", DEMO_PASSWORD="").items():
        monkeypatch.setenv(k, str(v))
    monkeypatch.delenv("DEMO_ACCOUNTS", raising=False)
    import streamlit as st
    st.cache_resource.clear()                       # each test gets a hub over its own temporary files
    yield tmp_path
    st.cache_resource.clear()


def sign_in(name, password="password"):
    at = AppTest.from_file(APP, default_timeout=300).run()
    at.text_input[0].input(name)
    at.text_input[1].input(password)
    return at.button[0].click().run()


def labelled(at, label):
    return [b for b in at.button if b.label == label]


def test_demo_accounts_and_usernames(env):
    s = StaffDirectory()
    assert len(s.ensure_demo()) == 4 and s.ensure_demo() == []            # created once, then left alone
    assert s.check("marcus", "password")["role"] == "supervisor"
    assert s.check("Admin", "password")["name"] == "Marcus Adeyemi"
    assert [s.check(n, "password")["role"] for n in ("asha", "Sofia", "ravi@cloudserve.example")] == ["agent"] * 3
    assert s.check("asha", "nope-nope") is None and s.check("nobody", "password") is None
    assert "password" not in (env / "staff.json").read_text()                # salted hashes only


def test_no_demo_accounts_when_turned_off(env, monkeypatch):
    monkeypatch.setenv("DEMO_ACCOUNTS", "off")
    assert StaffDirectory().ensure_demo() == [] and StaffDirectory().list() == []


def test_supervisor_assigns_and_agent_sends_in_streamlit(env):
    bad = sign_in("marcus", "wrong-password")
    assert "Wrong username or password." in [e.value for e in bad.error]

    sup = sign_in("marcus")
    assert not sup.exception and sup.session_state["user"]["role"] == "supervisor"
    labelled(sup, "Add sample tickets")[0].click().run()
    assert labelled(sup, "Assign"), "sample tickets should arrive unassigned"
    labelled(sup, "Assign")[0].click().run()
    assert not sup.exception
    metrics = {m.label: m.value for m in sup.metric}
    assert metrics["Assigned"] == "1" and int(metrics["Tickets"]) >= 8
    assert any(s.value.startswith("Every ticket is logged") for s in sup.success)

    agent = sign_in("asha")
    assert not agent.exception and agent.session_state["user"]["name"] == "Asha"
    reply = agent.text_area[0].value
    assert "[1]" in reply                                                   # cited reply prepared by Cirrus
    text = " ".join(m.value for m in agent.markdown)
    assert not [w for w in ("RAG", "Tier ", "intent", "threshold", "extractive", "lexical") if w in text]
    labelled(agent, "Send reply")[0].click().run()
    assert not agent.exception and "Reply sent." in [t.value for t in agent.toast]
    row = sqlite3.connect(env / "tickets.db").execute(
        "select status, work_status from tickets where assignee='asha@cloudserve.example'").fetchone()
    assert row == ("resolved", "resolved")


def test_agent_cannot_open_the_supervisor_screen(env):
    agent = sign_in("ravi")
    assert not labelled(agent, "Add sample tickets") and not labelled(agent, "Assign")
    assert "My tickets" in " ".join(m.value for m in agent.markdown)
