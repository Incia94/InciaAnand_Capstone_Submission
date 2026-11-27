"""The support workspace: staff sign-in, the supervisor's queue and reports, the agent's tickets.

A supervisor sees every ticket from every channel and assigns it; an agent sees only their own
tickets, checks Cirrus's cited reply and sends it. Everything is logged, and nothing on screen uses
the pipeline's technical vocabulary."""
import json
import secrets
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src import api as A
from src.classify import Classifier
from src.generate import Generator
from src.guardrail import Guardrails
from src.hub import SupportHub
from src.ingest import load_tickets
from src.logging_store import DecisionStore
from src.retrieve import Corpus, Retriever, _labelled, approved_threshold_from, best_threshold, score_tickets
from src.route import Router, RoutingConfig, _label_sets
from src.staff import COOKIE, StaffDirectory
from src.workspace import plain

DATA = Path(__file__).resolve().parents[1] / "data"
WEB = Path(__file__).resolve().parents[1] / "src" / "web"
PASSWORD = secrets.token_urlsafe(12)            # made up per test run: no password is written in the code
ROLLBACK = ("How do I revert to an earlier revision? The release we put out this morning is causing errors and I "
            "would rather roll back than try to fix forward.")
JARGON = ("RAG", "Tier ", "tier_", "intent", "threshold", "extractive", "lexical", "passage", "generator",
          "REPLY_MODE", "guardrail", "Degraded")


@pytest.fixture(scope="module")
def parts():
    dev = load_tickets(DATA / "development_tickets.json")
    corpus = Corpus.from_file(DATA / "documentation.json")
    retriever = Retriever(corpus)
    rows = score_tickets(retriever.fallback, corpus, _labelled(dev))
    retriever.thresholds["lexical"] = th = best_threshold(rows)
    retriever.thresholds["lexical_approved"] = approved_threshold_from(rows, th)
    must, docs = _label_sets(dev)
    router = Router(RoutingConfig(confidence_threshold=0.5, must_escalate_intents=must, intent_docs=docs,
                                  use_intent_hint=True))
    return Classifier().fit(dev), retriever, router, corpus


@pytest.fixture
def app(tmp_path, parts, monkeypatch):
    monkeypatch.setenv("KILL_SWITCH_FILE", str(tmp_path / "KS"))
    monkeypatch.delenv("KILL_SWITCH", raising=False)
    monkeypatch.delenv("REPLY_MODE", raising=False)                 # review mode: agents send every reply
    monkeypatch.setenv("REVIEW_DB", str(tmp_path / "reviews.db"))
    monkeypatch.setenv("TICKETS_DB", str(tmp_path / "tickets.db"))
    staff = StaffDirectory(tmp_path / "staff.json")
    staff.add("boss@cloudserve.example", "Marcus", "Support Operations", "supervisor", PASSWORD)
    staff.add("asha@cloudserve.example", "Asha", "API Support", "agent", PASSWORD)
    staff.add("ravi@cloudserve.example", "Ravi", "Deployments", "agent", PASSWORD)
    classifier, retriever, router, corpus = parts
    p = A.SupportPipeline(classifier, retriever, router, Generator(None, unavailable_reason="no key"),
                          Guardrails(corpus=corpus), DecisionStore(f"sqlite:///{tmp_path / 'decisions.db'}"),
                          A.Metrics())
    application = A.create_app(p)
    application.state.staff = staff
    application.state.hub = SupportHub(p)
    yield application
    p.store.close()


def client_as(app, email):
    c = TestClient(app)
    r = c.post("/staff/login", json={"email": email, "password": PASSWORD})
    assert r.status_code == 200, r.text
    return c


def chat(text, session):
    return {"session_id": session, "visitor": {"name": "Kavya Patel", "customer_id": "CUST-1177"},
            "messages": [{"from": "visitor", "text": text}]}


# ================================================================= accounts and sign-in
def test_passwords_are_stored_only_as_salted_hashes(tmp_path):
    s = StaffDirectory(tmp_path / "staff.json")
    s.add("a@cloudserve.example", "A", "T", "agent", PASSWORD)
    raw = (tmp_path / "staff.json").read_text()
    assert PASSWORD not in raw and json.loads(raw)["a@cloudserve.example"]["hash"]
    assert s.check("a@cloudserve.example", PASSWORD)["role"] == "agent"
    assert s.check("a@cloudserve.example", "wrong-password") is None
    with pytest.raises(ValueError):
        s.add("b@cloudserve.example", "B", "T", "agent", "short")


def test_sign_in_page_and_wrong_password(app):
    c = TestClient(app)
    assert "Staff sign-in" in c.get("/").text
    r = c.post("/staff/login", json={"email": "asha@cloudserve.example", "password": "nope-nope"})
    assert r.status_code == 401 and r.json()["detail"] == "Wrong username or password."
    assert c.get("/staff/me").status_code == 401


def test_each_role_lands_on_its_own_screen_and_cannot_open_the_other(app):
    sup = client_as(app, "boss@cloudserve.example")
    assert sup.get("/", follow_redirects=False).headers["location"] == "/supervisor"
    assert "Ticket queue" in sup.get("/supervisor").text
    agent = client_as(app, "asha@cloudserve.example")
    assert agent.get("/", follow_redirects=False).headers["location"] == "/agent"
    assert "My tickets" in agent.get("/agent").text
    assert agent.get("/supervisor", follow_redirects=False).headers["location"] == "/"
    assert agent.get("/staff/queue").status_code == 403 and agent.get("/staff/reports").status_code == 403
    assert TestClient(app).get("/agent", follow_redirects=False).headers["location"] == "/"
    cookie = agent.cookies.get(COOKIE)
    assert cookie and agent.post("/staff/logout").status_code == 200
    assert agent.get("/staff/me").status_code == 401


# ================================================================= supervisor: queue, assignment, reports
def test_supervisor_sees_every_channel_and_assigns(app):
    sup = client_as(app, "boss@cloudserve.example")
    assert sup.post("/staff/sample?count=8").json()["added"] >= 8
    queue = sup.get("/staff/queue?tab=unassigned").json()["tickets"]
    assert queue and {t["channel"] for t in queue} >= {"Email", "Live chat"}
    assert all(t["ticket_id"].startswith("CS-") and t["priority"] in ("High", "Medium", "Low") for t in queue)
    agents = sup.get("/staff/agents").json()["agents"]
    assert [a["name"] for a in agents] == ["Asha", "Ravi"]
    tid = queue[0]["ticket_id"]
    assert sup.post(f"/staff/tickets/{tid}/assign", json={"agent": "asha@cloudserve.example"}).status_code == 200
    assert tid not in [t["ticket_id"] for t in sup.get("/staff/queue?tab=unassigned").json()["tickets"]]
    assigned = sup.get("/staff/queue?tab=assigned").json()["tickets"]
    assert assigned[0]["assignee"] == "Asha · API Support"
    assert sup.post(f"/staff/tickets/{tid}/assign", json={"agent": "boss@cloudserve.example"}).status_code == 422
    log = app.state.hub.pipeline.store.export(ticket_id=tid)
    assert log[-1]["stage"] == "assignment" and "Marcus assigned the ticket to Asha" in log[-1]["reason"]


def test_reports_volume_channels_and_decision_log(app):
    sup = client_as(app, "boss@cloudserve.example")
    sup.post("/staff/sample?count=8")
    r = sup.get("/staff/reports").json()
    assert r["total"] >= 8 and len(r["volume"]) == 14 and r["volume"][-1]["tickets"] == r["total"]
    assert sum(c["tickets"] for c in r["by_channel"]) == r["total"]
    log = r["decision_log"]
    assert log["every_ticket_logged"] and log["tickets"] == r["total"] and len(log["tickets_log"]) == r["total"]
    tid = log["tickets_log"][0]["ticket_id"]
    steps = sup.get(f"/staff/tickets/{tid}/steps").json()["steps"]
    assert steps[0]["step"] == "Duplicate check" and any(s["step"] == "Result" for s in steps)
    text = json.dumps(r) + json.dumps(steps)
    assert not [w for w in JARGON if w in text], [w for w in JARGON if w in text]


# ================================================================= agent: own tickets, check, send
def assigned_ticket(app, text=ROLLBACK, session="s1", to="asha@cloudserve.example"):
    hub = app.state.hub
    res = hub.receive("chat", chat(text, session))
    hub.assign(res["ticket_id"], app.state.staff.get(to), app.state.staff.get("boss@cloudserve.example"))
    return res["ticket_id"]


def test_agent_sees_only_their_tickets_with_cirrus_findings_and_a_cited_reply(app):
    mine = assigned_ticket(app)
    other = assigned_ticket(app, "Webhook signature verification fails for every delivery.", "s2",
                            to="ravi@cloudserve.example")
    agent = client_as(app, "asha@cloudserve.example")
    assert [t["ticket_id"] for t in agent.get("/staff/my-tickets").json()["tickets"]] == [mine]
    assert agent.get(f"/staff/tickets/{other}").status_code == 403
    t = agent.get(f"/staff/tickets/{mine}").json()
    assert t["can_send"] and "Cirrus found the answer in the help articles" in t["findings"]
    assert t["articles"][0]["doc_id"] == "DOC-DEPLOY-002" and t["articles"][0]["url"].startswith("https://")
    assert "[1]" in t["reply"] and "Rolling back a failed release" in t["reply"]          # cited, with sources
    assert not [w for w in JARGON if w in json.dumps({k: t[k] for k in ("findings", "trail", "handled")})]


def test_agent_sends_the_checked_reply_and_it_is_logged(app):
    tid = assigned_ticket(app)
    agent = client_as(app, "asha@cloudserve.example")
    reply = agent.get(f"/staff/tickets/{tid}").json()["reply"]
    r = agent.post(f"/staff/tickets/{tid}/send", json={"text": reply})
    assert r.status_code == 200 and r.json() == {"sent": True, "edited": False, "flagged": []}
    t = agent.get(f"/staff/tickets/{tid}").json()
    assert not t["can_send"] and t["sent_reply"] == reply and t["work_status"] == "Resolved"
    review = [d for d in app.state.hub.pipeline.store.export(ticket_id=tid) if d["stage"] == "human_review"]
    assert review[0]["action_taken"] == "sent_after_review" and review[0]["prediction"]["value"] == "Asha"
    assert agent.post(f"/staff/tickets/{tid}/send", json={"text": reply}).status_code == 409   # not twice


def test_edited_replies_are_checked_and_saved_drafts_kept(app):
    tid = assigned_ticket(app)
    agent = client_as(app, "asha@cloudserve.example")
    agent.post(f"/staff/tickets/{tid}/save", json={"text": "Draft in progress"})
    t = agent.get(f"/staff/tickets/{tid}").json()
    assert t["reply"] == "Draft in progress" and t["work_status"] == "In progress"
    leak = agent.post(f"/staff/tickets/{tid}/send", json={"text": "Use key sk-live-ABCDEF1234567890ABCDEF12"})
    assert leak.status_code == 422 and "private_data" in leak.json()["detail"]
    ok = agent.post(f"/staff/tickets/{tid}/send", json={"text": "Roll back from the revisions page, then redeploy."})
    assert ok.json()["edited"] is True


def test_mark_resolved_without_a_reply_and_status_changes(app):
    tid = assigned_ticket(app)
    agent = client_as(app, "asha@cloudserve.example")
    assert agent.post(f"/staff/tickets/{tid}/status", json={"status": "in_progress"}).status_code == 200
    assert agent.post(f"/staff/tickets/{tid}/status", json={"status": "bogus"}).status_code == 422
    assert agent.post(f"/staff/tickets/{tid}/resolve", json={"note": "solved on a call"}).status_code == 200
    assert agent.get(f"/staff/tickets/{tid}").json()["work_status"] == "Resolved"
    log = app.state.hub.pipeline.store.export(ticket_id=tid)
    assert log[-1]["action_taken"] == "resolved_without_reply"


def test_only_the_assigned_agent_can_act(app):
    tid = assigned_ticket(app)
    ravi = client_as(app, "ravi@cloudserve.example")
    assert ravi.post(f"/staff/tickets/{tid}/send", json={"text": "hi there"}).status_code == 403
    sup = client_as(app, "boss@cloudserve.example")
    assert sup.get(f"/staff/tickets/{tid}").status_code == 200                   # supervisors can look
    assert sup.post(f"/staff/tickets/{tid}/send", json={"text": "hi there"}).status_code == 403


# ================================================================= plain words, safe pages
def test_plain_wording():
    assert plain("Tier 2: RAG generation: the best article scored 0.259, relevant but below the 0.292 "
                 "approved-content threshold.") == "Reply written from the help articles: based on how closely the " \
                                                   "help article matches."
    assert plain("Escalated to a person. No help article matched closely enough (best match scored 0.13; at least "
                 "0.23 is needed).") == "No help article matched closely enough."


def test_screens_escape_ticket_text_and_use_no_jargon():
    for name in ("login.html", "supervisor.html", "agent.html", "app.js"):
        text = (WEB / name).read_text()
        assert "RAG" not in text and "intent" not in text.lower().replace("intention", "")
    assert "const esc = " in (WEB / "app.js").read_text()
