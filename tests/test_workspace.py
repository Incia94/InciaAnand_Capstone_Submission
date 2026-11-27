"""The support workspace's rules and views (src/hub.py, src/workspace.py, src/staff.py).

The screens are the Streamlit app (tests/test_streamlit_app.py drives it). These tests check the rules
underneath, which hold whatever screen calls them: a supervisor sees every ticket and assigns it; an
agent acts only on their own tickets, checks Cirrus's cited reply and sends it; everything is logged;
and nothing shown to staff uses the pipeline's technical vocabulary."""
import json
from pathlib import Path

import pytest

from src import api as A
from src import workspace as W
from src.classify import Classifier
from src.generate import Generator
from src.guardrail import Guardrails
from src.hub import SupportHub
from src.ingest import load_tickets
from src.logging_store import DecisionStore
from src.retrieve import Corpus, Retriever, _labelled, approved_threshold_from, best_threshold, score_tickets
from src.review import ReviewError
from src.route import Router, RoutingConfig, _label_sets
from src.staff import StaffDirectory
from tests import fakes

DATA = Path(__file__).resolve().parents[1] / "data"
PASSWORD = fakes.password()                      # made up per test run: no password is written in the code
WRONG = fakes.password()
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
def ws(tmp_path, parts, monkeypatch):
    """A hub in review mode (agents send every reply) and three staff accounts."""
    monkeypatch.setenv("KILL_SWITCH_FILE", str(tmp_path / "KS"))
    monkeypatch.delenv("KILL_SWITCH", raising=False)
    monkeypatch.delenv("REPLY_MODE", raising=False)
    monkeypatch.setenv("REVIEW_DB", str(tmp_path / "reviews.db"))
    monkeypatch.setenv("TICKETS_DB", str(tmp_path / "tickets.db"))
    staff = StaffDirectory(tmp_path / "staff.json")
    boss = staff.add("boss@cloudserve.example", "Marcus", "Support Operations", "supervisor", PASSWORD)
    asha = staff.add("asha@cloudserve.example", "Asha", "API Support", "agent", PASSWORD)
    ravi = staff.add("ravi@cloudserve.example", "Ravi", "Deployments", "agent", PASSWORD)
    classifier, retriever, router, corpus = parts
    p = A.SupportPipeline(classifier, retriever, router, Generator(None, unavailable_reason="no key"),
                          Guardrails(corpus=corpus), DecisionStore(f"sqlite:///{tmp_path / 'decisions.db'}"),
                          A.Metrics())
    yield SupportHub(p), staff, boss, asha, ravi
    p.store.close()


def chat(text, session):
    return {"session_id": session, "visitor": {"name": "Kavya Patel", "customer_id": "CUST-1177"},
            "messages": [{"from": "visitor", "text": text}]}


def assigned(hub, boss, agent, text=ROLLBACK, session="s1"):
    tid = hub.receive("chat", chat(text, session))["ticket_id"]
    hub.assign(tid, agent, boss)
    return tid


# ================================================================= accounts
def test_passwords_are_stored_only_as_salted_hashes(tmp_path):
    s = StaffDirectory(tmp_path / "staff.json")
    s.add("a@cloudserve.example", "A", "T", "agent", PASSWORD)
    raw = (tmp_path / "staff.json").read_text()
    assert PASSWORD not in raw and json.loads(raw)["a@cloudserve.example"]["hash"]
    assert s.check("a@cloudserve.example", PASSWORD)["role"] == "agent"
    assert s.check("a", PASSWORD)["role"] == "agent"                       # username = the part before @
    assert s.check("a@cloudserve.example", WRONG) is None
    with pytest.raises(ValueError):
        s.add("b@cloudserve.example", "B", "T", "agent", "short")


# ================================================================= supervisor: queue, assignment, reports
def test_supervisor_sees_every_channel_and_assigns(ws):
    hub, staff, boss, asha, ravi = ws
    hub.simulate(count=8)
    queue = W.queue(hub, "unassigned")
    assert queue and {t["channel"] for t in queue} >= {"Email", "Live chat"}
    assert all(t["ticket_id"].startswith("CS-") and t["priority"] in ("High", "Medium", "Low") for t in queue)
    assert [a["name"] for a in staff.list("agent")] == ["Asha", "Ravi"]
    tid = queue[0]["ticket_id"]
    hub.assign(tid, asha, boss)
    assert tid not in [t["ticket_id"] for t in W.queue(hub, "unassigned")]
    assert W.queue(hub, "assigned")[0]["assignee"] == "Asha · API Support"
    log = hub.pipeline.store.export(ticket_id=tid)
    assert log[-1]["stage"] == "assignment" and "Marcus assigned the ticket to Asha" in log[-1]["reason"]
    with pytest.raises(LookupError):
        hub.assign("CS-9999", asha, boss)


def test_reports_volume_channels_and_decision_log(ws):
    hub = ws[0]
    hub.simulate(count=8)
    r = W.reports(hub)
    assert r["total"] >= 8 and len(r["volume"]) == 14 and r["volume"][-1]["tickets"] == r["total"]
    assert sum(c["tickets"] for c in r["by_channel"]) == r["total"]
    log = r["decision_log"]
    assert log["every_ticket_logged"] and log["tickets"] == r["total"] and len(log["tickets_log"]) == r["total"]
    steps = W.ticket_steps(hub, log["tickets_log"][0]["ticket_id"])
    assert steps[0]["step"] == "Duplicate check" and any(s["step"] == "Result" for s in steps)
    text = json.dumps(r) + json.dumps(steps)
    assert not [w for w in JARGON if w in text], [w for w in JARGON if w in text]


# ================================================================= agent: own tickets, check, send
def test_agent_sees_their_tickets_with_cirrus_findings_and_a_cited_reply(ws):
    hub, staff, boss, asha, ravi = ws
    mine = assigned(hub, boss, asha)
    assigned(hub, boss, ravi, "Webhook signature verification fails for every delivery.", "s2")
    assert [t["ticket_id"] for t in hub.tickets.assigned_to(asha["email"])] == [mine]
    t = W.agent_view(hub, mine)
    assert t["can_send"] and "Cirrus found the answer in the help articles" in t["findings"]
    assert t["articles"][0]["doc_id"] == "DOC-DEPLOY-002" and t["articles"][0]["url"].startswith("https://")
    assert "[1]" in t["reply"] and "Rolling back a failed release" in t["reply"]          # cited, with sources
    assert not [w for w in JARGON if w in json.dumps({k: t[k] for k in ("findings", "trail", "handled")})]


def test_agent_sends_the_checked_reply_and_it_is_logged(ws):
    hub, staff, boss, asha, ravi = ws
    tid = assigned(hub, boss, asha)
    reply = W.agent_view(hub, tid)["reply"]
    res = hub.send(tid, asha, reply)
    assert res["edited"] is False
    t = W.agent_view(hub, tid)
    assert not t["can_send"] and t["sent_reply"] == reply and t["work_status"] == "Resolved"
    review = [d for d in hub.pipeline.store.export(ticket_id=tid) if d["stage"] == "human_review"]
    assert review[0]["action_taken"] == "sent_after_review" and review[0]["prediction"]["value"] == "Asha"
    with pytest.raises(ReviewError, match="already"):
        hub.send(tid, asha, reply)                                                   # not twice


def test_edited_replies_are_checked_and_saved_drafts_kept(ws):
    hub, staff, boss, asha, ravi = ws
    tid = assigned(hub, boss, asha)
    hub.save_reply(tid, "Draft in progress", asha)
    t = W.agent_view(hub, tid)
    assert t["reply"] == "Draft in progress" and t["work_status"] == "In progress"
    with pytest.raises(ReviewError, match="private_data"):
        hub.send(tid, asha, f"Use key {fakes.api_key(24)}")
    assert hub.send(tid, asha, "Roll back from the revisions page, then redeploy.")["edited"] is True


def test_mark_resolved_without_a_reply_and_status_changes(ws):
    hub, staff, boss, asha, ravi = ws
    tid = assigned(hub, boss, asha)
    hub.set_work_status(tid, "in_progress", asha)
    with pytest.raises(ValueError):
        hub.set_work_status(tid, "bogus", asha)
    hub.resolve_without_reply(tid, asha, "solved on a call")
    assert W.agent_view(hub, tid)["work_status"] == "Resolved"
    assert hub.pipeline.store.export(ticket_id=tid)[-1]["action_taken"] == "resolved_without_reply"


def test_only_the_assigned_agent_can_act(ws):
    hub, staff, boss, asha, ravi = ws
    tid = assigned(hub, boss, asha)
    for act in (lambda: hub.send(tid, ravi, "hi there"), lambda: hub.save_reply(tid, "hi", ravi),
                lambda: hub.resolve_without_reply(tid, ravi), lambda: hub.set_work_status(tid, "in_progress", ravi),
                lambda: hub.send(tid, boss, "hi there")):
        with pytest.raises(PermissionError):
            act()
    assert W.agent_view(hub, tid)["can_send"]                                      # nothing changed


# ================================================================= plain words
def test_plain_wording():
    assert W.plain("Tier 2: RAG generation: the best article scored 0.259, relevant but below the 0.292 "
                   "approved-content threshold.") == "Reply written from the help articles: based on how closely " \
                                                     "the help article matches."
    assert W.plain("Escalated to a person. No help article matched closely enough (best match scored 0.13; at "
                   "least 0.23 is needed).") == "No help article matched closely enough."


def test_streamlit_screens_use_no_jargon():
    text = (Path(__file__).resolve().parents[1] / "streamlit_app.py").read_text()
    shown = " ".join(s for s in __import__("re").findall(r'"([^"\n]*)"', text))
    assert "RAG" not in shown and " intent" not in shown.lower()
