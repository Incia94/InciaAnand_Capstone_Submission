"""The support resolution hub: unified schema, four channels, channel-aware formatting, central
storage, cross-channel deduplication, conversation history, the employee dashboard, decision logging."""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src import api as A
from src.channels import CHANNELS, render, simulate, to_unified
from src.classify import Classifier
from src.formatting import format_reply, split_reply
from src.generate import Generator
from src.guardrail import Guardrails
from src.hub import SupportHub, TicketStore, similarity
from src.ingest import load_tickets, normalise_ticket
from src.logging_store import DecisionStore
from src.retrieve import Corpus, Retriever, _labelled, approved_threshold_from, best_threshold, score_tickets
from src.route import Router, RoutingConfig, _label_sets

DATA = Path(__file__).resolve().parents[1] / "data"
VAL = json.loads((DATA / "validation_tickets.json").read_text())
ROLLBACK = ("How do I revert to an earlier revision? The release we put out this morning is causing errors and I "
            "would rather roll back than try to fix forward.")
SECURITY = ("A former employee still has access three weeks after leaving. We have found API calls made under "
            "their account. This is urgent.")
REPLY = ("Thanks for getting in touch.\n\nOpen the revisions page and pick the last healthy revision. [1] Select Roll "
         "back. [1] Confirm the health checks pass. [2] Redeploy the fix. [1]\n\nIf this does not resolve it, reply "
         "to this message and a member of the team will pick it up.\n\nThis reply was prepared automatically from "
         "the CloudServe help articles listed below.\n\nSources:\n[1] Rolling back a failed release (DOC-DEPLOY-002)"
         "\n[2] Container deployments failing during the health check phase (DOC-DEPLOY-001)")


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
def hub(tmp_path, parts, monkeypatch):
    monkeypatch.setenv("KILL_SWITCH_FILE", str(tmp_path / "KS"))
    monkeypatch.delenv("KILL_SWITCH", raising=False)
    monkeypatch.setenv("REVIEW_DB", str(tmp_path / "reviews.db"))
    monkeypatch.setenv("TICKETS_DB", str(tmp_path / "tickets.db"))
    classifier, retriever, router, corpus = parts
    p = A.SupportPipeline(classifier, retriever, router, Generator(None, unavailable_reason="no key"),
                          Guardrails(corpus=corpus), DecisionStore(f"sqlite:///{tmp_path / 'decisions.db'}"),
                          A.Metrics())
    yield SupportHub(p)
    p.store.close()


def email(text, sender="Kavya Patel <kavya.patel@example.com>", cid="CUST-1177", mid="<m1@mail>", subject="Help"):
    return {"message_id": mid, "from": sender, "subject": subject, "text": text,
            "account": {"customer_id": cid, "tier": "business", "region": "europe", "fluency": "fluent"}}


def chat(text, session="s1", cid="CUST-1177", history=()):
    return {"session_id": session, "visitor": {"name": "Kavya Patel", "email": "kavya.patel@example.com",
                                               "customer_id": cid},
            "messages": [*history, {"from": "visitor", "text": text}]}


# ================================================================= unified schema + four channels
def test_every_channel_maps_onto_the_unified_schema():
    t = normalise_ticket(to_unified("email", email(ROLLBACK)))
    assert (t.channel.value, t.customer_id, t.customer_email, t.thread_id) == (
        "email", "CUST-1177", "kavya.patel@example.com", "<m1@mail>")
    assert t.customer_name == "Kavya Patel" and "Kavya" not in t.text and "example.com" not in t.text
    c = normalise_ticket(to_unified("chat", chat("still broken", history=[
        {"from": "visitor", "text": "my deploy fails"}, {"from": "bot", "text": "Try a rollback [1]"}])))
    assert c.body == "still broken" and [m["role"] for m in c.conversation] == ["customer", "cirrus"]
    assert c.thread_id == "s1"
    forum = {"thread_id": "th9", "post_id": "p1", "title": "Webhooks not arriving", "body": "Signature check fails",
             "author": {"username": "elif"}}
    f = normalise_ticket(to_unified("forum", forum))
    assert (f.channel.value, f.subject, f.thread_id) == ("forum", "Webhooks not arriving", "th9")
    comment = {"comment_id": "c1", "page": {"doc_id": "DOC-API-001", "title": "Rate limits"},
               "comment": "Is search limited?"}
    d = normalise_ticket(to_unified("docs_comment", comment))
    assert d.channel.value == "docs_comment" and d.subject == "Comment on: Rate limits"


def test_customer_key_links_channels_without_exposing_who():
    a = normalise_ticket(to_unified("email", email("x" * 20)))
    b = normalise_ticket(to_unified("chat", chat("y" * 20)))
    assert a.customer_key == b.customer_key and "1177" not in a.customer_key and len(a.customer_key) == 16


def test_malformed_channel_payloads_are_still_tickets():
    assert normalise_ticket(to_unified("email", "not json")).processable
    assert not normalise_ticket(to_unified("chat", {"messages": []})).processable


def test_simulated_traffic_covers_four_channels_a_duplicate_and_a_follow_up():
    stream = simulate(12, seed=7)
    assert {ch for ch, _ in stream} == set(CHANNELS) and len(stream) == 14
    sessions = [p["session_id"] for ch, p in stream if ch == "chat"]
    assert len(sessions) != len(set(sessions))                    # a follow-up in an existing session
    for t in VAL[:8]:
        ch, payload = render(t)
        assert normalise_ticket(to_unified(ch, payload)).customer_id == t["customer_id"]


# ================================================================= channel-aware formatting
def test_one_reply_formatted_for_each_channel():
    answer, sources, _ = split_reply(REPLY)
    assert [s[2] for s in sources] == ["DOC-DEPLOY-002", "DOC-DEPLOY-001"]
    e = format_reply("email", REPLY, ticket_id="CS-1", subject="Bad release")
    assert e["subject"] == "Re: Bad release" and e["body"].startswith("Hello,")
    assert "https://docs.cloudserve.example/articles/DOC-DEPLOY-002" in e["body"]
    assert "Ticket reference: CS-1" in e["body"]
    c = format_reply("chat", REPLY)
    assert "Thanks for getting in touch" not in c["body"] and "Redeploy the fix" not in c["body"]   # first 3 steps
    assert "](https://docs.cloudserve.example/articles/DOC-DEPLOY-002)" in c["body"]
    f = format_reply("forum", REPLY, ticket_id="CS-1")
    assert "CS-1" not in f["body"] and "**Sources**" in f["body"] and "moderator" in f["body"]
    d = format_reply("docs_comment", REPLY)
    assert "Select Roll back" in d["body"] and "Confirm the health checks" not in d["body"] and "See: [" in d["body"]
    assert format_reply("chat", "Hello!", fixed=True)["body"] == "Hello!"


# ================================================================= common API + central storage
def test_each_channel_endpoint_processes_stores_and_logs(hub):
    with TestClient(A.create_app(hub.pipeline)) as client:
        client.app.state.hub = hub
        for ch, payload in (("email", email(ROLLBACK)), ("chat", chat("Tell me a joke")),
                            ("forum", {"post_id": "p1", "title": "Webhooks", "body": "Webhook signature verification "
                                       "fails for every delivery.", "author": {"username": "elif"}}),
                            ("docs_comment", {"comment_id": "c1", "comment": "Are search endpoints rate limited "
                                              "differently? We get 429 on search."})):
            r = client.post(f"/channels/{ch}", json=payload)
            assert r.status_code == 200 and r.json()["channel"] == ch, r.text
            tid = r.json()["ticket_id"]
            stored = hub.tickets.get(tid)
            assert stored and stored["channel"] == ch and hub.tickets.messages(tid)
            assert [d["stage"] for d in hub.pipeline.store.export(ticket_id=tid)][-1] in ("outcome", "human_review")
        assert client.post("/channels/sms", json={}).status_code == 404
        assert len(client.get("/hub/tickets").json()["tickets"]) == 4


def test_post_tickets_still_returns_the_outcome_and_now_stores_centrally(hub):
    with TestClient(A.create_app(hub.pipeline)) as client:
        client.app.state.hub = hub
        body = client.post("/tickets", json={"ticket_id": "U-1", "channel": "chat", "body": ROLLBACK}).json()
        assert body["ticket_id"] == "U-1" and body["outcome"] in ("sent", "pending_review")
        assert hub.tickets.get("U-1")["status"] in ("answered", "pending_review")


def test_storage_masks_personal_data_in_text(hub):
    hub.receive("email", email(ROLLBACK + " Call me on +44 20 7946 0958 or kavya.patel@example.com."))
    t = hub.tickets.list()[0]
    assert "kavya.patel@example.com" not in t["text"] and "7946" not in t["text"]
    assert t["customer_name"] == "Kavya Patel"                     # kept, in its own field, for staff


# ================================================================= cross-channel deduplication
def test_same_open_problem_on_another_channel_is_merged(hub):
    first = hub.receive("email", email(SECURITY, mid="<sec@mail>"), run_id="DUP")
    assert first["status"] == "escalated"
    second = hub.receive("chat", chat(SECURITY, session="s-dup"), run_id="DUP")
    assert second["status"] == "merged" and second["duplicate_of"] == first["ticket_id"] and second["tier"] == 0
    assert first["ticket_id"] in second["reply"]["body"]
    assert any("(also sent on chat)" in m["text"] for m in hub.tickets.messages(first["ticket_id"]))
    trail = hub.pipeline.store.export(ticket_id=second["ticket_id"])
    assert [r["stage"] for r in trail] == ["ingestion", "deduplication", "outcome"]
    assert trail[1]["action_taken"] == "merged" and trail[1]["prediction"]["value"] == first["ticket_id"]
    rep = hub.pipeline.store.reconcile("DUP", [first["ticket_id"], second["ticket_id"]])
    assert rep["reconciled"] and rep["outcomes"] == {"escalated": 1, "merged": 1}


def test_not_merged_different_customer_different_problem_or_closed(hub):
    first = hub.receive("email", email(SECURITY, mid="<a@mail>"))
    other = hub.receive("chat", chat(SECURITY, session="s-x", cid="CUST-9999"))
    assert other["status"] != "merged"                                         # someone else
    diff = hub.receive("chat", chat("How do I add a team member to my project?", session="s-y"))
    assert diff["status"] != "merged"                                          # a different problem
    hub.tickets.update(first["ticket_id"], status="resolved")
    again = hub.receive("forum", {"post_id": "p-z", "title": "Former employee", "body": SECURITY,
                                  "author": {"customer_id": "CUST-1177", "username": "kavya"}})
    assert again["status"] != "merged"                                         # the original is closed
    assert similarity(SECURITY, SECURITY) > 0.99 and similarity(SECURITY, ROLLBACK) < 0.5


# ================================================================= conversation history
def test_chat_messages_in_one_session_are_separate_tickets_and_redelivery_is_idempotent():
    a = to_unified("chat", chat("first message", session="s"))["ticket_id"]
    b = to_unified("chat", chat("second message", session="s"))["ticket_id"]
    assert a != b and a == to_unified("chat", chat("first message", session="s"))["ticket_id"]


def test_a_follow_up_in_the_same_thread_carries_the_history(hub, monkeypatch):
    monkeypatch.setenv("REPLY_MODE", "auto")
    first = hub.receive("chat", chat(ROLLBACK, session="s-h"))
    assert first["status"] == "answered"
    follow = hub.receive("chat", chat("Thanks. I tried that and it is still not working.", session="s-h"))
    assert follow["history"] == 2                                              # the question and Cirrus's answer
    assert follow["status"] == "escalated"                                     # the answer did not work: a person
    trail = hub.pipeline.store.export(ticket_id=follow["ticket_id"])
    assert any("did not solve it" in r["reason"] for r in trail)
    view = hub.ticket_view(follow["ticket_id"])
    assert any(m["ticket_id"] == first["ticket_id"] for m in view["customer_history"])


def test_customer_history_spans_channels(hub):
    hub.receive("email", email(ROLLBACK, mid="<h1@mail>"))
    hub.receive("forum", {"post_id": "p-h", "title": "Rate limits", "body": "Search requests get 429 constantly.",
                          "author": {"customer_id": "CUST-1177", "username": "kavya"}})
    key = normalise_ticket(to_unified("email", email("x"))).customer_key
    assert {m["channel"] for m in hub.tickets.history(key)} == {"email", "forum"}


# ================================================================= the hub's data API
@pytest.mark.review_mode
def test_hub_data_api(hub):
    with TestClient(A.create_app(hub.pipeline)) as client:
        client.app.state.hub = hub
        results = client.post("/hub/simulate?count=8").json()["results"]
        assert {r["channel"] for r in results} == set(CHANNELS)
        summary = client.get("/hub/summary").json()
        assert summary["tickets"] == len(results) and set(summary["by_channel"]) == set(CHANNELS)
        assert all(r["ticket_id"].startswith("CS-") for r in results)
        held = next(r for r in results if r["status"] == "pending_review")
        view = client.get(f"/hub/tickets/{held['ticket_id']}").json()
        assert view["review"]["status"] == "pending_review" and view["formatted"]["body"]
        assert view["retrieved"] and view["decisions"] and view["messages"]
        r = client.post(f"/reviews/{held['ticket_id']}/approve", json={"reviewer": "Sofia Restrepo"})
        assert r.status_code == 200 and r.json()["formatted"]["channel"] == held["channel"]
        after = client.get(f"/hub/tickets/{held['ticket_id']}").json()
        assert after["status"] == "resolved" and after["resolved_by"] == "Sofia Restrepo"
        assert after["decisions"][-1]["stage"] == "human_review"
        key = after["customer_key"]
        assert client.get(f"/customers/{key}/history").json()["messages"]


def test_channel_tickets_get_cloudserve_numbers_and_redelivery_is_one_ticket(hub):
    first = hub.receive("email", email(ROLLBACK, mid="<n1@mail>"))
    second = hub.receive("email", email(SECURITY, mid="<n2@mail>"))
    assert (first["ticket_id"], second["ticket_id"]) == ("CS-1001", "CS-1002")
    again = hub.receive("email", email(ROLLBACK, mid="<n1@mail>"))
    assert again["ticket_id"] == "CS-1001" and again["duplicate_delivery"]
    assert hub.tickets.get("CS-1001")["source_ref"] == "email:<n1@mail>"


def test_ticket_store_survives_restart(tmp_path):
    s = TicketStore(tmp_path / "t.db")
    s.save(ticket_id="X", channel="email", customer_key="k", status="escalated", text="hello", thread_key="email:X")
    s.add_message("X", "email:X", "k", "email", "customer", "hello")
    s2 = TicketStore(tmp_path / "t.db")
    assert s2.get("X")["status"] == "escalated" and s2.history("k")[0]["text"] == "hello"
