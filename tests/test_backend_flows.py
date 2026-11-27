"""The three things the backend must do well, end to end:

  1 intelligent documentation search, with citations that resolve to real articles;
  2 an automated, cited reply that a person reviews, edits if needed, and then sends;
  3 decision logging: every step of every ticket, the review included, recorded and reconciled.

These run the real pipeline (classifier, hybrid-capable retriever in its keyword mode, router,
guardrails, decision log) with no language model, through Python, the HTTP API and the command line.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src import api as A
from src.classify import Classifier
from src.generate import Generator
from src.guardrail import Guardrails
from src.ingest import load_tickets
from src.logging_store import DecisionStore
from src.retrieve import Corpus, Retriever, _labelled, approved_threshold_from, best_threshold, cited_results, \
    score_tickets
from src.review import ReviewError, approve, reject
from src.route import Router, RoutingConfig, _label_sets
from tests import fakes

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ROLLBACK = {"ticket_id": "R-1", "channel": "chat", "customer_name": "Asha Rao", "customer_id": "CUST-1042",
            "customer_tier": "business", "customer_region": "europe", "language_fluency": "fluent",
            "body": "How do I revert to an earlier revision? The release we put out this morning is causing errors "
                    "and I would rather roll back than try to fix forward. You can reach me at asha.rao@example.com."}
SECURITY = {"ticket_id": "S-1", "channel": "email", "subject": "Possible account compromise",
            "body": "A former employee appears to still have access three weeks after leaving. We have found API "
                    "calls made under their account. This is urgent."}


@pytest.fixture(scope="module")
def corpus():
    return Corpus.from_file(DATA / "documentation.json")


@pytest.fixture(scope="module")
def parts(corpus):
    """Trained once for the file: the classifier, the calibrated retriever and the router."""
    dev = load_tickets(DATA / "development_tickets.json")
    retriever = Retriever(corpus)
    rows = score_tickets(retriever.fallback, corpus, _labelled(dev))
    retriever.thresholds["lexical"] = th = best_threshold(rows)
    retriever.thresholds["lexical_approved"] = approved_threshold_from(rows, th)
    must, docs = _label_sets(dev)
    router = Router(RoutingConfig(confidence_threshold=0.5, must_escalate_intents=must, intent_docs=docs,
                                  use_intent_hint=True))
    return Classifier().fit(dev), retriever, router


@pytest.fixture
def pipeline(tmp_path, corpus, parts, monkeypatch):
    """A fresh decision log and review queue for every test."""
    monkeypatch.setenv("KILL_SWITCH_FILE", str(tmp_path / "KS"))
    monkeypatch.delenv("KILL_SWITCH", raising=False)
    monkeypatch.setenv("REVIEW_DB", str(tmp_path / "reviews.db"))
    classifier, retriever, router = parts
    p = A.SupportPipeline(classifier, retriever, router, Generator(None, unavailable_reason="no key"),
                          Guardrails(corpus=corpus), DecisionStore(f"sqlite:///{tmp_path / 'decisions.db'}"),
                          A.Metrics())
    yield p
    p.store.close()


# ================================================================= 1 search with citations
def test_search_returns_numbered_citations_that_resolve(pipeline, corpus):
    res = cited_results(pipeline.retriever, "our search requests keep getting 429 too many requests")
    assert res["status"] == "ok" and res["results"]
    first = res["results"][0]
    assert first["citation"] == "[1]" and first["doc_id"] == "DOC-API-001"
    assert corpus.resolve(first["chunk_id"]) is not None and first["excerpt"]
    assert res["sources"][0] == "[1] Understanding API rate limits and quota tiers (DOC-API-001)"
    assert [x["citation"] for x in res["results"]] == [f"[{i}]" for i in range(1, len(res["results"]) + 1)]


def test_search_understands_wording_and_error_codes(pipeline):
    for query, doc in (("can't get into the console, invalid credentials", "DOC-AUTH-001"),
                       ("webhook signature verification fails", "DOC-API-003"),
                       ("how do I go back to the previous release", "DOC-DEPLOY-002"),
                       ("HTTP 429 on the search endpoint", "DOC-API-001")):
        assert cited_results(pipeline.retriever, query)["results"][0]["doc_id"] == doc, query


def test_search_filters_by_metadata(pipeline):
    q = "my deployment fails at the health check"
    assert cited_results(pipeline.retriever, q, filters={"category": ["deployment"]})["results"]
    assert not cited_results(pipeline.retriever, q, filters={"category": ["billing"]})["results"]
    assert not cited_results(pipeline.retriever, q, filters={"doc_version": ["2.0"]})["results"]


def test_search_cites_nothing_rather_than_noise_and_records_the_gap(pipeline):
    res = cited_results(pipeline.retriever, "how do I bake sourdough bread", store=pipeline.store)
    assert res["results"] == [] and res["sources"] == [] and res["note"]
    gap = [r for r in pipeline.store._rows("WHERE stage = 'search'")]
    assert gap and gap[0]["action_taken"] == "no_result"


def test_every_automated_reply_cites_the_articles_it_uses(pipeline, corpus):
    o = pipeline.process(ROLLBACK)
    assert o.outcome == "sent" and o.sources
    assert "\nSources:\n[1] " in o.reply
    for s in o.sources:
        assert corpus.resolve(s["chunk_id"]) is not None
    markers = {int(n) for n in re.findall(r"\[(\d+)\]", o.reply.split("\nSources:")[0])}
    assert markers and markers <= set(range(1, len(o.sources) + 1))       # every marker points at a source


def test_search_over_http_and_the_command_line(pipeline):
    with TestClient(A.create_app(pipeline)) as c:
        r = c.get("/search", params={"q": "webhook signature verification fails", "category": "api"})
        assert r.status_code == 200 and r.json()["results"][0]["doc_id"] == "DOC-API-003"
    out = subprocess.run([sys.executable, "-m", "src.retrieve", "--backend", "lexical", "search",
                          "rate limit 429 on search"], capture_output=True, text=True, cwd=ROOT, timeout=120)
    assert out.returncode == 0 and "[1] Understanding API rate limits" in out.stdout and "Sources:" in out.stdout


# ================================================================= 2 automated reply, reviewed, then sent
@pytest.mark.review_mode
def test_reply_is_held_until_a_person_approves_it(pipeline):
    o = pipeline.process(ROLLBACK, run_id="REV")
    assert o.outcome == "pending_review" and "Held for a person to approve" in o.reason
    item = pipeline.reviews.get("R-1")
    assert item["status"] == "pending_review" and item["proposed_reply"] == o.reply
    assert item["sources"] and "asha.rao@example.com" not in item["ticket_text"]        # masked in the queue
    res = approve(pipeline.reviews, pipeline.store, pipeline.guardrails, "R-1", "Sofia Restrepo")
    assert res["action"] == "sent_after_review" and res["reply"] == o.reply and not res["edited"]
    assert pipeline.reviews.get("R-1")["status"] == "sent"
    with pytest.raises(ReviewError, match="already handled"):
        approve(pipeline.reviews, pipeline.store, pipeline.guardrails, "R-1", "Someone else")


@pytest.mark.review_mode
def test_reviewer_can_edit_and_the_edit_is_checked(pipeline):
    o = pipeline.process(ROLLBACK)
    edited = o.reply.replace("Thanks for getting in touch.", "Thanks for flagging this, we're on it.")
    res = approve(pipeline.reviews, pipeline.store, pipeline.guardrails, "R-1", "Sofia Restrepo", edited)
    assert res["action"] == "edited_and_sent" and res["edited"] and res["reply"] == edited.strip()


@pytest.mark.review_mode
def test_an_edit_that_leaks_a_secret_or_private_data_is_not_sent(pipeline):
    pipeline.process(ROLLBACK)
    for leak in (f"Use this key instead: {fakes.api_key(32)}",
                 "Contact our other customer at jane.doe@othercompany.com for their setup."):
        with pytest.raises(ReviewError, match="not sent"):
            approve(pipeline.reviews, pipeline.store, pipeline.guardrails, "R-1", "Sofia Restrepo", leak)
    assert pipeline.reviews.get("R-1")["status"] == "pending_review"          # still waiting
    blocked = [r for r in pipeline.store.trail("R-1") if r["stage"] == "review"]
    assert [r["action_taken"] for r in blocked] == ["blocked", "blocked"]


@pytest.mark.review_mode
def test_money_promises_in_an_edit_are_flagged_for_the_reviewer(pipeline):
    pipeline.process(ROLLBACK)
    res = approve(pipeline.reviews, pipeline.store, pipeline.guardrails, "R-1", "Sofia Restrepo",
                  "We will refund you $200 for the trouble. Roll back from the revisions page.")
    assert res["checks"]["warnings"] and any("forbidden_claims" in w for w in res["checks"]["warnings"])


@pytest.mark.review_mode
def test_escalations_wait_with_a_cited_draft_for_the_agent(pipeline):
    o = pipeline.process(SECURITY)
    assert o.outcome == "escalated"
    item = pipeline.reviews.get("S-1")
    assert item["status"] == "needs_agent" and item["kind"] == "escalation"
    assert item["handoff"]["suggested_team"] == "security"
    if item["proposed_reply"]:                                    # a draft was prepared from the articles
        assert "Sources:" in item["proposed_reply"]
    res = approve(pipeline.reviews, pipeline.store, pipeline.guardrails, "S-1", "Priya (security)",
                  "We have disabled the former employee's access and are reviewing the API calls with you.")
    assert res["action"] == "edited_and_sent"


@pytest.mark.review_mode
def test_reject_sends_nothing(pipeline):
    pipeline.process(ROLLBACK)
    res = reject(pipeline.reviews, pipeline.store, "R-1", "Sofia Restrepo", "customer is on a custom contract")
    assert res["status"] == "rejected" and pipeline.reviews.get("R-1")["final_reply"] is None
    assert pipeline.reviews.list() == []


def test_auto_mode_still_sends_without_review(pipeline, monkeypatch):
    monkeypatch.setenv("REPLY_MODE", "auto")
    assert pipeline.process(ROLLBACK).outcome == "sent"
    assert pipeline.reviews.get("R-1") is None


@pytest.mark.review_mode
def test_review_over_http(pipeline):
    with TestClient(A.create_app(pipeline)) as c:
        assert c.post("/tickets", json=ROLLBACK).json()["outcome"] == "pending_review"
        items = c.get("/reviews").json()["items"]
        assert [i["ticket_id"] for i in items] == ["R-1"]
        assert c.post("/reviews/R-1/approve", json={}).status_code == 422             # reviewer required
        r = c.post("/reviews/R-1/approve", json={"reviewer": "Sofia Restrepo"})
        assert r.status_code == 200 and r.json()["status"] == "sent"
        assert c.post("/reviews/R-1/approve", json={"reviewer": "Sofia Restrepo"}).status_code == 409
        assert c.get("/reviews").json()["items"] == []
        assert c.get("/reviews/R-1").json()["reviewer"] == "Sofia Restrepo"


@pytest.mark.review_mode
def test_review_from_the_command_line(pipeline, tmp_path):
    pipeline.process(ROLLBACK)
    env = {"REVIEW_DB": str(tmp_path / "reviews.db"), "DATABASE_URL": f"sqlite:///{tmp_path / 'decisions.db'}",
           "PATH": "/usr/bin:/bin"}

    def cli(*args):
        return subprocess.run([sys.executable, "-m", "src.review", *args], capture_output=True, text=True,
                              cwd=ROOT, timeout=120, env=env)
    assert "R-1" in cli("list").stdout
    shown = cli("show", "R-1").stdout
    assert "Reply Cirrus prepared:" in shown and "Sources:" in shown
    (tmp_path / "reply.txt").write_text("Roll back from the revisions page, then redeploy the fix.")
    out = cli("approve", "R-1", "--reviewer", "Sofia Restrepo", "--text-file", str(tmp_path / "reply.txt"))
    assert out.returncode == 0 and "edited_and_sent" in out.stdout
    assert cli("approve", "R-1", "--reviewer", "Sofia Restrepo").returncode == 1


# ================================================================= 3 decision logging
EXPECTED_STAGES = ["ingest", "screen", "classify", "retrieve", "route", "tier", "generate", "guardrail", "outcome",
                   "review"]


@pytest.mark.review_mode
def test_one_ticket_has_a_complete_ordered_trail_from_ticket_to_sent_reply(pipeline):
    pipeline.process(ROLLBACK, run_id="TRAIL")
    approve(pipeline.reviews, pipeline.store, pipeline.guardrails, "R-1", "Sofia Restrepo")
    trail = pipeline.store.trail("R-1")
    assert [r["stage"] for r in trail] == EXPECTED_STAGES
    for r in trail:                     # the Governance Framework's minimum record, on every row
        assert r["decision_id"] and r["created_at"] and r["ticket_id"] == "R-1" and r["run_id"] == "TRAIL"
        assert r["action_taken"] and r["reason"] and r["model_name"] and r["requirement_ids"]
    by = {r["stage"]: r for r in trail}
    assert by["outcome"]["action_taken"] == "pending_review"
    assert by["review"]["action_taken"] == "sent_after_review" and by["review"]["prediction"] == "Sofia Restrepo"
    assert by["generate"]["sources_used"] and by["review"]["sources_used"]
    assert by["guardrail"]["guardrail_results"]


@pytest.mark.review_mode
def test_personal_data_is_masked_in_every_logged_row(pipeline):
    pipeline.process(ROLLBACK)
    approve(pipeline.reviews, pipeline.store, pipeline.guardrails, "R-1", "Sofia Restrepo")
    raw = json.dumps(pipeline.store.trail("R-1"), default=str)
    assert "asha.rao@example.com" not in raw and "CUST-1042" not in raw and "Asha Rao" not in raw


@pytest.mark.review_mode
def test_rejections_and_blocked_edits_are_logged(pipeline):
    pipeline.process(ROLLBACK)
    with pytest.raises(ReviewError):
        approve(pipeline.reviews, pipeline.store, pipeline.guardrails, "R-1", "Sofia",
                f"key: {fakes.api_key(16, 'XYZ')}")
    reject(pipeline.reviews, pipeline.store, "R-1", "Sofia", "wrong product")
    actions = [r["action_taken"] for r in pipeline.store.trail("R-1") if r["stage"] == "review"]
    assert actions == ["blocked", "rejected"]
    assert fakes.api_key(16, "XYZ") not in json.dumps(pipeline.store.trail("R-1"), default=str)


@pytest.mark.review_mode
def test_a_whole_run_reconciles_one_outcome_per_ticket(pipeline):
    val = json.loads((DATA / "validation_tickets.json").read_text())
    for t in val:
        pipeline.process(t, run_id="VAL")
    rep = pipeline.store.reconcile("VAL", [t["ticket_id"] for t in val])
    assert rep["reconciled"] and rep["tickets_with_outcome"] == 80 and not rep["duplicate_outcome"]
    assert sum(rep["outcomes"].values()) == 80
    assert set(rep["outcomes"]) <= {"pending_review", "escalated", "sent"}     # "sent": fixed Tier 0 replies
    held = pipeline.reviews.list()
    assert len(held) == rep["outcomes"].get("pending_review", 0) + rep["outcomes"].get("escalated", 0)
    assert rep["by_stage"]["ingest"] == rep["by_stage"]["outcome"] == 80


def test_the_log_survives_a_database_failure(tmp_path):
    store = DecisionStore(f"sqlite:///{tmp_path / 'decisions.db'}")
    store._conn.close()
    store._conn = None                                   # the database is gone mid-run
    store.record("X-1", "ingest", "accepted", "test")
    assert store.fallback_path.is_file() and store.trail("X-1")


GOVERNANCE_KEYS = ["decision_id", "timestamp", "ticket_id", "run_id", "stage", "input_summary", "model", "prediction",
                   "alternatives", "sources_used", "threshold_applied", "action_taken", "reason", "guardrail_results",
                   "prompt_version", "requirement_ids"]


@pytest.mark.review_mode
def test_decisions_export_in_the_governance_record_format(pipeline):
    pipeline.process(ROLLBACK, run_id="GOV")
    approve(pipeline.reviews, pipeline.store, pipeline.guardrails, "R-1", "Sofia Restrepo")
    recs = pipeline.store.export(ticket_id="R-1")
    assert all(list(r) == GOVERNANCE_KEYS for r in recs)
    by = {r["stage"]: r for r in recs}
    assert {"classification", "routing", "generation", "validation", "human_review"} <= set(by)
    c = by["classification"]
    assert c["prediction"]["value"] == "rollback_request" and 0 <= c["prediction"]["confidence"] <= 1
    assert c["alternatives"] and set(c["alternatives"][0]) == {"value", "confidence"}
    assert c["model"]["name"].startswith("classifier")
    r = by["routing"]
    assert r["action_taken"] in ("auto_respond", "escalate") and r["threshold_applied"] is not None
    assert r["sources_used"] and set(r["sources_used"][0]) == {"doc_id", "score"}
    g = by["generation"]
    assert g["prompt_version"].startswith("PR-01 v3+PR-02 v1+PR-03 v1")
    v = by["validation"]
    assert v["action_taken"] == "auto_respond"
    assert v["guardrail_results"]["pii"] == v["guardrail_results"]["grounding"] == v["guardrail_results"]["tone"] \
        == "pass"
    for rec in recs:                            # which requirement produced each decision
        assert rec["requirement_ids"] and all(re.fullmatch(r"(FR-\d+|A\d+)", x) for x in rec["requirement_ids"])
    assert by["human_review"]["action_taken"] == "sent_after_review"


def test_a_blocked_reply_is_logged_as_block(pipeline, monkeypatch):
    original = pipeline.generator.generate

    def leaky(t, r, **kw):
        a = original(t, r, **kw)
        a.answer_text += " Email me at someone@example.com."
        return a
    monkeypatch.setattr(pipeline.generator, "generate", leaky)
    o = pipeline.process(ROLLBACK, run_id="BLK")
    assert o.outcome == "escalated"
    v = [r for r in pipeline.store.export(ticket_id="R-1") if r["stage"] == "validation"]
    assert v[0]["action_taken"] == "block" and v[0]["guardrail_results"]["pii"] == "block"


def test_coverage_catches_a_path_that_decides_without_recording(pipeline):
    pipeline.process(ROLLBACK, run_id="COV")
    assert pipeline.store.reconcile("COV", ["R-1"])["reconciled"]
    # a ticket that was answered with a generated reply but no validation record: the unobserved path
    s = pipeline.store
    s.record("GAP-1", "ingest", "normalised", "test", run_id="COV")
    s.record("GAP-1", "route", "auto_respond", "test", run_id="COV")
    s.record("GAP-1", "generate", "answered", "test", run_id="COV", details={"purpose": "reply"})
    s.record("GAP-1", "outcome", "sent", "test", run_id="COV")
    rep = s.reconcile("COV", ["R-1", "GAP-1"])
    assert not rep["reconciled"]
    assert rep["path_gaps"] == [{"ticket_id": "GAP-1", "gap": "a reply was generated without a validation record"}]
    s.record("GAP-2", "outcome", "sent", "test", run_id="COV")
    gaps = {g["gap"] for g in s.reconcile("COV")["path_gaps"] if g["ticket_id"] == "GAP-2"}
    assert gaps == {"no ingest record", "answered without a routing record"}


def test_export_and_reconcile_from_the_command_line(pipeline, tmp_path):
    pipeline.process(ROLLBACK, run_id="CLI")
    db = f"sqlite:///{tmp_path / 'decisions.db'}"
    out = subprocess.run([sys.executable, "-m", "src.logging_store", "--db", db, "export", "--run-id", "CLI"],
                         capture_output=True, text=True, cwd=ROOT, timeout=120)
    recs = [json.loads(line) for line in out.stdout.splitlines()]
    assert recs and all(list(r) == GOVERNANCE_KEYS for r in recs)
    rc = subprocess.run([sys.executable, "-m", "src.logging_store", "--db", db, "reconcile", "--run-id", "CLI"],
                        capture_output=True, text=True, cwd=ROOT, timeout=120)
    assert rc.returncode == 0 and json.loads(rc.stdout)["reconciled"] is True
