"""Cirrus AI as a support intelligence engine: BM25 and metadata filters in search (src/retrieve.py),
tiered resolution and channel policies (src/policy.py, the "tier" step in src/api.py)."""
from pathlib import Path

import pytest

from src import api as A
from src.classify import Classifier
from src.generate import Generator
from src.guardrail import Guardrails
from src.ingest import doc_products, load_tickets
from src.logging_store import DecisionStore
from src.policy import (CHANNEL_POLICIES, TECHNICAL_INTENTS, Tier, decide_tier, policy_for, similar_threads,
                        topic)
from src.retrieve import (BM25Search, Corpus, LexicalSearch, Retriever, _labelled, approved_threshold_from,
                          best_threshold, bm25_tokens, matches_filters, score_tickets)
from src.route import Router, RoutingConfig, _label_sets

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def corpus():
    return Corpus.from_file(DATA / "documentation.json")


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory, corpus):
    tmp = tmp_path_factory.mktemp("engine")
    dev = load_tickets(DATA / "development_tickets.json")
    retriever = Retriever(corpus)
    rows = score_tickets(retriever.fallback, corpus, _labelled(dev))
    retriever.thresholds["lexical"] = th = best_threshold(rows)
    retriever.thresholds["lexical_approved"] = approved_threshold_from(rows, th)
    must, docs = _label_sets(dev)
    router = Router(RoutingConfig(confidence_threshold=0.5, must_escalate_intents=must, intent_docs=docs,
                                  use_intent_hint=True))
    p = A.SupportPipeline(Classifier().fit(dev), retriever, router, Generator(None, unavailable_reason="no key"),
                          Guardrails(corpus=corpus), DecisionStore(f"sqlite:///{tmp / 'd.db'}"), A.Metrics())
    yield p
    p.store.close()


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("KILL_SWITCH_FILE", str(tmp_path / "KS"))
    monkeypatch.delenv("KILL_SWITCH", raising=False)
    monkeypatch.delenv("CHANNEL_POLICY", raising=False)
    monkeypatch.delenv("ROLLOUT_PHASE", raising=False)


ROLLBACK = ("How do I revert to an earlier revision? The release we put out this morning is causing errors and I "
            "would rather roll back than try to fix forward.")


# ---------------------------------------------------------------- Solution 1: search
def test_bm25_keeps_error_codes_and_identifiers_whole():
    assert "429" in bm25_tokens("We get HTTP 429 from the search endpoint")
    toks = bm25_tokens("ERR_QUOTA_EXCEEDED on x-api-key")
    assert "err_quota_exceeded" in toks and "quota" in toks and "x-api-key" in toks


def test_bm25_scores_are_on_a_fixed_scale(corpus):
    hits = BM25Search(corpus).search("429 too many requests rate limit", 3)
    assert hits and all(0 <= s <= 1 for _, s in hits) and hits[0][0].startswith("DOC-API-001")


def test_the_keyword_leg_blends_bm25(corpus, monkeypatch):
    assert LexicalSearch(corpus).bm25 is not None
    assert LexicalSearch(corpus, bm25_weight=0).bm25 is None
    monkeypatch.setenv("LEXICAL_BM25_WEIGHT", "0.5")
    assert LexicalSearch(corpus).bm25_weight == 0.5


def test_articles_carry_product_category_and_version(corpus):
    meta = [c.metadata for c in corpus.chunks]
    assert all({"category", "product", "doc_version"} <= set(m) for m in meta)
    assert doc_products("Business and Enterprise plans") == "All products"
    assert doc_products("Console, CLI, SDK") == "Console,CLI,SDK"


def test_metadata_filters(corpus):
    r = Retriever(corpus)
    facets = r.facets()
    assert "deployment" in facets["category"] and "Managed Database" in facets["product"]
    assert facets["doc_version"] == ["1.0"]
    q = {"ticket_id": "F", "body": "my deployment fails at the health check"}
    assert r.retrieve(q, filters={"category": "deployment"}).doc_ids[0].startswith("DOC-DEPLOY")
    assert r.retrieve(q, filters={"category": "billing"}).status == "no_relevant_passage"
    assert r.retrieve(q, filters={"doc_version": "2.0"}).status == "no_relevant_passage"
    assert matches_filters({"product": "All products"}, {"product": "Managed Database"})
    assert not matches_filters({"product": "Console,CLI"}, {"product": "Managed Database"})


def test_approved_threshold_is_stricter_than_relevance(corpus):
    dev = load_tickets(DATA / "development_tickets.json")
    rows = score_tickets(LexicalSearch(corpus), corpus, _labelled(dev))
    th = best_threshold(rows)
    approved = approved_threshold_from(rows, th)
    above = [r for r in rows if r["group"] in ("relevant", "wrong", "no_doc") and r["top_score"] >= approved]
    assert approved >= th and len(above) >= 20
    assert sum(r["top_correct"] for r in above) / len(above) >= 0.95


# ---------------------------------------------------------------- Solution 3: tiers
def test_tier_decisions():
    assert decide_tier("chat", "api_key_issue", 0.5, 0.3).tier == Tier.RETRIEVAL
    assert decide_tier("chat", "api_key_issue", 0.25, 0.3).tier == Tier.RAG
    assert decide_tier("chat", "api_key_issue", 0.5, 0.3, via_hint=True).tier == Tier.RAG   # a hint is never Tier 1


def test_every_outcome_records_its_tier(pipeline):
    assert pipeline.process({"ticket_id": "T0", "channel": "chat", "body": "Tell me a joke"}).tier == 0
    o = pipeline.process({"ticket_id": "T1", "channel": "chat", "body": ROLLBACK})
    assert o.outcome == "sent" and o.tier in (1, 2) and o.tier_reason.startswith("Tier")
    o = pipeline.process({"ticket_id": "T3", "channel": "chat",
                          "body": "Our account was hacked and someone created new API keys."})
    assert o.outcome == "escalated" and o.tier == 3


def test_tier_1_makes_no_model_call(pipeline, monkeypatch):
    calls = []
    original = pipeline.generator._generate
    monkeypatch.setattr(pipeline.generator, "_generate", lambda *a, **k: calls.append(1) or original(*a, **k))
    state = pipeline.run({"ticket_id": "T1B", "channel": "chat", "body": ROLLBACK})
    if state["tier"].tier == Tier.RETRIEVAL:
        assert calls == [] and state["answer"].generator == "extractive" and not state["answer"].degraded


# ---------------------------------------------------------------- Solution 5: channels
def test_channel_policies_match_the_rollout_plan():
    assert {k: v.rollout_phase for k, v in CHANNEL_POLICIES.items()} == {
        "docs_comment": 1, "chat": 2, "email": 3, "forum": 3}
    assert CHANNEL_POLICIES["email"].max_auto_tier == Tier.RETRIEVAL
    assert CHANNEL_POLICIES["forum"].suggest_only and CHANNEL_POLICIES["forum"].reply_once
    assert CHANNEL_POLICIES["docs_comment"].narrow_technical_only


def test_email_sends_only_approved_content(pipeline):
    d = decide_tier("email", "api_key_issue", 0.25, 0.3)
    assert d.escalate and d.tier == Tier.HUMAN and "only approved content" in d.reason
    assert not decide_tier("email", "api_key_issue", 0.5, 0.3).escalate


def test_docs_pages_answer_narrow_technical_questions_only():
    assert "billing_query" not in TECHNICAL_INTENTS
    assert decide_tier("docs_comment", "billing_query", 0.9, 0.3).escalate
    assert not decide_tier("docs_comment", "rate_limit", 0.9, 0.3).escalate


def test_forum_points_to_existing_answers_and_replies_once(pipeline):
    assert decide_tier("forum", "webhook_issue", 0.9, 0.3, already_replied=True).escalate
    assert similar_threads("Webhooks are not arriving at our endpoint") == ["Webhooks not arriving"]
    o = pipeline.process({"ticket_id": "FR1", "channel": "forum", "body": ROLLBACK})
    if o.outcome == "sent":
        assert o.tier == 1 and "Existing community threads on this:" in o.reply


def test_rollout_phase_holds_back_later_channels(pipeline, monkeypatch):
    monkeypatch.setenv("ROLLOUT_PHASE", "1")
    d = decide_tier("chat", "rollback_request", 0.9, 0.3)
    assert d.escalate and "rollout phase 2" in d.reason
    o = pipeline.process({"ticket_id": "RP1", "channel": "chat", "body": ROLLBACK})
    assert o.outcome == "escalated" and o.decided_by == "channel_policy" and o.draft_reply


def test_uniform_policy_switches_channel_rules_off(monkeypatch):
    monkeypatch.setenv("CHANNEL_POLICY", "uniform")
    assert policy_for("email").max_auto_tier == Tier.RAG
    assert not decide_tier("email", "api_key_issue", 0.25, 0.3).escalate


def test_topics_are_plain_language():
    assert topic("deployment_failure") == "Deployment troubleshooting"
    assert topic("billing_query") == topic("quota_or_overage") == "Billing and refunds"


# ---------------------------------------------------------------- Solution 6: knowledge gaps
def test_knowledge_gaps_are_read_from_the_decision_log(pipeline):
    from src.gaps import knowledge_gaps, record_search_gap
    pipeline.process({"ticket_id": "G1", "channel": "chat",
                      "body": "Are staged rollouts on the roadmap? We want to move traffic gradually."}, run_id="GAP")
    pipeline.process({"ticket_id": "G2", "channel": "chat", "body": ROLLBACK}, run_id="GAP")
    gaps = knowledge_gaps(pipeline.store, run_id="GAP")
    assert all(g["count"] >= 1 and g["examples"] for g in gaps)
    assert "G2" not in [t for g in gaps for t in g["tickets"]]               # answered: not a gap
    record_search_gap(pipeline.store, "how do I configure quantum widgets", "the Billing page",
                      topic_hint="Billing", run_id="GAP2")
    assert knowledge_gaps(pipeline.store, run_id="GAP2")[0]["topic"] == "Billing"


# ---------------------------------------------------------------- Solution 7: in-app help
def test_in_app_help_uses_page_and_error_code_without_a_model(corpus):
    from src.contextual import PAGES, suggest
    r = Retriever(corpus)
    h = suggest(r, "api", "429")
    assert h.error.startswith("429 Too Many Requests") and h.suggestions[0].doc_id == "DOC-API-001"
    assert [s.doc_id for s in suggest(r, "account").suggestions] == list(PAGES["account"].starters)
    h = suggest(r, "billing", query="why is my invoice higher this month")
    assert h.suggestions and all(s.doc_id.startswith("DOC-BILL") for s in h.suggestions)   # filtered to billing
    assert suggest(r, "billing", query="zzzz qqqq").gap
