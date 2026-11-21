"""Tests for src/route.py, one block per Build Specification bullet for Route."""
import re
from pathlib import Path

import pytest

from src import route as RT
from src.classify import Alternative, Classification, Classifier
from src.ingest import load_tickets, normalise_ticket
from src.retrieve import Corpus, Retriever, RetrievalResult, best_threshold, score_tickets, _labelled

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def dev():
    return load_tickets(DATA / "development_tickets.json")


@pytest.fixture(scope="module")
def val():
    return load_tickets(DATA / "validation_tickets.json")


@pytest.fixture(scope="module")
def clf(dev):
    return Classifier().fit(dev)


@pytest.fixture(scope="module")
def retriever(dev):
    corpus = Corpus.from_file(DATA / "documentation.json")
    r = Retriever(corpus)
    r.thresholds["lexical"] = best_threshold(score_tickets(r.fallback, corpus, _labelled(dev)))
    return r


@pytest.fixture(scope="module")
def config(dev):
    must, docs = RT._label_sets(dev)
    return RT.RoutingConfig(confidence_threshold=0.9, must_escalate_intents=must, intent_docs=docs)


@pytest.fixture(scope="module")
def router(config):
    return RT.Router(config)


@pytest.fixture(scope="module")
def decisions(val, clf, retriever, router):
    return RT.run_chain(val, clf, retriever, router)


def make_classification(intent="api_key_issue", conf=0.95, fallback=False):
    alts = [Alternative(label=intent, probability=conf), Alternative(label="rate_limit", probability=0.03)]
    return Classification(ticket_id="T1", intent=intent, intent_confidence=conf, urgency="medium",
                          urgency_confidence=0.5, intent_alternatives=alts, is_fallback=fallback,
                          fallback_reason="no_usable_text" if fallback else None, familiarity=0.9)


TICKET = {"ticket_id": "T1", "channel": "email", "subject": "API key returns 401 Unauthorized",
          "body": "My production API key suddenly returns 401 unauthorized. Do I need to rotate the key "
                  "or change its scope?", "customer_id": "C1",
          "customer_name": "Asha Rao", "customer_tier": "business", "customer_region": "europe",
          "language_fluency": "fluent"}


# ---------------------------------------------------------------- 1. answer or escalate, threshold from data
def test_req1_every_ticket_gets_one_of_two_actions(decisions, val):
    assert len(decisions) == len(val)
    assert {d.action for d in decisions} <= {"auto_respond", "escalate"}
    assert {d.action for d in decisions} == {"auto_respond", "escalate"}     # both happen on real data


def test_req1_threshold_decides_at_the_boundary(router, retriever):
    r = retriever.retrieve(TICKET)
    assert r.passages and r.passages[0].doc_id == "DOC-AUTH-004"
    th = router.config.confidence_threshold
    above = router.decide(TICKET, make_classification(conf=th), r)
    below = router.decide(TICKET, make_classification(conf=round(th - 0.01, 2)), r)
    assert above.action == "auto_respond" and below.action == "escalate"
    assert below.decided_by == "confidence"


def test_req1_calibrate_chooses_threshold_from_data(dev, retriever, tmp_path):
    subset = dev[::2]                                           # 250 tickets keeps the test quick
    cfg = RT.calibrate(subset, retriever, cost_ratio=5, config_path=tmp_path / "routing.json")
    assert cfg.calibrated and cfg.calibrated_on == len(subset)
    assert len(cfg.curve) == len(RT.THRESHOLD_GRID) and cfg.confidence_threshold in RT.THRESHOLD_GRID
    best = RT.choose_threshold(cfg.curve, 5)
    assert cfg.confidence_threshold == best["threshold"]
    costs = [5 * p["wrong_auto_rate"] + p["unnecessary_escalation_rate"] for p in cfg.curve]
    assert min(costs) == pytest.approx(best["cost"], abs=1e-4)
    # a cheaper wrong answer never pushes the threshold up
    ratios = sorted(cfg.sensitivity, key=float)
    assert [cfg.sensitivity[r] for r in ratios] == sorted(cfg.sensitivity[r] for r in ratios)
    saved = RT.RoutingConfig.load(tmp_path / "routing.json")
    assert saved.confidence_threshold == cfg.confidence_threshold and saved.must_escalate_intents


def test_req1_must_escalate_intents_come_from_labels(dev, config):
    assert set(config.must_escalate_intents) == {"compliance_request", "feature_request",
                                                 "security_incident", "unclear_request"}


def test_req1_must_not_auto_respond_is_never_auto_answered(decisions, val):
    by_id = {t.ticket_id: t for t in val}
    for d in decisions:
        if by_id[d.ticket_id].eval_labels["must_not_auto_respond"]:
            assert d.action == "escalate", d.reason


def test_req1_policy_beats_confidence(router, retriever):
    d = router.decide(TICKET, make_classification(intent="security_incident", conf=0.99), retriever.retrieve(TICKET))
    assert d.action == "escalate" and d.decided_by == "policy"


def test_req1_missing_config_uses_env_threshold(tmp_path, monkeypatch):
    monkeypatch.setenv("CONFIDENCE_THRESHOLD", "0.7")
    assert RT.RoutingConfig.load(tmp_path / "none.json").confidence_threshold == 0.7


# ---------------------------------------------------------------- 2. deterministic
def test_req2_same_input_same_decision(router, clf, retriever, val):
    for t in val[:20]:
        c, r = clf.classify(t), retriever.retrieve(t)
        a, b = router.decide(t, c, r), router.decide(t, c, r)
        assert a.model_dump(exclude={"latency_ms"}) == b.model_dump(exclude={"latency_ms"})


def test_req2_fresh_router_and_rerun_give_identical_decisions(config, decisions, val, clf, retriever):
    again = RT.run_chain(val, clf, retriever, RT.Router(config.model_copy()))
    assert [d.model_dump(exclude={"latency_ms"}) for d in again] == \
        [d.model_dump(exclude={"latency_ms"}) for d in decisions]


def test_req2_digest_changes_when_input_changes(router, retriever):
    r = retriever.retrieve(TICKET)
    a = router.decide(TICKET, make_classification(conf=0.95), r)
    b = router.decide(TICKET, make_classification(conf=0.96), r)
    assert a.inputs_digest != b.inputs_digest
    assert a.config_version == b.config_version


# ---------------------------------------------------------------- 3. reasons a support manager can read
def test_req3_reasons_are_plain_language(decisions):
    for d in decisions:
        assert re.match(r"^(Answered automatically: |Escalated to a person\. )", d.reason), d.reason
        assert "_" not in d.reason and "intent_confidence" not in d.reason and "None" not in d.reason
        assert d.reason.endswith(".") and len(d.reason) < 320


def test_req3_every_check_explained(decisions):
    for d in decisions:
        assert [c.name for c in d.checks] == ["readable", "classified", "policy", "documentation",
                                              "confidence", "consistency"]
        for c in d.checks:
            assert c.detail and c.detail.endswith(".") and "_" not in c.detail


def test_req3_confidence_reason_states_both_numbers(router, retriever):
    d = router.decide(TICKET, make_classification(conf=0.42), retriever.retrieve(TICKET))
    assert "42%" in d.reason and f"{router.config.confidence_threshold:.0%}" in d.reason


@pytest.mark.parametrize("raw,decided_by", [({"ticket_id": "E", "channel": "chat", "body": ""}, "readable")])
def test_req3_unreadable_ticket_reason(router, retriever, clf, raw, decided_by):
    t = normalise_ticket(raw)
    d = router.decide(t, clf.classify(t), retriever.retrieve(t))
    assert d.action == "escalate" and d.decided_by == decided_by and "no readable text" in d.reason


# ---------------------------------------------------------------- 4. escalations carry summary and sources
def test_req4_every_escalation_has_a_handoff(decisions, val):
    by_id = {t.ticket_id: t for t in val}
    for d in [x for x in decisions if x.action == "escalate"]:
        h = d.handoff
        assert h is not None and h.summary and h.uncertainties
        assert "Escalated because:" in h.summary
        assert by_id[d.ticket_id].text.split()[0] in h.summary            # the customer's words are in it
        assert h.priority in {"P1", "P2", "P3"} and h.suggested_team
        assert h.original_text == by_id[d.ticket_id].original_body


def test_req4_handoff_sources_are_real_and_marked(decisions, retriever):
    for d in [x for x in decisions if x.action == "escalate"]:
        assert d.handoff.sources, d.ticket_id
        for s in d.handoff.sources:
            assert retriever.corpus.resolve(s.chunk_id) is not None and s.title
        if d.decided_by == "documentation":
            assert all(not s.above_threshold for s in d.handoff.sources)   # near misses, clearly marked


def test_req4_automatic_answers_carry_sources_but_no_handoff(decisions):
    autos = [d for d in decisions if d.action == "auto_respond"]
    assert autos
    for d in autos:
        assert d.handoff is None and d.sources and all(s.above_threshold for s in d.sources)


def test_req4_low_confidence_handoff_lists_alternatives(router, retriever):
    d = router.decide(TICKET, make_classification(conf=0.4), retriever.retrieve(TICKET))
    assert any("could also be" in u for u in d.handoff.uncertainties)


def test_req4_routing_error_still_escalates_with_handoff(router):
    d = router.decide(TICKET, object(), RetrievalResult(status="ok"))        # nonsense classification
    assert d.action == "escalate" and d.decided_by == "routing_error" and d.handoff.summary


def test_req4_classifier_fallback_escalates_with_handoff(router, retriever):
    d = router.decide(TICKET, make_classification(intent="unclassified", conf=0.0, fallback=True),
                      retriever.retrieve(TICKET))
    assert d.action == "escalate" and d.decided_by == "classified" and "could not be identified" in d.handoff.summary


# ---------------------------------------------------------------- consistency check, CLI
def test_disagreeing_classifier_and_search_escalate(router, retriever):
    billing = {**TICKET, "subject": "Invoice higher than expected", "body": "Our invoice is much higher this month."}
    r = retriever.retrieve(billing)
    d = router.decide(billing, make_classification(intent="api_key_issue", conf=0.99), r)
    if r.passages:                                   # the article found is a billing one
        assert d.action == "escalate" and d.decided_by == "consistency" and "disagree" in d.reason


def test_cli_run(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("CHROMA_PATH", str(tmp_path / "none"))
    monkeypatch.setenv("RETRIEVAL_BACKEND", "lexical")
    monkeypatch.setenv("CLASSIFIER_PATH", str(tmp_path / "clf.joblib"))
    out = tmp_path / "d.jsonl"
    assert RT.main(["--config", str(tmp_path / "none.json"), "run", "--input",
                    str(DATA / "validation_tickets.json"), "--output", str(out)]) == 0
    assert len(out.read_text().splitlines()) == 80
    assert '"must_not_auto_respond_auto_answered": 0' in capsys.readouterr().out
