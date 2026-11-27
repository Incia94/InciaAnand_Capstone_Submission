"""The ten operating requirements, one section each, tested through the real pipeline.

 1 search is not keyword-only                       6 personal data handled for compliance
 2 automated replies always cite an article         7 findability escalations removed; genuine ones kept
 3 roadmap / timing / features / incidents escalate 8 escalations carry summary, conversation, documents
 4 security issues always escalate                  9 customers who do not write in English
 5 billing: no claims about money                   10 a cited draft for the agent to review and send
"""
import json
import re
from pathlib import Path

import pytest

from src import api as A
from src import generate as G
from src.classify import Alternative, Classification, Classifier
from src.guardrail import Guardrails
from src.ingest import detect_language, load_tickets, normalise_ticket
from src.logging_store import DecisionStore
from src.retrieve import Corpus, HybridSearch, Retriever, _labelled, best_threshold, score_tickets
from src.route import Router, RoutingConfig, _label_sets, detect_topics

DATA = Path(__file__).resolve().parents[1] / "data"
SPANISH = ("Hola, nuestra clave de API devuelve 401 desde esta mañana y no podemos desplegar. "
           "¿Tenemos que rotar la clave o cambiar sus permisos? Gracias por la ayuda.")
SPANISH_EN = ("Hello, our API key has been returning 401 since this morning and we cannot deploy. "
              "Do we need to rotate the key or change its permissions? Thanks for the help.")


class RoleLLM:
    """A stand-in model that behaves by task: translates tickets, translates replies, and drafts
    answers from the first resolution step of the first passage (so the claim is supported)."""
    model = "fake-model"

    def __init__(self, english=SPANISH_EN, reply_prefix="RESPUESTA: "):
        self.english, self.reply_prefix, self.calls = english, reply_prefix, []

    def complete(self, messages, temperature=0.0, max_tokens=600):
        self.calls.append(messages)
        system, user = messages[0]["content"], messages[-1]["content"]
        if system.startswith("You translate customer support tickets"):
            return self.english
        if system.startswith("You translate CloudServe support replies"):
            return self.reply_prefix + re.search(r"<reply>\n(.*)\n</reply>", user, re.S).group(1)
        step = re.search(r"Resolution:\s*\n?\s*1\.\s+(.+)", user)
        return json.dumps({"status": "answered", "claims": [{"text": step.group(1).strip(), "citations": ["P1"]}],
                           "missing_information": None})

    def kinds(self):
        return [m[0]["content"].split("\n")[0][:40] for m in self.calls]


@pytest.fixture(scope="module")
def parts():
    dev = load_tickets(DATA / "development_tickets.json")
    corpus = Corpus.from_file(DATA / "documentation.json")
    retriever = Retriever(corpus)
    retriever.thresholds["lexical"] = best_threshold(score_tickets(retriever.fallback, corpus, _labelled(dev)))
    must, docs = _label_sets(dev)
    return Classifier().fit(dev), retriever, must, docs, corpus


def make_pipeline(parts, tmp_path, llm=None, hint=True, threshold=0.9):
    clf, retriever, must, docs, corpus = parts
    router = Router(RoutingConfig(confidence_threshold=threshold, must_escalate_intents=must, intent_docs=docs,
                                  use_intent_hint=hint))
    gen = G.Generator(llm) if llm else G.Generator(None, unavailable_reason="no API key configured")
    return A.SupportPipeline(clf, retriever, router, gen, Guardrails(corpus=corpus),
                             DecisionStore(f"sqlite:///{tmp_path / 'decisions.db'}"), A.Metrics())


@pytest.fixture
def pipeline(parts, tmp_path, monkeypatch):
    monkeypatch.setenv("KILL_SWITCH_FILE", str(tmp_path / "KILL_SWITCH"))
    return make_pipeline(parts, tmp_path)


@pytest.fixture(scope="module")
def val():
    return json.loads((DATA / "validation_tickets.json").read_text())


def ticket(body, subject="", **kw):
    return {"ticket_id": kw.pop("ticket_id", "R-1"), "channel": kw.pop("channel", "email"), "subject": subject,
            "body": body, "customer_id": "CUST-77", "customer_tier": "business",
            "customer_region": kw.pop("region", "north_america"), "language_fluency": "fluent", **kw}


def confident(intent, conf=0.99):
    return Classification(ticket_id="R-1", intent=intent, intent_confidence=conf, urgency="medium",
                          urgency_confidence=0.6, intent_alternatives=[Alternative(label=intent, probability=conf)],
                          familiarity=0.9)


# ---------------------------------------------------------------- 1 search is not keyword-only
def test_1_default_search_combines_meaning_and_wording(parts):
    _, _, _, _, corpus = parts

    class MeaningOnly:                       # a dense search that knows billing == invoice
        name = "dense"

        def search(self, text, k):
            return [("DOC-BILL-001::full", 0.9)] + [(c.chunk_id, 0.1) for c in corpus.chunks
                                                    if c.chunk_id != "DOC-BILL-001::full"][:k - 1]
    hybrid = HybridSearch(MeaningOnly(), Retriever(corpus).fallback, len(corpus.chunks))
    # no word in common with the article title, but the meaning search finds it
    top = hybrid.search("why is the amount we pay so big this time", 3)[0]
    assert top[0] == "DOC-BILL-001::full"


# ---------------------------------------------------------------- 2 every automated reply cites an article
def test_2_every_sent_reply_cites_at_least_one_article(pipeline, val):
    from src.scope import CONVERSATIONAL           # fixed replies (a clarifying question) cite nothing
    sent = [o for o in (pipeline.process(t) for t in val) if o.outcome == "sent" and o.decided_by not in
            CONVERSATIONAL]
    assert sent
    for o in sent:
        assert o.sources and all(re.fullmatch(r"DOC-[A-Z]+-\d{3}", s["doc_id"]) for s in o.sources)
        assert "\nSources:\n[1] " in o.reply


def test_2_an_answer_without_a_cited_article_is_blocked(parts):
    corpus = parts[4]
    a = G.GeneratedAnswer(ticket_id="T", status="answered", claims=[], answer_text="Try again later.",
                          sources=[], generator="llm")
    res = Guardrails(corpus=corpus).check(normalise_ticket(ticket("x y z help")), a,
                                          parts[1].retrieve("webhook signature"))
    assert not res.released and any(f.kind == "no_article_cited" for c in res.checks for f in c.findings)


# ---------------------------------------------------------------- 3 roadmap / timing / features / incidents
@pytest.mark.parametrize("body,topic", [
    ("When will you add support for Terraform? Is it on your roadmap?", "roadmap"),
    ("Is there any plan to support ARM builds, and what is the ETA?", "roadmap"),
    ("We would like you to add a dark mode to the console.", "roadmap"),
    ("Is CloudServe down? Our deploys fail and the status page shows nothing.", "incident"),
])
def test_3_never_automate_topics_escalate_even_if_classifier_is_sure(pipeline, body, topic):
    t = normalise_ticket(ticket(body))
    r = pipeline.retriever.retrieve(t)
    d = pipeline.router.decide(t, confident("deployment_failure"), r)
    assert d.action == "escalate" and d.decided_by == "topic", d.reason
    assert d.handoff.escalation_category == "topic"
    assert d.handoff.suggested_team == {"roadmap": "product", "incident": "incident"}[topic]


def test_3_negated_mentions_do_not_count():
    assert detect_topics("It does not appear to be an outage, my colleague signs in fine.") == {}


def test_3_feature_requests_in_the_data_are_never_sent(pipeline):
    dev = json.loads((DATA / "development_tickets.json").read_text())
    feature = [t for t in dev if t["labels"]["intent"] == "feature_request"]
    assert all(pipeline.process(t).outcome == "escalated" for t in feature)


def test_3_replies_may_not_talk_about_the_future(parts):
    from tests.test_guardrail import API, answer
    corpus = parts[4]
    a = answer(extra_text=" This feature will be available in the next release.")
    res = Guardrails(corpus=corpus).check(normalise_ticket(ticket("429s")), a,
                                          G.RetrievalResult(status="ok", passages=[API]))
    assert not res.released and any(f.kind == "roadmap_claim" for c in res.checks for f in c.findings)


# ---------------------------------------------------------------- 4 security issues always escalate
def test_4_security_wording_escalates_whatever_the_classifier_says(pipeline):
    t = normalise_ticket(ticket("We see unrecognised API calls and think our account was compromised."))
    d = pipeline.router.decide(t, confident("api_usage_question"), pipeline.retriever.retrieve(t))
    assert d.action == "escalate" and d.decided_by == "topic" and d.handoff.suggested_team == "security"


def test_4_every_security_ticket_in_the_data_is_escalated(pipeline, val):
    dev = json.loads((DATA / "development_tickets.json").read_text())
    sec = [t for t in dev + val if t["labels"]["intent"] == "security_incident"]
    outs = [pipeline.process(t) for t in sec]
    assert len(sec) > 20 and all(o.outcome == "escalated" for o in outs)


# ---------------------------------------------------------------- 5 billing: no claims about money
@pytest.mark.parametrize("claim", [
    "Your plan costs $49 per month.", "You will be credited for the outage.", "We will waive the overage fee.",
    "You will not be charged for staging.", "A refund of the difference is on its way.", "That is 200 euros.",
])
def test_5_money_claims_are_blocked(parts, claim):
    from tests.test_guardrail import API, answer
    res = Guardrails(corpus=parts[4]).check(normalise_ticket(ticket("invoice")), answer(extra_text=" " + claim),
                                            G.RetrievalResult(status="ok", passages=[API]))
    kinds = {f.kind for c in res.checks for f in c.findings}
    assert not res.released and kinds & {"money_claim", "refund_promise"}, claim


def test_5_explaining_how_billing_works_is_allowed():
    from src.guardrail import _FORBIDDEN
    docs = json.loads((DATA / "documentation.json").read_text())
    billing = " ".join(d["content"] for d in docs if d["category"] == "billing")
    assert not [k for k, p in _FORBIDDEN if p.search(billing)]      # the articles themselves never trip it


def test_5_a_billing_dispute_goes_to_the_billing_team(pipeline):
    t = normalise_ticket(ticket("We were charged twice this month and want a refund."))
    d = pipeline.router.decide(t, confident("billing_query"), pipeline.retriever.retrieve(t))
    assert d.action == "escalate" and d.decided_by == "topic" and d.handoff.suggested_team == "billing"


# ---------------------------------------------------------------- 6 personal data and compliance
def test_6_personal_data_is_masked_before_it_reaches_a_model(parts):
    t = normalise_ticket(ticket("My key sk_live_abcdefghijklmnop1234 fails; email me at asha.rao@example.com "
                                "or call +44 20 7946 0958. Customer CUST-12345."))
    msgs, _ = G.build_messages("SYS", t, parts[1].retrieve("api key 401").passages or
                               parts[1].passages_for([("DOC-AUTH-004::full", 0.5)]))
    sent = msgs[1]["content"]
    for secret in ("sk_live_abcdefghijklmnop1234", "asha.rao@example.com", "7946 0958", "CUST-12345"):
        assert secret not in sent


def test_6_blocked_regions_never_call_an_external_model(parts, tmp_path, monkeypatch):
    monkeypatch.setenv("KILL_SWITCH_FILE", str(tmp_path / "KS"))
    monkeypatch.setenv("LLM_BLOCKED_REGIONS", "europe")
    llm = RoleLLM()
    p = make_pipeline(parts, tmp_path, llm=llm)
    val = json.loads((DATA / "validation_tickets.json").read_text())
    eu = [t for t in val if t["customer_region"] == "europe"][:10]
    states = [p.run(t) for t in eu]
    assert llm.calls == []
    drafted = [s["answer"] for s in states if s.get("answer") is not None]
    assert drafted and all(a.generator == "extractive" and "data residency" in a.data_policy for a in drafted)


def test_6_decision_log_holds_no_raw_personal_data(pipeline):
    pipeline.process(ticket("Contact asha.rao@example.com about key sk_live_abcdefghijklmnop1234 returning 401."),
                     run_id="PII")
    rows = pipeline.store.trail("R-1")
    dump = json.dumps(rows)
    assert "asha.rao@example.com" not in dump and "sk_live_abcdefghijklmnop1234" not in dump


# ---------------------------------------------------------------- 7 findability vs genuine escalations
def test_7_intent_hint_turns_findability_escalations_into_answers(parts, tmp_path, monkeypatch, val):
    monkeypatch.setenv("KILL_SWITCH_FILE", str(tmp_path / "KS"))
    monkeypatch.setenv("CHANNEL_POLICY", "uniform")      # findability only: no channel holding replies back
    avoidable = {}
    for hint in (False, True):
        (tmp_path / str(hint)).mkdir()
        p = make_pipeline(parts, tmp_path / str(hint), hint=hint)
        outs = {t["ticket_id"]: p.process(t) for t in val}
        avoidable[hint] = sum(o.outcome == "escalated" and t["labels"]["expected_route"] == "auto_respond"
                              and bool(t["labels"]["expected_doc_ids"]) for t, o in zip(val, outs.values()))
        if hint:
            policy = [t for t in val if t["labels"]["must_not_auto_respond"]]
            assert all(outs[t["ticket_id"]].outcome == "escalated" or outs[t["ticket_id"]].decided_by == "ambiguous"
                       for t in policy)                       # genuine ones still go (or get a clarifying question)
    assert avoidable[True] < avoidable[False] / 2, avoidable


def test_7_escalations_say_whether_they_are_findability_or_genuine(pipeline, val):
    cats = {o.handoff["escalation_category"] for o in (pipeline.process(t) for t in val) if o.outcome == "escalated"}
    assert cats <= {"policy", "topic", "no_documentation", "low_confidence", "signals_disagree",
                    "documentation_incomplete", "guardrail_block", "language", "unclassified", "unreadable",
                    "channel_policy"}


# ---------------------------------------------------------------- 8 the handoff
def test_8_handoff_has_summary_conversation_and_where_the_solution_is(pipeline):
    raw = ticket("A former employee still has access to our project after leaving. Please help.",
                 subject="Ex-employee access",
                 messages=[{"role": "customer", "text": "First message: we noticed odd logins.",
                            "at": "2026-11-20T09:00Z"},
                           {"role": "agent", "text": "Thanks, looking into it."}])
    o = pipeline.process(raw)
    h = o.handoff
    assert o.outcome == "escalated" and h["summary"].startswith("Business customer")
    assert [m["role"] for m in h["conversation"]] == ["customer", "agent", "customer"]
    assert "former employee" in h["conversation"][-1]["text"]
    assert h["documents"] and h["documents"][0]["doc_id"].startswith("DOC-") and h["documents"][0]["excerpt"]
    assert {c["name"] for c in h["tier1_checks"]} >= {"policy", "topic", "documentation", "confidence"}
    assert h["suggested_team"] == "security" and h["priority"] in ("P1", "P2", "P3")


# ---------------------------------------------------------------- 9 customers who do not write in English
def test_9_language_is_detected():
    assert detect_language(SPANISH)[0] == "es"
    assert detect_language("Notre facture est plus élevée que prévu pour ce mois")[0] == "fr"
    assert detect_language("APIキーが401を返します")[0] == "ja"
    assert detect_language("builds that work last week are now fail during dependency resolution.")[0] == "en"


def test_9_non_english_ticket_is_translated_answered_and_replied_in_its_language(parts, tmp_path, monkeypatch):
    monkeypatch.setenv("KILL_SWITCH_FILE", str(tmp_path / "KS"))
    monkeypatch.delenv("LLM_BLOCKED_REGIONS", raising=False)
    llm = RoleLLM()
    p = make_pipeline(parts, tmp_path, llm=llm, threshold=0.5)
    state = p.run(ticket(SPANISH, region="latin_america"))
    t, o = state["ticket"], state["outcome"]
    assert t.language == "es" and t.translated and t.text == SPANISH_EN and t.original_body == SPANISH
    assert llm.kinds()[0].startswith("You translate customer support tickets")
    assert o.language == "es"
    if o.outcome == "sent":
        assert o.reply.startswith("RESPUESTA: ") and G.SIMPLE_STYLE in llm.calls[1][0]["content"]
        assert "\n---\nEnglish version:\n" in o.reply
    else:   # escalated: the agent still gets the English translation and a draft
        assert o.handoff["english_translation"] == SPANISH_EN and o.handoff["language"] == "es"


def test_9_without_translation_a_person_who_reads_the_language_gets_it(pipeline):
    o = pipeline.process(ticket(SPANISH))
    assert o.outcome == "escalated" and o.decided_by == "language"
    assert o.handoff["suggested_team"] == "tier1-es" and o.handoff["language"] == "es"


def test_9_a_translation_that_changes_numbers_is_blocked(parts):
    from tests.test_guardrail import API, answer
    a = answer()
    a.answer_text_en = a.answer_text
    a.reply_language = "es"
    from src.guardrail import ENGLISH_DIVIDER
    a.answer_text = "Su límite es 999 solicitudes [1]." + ENGLISH_DIVIDER + a.answer_text_en
    res = Guardrails(corpus=parts[4]).check(normalise_ticket(ticket("429")), a,
                                            G.RetrievalResult(status="ok", passages=[API]))
    assert not res.released and any(f.kind == "translation_changed_numbers" for c in res.checks for f in c.findings)


# ---------------------------------------------------------------- 10 a cited draft for the agent
def test_10_escalations_carry_a_cited_draft_that_is_not_sent(pipeline, val):
    escalated = [o for o in (pipeline.process(t) for t in val) if o.outcome == "escalated"]
    drafted = [o for o in escalated if (o.draft_reply or {}).get("status") == "drafted"]
    assert drafted, "no escalation got a draft"
    for o in drafted:
        d = o.draft_reply
        assert d["sources"] and d["text"] and "Sources:" in d["text"] and "Not sent" in d["note"]
        assert o.reply is None and o.handoff["draft_reply"] == d
    for o in escalated:
        assert o.draft_reply is None or o.draft_reply["status"] in ("drafted", "not_drafted")


def test_10_security_escalation_still_gets_a_draft_from_the_security_article(pipeline):
    o = pipeline.process(ticket("A former employee still has access after leaving and made API calls. Urgent."))
    assert o.outcome == "escalated" and o.draft_reply["status"] == "drafted"
    assert any(s["doc_id"].startswith("DOC-SEC") for s in o.draft_reply["sources"])


def test_10_drafts_are_logged_but_not_counted_as_sent(pipeline):
    o = pipeline.process(ticket("A former employee still has access after leaving. Urgent."), run_id="D")
    stages = [r["stage"] for r in pipeline.store.trail(o.ticket_id) if r["run_id"] == "D"]
    assert "draft" in stages and stages[-1] == "outcome"
    assert pipeline.store.reconcile("D", [o.ticket_id])["reconciled"]
