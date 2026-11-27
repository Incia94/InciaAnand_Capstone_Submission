"""Small talk, requests for a person, and unrelated messages (src/conversation.py and its use in
routing): answered with fixed replies instead of being escalated, without touching real tickets."""
import json
from pathlib import Path

import pytest

from src import api as A
from src.classify import Classifier
from src.conversation import DOMAIN_WORDS, PROFANITY, REPLIES, needs_detail, out_of_scope, reply, small_talk
from src.generate import Generator
from src.guardrail import Guardrails
from src.ingest import detect_language, load_tickets
from src.logging_store import DecisionStore
from src.retrieve import Corpus, Retriever, _labelled, best_threshold, score_tickets
from src.route import Router, RoutingConfig, _label_sets

DATA = Path(__file__).resolve().parents[1] / "data"
ALL = json.loads((DATA / "development_tickets.json").read_text()) + \
    json.loads((DATA / "validation_tickets.json").read_text())


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("conv")
    dev = load_tickets(DATA / "development_tickets.json")
    corpus = Corpus.from_file(DATA / "documentation.json")
    retriever = Retriever(corpus)
    retriever.thresholds["lexical"] = best_threshold(score_tickets(retriever.fallback, corpus, _labelled(dev)))
    must, docs = _label_sets(dev)
    router = Router(RoutingConfig(confidence_threshold=0.5, must_escalate_intents=must, intent_docs=docs,
                                  use_intent_hint=True))
    p = A.SupportPipeline(Classifier().fit(dev), retriever, router, Generator(None, unavailable_reason="no key"),
                          Guardrails(corpus=corpus), DecisionStore(f"sqlite:///{tmp / 'd.db'}"), A.Metrics())
    yield p
    p.store.close()


@pytest.fixture(autouse=True)
def no_kill_switch(tmp_path, monkeypatch):
    monkeypatch.setenv("KILL_SWITCH_FILE", str(tmp_path / "KS"))
    monkeypatch.delenv("KILL_SWITCH", raising=False)


def ask(pipeline, text, **kw):
    return pipeline.process({"ticket_id": kw.pop("ticket_id", "C-1"), "channel": "chat", "body": text, **kw})


# ---------------------------------------------------------------- never touches a real ticket
def test_no_ticket_in_the_data_is_small_talk():
    assert not [t["ticket_id"] for t in ALL if small_talk(f"{t['subject']} {t['body']}") or small_talk(t["body"])]


def test_every_ticket_in_the_data_has_a_cloudserve_word():
    assert not [t["ticket_id"] for t in ALL if not DOMAIN_WORDS.search(f"{t['subject']} {t['body']}")]


def test_no_validation_ticket_is_routed_as_conversation(pipeline):
    outs = [pipeline.process(t) for t in ALL[500:]]
    assert not [o.ticket_id for o in outs
                if o.decided_by in ("small_talk", "out_of_scope", "needs_detail", "conversation")]


def test_no_ticket_in_the_data_swears_or_is_out_of_scope():
    assert not [t["ticket_id"] for t in ALL if PROFANITY.search(f"{t['subject']} {t['body']}")]
    assert not [t["ticket_id"] for t in ALL if needs_detail(t["body"]) or out_of_scope(t["body"], 0.0)]


# ---------------------------------------------------------------- small talk
@pytest.mark.parametrize("text,kind,lang", [
    ("Hello", "greet", "en"), ("hi there!", "greet", "en"), ("Como estas?", "how", "es"),
    ("Hola, ¿qué tal?", "how", "es"), ("how are you doing today?", "how", "en"), ("thanks!", "thanks", "en"),
    ("Merci beaucoup", "thanks", "fr"), ("Guten Morgen", "greet", "de"), ("ciao", "greet", "it"),
    ("obrigado", "thanks", "pt"), ("bye", "bye", "en"), ("are you a bot?", "identity", "en"),
    ("ok", "ack", "en"), ("help", "greet", "en"), ("I want to talk to a human", "human", "en"),
    ("Quiero hablar con una persona", "human", "es"),
])
def test_small_talk_is_recognised(text, kind, lang):
    st = small_talk(text)
    assert st is not None and (st.kind, st.language) == (kind, lang)


@pytest.mark.parametrize("text", ["Hello, my API key returns 401", "hi, how do I roll back a release?",
                                  "I need help with my deployment", "thanks, but the webhook still fails"])
def test_a_greeting_with_a_real_question_is_not_small_talk(text):
    assert small_talk(text) is None


def test_small_talk_gets_a_fixed_reply_in_the_customers_language(pipeline):
    o = ask(pipeline, "Como estas?")
    assert o.outcome == "sent" and o.decided_by == "greeting" and o.reply == REPLIES["es"]["how"]
    o = ask(pipeline, "Hello")
    assert o.outcome == "sent" and o.reply == REPLIES["en"]["greet"] and not o.handoff


def test_asking_for_a_person_escalates_with_a_handoff(pipeline):
    o = ask(pipeline, "Can I talk to a human please?")
    assert o.outcome == "escalated" and o.decided_by == "conversation"
    assert o.handoff["escalation_category"] == "person_requested" and o.handoff["summary"]


# ---------------------------------------------------------------- unrelated messages
@pytest.mark.parametrize("text", ["When will my groceries arrive?", "What's the weather in Paris tomorrow?",
                                  "Book me a flight to Delhi", "Where is my parcel?"])
def test_unrelated_messages_are_called_out_not_escalated(pipeline, text):
    o = ask(pipeline, text)
    assert o.outcome == "sent" and o.decided_by == "out_of_scope" and o.reply == REPLIES["en"]["out_of_scope"]


@pytest.mark.parametrize("text", ["it is not working, please help", "why is my bill so high", "cant log in",
                                  "is the service down?", "my app is slow"])
def test_vague_support_messages_are_not_out_of_scope(pipeline, text):
    assert ask(pipeline, text).decided_by not in ("out_of_scope", "small_talk")


def test_an_untranslated_non_english_ticket_is_never_called_out_of_scope(pipeline):
    o = ask(pipeline, "Mi despliegue falla en el health check desde esta mañana")
    assert o.outcome == "escalated" and o.decided_by == "language"


def test_an_unrelated_spanish_message(pipeline):
    """Untranslated, the system cannot know what it says, so a Spanish reader gets it. Translated,
    it is recognised as unrelated and answered in Spanish."""
    from tests.test_requirements import RoleLLM
    assert detect_language("¿Cuándo llegan mis compras?")[0] == "es"
    assert ask(pipeline, "¿Cuándo llegan mis compras?").decided_by == "language"
    translating = A.SupportPipeline(pipeline.classifier, pipeline.retriever, pipeline.router,
                                    Generator(RoleLLM(english="When will my groceries arrive?")),
                                    pipeline.guardrails, pipeline.store, A.Metrics())
    o = ask(translating, "¿Cuándo llegan mis compras?")
    assert o.decided_by == "out_of_scope" and o.reply == REPLIES["es"]["out_of_scope"]


def test_out_of_scope_rules():
    assert out_of_scope("When will my groceries arrive?", 0.15)
    assert out_of_scope("Where are my groceries", 0.90)                  # short: no classifier needed
    long = " ".join(["Could you tell me when the groceries I ordered for the party will finally arrive"] * 2)
    assert out_of_scope(long, 0.15) and not out_of_scope(long, 0.60)     # long: familiarity decides
    assert not out_of_scope("why is my bill so high", 0.15)              # a CloudServe word: route normally


@pytest.mark.parametrize("text", ["Wtf is worng with my system", "wtf is wrong with my system",
                                  "this is useless, nothing works", "what the hell is going on??"])
def test_angry_messages_without_detail_are_asked_for_detail_not_escalated(pipeline, text):
    o = ask(pipeline, text)
    assert o.outcome == "sent" and o.decided_by == "frustration" and o.reply == REPLIES["en"]["needs_detail"]


@pytest.mark.parametrize("text", ["wtf, my deployment fails with a 502", "my api key is shit, it returns 401",
                                  "this stupid invoice charged us twice, we want a refund"])
def test_angry_messages_that_say_what_is_wrong_are_routed(pipeline, text):
    assert ask(pipeline, text).decided_by not in ("needs_detail", "out_of_scope", "small_talk", "frustration")


# ---------------------------------------------------------------- governance still holds
def test_conversation_is_logged_and_reconciles(pipeline):
    for i, text in enumerate(["Hello", "When will my groceries arrive?", "talk to an agent"]):
        pipeline.process(
            {"ticket_id": f"C-L{i}", "channel": "chat", "body": text}, run_id="CONV")
    rep = pipeline.store.reconcile("CONV", ["C-L0", "C-L1", "C-L2"])
    assert rep["reconciled"]
    route = next(r for r in pipeline.store.trail("C-L0") if r["stage"] == "route" and r["run_id"] == "CONV")
    assert route["action_taken"] == "converse"


def test_kill_switch_stops_conversational_replies_too(pipeline, tmp_path):
    (tmp_path / "KS").write_text("on")
    assert ask(pipeline, "Hello").decided_by == "kill_switch"


def test_every_reply_exists_in_every_language():
    kinds = set(REPLIES["en"])
    for lang, table in REPLIES.items():
        assert set(table) == kinds, lang
        assert all(reply(k, lang) for k in kinds)
