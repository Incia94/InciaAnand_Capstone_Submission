"""Route: decide whether to answer a ticket automatically or escalate it to a person, and say why.

Build Specification (Route) -> where it is met                              -> tests (tests/test_route.py)
-----------------------------------------------------------------------------------------------------------
Answer-or-escalate using a threshold from data   calibrate(): threshold chosen on out-of-fold data  test_req1_*
Deterministic: same input, same decision         decide() is a pure function of its inputs+config   test_req2_*
Reason a support manager can read                RoutingDecision.reason (+ every check's detail)    test_req3_*
Escalations carry a drafted summary + sources    RoutingDecision.handoff                            test_req4_*

Usage (run from the project root)
-----
    python -m src.route calibrate                                   # sets the threshold from data
    python -m src.route run --input data/validation_tickets.json --sample

The decision
------------
Eight checks, in this order. The first that fails decides the escalation; all are recorded.

  1 readable       the ticket has usable text
  2 language       the ticket is in English, or was translated into English for processing
                   (a ticket that could not be translated goes to a person who reads it)
  3 classified     the classifier produced an intent (not its fallback)
  4 policy         the intent is not one that must always go to a person. The set is taken from
                   the training labels (must_not_auto_respond): security_incident,
                   compliance_request, feature_request, unclear_request
  5 topic          the text itself is not about a subject that is never answered automatically,
                   whatever the classifier said (a second, independent safety net; see TOPICS):
                   security problems, roadmap / timing / feature requests, service incidents,
                   billing disputes, and requests for compliance commitments
  6 documentation  retrieval found at least one passage above its relevance threshold (or, when
                   switched on, an intent-guided match; see src/retrieve.py)
  7 confidence     intent confidence >= the threshold
  8 consistency    the best article is one that tickets of this intent have needed before
                   (learned from expected_doc_ids), so classifier and search agree

If all pass, the ticket is answered automatically.

Before the checks, messages that are not support tickets are handled by src/conversation.py: a
message that is only small talk gets a fixed reply (action "converse"), a request for a person is
escalated, and a message with no CloudServe terms that is unlike every past ticket is told what the
assistant covers instead of being escalated.

The threshold
-------------
`calibrate` classifies every development ticket with a classifier that did not see it (two
cross-validations: one holding out whole groups of near-duplicates, i.e. new wording, one ordinary
split, i.e. familiar wording), retrieves for it, and replays the checks at every candidate
threshold from 0.50 to 0.99. Each threshold is scored as

    cost = cost_ratio x (wrong automatic answers) + (unnecessary escalations)

A wrong automatic answer is one where the label says escalate, or the intent was wrong, or the
article was not one a correct answer should cite. The default cost ratio is 5 (a wrong answer
costs a customer a bad reply and the team a complaint on top of the original ticket). That ratio
is a business judgement, not a fact in the data: it is a setting (ROUTING_COST_RATIO), and
calibrate reports the threshold each ratio from 2 to 10 would choose. The lowest-cost threshold is
saved in storage/routing.json with the whole cost curve. Lower thresholds win ties, so automation
is never given up for nothing.

Findability (INTENT_HINT=on|off|auto, default on)
-----------
Calibrate replays everything twice, with and without intent-guided search, and stores both costs
(hint_comparison). "on" uses it, so tickets are not escalated just because the customer's wording
missed the article; "auto" uses it only if it lowers the cost at the chosen cost ratio. On the
development data it removes most findability escalations; the price is a few more automatic
replies to tickets labelled escalate, many of which have twins with identical text labelled
auto_respond (discovery finding E24). Which way to set it is a business decision for the report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from pydantic import BaseModel

from src.classify import Classification, Classifier, near_duplicate_groups
from src.conversation import needs_detail, out_of_scope, reply as conversation_reply, small_talk
from src.ingest import LANGUAGE_NAMES, NormalisedTicket, load_tickets, normalise_ticket
from src.retrieve import RetrievalResult, intent_doc_map

log = logging.getLogger("route")

DEFAULT_CONFIG = "./storage/routing.json"
DEFAULT_THRESHOLD = 0.80          # only until `calibrate` has run (the Setup Guide's starting value)
DEFAULT_COST_RATIO = 5.0
DEFAULT_MUST_ESCALATE = ("compliance_request", "feature_request", "security_incident", "unclear_request")
THRESHOLD_GRID = [round(0.50 + 0.01 * i, 2) for i in range(50)]   # 0.50 .. 0.99
SENSITIVITY_RATIOS = (2.0, 3.0, 5.0, 10.0)
LOW_FAMILIARITY = 0.5

# Where an escalation should go. A suggestion for the receiving queue, based on the tier one /
# tier two split described in the stakeholder interviews; adjust it to CloudServe's real teams.
SUGGESTED_TEAM = {
    "security_incident": "security", "compliance_request": "compliance", "feature_request": "product",
    "performance_degradation": "tier2", "database_issue": "tier2", "deployment_failure": "tier2",
    "integration_help": "tier2", "webhook_issue": "tier2", "data_residency": "tier2",
}
PRIORITY = {"high": "P1", "medium": "P2", "low": "P3"}

# Subjects never answered automatically, detected from the ticket's own words (case-insensitive).
# A match escalates even when the classifier thought the ticket was something else. Each pattern
# was checked against the 580 labelled tickets: security matches 25 tickets, all labelled
# escalate; roadmap matches 20 feature requests and 0 tickets labelled auto_respond. A phrase
# preceded by a negation ("it does not look like an outage") does not count.
TOPICS: dict[str, tuple[str, str, str]] = {   # name -> (pattern, team, plain-language reason)
    "security": (
        r"\b(?:compromis\w*|breach\w*|hack(?:ed|er|ers)?|unauthori[sz]ed (?:access|activity|logins?|use|api calls?)"
        r"|unrecogni[sz]ed (?:activity|logins?|api calls?|resources?)|suspicious (?:activity|logins?|traffic)"
        r"|vulnerabilit\w*|exploit\w*|malware|ransomware|phishing|cve-\d+|penetration test\w*"
        r"|someone (?:else )?(?:has|had|got|gained) access|still (?:has|have) access|retained access"
        r"|after (?:they )?(?:left|leaving)[^.]{0,40}access|access[^.]{0,40}after (?:they )?(?:left|leaving))",
        "security", "it describes a possible security problem, which a person must always handle"),
    "roadmap": (
        r"\b(?:roadmap|when will (?:you|it|this|that|cloudserve|the feature)|eta\b|timeline for|release date"
        r"|is there (?:any )?plans?|are there (?:any )?plans|any plans to|planned (?:feature|release)|coming soon"
        r"|future release|feature request|would (?:love|like) (?:you |cloudserve )?to (?:add|see|support|offer)"
        r"|please (?:add|consider adding)|could you add|can you add)",
        "product", "it asks about a new feature, the roadmap or timing, which the documentation does not cover "
                   "and only the product team can answer"),
    "incident": (
        r"\b(?:outage|is (?:cloudserve|the platform|your (?:service|platform|api|console)) down|status page"
        r"|platform[- ]wide|region[- ]wide|across all (?:regions|projects)|multiple customers"
        r"|other customers (?:are|have)|everyone (?:is|on))",
        "incident", "it may be a new service incident, which needs a person to investigate"),
    "billing_dispute": (
        r"\b(?:disput\w*|chargeback|overcharg\w*|charged (?:twice|double|two times)|double[- ]charg\w*"
        r"|(?:want|need|request|requesting|like) (?:a |our )?refund|refund (?:us|me|the|all|my|our)|money back"
        r"|(?:cancel|close|terminate) (?:my|our) (?:account|subscription|plan|contract)"
        r"|wrong(?:ly)? charged|incorrect(?:ly)? charged|not authori[sz]e[^.]{0,30}charge)",
        "billing", "it disputes a charge, asks for money back or asks to cancel the account, which only the "
                   "billing team can decide"),
    "compliance_commitment": (
        r"\b(?:legally binding|certif(?:y|ied|ication)|attest\w*|sign (?:a|our|the|your) (?:dpa|contract|agreement"
        r"|addendum)|\bdpa\b|data processing (?:agreement|addendum)|hipaa|soc ?2|iso ?27001|pci[- ]dss|audit report)",
        "compliance", "it asks CloudServe for a compliance or legal commitment, which a person must give"),
}
_TOPIC_RX = {k: re.compile(v[0], re.I) for k, v in TOPICS.items()}
_NEGATED = re.compile(r"\b(?:not|no|n't|never|without)\b[^.]{0,25}$", re.I)


def detect_topics(text: str) -> dict[str, str]:
    """{topic: the words that matched} for every never-automate topic found in the text."""
    found = {}
    for name, rx in _TOPIC_RX.items():
        for m in rx.finditer(text):
            if not _NEGATED.search(text[max(0, m.start() - 30):m.start()]):
                found[name] = m.group()
                break
    return found


# ------------------------------------------------------------------ configuration
class RoutingConfig(BaseModel):
    confidence_threshold: float = DEFAULT_THRESHOLD
    cost_ratio: float = DEFAULT_COST_RATIO
    must_escalate_intents: list[str] = list(DEFAULT_MUST_ESCALATE)
    intent_docs: dict[str, list[str]] = {}          # intent -> documents its tickets have needed
    use_intent_hint: bool = False                   # intent-guided search; set by calibrate from data
    hint_comparison: dict = {}                      # cost with and without it, from calibrate
    calibrated: bool = False
    calibrated_on: int = 0
    curve: list[dict] = []
    sensitivity: dict[str, float] = {}
    chosen: dict = {}

    @property
    def version(self) -> str:
        core = json.dumps({"t": self.confidence_threshold, "m": sorted(self.must_escalate_intents),
                           "d": {k: sorted(v) for k, v in sorted(self.intent_docs.items())},
                           "h": self.use_intent_hint, "topics": sorted(TOPICS)}, sort_keys=True)
        return "route-" + hashlib.sha256(core.encode()).hexdigest()[:10]

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "RoutingConfig":
        """routing.json if present (data-derived); otherwise CONFIDENCE_THRESHOLD or 0.80."""
        path = Path(path or os.getenv("ROUTING_CONFIG", DEFAULT_CONFIG))
        try:
            cfg = cls.model_validate_json(path.read_text())
            mode = os.getenv("INTENT_HINT", "").strip().lower()
            if mode in ("on", "off"):          # the setting wins over what calibrate stored
                cfg.use_intent_hint = mode == "on"
            return cfg
        except (OSError, ValueError):
            log.warning("no usable %s: using threshold %s until `python -m src.route calibrate` runs",
                        path, os.getenv("CONFIDENCE_THRESHOLD", DEFAULT_THRESHOLD))
            return cls(confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", DEFAULT_THRESHOLD)))

    def save(self, path: Optional[Path] = None) -> None:
        path = Path(path or os.getenv("ROUTING_CONFIG", DEFAULT_CONFIG))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))


# ------------------------------------------------------------------ output
class Check(BaseModel):
    name: str
    passed: Optional[bool]     # None: not applicable to this ticket
    detail: str


class SourceRef(BaseModel):
    chunk_id: str
    doc_id: str
    title: str = ""
    score: float
    above_threshold: bool      # False: a near miss, shown to the agent but not trusted


class DocumentPointer(BaseModel):
    """Where the solution is likely to be, for the agent: the article and the part that matters."""
    doc_id: str
    chunk_id: str
    title: str
    score: float
    matched: bool               # True: passed the relevance threshold; False: the closest near miss
    excerpt: str                # the resolution steps (or opening) of the article


class Handoff(BaseModel):
    """What a person receives with an escalated ticket: enough to act without re-reading
    everything or searching the documentation again."""
    summary: str                # one paragraph: who, what, why it was escalated, where to look
    escalation_category: str    # policy | topic | language | no_documentation | low_confidence | ...
    suggested_team: str
    priority: str
    uncertainties: list[str]
    tier1_checks: list[dict]    # every check the system ran and what it found
    sources: list[SourceRef]
    documents: list[DocumentPointer] = []
    intent_guess: str
    intent_alternatives: list[dict]
    customer: dict
    channel: str
    language: str = "en"
    conversation: list[dict]    # earlier messages, then this ticket, oldest first (original wording)
    english_translation: Optional[str] = None   # when the customer did not write in English
    original_text: str
    draft_reply: Optional[dict] = None          # added by the pipeline: a cited draft for the agent


class RoutingDecision(BaseModel):
    ticket_id: str
    action: str                     # "auto_respond" | "escalate" | "converse" (small talk, out of scope)
    decided_by: str                 # name of the check that decided, or "all_checks_passed"
    reason: str                     # plain language, for a support manager
    checks: list[Check]
    intent: str
    urgency: str
    confidence: float
    threshold: float
    sources: list[SourceRef]        # the passages an automatic answer must be grounded in
    handoff: Optional[Handoff] = None
    reply: Optional[str] = None     # the fixed reply, for action "converse" (src/conversation.py)
    config_version: str
    inputs_digest: str              # identical inputs -> identical digest -> identical decision
    latency_ms: float = 0.0


# ------------------------------------------------------------------ helpers
_ACRONYMS = {"api": "API", "sso": "SSO"}


def _name(intent: str) -> str:
    """'api_key_issue' -> 'API key issue', quoted where it reads as a category."""
    words = [_ACRONYMS.get(w, w) for w in intent.split("_")]
    words[0] = words[0] if words[0].isupper() else words[0].capitalize()
    return "'" + " ".join(words) + "'"


def _a(intent: str) -> str:
    """'a' or 'an' before a category name, followed by the name."""
    name = _name(intent)
    return ("an " if name[1].upper() in "AEIOU" else "a ") + name


def _digest(ticket: NormalisedTicket, c: Classification, r: RetrievalResult, config: RoutingConfig) -> str:
    payload = {
        "ticket": [ticket.ticket_id, ticket.channel.value, ticket.text, ticket.processable],
        "classification": c.model_dump(exclude={"latency_ms"}),
        "retrieval": r.model_dump(exclude={"latency_ms"}),
        "config": config.version,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _sources(r: RetrievalResult, limit: int = 3) -> list[SourceRef]:
    refs = [SourceRef(chunk_id=p.chunk_id, doc_id=p.doc_id, title=p.title, score=p.score, above_threshold=True)
            for p in r.passages[:limit]]
    if not refs:   # nothing passed the threshold: show the near misses, clearly marked
        refs = [SourceRef(chunk_id=c.chunk_id, doc_id=c.chunk_id.split("::")[0], title=c.title, score=c.score,
                          above_threshold=False) for c in r.candidates[:limit]]
    return refs


def _excerpt(text: str, limit: int = 280) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "…"


def _documents(r: RetrievalResult, limit: int = 3) -> list[DocumentPointer]:
    out, seen = [], set()
    for p in r.passages:
        if p.doc_id not in seen:
            seen.add(p.doc_id)
            out.append(DocumentPointer(doc_id=p.doc_id, chunk_id=p.chunk_id, title=p.title, score=p.score,
                                       matched=True, excerpt=_resolution(p.text)))
    for cand in r.candidates:
        doc = cand.chunk_id.split("::")[0]
        if len(out) >= limit:
            break
        if doc not in seen:
            seen.add(doc)
            out.append(DocumentPointer(doc_id=doc, chunk_id=cand.chunk_id, title=cand.title, score=cand.score,
                                       matched=False, excerpt=cand.excerpt))
    return out[:limit]


def _resolution(text: str, limit: int = 600) -> str:
    body = text.split("Resolution:", 1)[-1] if "Resolution:" in text else text
    body = body.split("Notes:", 1)[0]
    return _excerpt(body, limit)


_CATEGORIES = {"conversation": "person_requested", "readable": "unreadable", "language": "language",
               "classified": "unclassified", "policy": "policy",
               "topic": "topic", "documentation": "no_documentation", "confidence": "low_confidence",
               "consistency": "signals_disagree", "generate": "documentation_incomplete",
               "guardrail": "guardrail_block"}


def _category(check_name: str) -> str:
    """Why a person is needed, in a few fixed words, for reporting. 'policy', 'topic' and
    'no_documentation' are genuine needs for a person; 'low_confidence' and 'signals_disagree'
    are findability problems the system should keep to a minimum."""
    return _CATEGORIES.get(check_name, check_name)


def _team(failed: Check, t: NormalisedTicket, c: Classification) -> str:
    if failed.name == "topic":
        found = detect_topics(t.original_text or t.text) or detect_topics(t.text)
        if found:
            return TOPICS[next(iter(found))][1]
    if failed.name == "language":
        return f"tier1-{t.language}"
    return SUGGESTED_TEAM.get(c.intent, "tier1")


# ------------------------------------------------------------------ the router
class Router:
    def __init__(self, config: Optional[RoutingConfig] = None):
        self.config = config or RoutingConfig()

    def decide(self, ticket: Any, classification: Classification, retrieval: RetrievalResult) -> RoutingDecision:
        """Never raises: if anything goes wrong the ticket is escalated with that reason."""
        started = time.perf_counter()
        t = ticket if isinstance(ticket, NormalisedTicket) else normalise_ticket(ticket)
        try:
            decision = self._decide(t, classification, retrieval)
        except Exception as exc:
            log.exception("routing failed for %s", t.ticket_id)
            decision = self._error_decision(t, classification, retrieval, exc)
        decision.latency_ms = round((time.perf_counter() - started) * 1000, 2)
        return decision

    def _checks(self, t: NormalisedTicket, c: Classification, r: RetrievalResult) -> list[Check]:
        cfg = self.config
        th = cfg.confidence_threshold
        top = r.passages[0] if r.passages else None
        checks = [
            Check(name="readable", passed=t.processable,
                  detail="The ticket has readable text." if t.processable
                  else "The ticket has no readable text (it is empty or could not be read)."),
            self._language_check(t),
            Check(name="classified", passed=not c.is_fallback,
                  detail=f"The system identified the request as {_a(c.intent)} ticket." if not c.is_fallback
                  else f"The system could not work out what the ticket is about ({c.fallback_reason})."),
            Check(name="policy", passed=c.intent not in cfg.must_escalate_intents,
                  detail=f"{_name(c.intent)} tickets may be answered automatically."
                  if c.intent not in cfg.must_escalate_intents
                  else f"Company policy: {_name(c.intent)} tickets are always handled by a person."),
            self._topic_check(t),
        ]
        if top and top.via == "intent_hint":
            doc_detail = (f"Found the help article '{top.title}' (match score {top.score:.2f}): it scored below the "
                          f"usual threshold ({r.threshold:.2f}), but it is the article this kind of request needs, "
                          f"so the classifier and the search agree.")
        elif top:
            doc_detail = f"Found a matching help article: '{top.title}' (match score {top.score:.2f})."
            if r.degraded:
                doc_detail += " Search ran in backup mode."
        elif r.status == "unavailable":
            doc_detail = "Documentation search was unavailable, so the answer could not be checked."
        elif r.status == "empty_query":
            doc_detail = "There was no text to search the documentation with."
        else:
            doc_detail = (f"No help article matched closely enough (best match scored {r.top_score:.2f}; "
                          f"at least {r.threshold:.2f} is needed).")
        checks.append(Check(name="documentation", passed=top is not None, detail=doc_detail))
        conf_ok = c.intent_confidence >= th
        checks.append(Check(
            name="confidence", passed=conf_ok,
            detail=f"The system is {c.intent_confidence:.0%} sure this is {_a(c.intent)} ticket "
                   f"(at least {th:.0%} is required to answer without a person)."))
        known = cfg.intent_docs.get(c.intent)
        if not cfg.intent_docs:
            checks.append(Check(name="consistency", passed=None, detail="Consistency check not configured."))
        elif top is None:
            checks.append(Check(name="consistency", passed=None,
                                detail="Not applicable: no article was found to compare."))
        else:
            ok = bool(known) and top.doc_id in known
            checks.append(Check(
                name="consistency", passed=ok,
                detail=f"The article found is one that {_name(c.intent)} tickets normally need." if ok
                else f"The ticket looks like {_a(c.intent)} ticket, but the best article found "
                     f"('{top.title}') is not one used for that kind of request, so the two signals disagree."))
        return checks

    @staticmethod
    def _language_check(t: NormalisedTicket) -> Check:
        name = LANGUAGE_NAMES.get(t.language, t.language)
        if t.language in ("en", "und") or not t.processable:
            return Check(name="language", passed=None if t.language == "und" else True,
                         detail="The ticket is in English." if t.language == "en" else "Language not identified.")
        if t.translated:
            return Check(name="language", passed=True,
                         detail=f"The customer wrote in {name}; the ticket was translated into English to be "
                                f"processed, and any reply goes back in {name} with the English version attached.")
        return Check(name="language", passed=False,
                     detail=f"The customer wrote in {name} and the ticket could not be translated, so a person "
                            f"who reads {name} should handle it.")

    @staticmethod
    def _topic_check(t: NormalisedTicket) -> Check:
        found = detect_topics(t.original_text or t.text) or detect_topics(t.text)
        if not found:
            return Check(name="topic", passed=True, detail="Nothing in the ticket is on the never-automate list.")
        name, words = next(iter(found.items()))
        return Check(name="topic", passed=False,
                     detail=f"Company policy: not answered automatically because {TOPICS[name][2]} "
                            f"(the ticket says \"{words}\").")

    def _conversation(self, t: NormalisedTicket, c: Classification,
                      r: RetrievalResult) -> Optional[RoutingDecision]:
        """Messages that are not support tickets (src/conversation.py): small talk gets a fixed reply,
        a request for a person is escalated, an unrelated message is told what the assistant covers.
        None for everything else, which is routed by the checks as before."""
        if not t.processable or "plain_text_input" in t.flags:   # malformed input stays with the checks (A11)
            return None
        said = t.original_text or t.text
        st = small_talk(said)
        english = t.language in ("en", "und") or t.translated
        vague = st is None and english and needs_detail(said) and not detect_topics(said)
        if st is None and not vague and not (english and out_of_scope(t.text, c.familiarity)
                                             and not detect_topics(said)):
            return None
        common = dict(ticket_id=t.ticket_id, intent=c.intent, urgency=c.urgency, confidence=c.intent_confidence,
                      threshold=self.config.confidence_threshold, sources=[], config_version=self.config.version,
                      inputs_digest=_digest(t, c, r, self.config))
        readable = Check(name="readable", passed=True, detail="The ticket has readable text.")
        if st is not None and st.kind == "human":
            failed = Check(name="conversation", passed=False, detail="The customer asked to speak to a person.")
            checks = [readable, failed]
            return RoutingDecision(action="escalate", decided_by="conversation", checks=checks,
                                   reason="Escalated to a person. The customer asked to speak to a person.",
                                   handoff=self._handoff(t, c, r, checks, failed, _sources(r)), **common)
        if st is not None:
            detail = f"The message is only small talk ({st.kind}), so it gets a short fixed reply."
            checks = [readable, Check(name="conversation", passed=True, detail=detail)]
            return RoutingDecision(action="converse", decided_by="small_talk",
                                   reply=conversation_reply(st.kind, st.language), checks=checks,
                                   reason="Answered with a fixed reply: the message is small talk, not a support "
                                          "question.", **common)
        lang = t.language if t.language in ("es", "fr", "de", "it", "pt") else "en"
        if vague:
            detail = ("The message is angry but says nothing specific about what is wrong, so the customer is "
                      "asked for details before anything is escalated.")
            checks = [readable, Check(name="conversation", passed=True, detail=detail)]
            return RoutingDecision(action="converse", decided_by="needs_detail",
                                   reply=conversation_reply("needs_detail", lang), checks=checks,
                                   reason="Answered with a fixed reply asking for details: the message says "
                                          "nothing specific about what is wrong.", **common)
        detail = ("The message contains no CloudServe terms, so it is outside what support covers.")
        checks = [readable, Check(name="conversation", passed=True, detail=detail)]
        return RoutingDecision(action="converse", decided_by="out_of_scope",
                               reply=conversation_reply("out_of_scope", lang), checks=checks,
                               reason="Answered with a fixed reply: the message is not about CloudServe, so it was "
                                      "not escalated.", **common)

    def screened(self, t: NormalisedTicket, s: Any) -> RoutingDecision:
        """The decision for a message the scope screen recognised (src/scope.py): a fixed reply, made
        before classification, so it carries no intent or sources. Refused manipulation attempts are
        marked for review in the decision log."""
        import hashlib
        review = f" Flagged for review: {'; '.join(s.flags)}." if s.flags else ""
        checks = [Check(name="readable", passed=True, detail="The ticket has readable text."),
                  Check(name="screen", passed=True, detail=f"Intent: {s.intent.value} ({s.kind}). {s.detail}{review}")]
        digest = hashlib.sha256(f"{t.text}|{s.kind}|{self.config.version}".encode()).hexdigest()[:16]
        return RoutingDecision(ticket_id=t.ticket_id, action="converse", decided_by=s.intent.value,
                               reason=f"Answered with a fixed reply, no ticket: {s.detail}{review}", checks=checks,
                               intent=s.intent.value, urgency="low", confidence=1.0,
                               threshold=self.config.confidence_threshold, sources=[], reply=s.reply,
                               config_version=self.config.version, inputs_digest=digest)

    def _decide(self, t: NormalisedTicket, c: Classification, r: RetrievalResult) -> RoutingDecision:
        chat = self._conversation(t, c, r)
        if chat is not None:
            return chat
        checks = self._checks(t, c, r)
        failed = next((ch for ch in checks if ch.passed is False), None)
        sources = _sources(r)
        common = dict(ticket_id=t.ticket_id, checks=checks, intent=c.intent, urgency=c.urgency,
                      confidence=c.intent_confidence, threshold=self.config.confidence_threshold,
                      sources=sources, config_version=self.config.version,
                      inputs_digest=_digest(t, c, r, self.config))
        if failed is None:
            top = r.passages[0]
            reason = (f"Answered automatically: the system is {c.intent_confidence:.0%} sure this is "
                      f"{_a(c.intent)} ticket and found the matching help article '{top.title}'.")
            if r.degraded:
                reason += " Note: documentation search ran in backup mode."
            return RoutingDecision(action="auto_respond", decided_by="all_checks_passed", reason=reason, **common)
        reason = "Escalated to a person. " + failed.detail
        return RoutingDecision(action="escalate", decided_by=failed.name, reason=reason,
                               handoff=self._handoff(t, c, r, checks, failed, sources), **common)

    def _handoff(self, t: NormalisedTicket, c: Classification, r: RetrievalResult, checks: list[Check],
                 failed: Check, sources: list[SourceRef]) -> Handoff:
        uncertainties = [ch.detail for ch in checks if ch.passed is False]
        if not c.is_fallback and c.familiarity < LOW_FAMILIARITY:
            uncertainties.append("The wording is unlike past tickets, so the classification is less reliable.")
        if not c.is_fallback and c.intent_confidence < self.config.confidence_threshold:
            alts = [f"{_name(a.label)} ({a.probability:.0%})" for a in c.intent_alternatives[1:3]]
            if alts:
                uncertainties.append("It could also be: " + ", ".join(alts) + ".")
        if r.degraded and r.reason:
            uncertainties.append("Documentation search ran in backup mode.")

        documents = _documents(r)
        who = (f"{t.customer_tier.value.capitalize()} customer {t.customer_name or ''} "
               f"({t.customer_id or 'unknown id'})")
        lang = ""
        if t.language not in ("en", "und"):
            lang = f", writing in {LANGUAGE_NAMES.get(t.language, t.language)}"
        elif t.language_fluency.value == "non_fluent":
            lang = ", writing in a second language"
        guess = (f"Likely {_a(c.intent)} ticket ({c.intent_confidence:.0%} confidence), urgency {c.urgency}."
                 if not c.is_fallback else "The type of request could not be identified.")
        matched = [d for d in documents if d.matched]
        if matched:
            docs = "The solution is likely in: " + "; ".join(f"'{d.title}' ({d.doc_id})" for d in matched) + "."
        elif documents:
            docs = ("No article matched closely enough. Closest: "
                    + "; ".join(f"'{d.title}' ({d.doc_id}, score {d.score:.2f})" for d in documents) + ".")
        else:
            docs = "No documentation was found: this may be a gap in the knowledge base."
        said = _excerpt(t.text)
        summary = (f"{' '.join(who.split())} via {t.channel.value.replace('_', ' ')}{lang}. {guess} "
                   f"Customer wrote{' (translated)' if t.translated else ''}: \"{said}\" "
                   f"Escalated because: {failed.detail} {docs}")
        thread = list(t.conversation) + [{"role": "customer", "text": t.original_text or t.original_body or t.text,
                                          "at": t.received_at_raw}]
        return Handoff(
            summary=summary,
            escalation_category=_category(failed.name),
            suggested_team=_team(failed, t, c),
            priority=PRIORITY.get(c.urgency, "P2"),
            uncertainties=uncertainties,
            tier1_checks=[ch.model_dump() for ch in checks],
            sources=sources,
            documents=documents,
            intent_guess=c.intent,
            intent_alternatives=[a.model_dump() for a in c.intent_alternatives[:3]],
            customer={"id": t.customer_id, "name": t.customer_name, "tier": t.customer_tier.value,
                      "region": t.customer_region.value, "language_fluency": t.language_fluency.value},
            channel=t.channel.value,
            language=t.language,
            conversation=thread,
            english_translation=t.text if t.translated else None,
            original_text=t.original_body if t.original_body else t.text,
        )

    def escalate_after(self, ticket: Any, classification: Classification, retrieval: RetrievalResult,
                       check_name: str, detail: str, extra_uncertainties: Sequence[str] = ()) -> RoutingDecision:
        """Escalate a ticket that routing had approved, because a later stage (the guardrails)
        stopped the reply. The handoff carries the same summary and sources as any escalation."""
        t = ticket if isinstance(ticket, NormalisedTicket) else normalise_ticket(ticket)
        checks = self._checks(t, classification, retrieval) + [Check(name=check_name, passed=False, detail=detail)]
        sources = _sources(retrieval)
        handoff = self._handoff(t, classification, retrieval, checks, checks[-1], sources)
        handoff.uncertainties += list(extra_uncertainties)
        return RoutingDecision(
            ticket_id=t.ticket_id, action="escalate", decided_by=check_name, reason="Escalated to a person. " + detail,
            checks=checks, intent=classification.intent, urgency=classification.urgency,
            confidence=classification.intent_confidence, threshold=self.config.confidence_threshold,
            sources=sources, handoff=handoff, config_version=self.config.version,
            inputs_digest=_digest(t, classification, retrieval, self.config))

    def _error_decision(self, t: NormalisedTicket, c: Any, r: Any, exc: Exception) -> RoutingDecision:
        detail = f"An internal error occurred while routing ({type(exc).__name__}), so a person should check it."
        check = Check(name="routing_error", passed=False, detail=detail)
        intent = getattr(c, "intent", "unclassified")
        return RoutingDecision(
            ticket_id=t.ticket_id, action="escalate", decided_by="routing_error",
            reason="Escalated to a person. " + detail, checks=[check],
            intent=intent, urgency=getattr(c, "urgency", "medium"),
            confidence=float(getattr(c, "intent_confidence", 0.0) or 0.0),
            threshold=self.config.confidence_threshold, sources=[],
            handoff=Handoff(summary=f"Ticket {t.ticket_id} via {t.channel.value}: {detail} "
                                    f"Customer wrote: \"{_excerpt(t.text)}\"",
                            escalation_category="error", suggested_team="tier1", priority="P2",
                            uncertainties=[detail], tier1_checks=[check.model_dump()], sources=[],
                            intent_guess=intent, intent_alternatives=[], customer={"id": t.customer_id},
                            channel=t.channel.value, language=t.language,
                            conversation=[{"role": "customer", "text": t.original_text or t.text, "at": ""}],
                            original_text=t.original_body or t.text),
            config_version=self.config.version, inputs_digest="error")


# ------------------------------------------------------------------ calibration from data
def _label_sets(tickets: Sequence[NormalisedTicket]) -> tuple[list[str], dict[str, list[str]]]:
    must = sorted({t.eval_labels["intent"] for t in tickets if t.eval_labels.get("must_not_auto_respond")})
    return must or list(DEFAULT_MUST_ESCALATE), intent_doc_map(tickets)


def out_of_fold_classifications(tickets: Sequence[NormalisedTicket],
                                folds: int = 5) -> list[tuple[int, Classification]]:
    """Each ticket classified twice by a model that did not see it: once with its near-duplicate
    group held out (new wording), once in an ordinary stratified split (familiar wording)."""
    import numpy as np
    from sklearn.model_selection import GroupKFold, StratifiedKFold
    y = np.array([t.eval_labels["intent"] for t in tickets])
    groups = near_duplicate_groups([t.body or t.text for t in tickets], y)
    out = []
    for splits in (GroupKFold(folds).split(tickets, y, groups),
                   StratifiedKFold(folds, shuffle=True, random_state=0).split(tickets, y)):
        for tr, te in splits:
            clf = Classifier().fit([tickets[i] for i in tr])
            out += [(int(i), clf.classify(tickets[i])) for i in te]
    return out


def hint_docs_for(config: RoutingConfig, c: Classification) -> Optional[list[str]]:
    """The articles intent-guided search may use for this classification, or None when it is off
    or the classifier is not confident enough for its intent to guide the search."""
    if not config.use_intent_hint or c.is_fallback or c.intent_confidence < config.confidence_threshold:
        return None
    return config.intent_docs.get(c.intent) or None


def replay(records: Sequence[dict], threshold: float, config: RoutingConfig) -> dict:
    """Outcome rates for one threshold over pre-computed (ticket, classification, retrieval) records.
    Records may carry `retrieval_hint` (the result with intent-guided search), used when the config
    switches it on and the classification clears the threshold."""
    cfg = config.model_copy(update={"confidence_threshold": threshold})
    router = Router(cfg)
    auto = wrong = missed = 0
    for rec in records:
        t, c = rec["ticket"], rec["classification"]
        r = rec.get("retrieval_hint") if (rec.get("retrieval_hint") is not None and hint_docs_for(cfg, c)) \
            else rec["retrieval"]
        action = router._decide(t, c, r).action
        labels = t.eval_labels
        if action == "auto_respond":
            auto += 1
            top_doc = r.passages[0].doc_id if r.passages else None
            if (labels["expected_route"] == "escalate" or c.intent != labels["intent"]
                    or top_doc not in (labels.get("expected_doc_ids") or [])):
                wrong += 1
        elif labels["expected_route"] == "auto_respond":
            missed += 1
    n = max(len(records), 1)
    return {"threshold": threshold, "auto_rate": round(auto / n, 4), "wrong_auto_rate": round(wrong / n, 4),
            "unnecessary_escalation_rate": round(missed / n, 4)}


def choose_threshold(curve: Sequence[dict], cost_ratio: float) -> dict:
    scored = [{**p, "cost": round(cost_ratio * p["wrong_auto_rate"] + p["unnecessary_escalation_rate"], 4)}
              for p in curve]
    return min(scored, key=lambda p: (p["cost"], p["threshold"]))


def calibrate(tickets: Iterable[Any], retriever: Any, cost_ratio: Optional[float] = None,
              config_path: Optional[Path] = None) -> RoutingConfig:
    rows = [t if isinstance(t, NormalisedTicket) else normalise_ticket(t) for t in tickets]
    rows = [t for t in rows if t.processable and t.eval_labels and t.eval_labels.get("intent")]
    cost_ratio = float(cost_ratio or os.getenv("ROUTING_COST_RATIO", DEFAULT_COST_RATIO))
    must, intent_docs = _label_sets(rows)
    base = RoutingConfig(cost_ratio=cost_ratio, must_escalate_intents=must, intent_docs=intent_docs)
    retrievals = {t.ticket_id: retriever.retrieve(t) for t in rows}
    records = []
    for i, c in out_of_fold_classifications(rows):
        t = rows[i]
        plain = retrievals[t.ticket_id]
        hint = None
        if not plain.passages and not c.is_fallback and intent_docs.get(c.intent):
            hint = retriever.retrieve(t, hint_docs=intent_docs[c.intent])
        records.append({"ticket": t, "classification": c, "retrieval": plain, "retrieval_hint": hint})
    # two curves: without and with intent-guided search; keep whichever costs less (ties: without)
    curves = {flag: [replay(records, th, base.model_copy(update={"use_intent_hint": flag})) for th in THRESHOLD_GRID]
              for flag in (False, True)}
    picks = {flag: choose_threshold(curves[flag], cost_ratio) for flag in (False, True)}
    data_choice = picks[True]["cost"] < picks[False]["cost"]
    mode = os.getenv("INTENT_HINT", "on").strip().lower()      # on | off | auto (follow the data)
    use_hint = data_choice if mode == "auto" else mode != "off"
    curve, best = curves[use_hint], picks[use_hint]
    sensitivity = {f"{r:g}": choose_threshold(curve, r)["threshold"] for r in SENSITIVITY_RATIOS}
    cfg = base.model_copy(update={"confidence_threshold": best["threshold"], "calibrated": True,
                                  "calibrated_on": len(rows), "curve": curve, "sensitivity": sensitivity,
                                  "chosen": best, "use_intent_hint": use_hint,
                                  "hint_comparison": {"without": picks[False], "with": picks[True],
                                                      "lower_cost_at_this_ratio": "with" if data_choice else "without",
                                                      "setting": mode}})
    cfg.save(config_path)
    return cfg


# ------------------------------------------------------------------ whole chain, for checking
def run_chain(tickets: Iterable[Any], classifier: Classifier, retriever: Any, router: Router) -> list[RoutingDecision]:
    out = []
    for raw in tickets:
        t = raw if isinstance(raw, NormalisedTicket) else normalise_ticket(raw)
        c = classifier.classify(t)
        out.append(router.decide(t, c, retriever.retrieve(t, hint_docs=hint_docs_for(router.config, c))))
    return out


def summarise(decisions: Sequence[RoutingDecision], tickets: Sequence[NormalisedTicket]) -> dict:
    by_id = {t.ticket_id: t for t in tickets}
    labelled = [(d, by_id[d.ticket_id]) for d in decisions if by_id.get(d.ticket_id) and by_id[d.ticket_id].eval_labels]
    out = {"tickets": len(decisions),
           "auto_respond": sum(d.action == "auto_respond" for d in decisions),
           "escalate": sum(d.action == "escalate" for d in decisions),
           "decided_by": dict(Counter(d.decided_by for d in decisions).most_common())}
    if labelled:
        agree = sum(d.action == t.eval_labels["expected_route"] for d, t in labelled)
        must = [(d, t) for d, t in labelled if t.eval_labels.get("must_not_auto_respond")]
        out["agreement_with_expected_route"] = round(agree / len(labelled), 3)
        out["must_not_auto_respond_auto_answered"] = sum(d.action == "auto_respond" for d, _ in must)
        out["escalation_rate"] = round(out["escalate"] / max(len(decisions), 1), 3)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    from src.classify import load_or_train
    from src.retrieve import build_retriever

    p = argparse.ArgumentParser(prog="python -m src.route", description="Answer-or-escalate routing.")
    p.add_argument("--config", type=Path, default=Path(os.getenv("ROUTING_CONFIG", DEFAULT_CONFIG)))
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("calibrate", help="choose the confidence threshold from labelled tickets")
    c.add_argument("--tickets", type=Path, default=Path(os.getenv("TRAIN_DATA", "./data/development_tickets.json")))
    c.add_argument("--cost-ratio", type=float, default=None,
                   help=f"cost of a wrong automatic answer relative to an unnecessary escalation "
                        f"(default ROUTING_COST_RATIO or {DEFAULT_COST_RATIO:g})")
    r = sub.add_parser("run", help="ingest, classify, retrieve and route a ticket file")
    r.add_argument("--input", type=Path, required=True)
    r.add_argument("--sample", action="store_true", help="print one automatic answer and one escalation")
    r.add_argument("--output", type=Path, help="write every decision as JSON Lines")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        if args.command == "calibrate":
            cfg = calibrate(load_tickets(args.tickets), build_retriever(), args.cost_ratio, args.config)
            print(json.dumps({"confidence_threshold": cfg.confidence_threshold, "cost_ratio": cfg.cost_ratio,
                              "chosen": cfg.chosen, "threshold_by_cost_ratio": cfg.sensitivity,
                              "must_escalate_intents": cfg.must_escalate_intents,
                              "use_intent_hint": cfg.use_intent_hint, "hint_comparison": cfg.hint_comparison},
                             indent=2))
            return 0
        tickets = load_tickets(args.input)
        decisions = run_chain(tickets, load_or_train(), build_retriever(), Router(RoutingConfig.load(args.config)))
    except (OSError, ValueError) as exc:
        log.error("%s failed: %s", args.command, exc)
        return 1
    if args.sample:
        for action in ("auto_respond", "escalate"):
            d = next((x for x in decisions if x.action == action), None)
            if d:
                print(d.model_dump_json(indent=2, exclude={"checks"}), "\n")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as fh:
            for d in decisions:
                fh.write(d.model_dump_json() + "\n")
    print(json.dumps(summarise(decisions, tickets), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
