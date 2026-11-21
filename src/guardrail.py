"""Guardrails: the release gate. Every generated reply passes through here before a customer sees it.

Build Specification (Validate) -> where it is met                              -> tests (tests/test_guardrail.py)
-------------------------------------------------------------------------------------------------------------------
Runs on every response before release           release() is the only function that returns reply   test_req1_*
                                                text to send; it always checks, and there is no
                                                switch to turn it off
Checks private data and unsupported claims       checks private_data and claim_support (and more)    test_req2_*
Can block                                        any blocking finding -> released=False, no text     test_req3_*
Records what it checked and found, either way    a `guardrail` row in the decision log for every     test_req4_*
                                                reply, released or blocked

Usage (run from the project root)
-----
    python -m src.guardrail run --input data/validation_tickets.json   # whole chain, then the gate

The checks (all always run, so the record is complete; any "block" finding stops the reply)
----------
structure        the answer is well-formed: answered replies have claims, every claim cites a
                 passage that was actually retrieved, and each cited passage is exactly the
                 corpus text with that id
private_data     no email addresses, phone numbers, payment card numbers (Luhn-checked), IP
                 addresses, customer ids, or secrets (API keys, tokens, passwords) in the reply
claim_support    every claim is supported by the passages it cites: at least SUPPORT_THRESHOLD of
                 its content words appear in them, and every number in it appears in them
forbidden_claims nothing the senior agents' must_not_claim lists rule out: that a refund has been
                 issued, that the issue has been fixed on our side, or a date for a fix
secret_request   the reply does not ask the customer to send a password, key, token or code
injection        the reply does not repeat our instructions or show signs of following a
                 customer's instructions (e.g. "ignore previous instructions")
tone             no insults or profanity (block); shouting in capitals is recorded as a warning

The support threshold (0.45)
----------------------------
Chosen on the 200 senior-agent reference answers: their factual sentences were scored against the
article they cite and against every other article. At 0.45, about 2% of correct sentences are
blocked, while 98.7% of sentences checked against the wrong article and 90% checked against a
closely related one are caught. Sentences the reference answers base on a different article than
the one labelled were set aside (the document-label noise noted in discovery). A wrongly blocked
reply is escalated, not lost, so the threshold leans towards blocking.

Personal data found in a reply is masked before it is written to the decision log.
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
from datetime import datetime, timezone
from typing import Any, Optional, Sequence

from pydantic import BaseModel

from src.generate import DONT_KNOW, SYSTEM_PROMPT, USER_PROMPT, GeneratedAnswer, PromptError, load_prompt
from src.ingest import NormalisedTicket, normalise_ticket
from src.logging_store import CARD_PATTERN, PHONE_PATTERN, PII_PATTERNS, DecisionStore, mask
from src.retrieve import Corpus, RetrievalResult

log = logging.getLogger("guardrail")

SUPPORT_THRESHOLD = 0.45


# ------------------------------------------------------------------ output
class Finding(BaseModel):
    kind: str                 # e.g. "email", "unsupported_claim", "refund_promise"
    severity: str             # "block" | "warn"
    detail: str               # readable; personal data masked
    claim_index: Optional[int] = None


class CheckResult(BaseModel):
    name: str
    passed: bool              # False if any finding blocks
    findings: list[Finding] = []
    stats: dict = {}


class GuardrailResult(BaseModel):
    ticket_id: str
    released: bool
    blocked_by: list[str]     # names of checks with blocking findings
    reason: str               # plain language
    checks: list[CheckResult]
    claims_checked: int = 0
    claims_unsupported: int = 0
    answer_digest: str
    generator: str = ""
    prompt_version: str = ""
    checked_at: str
    latency_ms: float = 0.0
    recorded: bool = False    # True once written to the decision log
    decision_id: str = ""


class Outbound(BaseModel):
    """What may be sent. `text` is None whenever the reply was blocked."""
    ticket_id: str
    released: bool
    text: Optional[str]
    guardrail: GuardrailResult


# ------------------------------------------------------------------ text helpers
_STOP = set("""a an the and or but if then than so to of in on at by for with from as is are was were be been being
it its this that these those you your we our us i they them their there here can could should would will may might
must do does did not no have has had which who what when where how all any each more most other some such only own
same too very just also into over under about again further once both few nor out up down off per""".split())
_NUMBER_WORDS = {w: str(i) for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen twenty".split())}
_NUMBER_WORDS.update({"thirty": "30", "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
                      "eighty": "80", "ninety": "90", "hundred": "100", "thousand": "1000"})


def _stem(w: str) -> str:
    for suf in ("ations", "ation", "ings", "ing", "edly", "ed", "es", "ly", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[:-len(suf)]
    return w


def _numbers_to_digits(words: list[str]) -> list[str]:
    """'two hundred' -> '200', 'twenty five' -> '25', 'one hundred and fifty' -> '150'."""
    out, value, active = [], 0, False
    for w in words + [""]:
        n = _NUMBER_WORDS.get(w)
        if n is not None and w not in ("hundred", "thousand"):
            value, active = value + int(n), True
        elif w in ("hundred", "thousand") and active:
            value *= int(_NUMBER_WORDS[w])
        elif w == "and" and active:
            continue
        else:
            if active:
                out.append(str(value))
            value, active = 0, False
            if w:
                out.append(_NUMBER_WORDS.get(w, w))
    return out


def content_words(text: str) -> list[str]:
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text.lower())          # 1,000 -> 1000
    words = _numbers_to_digits(re.findall(r"[a-z0-9]+", text))
    return [_stem(w) for w in words if w not in _STOP and (len(w) > 2 or w.isdigit())]


def support_score(claim: str, passages_text: str) -> tuple[float, list[str], list[str]]:
    """(share of the claim's content words found in the passages, words missing, numbers missing)."""
    words, available = content_words(claim), set(content_words(passages_text))
    if not words:
        return 1.0, [], []
    missing = [w for w in words if w not in available]
    numbers_missing = [w for w in words if w.isdigit() and w not in available]
    return 1 - len(missing) / len(words), missing, numbers_missing


def _luhn(digits: str) -> bool:
    total, alt = 0, False
    for d in reversed(digits):
        n = int(d) * (2 if alt else 1)
        total += n - 9 if n > 9 else n
        alt = not alt
    return total % 10 == 0


# ------------------------------------------------------------------ patterns for the other checks
_FORBIDDEN = [
    ("refund_promise", re.compile(
        r"\b(?:refund|reimburse)\w*\b[^.]{0,40}\b(?:issued|processed|approved|sent|been made|on (?:its|the) way)\b"
        r"|\b(?:we|i)(?: have|'ve| will|'ll)? (?:refunded|reimbursed|credited)\b"
        r"|\bwill (?:refund|reimburse)\b", re.I)),
    ("fix_claim", re.compile(
        r"\b(?:fixed|resolved|patched|corrected)\b[^.]{0,30}\b(?:on|at) our (?:side|end)\b"
        r"|\bwe(?: have|'ve)? (?:fixed|resolved|patched|deployed a fix)\b"
        r"|\b(?:the |this )?(?:issue|bug|problem|incident|outage) (?:has been|is now|was) (?:fixed|resolved|patched)\b",
        re.I)),
    ("fix_date", re.compile(
        r"\b(?:fix|patch|release|update|resolution)\w*\b[^.]{0,40}\b(?:by|on|before|within|in)\s+"
        r"(?:the )?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|tonight|today|"
        r"next (?:week|month)|end of (?:the )?(?:day|week|month)|\d+\s*(?:hours?|days?|weeks?)|"
        r"\d{1,2}(?:st|nd|rd|th)?\b|january|february|march|april|may|june|july|august|september|october|"
        r"november|december)", re.I)),
]
_FORBIDDEN_LABELS = {"refund_promise": "says a refund has been or will be issued",
                     "fix_claim": "says the issue has been fixed on our side",
                     "fix_date": "gives a date or time for a fix"}
_SECRET_REQUEST = re.compile(
    r"\b(send|share|provide|give|email|paste|tell|reply with|forward)\b[^.]{0,40}\b(password|api key|secret|"
    r"token|credential|private key|recovery code|mfa code|verification code|2fa code)", re.I)
_NEGATION = re.compile(r"\b(never|not|don't|do not|no need to|avoid)\b[^.]{0,20}$", re.I)
_INJECTION_SIGNS = re.compile(
    r"ignore (?:all |any )?(?:previous|prior|above) (?:instructions|rules)|as an ai (?:language )?model|"
    r"my (?:instructions|system prompt)|<\s*/?\s*(?:customer_ticket|passage|system)", re.I)
_INSULT = re.compile(r"\b(stupid|idiot\w*|dumb|moron\w*|incompetent|your (?:own )?fault|shut up|"
                     r"damn\w*|crap|wtf|f+u+c+k\w*|shit\w*)\b", re.I)
_SHOUT = re.compile(r"\b(?:[A-Z]{3,}\s+){3,}[A-Z]{3,}\b")
_ACRONYM_OK = {"API", "SSO", "SAML", "CLI", "SDK", "URL", "HTTP", "JSON", "MFA", "AWS", "CPU", "DNS"}


# ------------------------------------------------------------------ the guardrails
class Guardrails:
    def __init__(self, corpus: Optional[Corpus] = None, support_threshold: Optional[float] = None):
        self.corpus = corpus
        self.support_threshold = float(support_threshold if support_threshold is not None
                                       else os.getenv("GUARDRAIL_SUPPORT_THRESHOLD", SUPPORT_THRESHOLD))
        try:   # 8-word runs of every build prompt: seeing one in a reply means the prompt leaked
            words = " ".join(load_prompt(*p).text for p in (SYSTEM_PROMPT, USER_PROMPT)).split()
            self._prompt_ngrams = {" ".join(words[i:i + 8]).lower() for i in range(len(words) - 7)}
        except PromptError:
            self._prompt_ngrams = set()

    def check(self, ticket: Any, answer: GeneratedAnswer, retrieval: RetrievalResult) -> GuardrailResult:
        """Runs every check. Never raises: an error inside a check blocks the reply."""
        started = time.perf_counter()
        t = ticket if isinstance(ticket, NormalisedTicket) else normalise_ticket(ticket)
        checks = []
        for name, fn in (("structure", self._structure), ("private_data", self._private_data),
                         ("claim_support", self._claim_support), ("forbidden_claims", self._forbidden),
                         ("secret_request", self._secret_request), ("injection", self._injection),
                         ("tone", self._tone)):
            try:
                findings, stats = fn(t, answer, retrieval)
            except Exception as exc:   # a check that cannot run must not wave a reply through
                log.exception("check %s failed", name)
                findings, stats = [Finding(kind="check_error", severity="block",
                                           detail=f"The {name} check could not run ({type(exc).__name__}).")], {}
            checks.append(CheckResult(name=name, passed=not any(f.severity == "block" for f in findings),
                                      findings=findings, stats=stats))
        blocked = [c.name for c in checks if not c.passed]
        support = next(c for c in checks if c.name == "claim_support").stats
        return GuardrailResult(
            ticket_id=t.ticket_id, released=not blocked, blocked_by=blocked, reason=self._reason(checks, blocked),
            checks=checks, claims_checked=support.get("claims", 0), claims_unsupported=support.get("unsupported", 0),
            answer_digest=hashlib.sha256(answer.answer_text.encode()).hexdigest()[:16],
            generator=answer.generator, prompt_version=answer.prompt_version,
            checked_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            latency_ms=round((time.perf_counter() - started) * 1000, 2))

    @staticmethod
    def _reason(checks: Sequence[CheckResult], blocked: Sequence[str]) -> str:
        if not blocked:
            warns = [f.detail for c in checks for f in c.findings if f.severity == "warn"]
            return "Released: all checks passed." + (f" Warning: {warns[0]}" if warns else "")
        first = next(f for c in checks if c.name == blocked[0] for f in c.findings if f.severity == "block")
        more = f" ({len(blocked) - 1} other check(s) also failed.)" if len(blocked) > 1 else ""
        return f"Blocked before sending: {first.detail}{more}"

    # ---------------- checks: each returns (findings, stats)
    def _structure(self, t, a: GeneratedAnswer, r: RetrievalResult):
        f: list[Finding] = []
        if a.status == "failed":
            f.append(Finding(kind="generation_failed", severity="block", detail="Answer generation failed."))
        if not a.answer_text.strip():
            f.append(Finding(kind="empty", severity="block", detail="The reply is empty."))
        if a.status == "insufficient_information":
            if a.claims:
                f.append(Finding(kind="claims_with_dont_know", severity="block",
                                 detail="The reply says it does not know but still makes claims."))
            return f, {"status": a.status}
        if a.status == "answered" and not a.claims:
            f.append(Finding(kind="no_claims", severity="block", detail="An answered reply contains no claims."))
        retrieved = {p.chunk_id: p for p in r.passages}
        for i, c in enumerate(a.claims):
            if not c.citations:
                f.append(Finding(kind="uncited_claim", severity="block", claim_index=i,
                                 detail=f"Claim {i + 1} has no citation."))
            for cid in c.citations:
                if cid not in retrieved:
                    f.append(Finding(kind="citation_not_retrieved", severity="block", claim_index=i,
                                     detail=f"Claim {i + 1} cites {cid}, which was not among the retrieved passages."))
                elif self.corpus is not None and not self.corpus.verify(cid, retrieved[cid].text):
                    f.append(Finding(kind="citation_not_in_corpus", severity="block", claim_index=i,
                                     detail=f"Claim {i + 1} cites {cid}, which does not match the documentation."))
        cited = {cid for c in a.claims for cid in c.citations}
        if cited != {s.chunk_id for s in a.sources}:
            f.append(Finding(kind="source_list_mismatch", severity="block",
                             detail="The source list does not match the passages the claims cite."))
        return f, {"status": a.status, "claims": len(a.claims), "sources": len(a.sources)}

    def _private_data(self, t, a: GeneratedAnswer, r):
        text, f = a.answer_text, []
        for kind, pattern in PII_PATTERNS:
            for m in pattern.finditer(text):
                f.append(Finding(kind=kind, severity="block",
                                 detail=f"Possible {kind.replace('_', ' ')}: {mask(m.group())}."))
        for m in PHONE_PATTERN.finditer(text):
            digits = re.sub(r"\D", "", m.group())
            if 9 <= len(digits) <= 15 and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", m.group().strip()):
                f.append(Finding(kind="phone_number", severity="block",
                                 detail=f"Possible phone number: {mask(digits)}."))
        for m in CARD_PATTERN.finditer(text):
            digits = re.sub(r"\D", "", m.group())
            if 13 <= len(digits) <= 19 and _luhn(digits):
                f.append(Finding(kind="payment_card", severity="block",
                                 detail=f"Possible card number: {mask(digits)}."))
        if t.customer_name and len(t.customer_name) > 3 and t.customer_name.lower() in text.lower():
            f.append(Finding(kind="customer_name", severity="warn",
                             detail="The reply contains the customer's name, which the model was never given."))
        return f, {"detections": sum(x.severity == "block" for x in f)}

    def _claim_support(self, t, a: GeneratedAnswer, r: RetrievalResult):
        f, scores = [], []
        retrieved = {p.chunk_id: p.text for p in r.passages}
        for i, c in enumerate(a.claims):
            score, missing, numbers = support_score(c.text, " ".join(retrieved.get(cid, "") for cid in c.citations))
            scores.append(round(score, 3))
            if numbers:
                f.append(Finding(kind="unsupported_number", severity="block", claim_index=i,
                                 detail=f"Claim {i + 1} states {', '.join(numbers)}, which its sources do not."))
            elif score < self.support_threshold:
                f.append(Finding(kind="unsupported_claim", severity="block", claim_index=i,
                                 detail=f"Claim {i + 1} is not supported by its sources ({score:.0%} of its key "
                                        f"words appear there; {self.support_threshold:.0%} needed). "
                                        f"Unsupported words: {', '.join(missing[:6])}."))
        return f, {"claims": len(a.claims), "unsupported": len({x.claim_index for x in f}), "scores": scores,
                   "threshold": self.support_threshold}

    def _forbidden(self, t, a: GeneratedAnswer, r):
        text = a.answer_text if a.answer_text != DONT_KNOW else ""
        return [Finding(kind=k, severity="block", detail=f"The reply {_FORBIDDEN_LABELS[k]} (\"{m.group()[:60]}\").")
                for k, p in _FORBIDDEN for m in p.finditer(text)], {}

    def _secret_request(self, t, a: GeneratedAnswer, r):
        f = []
        for m in _SECRET_REQUEST.finditer(a.answer_text):
            if _NEGATION.search(a.answer_text[max(0, m.start() - 40):m.start()]):
                continue      # "never send a live credential to support" is advice, not a request
            f.append(Finding(kind="asks_for_secret", severity="block",
                             detail=f"The reply asks the customer for a {m.group(2).lower()}."))
        return f, {}

    def _injection(self, t, a: GeneratedAnswer, r):
        f = []
        if any(g in " ".join(a.answer_text.lower().split()) for g in self._prompt_ngrams):
            f.append(Finding(kind="instruction_leak", severity="block",
                             detail="The reply repeats part of the system's own instructions."))
        for m in _INJECTION_SIGNS.finditer(a.answer_text):
            f.append(Finding(kind="injection_sign", severity="block",
                             detail="The reply shows signs of following injected instructions "
                                    f"(\"{m.group()[:40]}\")."))
        attempt = bool(_INJECTION_SIGNS.search(t.text)) or "system:" in t.text.lower()
        return f, {"ticket_contains_injection_attempt": attempt}

    def _tone(self, t, a: GeneratedAnswer, r):
        f = [Finding(kind="insult_or_profanity", severity="block",
                     detail=f"The reply contains inappropriate language (\"{m.group()}\").")
             for m in _INSULT.finditer(a.answer_text)]
        for m in _SHOUT.finditer(a.answer_text):
            if not set(m.group().split()) <= _ACRONYM_OK:
                f.append(Finding(kind="shouting", severity="warn", detail="The reply contains a run of capitals."))
        return f, {}


# ------------------------------------------------------------------ the release gate
_default_store: Optional[DecisionStore] = None


def default_store() -> DecisionStore:
    global _default_store
    if _default_store is None:
        _default_store = DecisionStore()
    return _default_store


def release(ticket: Any, answer: GeneratedAnswer, retrieval: RetrievalResult,
            guardrails: Optional[Guardrails] = None, store: Optional[DecisionStore] = None,
            run_id: Optional[str] = None) -> Outbound:
    """The only way a reply is released. Always checks and always records. Never raises."""
    t = ticket if isinstance(ticket, NormalisedTicket) else normalise_ticket(ticket)
    guardrails = guardrails or Guardrails()
    try:
        result = guardrails.check(t, answer, retrieval)
    except Exception as exc:  # check() does not raise, but if it ever did, nothing is sent
        log.exception("guardrails crashed for %s", t.ticket_id)
        result = GuardrailResult(ticket_id=t.ticket_id, released=False, blocked_by=["guardrail_error"],
                                 reason=f"Blocked before sending: the guardrails could not run "
                                        f"({type(exc).__name__}).",
                                 checks=[], answer_digest="", checked_at=datetime.now(timezone.utc).isoformat())
    store = store or default_store()
    try:
        result.decision_id = store.log_guardrail(result, ticket_text=t.text, run_id=run_id)
        result.recorded = store.last_write_ok
    except Exception as exc:
        result.recorded = False
        log.error("could not record guardrail result for %s: %s", t.ticket_id, exc)
    return Outbound(ticket_id=t.ticket_id, released=result.released,
                    text=answer.answer_text if result.released else None, guardrail=result)


# ------------------------------------------------------------------ command line
def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    p = argparse.ArgumentParser(prog="python -m src.guardrail", description="Release gate for generated replies.")
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="ingest, classify, retrieve, route, generate, then the guardrails")
    r.add_argument("--input", required=True)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from collections import Counter

    from src.classify import load_or_train
    from src.generate import build_generator
    from src.ingest import load_tickets
    from src.retrieve import build_retriever
    from src.route import Router, RoutingConfig
    try:
        tickets = load_tickets(args.input)
    except (OSError, ValueError) as exc:
        log.error("could not read %s: %s", args.input, exc)
        return 1
    clf, ret, router, gen = load_or_train(), build_retriever(), Router(RoutingConfig.load()), build_generator()
    guardrails = Guardrails(corpus=ret.corpus)
    outs = []
    for t in tickets:
        rr = ret.retrieve(t)
        if router.decide(t, clf.classify(t), rr).action == "auto_respond":
            outs.append(release(t, gen.generate(t, rr), rr, guardrails))
    blocked = [o for o in outs if not o.released]
    for o in blocked[:3]:
        print(f"BLOCKED {o.ticket_id}: {o.guardrail.reason}")
    print(json.dumps({"checked": len(outs), "released": len(outs) - len(blocked), "blocked": len(blocked),
                      "blocked_by": dict(Counter(b for o in blocked for b in o.guardrail.blocked_by)),
                      "claims_checked": sum(o.guardrail.claims_checked for o in outs),
                      "claims_unsupported": sum(o.guardrail.claims_unsupported for o in outs)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
