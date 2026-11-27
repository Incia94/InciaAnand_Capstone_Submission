"""Contextual in-app help: suggestions shown inside the CloudServe application, before a ticket exists.

The cheapest support ticket is the one never raised. Each application page knows what it is and,
when something failed, the error code. That is enough to show the right articles, with no model:

  page            what is suggested
  deployment      known deployment errors: health checks, rollbacks, builds, configuration
  api             authentication and rate-limit guidance
  billing         invoice terms, plan changes, spend controls
  account         permissions, roles, sign-in and MFA
  support_form    as the customer types, the articles that would answer them, before they submit

Suggestions come from (1) the error code, looked up in KNOWN_ERRORS (codes the articles cover; the
mapping is maintained by the support team), (2) the customer's own words, searched with the page's
categories as a metadata filter (src/retrieve.py), and (3) the page's own starting articles.
A search that finds nothing is logged as a knowledge gap (src/gaps.py).

Account-specific information (the customer's own keys, usage or deployments) is deliberately left
out: it would need authorised, read-only integrations, planned for later.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from src.retrieve import _excerpt_of


@dataclass(frozen=True)
class Page:
    label: str
    categories: tuple
    starters: tuple                 # articles shown before anything is typed
    intro: str


PAGES = {
    "deployment": Page("Deployments", ("deployment", "performance", "integration"),
                       ("DOC-DEPLOY-001", "DOC-DEPLOY-002", "DOC-DEPLOY-003"),
                       "Having trouble with a deployment? These cover the most common failures."),
    "api": Page("API management", ("api", "authentication", "security"),
                ("DOC-AUTH-004", "DOC-API-001", "DOC-API-003"),
                "Authentication, keys and rate limits for the CloudServe API."),
    "billing": Page("Billing", ("billing",), ("DOC-BILL-001", "DOC-BILL-002", "DOC-BILL-003"),
                    "How your invoice, plan changes and spend controls work."),
    "account": Page("Account settings", ("account", "authentication", "security"),
                    ("DOC-ACCT-001", "DOC-AUTH-002", "DOC-AUTH-003"),
                    "Team members, roles and permissions, sign-in and multi-factor authentication."),
    "support_form": Page("Support form", (), ("DOC-ONB-001",),
                         "Before you submit, these articles may already answer your question."),
}

# Error codes the documentation covers, with the articles that explain them. Maintained by support.
KNOWN_ERRORS = {
    "401": ("Unauthorized: the API key or token is missing, expired, revoked or invalid.",
            ("DOC-AUTH-004", "DOC-AUTH-001")),
    "403": ("Forbidden: the key or user is authenticated but lacks the scope or role for this action.",
            ("DOC-AUTH-004", "DOC-ACCT-001")),
    "429": ("Too Many Requests: the request rate or quota for your plan was exceeded.",
            ("DOC-API-001", "DOC-BILL-003")),
    "HEALTH_CHECK_FAILED": ("The new release did not pass its health check, so traffic stayed on the old one.",
                            ("DOC-DEPLOY-001", "DOC-DEPLOY-002")),
    "BUILD_FAILED": ("The build stopped, often while resolving dependencies.", ("DOC-DEPLOY-003",)),
    "502": ("Bad Gateway: the application did not respond properly; check its health and logs.",
            ("DOC-DEPLOY-001", "DOC-PERF-001")),
    "503": ("Service Unavailable: no healthy instance could take the request.",
            ("DOC-PERF-002", "DOC-DEPLOY-001")),
    "504": ("Gateway Timeout: the application took too long to respond.", ("DOC-PERF-001", "DOC-PERF-003")),
    "SPEND_CAP_REACHED": ("Your spend cap was reached, so usage beyond it is paused.", ("DOC-BILL-003",)),
}


@dataclass
class Suggestion:
    doc_id: str
    title: str
    excerpt: str
    why: str                        # "error 429", "matches what you typed", "common on this page"
    score: Optional[float] = None


@dataclass
class Help:
    page: str
    intro: str
    error: Optional[str] = None     # the plain explanation of a known error code
    suggestions: list = field(default_factory=list)
    gap: bool = False               # the customer's words matched nothing: logged as a knowledge gap


def _article(retriever: Any, doc_id: str) -> Optional[Any]:
    return next((c for c in retriever.corpus.chunks if c.doc_id == doc_id), None)


def suggest(retriever: Any, page: str, error_code: Optional[str] = None, query: Optional[str] = None,
            k: int = 3, store: Any = None) -> Help:
    """What to show on this page. No model is called."""
    p = PAGES.get(page, PAGES["support_form"])
    out = Help(page=page, intro=p.intro)
    picked: dict[str, Suggestion] = {}

    def add(doc_id: str, why: str, score: Optional[float] = None):
        c = _article(retriever, doc_id)
        if c is not None and doc_id not in picked:
            picked[doc_id] = Suggestion(doc_id, c.metadata.get("title", doc_id), _excerpt_of(c.text, 240), why,
                                        score)

    code = (error_code or "").strip().upper().replace("HTTP ", "")
    if code:
        known = KNOWN_ERRORS.get(code)
        if known:
            out.error = f"{code} {known[0]}" if code[0].isdigit() else known[0]
            for d in known[1]:
                add(d, f"explains error {code}")
    if query and query.strip():
        filters = {"category": list(p.categories)} if p.categories else None
        r = retriever.retrieve({"ticket_id": f"HELP-{page}", "body": f"{query} {code}".strip()}, k=k,
                               filters=filters)
        for passage in r.passages:
            add(passage.doc_id, "matches what you described", passage.score)
        if not r.passages:
            out.gap = True
            if store is not None:
                from src.gaps import record_search_gap
                record_search_gap(store, query, f"the {p.label} page", topic_hint=p.label)
    if not query and not code:
        for d in p.starters:
            add(d, f"common on the {p.label} page")
    out.suggestions = list(picked.values())[:k]
    return out
