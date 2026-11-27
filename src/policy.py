"""Resolution tiers and channel policies: how much automation each ticket gets, and where.

Tiered resolution (cheapest first). Every outcome records the tier that resolved it.

  Tier 0  rules and approved responses   greetings, out-of-scope, manipulation attempts, short
                                         definitions (src/scope.py). No search, no model.
  Tier 1  approved documentation content the best article is a strong match (score at or above the
                                         calibrated "approved" threshold, where the top article was
                                         right for at least 95% of development tickets): its own
                                         resolution steps are sent, cited. Local search only, no model.
  Tier 2  RAG generation                 a relevant article was found but the match is not strong
                                         enough to send the article as written: the model writes a
                                         reply from the retrieved passages, every claim cited and
                                         checked by the guardrails.
  Tier 3  human-assisted resolution      sensitive, low-confidence, undocumented or account-specific:
                                         a person resolves it, with Cirrus's summary, likely articles
                                         and a cited draft (the agent copilot).

Channel policies (CHANNEL_POLICIES). The four channels behave differently, so each has its own
ceiling on automation, and the rollout phase (ROLLOUT_PHASE) switches channels on in order:

  Channel                 Policy                                                       Phase
  docs_comment (API docs) answer narrow technical questions (Tiers 0-2); link related    1
                          articles; anything else goes to a person
  internal dashboard      the agent copilot: summaries, classification, drafts          1
  chat (live chat)        instant answers (Tiers 0-2), confidence-based escalation       2
  email                   send only Tiers 0-1 (approved content); a Tier 2 answer        3
                          becomes a draft for an agent to check and send (Tier 3)
  forum (community)       point to the documentation and existing threads (Tier 1, no   3
                          generated answer), and never reply twice in the same thread

A channel not yet switched on gets no automatic reply: its tickets go to a person with the
copilot's preparation. CHANNEL_POLICY=uniform turns the channel rules off (every channel as live
chat), to compare with the system before channel policies.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional


class Tier(IntEnum):
    RULES = 0
    RETRIEVAL = 1
    RAG = 2
    HUMAN = 3


TIER_NAMES = {0: "Tier 0: rules and approved responses", 1: "Tier 1: approved documentation content",
              2: "Tier 2: RAG generation", 3: "Tier 3: human-assisted resolution"}

# Intents that count as narrow technical questions (answered automatically on documentation pages).
TECHNICAL_INTENTS = {"api_usage_question", "api_key_issue", "authentication_failure", "rate_limit",
                     "webhook_issue", "integration_help", "configuration_help", "deployment_failure",
                     "rollback_request", "performance_degradation", "database_issue", "data_export",
                     "sso_configuration", "onboarding"}

# Plain-language topics, for the agent workspace and the knowledge gap dashboard.
INTENT_TOPICS = {
    "api_key_issue": "API authentication", "authentication_failure": "Authentication",
    "sso_configuration": "Single sign-on", "account_access": "Account permissions",
    "deployment_failure": "Deployment troubleshooting", "rollback_request": "Deployment troubleshooting",
    "configuration_help": "Configuration", "billing_query": "Billing and refunds",
    "quota_or_overage": "Billing and refunds", "api_usage_question": "API usage", "rate_limit": "Rate limits",
    "webhook_issue": "Webhooks", "integration_help": "Integrations", "performance_degradation": "Performance",
    "database_issue": "Databases", "data_export": "Data migration", "data_residency": "Data residency",
    "compliance_request": "Compliance", "security_incident": "Security", "feature_request": "Feature requests",
    "onboarding": "Getting started", "unclear_request": "Unclear requests",
}


def topic(intent: Optional[str]) -> str:
    return INTENT_TOPICS.get(intent or "", (intent or "Other").replace("_", " ").capitalize())


@dataclass(frozen=True)
class ChannelPolicy:
    label: str
    rollout_phase: int              # switched on from this phase
    max_auto_tier: int              # highest tier sent without a person
    narrow_technical_only: bool = False
    suggest_related: bool = False   # append related articles to the reply
    suggest_only: bool = False      # reply with pointers (articles, threads), never a generated answer
    reply_once: bool = False        # never reply twice in the same thread
    description: str = ""
    where: str = ""                 # for sentences: "on API documentation pages"


CHANNEL_POLICIES = {
    "docs_comment": ChannelPolicy("API documentation", 1, Tier.RAG, narrow_technical_only=True, suggest_related=True,
                                  description="Suggest relevant articles and answer narrow technical questions",
                                  where="on API documentation pages"),
    "chat": ChannelPolicy("Live chat", 2, Tier.RAG,
                          description="Instant answers with confidence-based escalation", where="in live chat"),
    "email": ChannelPolicy("Email", 3, Tier.RETRIEVAL,
                           description="Send verified answers only; drafts for an agent otherwise",
                           where="by email"),
    "forum": ChannelPolicy("Community forum", 3, Tier.RETRIEVAL, suggest_only=True, reply_once=True,
                           description="Point to documentation and existing threads; no duplicate replies",
                           where="on the community forum"),
}
INTERNAL_DASHBOARD = ChannelPolicy("Internal support dashboard", 1, Tier.HUMAN,
                                   description="Suggested answers, summaries and classification for agents")
UNIFORM = ChannelPolicy("Any channel (uniform policy)", 1, Tier.RAG, where="on this channel")


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def rollout_phase() -> int:
    try:
        return int(os.getenv("ROLLOUT_PHASE", "3"))
    except ValueError:
        return 3


def policy_for(channel: str) -> ChannelPolicy:
    if os.getenv("CHANNEL_POLICY", "channel").lower() == "uniform":
        return UNIFORM
    return CHANNEL_POLICIES.get(channel, CHANNEL_POLICIES["chat"])


@dataclass(frozen=True)
class TierDecision:
    tier: Tier
    escalate: bool = False
    reason: str = ""
    policy: Optional[ChannelPolicy] = None


def decide_tier(channel: str, intent: Optional[str], top_score: float, approved_threshold: float,
                via_hint: bool = False, already_replied: bool = False, not_resolved: bool = False) -> TierDecision:
    """For a ticket the router would answer automatically: Tier 1 or Tier 2, or a person (Tier 3)
    when this channel's policy does not allow that automatic reply."""
    p = policy_for(channel)
    strong = top_score >= approved_threshold and not via_hint
    tier = Tier.RETRIEVAL if strong else Tier.RAG
    how = (f"the best article scored {top_score:.3f}, at or above the {approved_threshold:.3f} approved-content "
           "threshold" if strong else
           f"the best article scored {top_score:.3f}, relevant but below the {approved_threshold:.3f} approved-"
           "content threshold" + (" (found through the intent hint)" if via_hint else ""))
    if not_resolved:
        return TierDecision(Tier.HUMAN, True, "The customer says the earlier answer did not solve it, so a person "
                                              "takes over rather than sending the same answer again.", p)
    if p.rollout_phase > rollout_phase():
        return TierDecision(Tier.HUMAN, True, f"Automatic replies {p.where} start in rollout phase "
                                              f"{p.rollout_phase} (now phase {rollout_phase()}).", p)
    if p.reply_once and already_replied:
        return TierDecision(Tier.HUMAN, True, "Cirrus has already replied in this thread; it does not reply "
                                              "twice.", p)
    if p.narrow_technical_only and intent not in TECHNICAL_INTENTS:
        return TierDecision(Tier.HUMAN, True, f"{_cap(p.where)} only narrow technical questions are "
                                              f"answered automatically; this is about {topic(intent).lower()}.", p)
    if p.suggest_only:
        if not strong:
            return TierDecision(Tier.HUMAN, True, f"{_cap(p.where)} Cirrus only points to articles and threads "
                                                  f"it is sure about; {how}.", p)
        return TierDecision(Tier.RETRIEVAL, False, f"Tier 1: {how}; replies {p.where} point to existing answers.", p)
    if tier > p.max_auto_tier:
        return TierDecision(Tier.HUMAN, True, f"{_cap(p.where)} only approved content is sent "
                                              f"automatically; {how}, so an agent checks the drafted reply first.", p)
    return TierDecision(tier, False, f"{TIER_NAMES[tier]}: {how}.", p)


# ------------------------------------------------------------------ community forum: existing threads
THREAD_MIN_SIMILARITY = 0.20


class _Threads:
    """Earlier forum threads, from the development tickets (the only forum history in the data), as
    a stand-in for CloudServe's community forum. Matched by TF-IDF similarity; only resolved threads
    (first-contact resolution in their history) are suggested."""

    def __init__(self):
        import json
        from pathlib import Path

        from sklearn.feature_extraction.text import TfidfVectorizer
        path = Path(__file__).resolve().parents[1] / "data" / "development_tickets.json"
        rows = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        seen, self.subjects, texts = set(), [], []
        for t in rows:
            subject = (t.get("subject") or "").strip()
            if (t.get("channel") == "forum" and subject and subject.lower() not in seen
                    and (t.get("history") or {}).get("first_contact_resolution")):
                seen.add(subject.lower())
                self.subjects.append(subject)
                texts.append(f"{subject}. {t.get('body', '')}")
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english") if texts else None
        self.matrix = self.vec.fit_transform(texts) if texts else None

    def similar(self, text: str, k: int = 2) -> list[str]:
        if self.vec is None or not (text or "").strip():
            return []
        from sklearn.metrics.pairwise import cosine_similarity
        sims = cosine_similarity(self.vec.transform([text]), self.matrix)[0]
        order = sims.argsort()[::-1][:k]
        return [self.subjects[i] for i in order if sims[i] >= THREAD_MIN_SIMILARITY]


_THREADS: Optional[_Threads] = None


def similar_threads(text: str, k: int = 2) -> list[str]:
    """Titles of existing community threads that look like this question (personal data masked)."""
    global _THREADS
    from src.logging_store import mask_pii
    if _THREADS is None:
        _THREADS = _Threads()
    return [mask_pii(s) for s in _THREADS.similar(text, k)]
