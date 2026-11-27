"""The support workspace's views, in plain words for supervisors and agents.

The pipeline's own vocabulary (intent, tier, retrieval, guardrail...) is translated here into the
words a support team uses: topic, how it was handled, help articles, safety checks. The screens
(streamlit_app.py) only display what these functions return.
"""
from __future__ import annotations

import re
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Optional

from src.formatting import docs_url
from src.hub import OPEN, PRIORITY

CHANNELS = {"email": "Email", "chat": "Live chat", "forum": "Forum", "docs_comment": "Docs comment"}
HANDLED = {"answered": "Answered by Cirrus", "pending_review": "Reply ready to check", "escalated": "Needs an agent",
           "merged": "Duplicate (merged)", "resolved": "Resolved", "rejected": "Closed"}
WORK = {"open": "Open", "in_progress": "In progress", "resolved": "Resolved"}
STEPS = {"deduplication": "Duplicate check", "ingestion": "Received", "screening": "First check",
         "translation": "Translated", "classification": "Topic identified", "retrieval": "Help articles searched",
         "routing": "Answer or escalate", "tiering": "How to answer", "generation": "Reply written",
         "validation": "Safety checks", "agent_draft": "Draft for the agent", "outcome": "Result",
         "assignment": "Assigned", "human_review": "Agent action", "kill_switch": "Automatic replies paused",
         "error": "Error", "search": "Help article search"}
ACTIONS = {"auto_respond": "answer", "escalate": "send to an agent", "block": "stopped", "converse": "reply",
           "pending_review": "reply ready to check", "sent": "reply sent", "escalated": "sent to an agent",
           "merged": "merged into an open ticket", "sent_after_review": "reply sent as prepared",
           "edited_and_sent": "reply edited and sent", "rejected": "reply not sent",
           "resolved_without_reply": "resolved without a reply", "tier_0": "fixed reply", "tier_1":
           "reply from a help article", "tier_2": "reply written from help articles", "tier_3": "an agent",
           "normalised": "received", "unprocessable": "could not be read", "classified": "topic identified",
           "fallback": "topic unclear", "ok": "articles found", "no_relevant_passage": "no matching article",
           "answered": "reply written", "insufficient_information": "articles did not answer it",
           "drafted": "draft prepared", "not_drafted": "no draft", "release": "passed", "new": "not a duplicate",
           "support": "support question", "assigned": "assigned", "translated": "translated",
           "not_translated": "not translated", "trimmed": "off-topic part removed",
           "blocked": "not sent (failed a check)"}
_NOISE = [r"\s*Note: documentation search ran in backup mode\.", r"\s*\(REPLY_MODE=\w+\)",
          r"\s*Search ran in backup mode\.", r"^Escalated to a person\.\s*", r"\s*Degraded: [^.]*\.?",
          r"\s*Held for a person to approve before sending\.", r"\s*\((?:generate_answer|PR-)[^)]*\)"]
# The pipeline's technical phrases, in the words a support team uses.
_WORDS = [(r"Tier 0: rules and approved responses", "A fixed reply"),
          (r"Tier 1: approved documentation content", "Reply taken from a help article"),
          (r"Tier 2: RAG generation", "Reply written from the help articles"),
          (r"Tier 3: human-assisted resolution", "An agent answers"),
          (r"Drafted (\d+) cited claim\(s\) with the \w+ generator\.?",
           r"Reply written: \1 points, each citing a help article."),
          (r"Cited draft prepared for the agent; not sent\.", "Draft reply prepared for the agent."),
          (r"Guardrails found nothing\.", "It passed the safety checks."),
          (r"Released: all checks passed\.", "Passed every safety check."),
          (r"Intent (\w+) at (\d+)%, urgency (\w+) at \d+%\.",
           lambda m: f"Topic: {m.group(1).replace('_', ' ')} ({m.group(2)}% sure), urgency {m.group(3)}."),
          (r"(\d+) passage\(s\) above threshold via \w+\.", r"\1 matching help article(s) found."),
          (r"No passage returned: .*", "No help article matched closely enough."),
          (r"the best article scored [\d.]+, (?:at or above|relevant but below) the [\d.]+ approved-content threshold",
           "based on how closely the help article matches"),
          (r"\(best match scored [\d.]+; at least [\d.]+ is needed\)", ""),
          (r"the system is (\d+)% sure this is an? '([^']+)' ticket",
           lambda m: f"Cirrus is {m.group(1)}% sure this is about {m.group(2).lower()}"),
          (r"The system is (\d+)% sure this is an? '([^']+)' ticket \(at least \d+% is required to answer without a "
           r"person\)",
           lambda m: f"Cirrus is only {m.group(1)}% sure this is about {m.group(2).lower()}, not sure enough to "
                     f"answer on its own"),
          (r"Ticket from (\w+) normalised(?:; repairs: [^.]*)?\.", r"Received by \1."),
          (r"A support request: routed through the pipeline\.", "A support question."),
          (r"No open ticket from this customer on another channel looks like this one\.", "Not a duplicate."),
          (r"^Answered automatically:\s*", ""), (r"\s*\(found through the intent hint\)", ""),
          (r"Escalated to a person\.\s*", "")]


def plain(text: Optional[str]) -> str:
    out = text or ""
    for pattern in _NOISE:
        out = re.sub(pattern, "", out)
    for pattern, repl in _WORDS:
        out = re.sub(pattern, repl, out)
    out = re.sub(r"\s+([.,;:])", r"\1", out)
    return re.sub(r"\s{2,}", " ", out).strip()


def ago(ts: Any) -> str:
    try:
        secs = max(0, time.time() - float(ts))
    except (TypeError, ValueError):
        return ""
    if secs < 3600:
        return f"{int(secs // 60)}m ago"
    if secs < 86400:
        return f"{int(secs // 3600)}h ago"
    return f"{int(secs // 86400)}d ago"


def title(t: dict) -> str:
    text = (t.get("subject") or "").strip() or " ".join((t.get("text") or "").split())
    text = re.sub(r"^Comment on:\s*", "", text)
    sentences = [x.strip().rstrip(".") for x in re.split(r"(?<=[.?!])\s", text) if x.strip()]
    while len(sentences) > 1 and len(sentences[0].split()) <= 2:     # skip "Thanks." / "Hi there."
        sentences.pop(0)
    text = sentences[0] if sentences else text
    return text if len(text) <= 60 else text[:57].rsplit(" ", 1)[0] + "…"


def priority(t: dict) -> str:
    h = t.get("handoff") if isinstance(t.get("handoff"), dict) else {}
    return PRIORITY.get(h.get("priority") or t.get("priority") or t.get("urgency") or "", "Medium")


def card(t: dict) -> dict:
    return {"ticket_id": t["ticket_id"], "title": title(t), "channel": CHANNELS.get(t["channel"], t["channel"]),
            "priority": priority(t), "age": ago(t.get("created_at")), "handled": HANDLED.get(t["status"], t["status"]),
            "status": t["status"], "work_status": WORK.get(t.get("work_status") or "open", "Open"),
            "assignee": t.get("assignee_name") or "", "customer": t.get("customer_name") or "Unknown customer"}


def _articles(t: dict) -> list[dict]:
    src = t.get("sources") if isinstance(t.get("sources"), list) else []
    h = t.get("handoff") if isinstance(t.get("handoff"), dict) else {}
    docs = src or h.get("documents") or h.get("sources") or []
    seen, out = set(), []
    for d in docs:
        doc = d.get("doc_id")
        if doc and doc not in seen:
            seen.add(doc)
            out.append({"doc_id": doc, "title": d.get("title") or doc, "url": docs_url(doc)})
    return out[:3]


def findings(t: dict, review: Optional[dict] = None) -> str:
    from src.policy import topic
    what = topic(t.get("intent")).lower() if t.get("intent") and t.get("intent") != "not_a_ticket" else "support"
    status = t["status"]
    if status in ("resolved", "rejected") and review:            # describe it as it was when it arrived
        status = "pending_review" if review.get("kind") == "answer" else "escalated"
    if status == "pending_review":
        return (f"This looks like a {what} question. Cirrus found the answer in the help articles below and "
                "prepared a reply: check it, change it if needed, and send it.")
    if status == "escalated":
        why = plain(t.get("reason"))
        lead = f"This looks like a {what} question. " if what != "support" else ""
        why = why[:1].lower() + why[1:] if why else "it needs a person."
        return f"{lead}Cirrus did not answer it on its own: {why}"
    return plain(t.get("reason"))


def trail(decisions: list[dict]) -> list[dict]:
    out = []
    for d in decisions:
        step = STEPS.get(d["stage"], d["stage"].replace("_", " ").capitalize())
        action = ACTIONS.get(d["action_taken"], (d["action_taken"] or "").replace("_", " "))
        when = d.get("timestamp") or ""
        out.append({"step": step, "decision": action, "why": plain(d.get("reason")), "at": when[11:19]})
    return out


def agent_view(hub: Any, ticket_id: str) -> Optional[dict]:
    t = hub.tickets.get(ticket_id)
    if t is None:
        return None
    f = t.get("formatted") if isinstance(t.get("formatted"), dict) else {}
    review = hub.pipeline.reviews.get(ticket_id)
    can_send = bool(review and review["status"] in ("pending_review", "needs_agent"))
    return {**card(t), "customer_report": t.get("text") or "", "customer_id": t.get("customer_id") or "",
            "findings": findings(t, review), "articles": _articles(t),
            "reply": t.get("saved_reply") or f.get("body") or "", "subject": f.get("subject") or "",
            "reply_prepared": bool(f.get("body")), "can_send": can_send,
            "sent_reply": f.get("body") if t["status"] in ("resolved", "answered") else "",
            "conversation": [{"from": "Customer" if m["role"] == "customer" else (m.get("author") or "Cirrus"),
                              "channel": CHANNELS.get(m["channel"], m["channel"]), "text": m["text"]}
                             for m in hub.tickets.messages(ticket_id)],
            "trail": trail(hub.pipeline.store.export(ticket_id=ticket_id))}


def queue(hub: Any, tab: str = "unassigned") -> list[dict]:
    rows = [t for t in hub.tickets.list() if t["status"] in OPEN or t.get("assignee")]
    if tab == "unassigned":
        rows = [t for t in rows if not t.get("assignee") and t["status"] in OPEN]
    elif tab == "assigned":
        rows = [t for t in rows if t.get("assignee")]
    order = {"High": 0, "Medium": 1, "Low": 2}
    return sorted((card(t) for t in rows), key=lambda c: (order.get(c["priority"], 1), c["ticket_id"]))


def reports(hub: Any, days: int = 14) -> dict:
    tickets = [t for t in hub.tickets.list(limit=100000) if t["status"] != "new"]
    today = datetime.now(timezone.utc).date()
    per_day = Counter(datetime.fromtimestamp(float(t["created_at"]), timezone.utc).date() for t in tickets)
    volume = [{"day": (today.fromordinal(today.toordinal() - i)).strftime("%d %b"),
               "tickets": per_day.get(today.fromordinal(today.toordinal() - i), 0)} for i in range(days - 1, -1, -1)]
    store = hub.pipeline.store
    decisions = store.export()
    outcome = {d["ticket_id"]: d for d in decisions if d["stage"] == "outcome"}
    steps = Counter(d["ticket_id"] for d in decisions)
    missing = [t["ticket_id"] for t in tickets if t["ticket_id"] not in outcome]
    log = []
    for t in sorted(tickets, key=lambda x: float(x["created_at"]), reverse=True):
        o = outcome.get(t["ticket_id"]) or {}
        log.append({"ticket_id": t["ticket_id"], "title": title(t), "channel": CHANNELS.get(t["channel"], t["channel"]),
                    "decision": HANDLED.get(t["status"], t["status"]), "why": plain(o.get("reason") or t.get("reason")),
                    "at": (o.get("timestamp") or "")[11:16], "steps": steps.get(t["ticket_id"], 0),
                    "assignee": t.get("assignee_name") or ""})
    prepared = sum(1 for t in tickets if (t.get("formatted") or {}) and t["status"] != "merged")
    return {
        "total": len(tickets),
        "volume": volume,
        "by_channel": [{"channel": CHANNELS.get(k, k), "tickets": v}
                       for k, v in Counter(t["channel"] for t in tickets).most_common()],
        "handled": [{"how": HANDLED.get(k, k), "tickets": v} for k, v in Counter(t["status"] for t in tickets)
                    .most_common()],
        "work": {"unassigned": sum(1 for t in tickets if t["status"] in OPEN and not t.get("assignee")),
                 "assigned": sum(1 for t in tickets if t.get("assignee") and t["status"] in OPEN),
                 "resolved": sum(1 for t in tickets if t["status"] in ("resolved", "answered")),
                 "replies_prepared": prepared},
        "decision_log": {"decisions": len(decisions), "tickets": len(tickets), "every_ticket_logged": not missing,
                         "missing": missing[:20], "tickets_log": log},
    }


def ticket_steps(hub: Any, ticket_id: str) -> list[dict]:
    return trail(hub.pipeline.store.export(ticket_id=ticket_id))
