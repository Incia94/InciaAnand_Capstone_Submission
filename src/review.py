"""Review before sending: Cirrus prepares the reply, a person approves it.

REPLY_MODE (environment):
  review  (default) nothing Cirrus writes reaches a customer until a person approves it. A ticket
          Cirrus could answer is held as "pending_review" with its cited reply; an escalated ticket
          is held as "needs_agent" with the cited draft prepared for the agent.
  auto    Cirrus sends the replies that pass every check itself (the original behaviour); escalated
          tickets still wait here for an agent.
Fixed replies (greetings, out-of-scope, refusals: Tier 0, src/scope.py) are constant, approved text
and are sent in both modes.

For each held reply the reviewer can
  approve   send it as written;
  edit      send their own version: the edited text is checked again (private data and secrets
            block it; claims about money or fixes and the tone are flagged), and the change is logged;
  reject    send nothing (the ticket then goes to an agent to answer from scratch).

Every action is written to the decision log as the "review" stage, with who did it, what was
sent (personal data masked) and whether it was edited, so the audit trail runs from the ticket to
the reply the customer actually received. "Sending" here means recording the reply as approved
and returning it: delivery to email, chat or the forum is the channel integration's job.

    python -m src.review list                          # what is waiting
    python -m src.review show TICKET_ID                 # the ticket, the reply, its sources and checks
    python -m src.review approve TICKET_ID --reviewer "Sofia Restrepo"
    python -m src.review approve TICKET_ID --reviewer "Sofia Restrepo" --text-file my_reply.txt
    python -m src.review reject TICKET_ID --reviewer "Sofia Restrepo" --reason "needs account check"
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional, Sequence

from src.logging_store import mask_pii

DEFAULT_REVIEW_DB = "./storage/reviews.db"
PENDING = ("pending_review", "needs_agent")


def reply_mode() -> str:
    return "auto" if os.getenv("REPLY_MODE", "review").strip().lower() == "auto" else "review"


class ReviewError(ValueError):
    """The action cannot be done (unknown ticket, already handled, edited text fails a check)."""


class ReviewQueue:
    """Replies waiting for a person, in a small SQLite file (by default next to the decision log)."""

    COLUMNS = ("ticket_id", "created_at", "updated_at", "status", "kind", "channel", "tier", "intent", "reason",
               "ticket_text", "proposed_reply", "sources", "guardrails", "handoff", "final_reply", "reviewer",
               "edited", "review_note", "run_id")

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path or os.getenv("REVIEW_DB", DEFAULT_REVIEW_DB))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._conn() as c:
            c.execute("CREATE TABLE IF NOT EXISTS reviews (ticket_id TEXT PRIMARY KEY, created_at REAL, "
                      "updated_at REAL, status TEXT, kind TEXT, channel TEXT, tier INTEGER, intent TEXT, "
                      "reason TEXT, ticket_text TEXT, proposed_reply TEXT, sources TEXT, guardrails TEXT, "
                      "handoff TEXT, final_reply TEXT, reviewer TEXT, edited INTEGER, review_note TEXT, "
                      "run_id TEXT)")

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(str(self.path), timeout=10)
        c.row_factory = sqlite3.Row
        return c

    def hold(self, outcome: Any, ticket_text: str, kind: str) -> None:
        """Queue a reply (kind "answer") or an escalation's draft (kind "escalation")."""
        draft = outcome.draft_reply or {}
        drafted = draft.get("text") if draft.get("status") == "drafted" else ""
        proposed = outcome.reply if kind == "answer" else drafted
        row = {"ticket_id": outcome.ticket_id, "created_at": time.time(), "updated_at": time.time(),
               "status": "pending_review" if kind == "answer" else "needs_agent", "kind": kind,
               "channel": outcome.channel, "tier": outcome.tier, "intent": outcome.intent, "reason": outcome.reason,
               "ticket_text": mask_pii(ticket_text or ""), "proposed_reply": proposed or "",
               "sources": json.dumps(outcome.sources or draft.get("sources") or [], default=str),
               "guardrails": json.dumps(outcome.guardrail or {"draft_checks_passed": draft.get("guardrail_passed"),
                                                              "issues": draft.get("guardrail_issues", [])},
                                        default=str),
               "handoff": json.dumps(outcome.handoff or {}, default=str), "final_reply": None, "reviewer": None,
               "edited": 0, "review_note": None, "run_id": outcome.run_id}
        with self._lock:
            conn = self._conn()
            try:
                with conn:
                    conn.execute(f"INSERT OR REPLACE INTO reviews ({', '.join(row)}) "
                                 f"VALUES ({', '.join('?' * len(row))})", list(row.values()))
            finally:
                conn.close()

    def get(self, ticket_id: str) -> Optional[dict]:
        conn = self._conn()
        try:
            r = conn.execute("SELECT * FROM reviews WHERE ticket_id = ?", (ticket_id,)).fetchone()
        finally:
            conn.close()
        return self._out(r) if r else None

    def list(self, status: Optional[str] = "pending") -> list[dict]:
        conn = self._conn()
        try:
            if status == "pending":
                rows = conn.execute(f"SELECT * FROM reviews WHERE status IN {PENDING} ORDER BY created_at").fetchall()
            elif status:
                rows = conn.execute("SELECT * FROM reviews WHERE status = ? ORDER BY created_at", (status,)).fetchall()
            else:
                rows = conn.execute("SELECT * FROM reviews ORDER BY created_at").fetchall()
        finally:
            conn.close()
        return [self._out(r) for r in rows]

    def _set(self, ticket_id: str, **fields: Any) -> None:
        fields["updated_at"] = time.time()
        with self._lock:
            conn = self._conn()
            try:
                with conn:
                    conn.execute(f"UPDATE reviews SET {', '.join(f'{k} = ?' for k in fields)} WHERE ticket_id = ?",
                                 [*fields.values(), ticket_id])
            finally:
                conn.close()

    @staticmethod
    def _out(r: sqlite3.Row) -> dict:
        d = dict(r)
        for k in ("sources", "guardrails", "handoff"):
            try:
                d[k] = json.loads(d[k]) if d.get(k) else None
            except ValueError:
                pass
        d["edited"] = bool(d.get("edited"))
        return d


# ------------------------------------------------------------------ checking a reviewer's own text
def check_reviewed_text(text: str, item: dict, guardrails: Any) -> dict:
    """The guardrails on text a person wrote or changed. Private data and secrets block it: they
    must never reach a customer, whoever wrote them. Money or fix claims and tone are flagged for
    the reviewer, who may have the authority to make them. Citations are not required of a person."""
    from src.generate import GeneratedAnswer
    from src.ingest import normalise_ticket
    from src.retrieve import RetrievalResult
    t = normalise_ticket({"ticket_id": item["ticket_id"], "channel": item.get("channel") or "email",
                          "body": item.get("ticket_text") or ""})
    a = GeneratedAnswer(ticket_id=item["ticket_id"], status="answered", claims=[], answer_text=text, sources=[],
                        generator="human")
    r = RetrievalResult(ticket_id=item["ticket_id"], status="ok")
    block_checks = {"private_data": guardrails._private_data, "secret_request": guardrails._secret_request}
    warn_checks = {"forbidden_claims": guardrails._forbidden, "tone": guardrails._tone}
    blocked, warnings = [], []
    for name, fn in block_checks.items():
        findings, _ = fn(t, a, r)
        blocked += [f"{name}: {f.detail}" for f in findings if f.severity == "block"]
    for name, fn in warn_checks.items():
        findings, _ = fn(t, a, r)
        warnings += [f"{name}: {f.detail}" for f in findings]
    return {"passed": not blocked, "blocked": blocked, "warnings": warnings}


# ------------------------------------------------------------------ the actions
def approve(queue: ReviewQueue, store: Any, guardrails: Any, ticket_id: str, reviewer: str,
            text: Optional[str] = None) -> dict:
    """Send the held reply, or the reviewer's version of it. Returns what was sent and the checks."""
    item = queue.get(ticket_id)
    if item is None:
        raise ReviewError(f"nothing is waiting for review for {ticket_id}")
    if item["status"] not in PENDING:
        raise ReviewError(f"{ticket_id} was already handled ({item['status']} by {item['reviewer']})")
    if not (reviewer or "").strip():
        raise ReviewError("a reviewer name is required")
    final = (text if text is not None else item["proposed_reply"] or "").strip()
    if not final:
        raise ReviewError(f"{ticket_id} has no drafted reply: write one (--text-file) to send")
    edited = text is not None and final != (item["proposed_reply"] or "").strip()
    checks = check_reviewed_text(final, item, guardrails) if (edited or item["kind"] == "escalation") else \
        {"passed": True, "blocked": [], "warnings": [], "note": "unchanged reply that passed every guardrail"}
    if not checks["passed"]:
        store.record(ticket_id, "review", "blocked", f"{reviewer} tried to send a reply that failed a check: "
                     + "; ".join(checks["blocked"]), prediction=reviewer, run_id=item.get("run_id"),
                     input_text=final, model_name="person", details={"edited": edited, "checks": checks})
        raise ReviewError("not sent: " + "; ".join(checks["blocked"]))
    action = "edited_and_sent" if edited else "sent_after_review"
    note = ("Sent after review" + (", edited by the reviewer" if edited else ", as Cirrus wrote it") +
            (f". Flagged for the reviewer: {'; '.join(checks['warnings'])}" if checks["warnings"] else "."))
    store.record(ticket_id, "review", action, f"{reviewer}: {note}", prediction=reviewer, run_id=item.get("run_id"),
                 input_text=final, model_name="person",
                 sources=item.get("sources") or [],
                 details={"edited": edited, "kind": item["kind"], "checks": checks, "reply": mask_pii(final),
                          "proposed_reply": mask_pii(item["proposed_reply"] or "")})
    queue._set(ticket_id, status="sent", final_reply=final, reviewer=reviewer, edited=int(edited), review_note=note)
    return {"ticket_id": ticket_id, "status": "sent", "action": action, "reply": final, "edited": edited,
            "checks": checks}


def reject(queue: ReviewQueue, store: Any, ticket_id: str, reviewer: str, reason: str = "") -> dict:
    item = queue.get(ticket_id)
    if item is None:
        raise ReviewError(f"nothing is waiting for review for {ticket_id}")
    if item["status"] not in PENDING:
        raise ReviewError(f"{ticket_id} was already handled ({item['status']})")
    if not (reviewer or "").strip():
        raise ReviewError("a reviewer name is required")
    note = reason or "rejected by the reviewer"
    store.record(ticket_id, "review", "rejected", f"{reviewer} rejected the drafted reply: {note}",
                 prediction=reviewer, run_id=item.get("run_id"), model_name="person",
                 details={"kind": item["kind"], "proposed_reply": mask_pii(item["proposed_reply"] or "")})
    queue._set(ticket_id, status="rejected", reviewer=reviewer, review_note=note)
    return {"ticket_id": ticket_id, "status": "rejected", "reason": note}


def close(queue: ReviewQueue, store: Any, ticket_id: str, reviewer: str, note: str = "") -> dict:
    """Resolved without sending this reply (for example, solved on a call). Logged."""
    item = queue.get(ticket_id)
    if item is None:
        raise ReviewError(f"nothing is waiting for review for {ticket_id}")
    if item["status"] not in PENDING:
        raise ReviewError(f"{ticket_id} was already handled ({item['status']})")
    if not (reviewer or "").strip():
        raise ReviewError("a reviewer name is required")
    note = note or "resolved by the agent without this reply"
    store.record(ticket_id, "review", "resolved_without_reply", f"{reviewer}: {note}", prediction=reviewer,
                 run_id=item.get("run_id"), model_name="person", details={"kind": item["kind"]})
    queue._set(ticket_id, status="resolved", reviewer=reviewer, review_note=note)
    return {"ticket_id": ticket_id, "status": "resolved", "note": note}


# ------------------------------------------------------------------ command line
def _print_item(item: dict) -> None:
    print(f"{item['ticket_id']}  [{item['status']}]  {item['channel']} · tier {item['tier']} · {item['intent']}")
    print(f"\nWhy: {item['reason']}\n\nCustomer wrote (personal data masked):\n{item['ticket_text']}\n")
    print("Reply Cirrus prepared:" if item["proposed_reply"] else "No reply drafted.")
    if item["proposed_reply"]:
        print(item["proposed_reply"])
    src = item.get("sources") or []
    if src and "Sources:" not in (item["proposed_reply"] or ""):
        print("\nSources: " + ", ".join(f"{s.get('title', '')} ({s.get('doc_id')})" for s in src))
    if item.get("final_reply"):
        print(f"\nSent by {item['reviewer']}{' (edited)' if item['edited'] else ''}:\n{item['final_reply']}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    p = argparse.ArgumentParser(prog="python -m src.review", description="Review replies before they are sent.")
    p.add_argument("--db", type=Path, default=None, help=f"review queue (default {DEFAULT_REVIEW_DB})")
    sub = p.add_subparsers(dest="command", required=True)
    ls = sub.add_parser("list", help="replies waiting for review")
    ls.add_argument("--status", default="pending", help="pending (default), sent, rejected or all")
    sh = sub.add_parser("show", help="one held reply in full")
    sh.add_argument("ticket_id")
    ap = sub.add_parser("approve", help="send the reply, as written or edited")
    ap.add_argument("ticket_id")
    ap.add_argument("--reviewer", required=True)
    ap.add_argument("--text-file", type=Path, help="send this text instead (an edited reply)")
    rj = sub.add_parser("reject", help="send nothing")
    rj.add_argument("ticket_id")
    rj.add_argument("--reviewer", required=True)
    rj.add_argument("--reason", default="")
    args = p.parse_args(argv)

    queue = ReviewQueue(args.db)
    if args.command == "list":
        items = queue.list(None if args.status == "all" else args.status)
        if not items:
            print("Nothing waiting." if args.status == "pending" else "None.")
        for it in items:
            first = " ".join((it["ticket_text"] or "").split())[:70]
            print(f"{it['ticket_id']:<14} {it['status']:<15} {it['channel']:<13} tier {it['tier']}  {first}")
        return 0
    if args.command == "show":
        item = queue.get(args.ticket_id)
        if item is None:
            print(f"No review item for {args.ticket_id}.", file=sys.stderr)
            return 1
        _print_item(item)
        return 0
    from src.guardrail import Guardrails
    from src.logging_store import DecisionStore
    store = DecisionStore()
    try:
        if args.command == "approve":
            text = args.text_file.read_text(encoding="utf-8") if args.text_file else None
            res = approve(queue, store, Guardrails(), args.ticket_id, args.reviewer, text)
            print(f"Sent ({res['action']}).")
            for w in res["checks"]["warnings"]:
                print(f"Flagged: {w}")
            print("\n" + res["reply"])
        else:
            res = reject(queue, store, args.ticket_id, args.reviewer, args.reason)
            print(f"Rejected: {res['reason']}")
        return 0
    except ReviewError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        store.close()


if __name__ == "__main__":
    sys.exit(main())
