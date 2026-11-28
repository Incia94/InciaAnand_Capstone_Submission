"""Cirrus AI, the support resolution hub: one intelligence engine serving customers and employees.

Every ticket, from every channel, takes the same path:

  native payload --(src/channels.py adapter)--> unified ticket
      --> conversation history: earlier messages in the same thread are attached
      --> cross-channel deduplication: the same open problem from the same customer on another
          channel is merged into the existing ticket instead of being worked twice
      --> the pipeline (src/api.py): screen, classify, search, route, tier, generate, guardrails, log
      --> channel-aware formatting (src/formatting.py)
      --> central ticket storage: the ticket, its status and every message, for the employee dashboard

Central storage (TICKETS_DB, default tickets.db beside the decision log) holds two tables:
  tickets   one row per ticket: channel, customer, status, how it was handled (tier, decided_by),
            the reply and its channel-formatted version, and duplicate_of for merged tickets;
  messages  the conversation: every customer message, every reply Cirrus or an agent sent, per ticket,
            with the customer key, so a customer's history can be read across all four channels.
Personal data in stored text is masked (src/logging_store.mask_pii); the customer's name and id are
kept in their own fields, for staff.

Statuses: answered (Cirrus replied), pending_review (Cirrus's reply waits for a person, REPLY_MODE=
review), escalated (an agent resolves it, with Cirrus's draft), merged (a duplicate of an open ticket),
resolved (a person sent the reply), rejected (the reviewer sent nothing).

Deduplication rule: same customer (customer key), the earlier ticket still open (pending_review or
escalated), received within DEDUP_WINDOW_HOURS (default 72), in a different thread, and the two texts
at least DEDUP_SIMILARITY (default 0.5) alike by character n-gram TF-IDF cosine. A message in the same
thread is never a duplicate: it is the next turn of that conversation. Each dedup decision is logged.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Optional

from src.channels import to_unified
from src.formatting import format_reply
from src.ingest import normalise_ticket
from src.logging_store import mask_pii

OPEN = ("pending_review", "escalated")
# Who is working a ticket, for the support workspace (supervisor assigns, agent resolves).
WORK_COLUMNS = ("assignee", "assignee_name", "assigned_by", "assigned_at", "work_status", "saved_reply",
                "source_ref")              # source_ref: the channel's own id for the message (email Message-ID...)
WORK_STATUSES = ("open", "in_progress", "resolved")
PRIORITY = {"P1": "High", "P2": "Medium", "P3": "Low", "high": "High", "medium": "Medium", "low": "Low"}
TIER_LABELS = {0: "Tier 0 · rules", 1: "Tier 1 · approved article", 2: "Tier 2 · RAG answer",
               3: "Tier 3 · agent"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


class TicketStore:
    """Centralised ticket storage for all four channels."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._conn() as c:
            c.execute("CREATE TABLE IF NOT EXISTS tickets (ticket_id TEXT PRIMARY KEY, created_at REAL, "
                      "updated_at REAL, channel TEXT, customer_key TEXT, customer_name TEXT, customer_id TEXT, "
                      "customer_tier TEXT, thread_key TEXT, subject TEXT, text TEXT, status TEXT, outcome TEXT, "
                      "decided_by TEXT, intent TEXT, urgency TEXT, priority TEXT, team TEXT, tier INTEGER, "
                      "confidence REAL, reason TEXT, reply TEXT, formatted TEXT, sources TEXT, handoff TEXT, "
                      "draft TEXT, duplicate_of TEXT, language TEXT, run_id TEXT, resolved_by TEXT)")
            have = {r[1] for r in c.execute("PRAGMA table_info(tickets)")}
            for col in WORK_COLUMNS:                         # added for the support workspace
                if col not in have:
                    c.execute(f"ALTER TABLE tickets ADD COLUMN {col} TEXT")
            c.execute("CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY AUTOINCREMENT, ticket_id TEXT, "
                      "thread_key TEXT, customer_key TEXT, channel TEXT, role TEXT, author TEXT, text TEXT, "
                      "at REAL)")
            c.execute("CREATE INDEX IF NOT EXISTS messages_customer ON messages (customer_key, at)")
            c.execute("CREATE INDEX IF NOT EXISTS messages_thread ON messages (thread_key, at)")

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(str(self.path), timeout=10)
        c.row_factory = sqlite3.Row
        return c

    def _write(self, sql: str, params: Any) -> None:
        with self._lock:
            conn = self._conn()
            try:
                with conn:
                    conn.execute(sql, params)
            finally:
                conn.close()

    def _read(self, sql: str, params: Any = ()) -> list[dict]:
        conn = self._conn()
        try:
            return [dict(r) for r in conn.execute(sql, params).fetchall()]
        finally:
            conn.close()

    def save(self, **row: Any) -> None:
        row.setdefault("created_at", time.time())
        row["updated_at"] = time.time()
        for k in ("formatted", "sources", "handoff", "draft"):
            if k in row and not isinstance(row[k], (str, type(None))):
                row[k] = json.dumps(row[k], default=str)
        self._write(f"INSERT OR REPLACE INTO tickets ({', '.join(row)}) VALUES ({', '.join('?' * len(row))})",
                    list(row.values()))

    def update(self, ticket_id: str, **fields: Any) -> None:
        fields["updated_at"] = time.time()
        for k in ("formatted", "sources", "handoff", "draft"):
            if k in fields and not isinstance(fields[k], (str, type(None))):
                fields[k] = json.dumps(fields[k], default=str)
        self._write(f"UPDATE tickets SET {', '.join(f'{k} = ?' for k in fields)} WHERE ticket_id = ?",
                    [*fields.values(), ticket_id])

    def add_message(self, ticket_id: str, thread_key: str, customer_key: str, channel: str, role: str, text: str,
                    author: str = "", at: Optional[float] = None) -> None:
        self._write("INSERT INTO messages (ticket_id, thread_key, customer_key, channel, role, author, text, at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (ticket_id, thread_key, customer_key, channel, role, author, mask_pii(text or ""),
                     at or time.time()))

    @staticmethod
    def _decode(row: dict) -> dict:
        for k in ("formatted", "sources", "handoff", "draft"):
            if row.get(k):
                try:
                    row[k] = json.loads(row[k])
                except ValueError:
                    pass
        return row

    def get(self, ticket_id: str) -> Optional[dict]:
        rows = self._read("SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id,))
        return self._decode(rows[0]) if rows else None

    def list(self, status: Optional[str] = None, channel: Optional[str] = None, limit: int = 500) -> list[dict]:
        where, params = [], []
        if status == "open":
            where.append(f"status IN {OPEN}")
        elif status:
            where.append("status = ?")
            params.append(status)
        if channel:
            where.append("channel = ?")
            params.append(channel)
        sql = "SELECT * FROM tickets" + (" WHERE " + " AND ".join(where) if where else "")
        return [self._decode(r) for r in self._read(sql + " ORDER BY created_at DESC LIMIT ?", (*params, limit))]

    def thread(self, thread_key: str) -> list[dict]:
        return self._read("SELECT * FROM messages WHERE thread_key = ? ORDER BY at, id", (thread_key,))

    def messages(self, ticket_id: str) -> list[dict]:
        return self._read("SELECT * FROM messages WHERE ticket_id = ? ORDER BY at, id", (ticket_id,))

    def history(self, customer_key: str) -> list[dict]:
        return self._read("SELECT * FROM messages WHERE customer_key = ? ORDER BY at, id", (customer_key,))

    def next_number(self) -> str:
        """CloudServe ticket numbers for channel tickets: CS-1001, CS-1002..."""
        rows = self._read("SELECT ticket_id FROM tickets WHERE ticket_id LIKE 'CS-%'")
        nums = [int(r["ticket_id"][3:]) for r in rows if r["ticket_id"][3:].isdigit()]
        return f"CS-{max(nums, default=1000) + 1}"

    def by_source(self, source_ref: str) -> Optional[dict]:
        rows = self._read("SELECT * FROM tickets WHERE source_ref = ?", (source_ref,)) if source_ref else []
        return self._decode(rows[0]) if rows else None

    def assigned_to(self, email: str) -> list[dict]:
        rows = self._read("SELECT * FROM tickets WHERE assignee = ? ORDER BY created_at DESC", (email,))
        return [self._decode(r) for r in rows]

    def open_for(self, customer_key: str, since: float) -> list[dict]:
        return self._read(f"SELECT * FROM tickets WHERE customer_key = ? AND status IN {OPEN} AND created_at >= ? "
                          "ORDER BY created_at", (customer_key, since))

    def summary(self) -> dict:
        rows = self._read("SELECT channel, status, tier, duplicate_of FROM tickets")
        return {"tickets": len(rows),
                "by_channel": dict(Counter(r["channel"] for r in rows)),
                "by_status": dict(Counter(r["status"] for r in rows)),
                "by_tier": {TIER_LABELS.get(k, str(k)): v for k, v in sorted(
                    Counter(r["tier"] for r in rows if r["tier"] is not None).items())},
                "merged_duplicates": sum(bool(r["duplicate_of"]) for r in rows),
                "open": sum(r["status"] in OPEN for r in rows)}


def similarity(a: str, b: str) -> float:
    """Character n-gram TF-IDF cosine of two texts (robust to typos and rewording of the same issue)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    if not (a or "").strip() or not (b or "").strip():
        return 0.0
    m = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit_transform([a, b])
    return float(cosine_similarity(m[0], m[1])[0][0])


class SupportHub:
    """The one entry point: any channel in, a handled, stored, logged ticket out."""

    def __init__(self, pipeline: Any, tickets: Optional[TicketStore] = None):
        self.pipeline = pipeline
        path = os.getenv("TICKETS_DB") or str(pipeline.store.path.with_name("tickets.db"))
        self.tickets = tickets or TicketStore(Path(path))
        self._numbering = threading.Lock()

    # ---------------- receiving
    def receive(self, channel: str, payload: Any, run_id: Optional[str] = None, native: bool = True) -> dict:
        """A ticket from a channel. native=True: the channel's own payload (src/channels.py); native=False:
        already in the unified schema (POST /tickets, the evaluation files)."""
        if not native:
            return self.handle(payload, channel, run_id)[0]
        raw = to_unified(channel, payload)
        if not isinstance(raw, dict):
            return self.handle(raw, channel, run_id)[0]
        source = f"{channel}:{raw.get('ticket_id')}" if raw.get("ticket_id") else ""
        with self._numbering:
            known = self.tickets.by_source(source)
            if known is not None:                          # the same message delivered twice: one ticket
                return {"ticket_id": known["ticket_id"], "channel": channel, "status": known["status"],
                        "duplicate_delivery": True, "reason": "This message was already received."}
            raw["ticket_id"] = self.tickets.next_number()
            self.tickets.save(ticket_id=raw["ticket_id"], channel=channel, status="new", source_ref=source,
                              thread_key="", customer_key="")    # reserves the number
        result = self.handle(raw, channel, run_id)[0]
        self.tickets.update(result["ticket_id"], source_ref=source)
        return result

    def handle(self, raw: Any, channel: str = "", run_id: Optional[str] = None) -> tuple[dict, Any]:
        """(the hub's result, the pipeline's TicketOutcome) for a unified ticket."""
        if isinstance(raw, dict):
            raw = dict(raw)
            if channel:
                raw.setdefault("channel", channel)
            if not str(raw.get("ticket_id") or "").strip():
                raw["ticket_id"] = f"CS-{uuid.uuid4().hex[:8].upper()}"
        t = normalise_ticket(raw)
        thread_key = f"{t.channel.value}:{t.thread_id}" if t.thread_id else f"{t.channel.value}:{t.ticket_id}"
        earlier = self.tickets.thread(thread_key)
        if isinstance(raw, dict) and earlier:                 # conversation history from central storage
            stored = [{"role": m["role"] if m["role"] != "agent" else "cirrus", "text": m["text"],
                       "at": m["at"]} for m in earlier]
            raw["conversation"] = stored + [m for m in (raw.get("conversation") or [])
                                            if m.get("text") not in {s["text"] for s in stored}]
        dup = self._duplicate_of(t, thread_key)
        if dup is not None:
            return self._merge(t, thread_key, dup, run_id)
        self.pipeline.store.record(t.ticket_id, "dedup", "new", (
            "No open ticket from this customer on another channel looks like this one." if t.customer_key else
            "The customer is not identified, so it cannot be matched across channels."),
            run_id=run_id, input_text=t.text)
        state = self.pipeline.run(raw, run_id=run_id)
        return self._store(state, thread_key, earlier), state["outcome"]

    def _duplicate_of(self, t: Any, thread_key: str) -> Optional[dict]:
        if not t.customer_key or not t.processable:
            return None
        window = _env_float("DEDUP_WINDOW_HOURS", 72) * 3600
        threshold = _env_float("DEDUP_SIMILARITY", 0.5)
        best, best_score = None, 0.0
        for other in self.tickets.open_for(t.customer_key, time.time() - window):
            if other["thread_key"] == thread_key or other["ticket_id"] == t.ticket_id:
                continue
            score = similarity(t.text, other["text"])
            if score >= threshold and score > best_score:
                best, best_score = other, score
        if best is not None:
            best["similarity"] = round(best_score, 3)
        return best

    def _merge(self, t: Any, thread_key: str, original: dict, run_id: Optional[str]) -> dict:
        """A duplicate: logged as its own ticket (ingest, dedup, outcome), stored as merged, and its
        message added to the original ticket's conversation. One fixed reply, no model call."""
        store = self.pipeline.store
        store.log_ingest(t, run_id)
        reason = (f"Merged into {original['ticket_id']} ({original['channel']}, {original['status']}): same "
                  f"customer, similarity {original['similarity']:.2f}, still open.")
        store.record(t.ticket_id, "dedup", "merged", reason, prediction=original["ticket_id"],
                     confidence=original["similarity"], threshold=_env_float("DEDUP_SIMILARITY", 0.5),
                     run_id=run_id, input_text=t.text)
        store.log_outcome(t.ticket_id, "merged", reason, run_id, t.text,
                          details={"decided_by": "dedup", "duplicate_of": original["ticket_id"], "channel":
                                   t.channel.value})
        text = (f"Thanks, we already have this problem open as ticket {original['ticket_id']} and the team is "
                "working on it. We've added your message to that ticket, so you don't need to do anything else.")
        formatted = format_reply(t.channel.value, text, ticket_id=original["ticket_id"], subject=t.subject, fixed=True)
        self.tickets.save(ticket_id=t.ticket_id, channel=t.channel.value, customer_key=t.customer_key,
                          customer_name=t.customer_name, customer_id=t.customer_id,
                          customer_tier=t.customer_tier.value, thread_key=thread_key, subject=t.subject,
                          text=mask_pii(t.text), status="merged", outcome="merged", decided_by="dedup",
                          tier=0, reason=reason, reply=text, formatted=formatted, duplicate_of=original["ticket_id"],
                          language=t.language, run_id=run_id)
        self.tickets.add_message(t.ticket_id, thread_key, t.customer_key, t.channel.value, "customer", t.text)
        self.tickets.add_message(t.ticket_id, thread_key, t.customer_key, t.channel.value, "cirrus", text,
                                 author="Cirrus AI")
        self.tickets.add_message(original["ticket_id"], original["thread_key"], t.customer_key, t.channel.value,
                                 "customer", f"(also sent on {t.channel.value}) {t.text}")
        from src.api import TicketOutcome
        outcome = TicketOutcome(ticket_id=t.ticket_id, channel=t.channel.value, outcome="merged", reply=text,
                                reason=reason, decided_by="dedup", tier=0, run_id=run_id, language=t.language,
                                tier_reason="Tier 0: merged into an open ticket, no model call.")
        return ({"ticket_id": t.ticket_id, "channel": t.channel.value, "status": "merged",
                 "duplicate_of": original["ticket_id"], "handled_by": "deduplication", "tier": 0,
                 "reply": formatted, "reason": reason}, outcome)

    def _store(self, state: dict, thread_key: str, earlier: list) -> dict:
        from src.scope import CONVERSATIONAL
        t, o = state["ticket"], state["outcome"]
        fixed = o.decided_by in CONVERSATIONAL
        status = {"sent": "answered", "pending_review": "pending_review"}.get(o.outcome, "escalated")
        h = o.handoff or {}
        draft = o.draft_reply or {}
        proposed = o.reply or (draft.get("text") if draft.get("status") == "drafted" else None)
        formatted = format_reply(t.channel.value, proposed, ticket_id=t.ticket_id, subject=t.subject,
                                 fixed=fixed) if proposed else None
        self.tickets.save(ticket_id=t.ticket_id, channel=t.channel.value, customer_key=t.customer_key,
                          customer_name=t.customer_name, customer_id=t.customer_id,
                          customer_tier=t.customer_tier.value, thread_key=thread_key, subject=t.subject,
                          text=mask_pii(t.original_text or t.text), status=status, outcome=o.outcome,
                          decided_by=o.decided_by, intent=o.intent, urgency=o.urgency,
                          priority=h.get("priority"), team=h.get("suggested_team"), tier=o.tier,
                          confidence=o.confidence, reason=o.reason, reply=proposed, formatted=formatted,
                          sources=o.sources, handoff=h, draft=draft or None, language=t.language,
                          run_id=state.get("run_id"))
        self.tickets.add_message(t.ticket_id, thread_key, t.customer_key, t.channel.value, "customer",
                                 t.original_text or t.text)
        if o.outcome == "sent" and o.reply:
            self.tickets.add_message(t.ticket_id, thread_key, t.customer_key, t.channel.value, "cirrus",
                                     formatted["body"] if formatted else o.reply, author="Cirrus AI")
        return {"ticket_id": t.ticket_id, "channel": t.channel.value, "status": status, "outcome": o.outcome,
                "handled_by": TIER_LABELS.get(o.tier, ""), "tier": o.tier, "decided_by": o.decided_by,
                "intent": o.intent, "reply": formatted if o.outcome == "sent" else None,
                "suggested_reply": formatted if o.outcome != "sent" else None, "sources": o.sources,
                "reason": o.reason, "history": len(earlier)}

    # ---------------- employees: review and send
    def approve(self, ticket_id: str, reviewer: str, text: Optional[str] = None) -> dict:
        from src.review import approve
        p = self.pipeline
        res = approve(p.reviews, p.store, p.guardrails, ticket_id, reviewer, text)
        t = self.tickets.get(ticket_id)
        if t is not None:
            formatted = format_reply(t["channel"], res["reply"], ticket_id=ticket_id, subject=t.get("subject") or "",
                                     fixed=res["edited"] or t.get("decided_by") == "kill_switch")
            self.tickets.update(ticket_id, status="resolved", reply=res["reply"], formatted=formatted,
                                resolved_by=reviewer)
            self.tickets.add_message(ticket_id, t["thread_key"], t["customer_key"], t["channel"],
                                     "agent" if res["edited"] or t["status"] == "escalated" else "cirrus",
                                     formatted["body"], author=reviewer)
            res["formatted"] = formatted
        return res

    def reject(self, ticket_id: str, reviewer: str, reason: str = "") -> dict:
        from src.review import reject
        res = reject(self.pipeline.reviews, self.pipeline.store, ticket_id, reviewer, reason)
        if self.tickets.get(ticket_id) is not None:
            self.tickets.update(ticket_id, status="rejected", resolved_by=reviewer)
        return res

    # ---------------- the support workspace: supervisors assign, agents resolve
    def assign(self, ticket_id: str, agent: dict, supervisor: dict) -> dict:
        t = self.tickets.get(ticket_id)
        if t is None:
            raise LookupError(f"no ticket {ticket_id}")
        if t["status"] not in OPEN:
            raise ValueError(f"{ticket_id} does not need an agent ({t['status']})")
        self.tickets.update(ticket_id, assignee=agent["email"], assignee_name=f"{agent['name']} · {agent['team']}",
                            assigned_by=supervisor["name"], assigned_at=str(time.time()),
                            work_status=t.get("work_status") or "open")
        self.pipeline.store.record(ticket_id, "assignment", "assigned",
                                   f"{supervisor['name']} assigned the ticket to {agent['name']} ({agent['team']}).",
                                   prediction=agent["name"], run_id=t.get("run_id"), model_name="person")
        return self.tickets.get(ticket_id)

    def _assigned(self, ticket_id: str, agent: Optional[dict]) -> dict:
        """The ticket, if `agent` may act on it: only the agent it is assigned to can."""
        t = self.tickets.get(ticket_id)
        if t is None:
            raise LookupError(f"no ticket {ticket_id}")
        if agent is not None and t.get("assignee") != agent.get("email"):
            raise PermissionError(f"{ticket_id} is not assigned to {agent.get('name', 'you')}")
        return t

    def save_reply(self, ticket_id: str, text: str, agent: Optional[dict] = None) -> None:
        self._assigned(ticket_id, agent)
        self.tickets.update(ticket_id, saved_reply=text, work_status="in_progress")

    def set_work_status(self, ticket_id: str, status: str, agent: Optional[dict] = None) -> None:
        self._assigned(ticket_id, agent)
        if status not in WORK_STATUSES:
            raise ValueError(f"status must be one of {WORK_STATUSES}")
        self.tickets.update(ticket_id, work_status=status)

    def send(self, ticket_id: str, agent: dict, text: str) -> dict:
        """Send the reply an agent checked. Unchanged: Cirrus's reply as it was prepared for the channel.
        Edited: exactly the agent's text, checked again first (src/review.py)."""
        from src.review import approve
        t = self._assigned(ticket_id, agent)
        prepared = (t.get("formatted") or {}).get("body") if isinstance(t.get("formatted"), dict) else None
        unchanged = prepared is not None and text.strip() == prepared.strip()
        p = self.pipeline
        res = approve(p.reviews, p.store, p.guardrails, ticket_id, agent["name"], None if unchanged else text)
        formatted = dict(t["formatted"]) if unchanged else {
            "channel": t["channel"], "format": "text/plain" if t["channel"] == "email" else "text/markdown",
            "subject": (t.get("formatted") or {}).get("subject", "") if isinstance(t.get("formatted"), dict) else "",
            "body": text.strip()}
        self.tickets.update(ticket_id, status="resolved", work_status="resolved", reply=res["reply"],
                            formatted=formatted, resolved_by=agent["name"], saved_reply=None)
        self.tickets.add_message(ticket_id, t["thread_key"], t["customer_key"], t["channel"],
                                 "agent" if not unchanged else "cirrus", formatted["body"], author=agent["name"])
        return {**res, "formatted": formatted}

    def resolve_without_reply(self, ticket_id: str, agent: dict, note: str = "") -> dict:
        from src.review import close
        self._assigned(ticket_id, agent)
        res = close(self.pipeline.reviews, self.pipeline.store, ticket_id, agent["name"], note)
        self.tickets.update(ticket_id, status="resolved", work_status="resolved", resolved_by=agent["name"])
        return res

    # ---------------- employees: one view of a ticket
    def ticket_view(self, ticket_id: str) -> Optional[dict]:
        t = self.tickets.get(ticket_id)
        if t is None:
            return None
        trail = self.pipeline.store.export(ticket_id=ticket_id)
        retrieval = next((r for r in trail if r["stage"] == "retrieval"), None)
        return {**t, "messages": self.tickets.messages(ticket_id),
                "customer_history": [m for m in self.tickets.history(t["customer_key"])
                                     if m["ticket_id"] != ticket_id] if t["customer_key"] else [],
                "duplicates": [d["ticket_id"] for d in self.tickets.list() if d.get("duplicate_of") == ticket_id],
                "retrieved": retrieval["sources_used"] if retrieval else [],
                "review": self.pipeline.reviews.get(ticket_id), "decisions": trail}

    def simulate(self, count: int = 12, seed: int = 7, run_id: Optional[str] = None) -> list[dict]:
        from src.channels import simulate
        run_id = run_id or f"sim-{uuid.uuid4().hex[:6]}"
        return [self.receive(ch, payload, run_id=run_id) for ch, payload in simulate(count, seed)]

    def load(self, tickets: list[dict], run_id: Optional[str] = None) -> list[dict]:
        """Bring real tickets (dataset records) into the queue, each through its own channel. Loading the
        same file again adds nothing: every ticket keeps the same delivery reference."""
        from src.channels import render
        run_id = run_id or f"load-{uuid.uuid4().hex[:6]}"
        return [self.receive(*render(t), run_id=run_id) for t in tickets]


# ---------------------------------------------------------------- terminal: start over, load tickets
def read_tickets(path: Path) -> list[dict]:
    """A ticket file as the evaluation reads it: a JSON list, {"tickets": [...]}, or JSON Lines."""
    text = path.read_text(encoding="utf-8").strip()
    if text.startswith("[") or text.startswith("{\"tickets\""):
        data = json.loads(text)
        return data["tickets"] if isinstance(data, dict) else data
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def store_paths() -> dict[str, Path]:
    """Where the live queue keeps its data (the same settings the API and Streamlit use)."""
    from src.logging_store import DEFAULT_URL, _sqlite_path
    decisions = Path(_sqlite_path(os.getenv("DATABASE_URL", DEFAULT_URL)))
    return {"tickets": Path(os.getenv("TICKETS_DB") or decisions.with_name("tickets.db")),
            "replies waiting for review": Path(os.getenv("REVIEW_DB") or decisions.with_name("reviews.db")),
            "decision log": decisions}


def main(argv: Optional[list] = None) -> int:
    import argparse
    import sys
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    p = argparse.ArgumentParser(prog="python -m src.hub", description="The live ticket queue the staff screens show.")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("reset", help="delete every ticket, waiting reply and logged decision (asks you to confirm)")
    ld = sub.add_parser("load", help="bring tickets from a file into the queue, each through its own channel")
    ld.add_argument("file", type=Path, help="a JSON list, {\"tickets\": [...]}, or JSON Lines")
    ld.add_argument("--count", type=int, help="only the first N tickets")
    ld.add_argument("--skip", type=int, default=0, help="start after the first N tickets")
    args = p.parse_args(argv)

    if args.command == "reset":
        paths = store_paths()
        print("This deletes, for the staff screens and the API:")
        for what, path in paths.items():
            print(f"  {what:<27} {path}{'' if path.exists() else '  (empty already)'}")
        print("Evaluation runs keep their own files in evaluation/results and are not touched.")
        if input("Type reset to continue: ").strip().lower() != "reset":
            print("Nothing deleted.")
            return 1
        for path in paths.values():
            for f in (path, path.with_name(path.name + "-wal"), path.with_name(path.name + "-shm")):
                f.unlink(missing_ok=True)
        print("Done. Restart Streamlit (and the API, if it is running) so they start from the empty queue.")
        return 0

    if not args.file.is_file():
        print(f"No such file: {args.file}", file=sys.stderr)
        return 1
    tickets = read_tickets(args.file)[args.skip:]
    tickets = tickets[:args.count] if args.count else tickets
    from src.api import SupportPipeline
    hub = SupportHub(SupportPipeline.from_env())
    print(f"Loading {len(tickets)} ticket(s) from {args.file}...")
    results = hub.load(tickets)
    new = [r for r in results if not r.get("duplicate_delivery")]
    print(f"Added {len(new)} ticket(s) to the queue"
          f"{f'; {len(results) - len(new)} were already there' if len(new) < len(results) else ''}. "
          "Click Refresh in Streamlit (or press R) to see them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
