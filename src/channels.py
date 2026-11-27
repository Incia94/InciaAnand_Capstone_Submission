"""The four support channels: their native payloads, and the adapters that turn each into the one
unified ticket schema (src/ingest.py: NormalisedTicket).

Every channel reaches CloudServe in its own shape. An adapter maps that shape onto the unified ticket
dictionary that normalise_ticket() reads, so everything after this point (screening, classification,
search, routing, generation, guardrails, decision logging, storage) is the same for all four.

  email         {"message_id", "in_reply_to", "from": "Name <address>", "subject", "text" | "html", "date",
                 "account": {...}}
  chat          {"session_id", "visitor": {"name", "email", "customer_id"}, "messages": [{"from", "text",
                 "ts"}], "account": {...}}            the last visitor message is the ticket; the rest is history
  forum         {"thread_id", "post_id", "title", "body", "author": {"username", "email", "customer_id"},
                 "created_at", "account": {...}}
  docs_comment  {"comment_id", "page": {"url", "doc_id", "title"}, "comment", "author": {"name", "email",
                 "customer_id"}, "created_at", "account": {...}}

"account" is what the channel connector looks up in the CRM for the sender (customer id, plan,
region, English fluency). It is optional: without it the ticket is processed as an unknown customer.

simulate() renders tickets from the dataset into these native payloads, for a realistic mixed stream
across all four channels, including a customer who raises the same problem on two channels and a
customer who follows up in the same thread:

    python -m src.channels simulate --count 20                       # print the payloads (JSON Lines)
    python -m src.channels simulate --count 20 --post http://127.0.0.1:8000   # send them to the API
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

CHANNELS = ("email", "chat", "forum", "docs_comment")
_ADDRESS = re.compile(r"^\s*(?:\"?([^\"<]*?)\"?\s*)?<\s*([^>\s]+@[^>\s]+)\s*>\s*$|^\s*([^\s<>]+@[^\s<>]+)\s*$")
_TAGS = re.compile(r"<[^>]+>")


def _account(payload: dict) -> dict:
    a = payload.get("account") or {}
    return {"customer_id": a.get("customer_id", ""), "customer_tier": a.get("tier", ""),
            "customer_region": a.get("region", ""), "language_fluency": a.get("fluency", "")}


def from_email(p: dict) -> dict:
    m = _ADDRESS.match(str(p.get("from") or ""))
    name, address = ((m.group(1) or "").strip(), (m.group(2) or m.group(3) or "").strip()) if m else ("", "")
    text = p.get("text") or _TAGS.sub(" ", str(p.get("html") or ""))
    return {"ticket_id": p.get("message_id") or "", "channel": "email", "subject": p.get("subject", ""),
            "body": text, "received_at": p.get("date"), "customer_name": name, "customer_email": address,
            "thread_id": p.get("in_reply_to") or p.get("message_id") or "", **_account(p)}


def from_chat(p: dict) -> dict:
    visitor = p.get("visitor") or {}
    msgs = [m for m in (p.get("messages") or []) if isinstance(m, dict) and str(m.get("text") or "").strip()]
    mine = [i for i, m in enumerate(msgs) if m.get("from") in ("visitor", "customer", None)]
    last = mine[-1] if mine else len(msgs) - 1
    history = [{"role": "customer" if m.get("from") in ("visitor", "customer", None) else "cirrus",
                "text": m.get("text", ""), "at": m.get("ts", "")} for m in msgs[:last]] if last > 0 else []
    body = msgs[last]["text"] if msgs else ""
    session = p.get("session_id") or ""
    # one ticket per visitor message: the message's own id, else a hash of its text and time, so a follow-up
    # in the same session is a new ticket and the same message delivered twice is the same ticket
    msg = msgs[last] if msgs else {}
    mid = msg.get("id") or hashlib.sha256(f"{body}|{msg.get('ts')}|{last}".encode()).hexdigest()[:8]
    return {"ticket_id": f"{session}-{mid}" if session else "", "channel": "chat", "subject": "", "body": body,
            "received_at": msgs[last].get("ts") if msgs else None, "customer_name": visitor.get("name", ""),
            "customer_email": visitor.get("email", ""), "thread_id": session, "conversation": history,
            **{**_account(p), **({"customer_id": visitor["customer_id"]} if visitor.get("customer_id") else {})}}


def from_forum(p: dict) -> dict:
    author = p.get("author") or {}
    return {"ticket_id": p.get("post_id") or "", "channel": "forum", "subject": p.get("title", ""),
            "body": p.get("body", ""), "received_at": p.get("created_at"), "customer_name": author.get("username", ""),
            "customer_email": author.get("email", ""), "thread_id": p.get("thread_id", ""),
            **{**_account(p), **({"customer_id": author["customer_id"]} if author.get("customer_id") else {})}}


def from_docs_comment(p: dict) -> dict:
    author, page = p.get("author") or {}, p.get("page") or {}
    title = page.get("title") or ""
    return {"ticket_id": p.get("comment_id") or "", "channel": "docs_comment",
            "subject": p.get("subject") or (f"Comment on: {title}" if title else ""), "body": p.get("comment", ""),
            "received_at": p.get("created_at"), "customer_name": author.get("name", ""),
            "customer_email": author.get("email", ""), "thread_id": p.get("parent_id") or p.get("comment_id") or "",
            **{**_account(p), **({"customer_id": author["customer_id"]} if author.get("customer_id") else {})}}


ADAPTERS: dict[str, Callable[[dict], dict]] = {
    "email": from_email, "chat": from_chat, "forum": from_forum, "docs_comment": from_docs_comment}


def to_unified(channel: str, payload: Any) -> Any:
    """The unified ticket dictionary for a native payload. Anything that is not a dict, or a channel
    with no adapter, is passed through unchanged for normalise_ticket() to handle (and flag)."""
    adapter = ADAPTERS.get(channel)
    if adapter is None or not isinstance(payload, dict):
        return payload
    try:
        return adapter(payload)
    except Exception:                    # a malformed payload is still a ticket: normalise flags it
        return {"channel": channel, "body": json.dumps(payload, default=str)[:4000]}


# ------------------------------------------------------------------ simulated traffic
def _address(name: str) -> str:
    return re.sub(r"[^a-z.]", "", name.lower().replace(" ", ".")) + "@example.com"


def render(t: dict, channel: Optional[str] = None, n: int = 0) -> tuple[str, dict]:
    """A dataset ticket as its channel would deliver it."""
    ch = channel or t.get("channel", "email")
    name, cid = t.get("customer_name", ""), t.get("customer_id", "")
    account = {"customer_id": cid, "tier": t.get("customer_tier"), "region": t.get("customer_region"),
               "fluency": t.get("language_fluency")}
    when, sid = t.get("received_at"), f"{t['ticket_id']}-{ch}-{n}"
    subject, body = t.get("subject", ""), t.get("body", "")
    if ch == "email":
        return ch, {"message_id": f"<{sid}@mail.example.com>", "from": f"{name} <{_address(name)}>",
                    "to": "support@cloudserve.example", "subject": subject or "Support request", "text": body,
                    "date": when, "account": account}
    if ch == "chat":
        return ch, {"session_id": f"chat-{sid}", "visitor": {"name": name, "email": _address(name), "customer_id": cid},
                    "messages": [{"from": "visitor", "text": f"{subject}. {body}" if subject else body, "ts": when}],
                    "account": account}
    if ch == "forum":
        return ch, {"thread_id": f"thread-{sid}", "post_id": f"post-{sid}", "title": subject or body[:60],
                    "body": body, "author": {"username": name.split()[0].lower() + (cid[-3:] if cid else ""),
                                             "email": _address(name), "customer_id": cid},
                    "created_at": when, "account": account}
    doc = (t.get("labels") or {}).get("expected_doc_ids") or ["DOC-ONB-001"]
    return "docs_comment", {"comment_id": f"comment-{sid}",
                            "page": {"url": f"https://docs.cloudserve.example/{doc[0]}", "doc_id": doc[0],
                                     "title": subject or "Documentation"},
                            "comment": body, "author": {"name": name, "email": _address(name), "customer_id": cid},
                            "created_at": when, "account": account}


def simulate(count: int = 20, seed: int = 7, tickets: Optional[Sequence[dict]] = None) -> list[tuple[str, dict]]:
    """A mixed stream across the four channels, in arrival order. Besides `count` tickets in their own
    channels it adds (1) one customer raising an open problem again on a different channel (a cross-
    channel duplicate) and (2) one customer replying in the same chat session (conversation history)."""
    rng = random.Random(seed)
    path = Path(__file__).resolve().parents[1] / "data" / "validation_tickets.json"
    pool = list(tickets) if tickets is not None else json.loads(path.read_text(encoding="utf-8"))
    by_channel = {ch: [t for t in pool if t.get("channel") == ch] for ch in CHANNELS}
    picked: list[dict] = []
    for i in range(count):                               # round-robin, so every channel is represented
        options = by_channel[CHANNELS[i % 4]] or pool
        picked.append(options[rng.randrange(len(options))])
    stream = [render(t, n=i) for i, t in enumerate(picked)]
    escalate = next((t for t in picked if (t.get("labels") or {}).get("expected_route") == "escalate"
                     and t.get("channel") != "chat" and t.get("customer_id")), None)
    if escalate is not None:                             # the same open problem, again, on live chat
        stream.append(render(escalate, "chat", n=count))
    chat = next(((ch, p) for ch, p in stream if ch == "chat"), None)
    if chat is not None:                                 # a follow-up in the same chat session
        follow = json.loads(json.dumps(chat[1]))
        follow["messages"] += [{"from": "bot", "text": "(Cirrus replied)", "ts": None},
                               {"from": "visitor", "text": "Thanks. I tried that and it is still not working.",
                                "ts": None}]
        stream.append(("chat", follow))
    return stream


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="python -m src.channels", description="Simulated four-channel traffic.")
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("simulate", help="generate native channel payloads")
    s.add_argument("--count", type=int, default=20)
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--post", help="send each payload to this API (e.g. http://127.0.0.1:8000)")
    args = p.parse_args(argv)
    stream = simulate(args.count, args.seed)
    if not args.post:
        for ch, payload in stream:
            print(json.dumps({"channel": ch, "payload": payload}, ensure_ascii=False, default=str))
        return 0
    import httpx
    with httpx.Client(base_url=args.post, timeout=120) as client:
        for ch, payload in stream:
            r = client.post(f"/channels/{ch}", json=payload)
            body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
            print(f"{ch:13} {r.status_code}  {body.get('ticket_id', '')}  {body.get('status', '')}  "
                  f"{body.get('handled_by', '')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
