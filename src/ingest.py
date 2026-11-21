"""Ingest: everything that brings data into the system, in one file.

Part 1, tickets: normalise a ticket from any of the four channels (email, live chat,
documentation comments, community forum) into one NormalisedTicket. This runs for every ticket,
and downstream components (classify, retrieve, route, generate, guardrails, the decision log and
the evaluation harness) read only NormalisedTicket. Follows the ticket schema in the Dataset Guide.

Part 2, docs: load, validate, normalise, chunk, embed and store the documentation corpus in
Chroma. This runs once, offline, to build the index that retrieval searches.

Usage
-----
    # tickets: print one normalised ticket per channel (build checkpoint, day one)
    python -m src.ingest tickets --input data/validation_tickets.json --sample
    # tickets: normalise a whole file to JSON Lines and print a summary
    python -m src.ingest tickets --input data/validation_tickets.json --output storage/normalised.jsonl

    # docs: build the vector index (paths and model from .env: CHROMA_PATH, EMBEDDING_MODEL)
    python -m src.ingest docs
    python -m src.ingest docs --docs data/documentation.json --check "my api key returns 401"

Docs outputs: <persist-dir>/ (Chroma, collection kb_chunks, cosine distance),
<persist-dir>/chunks.jsonl (one line per chunk, used to resolve citations) and
<persist-dir>/ingest_report.json (counts, skipped docs, model, timestamp).

Build Specification (Ingest) -> where it is met            -> tests (tests/test_ingest_tickets.py)
-----------------------------------------------------------------------------------------------
Accepts email, live chat, docs comments, forum             parse_channel, CHANNEL_ALIASES   test_req1_*
One normalised representation regardless of channel        NormalisedTicket                 test_req2_*
Preserves the original text and the channel                original_subject/_body/_channel  test_req3_*
Handles missing fields, unusual characters, empty bodies   to_str, clean_text, flags,       test_req4_*
                                                           processable, the outer try       (incl. a 2,000-ticket fuzz)

Design rules
------------
1. Never raise on a bad ticket. Every input becomes a NormalisedTicket; problems are recorded in
   `flags`, and a ticket with no usable text has `processable=False` so the router escalates it
   with a reason instead of the run stopping (acceptance criteria A9 and A11).
2. Keep the original. `original_subject`, `original_body` and `original_channel` are stored as
   received alongside the cleaned values (Build Specification, Ingest).
3. Keep channel. It matters downstream: chat customers expect minutes, docs comments are narrow,
   forum threads may already have replies.
4. Keep evaluation data apart. `labels` and `history` are carried in `eval_labels` and
   `eval_history` for the harness only. Nothing in the pipeline may read them, because that would
   leak the answer into the prediction. The customer's name is kept for the record but is never
   put into `text`, the field models see.
5. Do not rewrite the customer's words. Non-fluent English is kept as written; cleaning only
   removes markup, invisible characters and quoted earlier messages.

Docs chunking
-------------
Default "document": one chunk per article, headed by its title and applies-to line. Chosen by
measurement (`python -m src.retrieve compare`) over three alternatives: "section" (problem /
resolution / notes), "step" (one chunk per resolution step) and "window" (~800 characters with 120
overlap, the Setup Guide's starting point). Select another with --strategy or CHUNK_STRATEGY.
Chunk ids are stable across runs (e.g. DOC-AUTH-001::full), so logged citations stay resolvable.
An article that does not follow the usual layout is kept as one chunk rather than dropped.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import logging
import os
import re
import sys
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Optional, Protocol, Sequence

from pydantic import BaseModel, Field

log = logging.getLogger("ingest")


# ################################################################## PART 1: TICKETS

MAX_BODY_CHARS = 6000          # longer bodies are cut (flagged) to bound model cost and latency
MIN_USABLE_CHARS = 3           # below this there is nothing to classify


# ------------------------------------------------------------------ tickets: allowed values (Dataset Guide)
class Channel(str, Enum):
    email = "email"
    chat = "chat"
    docs_comment = "docs_comment"
    forum = "forum"
    unknown = "unknown"


class Tier(str, Enum):
    enterprise = "enterprise"
    business = "business"
    standard = "standard"
    unknown = "unknown"


class Region(str, Enum):
    north_america = "north_america"
    europe = "europe"
    asia_pacific = "asia_pacific"
    latin_america = "latin_america"
    unknown = "unknown"


class Fluency(str, Enum):
    fluent = "fluent"
    non_fluent = "non_fluent"
    unknown = "unknown"


# Spellings other systems commonly use, mapped to the Dataset Guide's values.
CHANNEL_ALIASES = {
    "email": "email", "e-mail": "email", "mail": "email",
    "chat": "chat", "live_chat": "chat", "livechat": "chat", "live chat": "chat", "web_chat": "chat",
    "docs_comment": "docs_comment", "docs-comment": "docs_comment", "doc_comment": "docs_comment",
    "docs comment": "docs_comment", "documentation_comment": "docs_comment", "docs": "docs_comment",
    "api_docs": "docs_comment", "documentation comment": "docs_comment", "doc comment": "docs_comment",
    "forum": "forum", "forum post": "forum", "community": "forum",
    "community_forum": "forum", "community forum": "forum",
}
REGION_ALIASES = {"na": "north_america", "us": "north_america", "eu": "europe", "emea": "europe",
                  "apac": "asia_pacific", "latam": "latin_america"}
FIELD_ALIASES = {  # canonical field -> other names accepted on input
    "ticket_id": ("id", "ticketId"),
    "subject": ("title", "topic"),
    "body": ("message", "text", "content", "description"),
    "received_at": ("created_at", "timestamp", "receivedAt", "createdAt"),
    "customer_id": ("customerId",),
    "customer_name": ("customerName", "name"),
    "customer_tier": ("tier", "plan"),
    "customer_region": ("region",),
    "language_fluency": ("fluency",),
}


# ------------------------------------------------------------------ tickets: the internal representation
class NormalisedTicket(BaseModel):
    ticket_id: str
    channel: Channel
    original_channel: str = ""              # the channel value exactly as received
    subject: str = ""                       # cleaned; always "" for chat
    body: str = ""                          # cleaned
    text: str = ""                          # what classify/retrieve/generate read: subject + body
    original_subject: str = ""              # exactly as received
    original_body: str = ""                 # exactly as received
    received_at: Optional[datetime] = None  # timezone-aware UTC, or None if unreadable
    received_at_raw: str = ""
    customer_id: str = ""
    customer_name: str = Field("", description="for the record only; never placed in `text`")
    customer_tier: Tier = Tier.unknown
    customer_region: Region = Region.unknown
    language_fluency: Fluency = Fluency.unknown
    referenced_doc_ids: list[str] = []      # e.g. DOC-API-001 mentioned in the ticket
    word_count: int = 0
    processable: bool = True                # False -> nothing usable to answer; route must escalate
    flags: list[str] = []                   # every repair or problem noticed, for the decision log
    eval_labels: Optional[dict] = Field(None, description="evaluation only; never read by the pipeline")
    eval_history: Optional[dict] = Field(None, description="baseline only; never read by the pipeline")

    def segment(self) -> dict[str, str]:
        """The fairness-audit segments (Dataset Guide, 'The fairness segments')."""
        return {"customer_tier": self.customer_tier.value, "customer_region": self.customer_region.value,
                "language_fluency": self.language_fluency.value, "channel": self.channel.value}


# ------------------------------------------------------------------ tickets: text cleaning
# zero-width characters, soft hyphen and bidirectional controls (these can disguise what text says)
_ZERO_WIDTH = re.compile("[\u00ad\u200b-\u200f\u202a-\u202e\u2060-\u2064\u2066-\u2069\ufeff]")
_SURROGATE = re.compile("[\ud800-\udfff]")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_TAG = re.compile(r"<[^>]{1,200}>")
_BREAK_TAGS = re.compile(r"<\s*(br|/p|/div|/li)\s*/?>", re.I)
_DOC_ID = re.compile(r"\bDOC-[A-Z]+-\d{3}\b")


def to_str(value: Any, flags: list[str]) -> str:
    """Turn any field value into a str that can always be written out as UTF-8 JSON.

    Also used for the preserved originals, which therefore stay exactly as received except for
    code points that cannot exist in valid UTF-8 (lone surrogates): those become U+FFFD, flagged.
    """
    if value is None:
        return ""
    if isinstance(value, (bytes, bytearray)):
        flags.append("bytes_decoded")
        value = bytes(value).decode("utf-8", errors="replace")
    elif isinstance(value, (list, tuple)):
        flags.append("list_joined")
        value = "\n".join(to_str(v, flags) for v in value)
    elif isinstance(value, dict):
        flags.append("nested_object_flattened")
        inner = next((value[k] for k in ("text", "body", "content", "message", "value") if k in value), None)
        value = to_str(inner, flags) if inner is not None else json.dumps(value, ensure_ascii=False, default=str)
    elif not isinstance(value, str):
        flags.append("non_text_field_coerced")
        value = str(value)
    if _SURROGATE.search(value):
        flags.append("invalid_unicode_replaced")
        value = _SURROGATE.sub("�", value)
    return value


def clean_text(value: Any, flags: list[str]) -> str:
    """Make text safe and uniform without changing what the customer said."""
    text = unicodedata.normalize("NFKC", to_str(value, flags))
    if _ZERO_WIDTH.search(text) or _CONTROL.search(text):
        flags.append("invisible_characters_removed")
        text = _CONTROL.sub(" ", _ZERO_WIDTH.sub("", text))
    if _TAG.search(text):
        flags.append("html_removed")
        text = _TAG.sub(" ", _BREAK_TAGS.sub("\n", text))
    if "&" in text:
        text = html.unescape(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
    text = re.sub(r"[  ]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ------------------------------------------------------------------ tickets: channel-specific cleaning
_EMAIL_CUTS = [
    re.compile(r"^-{2,}\s*original message\s*-{2,}", re.I | re.M),
    re.compile(r"^on .{3,120} wrote:\s*$", re.I | re.M),
    re.compile(r"^from: .+\nsent: .+", re.I | re.M),
    re.compile(r"^-- ?$", re.M),                                  # standard signature delimiter
    re.compile(r"^sent from my (iphone|ipad|android|phone|mobile).*$", re.I | re.M),
]
_SUBJECT_PREFIX = re.compile(r"^\s*((re|fw|fwd|aw|sv)\s*:\s*)+", re.I)
_CHAT_LINE = re.compile(r"^\s*(\[[0-9:apm .]{3,12}\]\s*)?(?P<who>[A-Za-z ]{2,20}):\s?(?P<msg>.*)$")
_CHAT_STAFF = {"agent", "support", "bot", "assistant", "cloudserve", "system", "operator"}
_FORUM_QUOTE = re.compile(r"\[quote[^\]]*\].*?\[/quote\]", re.I | re.S)


def _drop_quoted_lines(text: str, flags: list[str], flag: str) -> str:
    kept = [line for line in text.split("\n") if not line.lstrip().startswith(">")]
    if len(kept) != len(text.split("\n")):
        flags.append(flag)
    return "\n".join(kept)


def clean_email(body: str, flags: list[str]) -> str:
    """Keep the newest message: cut quoted history and signatures, but only if text remains."""
    cut = len(body)
    for pattern in _EMAIL_CUTS:
        m = pattern.search(body)
        if m and m.start() > 0:
            cut = min(cut, m.start())
    if cut < len(body) and body[:cut].strip():
        flags.append("email_quoted_history_removed")
        body = body[:cut]
    return _drop_quoted_lines(body, flags, "email_quoted_lines_removed").strip()


def clean_chat(body: str, flags: list[str]) -> str:
    """A transcript with speaker labels keeps only the customer's lines: staff and bot lines are
    our own words and must not be treated as the customer's request."""
    lines = [line for line in body.split("\n") if line.strip()]
    parsed = [_CHAT_LINE.match(line) for line in lines]
    if len(lines) > 1 and all(parsed):
        customer = [m["msg"].strip() for m in parsed if m["who"].strip().lower() not in _CHAT_STAFF]
        if customer and len(customer) < len(lines):
            flags.append("chat_staff_lines_removed")
        return "\n".join(customer) if customer else body
    return body


def clean_forum(body: str, flags: list[str]) -> str:
    """Forum posts often quote earlier posts by other customers; keep only this author's words."""
    if _FORUM_QUOTE.search(body):
        flags.append("forum_quotes_removed")
        body = _FORUM_QUOTE.sub(" ", body)
    return _drop_quoted_lines(body, flags, "forum_quoted_lines_removed").strip()


CHANNEL_CLEANERS = {Channel.email: clean_email, Channel.chat: clean_chat, Channel.forum: clean_forum}


# ------------------------------------------------------------------ tickets: field helpers
def _get(raw: dict, field: str) -> Any:
    if field in raw and raw[field] not in (None, ""):
        return raw[field]
    for alt in FIELD_ALIASES.get(field, ()):
        if alt in raw and raw[alt] not in (None, ""):
            return raw[alt]
    customer = raw.get("customer")  # nested {"customer": {"id":..., "tier":...}}
    if isinstance(customer, dict) and field.startswith("customer_"):
        return customer.get(field.removeprefix("customer_"))
    return raw.get(field)


def parse_channel(value: Any, flags: list[str]) -> Channel:
    """Exact names and aliases first, then keywords, so 'Documentation Comments', 'Live-Chat',
    'forum_post' or 'Email (support inbox)' all land on the right channel."""
    key = re.sub(r"[\s\-_]+", " ", to_str(value, []).strip().lower())
    if not key:
        flags.append("channel_missing")
        return Channel.unknown
    for candidate in (key, key.rstrip("s")):
        for spelling in (candidate, candidate.replace(" ", "_")):
            if spelling in CHANNEL_ALIASES:
                return Channel(CHANNEL_ALIASES[spelling])
    if "doc" in key and "comment" in key:
        return Channel.docs_comment
    for word, channel in (("chat", Channel.chat), ("mail", Channel.email),
                          ("forum", Channel.forum), ("community", Channel.forum)):
        if word in key:
            return channel
    flags.append("channel_unrecognised")
    return Channel.unknown


def _enum(value: Any, enum_cls, flags: list[str], name: str, aliases: dict | None = None):
    key = re.sub(r"\s+", " ", to_str(value, [])).strip().lower()
    key = (aliases or {}).get(key, key).replace(" ", "_").replace("-", "_")
    try:
        return enum_cls(key)
    except ValueError:
        flags.append(f"{name}_missing" if not key else f"{name}_unrecognised")
        return enum_cls("unknown")


def parse_timestamp(value: Any, flags: list[str]) -> Optional[datetime]:
    if value in (None, ""):
        flags.append("received_at_missing")
        return None
    try:
        if isinstance(value, bool):
            raise ValueError("boolean is not a timestamp")
        if isinstance(value, (int, float)):
            ts = float(value) / (1000 if value > 1e11 else 1)          # epoch seconds or ms
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        dt = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
        if dt.tzinfo is None:
            flags.append("received_at_assumed_utc")
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (ValueError, OverflowError, OSError):
        flags.append("received_at_unreadable")
        return None


def _fallback_id(raw: Any) -> str:
    """Stable id for a ticket that arrived without one, so re-runs log the same id."""
    blob = json.dumps(raw, sort_keys=True, default=str) if isinstance(raw, dict) else repr(raw)
    return "NOID-" + hashlib.sha1(blob.encode("utf-8", "replace")).hexdigest()[:10]


# ------------------------------------------------------------------ tickets: the main function
def normalise_ticket(raw: Any) -> NormalisedTicket:
    """Normalise one ticket. Never raises."""
    flags: list[str] = []
    originals = {"original_subject": "", "original_body": "", "original_channel": ""}
    try:
        if isinstance(raw, (bytes, bytearray)):
            raw = to_str(raw, flags)
        if isinstance(raw, str):                          # a JSON string, or plain text
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                raw = {"body": raw}
                flags.append("plain_text_input")
        if not isinstance(raw, dict):
            return NormalisedTicket(ticket_id=_fallback_id(raw), channel=Channel.unknown,
                                    original_body=to_str(raw, []), processable=False,
                                    flags=flags + ["not_a_ticket_object"])

        # Capture the originals first, so they survive whatever happens below.
        raw_subject, raw_body, raw_channel = _get(raw, "subject"), _get(raw, "body"), _get(raw, "channel")
        originals = {"original_subject": to_str(raw_subject, []), "original_body": to_str(raw_body, []),
                     "original_channel": to_str(raw_channel, [])}

        ticket_id = to_str(_get(raw, "ticket_id"), []).strip()
        if not ticket_id:
            ticket_id = _fallback_id(raw)
            flags.append("ticket_id_generated")

        channel = parse_channel(raw_channel, flags)
        subject = _SUBJECT_PREFIX.sub("", clean_text(raw_subject, flags))
        body = clean_text(raw_body, flags)

        if channel is Channel.chat and subject:
            # The Dataset Guide says chat has no subject; keep any we get as part of the message.
            body = f"{subject}\n{body}".strip()
            subject = ""
            flags.append("chat_subject_moved_to_body")
        cleaner = CHANNEL_CLEANERS.get(channel)
        if cleaner and body:
            body = cleaner(body, flags)
        if len(body) > MAX_BODY_CHARS:
            body = body[:MAX_BODY_CHARS].rsplit(" ", 1)[0]
            flags.append("body_truncated")

        if not body:
            flags.append("empty_body")
        # Subject first: for email, docs comments and forum it is the customer's own summary.
        # Skip it when the body already starts with it, so the same words are not counted twice.
        if subject and not body.lower().startswith(subject.lower()):
            text = f"{subject}\n\n{body}".strip()
        else:
            text = body or subject
        processable = len(text) >= MIN_USABLE_CHARS
        if not processable:
            flags.append("no_usable_text")

        customer_id = to_str(_get(raw, "customer_id"), flags).strip()
        if not customer_id:
            flags.append("customer_id_missing")
        received_at = parse_timestamp(_get(raw, "received_at"), flags)
        tier = _enum(_get(raw, "customer_tier"), Tier, flags, "customer_tier")
        region = _enum(_get(raw, "customer_region"), Region, flags, "customer_region", REGION_ALIASES)
        fluency = _enum(_get(raw, "language_fluency"), Fluency, flags, "language_fluency")

        labels, history = raw.get("labels"), raw.get("history")
        return NormalisedTicket(
            ticket_id=ticket_id,
            channel=channel,
            subject=subject,
            body=body,
            text=text,
            **originals,
            received_at=received_at,
            received_at_raw=to_str(_get(raw, "received_at"), []),
            customer_id=customer_id,
            customer_name=clean_text(_get(raw, "customer_name"), []),
            customer_tier=tier,
            customer_region=region,
            language_fluency=fluency,
            referenced_doc_ids=sorted(set(_DOC_ID.findall(text))),
            word_count=len(text.split()),
            processable=processable,
            flags=list(dict.fromkeys(flags)),             # de-duplicate, keep order
            eval_labels=labels if isinstance(labels, dict) else None,
            eval_history=history if isinstance(history, dict) else None,
        )
    except Exception as exc:  # last line of defence: one bad ticket must never stop a run
        log.exception("unexpected error normalising ticket")
        return NormalisedTicket(ticket_id=_fallback_id(raw), channel=Channel.unknown, processable=False,
                                flags=flags + [f"normalisation_error:{type(exc).__name__}"], **originals)


# ------------------------------------------------------------------ tickets: files
def read_ticket_file(path: Path) -> list[Any]:
    """Accepts a JSON list, {"tickets": [...]}, a single ticket object, or JSON Lines."""
    text = Path(path).read_text(encoding="utf-8-sig")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        records = []
        for n, line in enumerate(text.splitlines(), 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    log.warning("line %d is not valid JSON; kept as an unprocessable ticket", n)
                    records.append(line)
        return records
    if isinstance(data, dict):
        return data["tickets"] if isinstance(data.get("tickets"), list) else [data]
    if isinstance(data, list):
        return data
    raise ValueError(f"{path}: expected a list of tickets")


def load_tickets(path: Path) -> list[NormalisedTicket]:
    return [normalise_ticket(r) for r in read_ticket_file(path)]


def summarise(tickets: Iterable[NormalisedTicket]) -> dict:
    tickets = list(tickets)
    return {
        "tickets": len(tickets),
        "processable": sum(t.processable for t in tickets),
        "by_channel": dict(Counter(t.channel.value for t in tickets)),
        "flags": dict(Counter(f for t in tickets for f in t.flags).most_common()),
        "duplicate_ids": [i for i, c in Counter(t.ticket_id for t in tickets).items() if c > 1],
    }


# ################################################################## PART 2: DOCS

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_COLLECTION = "kb_chunks"
REQUIRED_FIELDS = ("doc_id", "title", "content")
# all-MiniLM-L6-v2 reads at most 256 word pieces; text past that is silently ignored when
# embedding. ~180 words is a conservative proxy, used only to warn.
MAX_WORDS_BEFORE_TRUNCATION = 180

# Which markdown "## " sections go into which chunk.
SECTION_GROUPS = {
    "problem": ("symptoms", "common causes"),
    "resolution": ("resolution",),
    "notes": ("notes",),
}


# ------------------------------------------------------------------ docs: data shapes
@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    section: str
    text: str
    metadata: dict = field(default_factory=dict)


class Embedder(Protocol):
    model_name: str

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class SentenceTransformerEmbedder:
    """all-MiniLM-L6-v2 by default; vectors are L2-normalised so cosine distance is meaningful."""

    def __init__(self, model_name: str = DEFAULT_MODEL, batch_size: int = 32):
        from sentence_transformers import SentenceTransformer  # imported lazily: heavy

        self.model_name = model_name
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._model.encode(list(texts), batch_size=self.batch_size,
                                     normalize_embeddings=True, show_progress_bar=False)
        return [v.tolist() for v in vectors]


# ------------------------------------------------------------------ docs: normalise
def normalise_doc_text(text: str) -> str:
    """Unicode NFKC, unified newlines, markdown emphasis removed, whitespace tidied.

    Inline code (`cloudserve auth login`) is kept, backticks included: commands are exactly
    what a customer needs to copy.
    """
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace(" ", " ")
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)           # **bold** -> bold
    text = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"\1", text)     # _italic_ -> italic
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_sections(content: str) -> dict[str, str]:
    """Map lower-cased '## heading' -> section body. Text before the first heading is ignored
    (it repeats the title and the applies-to line, which every chunk carries anyway)."""
    sections: dict[str, str] = {}
    parts = re.split(r"^##\s+(.+?)\s*$", content, flags=re.MULTILINE)
    # parts = [preamble, heading1, body1, heading2, body2, ...]
    for heading, body in zip(parts[1::2], parts[2::2]):
        key = heading.strip().lower()
        body = body.strip()
        if body:
            sections[key] = body
    return sections


# ------------------------------------------------------------------ docs: validate
def validate_docs(raw: object) -> tuple[list[dict], list[dict]]:
    """Return (valid_docs, skipped) where skipped entries record why. Never raises on bad rows."""
    if not isinstance(raw, list):
        raise ValueError("documentation file must contain a JSON list of articles")
    valid, skipped, seen = [], [], set()
    for i, doc in enumerate(raw):
        ref = doc.get("doc_id", f"<index {i}>") if isinstance(doc, dict) else f"<index {i}>"
        if not isinstance(doc, dict):
            skipped.append({"doc": ref, "reason": "not an object"})
            continue
        missing = [f for f in REQUIRED_FIELDS if not str(doc.get(f) or "").strip()]
        if missing:
            skipped.append({"doc": ref, "reason": f"missing or empty: {', '.join(missing)}"})
            continue
        if doc["doc_id"] in seen:
            skipped.append({"doc": ref, "reason": "duplicate doc_id (first occurrence kept)"})
            continue
        seen.add(doc["doc_id"])
        valid.append(doc)
    return valid, skipped


# ------------------------------------------------------------------ docs: chunk
CHUNK_STRATEGIES = ("section", "document", "step", "window")
DEFAULT_STRATEGY = "document"   # chosen by `python -m src.retrieve compare`; see src/retrieve.py
WINDOW_CHARS, WINDOW_OVERLAP = 800, 120   # the Setup Guide's starting point, used by "window"


def _char_windows(text: str, size: int = WINDOW_CHARS, overlap: int = WINDOW_OVERLAP) -> list[str]:
    """Fixed-size windows cut at whitespace, like a recursive character splitter."""
    out, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = max(text.rfind("\n", start, end), text.rfind(" ", start, end))
            end = cut if cut > start + size // 2 else end
        out.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
        while start < len(text) and not text[start - 1].isspace():   # begin on a word boundary
            start += 1
    return [w for w in out if w]


def _chunk_groups(content: str, strategy: str) -> list[tuple[str, str, bool]]:
    """(section name, body, add title header?) for one article under a chunking strategy."""
    sections = split_sections(content)
    if strategy == "window":
        body = re.sub(r"^#\s+.*\n?", "", content).strip()
        return [(f"w{i}", w, False) for i, w in enumerate(_char_windows(body))]
    if "resolution" not in sections:
        return []
    named = {name: "\n\n".join(f"{h.capitalize()}:\n{sections[h]}" for h in wanted if h in sections)
             for name, wanted in SECTION_GROUPS.items()}
    if strategy == "document":
        return [("full", "\n\n".join(v for v in named.values() if v), True)]
    if strategy == "step":
        steps = [m.strip() for m in re.split(r"\n(?=\d+\.\s)", sections["resolution"]) if m.strip()]
        groups = [("problem", named["problem"], True)] if named.get("problem") else []
        groups += [(f"resolution-{i + 1}", f"Resolution step:\n{st}", True) for i, st in enumerate(steps)]
        return groups + ([("notes", named["notes"], True)] if named.get("notes") else [])
    return [(name, body, True) for name, body in named.items() if body]   # "section"


def chunk_document(doc: dict, source_file: str = "", strategy: str = DEFAULT_STRATEGY) -> list[Chunk]:
    """Split one article into chunks. Strategies (compared in src/retrieve.py `compare`):

    document  the whole article as one chunk, with the title (the default)
    section   problem (symptoms + causes) / resolution (all steps) / notes, each with the title
    step      problem / one chunk per resolution step / notes, each with the title
    window    ~800-character windows with 120 overlap and no title: the Setup Guide's starting point
    """
    if strategy not in CHUNK_STRATEGIES:
        raise ValueError(f"unknown chunking strategy {strategy!r}; choose from {CHUNK_STRATEGIES}")
    doc_id = str(doc["doc_id"]).strip()
    title = normalise_doc_text(str(doc["title"]))
    applies_to = normalise_doc_text(str(doc.get("applies_to") or "All plans"))
    category = normalise_doc_text(str(doc.get("category") or "uncategorised"))
    related = doc.get("related_docs") or []
    content = normalise_doc_text(str(doc["content"]))
    header = f"{title}\nApplies to: {applies_to}"

    groups = _chunk_groups(content, strategy)
    if not groups:
        # Unexpected layout: keep the whole article as one chunk rather than lose it.
        log.warning("%s: no Resolution section found; storing as a single chunk", doc_id)
        groups = [("full", re.sub(r"^#\s+.*\n?", "", content).strip(), True)]

    chunks = []
    for index, (section, body, with_header) in enumerate(groups):
        text = f"{header}\n\n{body}" if with_header else body
        if len(text.split()) > MAX_WORDS_BEFORE_TRUNCATION:
            log.debug("%s::%s has %d words; the embedding may ignore the end of it",
                      doc_id, section, len(text.split()))
        chunks.append(Chunk(
            chunk_id=f"{doc_id}::{section}",
            doc_id=doc_id,
            section=section,
            text=text,
            metadata={
                "doc_id": doc_id,
                "title": title,
                "category": category,
                "applies_to": applies_to,
                "section": section,
                "chunk_index": index,
                "strategy": strategy,
                # Chroma metadata values must be scalars, so lists are joined.
                "related_docs": ",".join(str(r) for r in related),
                "last_reviewed_days_ago": int(doc.get("last_reviewed_days_ago") or 0),
                "content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
                "source_file": source_file,
            },
        ))
    return chunks


def chunk_corpus(docs: Iterable[dict], source_file: str = "", strategy: str = DEFAULT_STRATEGY) -> list[Chunk]:
    chunks = [c for d in docs for c in chunk_document(d, source_file, strategy)]
    ids = [c.chunk_id for c in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("chunk ids are not unique")  # cannot happen after validate_docs
    return chunks


def chroma_client(persist_dir: Path):
    """A local Chroma client with anonymous usage telemetry switched off: nothing about the
    corpus or the queries leaves the machine, and no telemetry errors clutter the logs."""
    import chromadb
    from chromadb.config import Settings
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
    return chromadb.PersistentClient(path=str(persist_dir), settings=Settings(anonymized_telemetry=False))


# ------------------------------------------------------------------ docs: store
def store_chunks(chunks: Sequence[Chunk], embedder: Embedder, persist_dir: Path,
                 collection_name: str = DEFAULT_COLLECTION) -> int:
    """Rebuild the collection from scratch, so re-running ingest never leaves stale chunks."""

    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chroma_client(persist_dir)
    try:
        client.delete_collection(collection_name)
    except Exception:  # collection did not exist yet
        pass
    # embedding_function=None: we always pass our own vectors, so Chroma never downloads its own model.
    collection = client.create_collection(
        name=collection_name,
        embedding_function=None,
        metadata={"hnsw:space": "cosine", "embedding_model": embedder.model_name},
    )
    vectors = embedder.embed([c.text for c in chunks])
    if len(vectors) != len(chunks):
        raise RuntimeError("embedder returned a different number of vectors than chunks")
    collection.add(
        ids=[c.chunk_id for c in chunks],
        documents=[c.text for c in chunks],
        metadatas=[{**c.metadata, "embedding_model": embedder.model_name} for c in chunks],
        embeddings=vectors,
    )
    return collection.count()


def write_manifest(chunks: Sequence[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps({"chunk_id": c.chunk_id, "text": c.text, "metadata": c.metadata},
                                ensure_ascii=False) + "\n")


def quick_check(query: str, embedder: Embedder, persist_dir: Path,
                collection_name: str = DEFAULT_COLLECTION, k: int = 3) -> list[dict]:
    """Day-two checkpoint helper: show the top-k chunks for a question."""

    collection = chroma_client(persist_dir).get_collection(collection_name)
    res = collection.query(query_embeddings=embedder.embed([query]), n_results=k,
                           include=["distances", "metadatas"])
    return [{"chunk_id": cid, "similarity": round(1 - dist, 3), "title": meta["title"]}
            for cid, dist, meta in zip(res["ids"][0], res["distances"][0], res["metadatas"][0])]


# ------------------------------------------------------------------ docs: pipeline
def ingest_docs(docs_path: Path, persist_dir: Path, collection_name: str = DEFAULT_COLLECTION,
                embedder: Embedder | None = None, manifest_path: Path | None = None,
                strategy: str = DEFAULT_STRATEGY) -> dict:
    started = time.time()
    raw = json.loads(Path(docs_path).read_text(encoding="utf-8"))
    docs, skipped = validate_docs(raw)
    for s in skipped:
        log.warning("skipped %s: %s", s["doc"], s["reason"])
    chunks = chunk_corpus(docs, source_file=Path(docs_path).name, strategy=strategy)
    if not chunks:
        raise ValueError("no usable articles found; nothing was ingested")

    embedder = embedder or SentenceTransformerEmbedder()
    stored = store_chunks(chunks, embedder, persist_dir, collection_name)
    manifest_path = manifest_path or persist_dir / "chunks.jsonl"
    write_manifest(chunks, manifest_path)

    by_section: dict[str, int] = {}
    for c in chunks:
        by_section[c.section] = by_section.get(c.section, 0) + 1
    report = {
        "docs_in_file": len(raw), "docs_ingested": len(docs), "docs_skipped": skipped,
        "chunks": len(chunks), "chunks_stored": stored, "chunks_by_section": by_section,
        "collection": collection_name, "embedding_model": embedder.model_name, "strategy": strategy,
        "persist_dir": str(persist_dir), "manifest": str(manifest_path),
        "seconds": round(time.time() - started, 2),
        "ingested_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (persist_dir / "ingest_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


# ================================================================== command line
def _run_tickets(args: argparse.Namespace) -> int:
    try:
        tickets = load_tickets(args.input)
    except (OSError, ValueError) as exc:
        log.error("could not read %s: %s", args.input, exc)
        return 1
    exclude = None if args.include_eval else {"eval_labels", "eval_history"}
    if args.sample:
        seen = set()
        for t in tickets:
            if t.channel not in seen:
                seen.add(t.channel)
                print(t.model_dump_json(indent=2, exclude=exclude), "\n")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as fh:
            for t in tickets:
                fh.write(t.model_dump_json(exclude=exclude) + "\n")
    print(json.dumps(summarise(tickets), indent=2))
    return 0


def _run_docs(args: argparse.Namespace) -> int:
    if not args.docs.is_file():
        log.error("documentation file not found: %s", args.docs)
        return 1
    try:
        embedder = SentenceTransformerEmbedder(args.model)
    except Exception as exc:  # network, disk or package problem; nothing to ingest without it
        log.error("could not load embedding model %r: %s. The first run downloads it (about 90 MB) "
                  "from Hugging Face; after that it is cached locally.", args.model, exc)
        return 2
    try:
        report = ingest_docs(args.docs, args.persist_dir, args.collection, embedder, args.manifest,
                             strategy=args.strategy)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        log.error("docs ingest failed: %s", exc)
        return 1
    log.info("ingested %d docs into %d chunks %s in %.1fs (%d skipped)", report["docs_ingested"],
             report["chunks"], report["chunks_by_section"], report["seconds"], len(report["docs_skipped"]))
    if args.check:
        for hit in quick_check(args.check, embedder, args.persist_dir, args.collection):
            print(f"{hit['similarity']:.3f}  {hit['chunk_id']:<28} {hit['title']}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    # Defaults come from .env (see .env.example), so ingest and retrieval read the same store.
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:  # python-dotenv missing: fall back to real environment variables
        pass
    env_path = Path(os.getenv("CHROMA_PATH", "./storage/chroma"))
    env_model = os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL)

    p = argparse.ArgumentParser(prog="python -m src.ingest",
                                description="Bring tickets and documentation into the system.")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    pt = sub.add_parser("tickets", help="normalise tickets from all four channels")
    pt.add_argument("--input", required=True, type=Path, help="ticket file (JSON list or JSON Lines)")
    pt.add_argument("--output", type=Path, help="write normalised tickets here as JSON Lines")
    pt.add_argument("--sample", action="store_true", help="print one normalised ticket per channel")
    pt.add_argument("--include-eval", action="store_true", help="keep eval_labels/eval_history in output")

    pd = sub.add_parser("docs", help="build the documentation vector index")
    pd.add_argument("--docs", default=Path("data/documentation.json"), type=Path,
                    help="path to documentation.json (default data/documentation.json)")
    pd.add_argument("--persist-dir", default=env_path, type=Path,
                    help=f"Chroma folder (default CHROMA_PATH or ./storage/chroma; now {env_path})")
    pd.add_argument("--collection", default=os.getenv("CHROMA_COLLECTION", DEFAULT_COLLECTION))
    pd.add_argument("--model", default=env_model,
                    help=f"sentence-transformers model (default EMBEDDING_MODEL; now {env_model})")
    pd.add_argument("--manifest", type=Path, help="chunk manifest path (default <persist-dir>/chunks.jsonl)")
    pd.add_argument("--strategy", choices=CHUNK_STRATEGIES, default=os.getenv("CHUNK_STRATEGY", DEFAULT_STRATEGY),
                    help="chunking strategy (default CHUNK_STRATEGY or section); compare with src.retrieve compare")
    pd.add_argument("--check", metavar="QUERY", help="after ingesting, print the top 3 chunks for QUERY")

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    return _run_tickets(args) if args.command == "tickets" else _run_docs(args)


if __name__ == "__main__":
    sys.exit(main())
