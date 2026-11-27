"""Channel-aware response formatting: one checked reply, presented the way each channel expects.

The pipeline produces one reply: cited claims, then a "Sources:" list (src/generate.py), checked by
the guardrails. format_reply() then lays it out for the channel it goes back to, with fixed text only:
no model, and nothing from the customer (no name, no address), so the guardrails' verdict still holds.

  email         plain text. A subject line ("Re: ..."), a greeting, the answer, the sources as full
                article links, the ticket reference for replies, and a signature.
  chat          short. The answer without the boilerplate around it, and the sources as compact links.
                Long answers are cut to the first steps, with a pointer to the full article.
  forum         public Markdown. No ticket reference or account details (anyone can read it), sources
                as Markdown links, and a note that this is an automated answer that a moderator can correct.
  docs_comment  brief, under the article. The first steps only, and a link to the article that covers it.

DOCS_BASE_URL (environment) is where articles live; the default is an illustrative address.
"""
from __future__ import annotations

import os
import re
from typing import Optional, Sequence

CHAT_MAX_STEPS = 3
DOCS_MAX_STEPS = 2
SIGNATURE = "CloudServe Support\n(prepared with Cirrus AI)"
_SOURCES = re.compile(r"\n+Sources:\n(.*)\Z", re.S)
_BOILERPLATE = (r"^Thanks for getting in touch\.\s*", r"\n+If this does not resolve it[^\n]*",
                r"\n+This reply was prepared automatically[^\n]*")


def docs_url(doc_id: str) -> str:
    return f"{os.getenv('DOCS_BASE_URL', 'https://docs.cloudserve.example/articles').rstrip('/')}/{doc_id}"


def split_reply(reply: str) -> tuple[str, list[tuple[str, str, str]], str]:
    """(the answer, [(marker, title, doc_id)], extras) from a pipeline reply. extras holds what the
    channel policy appended after the sources (related articles, existing forum threads)."""
    text = reply or ""
    sources: list[tuple[str, str, str]] = []
    extras: list[str] = []
    m = _SOURCES.search(text)
    if m:
        for line in m.group(1).splitlines():
            s = re.match(r"\s*(\[\d+\])\s*(.*?)\s*\((DOC-[A-Z]+-\d{3})\)\s*$", line)
            if s:
                sources.append(s.groups())
            elif line.strip():
                extras.append(line.rstrip())
        text = text[:m.start()]
    for pattern in _BOILERPLATE:
        text = re.sub(pattern, "", text.strip())
    return text.strip(), sources, "\n".join(extras)


def _steps(answer: str) -> list[str]:
    """The answer's sentences, each with its citation marker."""
    return [s.strip() for s in re.split(r"(?<=\])\s+(?=[A-Z])", answer) if s.strip()]


def format_reply(channel: str, reply: str, ticket_id: str = "", subject: str = "",
                 fixed: bool = False) -> dict:
    """{"channel", "format", "subject", "body"} for one reply. fixed=True is a constant reply
    (greeting, refusal, clarifying question): it has no sources and is laid out as it is."""
    answer, sources, extras = (reply.strip(), [], "") if fixed else split_reply(reply)
    links = [(mark, title, doc, docs_url(doc)) for mark, title, doc in sources]
    if channel == "email":
        lines = ["Hello,", "", answer]
        if links:
            lines += ["", "Sources:"] + [f"{m} {t}: {u}" for m, t, _, u in links]
        if not fixed:
            lines += ["", "If this does not solve it, just reply to this email and a member of the team will help."]
        if ticket_id:
            lines += ["", f"Ticket reference: {ticket_id}"]
        lines += ["", SIGNATURE]
        subj = subject.strip()
        subj = subj if not subj or subj.lower().startswith("re:") else f"Re: {subj}"
        return {"channel": channel, "format": "text/plain", "subject": subj or "Your CloudServe support request",
                "body": "\n".join(lines)}
    if channel == "chat":
        steps = _steps(answer)
        short = " ".join(steps[:CHAT_MAX_STEPS])
        if len(steps) > CHAT_MAX_STEPS and links:
            short += f" The full steps are in {links[0][1]}."
        body = short + ("\n\n" + " · ".join(f"{m} [{t}]({u})" for m, t, _, u in links) if links else "")
        return {"channel": channel, "format": "text/markdown", "subject": "", "body": body}
    if channel == "forum":
        body = answer
        if links:
            body += "\n\n**Sources**\n" + "\n".join(f"- {m} [{t}]({u})" for m, t, _, u in links)
        if extras:
            body += "\n\n" + extras
        body += ("\n\n*Answered automatically by Cirrus AI from the CloudServe documentation. A moderator can "
                 "correct or add to this answer.*")
        return {"channel": channel, "format": "text/markdown", "subject": "", "body": body}
    if channel == "docs_comment":
        steps = _steps(answer)
        body = " ".join(steps[:DOCS_MAX_STEPS])
        if links:
            body += "\n\n" + "See: " + ", ".join(f"[{t}]({u})" for _, t, _, u in links)
        if extras:
            body += "\n\n" + extras
        return {"channel": channel, "format": "text/markdown", "subject": "", "body": body}
    return {"channel": channel, "format": "text/plain", "subject": "", "body": reply}


def formatted_sources(reply: str) -> Sequence[dict]:
    return [{"citation": m, "title": t, "doc_id": d, "url": docs_url(d)} for m, t, d in split_reply(reply)[1]]


def preview(channel: str, reply: Optional[str], **kw) -> Optional[dict]:
    return format_reply(channel, reply, **kw) if reply else None
