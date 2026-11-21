"""Generate: draft a reply grounded in the retrieved passages, with a citation on every claim.

Build Specification (Generate) -> where it is met                            -> tests (tests/test_generate.py)
--------------------------------------------------------------------------------------------------------------
Grounded answer, citations attached to claims   Claim.citations; every label checked against the   test_req1_*
                                                passages actually retrieved
States plainly when it does not know             status "insufficient_information" + fixed wording  test_req2_*
Ticket content separated from instructions       instructions only in the system message; ticket    test_req3_*
                                                fenced as data, fence-breaking text removed
Defined structure the code can parse reliably    JSON schema -> pydantic; one repair attempt;      test_req4_*
                                                otherwise the extractive fallback

Usage (run from the project root)
-----
    python -m src.generate check                                            # one live call to the provider
    python -m src.generate run --input data/validation_tickets.json --sample

Generators
----------
llm         LangChain's ChatOpenAI against an OpenAI-compatible API (OpenRouter by default; Groq
            works by changing LLM_BASE_URL). Temperature 0, responses cached on disk
            (LLM_CACHE_DIR) so reruns are reproducible and cost nothing. Rate limits and server
            errors are retried with backoff. LLM_CLIENT=http swaps in a plain HTTP client.
extractive  no model: the numbered resolution steps of the best passage, each cited. Used when
            no API key is configured, when the provider is unavailable, or when the model's output
            is still invalid after one repair attempt. Results then say degraded=True and why.

What the model sees
-------------------
All three prompts are versioned .txt files in prompts/build/, read by load_prompt() below:
System message: generate_answer (rules, the JSON shape). It never contains customer text.
User message: generate_answer_user, filled with the passages, labelled P1..Pn, and the ticket's
subject and body inside <customer_ticket> ... </customer_ticket>, introduced as data that must not
be obeyed. A reply that fails validation gets repair_reply with the exact error, once. Anything in
the ticket that looks like one of these tags is replaced before sending, so a customer cannot
close the data block and write their own instructions. The customer's name, id, tier and region
are never sent to the model: it does not need them.

Personal data in the ticket (emails, customer ids, keys, phone and card numbers) is masked before
the text is sent to any model, so it never leaves CloudServe in a prompt. Customers in a region
listed in LLM_BLOCKED_REGIONS (for example LLM_BLOCKED_REGIONS=europe, to keep their ticket text
inside CloudServe for data-residency reasons) get the extractive reply: no model is called for them.

Languages: a ticket not written in English is translated into English first (translate_ticket);
the reply is drafted and verified in English, then translated into the customer's language
(translate_reply) and sent with the English version below it, so the cited, checked wording is
always there. Customers writing in a second language get a style instruction for short, plain
sentences.

The reply shown to the customer is assembled by code from the validated claims: a fixed opening,
the claims with numbered citation markers, a fixed closing and a numbered source list. The model
cannot add text outside a cited claim.
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
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any, Callable, Optional, Protocol, Sequence

from pydantic import BaseModel, ValidationError, field_validator

from src.ingest import LANGUAGE_NAMES, NormalisedTicket, normalise_ticket
from src.logging_store import mask_pii
from src.retrieve import Passage, RetrievalResult

log = logging.getLogger("generate")

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "meta-llama/llama-3.1-8b-instruct"
DEFAULT_CACHE = "./storage/llm_cache"
MAX_PASSAGES = 3
MAX_ATTEMPTS = 2                      # first try + one repair

OPENING = "Thanks for getting in touch."
CLOSING = "If this does not resolve it, reply to this message and a member of the team will pick it up."
# Customers asked to know when a reply is automated, so they can decide how much to verify it.
AUTOMATED_NOTE = "This reply was prepared automatically from the CloudServe help articles listed below."
# The prompt register (Stage 3) ids, so the decision log can name the prompt as PR-xx vN.
PROMPT_IDS = {"generate_answer": "PR-01", "generate_answer_user": "PR-02", "repair_reply": "PR-03",
              "translate_ticket": "PR-04", "translate_ticket_user": "PR-05", "translate_reply": "PR-06",
              "translate_reply_user": "PR-07", "judge_answer": "PR-08"}
DONT_KNOW = ("I'm sorry, but I don't have documentation that answers this, so I won't guess. "
             "I've passed your ticket to a member of the support team, who will follow up with you.")
_FENCE = re.compile(r"<\s*/?\s*(customer_ticket|passages?|system|instructions)\b[^>]*>", re.I)


# ------------------------------------------------------------------ the prompt library
# Prompts are .txt files in prompts/build (run inside the system) and prompts/evaluation (used to
# judge output), named <name>.v<N>.txt. Each caller pins the version it loads; PROMPT_<NAME>=vN in
# the environment selects another version for a trial run without a code change.
PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"
SYSTEM_PROMPT = ("generate_answer", 2)        # prompts/build/generate_answer.v2.txt
USER_PROMPT = ("generate_answer_user", 1)     # prompts/build/generate_answer_user.v1.txt
REPAIR_PROMPT = ("repair_reply", 1)           # prompts/build/repair_reply.v1.txt
TRANSLATE_TICKET = ("translate_ticket", 1)            # prompts/build/translate_ticket.v1.txt
TRANSLATE_TICKET_USER = ("translate_ticket_user", 1)  # prompts/build/translate_ticket_user.v1.txt
TRANSLATE_REPLY = ("translate_reply", 1)              # prompts/build/translate_reply.v1.txt
TRANSLATE_REPLY_USER = ("translate_reply_user", 1)    # prompts/build/translate_reply_user.v1.txt
SIMPLE_STYLE = ("The customer is not writing in their first language: use short, simple sentences and "
                "common words, and avoid idioms.")


class PromptError(RuntimeError):
    pass


@dataclass(frozen=True)
class Prompt:
    name: str
    number: int
    text: str

    @property
    def version(self) -> str:
        """'<name>.v<N>@<sha8>': recorded with every decision, so an edited file is visible."""
        return f"{self.name}.v{self.number}@{hashlib.sha256(self.text.encode()).hexdigest()[:8]}"

    @property
    def ref(self) -> str:
        """'PR-01 v2': the prompt register id and version."""
        return f"{PROMPT_IDS.get(self.name, self.name)} v{self.number}"

    @property
    def placeholders(self) -> set[str]:
        return {m.group("named") or m.group("braced") for m in Template.pattern.finditer(self.text)
                if m.group("named") or m.group("braced")}

    def render(self, **values: Any) -> str:
        """Fill every $placeholder. Missing values raise; values are inserted as-is, never parsed."""
        missing = self.placeholders - set(values)
        if missing:
            raise PromptError(f"{self.name}: missing values for {sorted(missing)}")
        return Template(self.text).substitute(**{k: str(v) for k, v in values.items()})


def load_prompt(name: str, number: int = 1, kind: str = "build", root: Optional[Path] = None) -> Prompt:
    """Read prompts/<kind>/<name>.v<number>.txt."""
    override = os.getenv(f"PROMPT_{name.upper()}", "").strip().lower().lstrip("v")
    number = int(override) if override.isdigit() else number
    path = Path(root or os.getenv("PROMPTS_DIR", PROMPTS_DIR)) / kind / f"{name}.v{number}.txt"
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise PromptError(f"prompt file not found: {path}") from exc
    if not text:
        raise PromptError(f"prompt file is empty: {path}")
    return Prompt(name=name, number=number, text=text)


# ------------------------------------------------------------------ output
class Claim(BaseModel):
    text: str
    citations: list[str]              # chunk ids, e.g. DOC-API-001::full


class Citation(BaseModel):
    number: int                       # the [n] marker in answer_text
    chunk_id: str
    doc_id: str
    title: str


class GeneratedAnswer(BaseModel):
    ticket_id: str
    status: str                       # answered | insufficient_information | failed
    claims: list[Claim] = []
    answer_text: str = ""             # the reply as the customer would see it
    sources: list[Citation] = []
    missing_information: Optional[str] = None
    generator: str = ""               # "llm" | "extractive" | "none"
    model: str = ""
    prompt_version: str = ""
    attempts: int = 0
    degraded: bool = False
    error: str = ""
    raw_output: str = ""              # the model's last raw reply, for the decision log
    purpose: str = "reply"            # "reply" (may be sent) | "agent_draft" (for a person to review)
    prompt_ref: str = ""              # prompt register ids, e.g. "PR-01 v2+PR-02 v1+PR-03 v1"
    reply_language: str = "en"        # language of answer_text's main part
    answer_text_en: str = ""          # the English reply the guardrails verify (= answer_text if English)
    translation_error: str = ""       # why a reply could not be translated, if it could not
    data_policy: str = ""             # e.g. why no external model was used for this customer
    latency_ms: float = 0.0


class _ModelReply(BaseModel):
    """The JSON shape the model must return."""
    status: str
    claims: list[Any] = []
    missing_information: Optional[str] = None

    @field_validator("status")
    @classmethod
    def _status(cls, v: str) -> str:
        v = str(v).strip().lower().replace(" ", "_").replace("-", "_")
        aliases = {"answer": "answered", "insufficient": "insufficient_information",
                   "unknown": "insufficient_information", "not_enough_information": "insufficient_information"}
        v = aliases.get(v, v)
        if v not in ("answered", "insufficient_information"):
            raise ValueError(f"status must be 'answered' or 'insufficient_information', not {v!r}")
        return v


# ------------------------------------------------------------------ the model client
class LLMError(RuntimeError):
    pass


class LLMClient(Protocol):
    model: str

    def complete(self, messages: list[dict], temperature: float = 0.0, max_tokens: int = 600) -> str: ...


class OpenAICompatibleClient:
    """Chat completions over HTTP (OpenRouter, Groq, ...), with retries and an on-disk cache."""

    RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, base_url: str = DEFAULT_BASE_URL,
                 timeout: float = 30.0, max_retries: int = 3, cache_dir: Optional[Path] = None,
                 sleep: Callable[[float], None] = time.sleep):
        if not api_key:
            raise LLMError("no API key configured")
        self.api_key, self.model, self.base_url = api_key.strip(), model, base_url.rstrip("/")
        self.timeout, self.max_retries, self.sleep = timeout, max_retries, sleep
        self.cache_dir = Path(cache_dir) if cache_dir else None

    def _cache_path(self, payload: dict) -> Optional[Path]:
        if not self.cache_dir:
            return None
        key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        return self.cache_dir / f"{key}.json"

    def complete(self, messages: list[dict], temperature: float = 0.0, max_tokens: int = 600) -> str:
        import requests
        payload = {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        cache = self._cache_path(payload)
        if cache and cache.is_file():
            return json.loads(cache.read_text())["content"]
        last = ""
        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.post(f"{self.base_url}/chat/completions", json=payload, timeout=self.timeout,
                                     headers={"Authorization": f"Bearer {self.api_key}",
                                              "Content-Type": "application/json"})
            except requests.RequestException as exc:      # timeout, connection refused, DNS ...
                last = f"{type(exc).__name__}: {exc}"
            else:
                if resp.status_code == 200:
                    try:
                        content = resp.json()["choices"][0]["message"]["content"] or ""
                    except (ValueError, KeyError, IndexError, TypeError) as exc:
                        raise LLMError(f"unexpected response shape: {exc}") from exc
                    if cache:
                        cache.parent.mkdir(parents=True, exist_ok=True)
                        cache.write_text(json.dumps({"content": content, "model": self.model}))
                    return content
                last = f"HTTP {resp.status_code}: {resp.text[:200]}"
                if resp.status_code not in self.RETRY_STATUS:
                    raise LLMError(last)                  # 400/401/403/404: retrying will not help
                retry_after = resp.headers.get("Retry-After", "")
                if retry_after.replace(".", "", 1).isdigit():
                    self.sleep(min(float(retry_after), 20.0))
                    continue
            if attempt < self.max_retries:
                self.sleep(min(2 ** attempt, 8))
        raise LLMError(f"provider unavailable after {self.max_retries + 1} attempts ({last})")


class LangChainChatClient:
    """The model call through LangChain: `ChatOpenAI` pointed at an OpenAI-compatible API
    (OpenRouter by default). LangChain's client retries rate limits and server errors with
    backoff (max_retries); responses are cached on disk on top, as with the HTTP client.
    Any chat model can be passed in instead, which is how tests use LangChain's fake model."""

    def __init__(self, chat_model: Any = None, *, api_key: str = "", model: str = DEFAULT_MODEL,
                 base_url: str = DEFAULT_BASE_URL, timeout: float = 30.0, max_retries: int = 3,
                 max_tokens: int = 600, cache_dir: Optional[Path] = None):
        if chat_model is None:
            if not api_key:
                raise LLMError("no API key configured")
            from langchain_openai import ChatOpenAI
            chat_model = ChatOpenAI(model=model, api_key=api_key.strip(), base_url=base_url, temperature=0,
                                    timeout=timeout, max_retries=max_retries, max_tokens=max_tokens)
        self.chat_model = chat_model
        self.model = model
        self.cache_dir = Path(cache_dir) if cache_dir else None

    def complete(self, messages: list[dict], temperature: float = 0.0, max_tokens: int = 600) -> str:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
        key = hashlib.sha256(json.dumps({"model": self.model, "messages": messages, "temperature": temperature,
                                         "max_tokens": max_tokens}, sort_keys=True).encode()).hexdigest()
        cache = self.cache_dir / f"{key}.json" if self.cache_dir else None
        if cache and cache.is_file():
            return json.loads(cache.read_text())["content"]
        kinds = {"system": SystemMessage, "user": HumanMessage, "assistant": AIMessage}
        lc_messages = [kinds[m["role"]](content=m["content"]) for m in messages]
        try:
            result = self.chat_model.invoke(lc_messages)
        except Exception as exc:            # rate limited past retries, timeout, outage, bad key
            raise LLMError(f"provider call failed: {type(exc).__name__}: {exc}") from exc
        content = result.content if isinstance(result.content, str) else "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in result.content)
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps({"content": content, "model": self.model}))
        return content


# ------------------------------------------------------------------ prompt construction
def sanitise_ticket_text(text: str) -> tuple[str, bool]:
    """Replace anything shaped like our own tags, so ticket text cannot leave its data block."""
    cleaned, n = _FENCE.subn("[removed]", text)
    return cleaned, n > 0


def build_messages(system_prompt: str, ticket: NormalisedTicket, passages: Sequence[Passage],
                   user_template: Optional[Prompt] = None) -> tuple[list[dict], dict]:
    """The chat messages and a label map P1.. -> passage. The user message comes from the
    versioned template prompts/build/generate_answer_user.vN.txt."""
    user_template = user_template or load_prompt(*USER_PROMPT)
    labels = {f"P{i + 1}": p for i, p in enumerate(passages[:MAX_PASSAGES])}
    blocks = [f'<passage id="{label}" title="{p.title}">\n{p.text}\n</passage>' for label, p in labels.items()]
    subject, _ = sanitise_ticket_text(mask_pii(ticket.subject))
    body, _ = sanitise_ticket_text(mask_pii(ticket.body or ticket.text))
    ticket_block = (f"Subject: {subject}\n" if subject else "") + f"Message: {body}"
    user = user_template.render(passages="\n\n".join(blocks), ticket=ticket_block)
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}], labels


# ------------------------------------------------------------------ parsing and validation
def _extract_json(raw: str) -> dict:
    text = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", raw.strip(), flags=re.I)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object found in the reply")
    return json.loads(text[start:end + 1])


def _label(value: Any) -> str:
    v = str(value).strip().strip("[]").upper().replace(" ", "")
    return f"P{v}" if v.isdigit() else v


def parse_reply(raw: str, labels: dict[str, Passage]) -> tuple[str, list[Claim], Optional[str]]:
    """Validate the model's reply against the schema and the passages. Raises ValueError with a
    message the model can act on."""
    try:
        reply = _ModelReply.model_validate(_extract_json(raw))
    except (ValueError, ValidationError) as exc:      # JSONDecodeError is a ValueError
        raise ValueError(f"the reply is not valid JSON in the required shape ({str(exc).splitlines()[0]})")
    if reply.status == "insufficient_information":
        return reply.status, [], (reply.missing_information or "").strip() or None
    claims = []
    for i, item in enumerate(reply.claims, 1):
        if not isinstance(item, dict):
            raise ValueError(f"claim {i} is not an object with 'text' and 'citations'")
        text = " ".join(str(item.get("text", "")).split())
        cites = item.get("citations") or []
        cites = [cites] if isinstance(cites, (str, int)) else cites
        cites = list(dict.fromkeys(_label(c) for c in cites))
        if not text:
            raise ValueError(f"claim {i} has no text")
        if not cites:
            raise ValueError(f"claim {i} has no citations; every claim must cite a passage label")
        unknown = [c for c in cites if c not in labels]
        if unknown:
            raise ValueError(f"claim {i} cites {unknown}, which are not passage labels {list(labels)}")
        claims.append(Claim(text=text, citations=[labels[c].chunk_id for c in cites]))
    if not claims:
        raise ValueError("status is 'answered' but there are no claims")
    return reply.status, claims, None


def render(claims: Sequence[Claim], passages: Sequence[Passage]) -> tuple[str, list[Citation]]:
    """Assemble the customer-facing reply: numbered citation markers in order of first use."""
    by_chunk = {p.chunk_id: p for p in passages}
    numbers: dict[str, int] = {}
    sentences = []
    for c in claims:
        marks = []
        for cid in c.citations:
            numbers.setdefault(cid, len(numbers) + 1)
            marks.append(f"[{numbers[cid]}]")
        text = c.text if c.text.endswith((".", "!", "?")) else c.text + "."
        sentences.append(f"{text} {''.join(marks)}")
    sources = [Citation(number=n, chunk_id=cid, doc_id=by_chunk[cid].doc_id, title=by_chunk[cid].title)
               for cid, n in numbers.items()]
    src = "\n".join(f"[{s.number}] {s.title} ({s.doc_id})" for s in sources)
    return (f"{OPENING}\n\n" + " ".join(sentences) + f"\n\n{CLOSING}\n\n{AUTOMATED_NOTE}\n\nSources:\n{src}",
            sources)


# ------------------------------------------------------------------ extractive fallback
_STEP = re.compile(r"^\s*\d+\.\s+(.*\S)\s*$", re.M)


def extractive_claims(passages: Sequence[Passage]) -> list[Claim]:
    """The best passage's numbered resolution steps, each cited to it. No model involved."""
    if not passages:
        return []
    top = passages[0]
    steps = _STEP.findall(top.text.split("Resolution:", 1)[-1].split("Notes:", 1)[0])
    if not steps:
        sentences = re.split(r"(?<=[.!?])\s+", top.text.split("\n\n", 1)[-1])
        steps = [s for s in sentences if len(s.split()) > 3][:2]
    return [Claim(text=s.strip(), citations=[top.chunk_id]) for s in steps]


# ------------------------------------------------------------------ the generator
class CallBudget:
    """A cap on model calls per rolling hour (LLM_HOURLY_CALL_BUDGET; 0 = no cap), so a surge in
    volume cannot run up cost or exhaust a free tier unnoticed. Over the cap, replies are built from
    the articles without a model and say so (degraded), and translation is not attempted."""

    def __init__(self, limit: Optional[int] = None):
        self.limit = int(os.getenv("LLM_HOURLY_CALL_BUDGET", "600") if limit is None else limit)
        self.calls: list[float] = []

    def take(self) -> None:
        now = time.time()
        self.calls = [c for c in self.calls if now - c < 3600]
        if self.limit and len(self.calls) >= self.limit:
            raise LLMError(f"hourly model-call budget of {self.limit} reached (LLM_HOURLY_CALL_BUDGET)")
        self.calls.append(now)

    @property
    def used_this_hour(self) -> int:
        now = time.time()
        return sum(1 for c in self.calls if now - c < 3600)


class Generator:
    """Drafts answers. Its three prompts come from prompts/build/ at their pinned versions:
    generate_answer (system), generate_answer_user (user message), repair_reply (follow-up)."""

    def __init__(self, client: Optional[LLMClient] = None, prompts_dir: Optional[Path] = None,
                 unavailable_reason: str = ""):
        self.client = client
        self.unavailable_reason = unavailable_reason
        self.budget = CallBudget()
        try:
            self.system = load_prompt(*SYSTEM_PROMPT, root=prompts_dir)
            self.user_template = load_prompt(*USER_PROMPT, root=prompts_dir)
            self.repair = load_prompt(*REPAIR_PROMPT, root=prompts_dir)
            self.system_prompt = self.system.text
            self.prompt_version = "+".join(p.version for p in (self.system, self.user_template, self.repair))
            self.prompt_ref = "+".join(p.ref for p in (self.system, self.user_template, self.repair))
            self.translate_prompts = {k: load_prompt(*v, root=prompts_dir) for k, v in (
                ("ticket", TRANSLATE_TICKET), ("ticket_user", TRANSLATE_TICKET_USER),
                ("reply", TRANSLATE_REPLY), ("reply_user", TRANSLATE_REPLY_USER))}
        except PromptError as exc:
            self.system = self.user_template = self.repair = None
            self.translate_prompts = {}
            self.system_prompt, self.prompt_version, self.prompt_ref = "", "", ""
            self.client = None
            self.unavailable_reason = f"prompt file missing: {exc}"

    def generate(self, ticket: Any, retrieval: RetrievalResult, purpose: str = "reply") -> GeneratedAnswer:
        """Never raises. purpose="agent_draft" drafts a reply for a person to review and send."""
        started = time.perf_counter()
        t = ticket if isinstance(ticket, NormalisedTicket) else normalise_ticket(ticket)
        try:
            out = self._generate(t, retrieval)
            if out.status == "answered":
                self._localise(t, out)
        except Exception as exc:  # last line of defence
            log.exception("generation failed for %s", t.ticket_id)
            out = self._dont_know(t, generator="none", error=f"{type(exc).__name__}: {exc}", status="failed")
        out.purpose = purpose
        out.prompt_ref = self.prompt_ref
        if not out.answer_text_en:
            out.answer_text_en = out.answer_text
        out.latency_ms = round((time.perf_counter() - started) * 1000, 2)
        return out

    # ---------------- data policy and languages
    @staticmethod
    def blocked_region(t: NormalisedTicket) -> str:
        """Non-empty (the reason) when this customer's text must not be sent to an external model."""
        blocked = {r.strip().lower() for r in os.getenv("LLM_BLOCKED_REGIONS", "").split(",") if r.strip()}
        region = t.customer_region.value
        if region in blocked:
            return (f"data residency policy: ticket text from {region} customers is not sent to an external "
                    f"model (LLM_BLOCKED_REGIONS)")
        return ""

    def can_translate(self, t: NormalisedTicket) -> str:
        """'' if translation is possible for this ticket, else the reason it is not."""
        if self.client is None:
            return self.unavailable_reason or "no model configured"
        if not self.translate_prompts:
            return "translation prompts missing"
        return self.blocked_region(t)

    def translate_ticket(self, t: NormalisedTicket) -> tuple[Optional[str], str]:
        """(English text, '') or (None, reason). Personal data is masked before sending."""
        reason = self.can_translate(t)
        if reason:
            return None, reason
        text, _ = sanitise_ticket_text(mask_pii(t.original_text or t.text))
        messages = [{"role": "system", "content": self.translate_prompts["ticket"].text},
                    {"role": "user", "content": self.translate_prompts["ticket_user"].render(
                        language=LANGUAGE_NAMES.get(t.language, t.language), ticket=text)}]
        try:
            self.budget.take()
            out = self.client.complete(messages, max_tokens=800).strip()
        except Exception as exc:
            return None, f"translation failed: {exc}"
        return (out, "") if out else (None, "translation was empty")

    def _localise(self, t: NormalisedTicket, a: GeneratedAnswer) -> None:
        """Translate an answered reply into the customer's language, keeping the English below."""
        a.answer_text_en = a.answer_text
        if t.language in ("en", "und"):
            return
        reason = self.can_translate(t)
        if reason:
            a.translation_error = reason
            return
        messages = [{"role": "system", "content": self.translate_prompts["reply"].render(
                        language=LANGUAGE_NAMES.get(t.language, t.language))},
                    {"role": "user", "content": self.translate_prompts["reply_user"].render(reply=a.answer_text)}]
        try:
            self.budget.take()
            local = self.client.complete(messages, max_tokens=1000).strip()
        except Exception as exc:
            a.translation_error = f"translation failed: {exc}"
            return
        if not local:
            a.translation_error = "translation was empty"
            return
        a.reply_language = t.language
        a.answer_text = f"{local}\n\n---\nEnglish version:\n{a.answer_text_en}"

    def _generate(self, t: NormalisedTicket, r: RetrievalResult) -> GeneratedAnswer:
        passages = list(r.passages[:MAX_PASSAGES])
        if not passages:  # nothing to ground an answer in: say so, do not ask the model to improvise
            return self._dont_know(t, generator="none", missing="No documentation passage matched this ticket.")
        if self.client is None:
            return self._extractive(t, passages, self.unavailable_reason or "no model client configured")
        policy = self.blocked_region(t)
        if policy:
            out = self._extractive(t, passages, policy)
            out.degraded, out.data_policy = False, policy      # a policy choice, not a failure
            return out

        style = SIMPLE_STYLE if (t.language_fluency.value == "non_fluent" or t.translated) else ""
        system = self.system.render(style=style) if "style" in self.system.placeholders else self.system_prompt
        messages, labels = build_messages(system, t, passages, self.user_template)
        raw, last_error = "", ""
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                self.budget.take()
                raw = self.client.complete(messages)
            except Exception as exc:          # provider down, rate limited past retries, bad key, budget
                return self._extractive(t, passages, f"model unavailable: {exc}", attempts=attempt)
            try:
                status, claims, missing = parse_reply(raw, labels)
            except ValueError as exc:
                last_error = str(exc)
                messages = messages + [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": self.repair.render(error=last_error)}]
                continue
            if status == "insufficient_information":
                return self._dont_know(t, generator="llm", missing=missing, attempts=attempt, raw=raw)
            text, sources = render(claims, passages)
            return GeneratedAnswer(ticket_id=t.ticket_id, status="answered", claims=claims, answer_text=text,
                                   sources=sources, generator="llm", model=self.client.model,
                                   prompt_version=self.prompt_version, attempts=attempt, raw_output=raw)
        return self._extractive(t, passages, f"model output invalid after {MAX_ATTEMPTS} attempts: {last_error}",
                                attempts=MAX_ATTEMPTS, raw=raw)

    def _extractive(self, t: NormalisedTicket, passages: Sequence[Passage], reason: str,
                    attempts: int = 0, raw: str = "") -> GeneratedAnswer:
        claims = extractive_claims(passages)
        if not claims:
            return self._dont_know(t, generator="extractive", error=reason, degraded=True)
        text, sources = render(claims, passages)
        return GeneratedAnswer(ticket_id=t.ticket_id, status="answered", claims=claims, answer_text=text,
                               sources=sources, generator="extractive", prompt_version=self.prompt_version,
                               attempts=attempts, degraded=True, error=reason, raw_output=raw)

    def _dont_know(self, t: NormalisedTicket, generator: str, missing: Optional[str] = None, error: str = "",
                   attempts: int = 0, raw: str = "", status: str = "insufficient_information",
                   degraded: bool = False) -> GeneratedAnswer:
        return GeneratedAnswer(
            ticket_id=t.ticket_id, status=status, claims=[], answer_text=DONT_KNOW, sources=[],
            missing_information=missing or "The documentation does not cover what this ticket asks.",
            generator=generator, model=getattr(self.client, "model", "") if generator == "llm" else "",
            prompt_version=self.prompt_version, attempts=attempts, degraded=degraded, error=error, raw_output=raw)


def build_generator() -> Generator:
    """From the environment (.env). Without an API key the generator runs extractively and says so."""
    mode = os.getenv("GENERATOR", "llm").lower()
    key = os.getenv("OPENROUTER_API_KEY") or os.getenv("LLM_API_KEY") or ""
    if mode == "extractive":
        return Generator(None, unavailable_reason="GENERATOR=extractive")
    settings = dict(api_key=key, model=os.getenv("MODEL_NAME", DEFAULT_MODEL),
                    base_url=os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL),
                    timeout=float(os.getenv("LLM_TIMEOUT", "30")),
                    cache_dir=Path(os.getenv("LLM_CACHE_DIR", DEFAULT_CACHE)) if os.getenv("LLM_CACHE", "1") != "0"
                    else None)
    try:
        # LangChain by default (LLM_CLIENT=langchain); LLM_CLIENT=http uses the plain HTTP client.
        client: LLMClient = (OpenAICompatibleClient(**settings) if os.getenv("LLM_CLIENT", "langchain") == "http"
                             else LangChainChatClient(**settings))
    except LLMError as exc:
        log.warning("model client not available (%s): answers will be extractive", exc)
        return Generator(None, unavailable_reason=str(exc))
    return Generator(client)


# ------------------------------------------------------------------ command line
def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    p = argparse.ArgumentParser(prog="python -m src.generate", description="Draft grounded, cited answers.")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="make one live call to the model provider")
    r = sub.add_parser("run", help="ingest, classify, retrieve, route, then draft answers for auto_respond")
    r.add_argument("--input", type=Path, required=True)
    r.add_argument("--sample", action="store_true", help="print the first few drafted answers")
    r.add_argument("--output", type=Path, help="write every answer as JSON Lines")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    gen = build_generator()
    if args.command == "check":
        if gen.client is None:
            print(f"No model client: {gen.unavailable_reason}. Set OPENROUTER_API_KEY in .env.")
            return 1
        try:
            print(gen.client.complete([{"role": "user", "content": "Reply with the word ready."}], max_tokens=5))
            return 0
        except Exception as exc:
            print(f"Provider call failed: {exc}")
            return 1

    from src.classify import load_or_train
    from src.ingest import load_tickets
    from src.retrieve import build_retriever
    from src.route import Router, RoutingConfig
    try:
        tickets = load_tickets(args.input)
    except (OSError, ValueError) as exc:
        log.error("could not read %s: %s", args.input, exc)
        return 1
    clf, ret, router = load_or_train(), build_retriever(), Router(RoutingConfig.load())
    answers = []
    for t in tickets:
        rr = ret.retrieve(t)
        d = router.decide(t, clf.classify(t), rr)
        if d.action == "auto_respond":
            answers.append(gen.generate(t, rr))
    if args.sample:
        for a in answers[:3]:
            print(f"--- {a.ticket_id} ({a.generator}{', degraded' if a.degraded else ''})\n{a.answer_text}\n")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as fh:
            for a in answers:
                fh.write(a.model_dump_json() + "\n")
    from collections import Counter
    print(json.dumps({"tickets": len(tickets), "drafted": len(answers),
                      "by_status": dict(Counter(a.status for a in answers)),
                      "by_generator": dict(Counter(a.generator for a in answers)),
                      "degraded": sum(a.degraded for a in answers)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
