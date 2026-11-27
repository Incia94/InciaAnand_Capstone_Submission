"""Decision log: every automated decision, persisted in SQLite, reconstructable later (A8).

Usage (run from the project root)
-----
    python -m src.logging_store trail VAL-0001              # every decision for one ticket
    python -m src.logging_store reconcile --run-id <id>     # decisions vs tickets for one run
    python -m src.logging_store summary

Schema
------
The Setup Guide's table, column for column, plus six columns the Project Brief and the
acceptance criteria need:

    decision_id, created_at, ticket_id, stage, prediction, confidence, threshold, action_taken,
    reason, sources_used, guardrails, prompt_version, requirement_ids       (Setup Guide)
    run_id          groups one batch or harness run, so decisions reconcile against tickets (A8)
    input_digest    SHA-256 of the full input, to prove which input a decision was made on
    input_text      the input with personal data masked: the Brief requires the input to be
                    logged, and the log must not become a store of customers' personal data
    model_version   classifier / embedding / LLM version behind the decision
    details         JSON with the stage's full output (alternatives, scores, checks, handoff)
    latency_ms
and four columns the Governance Framework's minimum record names explicitly:
    input_summary      a short masked summary of the input (first 200 characters)
    model_name         which model or component decided: classifier, retriever, router, the LLM
    alternatives       the other options considered, with confidences (classification)
    guardrail_results  one word per check, e.g. {"pii": "pass", "grounding": "pass", "tone": "pass"}
requirement_ids holds the PRD requirement ids (FR-xx) and the acceptance criteria (Axx) a stage
serves; prompt_version holds the prompt register ids (PR-xx vN) and the exact file versions.
A database created by an earlier version is upgraded in place (the new columns are added).

One row per stage per ticket: ingest, classify, retrieve, route, generate, guardrail, and a final
`outcome` row (sent or escalated). A ticket that fails part-way still gets an `error` row and an
`outcome` row, so the count of outcomes always equals the count of tickets.

Recording never raises. If SQLite cannot be written, the record goes to a JSON Lines file next
to the database (<db>.fallback.jsonl), which `reconcile` also reads, and the failure is counted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sqlite3
import sys
import threading
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

log = logging.getLogger("logging_store")

DEFAULT_URL = "sqlite:///./storage/decisions.db"
STAGES = ("ingest", "classify", "retrieve", "route", "generate", "guardrail", "outcome", "error")
# Which requirement each stage serves. The Build Specification's acceptance criteria are given;
# add your own PRD requirement ids (e.g. "FR-03") so every logged decision traces to a requirement.
REQUIREMENT_IDS = {   # PRD functional requirements (FR) and Build Specification criteria (A) per stage
    "ingest": ["FR-01", "A2"], "translate": ["FR-12", "FR-09"], "classify": ["FR-02", "A3"],
    "retrieve": ["FR-03", "FR-10", "A4"], "route": ["FR-04", "FR-07", "FR-08", "FR-10", "FR-11", "A5"],
    "generate": ["FR-05", "FR-12", "A6"], "guardrail": ["FR-06", "FR-08", "FR-09", "A7"],
    "draft": ["FR-13"], "outcome": ["FR-14", "FR-15", "A8", "A9"], "error": ["FR-16", "A11"],
    "kill_switch": ["FR-16"], "screen": ["FR-04", "FR-06"], "tier": ["FR-04", "FR-05"],
    "review": ["FR-13", "FR-15"], "search": ["FR-03"], "dedup": ["FR-01"], "assignment": ["FR-15"],
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    decision_id     TEXT PRIMARY KEY,
    created_at      TEXT NOT NULL,
    ticket_id       TEXT NOT NULL,
    stage           TEXT NOT NULL,
    prediction      TEXT,
    confidence      REAL,
    threshold       REAL,
    action_taken    TEXT NOT NULL,
    reason          TEXT NOT NULL,
    sources_used    TEXT,
    guardrails      TEXT,
    prompt_version  TEXT,
    requirement_ids TEXT,
    run_id          TEXT,
    input_digest    TEXT,
    input_text      TEXT,
    model_version   TEXT,
    details         TEXT,
    latency_ms      REAL,
    input_summary   TEXT,
    model_name      TEXT,
    alternatives    TEXT,
    guardrail_results TEXT
);
CREATE INDEX IF NOT EXISTS idx_decisions_ticket ON decisions (ticket_id);
CREATE INDEX IF NOT EXISTS idx_decisions_run ON decisions (run_id, stage);
"""
COLUMNS = ("decision_id", "created_at", "ticket_id", "stage", "prediction", "confidence", "threshold",
           "action_taken", "reason", "sources_used", "guardrails", "prompt_version", "requirement_ids",
           "run_id", "input_digest", "input_text", "model_version", "details", "latency_ms",
           "input_summary", "model_name", "alternatives", "guardrail_results")
ADDED_COLUMNS = {"input_summary": "TEXT", "model_name": "TEXT", "alternatives": "TEXT", "guardrail_results": "TEXT"}
# Framework names for the guardrail checks (others keep their own names).
GUARDRAIL_NAMES = {"private_data": "pii", "claim_support": "grounding", "injection": "instruction_integrity",
                   "forbidden_claims": "scope_and_commitments"}


# ------------------------------------------------------------------ masking personal data
# Shared with the guardrails, which import these patterns.
PII_PATTERNS = [
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")),
    ("ip_address", re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")),
    ("customer_id", re.compile(r"\bCUST-\d+\b", re.I)),
    ("api_key", re.compile(r"\b(?:sk|pk|rk)[-_](?:live|test|proj)?[-_]?[A-Za-z0-9]{16,}\b")),
    ("aws_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b")),
    ("jwt", re.compile(r"\beyJ[\w-]{8,}\.[\w-]{8,}\.[\w-]{8,}\b")),
    ("secret_assignment", re.compile(r"\b(?:password|passwd|api[_ -]?key|secret|token)\s*[:=]\s*\S{4,}", re.I)),
    ("long_token", re.compile(r"\b(?=[A-Za-z0-9_-]*\d)(?=[A-Za-z0-9_-]*[A-Za-z])[A-Za-z0-9_-]{32,}\b")),
    ("postal_address", re.compile(r"\b\d{1,5}\s+(?:[A-Z][a-z]+\s){1,3}(?:Street|St|Road|Rd|Avenue|Ave|Lane|Ln|"
                                  r"Boulevard|Blvd|Drive|Dr|Way|Court|Ct|Place|Pl)\b\.?")),
    ("account_number", re.compile(r"\b(?:account|acct|iban)\s*(?:no\.?|number|#)?\s*[:#]?\s*(?=[A-Z0-9-]*\d)"
                                  r"[A-Z0-9][A-Z0-9-]{5,}\b", re.I)),
    ("iban", re.compile(r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){3,7}\b")),
]
PHONE_PATTERN = re.compile(r"(?<![\w-])\+?\d[\d\s().-]{7,}\d(?![\w-])")
CARD_PATTERN = re.compile(r"\b(?:\d[ -]?){13,19}\b")


def mask(value: str) -> str:
    v = value.strip()
    return v[0] + "*" * max(len(v) - 2, 1) + v[-1] if len(v) > 2 else "*" * len(v)


def mask_pii(text: Optional[str]) -> str:
    """Mask emails, ids, secrets, phone and card numbers, keeping the rest readable."""
    if not text:
        return ""
    for _, pattern in PII_PATTERNS:
        text = pattern.sub(lambda m: mask(m.group()), text)
    text = CARD_PATTERN.sub(lambda m: mask(m.group()) if len(re.sub(r"\D", "", m.group())) >= 13 else m.group(), text)
    return PHONE_PATTERN.sub(lambda m: mask(m.group()) if len(re.sub(r"\D", "", m.group())) >= 9 else m.group(), text)


def _mask_deep(value: Any) -> Any:
    """Mask personal data in every string inside a structure, leaving numbers and keys alone."""
    if isinstance(value, str):
        return mask_pii(value)
    if isinstance(value, dict):
        return {k: _mask_deep(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_mask_deep(v) for v in value]
    return value


STAGE_MODEL = {"ingest": "ingest (rules)", "translate": "llm", "classify": "classifier (tf-idf + logistic "
               "regression)", "retrieve": "retriever", "route": "router (rules + threshold)", "generate": "llm",
               "guardrail": "guardrails (rules)", "draft": "llm", "outcome": "pipeline", "error": "pipeline",
               "kill_switch": "pipeline", "screen": "scope screen (rules)",
               "tier": "tier and channel policy (rules)", "review": "person", "search": "retriever",
               "dedup": "deduplication (same customer, TF-IDF similarity)", "assignment": "person"}


def _summary(text: Optional[str], limit: int = 200) -> Optional[str]:
    if not text:
        return None
    flat = " ".join(mask_pii(text).split())
    return flat if len(flat) <= limit else flat[:limit].rsplit(" ", 1)[0] + "…"


def _handoff_for_log(h: Any) -> Optional[dict]:
    """The handoff as logged: the customer's name masked (the agent sees it; the audit log need not)."""
    if h is None:
        return None
    out = h.model_dump() if hasattr(h, "model_dump") else dict(h)
    name = (out.get("customer") or {}).get("name")
    if name:
        out["customer"]["name"] = mask(name)
        out["summary"] = (out.get("summary") or "").replace(name, mask(name))
    return out


def guardrail_summary(checks: Sequence[Any]) -> dict:
    """{framework name: pass | block | warn} for a list of guardrail CheckResults."""
    out = {}
    for c in checks:
        sev = {f.severity for f in c.findings}
        out[GUARDRAIL_NAMES.get(c.name, c.name)] = "block" if not c.passed else ("warn" if "warn" in sev else "pass")
    return out


def _json(value: Any) -> Optional[str]:
    """Serialise for the log. Every string inside is PII-masked first: free text such as a handoff
    summary quotes the customer, so masking only input_text would not be enough."""
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    value = json.loads(json.dumps(value, ensure_ascii=False, default=str))
    return json.dumps(_mask_deep(value), ensure_ascii=False)


def _sqlite_path(url: str) -> Path:
    if url.startswith("sqlite:///"):
        return Path(url[len("sqlite:///"):])
    if "://" in url:
        raise ValueError(f"only sqlite:/// URLs are supported, got {url!r}")
    return Path(url)


# ------------------------------------------------------------------ the store
class DecisionStore:
    def __init__(self, url: Optional[str] = None):
        self.path = _sqlite_path(url or os.getenv("DATABASE_URL", DEFAULT_URL))
        self.fallback_path = self.path.with_name(self.path.name + ".fallback.jsonl")
        self.write_failures = 0
        self.last_write_ok = True     # False only if a record could be written nowhere at all
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.path), timeout=10, check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(SCHEMA)
            have = {row[1] for row in self._conn.execute("PRAGMA table_info(decisions)")}
            for col, kind in ADDED_COLUMNS.items():          # upgrade a database from an earlier version
                if col not in have:
                    self._conn.execute(f"ALTER TABLE decisions ADD COLUMN {col} {kind}")
            self._conn.commit()
        except (sqlite3.Error, OSError) as exc:
            log.error("decision database %s unavailable (%s); writing to %s", self.path, exc, self.fallback_path)
            self._conn = None

    @property
    def healthy(self) -> bool:
        return self._conn is not None

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    # ---------------- writing
    def record(self, ticket_id: str, stage: str, action_taken: str, reason: str, *, prediction: Any = None,
               confidence: Optional[float] = None, threshold: Optional[float] = None, sources: Any = None,
               guardrails: Any = None, prompt_version: Optional[str] = None,
               requirement_ids: Optional[Sequence[str]] = None, run_id: Optional[str] = None,
               input_text: Optional[str] = None, model_version: Optional[str] = None, details: Any = None,
               latency_ms: Optional[float] = None, model_name: Optional[str] = None, alternatives: Any = None,
               guardrail_results: Any = None) -> str:
        """Write one decision. Returns its id. Never raises."""
        row = {
            "decision_id": uuid.uuid4().hex,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "ticket_id": str(ticket_id or "unknown"),
            "stage": stage,
            "prediction": prediction if isinstance(prediction, (str, type(None))) else _json(prediction),
            "confidence": None if confidence is None else float(confidence),
            "threshold": None if threshold is None else float(threshold),
            "action_taken": str(action_taken or "none"),
            "reason": mask_pii(str(reason or "no reason recorded")),
            "sources_used": _json(sources),
            "guardrails": _json(guardrails),
            "prompt_version": prompt_version,
            "requirement_ids": _json(list(requirement_ids) if requirement_ids is not None
                                     else REQUIREMENT_IDS.get(stage, [])),
            "run_id": run_id,
            "input_digest": hashlib.sha256(input_text.encode("utf-8", "replace")).hexdigest() if input_text else None,
            "input_text": mask_pii(input_text) if input_text else None,
            "model_version": model_version,
            "details": _json(details),
            "latency_ms": latency_ms,
            "input_summary": _summary(input_text),
            "model_name": model_name or STAGE_MODEL.get(stage),
            "alternatives": _json(alternatives),
            "guardrail_results": _json(guardrail_results),
        }
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.execute(f"INSERT INTO decisions ({', '.join(COLUMNS)}) VALUES "
                                       f"({', '.join('?' for _ in COLUMNS)})", [row[c] for c in COLUMNS])
                    self._conn.commit()
                    self.last_write_ok = True
                    return row["decision_id"]
                except sqlite3.Error as exc:
                    log.error("decision write failed (%s); using fallback file", exc)
            self.write_failures += 1
            try:
                self.fallback_path.parent.mkdir(parents=True, exist_ok=True)
                with self.fallback_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                self.last_write_ok = True
            except OSError as exc:  # nowhere left to write: say so loudly, but do not stop the run
                self.last_write_ok = False
                log.critical("DECISION NOT RECORDED for %s/%s: %s", ticket_id, stage, exc)
        return row["decision_id"]

    # ---------------- convenience writers for each pipeline stage
    def log_ingest(self, ticket: Any, run_id: Optional[str] = None) -> str:
        action = "normalised" if ticket.processable else "unprocessable"
        reason = (f"Ticket from {ticket.channel.value} normalised"
                  + (f"; repairs: {', '.join(ticket.flags)}" if ticket.flags else "") + ".")
        return self.record(ticket.ticket_id, "ingest", action, reason, prediction=ticket.channel.value,
                           run_id=run_id, input_text=ticket.text,
                           details={"flags": ticket.flags, "segment": ticket.segment(),
                                    "original_channel": ticket.original_channel})

    def log_classify(self, c: Any, ticket_text: str = "", run_id: Optional[str] = None) -> str:
        action = "fallback" if c.is_fallback else "classified"
        reason = (f"Could not classify ({c.fallback_reason})." if c.is_fallback else
                  f"Intent {c.intent} at {c.intent_confidence:.0%}, urgency {c.urgency} at {c.urgency_confidence:.0%}.")
        return self.record(c.ticket_id, "classify", action, reason, prediction=c.intent, confidence=c.intent_confidence,
                           run_id=run_id, input_text=ticket_text, model_version=c.model_version,
                           latency_ms=c.latency_ms,
                           alternatives=[{"value": a.label, "confidence": a.probability}
                                         for a in c.intent_alternatives[1:]],
                           details={"urgency": c.urgency, "urgency_confidence": c.urgency_confidence,
                                    "intent_alternatives": [a.model_dump() for a in c.intent_alternatives],
                                    "urgency_alternatives": [a.model_dump() for a in c.urgency_alternatives],
                                    "familiarity": c.familiarity})

    def log_retrieve(self, r: Any, ticket_text: str = "", run_id: Optional[str] = None) -> str:
        reason = (f"{len(r.passages)} passage(s) above threshold via {r.backend}." if r.passages
                  else f"No passage returned: {r.reason}")
        return self.record(r.ticket_id, "retrieve", r.status, reason,
                           prediction=r.passages[0].doc_id if r.passages else None, confidence=r.top_score,
                           threshold=r.threshold, sources=[{"doc_id": p.doc_id, "chunk_id": p.chunk_id,
                                                            "score": p.score} for p in r.passages],
                           model_name=f"retriever ({r.backend})",
                           run_id=run_id, input_text=ticket_text, latency_ms=r.latency_ms,
                           details={"candidates": [x.model_dump() for x in r.candidates], "backend": r.backend,
                                    "degraded": r.degraded, "reason": r.reason})

    def log_route(self, d: Any, ticket_text: str = "", run_id: Optional[str] = None) -> str:
        return self.record(d.ticket_id, "route", d.action, d.reason, prediction=d.intent, confidence=d.confidence,
                           threshold=d.threshold, sources=[s.model_dump() for s in d.sources], run_id=run_id,
                           input_text=ticket_text, model_version=d.config_version, latency_ms=d.latency_ms,
                           details={"decided_by": d.decided_by, "checks": [c.model_dump() for c in d.checks],
                                    "inputs_digest": d.inputs_digest,
                                    "handoff": _handoff_for_log(d.handoff)})

    def log_generate(self, a: Any, ticket_text: str = "", run_id: Optional[str] = None) -> str:
        reason = {"answered": f"Drafted {len(a.claims)} cited claim(s) with the {a.generator} generator.",
                  "insufficient_information": "The sources do not answer the question; said so instead of guessing.",
                  }.get(a.status, f"Generation failed: {a.error}")
        if a.degraded:
            reason += f" Degraded: {a.error}"
        ref = getattr(a, "prompt_ref", "")
        return self.record(a.ticket_id, "generate", a.status, reason, prediction=a.generator,
                           sources=[s.model_dump() for s in a.sources],
                           prompt_version=f"{ref} ({a.prompt_version})" if ref else a.prompt_version,
                           model_name=f"llm: {a.model}" if a.model else a.generator,
                           run_id=run_id, input_text=ticket_text, model_version=a.model or a.generator,
                           latency_ms=a.latency_ms,
                           details={"claims": [c.model_dump() for c in a.claims], "attempts": a.attempts,
                                    "answer_text": mask_pii(a.answer_text), "raw_output": mask_pii(a.raw_output),
                                    "missing_information": a.missing_information,
                                    "purpose": getattr(a, "purpose", "reply"),
                                    "reply_language": getattr(a, "reply_language", "en"),
                                    "translation_error": getattr(a, "translation_error", ""),
                                    "data_policy": getattr(a, "data_policy", "")})

    def log_guardrail(self, v: Any, ticket_text: str = "", run_id: Optional[str] = None) -> str:
        return self.record(v.ticket_id, "guardrail", "release" if v.released else "block", v.reason,
                           guardrails=[c.model_dump() for c in v.checks], prompt_version=v.prompt_version,
                           guardrail_results=guardrail_summary(v.checks),
                           run_id=run_id, input_text=ticket_text, model_version=v.generator,
                           latency_ms=v.latency_ms,
                           details={"blocked_by": v.blocked_by, "claims_checked": v.claims_checked,
                                    "claims_unsupported": v.claims_unsupported, "answer_digest": v.answer_digest})

    def log_outcome(self, ticket_id: str, outcome: str, reason: str, run_id: Optional[str] = None,
                    ticket_text: str = "", details: Any = None, latency_ms: Optional[float] = None) -> str:
        return self.record(ticket_id, "outcome", outcome, reason, run_id=run_id, input_text=ticket_text,
                           details=details, latency_ms=latency_ms)

    def log_error(self, ticket_id: str, where: str, exc: BaseException, run_id: Optional[str] = None,
                  ticket_text: str = "") -> str:
        return self.record(ticket_id, "error", f"error_in_{where}", f"{type(exc).__name__}: {exc}",
                           run_id=run_id, input_text=ticket_text)

    # ---------------- reading
    def _rows(self, where: str = "", params: Sequence[Any] = ()) -> list[dict]:
        rows: list[dict] = []
        with self._lock:
            if self._conn is not None:
                try:
                    cur = self._conn.execute(f"SELECT {', '.join(COLUMNS)} FROM decisions {where} "
                                             f"ORDER BY created_at, rowid", params)
                    rows = [dict(zip(COLUMNS, r)) for r in cur.fetchall()]
                except sqlite3.Error as exc:
                    log.error("decision read failed: %s", exc)
        if self.fallback_path.is_file():
            for line in self.fallback_path.read_text(encoding="utf-8").splitlines():
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
        return rows

    def trail(self, ticket_id: str) -> list[dict]:
        """Every decision for one ticket, in order, with JSON columns decoded."""
        out = []
        for r in self._rows("WHERE ticket_id = ?", (ticket_id,)):
            if r.get("ticket_id") != ticket_id:
                continue
            for col in ("sources_used", "guardrails", "requirement_ids", "details", "alternatives",
                        "guardrail_results"):
                if isinstance(r.get(col), str):
                    try:
                        r[col] = json.loads(r[col])
                    except ValueError:
                        pass
            out.append(r)
        return out

    def reconcile(self, run_id: str, ticket_ids: Optional[Iterable[str]] = None) -> dict:
        """Do the logged decisions account for every ticket processed in this run? (A8)

        Coverage is checked two ways. Counts: every processed ticket has exactly one outcome.
        Paths: every ticket's trail holds the decisions its path requires, so no route through the
        system decides without recording it: an ingest record for every ticket; a routing record for
        every ticket that was answered; a validation (guardrail) record for every generated reply; and
        no answer sent after validation blocked it. Any gap makes the run not reconciled."""
        rows = [r for r in self._rows("WHERE run_id = ?", (run_id,)) if r.get("run_id") == run_id]
        outcomes = Counter(r["ticket_id"] for r in rows if r["stage"] == "outcome")
        seen = set(r["ticket_id"] for r in rows)
        expected = list(ticket_ids) if ticket_ids is not None else sorted(seen)
        missing = [t for t in expected if outcomes.get(t, 0) == 0]
        duplicated = [t for t, n in outcomes.items() if n > 1]
        path_gaps = self._path_gaps(rows)
        return {
            "run_id": run_id, "tickets_expected": len(expected), "tickets_with_outcome": len(outcomes),
            "decisions": len(rows), "by_stage": dict(Counter(r["stage"] for r in rows)),
            "outcomes": dict(Counter(r["action_taken"] for r in rows if r["stage"] == "outcome")),
            "missing_outcome": missing, "duplicate_outcome": duplicated,
            "unexpected_tickets": sorted(set(outcomes) - set(expected)) if ticket_ids is not None else [],
            "write_failures": self.write_failures,
            "path_gaps": path_gaps,
            "decisions_per_ticket": round(len(rows) / len(seen), 2) if seen else 0.0,
            "reconciled": not missing and not duplicated and not path_gaps,
        }

    @staticmethod
    def _path_gaps(rows: Sequence[dict]) -> list[dict]:
        by: dict[str, list[dict]] = {}
        for r in rows:
            by.setdefault(r["ticket_id"], []).append(r)
        gaps = []
        for tid, trail in by.items():
            stages = [r["stage"] for r in trail]
            if "outcome" not in stages:
                continue                                        # counted as missing_outcome
            outcome = next(r["action_taken"] for r in trail if r["stage"] == "outcome")
            replies = [r for r in trail if r["stage"] == "generate"
                       and (_decode(r.get("details")) or {}).get("purpose", "reply") != "agent_draft"]
            checks = [r for r in trail if r["stage"] == "guardrail"]
            if "ingest" not in stages:
                gaps.append({"ticket_id": tid, "gap": "no ingest record"})
            if outcome in ("sent", "pending_review") and "route" not in stages:
                gaps.append({"ticket_id": tid, "gap": "answered without a routing record"})
            if replies and not checks:
                gaps.append({"ticket_id": tid, "gap": "a reply was generated without a validation record"})
            if outcome in ("sent", "pending_review") and replies and checks and checks[-1]["action_taken"] != "release":
                gaps.append({"ticket_id": tid, "gap": "answered although validation blocked the reply"})
        return gaps

    def export(self, run_id: Optional[str] = None, ticket_id: Optional[str] = None) -> list[dict]:
        """Decisions in the governance record format (governance_record), oldest first."""
        where, params = [], []
        if run_id:
            where.append("run_id = ?")
            params.append(run_id)
        if ticket_id:
            where.append("ticket_id = ?")
            params.append(ticket_id)
        rows = self._rows(("WHERE " + " AND ".join(where)) if where else "", params)
        rows = [r for r in rows if (not run_id or r.get("run_id") == run_id)
                and (not ticket_id or r.get("ticket_id") == ticket_id)]
        return [governance_record(r) for r in rows]

    def summary(self) -> dict:
        rows = self._rows()
        return {"decisions": len(rows), "tickets": len({r["ticket_id"] for r in rows}),
                "runs": len({r["run_id"] for r in rows if r.get("run_id")}),
                "by_stage": dict(Counter(r["stage"] for r in rows)), "database": str(self.path),
                "fallback_records": (len(self.fallback_path.read_text(encoding="utf-8").splitlines())
                                     if self.fallback_path.is_file() else 0)}


def new_run_id(prefix: str = "run") -> str:
    return f"{prefix}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:6]}"


# ------------------------------------------------------------------ the governance record format
STAGE_NAMES = {"ingest": "ingestion", "screen": "screening", "translate": "translation",
               "classify": "classification", "retrieve": "retrieval", "route": "routing", "tier": "tiering",
               "generate": "generation", "guardrail": "validation", "draft": "agent_draft", "outcome": "outcome",
               "review": "human_review", "kill_switch": "kill_switch", "error": "error", "search": "search",
               "dedup": "deduplication", "assignment": "assignment"}
_ACTIONS = {("route", "converse"): "auto_respond", ("guardrail", "release"): "auto_respond",
            ("guardrail", "block"): "block", ("review", "blocked"): "block"}


def _decode(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def governance_record(r: dict) -> dict:
    """One logged decision in the Governance Framework's record format. stage uses the framework's
    names (classification, routing, generation, validation...); action_taken is auto_respond,
    escalate or block where the stage decides that, and the stage's own action otherwise.
    prompt_version and requirement_ids say which prompt and which requirement produced it, so
    "was this behaviour intended?" can be answered from the record alone."""
    sources = _decode(r.get("sources_used")) or []
    alternatives = _decode(r.get("alternatives")) or []
    return {
        "decision_id": r.get("decision_id"),
        "timestamp": r.get("created_at"),
        "ticket_id": r.get("ticket_id"),
        "run_id": r.get("run_id"),
        "stage": STAGE_NAMES.get(r.get("stage"), r.get("stage")),
        "input_summary": r.get("input_summary"),
        "model": {"name": r.get("model_name"), "version": r.get("model_version")},
        "prediction": {"value": r.get("prediction"), "confidence": r.get("confidence")},
        "alternatives": [{"value": a.get("value"), "confidence": a.get("confidence")} for a in alternatives
                         if isinstance(a, dict)],
        "sources_used": [{"doc_id": s_.get("doc_id"), "score": s_.get("score")} for s_ in sources
                         if isinstance(s_, dict)],
        "threshold_applied": r.get("threshold"),
        "action_taken": _ACTIONS.get((r.get("stage"), r.get("action_taken")), r.get("action_taken")),
        "reason": r.get("reason"),
        "guardrail_results": _decode(r.get("guardrail_results")),
        "prompt_version": r.get("prompt_version"),
        "requirement_ids": _decode(r.get("requirement_ids")) or [],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    p = argparse.ArgumentParser(prog="python -m src.logging_store", description="Inspect the decision log.")
    p.add_argument("--db", default=os.getenv("DATABASE_URL", DEFAULT_URL))
    sub = p.add_subparsers(dest="command", required=True)
    t = sub.add_parser("trail", help="every decision for one ticket")
    t.add_argument("ticket_id")
    r = sub.add_parser("reconcile", help="check decisions against tickets for one run")
    r.add_argument("--run-id", required=True)
    sub.add_parser("summary", help="counts by stage")
    ex = sub.add_parser("export", help="decisions in the governance record format, one JSON object per line")
    ex.add_argument("--run-id")
    ex.add_argument("--ticket-id")
    args = p.parse_args(argv)
    store = DecisionStore(args.db)
    if args.command == "trail":
        print(json.dumps(store.trail(args.ticket_id), indent=2, ensure_ascii=False))
    elif args.command == "reconcile":
        rep = store.reconcile(args.run_id)
        print(json.dumps(rep, indent=2))
        return 0 if rep["reconciled"] else 2
    elif args.command == "export":
        for rec in store.export(args.run_id, args.ticket_id):
            print(json.dumps(rec, ensure_ascii=False, default=str))
    else:
        print(json.dumps(store.summary(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
