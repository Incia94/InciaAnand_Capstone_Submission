"""Evaluation harness: run every ticket in a file through the pipeline, unattended, and report.

    python -m evaluation.harness --input data/validation_tickets.json --output evaluation/results/

Build Specification section 04 (the end-to-end run) -> where it is met
------------------------------------------------------------------------------------------------
1 one documented command, input and output paths    this module; --input and --output are required
2 every ticket in the file is processed             one pass over every record, whatever it holds
3 nobody intervenes                                 no prompts; failures inside a ticket become a
                                                    logged escalation; missing set-up is prepared
                                                    automatically (see "Preparation")
4 every ticket ends sent or escalated               SupportPipeline.run always yields an outcome
5 every decision is logged                          a decision log per run, reconciled at the end
6 metrics report without further work              metrics.json and metrics.md, written at the end
7 reasonable time                                   duration and per-ticket latency are reported

What it writes, in <output>/<run_id>/:
    metrics.json     every figure, machine-readable
    metrics.md       the same figures as a readable report, with the Brief's targets beside them
    outcomes.jsonl   one line per ticket: outcome, reason, intent, sources, guardrail result
    decisions.db     the decision log for this run (SQLite; the same schema as storage/decisions.db)

Preparation (skip with --no-prepare): before the run, anything missing is built from the
DEVELOPMENT data (TRAIN_DATA), never from the input file: the documentation index, the retrieval
thresholds, the classifier and the routing threshold. The input file is only ever read to be
processed, so running on the hidden set never tunes anything to it.

Labels: if tickets carry `labels` (as the development and validation sets do), accuracy-type
metrics are computed against them. Tickets without labels are still processed and counted; the
metrics that need labels report how many tickets they rest on.

Exit code: 0 when the run completed and the decision log reconciles; 2 if it completed but the
log does not reconcile; 1 if the input could not be read at all.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from src.query_scope import CONVERSATIONAL       # fixed replies, no ticket (src/scope.py)

log = logging.getLogger("harness")

DEFAULT_TRAIN = "./data/development_tickets.json"
DEFAULT_TRUTH = "./data/ground_truth_responses.json"
PII_KINDS = {"email", "ip_address", "customer_id", "api_key", "aws_key", "github_token", "jwt",
             "secret_assignment", "long_token", "phone_number", "payment_card"}
SEGMENTS = ("customer_tier", "customer_region", "language_fluency", "channel", "ticket_length")
# Ticket length bands from the quartiles of the 580 labelled tickets (24 and 32 words).
SHORT_WORDS, LONG_WORDS = 25, 33
FAIRNESS_GAP = 0.05          # Brief: under five percentage points between groups
FAIRNESS_MIN_N = 10          # smaller segments are reported but not judged

# The Brief's targets (section 07), so the report can state met / not met beside each figure.
TARGETS = {
    "first_contact_resolution": (">=", 0.60, "Business: first contact resolution 60% or better"),
    "escalation_rate": ("<=", 0.30, "Business: escalation rate 30% or lower"),
    "response_time_mean_s_sent": ("<=", 300.0, "Business: time to first reply under 5 minutes"),
    "intent_precision_macro": (">=", 0.85, "Technical: intent classification precision 85% or better"),
    "hallucination_rate_sent": ("<=", 0.05, "Technical: hallucination rate 5% or lower"),
    "citation_accuracy": (">=", 0.95, "Technical: citation accuracy 95% or better"),
    "latency_p95_s": ("<", 3.0, "Technical: response latency, 95th percentile, under 3 seconds"),
    "private_data_in_sent_replies": ("==", 0, "Governance: zero private data in outbound responses"),
    "fairness_max_gap": ("<", FAIRNESS_GAP, "Governance: under 5 points between customer groups"),
    "calibration_gap": ("<=", 0.05, "Governance: stated confidence within 5 points of accuracy"),
    "decision_logging_complete": ("==", True, "Governance: every automated decision logged"),
}


# ------------------------------------------------------------------ preparation (development data only)

def prepare(train_path: Path, say: Callable[[str], None]) -> list[str]:
    """Build whatever is missing, from the development set only. Returns what was done."""
    from src import ingest, retrieve, route
    from src.classify import load_or_train
    from src.ingest import load_tickets
    done = []
    chroma = Path(os.getenv("CHROMA_PATH", "./storage/chroma"))
    needs_index = os.getenv("RETRIEVAL_BACKEND", "hybrid") in ("dense", "hybrid")
    if needs_index and not (chroma / "ingest_report.json").is_file():
        say(f"preparing: building the documentation index in {chroma}")
        try:
            ingest.ingest_docs(Path(os.getenv("DOCS_PATH", "./data/documentation.json")), chroma,
                               os.getenv("CHROMA_COLLECTION", ingest.DEFAULT_COLLECTION),
                               ingest.SentenceTransformerEmbedder(os.getenv("EMBEDDING_MODEL", ingest.DEFAULT_MODEL)),
                               strategy=os.getenv("CHUNK_STRATEGY", ingest.DEFAULT_STRATEGY))
            done.append("documentation index built")
        except Exception as exc:   # the run continues on lexical search, and says so
            say(f"preparing: index could not be built ({exc}); retrieval will use lexical search")
            done.append(f"documentation index NOT built: {exc}")
    load_or_train(train_path=train_path)        # trains and saves only if no usable model exists
    retrieval_cfg = Path(os.getenv("RETRIEVAL_CONFIG", retrieve.DEFAULT_CONFIG))
    routing_cfg = Path(os.getenv("ROUTING_CONFIG", route.DEFAULT_CONFIG))
    if not retrieval_cfg.is_file() or not routing_cfg.is_file():
        train = load_tickets(train_path)
        retriever = retrieve.build_retriever()
        if not retrieval_cfg.is_file():
            say("preparing: calibrating retrieval thresholds on the development set")
            retrieve.calibrate(retriever, retrieve._labelled(train), retrieval_cfg)
            done.append("retrieval thresholds calibrated")
        if not routing_cfg.is_file():
            say("preparing: calibrating the routing threshold on the development set (about a minute)")
            route.calibrate(train, retriever, config_path=routing_cfg)
            done.append("routing threshold calibrated")
    return done


# ------------------------------------------------------------------ one row per ticket
def ticket_row(state: dict, corpus: Any) -> dict:
    """Everything the metrics need about one ticket, taken from the pipeline's final state."""
    t, o = state["ticket"], state["outcome"]
    c, r, a, out = state.get("classification"), state.get("retrieval"), state.get("answer"), state.get("outbound")
    g = out.guardrail if out is not None else None
    labels = t.eval_labels or {}
    kinds = [f.kind for ch in (g.checks if g else []) for f in ch.findings if f.severity == "block"]
    cited = [cid for cl in (a.claims if a else []) for cid in cl.citations]
    return {
        # A fixed reply (small talk, a clarifying question...) is neither an answer nor an escalation.
        "ticket_id": t.ticket_id, "outcome": "conversation" if o.decided_by in CONVERSATIONAL else o.outcome,
        "decided_by": o.decided_by, "reason": o.reason,
        "channel": t.channel.value, "customer_tier": t.customer_tier.value,
        "customer_region": t.customer_region.value, "language_fluency": t.language_fluency.value,
        "processable": t.processable,
        "intent": c.intent if c else None, "intent_confidence": c.intent_confidence if c else None,
        "urgency": c.urgency if c else None, "familiarity": c.familiarity if c else None,
        "classifier_fallback": bool(c and c.is_fallback),
        "retrieval_status": r.status if r else None, "retrieved_docs": r.doc_ids if r else [],
        "retrieval_backend": r.backend if r else None, "retrieval_degraded": bool(r and r.degraded),
        "drafted": a is not None, "answer_status": a.status if a else None,
        "generator": a.generator if a else None, "generator_degraded": bool(a and a.degraded),
        "claims": len(a.claims) if a else 0, "cited_chunks": cited,
        "cited_docs": sorted({cid.split("::")[0] for cid in cited}),
        "citations_resolve": all(corpus.resolve(cid) is not None for cid in cited) if cited else None,
        "guardrail_checked": g is not None, "guardrail_released": g.released if g else None,
        "guardrail_blocked_by": list(g.blocked_by) if g else [], "guardrail_finding_kinds": kinds,
        "claims_checked": g.claims_checked if g else 0, "claims_unsupported": g.claims_unsupported if g else 0,
        "prompt_version": a.prompt_version if a else None,
        "latency_ms": o.latency_ms, "degraded": list(o.degraded), "error": state.get("error"),
        "labels": labels or None, "history": t.eval_history or None,
        "reply": o.reply, "handoff_summary": (o.handoff or {}).get("summary"),
        "ticket_length": ("short (under 25 words)" if t.word_count < SHORT_WORDS
                          else "long (over 32 words)" if t.word_count >= LONG_WORDS else "medium (25 to 32 words)"),
        "escalation_category": (o.handoff or {}).get("escalation_category"),
        "suggested_team": (o.handoff or {}).get("suggested_team"),
        "draft_status": (o.draft_reply or {}).get("status"),
        "draft_guardrail_passed": (o.draft_reply or {}).get("guardrail_passed"),
        "draft_sources": [s_["doc_id"] for s_ in (o.draft_reply or {}).get("sources", [])],
        "language": t.language, "translated": t.translated, "reply_language": a.reply_language if a else None,
        "intent_hint_used": bool(r and r.hinted),
    }


# ------------------------------------------------------------------ metrics
def _rate(num: float, den: float) -> Optional[float]:
    return round(num / den, 4) if den else None


def _pct(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return round(s[lo] + (s[hi] - s[lo]) * (k - lo), 4)


def _correct_answer(row: dict) -> Optional[bool]:
    """A sent reply that a senior agent would accept: right intent, an expected document cited,
    and a ticket the labels say may be answered automatically. None without labels."""
    lab = row["labels"]
    if not lab:
        return None
    return (row["outcome"] == "sent" and row["intent"] == lab.get("intent")
            and lab.get("expected_route") == "auto_respond"
            and bool(set(row["cited_docs"]) & set(lab.get("expected_doc_ids") or [])))


def compute_metrics(rows: list[dict], reconcile: dict) -> dict:
    from sklearn.metrics import classification_report, f1_score

    from src.classify import expected_calibration_error, reliability_table
    n = len(rows)
    sent = [r for r in rows if r["outcome"] == "sent"]
    esc = [r for r in rows if r["outcome"] == "escalated"]
    lab = [r for r in rows if r["labels"]]
    lat_all = [r["latency_ms"] / 1000 for r in rows]
    lat_sent = [r["latency_ms"] / 1000 for r in sent]

    # Why tickets were escalated. Against the labels, an escalation is "genuine" when the label
    # says a person is needed, and "avoidable" (a findability problem) when the label says it could
    # have been answered automatically from the documentation.
    esc_lab = [r for r in esc if r["labels"]]
    avoidable = [r for r in esc_lab if r["labels"].get("expected_route") == "auto_respond"
                 and r["labels"].get("expected_doc_ids")]
    escalations = {
        "by_category": dict(Counter(r["escalation_category"] or r["decided_by"] for r in esc).most_common()),
        "by_team": dict(Counter(r["suggested_team"] or "unknown" for r in esc).most_common()),
        "labelled_escalations": len(esc_lab),
        "genuine": sum(r["labels"].get("expected_route") == "escalate" for r in esc_lab),
        "avoidable_findability": len(avoidable),
        "avoidable_share_of_escalations": _rate(len(avoidable), len(esc_lab)),
        "avoidable_by_category": dict(Counter(r["escalation_category"] or r["decided_by"]
                                              for r in avoidable).most_common()),
        "note": "genuine = labelled escalate; avoidable = labelled auto_respond with a documented answer.",
        "intent_hint_answers": sum(r["intent_hint_used"] and r["outcome"] == "sent" for r in rows),
        "drafts_for_agents": sum(r["draft_status"] == "drafted" for r in esc),
        "drafts_passing_guardrails": sum(r["draft_status"] == "drafted" and bool(r["draft_guardrail_passed"])
                                         for r in esc),
        "escalations_without_draft": sum(r["draft_status"] != "drafted" for r in esc),
    }
    languages = {"by_language": dict(Counter(r["language"] for r in rows)),
                 "translated": sum(r["translated"] for r in rows),
                 "replies_in_customer_language": sum(r["outcome"] == "sent" and r["reply_language"] not in
                                                     (None, "en") for r in rows)}

    volume = {
        "tickets_processed": n, "answered_automatically": len(sent), "escalated": len(esc),
        "blocked_by_guardrails": sum(r["decided_by"] == "guardrail" for r in rows),
        "escalated_by": dict(Counter(r["decided_by"] for r in esc).most_common()),
        "by_channel": dict(Counter(r["channel"] for r in rows)),
        "errors": sum(bool(r["error"]) for r in rows),
        "unprocessable_input": sum(not r["processable"] for r in rows),
        "conversational_replies": sum(r["decided_by"] in CONVERSATIONAL for r in rows),
        "tickets_with_labels": len(lab),
    }

    hist = [r["history"] for r in rows if r["history"]]
    baseline = {
        "tickets": len(hist),
        "first_contact_resolution": _rate(sum(bool(h.get("first_contact_resolution")) for h in hist), len(hist)),
        "escalation_rate": _rate(sum(bool(h.get("escalated")) for h in hist), len(hist)),
        "resolution_minutes_mean": round(statistics.mean(h["resolution_time_minutes"] for h in hist), 1)
        if hist else None,
        "resolution_minutes_median": statistics.median(h["resolution_time_minutes"] for h in hist) if hist else None,
        "csat_mean": round(statistics.mean(h["csat_rating"] for h in hist), 2) if hist else None,
    } if hist else None
    correct = [_correct_answer(r) for r in lab]
    business = {
        "first_contact_resolution": _rate(len(sent), n),
        "first_contact_resolution_note": "Brief definition: tickets closed without escalation / total tickets.",
        "correct_automatic_resolution": _rate(sum(bool(x) for x in correct), len(lab)),
        "correct_automatic_resolution_note": "Sent, right intent, an expected document cited, and labelled "
                                             "auto_respond. Labelled tickets only.",
        "wrong_automatic_answers": sum(r["outcome"] == "sent" and _correct_answer(r) is False for r in lab),
        "escalation_rate": _rate(len(esc), n),
        "response_time_mean_s_sent": round(statistics.mean(lat_sent), 3) if lat_sent else None,
        "response_time_median_s_sent": round(statistics.median(lat_sent), 3) if lat_sent else None,
        "response_time_mean_s_all": round(statistics.mean(lat_all), 3) if lat_all else None,
        "response_time_median_s_all": round(statistics.median(lat_all), 3) if lat_all else None,
        "response_time_note": "System processing time from arrival to sent reply or escalation. An escalated "
                              "ticket's time to a human reply is outside the system and not measured.",
        "baseline_from_history": baseline,
        "repeat_contacts_note": "Not measurable in a single run: needs follow-up tickets after a reply.",
    }

    technical: dict[str, Any] = {"labelled_tickets": len(lab)}
    if lab:
        y = [r["labels"].get("intent", "unknown") for r in lab]
        p = [r["intent"] or "unclassified" for r in lab]
        rep = classification_report(y, p, output_dict=True, zero_division=0)
        per_class = {k: {"precision": round(v["precision"], 3), "recall": round(v["recall"], 3),
                         "support": int(v["support"])} for k, v in rep.items()
                     if k not in ("accuracy", "macro avg", "weighted avg")}
        ok = [a == b for a, b in zip(y, p)]
        conf = [r["intent_confidence"] or 0.0 for r in lab]
        uy = [r["labels"].get("urgency") for r in lab]
        up = [r["urgency"] for r in lab]
        answerable = [r for r in lab if r["labels"].get("expected_doc_ids") and r["retrieval_status"] is not None]
        sent_lab = [r for r in sent if r["labels"]]
        cites = [(d, r["labels"].get("expected_doc_ids") or []) for r in sent_lab for d in r["cited_docs"]]
        must = [r for r in lab if r["labels"].get("must_not_auto_respond")]
        technical.update({
            "intent_accuracy": round(sum(ok) / len(ok), 4),
            "intent_precision_macro": round(rep["macro avg"]["precision"], 4),
            "intent_recall_macro": round(rep["macro avg"]["recall"], 4),
            "intent_f1_macro": round(f1_score(y, p, average="macro", zero_division=0), 4),
            "intent_per_class": per_class,
            "urgency_accuracy": _rate(sum(a == b for a, b in zip(uy, up)), len(uy)),
            "calibration_ece": round(expected_calibration_error(conf, ok), 4),
            "calibration_gap": round(abs(statistics.mean(conf) - sum(ok) / len(ok)), 4),
            "calibration_reliability": reliability_table(conf, ok),
            "routing_agreement_with_labels": _rate(sum((r["outcome"] == "sent") == (r["labels"].get(
                "expected_route") == "auto_respond") for r in lab), len(lab)),
            "routing_confusion": dict(Counter(f"expected {r['labels'].get('expected_route')} -> {r['outcome']}"
                                              for r in lab)),
            "retrieval_hit_rate": _rate(sum(bool(set(r["retrieved_docs"]) & set(r["labels"]["expected_doc_ids"]))
                                            for r in answerable), len(answerable)),
            "retrieval_hit_at_1": _rate(sum(bool(r["retrieved_docs"]) and r["retrieved_docs"][0] in
                                            r["labels"]["expected_doc_ids"] for r in answerable), len(answerable)),
            "retrieval_answerable_tickets": len(answerable),
            "citation_accuracy": _rate(sum(d in exp for d, exp in cites), len(cites)),
            "citations_checked": len(cites),
            "must_not_auto_respond_tickets": len(must),
        })
    drafted = [r for r in rows if r["guardrail_checked"]]
    technical.update({
        "citations_resolve_to_corpus": _rate(sum(bool(r["citations_resolve"]) for r in sent if r["cited_chunks"]),
                                             sum(bool(r["cited_chunks"]) for r in sent)),
        "hallucination_rate_drafts": _rate(sum(r["claims_unsupported"] for r in drafted),
                                           sum(r["claims_checked"] for r in drafted)),
        "hallucination_rate_sent": _rate(sum(r["claims_unsupported"] for r in sent),
                                         sum(r["claims_checked"] for r in sent)),
        "hallucination_note": "Claims the guardrails found unsupported by their cited passages. Drafts with "
                              "unsupported claims are blocked, so the sent rate measures leaks past the check.",
        "latency_p50_s": _pct(lat_all, 0.5), "latency_p95_s": _pct(lat_all, 0.95), "latency_max_s": _pct(lat_all, 1),
        "availability": _rate(n - sum(bool(r["error"]) for r in rows), n),
        "degraded_tickets": dict(Counter(d.split(":")[0] for r in rows for d in r["degraded"])),
        "generators_used": dict(Counter(r["generator"] for r in drafted)),
        "retrieval_backends_used": dict(Counter(r["retrieval_backend"] for r in rows if r["retrieval_backend"])),
    })

    activations = Counter(b for r in drafted for b in r["guardrail_blocked_by"])
    findings = Counter(k for r in drafted for k in r["guardrail_finding_kinds"])
    governance = {
        "decisions_logged": reconcile.get("decisions"),
        "decisions_by_stage": reconcile.get("by_stage"),
        "decision_logging_complete": bool(reconcile.get("reconciled")),
        "tickets_missing_outcome": reconcile.get("missing_outcome"),
        "replies_checked_by_guardrails": len(drafted),
        "guardrail_activations_by_type": dict(activations),
        "guardrail_findings_by_kind": dict(findings),
        "private_data_detections": sum(v for k, v in findings.items() if k in PII_KINDS),
        "private_data_in_sent_replies": sum(1 for r in sent for k in r["guardrail_finding_kinds"] if k in PII_KINDS),
        "must_not_auto_respond_answered_automatically": sum(
            1 for r in lab if r["labels"].get("must_not_auto_respond") and r["outcome"] == "sent"),
        "calibration_gap": technical.get("calibration_gap"),
        "fairness": fairness(rows),
    }
    gaps = [m["max_gap"] for d in governance["fairness"].values() for m in d["gaps"].values()
            if m["max_gap"] is not None]
    governance["fairness_max_gap"] = round(max(gaps), 4) if gaps else None

    flat = {**business, **technical, **governance}
    targets = []
    for key, (op, goal, label) in TARGETS.items():
        value = flat.get(key)
        met = None if value is None else {">=": value >= goal, "<=": value <= goal, "<": value < goal,
                                          "==": value == goal}[op]
        targets.append({"target": label, "metric": key, "value": value, "goal": f"{op} {goal}", "met": met})
    return {"volume": volume, "business": business, "technical": technical, "governance": governance,
            "escalations": escalations, "languages": languages, "targets": targets}


def fairness(rows: list[dict]) -> dict:
    """Per customer group: automation, correct automatic answers and escalation, with sample sizes."""
    out = {}
    for dim in SEGMENTS:
        groups: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            groups[r[dim]].append(r)
        table = {}
        for g, rs in sorted(groups.items()):
            lab = [r for r in rs if r["labels"]]
            judged = [r["judge"]["rating"] for r in rs if (r.get("judge") or {}).get("judged")]
            table[g] = {"tickets": len(rs), "auto_rate": _rate(sum(r["outcome"] == "sent" for r in rs), len(rs)),
                        "correct_automatic_resolution": _rate(sum(bool(_correct_answer(r)) for r in lab), len(lab)),
                        "escalation_rate": _rate(sum(r["outcome"] == "escalated" for r in rs), len(rs)),
                        "judge_rating": round(statistics.mean(judged), 2) if judged else None}
        gaps = {}
        for metric in ("auto_rate", "correct_automatic_resolution", "escalation_rate"):
            vals = [v[metric] for v in table.values() if v[metric] is not None and v["tickets"] >= FAIRNESS_MIN_N]
            gap = round(max(vals) - min(vals), 4) if len(vals) > 1 else None
            gaps[metric] = {"max_gap": gap, "over_threshold": None if gap is None else gap >= FAIRNESS_GAP}
        out[dim] = {"groups": table, "gaps": gaps,
                    "note": f"Groups under {FAIRNESS_MIN_N} tickets are shown but not used for the gap."}
    return out


# ------------------------------------------------------------------ the readable report
# Which figures are proportions (shown as %) and which are times in seconds; everything else as is.
_RATES = ("rate", "resolution", "accuracy", "precision", "recall", "f1", "ece", "gap", "hit", "availability",
          "resolve")


def _fmt(v: Any, key: str = "") -> str:
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if key.endswith("_s") or "_s_" in key:
        return f"{v:.3f} s"
    if isinstance(v, float) and any(word in key for word in _RATES):
        return f"{v:.1%}"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def render_markdown(report: dict) -> str:
    m, run = report["metrics"], report["run"]
    v, b, t, g = m["volume"], m["business"], m["technical"], m["governance"]
    lines = [f"# Evaluation report: {run['run_id']}", "",
             f"Input `{run['input']}` ({run['tickets_in_file']} tickets, sha256 `{run['input_sha256'][:16]}`), "
             f"run {run['started_at']} to {run['finished_at']} ({run['duration_s']} s). "
             f"Decision log reconciled: **{_fmt(g['decision_logging_complete'])}**.", "",
             "## Against the Brief's targets", "", "| Target | Value | Goal | Met |", "|---|---|---|---|"]
    for row in m["targets"]:
        goal = row["goal"].split(" ", 1)
        lines.append(f"| {row['target']} | {_fmt(row['value'], row['metric'])} | "
                     f"{goal[0]} {_fmt(float(goal[1]) if goal[1] not in ('True', '0') else goal[1], row['metric'])} | "
                     f"{_fmt(row['met'])} |")
    lines += ["", "## Volume", "", "| | |", "|---|---|"]
    for k in ("tickets_processed", "answered_automatically", "conversational_replies", "escalated",
              "blocked_by_guardrails", "errors", "unprocessable_input", "tickets_with_labels"):
        lines.append(f"| {k.replace('_', ' ')} | {v[k]} |")
    lines.append(f"| escalated by | {', '.join(f'{k} {n}' for k, n in v['escalated_by'].items()) or 'none'} |")
    e = m.get("escalations", {})
    if e:
        lines += ["", "## Escalations: genuine or avoidable", "", "| | |", "|---|---|",
                  f"| escalated tickets with labels | {e['labelled_escalations']} |",
                  f"| genuine (labelled escalate) | {e['genuine']} |",
                  f"| avoidable: findability (labelled auto_respond, documented) | {e['avoidable_findability']} |",
                  f"| avoidable share of escalations | {_fmt(e['avoidable_share_of_escalations'], 'rate')} |",
                  f"| by category | {e['by_category'] or 'none'} |",
                  f"| avoidable, by category | {e['avoidable_by_category'] or 'none'} |",
                  f"| by suggested team | {e['by_team'] or 'none'} |",
                  f"| answered through intent-guided search | {e['intent_hint_answers']} |",
                  f"| escalations with a cited draft for the agent | {e['drafts_for_agents']} "
                  f"({e['drafts_passing_guardrails']} passed every guardrail) |",
                  "", f"- {e['note']}"]
        lg = m["languages"]
        lines += [f"- Languages: {lg['by_language']}; translated {lg['translated']}; replies in the customer's "
                  f"language {lg['replies_in_customer_language']}."]
    lines += ["", "## Business outcomes", "", "| Measure | This run | Baseline (history) |", "|---|---|---|"]
    base = b["baseline_from_history"] or {}
    lines += [f"| First contact resolution | {_fmt(b['first_contact_resolution'], 'rate')} | "
              f"{_fmt(base.get('first_contact_resolution'), 'rate')} |",
              f"| Correct automatic resolution | {_fmt(b['correct_automatic_resolution'], 'rate')} | n/a |",
              f"| Escalation rate | {_fmt(b['escalation_rate'], 'rate')} | "
              f"{_fmt(base.get('escalation_rate'), 'rate')} |",
              f"| Response time, sent replies (mean / median) | {_fmt(b['response_time_mean_s_sent'], 'x_s')} / "
              f"{_fmt(b['response_time_median_s_sent'], 'x_s')} | {base.get('resolution_minutes_mean', 'n/a')} min / "
              f"{base.get('resolution_minutes_median', 'n/a')} min (time to resolution) |",
              f"| Customer satisfaction | {_fmt(b.get('satisfaction_proxy_mean_rating'))} (model-judged, "
              f"with --judge) | {_fmt(base.get('csat_mean'))} |",
              "", f"- {b['first_contact_resolution_note']}", f"- {b['correct_automatic_resolution_note']}",
              f"- {b['response_time_note']}", f"- {b['repeat_contacts_note']}"]
    lines += ["", "## Technical performance", "", "| Measure | Value |", "|---|---|"]
    for k in ("intent_accuracy", "intent_precision_macro", "intent_recall_macro", "intent_f1_macro",
              "urgency_accuracy", "calibration_ece", "calibration_gap", "routing_agreement_with_labels",
              "retrieval_hit_rate", "retrieval_hit_at_1", "citation_accuracy", "citations_resolve_to_corpus",
              "hallucination_rate_drafts", "hallucination_rate_sent", "latency_p50_s", "latency_p95_s",
              "availability"):
        if k in t:
            lines.append(f"| {k.replace('_', ' ')} | {_fmt(t[k], k)} |")
    if t.get("intent_per_class"):
        lines += ["", "Per intent:", "", "| Intent | Precision | Recall | Support |", "|---|---|---|---|"]
        for k, row in sorted(t["intent_per_class"].items()):
            lines.append(f"| {k} | {_fmt(row['precision'], 'precision')} | {_fmt(row['recall'], 'recall')} | "
                         f"{row['support']} |")
    lines += ["", f"Degraded tickets: {t['degraded_tickets'] or 'none'}. Generators: {t['generators_used'] or 'none'}. "
              f"Retrieval: {t['retrieval_backends_used']}.", "", "## Governance", "", "| Measure | Value |",
              "|---|---|"]
    for k in ("decisions_logged", "decision_logging_complete", "replies_checked_by_guardrails",
              "private_data_detections", "private_data_in_sent_replies",
              "must_not_auto_respond_answered_automatically", "fairness_max_gap"):
        lines.append(f"| {k.replace('_', ' ')} | {_fmt(g[k], k)} |")
    lines.append(f"| guardrail activations by type | {g['guardrail_activations_by_type'] or 'none'} |")
    lines += ["", "### Fairness audit (Governance Framework format)", "",
              "Resolution rate = answered automatically. Quality score = correct automatic resolution (sent, right "
              "intent, an expected article cited, labelled auto_respond); with --judge, the mean model-judged "
              "rating is shown too. Variation from best = best group's quality score minus this group's. "
              f"Groups under {FAIRNESS_MIN_N} tickets are listed but not used for the gap.", "",
              "| Segment | Tickets in segment | Resolution rate | Quality score | Variation from best | Explanation |",
              "|---|---|---|---|---|---|"]
    fw = [("customer_tier", "enterprise", "Enterprise customers"), ("customer_tier", "business", "Business customers"),
          ("customer_tier", "standard", "Standard customers"),
          ("language_fluency", "fluent", "Tickets in fluent English"),
          ("language_fluency", "non_fluent", "Tickets in non-fluent English"),
          ("ticket_length", "short (under 25 words)", "Short tickets (under 25 words)"),
          ("ticket_length", "long (over 32 words)", "Long tickets (over 32 words)")]
    for dim, grp, label in fw:
        d_ = g["fairness"].get(dim, {}).get("groups", {})
        row = d_.get(grp)
        if not row:
            lines.append(f"| {label} | 0 | n/a | n/a | n/a | |")
            continue
        best = max((v["correct_automatic_resolution"] for v in d_.values()
                    if v["correct_automatic_resolution"] is not None and v["tickets"] >= FAIRNESS_MIN_N), default=None)
        q = row["correct_automatic_resolution"]
        var = None if (best is None or q is None) else round(best - q, 4)
        judge = f" (judged {row['judge_rating']})" if row.get("judge_rating") is not None else ""
        lines.append(f"| {label} | {row['tickets']} | {_fmt(row['auto_rate'], 'rate')} | {_fmt(q, 'rate')}{judge} | "
                     f"{_fmt(var, 'gap')} | |")
    lines += ["", "The Explanation column is for your interpretation in the report.", "",
              "### Fairness by customer group", ""]
    for dim, d in g["fairness"].items():
        lines += [f"**{dim}**", "", "| Group | Tickets | Auto rate | Correct auto | Escalation |",
                  "|---|---|---|---|---|"]
        for grp, row in d["groups"].items():
            lines.append(f"| {grp} | {row['tickets']} | {_fmt(row['auto_rate'], 'rate')} | "
                         f"{_fmt(row['correct_automatic_resolution'], 'rate')} | "
                         f"{_fmt(row['escalation_rate'], 'rate')} |")
        gap_txt = "; ".join(f"{k.replace('_', ' ')} gap {_fmt(x['max_gap'], 'gap')}"
                            + (" (over 5 points)" if x["over_threshold"] else "") for k, x in d["gaps"].items())
        lines += ["", gap_txt, ""]
    lines += ["## Run details", "", "```", json.dumps(run["configuration"], indent=2), "```", "",
              "The figures above are measurements. What they mean for CloudServe belongs in the report."]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ the run
def run_harness(input_path: Path, output_dir: Path, *, pipeline: Any = None, prepare_first: bool = True,
                db_url: Optional[str] = None, limit: Optional[int] = None, judge: bool = False,
                truth_path: Optional[Path] = None, say: Callable[[str], None] = print) -> dict:
    from src.api import SupportPipeline
    from src.ingest import read_ticket_file
    from src.logging_store import DecisionStore, new_run_id

    input_path, output_dir = Path(input_path), Path(output_dir)
    raw = input_path.read_bytes()
    records = read_ticket_file(input_path)
    if limit:
        records = records[:limit]
    run_id = new_run_id("eval")
    out = output_dir / run_id
    out.mkdir(parents=True, exist_ok=True)
    started_at, t0 = datetime.now(timezone.utc), time.perf_counter()
    say(f"run {run_id}: {len(records)} tickets from {input_path} -> {out}")

    prepared = []
    if pipeline is None:
        if prepare_first:
            prepared = prepare(Path(os.getenv("TRAIN_DATA", DEFAULT_TRAIN)), say)
        pipeline = SupportPipeline.from_env(store=DecisionStore(db_url or f"sqlite:///{out / 'decisions.db'}"))
    store = pipeline.store
    corpus = pipeline.retriever.corpus

    rows, ids = [], []
    interrupted = False
    with (out / "outcomes.jsonl").open("w", encoding="utf-8") as fh:
        try:
            for i, rec in enumerate(records, 1):
                state = pipeline.run(rec, run_id=run_id)
                row = ticket_row(state, corpus)
                rows.append(row)
                ids.append(row["ticket_id"])
                fh.write(json.dumps({k: v for k, v in row.items() if k not in ("labels", "history")},
                                    ensure_ascii=False, default=str) + "\n")
                if i % 20 == 0 or i == len(records):
                    say(f"  {i}/{len(records)} processed ({time.perf_counter() - t0:.0f} s)")
        except KeyboardInterrupt:  # the report is still written for what was processed
            interrupted = True
            say("interrupted: writing the report for the tickets processed so far")

    if judge:
        judge_rows(rows, records, truth_path, say, corpus=corpus)
    reconcile = store.reconcile(run_id, ids)
    metrics = compute_metrics(rows, reconcile)
    if judge:
        judged = [r["judge"] for r in rows if r.get("judge", {}).get("judged")]
        metrics["business"]["satisfaction_proxy_mean_rating"] = (
            round(statistics.mean(j["rating"] for j in judged), 2) if judged else None)
        metrics["technical"]["judge_grounded_rate"] = _rate(sum(bool(j["grounded"]) for j in judged), len(judged))
        metrics["technical"]["judge_replies_reviewed"] = len(judged)
    report = {"run": {
        "run_id": run_id, "input": str(input_path), "input_sha256": hashlib.sha256(raw).hexdigest(),
        "tickets_in_file": len(records), "tickets_processed": len(rows), "interrupted": interrupted,
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "duration_s": round(time.perf_counter() - t0, 1), "output_dir": str(out),
        "decision_log": str(store.path), "prepared": prepared,
        "configuration": configuration(pipeline)}, "reconcile": reconcile, "metrics": metrics}
    (out / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    (out / "metrics.md").write_text(render_markdown(report))
    say(f"done: {len(rows)} processed, {metrics['volume']['answered_automatically']} sent, "
        f"{metrics['volume']['escalated']} escalated; decision log reconciled: {reconcile['reconciled']}")
    say(f"report: {out / 'metrics.md'}")
    return report


def configuration(pipeline: Any) -> dict:
    import importlib.metadata as md
    versions = {}
    for pkg in ("langgraph", "langchain-core", "chromadb", "scikit-learn", "sentence-transformers", "pydantic"):
        try:
            versions[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            versions[pkg] = None
    h = pipeline.health()
    return {"python": platform.python_version(), "packages": versions,
            "classifier": h["classifier"], "retrieval": {**h["retrieval"], "thresholds": pipeline.retriever.thresholds},
            "routing": h["routing"], "generator": h["generator"], "kill_switch": h["kill_switch"]}


def judge_rows(rows: list[dict], records: list, truth_path: Optional[Path], say: Callable[[str], None],
               corpus: Any = None, judge: Any = None) -> None:
    """Optional: a model reviews each sent reply (evaluation/judge.py). Needs an API key. The judge is
    given the passages the reply actually cited, so its 'grounded' verdict checks the reply against
    the same text the guardrails did."""
    from evaluation.judge import build_judge
    judge = judge or build_judge()
    if judge.client is None:
        say(f"judge skipped: {judge.unavailable_reason}")
        return
    truth = {}
    path = Path(truth_path or DEFAULT_TRUTH)
    if path.is_file():
        truth = {g["ticket_id"]: g for g in json.loads(path.read_text())}
    by_id = {r["ticket_id"]: r for r in rows}
    sent = [r for r in rows if r["outcome"] == "sent"]
    say(f"judging {len(sent)} sent replies")
    for rec in records:
        if not isinstance(rec, dict) or rec.get("ticket_id") not in by_id:
            continue
        row = by_id[rec["ticket_id"]]
        if row["outcome"] != "sent":
            continue
        cited = [corpus.resolve(cid) for cid in row["cited_chunks"]] if corpus is not None else []
        passages = [{"text": c.text} for c in cited if c is not None]
        row["judge"] = judge.judge(rec, row["reply"] or "", passages, truth.get(row["ticket_id"])).model_dump()


# ------------------------------------------------------------------ command line
def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    p = argparse.ArgumentParser(prog="python -m evaluation.harness",
                                description="Run every ticket in a file through the pipeline and write a "
                                            "metrics report.")
    p.add_argument("--input", required=True, type=Path, help="ticket file (JSON list or JSON Lines), any name")
    p.add_argument("--output", required=True, type=Path, help="folder for results; a dated run folder is made inside")
    p.add_argument("--no-prepare", action="store_true", help="do not build missing index/thresholds before the run")
    p.add_argument("--db", help="decision log URL (default: decisions.db inside the run folder)")
    p.add_argument("--limit", type=int, help="process only the first N tickets (for a quick check)")
    p.add_argument("--judge", action="store_true", help="also have a model review each sent reply (needs an API key)")
    p.add_argument("--truth", type=Path, default=Path(DEFAULT_TRUTH), help="reference answers for the judge")
    args = p.parse_args(argv)
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "WARNING"), format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpx2", "httpcore", "huggingface_hub", "sentence_transformers", "chromadb", "urllib3",
                  "torch", "openai"):   # library chatter; the harness prints its own progress
        logging.getLogger(noisy).setLevel(logging.WARNING)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if not args.input.is_file():
        print(f"input file not found: {args.input}", file=sys.stderr)
        return 1
    try:
        report = run_harness(args.input, args.output, prepare_first=not args.no_prepare, db_url=args.db,
                             limit=args.limit, judge=args.judge, truth_path=args.truth)
    except (OSError, ValueError) as exc:
        print(f"could not read {args.input}: {exc}", file=sys.stderr)
        return 1
    return 0 if report["reconcile"]["reconciled"] else 2


if __name__ == "__main__":
    sys.exit(main())
