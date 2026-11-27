"""Knowledge gaps: where Cirrus failed to find an answer, grouped by topic, for the technical writer.

A gap is recorded whenever the documentation did not hold what was needed:
  - search found no article above the relevance threshold for a ticket (retrieve: no_relevant_passage);
  - an article was found but did not answer the question (generate: insufficient_information);
  - an in-app help or documentation search returned nothing (stage "search", action "no_result",
    recorded by record_search_gap).

Nothing new is stored for tickets: the gaps are read back from the decision log, which already
records every one of these events, with the ticket's intent from the classify stage turned into a
plain-language topic (src/policy.py: INTENT_TOPICS). Texts are the log's masked summaries.

The improvement loop: the writer publishes an article for the biggest gap, the index is rebuilt
(`python -m src.ingest docs`), and the same questions are answered from then on. No model is
retrained.
"""
from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Optional

from src.policy import topic

GAP_EVENTS = {("retrieve", "no_relevant_passage"), ("generate", "insufficient_information"),
              ("search", "no_result")}


def record_search_gap(store: Any, query: str, where: str, topic_hint: Optional[str] = None,
                      run_id: Optional[str] = None) -> str:
    """Log a documentation or in-app help search that found nothing."""
    import uuid
    return store.record(f"SEARCH-{uuid.uuid4().hex[:8]}", "search", "no_result",
                        f"No article matched a search on {where}.", prediction=topic_hint, run_id=run_id,
                        input_text=query, model_name="retriever")


def knowledge_gaps(store: Any, run_id: Optional[str] = None, since: Optional[str] = None,
                   examples: int = 3) -> list[dict]:
    """[{topic, count, tickets, examples}] most frequent first. One count per ticket (or search)."""
    where, params = "WHERE stage IN ('classify', 'retrieve', 'generate', 'search')", []
    if run_id:
        where += " AND run_id = ?"
        params.append(run_id)
    if since:
        where += " AND created_at >= ?"
        params.append(since)
    rows = store._rows(where, params)
    intent: dict[str, str] = {}
    gap: dict[str, dict] = {}
    for r in rows:
        tid = r["ticket_id"]
        if r["stage"] == "classify" and r.get("prediction"):
            intent[tid] = r["prediction"]
        elif (r["stage"], r["action_taken"]) in GAP_EVENTS:
            details = r.get("details") or {}
            if isinstance(details, str):
                try:
                    details = json.loads(details)
                except ValueError:
                    details = {}
            if r["stage"] == "generate" and details.get("purpose") == "agent_draft":
                continue                                    # a draft for an agent is not a customer question
            gap.setdefault(tid, r)
    by_topic: dict[str, dict] = defaultdict(lambda: {"count": 0, "tickets": [], "examples": []})
    for tid, r in gap.items():
        name = r.get("prediction") if r["stage"] == "search" and r.get("prediction") else topic(intent.get(tid))
        g = by_topic[name]
        g["count"] += 1
        g["tickets"].append(tid)
        text = r.get("input_summary") or ""
        if text and len(g["examples"]) < examples and text not in g["examples"]:
            g["examples"].append(text)
    return sorted(({"topic": k, **v} for k, v in by_topic.items()), key=lambda g: (-g["count"], g["topic"]))
