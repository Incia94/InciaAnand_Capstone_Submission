"""Evaluation judge: a model reviews a drafted reply against its sources and a senior agent's answer.

Uses prompts/evaluation/judge_answer.v1.txt (JUDGE_PROMPT below, read with load_prompt). For each
reply it returns whether every statement is grounded in the source passages, which of the
reference answer's must_mention points are covered or missing, any must_not_claim statements made,
whether it answers the question, and a 1-5 rating "as the customer would likely rate it". The
Brief measures customer satisfaction as the mean rating across reviewed responses; this rating is
the stand-in for that, and should be reported as a model's judgement, not a customer's.

The judge never decides what is sent to a customer. It runs only in evaluation, offline, and
needs a model: without one it returns judged=False with the reason, and the harness reports the
deterministic metrics alone.

    from evaluation.judge import build_judge
    verdict = build_judge().judge(ticket, answer_text, passages, reference)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Sequence

from pydantic import BaseModel, Field, ValidationError, field_validator

from src.generate import DEFAULT_BASE_URL, DEFAULT_CACHE, DEFAULT_MODEL, LangChainChatClient, LLMClient, LLMError
from src.generate import REPAIR_PROMPT, _extract_json, load_prompt, sanitise_ticket_text
from src.ingest import NormalisedTicket, normalise_ticket

MAX_ATTEMPTS = 2
JUDGE_PROMPT = ("judge_answer", 1, "evaluation")   # prompts/evaluation/judge_answer.v1.txt


class JudgeVerdict(BaseModel):
    ticket_id: str
    judged: bool = False
    grounded: Optional[bool] = None
    unsupported_claims: list[str] = []
    must_mention_covered: list[str] = []
    must_mention_missing: list[str] = []
    forbidden_claims_made: list[str] = []
    answers_question: Optional[bool] = None
    rating: Optional[int] = Field(None, ge=1, le=5)
    rationale: str = ""
    prompt_version: str = ""
    judge_model: str = ""
    error: str = ""


class _Reply(BaseModel):
    grounded: bool
    unsupported_claims: list[str] = []
    must_mention_covered: list[str] = []
    must_mention_missing: list[str] = []
    forbidden_claims_made: list[str] = []
    answers_question: bool
    rating: int
    rationale: str = ""

    @field_validator("rating")
    @classmethod
    def _rating(cls, v: int) -> int:
        if not 1 <= int(v) <= 5:
            raise ValueError("rating must be between 1 and 5")
        return int(v)


class Judge:
    def __init__(self, client: Optional[LLMClient] = None, prompts_dir: Optional[Path] = None,
                 unavailable_reason: str = ""):
        self.client = client
        self.unavailable_reason = unavailable_reason
        self.prompt = load_prompt(*JUDGE_PROMPT, root=prompts_dir)
        self.repair = load_prompt(*REPAIR_PROMPT, root=prompts_dir)

    def judge(self, ticket: Any, answer_text: str, passages: Sequence[Any] = (),
              reference: Optional[dict] = None) -> JudgeVerdict:
        """Never raises. `passages` are objects or dicts with .text/["text"]; `reference` is a
        ground_truth_responses.json record (reference_response, must_mention, must_not_claim)."""
        t = ticket if isinstance(ticket, NormalisedTicket) else normalise_ticket(ticket)
        base = dict(ticket_id=t.ticket_id, prompt_version=self.prompt.version,
                    judge_model=getattr(self.client, "model", ""))
        if self.client is None:
            return JudgeVerdict(**base, error=self.unavailable_reason or "no judge model configured")
        reference = reference or {}
        text = lambda p: p.get("text", "") if isinstance(p, dict) else getattr(p, "text", "")  # noqa: E731
        messages = [{"role": "user", "content": self.prompt.render(
            ticket=sanitise_ticket_text(t.text)[0],
            passages="\n\n".join(text(p) for p in passages) or "(no passages were used)",
            reference=reference.get("reference_response") or "(no reference answer available)",
            must_mention="; ".join(reference.get("must_mention") or []) or "none listed",
            must_not_claim="; ".join(reference.get("must_not_claim") or []) or "none listed",
            answer=sanitise_ticket_text(answer_text or "")[0])}]
        error = ""
        for _ in range(MAX_ATTEMPTS):
            try:
                raw = self.client.complete(messages, max_tokens=500)
            except Exception as exc:
                return JudgeVerdict(**base, error=f"judge model unavailable: {exc}")
            try:
                reply = _Reply.model_validate(_extract_json(raw))
                return JudgeVerdict(**base, judged=True, **reply.model_dump())
            except (ValueError, ValidationError) as exc:
                error = str(exc).splitlines()[0]
                messages = messages + [{"role": "assistant", "content": raw},
                                       {"role": "user", "content": self.repair.render(error=error)}]
        return JudgeVerdict(**base, error=f"judge reply invalid after {MAX_ATTEMPTS} attempts: {error}")


def build_judge() -> Judge:
    """From the environment. JUDGE_MODEL (default MODEL_NAME) through LangChain; no key, no judge."""
    key = os.getenv("OPENROUTER_API_KEY") or os.getenv("LLM_API_KEY") or ""
    try:
        client = LangChainChatClient(
            api_key=key, model=os.getenv("JUDGE_MODEL", os.getenv("MODEL_NAME", DEFAULT_MODEL)),
            base_url=os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL), timeout=float(os.getenv("LLM_TIMEOUT", "30")),
            cache_dir=Path(os.getenv("LLM_CACHE_DIR", DEFAULT_CACHE)) if os.getenv("LLM_CACHE", "1") != "0" else None)
    except LLMError as exc:
        return Judge(None, unavailable_reason=str(exc))
    return Judge(client)
