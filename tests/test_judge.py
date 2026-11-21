"""Tests for evaluation/judge.py, which uses prompts/evaluation/judge_answer.vN.txt."""
import json

from evaluation.judge import JUDGE_PROMPT, Judge, JudgeVerdict, build_judge
from src.generate import load_prompt
from src.ingest import normalise_ticket

TICKET = normalise_ticket({"ticket_id": "T1", "channel": "chat", "body": "We keep getting 429 errors."})
REFERENCE = {"reference_response": "Limits are per organisation; use backoff.", "must_mention": ["per organisation",
             "backoff"], "must_not_claim": ["a refund has been issued"]}
VERDICT = json.dumps({"grounded": True, "unsupported_claims": [], "must_mention_covered": ["per organisation"],
                      "must_mention_missing": ["backoff"], "forbidden_claims_made": [], "answers_question": True,
                      "rating": 4, "rationale": "Correct but misses backoff."})


class FakeLLM:
    model = "fake-judge"

    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    def complete(self, messages, temperature=0.0, max_tokens=600):
        self.calls.append(messages)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def test_judge_uses_the_evaluation_prompt_and_parses_the_verdict():
    llm = FakeLLM(VERDICT)
    v = Judge(llm).judge(TICKET, "Limits apply per organisation.", [{"text": "Limits are enforced per organisation."}],
                         REFERENCE)
    assert v.judged and v.grounded and v.rating == 4 and v.must_mention_missing == ["backoff"]
    assert v.prompt_version == load_prompt(*JUDGE_PROMPT).version and v.judge_model == "fake-judge"
    sent = llm.calls[0][0]["content"]
    assert "per organisation; backoff" in sent and "a refund has been issued" in sent
    assert sent.index("<candidate>") < sent.index("Limits apply per organisation.") < sent.index("</candidate>")


def test_judge_repairs_once_then_gives_up_cleanly():
    v = Judge(FakeLLM("not json", VERDICT)).judge(TICKET, "answer", [], REFERENCE)
    assert v.judged and v.rating == 4
    v = Judge(FakeLLM("not json", json.dumps({"rating": 9}))).judge(TICKET, "answer", [], REFERENCE)
    assert not v.judged and "invalid after 2 attempts" in v.error


def test_judge_without_a_model_reports_why():
    v = Judge(None, unavailable_reason="no API key configured").judge(TICKET, "answer")
    assert isinstance(v, JudgeVerdict) and not v.judged and v.error == "no API key configured"


def test_judge_survives_a_provider_outage():
    v = Judge(FakeLLM(TimeoutError("down"))).judge(TICKET, "answer", [], REFERENCE)
    assert not v.judged and "unavailable" in v.error


def test_build_judge_without_key(monkeypatch):
    for k in ("OPENROUTER_API_KEY", "LLM_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    assert build_judge().client is None
