"""Tests for evaluation/harness.py: the unattended end-to-end run (A9) and its metrics report (A10)."""
import json
import shutil
from pathlib import Path

import pytest

from evaluation import harness as H
from src.api import Metrics, SupportPipeline
from src.classify import Classifier
from src.generate import Generator
from src.guardrail import Guardrails
from src.ingest import load_tickets
from src.logging_store import DecisionStore
from src.retrieve import Corpus, Retriever, _labelled, best_threshold, score_tickets
from src.route import Router, RoutingConfig, _label_sets

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def parts():
    dev = load_tickets(DATA / "development_tickets.json")
    corpus = Corpus.from_file(DATA / "documentation.json")
    retriever = Retriever(corpus)
    retriever.thresholds["lexical"] = best_threshold(score_tickets(retriever.fallback, corpus, _labelled(dev)))
    must, docs = _label_sets(dev)
    return Classifier().fit(dev), retriever, must, docs, corpus


@pytest.fixture
def pipeline(parts, tmp_path, monkeypatch):
    monkeypatch.setenv("KILL_SWITCH_FILE", str(tmp_path / "KILL_SWITCH"))
    clf, retriever, must, docs, corpus = parts
    router = Router(RoutingConfig(confidence_threshold=0.9, must_escalate_intents=must, intent_docs=docs))
    store = DecisionStore(f"sqlite:///{tmp_path / 'decisions.db'}")
    p = SupportPipeline(clf, retriever, router, Generator(None, unavailable_reason="no key"),
                        Guardrails(corpus=corpus), store, Metrics())
    yield p
    store.close()


def outcomes(report):
    path = Path(report["run"]["output_dir"]) / "outcomes.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def run(pipeline, path, tmp_path, **kw):
    return H.run_harness(path, tmp_path / "results", pipeline=pipeline, say=lambda *_: None, **kw)


# ---------------------------------------------------------------- the seven requirements of the end-to-end run
def test_1_takes_any_input_path_and_writes_to_the_output_path(pipeline, tmp_path):
    renamed = tmp_path / "hidden_set_never_seen.json"
    shutil.copy(DATA / "validation_tickets.json", renamed)
    report = run(pipeline, renamed, tmp_path)
    out = Path(report["run"]["output_dir"])
    assert out.parent == tmp_path / "results"
    for name in ("metrics.json", "metrics.md", "outcomes.jsonl"):
        assert (out / name).is_file()


def test_2_every_ticket_in_the_file_is_processed(pipeline, tmp_path):
    report = run(pipeline, DATA / "validation_tickets.json", tmp_path)
    assert report["run"]["tickets_in_file"] == report["run"]["tickets_processed"] == 80
    out = Path(report["run"]["output_dir"])
    assert len((out / "outcomes.jsonl").read_text().splitlines()) == 80


def test_3_bad_records_do_not_stop_the_run(pipeline, tmp_path):
    good = json.loads((DATA / "validation_tickets.json").read_text())[:5]
    f = tmp_path / "mixed.jsonl"
    f.write_text("\n".join([json.dumps(t) for t in good] + ["not json at all", json.dumps({"body": ""}),
                                                            json.dumps(42), json.dumps({"channel": "sms"})]))
    report = run(pipeline, f, tmp_path)
    assert report["run"]["tickets_processed"] == 9 and report["reconcile"]["reconciled"]
    assert report["metrics"]["volume"]["unprocessable_input"] >= 3


def test_3_internal_errors_are_absorbed(pipeline, tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline.retriever, "retrieve", lambda t: 1 / 0)
    report = run(pipeline, DATA / "validation_tickets.json", tmp_path, limit=10)
    assert report["run"]["tickets_processed"] == 10 and report["metrics"]["volume"]["errors"] == 10
    assert report["metrics"]["volume"]["escalated"] == 10 and report["reconcile"]["reconciled"]


def test_4_every_ticket_is_sent_or_escalated(pipeline, tmp_path):
    report = run(pipeline, DATA / "validation_tickets.json", tmp_path)
    rows = outcomes(report)
    # "conversation": a fixed reply and no ticket (src/scope.py); in this set only VAL-0017 and VAL-0035,
    # "It is not working. Please help.", which get one clarifying question.
    assert {r["outcome"] for r in rows} == {"sent", "escalated", "conversation"}
    assert [r["ticket_id"] for r in rows if r["outcome"] == "conversation"] == ["VAL-0017", "VAL-0035"]
    assert all((r["handoff_summary"] if r["outcome"] == "escalated" else r["reply"]) for r in rows)


def test_5_every_decision_is_logged_and_reconciles(pipeline, tmp_path):
    report = run(pipeline, DATA / "validation_tickets.json", tmp_path)
    rec = report["reconcile"]
    assert rec["reconciled"] and rec["tickets_with_outcome"] == 80 and rec["by_stage"]["outcome"] == 80
    assert rec["by_stage"]["ingest"] == 80
    assert rec["by_stage"]["classify"] == 80 - report["metrics"]["volume"]["conversational_replies"]
    assert report["metrics"]["governance"]["decision_logging_complete"] is True


def test_6_metrics_report_has_every_required_figure(pipeline, tmp_path):
    m = run(pipeline, DATA / "validation_tickets.json", tmp_path)["metrics"]
    for key in ("tickets_processed", "answered_automatically", "escalated", "blocked_by_guardrails"):
        assert key in m["volume"]
    for key in ("first_contact_resolution", "response_time_mean_s_sent", "response_time_median_s_sent",
                "escalation_rate"):
        assert m["business"][key] is not None
    for key in ("intent_per_class", "retrieval_hit_rate", "latency_p50_s", "latency_p95_s"):
        assert m["technical"][key] is not None
    assert all({"precision", "recall"} <= set(v) for v in m["technical"]["intent_per_class"].values())
    for key in ("decisions_logged", "guardrail_activations_by_type", "private_data_detections"):
        assert key in m["governance"]
    assert {t["metric"] for t in m["targets"]} == set(H.TARGETS)
    e = m["escalations"]
    assert e["genuine"] + e["avoidable_findability"] <= e["labelled_escalations"] == m["volume"]["escalated"]
    assert e["drafts_for_agents"] > 0 and m["languages"]["by_language"] == {"en": 80}


def test_6_readable_report_is_written(pipeline, tmp_path):
    report = run(pipeline, DATA / "validation_tickets.json", tmp_path)
    md = (Path(report["run"]["output_dir"]) / "metrics.md").read_text()
    for heading in ("Against the Brief's targets", "## Volume", "## Escalations: genuine or avoidable",
                    "## Business outcomes", "## Technical performance",
                    "## Governance", "Fairness by customer group"):
        assert heading in md
    assert " s |" in md and "%" in md                                 # seconds and proportions formatted apart


def test_7_duration_and_latency_are_reported(pipeline, tmp_path):
    report = run(pipeline, DATA / "validation_tickets.json", tmp_path)
    assert report["run"]["duration_s"] >= 0 and report["metrics"]["technical"]["latency_p95_s"] is not None


# ---------------------------------------------------------------- the hidden set, and the numbers themselves
def test_file_without_labels_still_runs(pipeline, tmp_path):
    tickets = json.loads((DATA / "validation_tickets.json").read_text())[:15]
    for t in tickets:
        t.pop("labels")
        t.pop("history")
    f = tmp_path / "unlabelled.json"
    f.write_text(json.dumps(tickets))
    m = run(pipeline, f, tmp_path)["metrics"]
    assert m["volume"]["tickets_processed"] == 15 and m["volume"]["tickets_with_labels"] == 0
    assert m["technical"]["labelled_tickets"] == 0 and m["business"]["baseline_from_history"] is None
    assert m["business"]["first_contact_resolution"] is not None


def test_figures_match_the_outcomes(pipeline, tmp_path):
    report = run(pipeline, DATA / "validation_tickets.json", tmp_path)
    m = report["metrics"]
    rows = outcomes(report)
    sent = sum(r["outcome"] == "sent" for r in rows)
    assert m["volume"]["answered_automatically"] == sent
    assert m["business"]["first_contact_resolution"] == round(sent / 80, 4)
    escalated = sum(r["outcome"] == "escalated" for r in rows)
    assert m["business"]["escalation_rate"] == round(escalated / 80, 4)
    assert m["governance"]["must_not_auto_respond_answered_automatically"] == 0


def test_guardrail_blocks_are_counted(pipeline, tmp_path, monkeypatch):
    original = pipeline.generator.generate

    def leaky(t, r, **kw):
        a = original(t, r)
        a.answer_text += " Contact a.rao@example.com."
        return a
    monkeypatch.setattr(pipeline.generator, "generate", leaky)
    m = run(pipeline, DATA / "validation_tickets.json", tmp_path)["metrics"]
    assert m["volume"]["blocked_by_guardrails"] > 0 and m["volume"]["answered_automatically"] == 0
    assert m["governance"]["guardrail_activations_by_type"]["private_data"] == m["volume"]["blocked_by_guardrails"]
    assert m["governance"]["private_data_detections"] > 0 and m["governance"]["private_data_in_sent_replies"] == 0


def test_targets_compare_correctly():
    m = H.compute_metrics([], {"reconciled": True, "decisions": 0})
    assert all(t["met"] is None or isinstance(t["met"], bool) for t in m["targets"])


def test_cli(tmp_path, monkeypatch, capsys):
    for k, v in {"CHROMA_PATH": str(tmp_path / "none"), "RETRIEVAL_BACKEND": "lexical", "GENERATOR": "extractive",
                 "CLASSIFIER_PATH": str(tmp_path / "clf.joblib"), "RETRIEVAL_CONFIG": str(tmp_path / "r.json"),
                 "ROUTING_CONFIG": str(tmp_path / "route.json"), "KILL_SWITCH_FILE": str(tmp_path / "KS")}.items():
        monkeypatch.setenv(k, v)
    code = H.main(["--input", str(DATA / "validation_tickets.json"), "--output", str(tmp_path / "out"),
                   "--no-prepare", "--limit", "10"])
    assert code == 0 and "decision log reconciled: True" in capsys.readouterr().out
    assert H.main(["--input", str(tmp_path / "missing.json"), "--output", str(tmp_path / "out")]) == 1


def test_judge_receives_the_cited_passages(pipeline, tmp_path):
    """The judge checks grounding against the passages the reply cited, not an empty list."""
    seen = []

    class FakeJudge:
        client = object()

        def judge(self, ticket, answer_text, passages, reference=None):
            from evaluation.judge import JudgeVerdict
            seen.append(passages)
            return JudgeVerdict(ticket_id=ticket["ticket_id"], judged=True, grounded=True, rating=4,
                                answers_question=True)
    report = run(pipeline, DATA / "validation_tickets.json", tmp_path, limit=10)
    rows = outcomes(report)
    records = json.loads((DATA / "validation_tickets.json").read_text())[:10]
    for r in rows:
        r.setdefault("cited_chunks", r.get("cited_chunks", []))
    H.judge_rows(rows, records, None, lambda *_: None, corpus=pipeline.retriever.corpus, judge=FakeJudge())
    assert seen and all(p and p[0]["text"] for p in seen)
