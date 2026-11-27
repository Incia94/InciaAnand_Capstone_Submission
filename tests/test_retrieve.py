"""Tests for src/retrieve.py, one block per Build Specification bullet for Retrieve.

The real dense model (all-MiniLM-L6-v2) is not downloaded in tests. Dense behaviour is tested
with a deterministic fake embedder through both the in-memory and the Chroma paths; retrieval
quality figures come from the lexical backend, which needs no model."""
import hashlib
import json
import math
from pathlib import Path

import pytest

from src import ingest
from src import retrieve as R
from src.ingest import load_tickets

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DOCS = DATA / "documentation.json"


class FakeEmbedder:
    model_name = "fake-hash-256"

    def embed(self, texts):
        out = []
        for t in texts:
            v = [0.0] * 256
            for w in t.lower().split():
                v[int(hashlib.md5(w.strip(".,:;()`'").encode()).hexdigest(), 16) % 256] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out


@pytest.fixture(scope="module")
def corpus():
    return R.Corpus.from_file(DOCS)


@pytest.fixture(scope="module")
def dev():
    return R._labelled(load_tickets(DATA / "development_tickets.json"))


@pytest.fixture(scope="module")
def retriever(corpus, dev):
    r = R.Retriever(corpus)                                   # no dense primary: lexical only
    for b in (r.fallback,):
        r.thresholds[b.name] = R.best_threshold(R.score_tickets(b, corpus, dev))
    return r


# ---------------------------------------------------------------- 1. ranked passages with scores
def test_req1_ranked_passages_with_scores(retriever):
    res = retriever.retrieve("my production api key suddenly returns 401 unauthorized")
    assert res.status == "ok" and res.passages
    assert [p.rank for p in res.passages] == list(range(1, len(res.passages) + 1))
    scores = [p.score for p in res.passages]
    assert scores == sorted(scores, reverse=True) and all(0 <= s <= 1 for s in scores)
    assert res.passages[0].doc_id == "DOC-AUTH-004"


@pytest.mark.parametrize("query,doc", [
    ("deployment rolls back after health check timeout", "DOC-DEPLOY-001"),
    ("we get 429 too many requests, what are the rate limits", "DOC-API-001"),
    ("download link from our export expired", "DOC-DATA-001"),
    ("SSO users returned to an error page from the identity provider", "DOC-AUTH-003"),
    ("webhook signature verification fails", "DOC-API-003"),
])
def test_req1_known_questions_find_the_right_article(retriever, query, doc):
    assert retriever.retrieve(query).passages[0].doc_id == doc


def test_req1_accepts_a_ticket_dict_or_normalised_ticket(retriever, dev):
    t = next(x for x in dev if x.eval_labels["expected_doc_ids"])
    a, b = retriever.retrieve(t), retriever.retrieve(t.model_dump())
    assert a.ticket_id == t.ticket_id and a.model_dump(exclude={"latency_ms"}) == b.model_dump(exclude={"latency_ms"})


def test_req1_k_limits_passages(retriever):
    assert len(retriever.retrieve("api key rotation scope 401", k=1).passages) <= 1


def test_req1_same_query_same_result(retriever):
    q = "cursor pagination stops after the first page"
    assert retriever.retrieve(q).model_dump(exclude={"latency_ms"}) == \
        retriever.retrieve(q).model_dump(exclude={"latency_ms"})


# ---------------------------------------------------------------- 2. identifiers resolve to the real corpus
def test_req2_every_returned_id_resolves_to_documentation(retriever, corpus, dev):
    real_docs = {d["doc_id"] for d in json.loads(DOCS.read_text())}
    for t in dev[:150]:
        res = retriever.retrieve(t)
        for p in res.passages:
            chunk = corpus.resolve(p.chunk_id)
            assert chunk is not None and chunk.doc_id == p.doc_id and p.doc_id in real_docs
            assert p.text == chunk.text and corpus.verify(p.chunk_id, p.text)
        assert all(corpus.resolve(c.chunk_id) for c in res.candidates)


def test_req2_verify_rejects_invented_or_altered_citations(corpus):
    real = corpus.chunks[0]
    assert corpus.verify(real.chunk_id, real.text)
    assert not corpus.verify("DOC-FAKE-999::full")
    assert not corpus.verify(real.chunk_id, real.text + " and a refund has been issued")


def test_req2_ids_from_a_stale_index_are_dropped(corpus):
    class StaleIndex:
        name = "dense"

        def search(self, text, k):
            return [("DOC-GONE-001::full", 0.99), (corpus.chunks[3].chunk_id, 0.9)]
    res = R.Retriever(corpus, primary=StaleIndex(), thresholds={"dense": 0.5}).retrieve("anything at all")
    assert [p.chunk_id for p in res.passages] == [corpus.chunks[3].chunk_id]
    assert "unresolvable" in res.reason


def test_req2_chroma_path_returns_resolvable_ids(tmp_path, corpus):
    fake = FakeEmbedder()
    ingest.ingest_docs(DOCS, tmp_path / "chroma", embedder=fake)
    search = R.ChromaSearch(tmp_path / "chroma", ingest.DEFAULT_COLLECTION, fake.embed)
    target = corpus.resolve("DOC-API-002::full")
    hits = search.search(target.text, 3)
    assert hits[0][0] == "DOC-API-002::full" and hits[0][1] == pytest.approx(1.0, abs=1e-3)
    assert all(corpus.resolve(cid) for cid, _ in hits)


# ---------------------------------------------------------------- 3. threshold: nothing rather than noise
@pytest.mark.parametrize("query", [
    "please add a dark mode to the dashboard",
    "what is the weather in Paris tomorrow",
    "zzqx flurb wobble",
])
def test_req3_irrelevant_queries_return_nothing(retriever, query):
    res = retriever.retrieve(query)
    assert res.passages == [] and res.status == "no_relevant_passage"
    assert "below" in res.reason or "no match" in res.reason
    assert res.top_score < res.threshold


def test_req3_candidates_still_recorded_when_nothing_returned(retriever):
    res = retriever.retrieve("please add a dark mode to the dashboard")
    assert res.candidates and all(c.score < res.threshold for c in res.candidates)


def test_req3_threshold_is_learned_from_data_not_fixed(retriever):
    assert retriever.thresholds["lexical"] != R.FALLBACK_THRESHOLDS["lexical"]


def test_req3_no_doc_intents_mostly_get_nothing(retriever, dev):
    nodoc = [t for t in dev if t.eval_labels["intent"] in R.no_doc_intents(dev)]
    returned = sum(bool(retriever.retrieve(t).passages) for t in nodoc)
    assert len(nodoc) >= 30 and returned / len(nodoc) <= 0.15


def test_req3_answerable_tickets_mostly_get_the_right_doc(retriever, dev):
    ans = [t for t in dev if t.eval_labels["expected_doc_ids"]]
    right = sum(any(d in t.eval_labels["expected_doc_ids"] for d in retriever.retrieve(t).doc_ids[:1])
                for t in ans)
    assert right / len(ans) >= 0.55


@pytest.mark.parametrize("raw", ["", "   ", None, {"channel": "chat", "body": ""}])
def test_req3_empty_query_returns_nothing(retriever, raw):
    res = retriever.retrieve(raw)
    assert res.passages == [] and res.status == "empty_query"


def test_req3_ambiguous_labels_excluded_from_threshold_fitting(retriever, corpus, dev):
    rows = R.score_tickets(retriever.fallback, corpus, dev)
    groups = {g: sum(r["group"] == g for r in rows) for g in ("relevant", "wrong", "no_doc", "ambiguous")}
    assert groups["no_doc"] == 35 and groups["ambiguous"] > 0
    assert sum(groups.values()) == len(dev)


# ---------------------------------------------------------------- 4. chunking strategy chosen by measurement
def test_req4_comparison_covers_all_strategies_and_picks_document(dev):
    truth = {g["ticket_id"]: g for g in json.loads((DATA / "ground_truth_responses.json").read_text())}
    report = R.compare_strategies(DOCS, dev, "lexical", truth=truth)
    assert set(report["results"]) == set(ingest.CHUNK_STRATEGIES)
    assert report["recommended"] == ingest.DEFAULT_STRATEGY == "document"
    doc = report["results"]["document"]["development"]
    for other in ("section", "step", "window"):
        o = report["results"][other]["development"]
        assert doc["must_mention_in_top1"] > o["must_mention_in_top1"]


def test_req4_dense_comparison_runs_with_any_embedder(dev):
    report = R.compare_strategies(DOCS, dev[:60], "dense", embed=FakeEmbedder().embed,
                                  strategies=("document", "section"))
    assert set(report["results"]) == {"document", "section"}


# ---------------------------------------------------------------- failure handling (A11)
def test_dense_failure_falls_back_to_lexical(corpus, retriever):
    class Broken:
        name = "dense"

        def search(self, text, k):
            raise TimeoutError("chroma not responding")
    r = R.Retriever(corpus, primary=Broken(), thresholds=retriever.thresholds)
    res = r.retrieve("my production api key suddenly returns 401 unauthorized")
    assert res.backend == "lexical" and res.degraded and "TimeoutError" in res.reason
    assert res.passages and res.passages[0].doc_id == "DOC-AUTH-004"


def test_build_retriever_without_index_degrades_not_crashes(tmp_path):
    r = R.build_retriever(DOCS, persist_dir=tmp_path / "missing", config_path=tmp_path / "none.json",
                          embed=FakeEmbedder().embed)
    res = r.retrieve("webhook signature verification fails")
    assert res.backend == "lexical" and res.degraded and "unavailable" in res.reason


def test_build_retriever_uses_chroma_when_present(tmp_path):
    fake = FakeEmbedder()
    ingest.ingest_docs(DOCS, tmp_path / "chroma", embedder=fake)
    r = R.build_retriever(DOCS, persist_dir=tmp_path / "chroma", config_path=tmp_path / "none.json", embed=fake.embed,
                          backend="dense")
    res = r.retrieve(r.corpus.resolve("DOC-BILL-002::full").text)
    assert res.backend == "dense" and not res.degraded and res.passages[0].doc_id == "DOC-BILL-002"


def test_hybrid_is_the_default_and_combines_both(tmp_path, monkeypatch):
    """Requirement: keyword search is not the only way in. The default backend scores every
    article by meaning (dense) and by wording (lexical) together."""
    monkeypatch.delenv("RETRIEVAL_BACKEND", raising=False)
    fake = FakeEmbedder()
    ingest.ingest_docs(DOCS, tmp_path / "chroma", embedder=fake)
    r = R.build_retriever(DOCS, persist_dir=tmp_path / "chroma", config_path=tmp_path / "none.json", embed=fake.embed)
    assert r.primary.name == "hybrid" and r.primary.dense.name == "dense" and r.primary.lexical.name == "lexical"
    text = r.corpus.resolve("DOC-BILL-002::full").text
    res = r.retrieve(text)
    assert res.backend == "hybrid" and not res.degraded and res.passages[0].doc_id == "DOC-BILL-002"
    dense, lexical = dict(r.primary.dense.search(text, 29)), dict(r.primary.lexical.search(text, 29))
    cid = "DOC-BILL-002::full"
    assert res.passages[0].score == pytest.approx(0.7 * dense[cid] + 0.3 * lexical[cid], abs=1e-3)


def test_hybrid_falls_back_to_lexical_when_dense_fails(corpus):
    class Broken:
        name = "dense"

        def search(self, text, k):
            raise RuntimeError("model down")
    r = R.Retriever(corpus, primary=R.HybridSearch(Broken(), R.LexicalSearch(corpus), len(corpus.chunks)))
    res = r.retrieve("webhook signature verification fails")
    assert res.backend == "lexical" and res.degraded and "model down" in res.reason and res.passages


def test_intent_hint_rescues_a_near_miss_only_from_the_intents_articles(corpus):
    r = R.Retriever(corpus, thresholds={"lexical": 0.99, "lexical_hint": 0.05})
    q = "my signatures never match on the deliveries you send"
    plain = r.retrieve(q)
    assert plain.status == "no_relevant_passage" and not plain.passages
    hinted = r.retrieve(q, hint_docs=["DOC-API-003"])
    assert hinted.passages and hinted.passages[0].doc_id == "DOC-API-003" and hinted.passages[0].via == "intent_hint"
    assert hinted.hinted and "intent-guided" in hinted.reason
    assert not r.retrieve(q, hint_docs=["DOC-NOPE-001"]).passages      # only the intent's own articles


def test_calibrate_sets_a_lower_hint_threshold(tmp_path, corpus, dev):
    out = R.calibrate(R.Retriever(corpus), dev, tmp_path / "retrieval.json")
    assert 0 < out["thresholds"]["lexical_hint"] <= out["thresholds"]["lexical"]
    assert out["metrics"]["lexical_hint"]["near_misses"] > 0


def test_calibrate_writes_thresholds(tmp_path, corpus, dev):
    r = R.Retriever(corpus)
    out = R.calibrate(r, dev, tmp_path / "retrieval.json")
    saved = json.loads((tmp_path / "retrieval.json").read_text())
    assert saved["thresholds"]["lexical"] == out["thresholds"]["lexical"] == r.thresholds["lexical"]
    assert saved["strategy"] == "document"


def test_cli_search_and_compare(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("CHROMA_PATH", str(tmp_path / "missing"))
    assert R.main(["--backend", "lexical", "search", "429 too many requests"]) == 0
    assert "[1] Understanding API rate limits" in capsys.readouterr().out          # readable, with citations
    assert R.main(["--backend", "lexical", "search", "429 too many requests", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["results"][0]["citation"] == "[1]"
    assert R.main(["--backend", "lexical", "compare", "--output", str(tmp_path / "c.json")]) == 0
    assert json.loads((tmp_path / "c.json").read_text())["recommended"] == "document"


@pytest.mark.parametrize("argv", [["--backend", "lexical", "compare"], ["compare", "--backend", "lexical"]])
def test_cli_backend_option_works_before_or_after_the_command(argv, tmp_path, capsys):
    assert R.main(argv + ["--output", str(tmp_path / "c.json")]) == 0
    assert json.loads((tmp_path / "c.json").read_text())["backend"] == "lexical"
    capsys.readouterr()
