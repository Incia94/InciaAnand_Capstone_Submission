"""Tests for documentation ingest (src/ingest.py, part 2). A deterministic fake embedder keeps them offline and fast."""
import hashlib
import json
import math
from pathlib import Path

import pytest

from src import ingest

DOCS = Path(__file__).resolve().parents[1] / "data" / "documentation.json"


class FakeEmbedder:
    """Bag-of-words hashed into 64 dims, L2-normalised. Similar wording -> similar vectors."""
    model_name = "fake-hash-64"

    def embed(self, texts):
        out = []
        for t in texts:
            v = [0.0] * 64
            for w in t.lower().split():
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out


@pytest.fixture(scope="module")
def corpus():
    return json.loads(DOCS.read_text())


def test_default_strategy_is_one_chunk_per_article(corpus):
    chunks = ingest.chunk_corpus(corpus)
    assert ingest.DEFAULT_STRATEGY == "document"
    assert [c.chunk_id for c in chunks] == [f"{d['doc_id']}::full" for d in corpus]
    for c, d in zip(chunks, corpus):                       # nothing from the article is lost
        for section in ("Symptoms", "Common causes", "Resolution", "Notes"):
            assert section in c.text, (d["doc_id"], section)


@pytest.mark.parametrize("strategy,per_doc", [("section", 3), ("document", 1)])
def test_fixed_strategies_chunk_counts(corpus, strategy, per_doc):
    assert len(ingest.chunk_corpus(corpus, strategy=strategy)) == per_doc * len(corpus)


def test_step_and_window_strategies(corpus):
    step = ingest.chunk_corpus(corpus, strategy="step")
    assert any(c.chunk_id == "DOC-AUTH-001::resolution-4" for c in step)
    window = ingest.chunk_corpus(corpus, strategy="window")
    assert all(len(c.text) <= ingest.WINDOW_CHARS for c in window)
    assert all(not c.text.startswith(c.metadata["title"]) for c in window)   # Setup Guide baseline: no header


def test_unknown_strategy_rejected(corpus):
    with pytest.raises(ValueError, match="unknown chunking strategy"):
        ingest.chunk_corpus(corpus, strategy="sentences")


def test_section_strategy_yields_problem_resolution_notes(corpus):
    chunks = ingest.chunk_corpus(corpus, strategy="section")
    assert len(chunks) == 3 * len(corpus)
    assert {c.section for c in chunks} == {"problem", "resolution", "notes"}


def test_resolution_steps_are_never_split(corpus):
    for doc in corpus:
        res = [c for c in ingest.chunk_document(doc, strategy="section") if c.section == "resolution"][0]
        steps = [line for line in doc["content"].split("## Resolution")[1].split("## ")[0].splitlines()
                 if line[:2].rstrip(".").isdigit()]
        for n in range(1, len(steps) + 1):
            assert f"\n{n}. " in res.text, f"{doc['doc_id']} step {n} missing from resolution chunk"


def test_chunk_ids_are_stable_and_resolve_to_real_docs(corpus):
    a = [c.chunk_id for c in ingest.chunk_corpus(corpus)]
    b = [c.chunk_id for c in ingest.chunk_corpus(corpus)]
    assert a == b and len(a) == len(set(a))
    doc_ids = {d["doc_id"] for d in corpus}
    assert all(cid.split("::")[0] in doc_ids for cid in a)


def test_section_chunks_carry_title_and_fit_the_model_window(corpus):
    for c in ingest.chunk_corpus(corpus, strategy="section"):
        assert c.text.startswith(c.metadata["title"])
        assert "Applies to:" in c.text
        assert len(c.text.split()) <= ingest.MAX_WORDS_BEFORE_TRUNCATION


def test_normalise_strips_emphasis_keeps_commands():
    raw = "**Applies to:** All\r\n\r\n\r\nRun `cloudserve auth login`  now and pass next_cursor"
    out = ingest.normalise_doc_text(raw)
    assert out == "Applies to: All\n\nRun `cloudserve auth login` now and pass next_cursor"


def test_bad_rows_are_skipped_not_fatal():
    raw = [
        {"doc_id": "D1", "title": "T", "content": "## Resolution\n1. Do it."},
        {"doc_id": "D1", "title": "dup", "content": "x"},
        {"doc_id": "D2", "title": "", "content": "x"},
        {"title": "no id", "content": "x"},
        "not an object",
    ]
    valid, skipped = ingest.validate_docs(raw)
    assert [d["doc_id"] for d in valid] == ["D1"]
    assert len(skipped) == 4


def test_unexpected_layout_falls_back_to_one_chunk():
    doc = {"doc_id": "X-1", "title": "Odd article", "content": "# Odd article\n\nJust prose, no headings."}
    chunks = ingest.chunk_document(doc)
    assert [c.chunk_id for c in chunks] == ["X-1::full"]
    assert "Just prose" in chunks[0].text


def test_non_list_file_is_rejected():
    with pytest.raises(ValueError):
        ingest.validate_docs({"doc_id": "D1"})


def test_end_to_end_store_query_and_manifest(tmp_path, corpus):
    report = ingest.ingest_docs(DOCS, tmp_path / "chroma", embedder=FakeEmbedder())
    assert report["docs_ingested"] == len(corpus)
    assert report["chunks_stored"] == report["chunks"] == len(corpus)
    manifest = [json.loads(line) for line in (tmp_path / "chroma" / "chunks.jsonl").read_text().splitlines()]
    assert len(manifest) == report["chunks"]
    # Plumbing check, not a quality check: a chunk's own text must come back first.
    target = next(m for m in manifest if m["chunk_id"] == "DOC-AUTH-001::full")
    hits = ingest.quick_check(target["text"], FakeEmbedder(), tmp_path / "chroma")
    assert hits[0]["chunk_id"] == "DOC-AUTH-001::full"
    assert hits[0]["similarity"] == pytest.approx(1.0, abs=1e-3)
    # every returned id resolves to a manifest entry (what citations will rely on)
    ids = {m["chunk_id"] for m in manifest}
    assert all(h["chunk_id"] in ids for h in hits)


def test_rerun_rebuilds_instead_of_duplicating(tmp_path):
    ingest.ingest_docs(DOCS, tmp_path / "c", embedder=FakeEmbedder())
    report = ingest.ingest_docs(DOCS, tmp_path / "c", embedder=FakeEmbedder())
    assert report["chunks_stored"] == report["chunks"]


def test_empty_corpus_fails_clearly(tmp_path):
    f = tmp_path / "empty.json"
    f.write_text("[]")
    with pytest.raises(ValueError, match="no usable articles"):
        ingest.ingest_docs(f, tmp_path / "c", embedder=FakeEmbedder())


def test_cli_reads_chroma_path_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("CHROMA_PATH", str(tmp_path / "from_env"))
    monkeypatch.setattr(ingest, "SentenceTransformerEmbedder", lambda model: FakeEmbedder())
    assert ingest.main(["docs", "--docs", str(DOCS)]) == 0
    assert (tmp_path / "from_env" / "chunks.jsonl").is_file()
    assert (tmp_path / "from_env" / "ingest_report.json").is_file()


def test_cli_missing_docs_returns_error(tmp_path):
    assert ingest.main(["docs", "--docs", str(tmp_path / "nope.json"), "--persist-dir", str(tmp_path)]) == 1
