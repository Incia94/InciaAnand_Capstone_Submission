"""Retrieve: find the documentation passages that bear on a ticket, or nothing.

Build Specification (Retrieve) -> where it is met                         -> tests (tests/test_retrieve.py)
-------------------------------------------------------------------------------------------------------------
Ranked passages with scores                       Retriever.retrieve -> RetrievalResult.passages  test_req1_*
Identifiers resolve to the real corpus            Corpus: every id checked against documentation.json test_req2_*
Relevance threshold: nothing rather than noise     thresholds calibrated from labelled tickets   test_req3_*
Chunking strategy chosen and justified             `compare` measures all four strategies        test_req4_*

Usage (run from the project root, after `python -m src.ingest docs`)
-----
    python -m src.retrieve search "my api key suddenly returns 401"
    python -m src.retrieve calibrate                       # sets thresholds from development tickets
    python -m src.retrieve evaluate --tickets data/validation_tickets.json
    python -m src.retrieve compare --backend hybrid        # chunking strategies, real embeddings + words
    python -m src.retrieve compare --backend dense         # chunking strategies, embeddings only
    python -m src.retrieve compare --backend lexical       # same, word matching (no model needed)

Backends (RETRIEVAL_BACKEND)
--------
hybrid   the default. Meaning (dense embeddings) and exact wording (lexical) combined: every
         article gets HYBRID_DENSE_WEIGHT x dense score + the rest x lexical score (0.7 / 0.3 by
         default). Dense search finds a passage when the customer uses different words from the
         article ("can't get in" for "sign-in fails"); lexical search keeps exact terms such as
         error codes, product names and "429" from being lost. Neither alone is relied on.
dense    all-MiniLM-L6-v2 embeddings in Chroma only (built by `python -m src.ingest docs`).
lexical  TF-IDF over words and character n-grams, built in memory from documentation.json.
         Needs no model and no network, so it is the fallback when the dense path fails
         (acceptance criterion A11). Results then carry degraded=True and the reason.
The backends score on different scales, so each has its own threshold.

Intent-guided search (findability)
----------------------------------
Many tickets that a person could answer from the documentation are escalated only because the
best passage scores just under the threshold: the customer described the problem in words the
article does not use. When the classifier is confident about the intent, `retrieve(hint_docs=...)`
is given the articles that tickets of that intent have needed before. If none of the normal
results passes the threshold but one of those articles is among the candidates and scores above
a second, lower threshold (the hint threshold, also calibrated from the development tickets),
that article is returned, marked via="intent_hint". Two independent signals then agree, and the
guardrails still check every claim against the article before anything is sent. Routing decides
from data whether this is switched on (storage/routing.json: use_intent_hint).

The threshold
-------------
For each labelled development ticket the top score is compared with whether the top passage
comes from a document a correct answer should cite (`expected_doc_ids`). The threshold is the
score that best separates "the best passage is right" from "the best passage is wrong, or the
intent has no documentation at all" (maximum of true-positive rate minus false-positive rate).
Tickets labelled unanswerable whose intent does have documentation are left out: the same text
carries both labels in this data (discovery finding E24). Below the threshold, retrieve returns
no passages, status "no_relevant_passage", and the router escalates. The closest candidates are
still recorded in `candidates` for the decision log.

Chunking (chosen by `compare`; see the README/report for the figures)
--------
"document": one chunk per article, headed by its title. The articles are short (133-209 words)
and each answers one problem, so splitting them separated the symptoms a customer describes from
the fix, which made relevance harder to judge and left answer points out of the top passage.
Caveat for the dense backend: all-MiniLM-L6-v2 reads about 256 word pieces, so for the 8 longest
articles the end (mostly Notes) is not embedded. The full text is still stored and returned.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np
from pydantic import BaseModel

from src.ingest import (CHUNK_STRATEGIES, DEFAULT_COLLECTION, DEFAULT_MODEL, DEFAULT_STRATEGY, Chunk,
                        NormalisedTicket, chroma_client, chunk_corpus, load_tickets, normalise_ticket,
                        validate_docs)

log = logging.getLogger("retrieve")

DEFAULT_DOCS = "./data/documentation.json"
DEFAULT_CHROMA = "./storage/chroma"
DEFAULT_CONFIG = "./storage/retrieval.json"
DEFAULT_TRAIN = "./data/development_tickets.json"
DEFAULT_K = 5
# Used only until `calibrate` has run (and written storage/retrieval.json).
FALLBACK_THRESHOLDS = {"dense": 0.45, "lexical": 0.20, "hybrid": 0.38}
HINT_FACTOR = 0.6                 # hint threshold = 0.6 x threshold until calibrate has run
APPROVED_PRECISION = 0.95         # Tier 1: the score above which the top article is right this often
APPROVED_MIN_TICKETS = 20
APPROVED_MARGIN = 0.10            # until calibrate has run: threshold + 0.10
DEFAULT_HYBRID_WEIGHT = 0.7
DEFAULT_BM25_WEIGHT = 0.3       # share of BM25 in the keyword leg (LEXICAL_BM25_WEIGHT)


# ------------------------------------------------------------------ output
class Passage(BaseModel):
    rank: int
    chunk_id: str          # e.g. DOC-AUTH-004::resolution; resolves through Corpus.resolve
    doc_id: str            # e.g. DOC-AUTH-004; an article in documentation.json
    title: str
    section: str
    score: float
    text: str
    via: str = "search"    # "search": passed the relevance threshold; "intent_hint": see module notes


class Candidate(BaseModel):
    chunk_id: str
    score: float
    title: str = ""
    excerpt: str = ""      # the start of the passage, so a person can see what it says


class RetrievalResult(BaseModel):
    ticket_id: str = ""
    status: str            # ok | no_relevant_passage | empty_query | unavailable
    passages: list[Passage] = []        # above threshold, best first; empty means "nothing relevant"
    candidates: list[Candidate] = []    # best matches regardless of threshold, for the decision log
    top_score: float = 0.0
    threshold: float = 0.0
    backend: str = ""
    degraded: bool = False              # True when the primary backend failed and lexical answered
    hinted: bool = False                # True when the passage came from intent-guided search
    reason: str = ""
    latency_ms: float = 0.0

    @property
    def doc_ids(self) -> list[str]:
        return list(dict.fromkeys(p.doc_id for p in self.passages))


# ------------------------------------------------------------------ the corpus: source of truth for ids
class Corpus:
    """Chunks rebuilt from documentation.json. Any id a backend returns is checked here, so a
    stale or tampered index can never produce a citation that does not exist."""

    def __init__(self, chunks: Sequence[Chunk], strategy: str):
        self.strategy = strategy
        self.chunks = list(chunks)
        self.by_id = {c.chunk_id: c for c in self.chunks}
        self.doc_ids = {c.doc_id for c in self.chunks}

    @classmethod
    def from_file(cls, docs_path: Path, strategy: str = DEFAULT_STRATEGY) -> "Corpus":
        docs, skipped = validate_docs(json.loads(Path(docs_path).read_text(encoding="utf-8")))
        for s in skipped:
            log.warning("corpus: skipped %s (%s)", s["doc"], s["reason"])
        return cls(chunk_corpus(docs, source_file=Path(docs_path).name, strategy=strategy), strategy)

    def resolve(self, chunk_id: str) -> Optional[Chunk]:
        return self.by_id.get(chunk_id)

    def verify(self, chunk_id: str, text: Optional[str] = None) -> bool:
        """True if the id exists and, when text is given, the text is exactly that chunk's text."""
        chunk = self.by_id.get(chunk_id)
        return chunk is not None and (text is None or text == chunk.text)


# ------------------------------------------------------------------ search backends
Hits = list[tuple[str, float]]  # (chunk_id, score), best first


class LexicalSearch:
    """The keyword leg: TF-IDF over words and character n-grams, blended with BM25 (LEXICAL_BM25_WEIGHT,
    default 0.3). TF-IDF's character n-grams tolerate misspellings ("deploymnet"); BM25 rewards exact
    terms and error codes ("429", "ERR_QUOTA") that appear in few articles. Chosen on the development
    tickets (cross-fitted Youden J 0.512 blended, 0.500 TF-IDF alone, 0.447 BM25 alone)."""
    name = "lexical"

    def __init__(self, corpus: Corpus, bm25_weight: Optional[float] = None):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import make_union
        self.ids = [c.chunk_id for c in corpus.chunks]
        self.vectoriser = make_union(
            TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english"),
            TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True))
        self.matrix = self.vectoriser.fit_transform([c.text for c in corpus.chunks])
        self.bm25_weight = float(os.getenv("LEXICAL_BM25_WEIGHT", DEFAULT_BM25_WEIGHT)
                                 if bm25_weight is None else bm25_weight)
        self.bm25 = BM25Search(corpus) if self.bm25_weight > 0 else None

    def search(self, text: str, k: int) -> Hits:
        from sklearn.metrics.pairwise import cosine_similarity
        q = self.vectoriser.transform([text])
        scores = cosine_similarity(q, self.matrix)[0] if q.nnz else np.zeros(len(self.ids))
        if self.bm25 is not None:
            raw, ceiling = self.bm25.raw_scores(text)
            bm = raw / ceiling if ceiling > 0 else np.zeros(len(self.ids))
            scores = (1 - self.bm25_weight) * scores + self.bm25_weight * bm
        if not scores.any():
            return []
        order = np.argsort(-scores, kind="stable")[:k]
        return [(self.ids[i], float(scores[i])) for i in order]


_TOKEN = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")
_BM25_STOP = frozenset("""a an the and or but if then so to of in on at by for with from about as is are was were be
been being do does did have has had i me my we our us you your it its they them their this that these those
what which who when where why how can could will would should may might must not no yes please""".split())


def bm25_tokens(text: str) -> list[str]:
    """Lower-case words and codes. Error codes and identifiers stay whole ("429", "err_quota_exceeded",
    "x-api-key", "v2.1"), and each part of a joined identifier is indexed too, so "quota" finds it."""
    out = []
    for tok in _TOKEN.findall((text or "").lower()):
        if tok in _BM25_STOP:
            continue
        out.append(tok)
        parts = re.split(r"[._-]", tok)
        if len(parts) > 1:
            out.extend(p for p in parts if p and p not in _BM25_STOP)
    return out


class BM25Search:
    """Okapi BM25 (k1=1.5, b=0.75) over the articles: exact keyword and error-code matching.

    BM25 scores have no fixed scale, which a relevance threshold needs. Each score is divided by the
    most a passage could score for that query (every query term present, saturated, in a passage of
    average length), so 1.0 means "contains every query term, many times" and scores compare across
    queries. No model and no network: this is also the fallback when dense search fails (A11)."""
    name = "lexical"
    method = "bm25"

    def __init__(self, corpus: Corpus, k1: float = 1.5, b: float = 0.75):
        from collections import Counter
        self.ids = [c.chunk_id for c in corpus.chunks]
        self.k1, self.b = k1, b
        docs = [bm25_tokens(c.text) for c in corpus.chunks]
        self.tf = [Counter(d) for d in docs]
        self.len = np.array([len(d) for d in docs], dtype=float)
        self.avg = float(self.len.mean()) if len(docs) else 1.0
        n = len(docs)
        df = Counter(t for d in docs for t in set(d))
        self.idf = {t: float(np.log(1 + (n - f + 0.5) / (f + 0.5))) for t, f in df.items()}

    def raw_scores(self, text: str) -> tuple[np.ndarray, float]:
        terms = [t for t in dict.fromkeys(bm25_tokens(text)) if t in self.idf]
        scores = np.zeros(len(self.ids))
        if not terms:
            return scores, 0.0
        norm = self.k1 * (1 - self.b + self.b * self.len / self.avg)
        for t in terms:
            f = np.array([tf.get(t, 0) for tf in self.tf], dtype=float)
            scores += self.idf[t] * f * (self.k1 + 1) / (f + norm)
        ceiling = sum(self.idf[t] for t in terms) * (self.k1 + 1)
        return scores, ceiling

    def search(self, text: str, k: int) -> Hits:
        raw, ceiling = self.raw_scores(text)
        if ceiling <= 0:
            return []
        scores = raw / ceiling
        order = np.argsort(-scores, kind="stable")[:k]
        return [(self.ids[i], float(scores[i])) for i in order]


class InMemoryDenseSearch:
    """Dense search without Chroma: used by `compare`, and by tests with a fake embedder."""
    name = "dense"

    def __init__(self, corpus: Corpus, embed: Callable[[Sequence[str]], list[list[float]]]):
        self.ids = [c.chunk_id for c in corpus.chunks]
        self.embed = embed
        m = np.asarray(embed([c.text for c in corpus.chunks]), dtype=float)
        self.matrix = m / np.clip(np.linalg.norm(m, axis=1, keepdims=True), 1e-12, None)

    def search(self, text: str, k: int) -> Hits:
        q = np.asarray(self.embed([text])[0], dtype=float)
        q = q / max(np.linalg.norm(q), 1e-12)
        scores = self.matrix @ q
        order = np.argsort(-scores, kind="stable")[:k]
        return [(self.ids[i], float(scores[i])) for i in order]


class ChromaSearch:
    """The production dense path: the collection written by `python -m src.ingest docs`."""
    name = "dense"

    def __init__(self, persist_dir: Path, collection: str, embed: Callable[[Sequence[str]], list[list[float]]]):
        self.collection = chroma_client(persist_dir).get_collection(collection)
        if self.collection.count() == 0:
            raise RuntimeError(f"collection {collection!r} in {persist_dir} is empty")
        self.embed = embed

    def search(self, text: str, k: int) -> Hits:
        k = min(k, self.collection.count())
        res = self.collection.query(query_embeddings=self.embed([text]), n_results=k, include=["distances"])
        return [(cid, float(1.0 - d)) for cid, d in zip(res["ids"][0], res["distances"][0])]


class HybridSearch:
    """Dense and lexical scores combined for every passage. The corpus is small (one passage per
    article), so both backends score all of it and nothing is missed by either one's cut-off."""
    name = "hybrid"

    def __init__(self, dense: Any, lexical: LexicalSearch, n_chunks: int, weight: float = DEFAULT_HYBRID_WEIGHT):
        self.dense, self.lexical, self.n, self.weight = dense, lexical, n_chunks, float(weight)

    def search(self, text: str, k: int) -> Hits:
        dense = dict(self.dense.search(text, self.n))          # raises if dense fails -> lexical fallback
        lexical = dict(self.lexical.search(text, self.n))
        ids = set(dense) | set(lexical)
        scores = {cid: self.weight * dense.get(cid, 0.0) + (1 - self.weight) * lexical.get(cid, 0.0) for cid in ids}
        return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]


# ------------------------------------------------------------------ the retriever
class Retriever:
    def __init__(self, corpus: Corpus, primary: Any = None, fallback: Optional[LexicalSearch] = None,
                 thresholds: Optional[dict] = None, k: int = DEFAULT_K, primary_error: str = ""):
        self.corpus = corpus
        self.primary = primary
        self.fallback = fallback if fallback is not None else LexicalSearch(corpus)
        self.thresholds = {**FALLBACK_THRESHOLDS, **(thresholds or {})}
        self.k = k
        self.primary_error = primary_error

    def retrieve(self, ticket: Any, k: Optional[int] = None,
                 hint_docs: Optional[Sequence[str]] = None, filters: Optional[dict] = None) -> RetrievalResult:
        """Never raises. Returns passages above the threshold, best first, or none. With hint_docs
        (the articles the ticket's intent has needed before), a near miss from one of those
        articles is returned when it clears the lower hint threshold (see module notes). With
        filters ({"category": ..., "product": ..., "doc_version": ...}, each a value or a list),
        only matching articles are considered (see matches_filters)."""
        started = time.perf_counter()
        k = k or self.k
        ticket_id, text = _query_of(ticket)
        result = self._retrieve(ticket_id, text, k, set(hint_docs or ()), filters or None)
        result.latency_ms = round((time.perf_counter() - started) * 1000, 2)
        return result

    def approved_threshold(self, backend: str) -> float:
        """Tier 1 (approved content, no model): a match this strong is right often enough that the
        article's own resolution steps are sent as the answer (see approved_threshold_from)."""
        base = self.thresholds.get(backend, FALLBACK_THRESHOLDS.get(backend, 0.3))
        return float(self.thresholds.get(f"{backend}_approved", base + APPROVED_MARGIN))

    def hint_threshold(self, backend: str) -> float:
        key = f"{backend}_hint"
        return float(self.thresholds.get(key, HINT_FACTOR * self.thresholds.get(backend, FALLBACK_THRESHOLDS[backend])))

    def passages_for(self, refs: Sequence[tuple[str, float]], via: str = "search") -> list[Passage]:
        """Passages for given (chunk_id, score) pairs, e.g. near misses offered to an agent."""
        out = []
        for cid, score in refs:
            c = self.corpus.resolve(cid)
            if c is not None:
                out.append(Passage(rank=len(out) + 1, chunk_id=cid, doc_id=c.doc_id, title=c.metadata.get("title", ""),
                                   section=c.section, score=round(score, 4), text=c.text, via=via))
        return out

    def facets(self) -> dict[str, list[str]]:
        """The values each metadata filter can take, from the indexed articles."""
        out: dict[str, set] = {"category": set(), "product": set(), "doc_version": set()}
        for c in self.corpus.chunks:
            out["category"].add(c.metadata.get("category", ""))
            out["doc_version"].add(c.metadata.get("doc_version", "1.0"))
            out["product"].update(p for p in str(c.metadata.get("product", "")).split(",") if p)
        return {k: sorted(v - {""}) for k, v in out.items()}

    def _retrieve(self, ticket_id: str, text: str, k: int, hint_docs: Optional[set] = None,
                  filters: Optional[dict] = None) -> RetrievalResult:
        if not text.strip():
            return RetrievalResult(ticket_id=ticket_id, status="empty_query", reason="ticket has no usable text")
        backend, degraded, reason = self.primary, False, ""
        if backend is None:
            backend, degraded = self.fallback, True
            reason = self.primary_error or "no dense index configured; lexical search used"
        hits: Hits = []
        wide = bool(hint_docs or filters)                   # filtering needs every article scored first
        try:
            hits = backend.search(text, len(self.corpus.chunks) if wide else max(k, DEFAULT_K))
        except Exception as exc:
            log.warning("%s search failed (%s); using lexical fallback", backend.name, exc)
            backend, degraded, reason = self.fallback, True, f"{backend.name} failed: {type(exc).__name__}: {exc}"
            try:
                hits = backend.search(text, len(self.corpus.chunks) if wide else max(k, DEFAULT_K))
            except Exception as exc2:  # nothing left to try
                return RetrievalResult(ticket_id=ticket_id, status="unavailable", degraded=True,
                                       reason=f"{reason}; lexical failed: {exc2}")

        verified = [(cid, s) for cid, s in hits if self.corpus.resolve(cid) is not None]
        if len(verified) < len(hits):
            reason = (reason + "; " if reason else "") + f"{len(hits) - len(verified)} unresolvable id(s) dropped"
        if filters:
            verified = [(cid, s) for cid, s in verified if matches_filters(self.corpus.resolve(cid).metadata, filters)]
            reason = (reason + "; " if reason else "") + "filtered by " + ", ".join(
                f"{key}={val}" for key, val in filters.items())
        threshold = float(self.thresholds.get(backend.name, FALLBACK_THRESHOLDS[backend.name]))
        top = verified[0][1] if verified else 0.0
        passages = []
        for cid, score in verified:
            if score < threshold or len(passages) >= k:
                break
            c = self.corpus.resolve(cid)
            passages.append(Passage(rank=len(passages) + 1, chunk_id=cid, doc_id=c.doc_id,
                                    title=c.metadata.get("title", ""), section=c.section,
                                    score=round(score, 4), text=c.text))
        hinted = False
        if not passages and hint_docs:
            hint_th = self.hint_threshold(backend.name)
            near = next(((cid, sc) for cid, sc in verified if self.corpus.resolve(cid).doc_id in hint_docs), None)
            if near and near[1] >= hint_th:
                passages = self.passages_for([near], via="intent_hint")
                hinted = True
                reason = (reason + "; " if reason else "") + (
                    f"no passage passed the {threshold:.3f} threshold; {near[0]} ({near[1]:.3f}) is an article this "
                    f"kind of ticket normally needs and passed the {hint_th:.3f} intent-guided threshold")
        status = "ok" if passages else "no_relevant_passage"
        if not passages:
            reason = (reason + "; " if reason else "") + (
                f"best match {verified[0][0]} scored {top:.3f}, below the {threshold:.3f} threshold"
                if verified else "no match at all")
        return RetrievalResult(
            ticket_id=ticket_id, status=status, passages=passages,
            candidates=[Candidate(chunk_id=c, score=round(s, 4),
                                  title=self.corpus.resolve(c).metadata.get("title", ""),
                                  excerpt=_excerpt_of(self.corpus.resolve(c).text))
                        for c, s in verified[:max(k, DEFAULT_K)]],
            top_score=round(top, 4), threshold=round(threshold, 4), backend=backend.name,
            degraded=degraded, hinted=hinted, reason=reason)


def cited_results(retriever: "Retriever", query: str, filters: Optional[dict] = None, k: int = DEFAULT_K,
                  store: Any = None) -> dict:
    """Documentation search for a person (an agent, or the API's /search): the matching passages,
    each numbered and citable as [n] with its article title and id. A search that finds nothing
    says so (no near misses dressed up as answers) and, given a decision log, is recorded as a
    knowledge gap (src/gaps.py)."""
    r = retriever.retrieve({"ticket_id": "SEARCH", "body": query}, k=k, filters=filters)
    results = []
    for n, p in enumerate(r.passages, 1):
        meta = retriever.corpus.resolve(p.chunk_id).metadata
        results.append({"citation": f"[{n}]", "doc_id": p.doc_id, "title": p.title, "chunk_id": p.chunk_id,
                        "score": p.score, "category": meta.get("category"), "product": meta.get("product"),
                        "doc_version": meta.get("doc_version"), "excerpt": _excerpt_of(p.text),
                        "via": p.via})
    if not results and store is not None and query.strip():
        from src.gaps import record_search_gap
        record_search_gap(store, query, "documentation search", topic_hint=", ".join(
            (filters or {}).get("category") or []) or None)
    return {"query": query, "filters": filters or {}, "backend": r.backend, "degraded": r.degraded,
            "threshold": r.threshold, "status": r.status, "results": results,
            "sources": [f"{x['citation']} {x['title']} ({x['doc_id']})" for x in results],
            "closest_below_threshold": [] if results else [
                {"doc_id": c.chunk_id.split("::")[0], "title": c.title, "score": c.score} for c in r.candidates[:3]],
            "note": "" if results else "No article is relevant enough to cite."}


def matches_filters(meta: dict, filters: dict) -> bool:
    """Metadata filtering. category and doc_version must match one of the values given. product
    matches an article written for that product, or one that applies to every product."""
    def wanted(key):
        v = filters.get(key)
        if v in (None, "", [], ()):
            return None
        return {str(x).lower() for x in (v if isinstance(v, (list, tuple, set)) else [v])}
    cat, ver, prod = wanted("category"), wanted("doc_version"), wanted("product")
    if cat and str(meta.get("category", "")).lower() not in cat:
        return False
    if ver and str(meta.get("doc_version", "1.0")).lower() not in ver:
        return False
    if prod:
        have = {p.strip().lower() for p in str(meta.get("product", "")).split(",") if p.strip()}
        if not (have & prod or "all products" in have):
            return False
    return True


def _excerpt_of(text: str, limit: int = 400) -> str:
    """The part of an article that holds the fix, if it is marked, else its opening."""
    body = text.split("Resolution:", 1)[-1] if "Resolution:" in text else text
    body = " ".join(body.split())
    return body if len(body) <= limit else body[:limit].rsplit(" ", 1)[0] + "…"


def _query_of(ticket: Any) -> tuple[str, str]:
    if isinstance(ticket, str):
        return "", ticket
    t = ticket if isinstance(ticket, NormalisedTicket) else normalise_ticket(ticket)
    return t.ticket_id, t.text if t.processable else ""


# ------------------------------------------------------------------ measurement
def _labelled(tickets: Iterable[Any]) -> list[NormalisedTicket]:
    rows = [t if isinstance(t, NormalisedTicket) else normalise_ticket(t) for t in tickets]
    return [t for t in rows if t.processable and t.eval_labels is not None]


def no_doc_intents(tickets: Sequence[NormalisedTicket]) -> set[str]:
    """Intents for which no ticket ever has an expected document (here: feature_request and
    unclear_request). Tickets of these intents are truly unanswerable from the docs."""
    has_doc: dict[str, bool] = {}
    for t in tickets:
        intent = t.eval_labels.get("intent", "")
        has_doc[intent] = has_doc.get(intent, False) or bool(t.eval_labels.get("expected_doc_ids"))
    return {i for i, h in has_doc.items() if not h}


def score_tickets(search: Any, corpus: Corpus, tickets: Sequence[NormalisedTicket], k: int = DEFAULT_K,
                  no_doc: Optional[set[str]] = None) -> list[dict]:
    """Per ticket: top score, rank of the first correct document, and which group it falls in:

    relevant   the top passage comes from an expected document
    wrong      the ticket has expected documents, but the top passage is from another one
    no_doc     the ticket's intent never has documentation: any passage is irrelevant
    ambiguous  labelled not answerable although its intent has documentation. Identical ticket
               text carries both labels in this data (discovery finding E24), so these tickets
               say nothing reliable about relevance and are left out of threshold fitting.
    """
    no_doc = no_doc_intents(tickets) if no_doc is None else no_doc
    out = []
    for t in tickets:
        expected = set(t.eval_labels.get("expected_doc_ids") or [])
        hits = [(cid, s) for cid, s in search.search(t.text, k) if corpus.resolve(cid)]
        docs = [corpus.resolve(cid).doc_id for cid, _ in hits]
        first = next((i + 1 for i, d in enumerate(docs) if d in expected), None)
        top_correct = bool(docs) and docs[0] in expected
        if expected:
            group = "relevant" if top_correct else "wrong"
        else:
            group = "no_doc" if t.eval_labels.get("intent") in no_doc else "ambiguous"
        out.append({"ticket_id": t.ticket_id, "top_score": hits[0][1] if hits else 0.0,
                    "top_correct": top_correct, "first_correct_rank": first, "group": group,
                    "top_text": corpus.resolve(hits[0][0]).text if hits else "",
                    "top3_text": " ".join(corpus.resolve(c).text for c, _ in hits[:3]),
                    "answerable": bool(expected)})
    return out


def intent_doc_map(tickets: Sequence[NormalisedTicket]) -> dict[str, list[str]]:
    """Intent -> the documents its labelled tickets were answered from."""
    docs: dict[str, set] = {}
    for t in tickets:
        for d in t.eval_labels.get("expected_doc_ids") or []:
            docs.setdefault(t.eval_labels["intent"], set()).add(d)
    return {k: sorted(v) for k, v in sorted(docs.items())}


def hint_rows(search: Any, corpus: Corpus, tickets: Sequence[NormalisedTicket],
              intent_docs: dict[str, list[str]], no_doc: Optional[set[str]] = None) -> list[dict]:
    """Like score_tickets, but the 'top' passage is the best-scoring one from the articles the
    ticket's intent has needed: what intent-guided search would return. Used to fit its threshold."""
    no_doc = no_doc_intents(tickets) if no_doc is None else no_doc
    out = []
    for t in tickets:
        hint = set(intent_docs.get(t.eval_labels.get("intent", ""), []))
        if not hint:
            continue
        expected = set(t.eval_labels.get("expected_doc_ids") or [])
        hits = [(cid, s) for cid, s in search.search(t.text, len(corpus.chunks)) if corpus.resolve(cid)]
        best = next(((cid, s) for cid, s in hits if corpus.resolve(cid).doc_id in hint), None)
        if best is None:
            continue
        correct = corpus.resolve(best[0]).doc_id in expected
        group = ("relevant" if correct else "wrong") if expected else (
            "no_doc" if t.eval_labels.get("intent") in no_doc else "ambiguous")
        out.append({"ticket_id": t.ticket_id, "top_score": best[1], "top_correct": correct, "group": group,
                    "first_correct_rank": 1 if correct else None, "answerable": bool(expected),
                    "top_text": "", "top3_text": ""})
    return out


def _fit_rows(rows: Sequence[dict]) -> list[dict]:
    return [r for r in rows if r["group"] != "ambiguous"]


def approved_threshold_from(rows: Sequence[dict], floor: float) -> float:
    """The lowest top score at or above which the top article is the right one for at least
    APPROVED_PRECISION of labelled tickets (and at least APPROVED_MIN_TICKETS of them). Never below
    the relevance threshold."""
    usable = sorted((r for r in rows if r["group"] in ("relevant", "wrong", "no_doc")),
                    key=lambda r: -r["top_score"])
    best, right = None, 0
    for n, r in enumerate(usable, 1):
        right += bool(r["top_correct"])
        if n >= APPROVED_MIN_TICKETS and right / n >= APPROVED_PRECISION:
            best = r["top_score"]
    return max(floor, best) if best is not None else floor + APPROVED_MARGIN


def best_threshold(rows: Sequence[dict]) -> float:
    """Score that best separates 'top passage is right' from 'top passage is wrong or there is
    nothing to find' (max TPR - FPR), ignoring ambiguous tickets."""
    rows = _fit_rows(rows)
    scores = np.array([r["top_score"] for r in rows])
    good = np.array([r["top_correct"] for r in rows])
    if len(rows) == 0:
        return 0.0
    if good.all() or (~good).all():
        return float(np.median(scores))
    best, best_j = 0.0, -1.0
    for th in np.unique(scores):
        j = (scores[good] >= th).mean() - (scores[~good] >= th).mean()
        if j > best_j:
            best, best_j = float(th), j
    return best


def metrics(rows: Sequence[dict], threshold: float, truth: Optional[dict] = None) -> dict:
    ans = [r for r in rows if r["answerable"]]
    nodoc = [r for r in rows if r["group"] == "no_doc"]
    fit = _fit_rows(rows)
    scores = np.array([r["top_score"] for r in fit])
    good = np.array([r["top_correct"] for r in fit])
    auc = None
    if good.any() and (~good).any():
        pos, neg = scores[good], scores[~good]
        auc = float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))
    ranks = [r["first_correct_rank"] for r in ans]
    returned = [r for r in fit if r["top_score"] >= threshold]
    out = {
        "tickets": len(rows), "answerable": len(ans), "no_doc": len(nodoc),
        "ambiguous_excluded": len(rows) - len(fit),
        "hit_at_1": round(np.mean([x == 1 for x in ranks]), 3) if ans else None,
        "hit_at_3": round(np.mean([x is not None and x <= 3 for x in ranks]), 3) if ans else None,
        "mrr": round(np.mean([1 / x if x else 0 for x in ranks]), 3) if ans else None,
        "separation_auc": round(auc, 3) if auc is not None else None,
        "threshold": round(threshold, 4),
        # with the threshold applied
        "retrieval_hit_rate": round(np.mean([r["top_score"] >= threshold and r["top_correct"] for r in ans]), 3)
        if ans else None,
        "returned_for_no_doc": round(np.mean([r["top_score"] >= threshold for r in nodoc]), 3) if nodoc else None,
        "precision_when_returned": round(np.mean([r["top_correct"] for r in returned]), 3) if returned else None,
    }
    if truth:  # share of the senior agents' must_mention points found in the passages
        cov1, cov3 = [], []
        for r in rows:
            mm = [m.lower() for m in (truth.get(r["ticket_id"], {}).get("must_mention") or [])]
            if mm:
                cov1.append(np.mean([m in r["top_text"].lower() for m in mm]))
                cov3.append(np.mean([m in r["top3_text"].lower() for m in mm]))
        out["must_mention_in_top1"] = round(float(np.mean(cov1)), 3) if cov1 else None
        out["must_mention_in_top3"] = round(float(np.mean(cov3)), 3) if cov3 else None
        out["must_mention_tickets"] = len(cov1)
    return out


def cross_fitted_metrics(rows: Sequence[dict], folds: int = 5) -> dict:
    """Threshold chosen on 4/5 of the tickets and applied to the other 1/5, so the thresholded
    figures are not flattered by choosing and scoring on the same tickets."""
    rows = _fit_rows(rows)
    idx = np.arange(len(rows))
    np.random.default_rng(0).shuffle(idx)
    applied = []
    for f in range(folds):
        test = set(idx[f::folds].tolist())
        th = best_threshold([rows[i] for i in idx if i not in test])
        applied += [(rows[i]["top_score"] >= th, rows[i]["top_correct"]) for i in test]
    ret = np.array([a for a, _ in applied], dtype=bool)
    good = np.array([g for _, g in applied], dtype=bool)
    if not good.any() or good.all():
        return {"cv_tpr": None, "cv_fpr": None, "cv_j": None}
    return {"cv_tpr": round(float(ret[good].mean()), 3), "cv_fpr": round(float(ret[~good].mean()), 3),
            "cv_j": round(float(ret[good].mean() - ret[~good].mean()), 3)}


def compare_strategies(docs_path: Path, train: Sequence[NormalisedTicket], backend: str,
                       embed: Optional[Callable] = None, validation: Optional[Sequence[NormalisedTicket]] = None,
                       strategies: Sequence[str] = CHUNK_STRATEGIES, truth: Optional[dict] = None) -> dict:
    results = {}
    no_doc = no_doc_intents(train)
    for strategy in strategies:
        corpus = Corpus.from_file(docs_path, strategy)
        search = LexicalSearch(corpus) if backend == "lexical" else InMemoryDenseSearch(corpus, embed)
        if backend == "hybrid":
            search = HybridSearch(search, LexicalSearch(corpus), len(corpus.chunks),
                                  float(os.getenv("HYBRID_DENSE_WEIGHT", DEFAULT_HYBRID_WEIGHT)))
        rows = score_tickets(search, corpus, train, no_doc=no_doc)
        th = best_threshold(rows)
        entry = {"chunks": len(corpus.chunks),
                 "mean_words": round(float(np.mean([len(c.text.split()) for c in corpus.chunks])), 1),
                 "development": {**metrics(rows, th, truth), **cross_fitted_metrics(rows)}}
        if validation:
            entry["validation"] = metrics(score_tickets(search, corpus, validation, no_doc=no_doc), th, truth)
        results[strategy] = entry

    def key(s):
        d = results[s]["development"]
        return (round(d["cv_j"] or 0, 2), d.get("must_mention_in_top1") or 0, d["mrr"] or 0)
    ranked = sorted(results, key=key, reverse=True)
    return {"backend": backend,
            "ranked_by": "cross-fitted J (right passage returned minus irrelevant returned), "
                         "then must_mention coverage of the top passage, then MRR",
            "recommended": ranked[0], "results": results}


# ------------------------------------------------------------------ building the retriever
def _embedder(model: str):
    from src.ingest import SentenceTransformerEmbedder
    return SentenceTransformerEmbedder(model).embed


def build_retriever(docs_path: Optional[Path] = None, persist_dir: Optional[Path] = None,
                    config_path: Optional[Path] = None, backend: Optional[str] = None,
                    embed: Optional[Callable] = None) -> Retriever:
    """Assemble the production retriever. Always returns something usable: if the dense path
    cannot be built, the retriever runs on lexical search and says so in every result."""
    docs_path = Path(docs_path or os.getenv("DOCS_PATH", DEFAULT_DOCS))
    persist_dir = Path(persist_dir or os.getenv("CHROMA_PATH", DEFAULT_CHROMA))
    config_path = Path(config_path or os.getenv("RETRIEVAL_CONFIG", DEFAULT_CONFIG))
    backend = backend or os.getenv("RETRIEVAL_BACKEND", "hybrid")
    config = _read_json(config_path)
    report = _read_json(persist_dir / "ingest_report.json")
    strategy = report.get("strategy") or config.get("strategy") or os.getenv("CHUNK_STRATEGY", DEFAULT_STRATEGY)
    corpus = Corpus.from_file(docs_path, strategy)

    primary, error = None, ""
    lexical = LexicalSearch(corpus)
    if backend in ("dense", "hybrid"):
        try:
            embed = embed or _embedder(report.get("embedding_model") or os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL))
            primary = ChromaSearch(persist_dir, os.getenv("CHROMA_COLLECTION", DEFAULT_COLLECTION), embed)
            if backend == "hybrid":
                primary = HybridSearch(primary, lexical, len(corpus.chunks),
                                       float(os.getenv("HYBRID_DENSE_WEIGHT", DEFAULT_HYBRID_WEIGHT)))
        except Exception as exc:
            error = f"{backend} unavailable (no dense index): {type(exc).__name__}: {exc}"
            log.warning("%s; retrieval will use lexical search", error)
    retriever = Retriever(corpus, primary=primary, fallback=lexical, thresholds=config.get("thresholds"),
                          primary_error=error)
    if not config.get("thresholds"):
        log.warning("no %s yet: using default thresholds; run `python -m src.retrieve calibrate`", config_path)
    return retriever


def calibrate(retriever: Retriever, tickets: Sequence[NormalisedTicket], config_path: Path) -> dict:
    """Choose a threshold per available backend from labelled tickets and save it."""
    out = {"strategy": retriever.corpus.strategy, "calibrated_on": len(tickets), "thresholds": {}, "metrics": {}}
    backends = [b for b in (retriever.primary, retriever.fallback) if b is not None]
    intent_docs = intent_doc_map(tickets)
    for b in backends:
        rows = score_tickets(b, retriever.corpus, tickets)
        th = best_threshold(rows)
        out.setdefault("ambiguous_excluded", sum(r["group"] == "ambiguous" for r in rows))
        out["thresholds"][b.name] = round(th, 4)
        out["metrics"][b.name] = {**metrics(rows, th), **cross_fitted_metrics(rows)}
        out["thresholds"][f"{b.name}_approved"] = round(approved_threshold_from(rows, th), 4)
        # intent-guided threshold: only the near misses matter, so fit it on tickets below th
        hrows = [r for r in hint_rows(b, retriever.corpus, tickets, intent_docs) if r["top_score"] < th]
        hth = best_threshold(hrows) if hrows else HINT_FACTOR * th
        out["thresholds"][f"{b.name}_hint"] = round(min(hth, th), 4)
        out["metrics"][f"{b.name}_hint"] = {"near_misses": len(hrows),
                                            "right_article": sum(r["top_correct"] for r in hrows),
                                            **cross_fitted_metrics(hrows)}
    retriever.thresholds.update(out["thresholds"])
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(out, indent=2))
    return out


def _read_json(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


# ------------------------------------------------------------------ command line
def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    p = argparse.ArgumentParser(prog="python -m src.retrieve", description="Search the documentation.")
    p.add_argument("--docs", type=Path, default=Path(os.getenv("DOCS_PATH", DEFAULT_DOCS)))
    p.add_argument("--config", type=Path, default=Path(os.getenv("RETRIEVAL_CONFIG", DEFAULT_CONFIG)))
    p.add_argument("--backend", choices=["hybrid", "dense", "lexical"],
                   default=os.getenv("RETRIEVAL_BACKEND", "hybrid"))
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("search", help="search for one query; results come with numbered citations")
    s.add_argument("query")
    s.add_argument("-k", type=int, default=DEFAULT_K)
    s.add_argument("--category", action="append", help="only this category (repeatable)")
    s.add_argument("--product", action="append", help="only articles for this product (repeatable)")
    s.add_argument("--doc-version", action="append", help="only this documentation version (repeatable)")
    s.add_argument("--json", action="store_true", help="print the full result as JSON")
    c = sub.add_parser("calibrate", help="set thresholds from labelled tickets")
    c.add_argument("--tickets", type=Path, default=Path(os.getenv("TRAIN_DATA", DEFAULT_TRAIN)))
    e = sub.add_parser("evaluate", help="hit rate and threshold behaviour on labelled tickets")
    e.add_argument("--tickets", type=Path, required=True)
    cmp = sub.add_parser("compare", help="compare chunking strategies")
    cmp.add_argument("--tickets", type=Path, default=Path(os.getenv("TRAIN_DATA", DEFAULT_TRAIN)))
    cmp.add_argument("--validation", type=Path, default=Path("./data/validation_tickets.json"))
    cmp.add_argument("--truth", type=Path, default=Path("./data/ground_truth_responses.json"))
    cmp.add_argument("--output", type=Path, default=Path("./storage/chunking_comparison.json"))
    # --backend also works after the command (python -m src.retrieve compare --backend dense);
    # SUPPRESS keeps the value given before the command when it is not repeated after it.
    for sp in (s, c, e, cmp):
        sp.add_argument("--backend", choices=["hybrid", "dense", "lexical"], default=argparse.SUPPRESS)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    try:
        if args.command == "compare":
            embed = (_embedder(os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL)) if args.backend in ("dense", "hybrid")
                     else None)
            val = _labelled(load_tickets(args.validation)) if args.validation.is_file() else None
            truth = {g["ticket_id"]: g for g in json.loads(args.truth.read_text())} if args.truth.is_file() else None
            report = compare_strategies(args.docs, _labelled(load_tickets(args.tickets)), args.backend, embed, val,
                                        truth=truth)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2))
            _print_comparison(report)
            return 0
        retriever = build_retriever(args.docs, config_path=args.config, backend=args.backend)
        if args.command == "search":
            filters = {k: v for k, v in (("category", args.category), ("product", args.product),
                                         ("doc_version", args.doc_version)) if v}
            res = cited_results(retriever, args.query, filters=filters or None, k=args.k)
            if args.json:
                print(json.dumps(res, indent=2))
            else:
                print(f"Search ({res['backend']}{', backup mode' if res['degraded'] else ''}): {args.query}")
                for x in res["results"]:
                    print(f"\n{x['citation']} {x['title']} ({x['doc_id']}) · score {x['score']:.2f} · "
                          f"{x['category']} · {x['product']} · v{x['doc_version']}\n    {x['excerpt']}")
                if not res["results"]:
                    print(res["note"] + "".join(f"\n  closest: {c['title']} ({c['doc_id']}, {c['score']:.2f})"
                                                for c in res["closest_below_threshold"]))
                else:
                    print("\nSources:\n" + "\n".join(res["sources"]))
        elif args.command == "calibrate":
            print(json.dumps(calibrate(retriever, _labelled(load_tickets(args.tickets)), args.config), indent=2))
        else:
            tickets = _labelled(load_tickets(args.tickets))
            out = {}
            for b in [x for x in (retriever.primary, retriever.fallback) if x is not None]:
                rows = score_tickets(b, retriever.corpus, tickets)
                out[b.name] = metrics(rows, retriever.thresholds.get(b.name, FALLBACK_THRESHOLDS.get(b.name, 0.4)))
            print(json.dumps(out, indent=2))
    except Exception as exc:
        log.error("%s failed: %s", args.command, exc)
        return 1
    return 0


def _print_comparison(report: dict) -> None:
    print(f"Chunking comparison ({report['backend']} backend)\nranked by {report['ranked_by']}\n")
    cols = [("hit@1", "hit_at_1"), ("MRR", "mrr"), ("AUC", "separation_auc"), ("cv J", "cv_j"),
            ("mm@1", "must_mention_in_top1"), ("mm@3", "must_mention_in_top3")]
    print(f"{'strategy':<10}{'chunks':>7}{'words':>7}" + "".join(f"{c:>7}" for c, _ in cols)
          + f"{'val hit':>9}{'val nodoc':>10}")
    for name, r in report["results"].items():
        d, v = r["development"], r.get("validation", {})
        print(f"{name:<10}{r['chunks']:>7}{r['mean_words']:>7}" + "".join(f"{str(d.get(k)):>7}" for _, k in cols)
              + f"{str(v.get('retrieval_hit_rate', '')):>9}{str(v.get('returned_for_no_doc', '')):>10}")
    print(f"\nrecommended: {report['recommended']}  (full figures in the JSON output)")


if __name__ == "__main__":
    sys.exit(main())
