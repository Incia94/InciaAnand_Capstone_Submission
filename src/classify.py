"""Classify: predict intent and urgency for a NormalisedTicket, with calibrated confidence.

Build Specification (Classify) -> where it is met                      -> tests (tests/test_classify.py)
----------------------------------------------------------------------------------------------------------
Assigns an intent and an urgency to every ticket   Classifier.classify always returns both  test_req1_*
Confidence reflects the probability of being right  familiarity-aware calibration (below)    test_req2_*
Defined fallback instead of an exception            FALLBACK_INTENT / FALLBACK_URGENCY       test_req3_*
Records the alternatives it considered              intent_alternatives, urgency_alternatives test_req4_*

Usage (run from the project root)
-----
    python -m src.classify train    --input data/development_tickets.json
    python -m src.classify evaluate --input data/validation_tickets.json
    python -m src.classify predict  --input data/validation_tickets.json --sample

The model file defaults to CLASSIFIER_PATH (./storage/models/classifier.joblib). If it is missing
or unreadable, `load_or_train` rebuilds it from TRAIN_DATA (./data/development_tickets.json), so a
clean checkout works without a manual training step.

How it works
------------
Intent: TF-IDF features on the ticket text (subject + body), words (1-2 grams) and characters
(2-5 grams within words, which copes with misspellings and non-fluent phrasing), into a
multinomial logistic regression. Local, free, deterministic and fast; no model provider involved.

Confidence (intent): a classifier's raw top probability is not the probability of being right.
On this data the gap depends mostly on whether the wording resembles anything seen in training:
reworded copies of a known message are almost always right, genuinely new wording much less
often. So confidence = P(correct | top probability, familiarity, familiarity to the predicted intent's
own examples, margin over the runner-up), where familiarity is the highest cosine similarity
between the ticket and any training ticket. That small calibration model (a logistic regression)
is fitted on out-of-fold predictions from two cross-validations: one that holds out whole groups
of near-duplicate tickets (new wording) and one ordinary stratified split (familiar wording).

Known limit: on familiar wording confidence tracks accuracy closely. On genuinely new wording it
is too high by about 7 points on average, and the gap varies by template (see
test_req2_intent_confidence_on_unseen_wording). Report it; routing should treat low familiarity
as a reason for caution.

Urgency: the urgency label varies a lot within identical ticket text, so predicting it from words
overfits. Urgency is derived from the intent distribution: P(urgency | ticket) =
sum over intents of P(intent | ticket) * P(urgency | intent), with the second term counted on
training data (add-one smoothed), then temperature-scaled on out-of-fold predictions. It never
uses customer tier or region.

The pipeline never reads `eval_labels` at prediction time; only `fit` uses them, as training labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from src.ingest import NormalisedTicket, load_tickets, normalise_ticket

log = logging.getLogger("classify")

URGENCIES = ("high", "medium", "low")
FALLBACK_INTENT = "unclassified"   # not one of the 22 intents, so routing can never mistake it for one
FALLBACK_URGENCY = "medium"         # neutral: a fallback ticket is escalated anyway, and should not jump
#                                     the queue just because it could not be read
N_ALTERNATIVES = 5
MAX_CONFIDENCE = 0.99               # never claim certainty: a 1.0 would say the system cannot be wrong
DEFAULT_MODEL_PATH = "./storage/models/classifier.joblib"
DEFAULT_TRAIN_DATA = "./data/development_tickets.json"
MODEL_FORMAT = 2                    # bump when the saved structure changes
RANDOM_STATE = 0


# ------------------------------------------------------------------ output
class Alternative(BaseModel):
    label: str
    probability: float


class Classification(BaseModel):
    # `model_version` is our field, not Pydantic's: allow the model_ prefix (older Pydantic warns).
    model_config = ConfigDict(protected_namespaces=())

    ticket_id: str
    intent: str
    intent_confidence: float = Field(ge=0, le=1)
    urgency: str
    urgency_confidence: float = Field(ge=0, le=1)
    intent_alternatives: list[Alternative] = []   # best first, chosen intent included
    urgency_alternatives: list[Alternative] = []  # all three levels, best first
    familiarity: float = 0.0                      # 0-1: closeness to anything seen in training
    is_fallback: bool = False
    fallback_reason: Optional[str] = None
    model_version: str = ""
    latency_ms: float = 0.0

    @property
    def confidence(self) -> float:
        """The figure routing thresholds on: confidence in the intent."""
        return self.intent_confidence


def fallback(ticket_id: str, reason: str, model_version: str = "") -> Classification:
    """The defined result when a ticket cannot be classified. Never raises."""
    return Classification(
        ticket_id=ticket_id or "unknown", intent=FALLBACK_INTENT, intent_confidence=0.0,
        urgency=FALLBACK_URGENCY, urgency_confidence=0.0,
        intent_alternatives=[Alternative(label=FALLBACK_INTENT, probability=1.0)],
        urgency_alternatives=[Alternative(label=u, probability=round(1 / 3, 4)) for u in URGENCIES],
        is_fallback=True, fallback_reason=reason, model_version=model_version)


# ------------------------------------------------------------------ helpers
def _vectoriser():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import make_union
    return make_union(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, lowercase=True),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, lowercase=True),
    )


def _intent_model(C: float):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    return make_pipeline(_vectoriser(), LogisticRegression(C=C, max_iter=5000))


def near_duplicate_groups(texts: Sequence[str], labels: Sequence[str], threshold: float = 0.5) -> np.ndarray:
    """Group tickets that are rewordings of one message (same label, cosine >= threshold), so
    cross-validation can hold a whole group out and measure performance on new wording."""
    from scipy.sparse.csgraph import connected_components
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    m = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit_transform(texts)
    lab = np.asarray(labels)
    adjacency = (cosine_similarity(m) >= threshold) & (lab[:, None] == lab[None, :])
    return connected_components(adjacency, directed=False)[1]


def calibration_features(proba: np.ndarray, sims: np.ndarray, train_labels: np.ndarray,
                         intents: Sequence[str]) -> np.ndarray:
    """What the confidence model looks at, per ticket:
    log of the top probability; familiarity (highest similarity to any training ticket);
    familiarity to training tickets of the predicted intent; margin over the runner-up."""
    top = proba.argmax(axis=1)
    pred = np.asarray(intents)[top]
    fam = sims.max(axis=1) if sims.shape[1] else np.zeros(len(proba))
    fam_pred = np.array([sims[i, train_labels == pred[i]].max() if (train_labels == pred[i]).any() else 0.0
                         for i in range(len(proba))])
    srt = np.sort(proba, axis=1)
    margin = srt[:, -1] - (srt[:, -2] if proba.shape[1] > 1 else 0.0)
    return np.c_[np.log(np.clip(srt[:, -1], 1e-12, 1.0)), fam, fam_pred, margin]


def _apply_temperature(p: np.ndarray, t: float) -> np.ndarray:
    logits = np.log(np.clip(p, 1e-12, 1.0)) / t
    logits -= logits.max(axis=-1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=-1, keepdims=True)


def _fit_temperature(p: np.ndarray, y_idx: np.ndarray) -> float:
    from scipy.optimize import minimize_scalar
    rows = np.arange(len(y_idx))
    res = minimize_scalar(lambda t: -np.log(_apply_temperature(p, t)[rows, y_idx] + 1e-12).mean(),
                          bounds=(0.05, 20.0), method="bounded")
    return float(res.x)


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    """Average gap between stated confidence and observed accuracy, weighted by bin size."""
    confidence, correct = np.asarray(confidence, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi) if lo > 0 else (confidence >= lo) & (confidence <= hi)
        if mask.any():
            total += mask.mean() * abs(confidence[mask].mean() - correct[mask].mean())
    return float(total)


def reliability_table(confidence: np.ndarray, correct: np.ndarray, bins: int = 5) -> list[dict]:
    confidence, correct = np.asarray(confidence, float), np.asarray(correct, float)
    rows, edges = [], np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence >= lo) & ((confidence < hi) if hi < 1 else (confidence <= hi))
        if mask.any():
            rows.append({"confidence_bin": f"{lo:.1f}-{hi:.1f}", "tickets": int(mask.sum()),
                         "mean_confidence": round(float(confidence[mask].mean()), 3),
                         "accuracy": round(float(correct[mask].mean()), 3)})
    return rows


def _as_ticket(ticket: Any) -> NormalisedTicket:
    return ticket if isinstance(ticket, NormalisedTicket) else normalise_ticket(ticket)


# ------------------------------------------------------------------ the classifier
class Classifier:
    def __init__(self, C: float = 30.0, n_folds: int = 5):
        self.C, self.n_folds = C, n_folds
        self.intent_model = None
        self.calibrator = None              # P(correct | log top prob, familiarity)
        self.urgency_given_intent = None    # array [n_intents, 3]
        self.urgency_temperature = 1.0
        self.train_matrix = None            # feature vectors of training tickets, for familiarity
        self.train_labels = None
        self.intents: list[str] = []
        self.version = ""
        self.training_report: dict = {}

    @property
    def is_ready(self) -> bool:
        return self.intent_model is not None and self.calibrator is not None

    # ---------------- training
    def fit(self, tickets: Iterable[Any]) -> "Classifier":
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.model_selection import GroupKFold, StratifiedKFold

        started = time.time()
        rows = [_as_ticket(t) for t in tickets]
        rows = [t for t in rows if t.processable and t.eval_labels and t.eval_labels.get("intent")]
        if len(rows) < 2 * self.n_folds:
            raise ValueError(f"need at least {2 * self.n_folds} labelled tickets to train, got {len(rows)}")
        X = np.array([t.text for t in rows], dtype=object)
        y = np.array([t.eval_labels["intent"] for t in rows])
        u = np.array([t.eval_labels.get("urgency", FALLBACK_URGENCY) for t in rows])
        self.intents = sorted(set(y))
        groups = near_duplicate_groups([t.body or t.text for t in rows], y)

        # out-of-fold predictions under two regimes: new wording (grouped) and familiar wording
        cal_x, cal_y, urg_p, urg_y, oof_report = [], [], [], [], {}
        splitters = {
            "new_wording": GroupKFold(n_splits=self.n_folds).split(X, y, groups),
            "familiar_wording": StratifiedKFold(self.n_folds, shuffle=True,
                                                random_state=RANDOM_STATE).split(X, y),
        }
        for regime, splits in splitters.items():
            correct_all = []
            for tr, te in splits:
                m = _intent_model(self.C).fit(X[tr], y[tr])
                p = self._full_proba(m, X[te])
                sims = cosine_similarity(m[0].transform(X[te]), m[0].transform(X[tr]))
                correct = np.array(self.intents)[p.argmax(axis=1)] == y[te]
                cal_x.append(calibration_features(p, sims, y[tr], self.intents))
                cal_y.append(correct)
                correct_all.append(correct)
                ugi = self._urgency_table(y[tr], u[tr])
                urg_p.append(p @ ugi)
                urg_y.append(u[te])
            oof_report[regime] = round(float(np.concatenate(correct_all).mean()), 3)

        cal_x, cal_y = np.vstack(cal_x), np.concatenate(cal_y)
        self.calibrator = LogisticRegression().fit(cal_x, cal_y) if len(set(cal_y)) > 1 else None
        urg_p, urg_y = np.vstack(urg_p), np.concatenate(urg_y)
        self.urgency_temperature = _fit_temperature(urg_p, np.array([URGENCIES.index(v) for v in urg_y]))
        if self.calibrator is None:  # every out-of-fold prediction right (or wrong): a constant
            self.calibrator = _ConstantCalibrator(float(cal_y.mean()))

        # final models on all the data
        self.intent_model = _intent_model(self.C).fit(X, y)
        self.train_matrix = self.intent_model[0].transform(X)
        self.train_labels = y
        self.urgency_given_intent = self._urgency_table(y, u)
        digest = hashlib.sha256("\n".join(sorted(f"{a}|{b}|{c}" for a, b, c in zip(X, y, u))).encode()).hexdigest()
        self.version = f"tfidf-lr-C{self.C:g}-{digest[:8]}"

        cal_conf = self.calibrator.predict_proba(cal_x)[:, 1]
        self.training_report = {
            "trained_on": len(rows), "intents": len(self.intents), "near_duplicate_groups": int(groups.max() + 1),
            "oof_intent_accuracy": oof_report,
            "oof_intent_ece_after_calibration": round(expected_calibration_error(cal_conf, cal_y), 3),
            "urgency_temperature": round(self.urgency_temperature, 3),
            "seconds": round(time.time() - started, 1), "model_version": self.version,
        }
        return self

    def _full_proba(self, model, texts) -> np.ndarray:
        """Probabilities over self.intents, with zeros for intents missing from a training fold."""
        p = model.predict_proba(texts)
        out = np.zeros((p.shape[0], len(self.intents)))
        for j, c in enumerate(model.classes_):
            out[:, self.intents.index(c)] = p[:, j]
        return out

    def _urgency_table(self, intents: np.ndarray, urgencies: np.ndarray) -> np.ndarray:
        table = np.ones((len(self.intents), len(URGENCIES)))  # add-one smoothing
        for i, v in zip(intents, urgencies):
            if v in URGENCIES:
                table[self.intents.index(i), URGENCIES.index(v)] += 1
        return table / table.sum(axis=1, keepdims=True)

    # ---------------- prediction
    def classify(self, ticket: Any) -> Classification:
        """Classify one ticket. Never raises: anything unexpected returns the defined fallback."""
        started = time.perf_counter()
        ticket_id = ""
        try:
            t = _as_ticket(ticket)
            ticket_id = t.ticket_id
            if not self.is_ready:
                return fallback(ticket_id, "model_unavailable", self.version)
            if not t.processable or not t.text.strip():
                return fallback(ticket_id, "no_usable_text", self.version)
            result = self._classify_text(ticket_id, t.text)
        except Exception as exc:  # last line of defence; the run must continue
            log.exception("classification failed for %s", ticket_id or "<unknown>")
            return fallback(ticket_id, f"classifier_error:{type(exc).__name__}", self.version)
        result.latency_ms = round((time.perf_counter() - started) * 1000, 2)
        return result

    def classify_many(self, tickets: Iterable[Any]) -> list[Classification]:
        return [self.classify(t) for t in tickets]

    def _classify_text(self, ticket_id: str, text: str) -> Classification:
        from sklearn.metrics.pairwise import cosine_similarity
        p = self._full_proba(self.intent_model, [text])[0]
        vec = self.intent_model[0].transform([text])
        sims = cosine_similarity(vec, self.train_matrix) if vec.nnz else np.zeros((1, self.train_matrix.shape[0]))
        familiarity = float(sims.max())
        order = np.argsort(-p, kind="stable")
        top = int(order[0])
        features = calibration_features(p[None, :], sims, self.train_labels, self.intents)
        confidence = float(self.calibrator.predict_proba(features)[0, 1])
        confidence = min(confidence, MAX_CONFIDENCE)
        # Alternatives: the chosen intent carries the calibrated confidence; the remaining
        # probability is shared among the others in proportion to the model's scores.
        rest = max(1.0 - p[top], 1e-12)
        intent_alts = [Alternative(label=self.intents[top], probability=round(confidence, 4))]
        intent_alts += [Alternative(label=self.intents[j], probability=round((1 - confidence) * p[j] / rest, 4))
                        for j in order[1:N_ALTERNATIVES]]

        pu = _apply_temperature((p @ self.urgency_given_intent)[None, :], self.urgency_temperature)[0]
        if pu.max() > MAX_CONFIDENCE:
            pu = pu * MAX_CONFIDENCE / pu.max()
            pu[np.argsort(pu)[:-1]] += (1 - pu.sum()) / (len(pu) - 1)
        u_order = np.argsort(-pu, kind="stable")
        urgency_alts = [Alternative(label=URGENCIES[j], probability=round(float(pu[j]), 4)) for j in u_order]
        return Classification(
            ticket_id=ticket_id, intent=self.intents[top], intent_confidence=round(confidence, 4),
            urgency=URGENCIES[int(u_order[0])], urgency_confidence=round(float(pu[u_order[0]]), 4),
            intent_alternatives=intent_alts, urgency_alternatives=urgency_alts,
            familiarity=round(familiarity, 4), model_version=self.version)

    # ---------------- persistence
    def save(self, path: Path) -> None:
        import joblib
        import sklearn
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"format": MODEL_FORMAT, "sklearn": sklearn.__version__, "classifier": self}, path)

    @classmethod
    def load(cls, path: Path) -> "Classifier":
        import joblib
        import sklearn
        blob = joblib.load(Path(path))
        if blob.get("format") != MODEL_FORMAT or blob.get("sklearn") != sklearn.__version__:
            raise ValueError(f"model file was saved by an incompatible version "
                             f"(format {blob.get('format')}, scikit-learn {blob.get('sklearn')})")
        return blob["classifier"]


class _ConstantCalibrator:
    """Used only if every out-of-fold prediction was right (or wrong), so no curve can be fitted."""

    def __init__(self, rate: float):
        self.rate = rate

    def predict_proba(self, x):
        n = len(x)
        return np.c_[np.full(n, 1 - self.rate), np.full(n, self.rate)]


# ------------------------------------------------------------------ loading for the pipeline
def load_or_train(model_path: Optional[Path] = None, train_path: Optional[Path] = None) -> Classifier:
    """Load the saved model; if missing or incompatible, train from the development set and save.
    If even that fails, return an untrained Classifier, whose every answer is the fallback."""
    model_path = Path(model_path or os.getenv("CLASSIFIER_PATH", DEFAULT_MODEL_PATH))
    train_path = Path(train_path or os.getenv("TRAIN_DATA", DEFAULT_TRAIN_DATA))
    if model_path.is_file():
        try:
            return Classifier.load(model_path)
        except Exception as exc:
            log.warning("could not load %s (%s); retraining", model_path, exc)
    try:
        clf = Classifier().fit(load_tickets(train_path))
        clf.save(model_path)
        log.info("trained classifier %s from %s", clf.version, train_path)
        return clf
    except Exception as exc:
        log.error("no classifier available (%s); every ticket will get the fallback", exc)
        return Classifier()


# ------------------------------------------------------------------ evaluation
def evaluate(clf: Classifier, tickets: Iterable[Any]) -> dict:
    """Accuracy, per-class precision/recall and calibration against labelled tickets."""
    from sklearn.metrics import accuracy_score, classification_report, f1_score
    rows = [_as_ticket(t) for t in tickets]
    labelled = [t for t in rows if t.eval_labels and t.eval_labels.get("intent")]
    preds = [clf.classify(t) for t in labelled]
    y = [t.eval_labels["intent"] for t in labelled]
    yu = [t.eval_labels.get("urgency") for t in labelled]
    pi = [c.intent for c in preds]
    pu = [c.urgency for c in preds]
    ic = np.array([c.intent_confidence for c in preds])
    uc = np.array([c.urgency_confidence for c in preds])
    i_ok = np.array(pi) == np.array(y)
    u_ok = np.array(pu) == np.array(yu)
    per_class = classification_report(y, pi, output_dict=True, zero_division=0)
    return {
        "tickets": len(rows), "labelled": len(labelled), "fallbacks": sum(c.is_fallback for c in preds),
        "model_version": clf.version,
        "intent": {
            "accuracy": round(accuracy_score(y, pi), 3) if y else None,
            "macro_f1": round(f1_score(y, pi, average="macro", zero_division=0), 3) if y else None,
            "mean_confidence": round(float(ic.mean()), 3) if len(ic) else None,
            "calibration_gap": round(float(abs(ic.mean() - i_ok.mean())), 3) if len(ic) else None,
            "ece": round(expected_calibration_error(ic, i_ok), 3) if len(ic) else None,
            "reliability": reliability_table(ic, i_ok),
            "per_class": {k: {"precision": round(v["precision"], 3), "recall": round(v["recall"], 3),
                              "support": int(v["support"])}
                          for k, v in per_class.items() if k not in ("accuracy", "macro avg", "weighted avg")},
        },
        "urgency": {
            "accuracy": round(float(u_ok.mean()), 3) if len(u_ok) else None,
            "majority_baseline": round(Counter(yu).most_common(1)[0][1] / len(yu), 3) if yu else None,
            "mean_confidence": round(float(uc.mean()), 3) if len(uc) else None,
            "ece": round(expected_calibration_error(uc, u_ok), 3) if len(uc) else None,
            "reliability": reliability_table(uc, u_ok),
        },
        "mean_familiarity": round(float(np.mean([c.familiarity for c in preds])), 3) if preds else None,
    }


# ------------------------------------------------------------------ command line
def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    p = argparse.ArgumentParser(prog="python -m src.classify", description="Intent and urgency classifier.")
    p.add_argument("--model", type=Path, default=Path(os.getenv("CLASSIFIER_PATH", DEFAULT_MODEL_PATH)))
    sub = p.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train", help="train on labelled tickets and save the model")
    t.add_argument("--input", type=Path, default=Path(os.getenv("TRAIN_DATA", DEFAULT_TRAIN_DATA)))
    e = sub.add_parser("evaluate", help="measure accuracy and calibration on labelled tickets")
    e.add_argument("--input", type=Path, required=True)
    e.add_argument("--output", type=Path, help="also write the report here as JSON")
    pr = sub.add_parser("predict", help="classify tickets")
    pr.add_argument("--input", type=Path, required=True)
    pr.add_argument("--sample", action="store_true", help="print one result per channel")
    pr.add_argument("--output", type=Path, help="write all results as JSON Lines")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    try:
        if args.command == "train":
            clf = Classifier().fit(load_tickets(args.input))
            clf.save(args.model)
            print(json.dumps(clf.training_report, indent=2))
            return 0
        tickets = load_tickets(args.input)
    except (OSError, ValueError) as exc:
        log.error("%s failed: %s", args.command, exc)
        return 1
    clf = load_or_train(args.model)
    if args.command == "evaluate":
        report = evaluate(clf, tickets)
        text = json.dumps(report, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text)
        print(text)
        return 0
    results = [(tk, clf.classify(tk)) for tk in tickets]
    if args.sample:
        seen = set()
        for tk, c in results:
            if tk.channel not in seen:
                seen.add(tk.channel)
                print(f"[{tk.channel.value}] {tk.text[:100]!r}\n{c.model_dump_json(indent=2)}\n")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as fh:
            for _, c in results:
                fh.write(c.model_dump_json() + "\n")
    print(json.dumps({"classified": len(results), "fallbacks": sum(c.is_fallback for _, c in results),
                      "intents": dict(Counter(c.intent for _, c in results).most_common())}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
