"""Tests for src/classify.py, one block per Build Specification bullet for Classify."""
import json
from pathlib import Path

import numpy as np
import pytest

from src import classify as C
from src.ingest import load_tickets, normalise_ticket

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def dev():
    return load_tickets(DATA / "development_tickets.json")


@pytest.fixture(scope="module")
def val():
    return load_tickets(DATA / "validation_tickets.json")


@pytest.fixture(scope="module")
def clf(dev):
    return C.Classifier().fit(dev)


ODD_INPUTS = [None, 42, "", {}, {"body": "   "}, {"channel": "chat", "body": "​"},
              {"channel": "sms", "body": "asdkjh qwe zzz"}, "plain text: my webhook stopped arriving"]


# ---------------------------------------------------------------- 1. intent and urgency for every ticket
def test_req1_every_real_ticket_gets_a_known_intent_and_urgency(clf, dev, val):
    for t in dev + val:
        c = clf.classify(t)
        assert c.intent in clf.intents and c.urgency in C.URGENCIES
        assert not c.is_fallback


@pytest.mark.parametrize("raw", ODD_INPUTS)
def test_req1_even_odd_input_gets_an_intent_and_urgency(clf, raw):
    c = clf.classify(raw)
    assert c.intent in clf.intents + [C.FALLBACK_INTENT]
    assert c.urgency in C.URGENCIES


# ---------------------------------------------------------------- 2. confidence reflects probability of being right
def test_req2_confidence_is_a_probability_and_never_certainty(clf, val):
    for t in val + [normalise_ticket(r) for r in ODD_INPUTS]:
        c = clf.classify(t)
        for v in (c.intent_confidence, c.urgency_confidence):
            assert 0.0 <= v <= C.MAX_CONFIDENCE


def test_req2_intent_confidence_calibrated_on_validation(clf, val):
    report = C.evaluate(clf, val)
    assert report["intent"]["ece"] <= 0.05          # Brief: within five points of observed accuracy


def test_req2_intent_confidence_on_unseen_wording(dev):
    """Hold out whole groups of near-duplicates (four different quarters), train on the rest, and
    compare stated confidence with accuracy on that genuinely new wording.

    Measured when written: gaps of roughly -0.09 to +0.16 by split, mean about +0.07 (too high).
    This test guards against it getting worse; the limit itself is documented and reported."""
    texts = [t.body or t.text for t in dev]
    groups = C.near_duplicate_groups(texts, [t.eval_labels["intent"] for t in dev])
    gaps, accs = [], []
    for offset in range(4):
        held = set(np.unique(groups)[offset::4])
        train = [t for t, g in zip(dev, groups) if g not in held]
        test = [t for t, g in zip(dev, groups) if g in held]
        intent = C.evaluate(C.Classifier().fit(train), test)["intent"]
        gaps.append(intent["mean_confidence"] - intent["accuracy"])
        accs.append(intent["accuracy"])
    assert abs(np.mean(gaps)) <= 0.10, gaps
    assert max(abs(g) for g in gaps) <= 0.20, gaps
    assert np.mean(accs) < 0.9          # sanity: this really is the harder regime


def test_req2_unfamiliar_text_gets_lower_confidence(clf, dev):
    familiar = clf.classify(dev[0])
    strange = clf.classify({"ticket_id": "S", "channel": "email", "body": "Purple elephants juggle quietly."})
    assert strange.familiarity < familiar.familiarity
    assert strange.intent_confidence < familiar.intent_confidence


# ---------------------------------------------------------------- 3. defined fallback, never an exception
@pytest.mark.parametrize("raw,reason", [
    (None, "no_usable_text"), ({}, "no_usable_text"), ({"ticket_id": "E", "body": ""}, "no_usable_text"),
])
def test_req3_unusable_tickets_get_the_fallback(clf, raw, reason):
    c = clf.classify(raw)
    assert c.is_fallback and c.fallback_reason == reason
    assert (c.intent, c.urgency) == (C.FALLBACK_INTENT, C.FALLBACK_URGENCY)
    assert c.intent_confidence == 0.0


def test_req3_untrained_classifier_falls_back(dev):
    c = C.Classifier().classify(dev[0])
    assert c.is_fallback and c.fallback_reason == "model_unavailable" and c.ticket_id == dev[0].ticket_id


def test_req3_internal_error_falls_back(clf, dev, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("simulated")
    monkeypatch.setattr(clf, "_classify_text", boom)
    c = clf.classify(dev[0])
    assert c.is_fallback and c.fallback_reason == "classifier_error:RuntimeError"


def test_req3_fallback_intent_is_not_a_real_intent(clf):
    assert C.FALLBACK_INTENT not in clf.intents


def test_req3_corrupt_model_file_is_rebuilt(tmp_path):
    bad = tmp_path / "m.joblib"
    bad.write_bytes(b"not a model")
    clf = C.load_or_train(bad, DATA / "development_tickets.json")
    assert clf.is_ready and C.Classifier.load(bad).version == clf.version


def test_req3_no_model_and_no_training_data_still_classifies(tmp_path, dev):
    clf = C.load_or_train(tmp_path / "none.joblib", tmp_path / "missing.json")
    c = clf.classify(dev[0])
    assert c.is_fallback and c.fallback_reason == "model_unavailable"


def test_req3_training_on_too_little_data_raises_clearly():
    with pytest.raises(ValueError, match="at least"):
        C.Classifier().fit([])


# ---------------------------------------------------------------- 4. records the alternatives considered
def test_req4_intent_alternatives_recorded(clf, val):
    for t in val:
        c = clf.classify(t)
        alts = c.intent_alternatives
        assert len(alts) == C.N_ALTERNATIVES
        assert alts[0].label == c.intent and alts[0].probability == c.intent_confidence
        assert len({a.label for a in alts}) == len(alts)
        assert all(a.probability >= b.probability for a, b in zip(alts[1:], alts[2:]))
        assert sum(a.probability for a in alts) <= 1.0 + 1e-6


def test_req4_urgency_alternatives_cover_all_levels(clf, val):
    for t in val:
        c = clf.classify(t)
        assert [a.label for a in c.urgency_alternatives][0] == c.urgency
        assert {a.label for a in c.urgency_alternatives} == set(C.URGENCIES)
        assert sum(a.probability for a in c.urgency_alternatives) == pytest.approx(1.0, abs=1e-3)


def test_req4_fallback_also_records_alternatives(clf):
    c = clf.classify(None)
    assert c.intent_alternatives and len(c.urgency_alternatives) == 3


# ---------------------------------------------------------------- determinism, persistence, separation, CLI
def test_same_ticket_same_result(clf, val):
    a, b = clf.classify(val[0]), clf.classify(val[0])
    assert a.model_dump(exclude={"latency_ms"}) == b.model_dump(exclude={"latency_ms"})


def test_training_is_deterministic(dev, clf):
    assert C.Classifier().fit(dev).version == clf.version


def test_save_load_round_trip(clf, val, tmp_path):
    clf.save(tmp_path / "m.joblib")
    other = C.Classifier.load(tmp_path / "m.joblib")
    for t in val[:10]:
        a = other.classify(t).model_dump(exclude={"latency_ms"})
        assert a == clf.classify(t).model_dump(exclude={"latency_ms"})


def test_prediction_never_reads_labels(clf, val):
    t = val[0]
    tampered = t.model_copy(update={"eval_labels": {"intent": "feature_request", "urgency": "low"}})
    a = clf.classify(tampered).model_dump(exclude={"latency_ms"})
    assert a == clf.classify(t).model_dump(exclude={"latency_ms"})


def test_urgency_does_not_depend_on_customer_segment(clf, val):
    t = val[0]
    other = t.model_copy(update={"customer_tier": "enterprise" if t.customer_tier.value != "enterprise" else "standard",
                                 "customer_region": "latin_america", "language_fluency": "non_fluent"})
    a, b = clf.classify(t), clf.classify(other)
    assert (a.intent, a.urgency, a.intent_confidence) == (b.intent, b.urgency, b.intent_confidence)


def test_cli_train_evaluate_predict(tmp_path, capsys):
    model = str(tmp_path / "m.joblib")
    assert C.main(["--model", model, "train", "--input", str(DATA / "development_tickets.json")]) == 0
    assert C.main(["--model", model, "evaluate", "--input", str(DATA / "validation_tickets.json"),
                   "--output", str(tmp_path / "r.json")]) == 0
    assert json.loads((tmp_path / "r.json").read_text())["labelled"] == 80
    assert C.main(["--model", model, "predict", "--input", str(DATA / "validation_tickets.json")]) == 0
    assert C.main(["--model", model, "evaluate", "--input", str(tmp_path / "missing.json")]) == 1
    capsys.readouterr()
