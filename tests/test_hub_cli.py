"""python -m src.hub: start the live queue over, and load real tickets from a file into it."""
import json
from pathlib import Path

import pytest

from src import hub as H

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture
def stores(tmp_path, monkeypatch):
    for k, v in dict(RETRIEVAL_BACKEND="lexical", GENERATOR="extractive", OPENROUTER_API_KEY="",
                     DATABASE_URL=f"sqlite:///{tmp_path / 'decisions.db'}", KILL_SWITCH_FILE=tmp_path / "KS",
                     REPLY_MODE="review").items():
        monkeypatch.setenv(k, str(v))
    monkeypatch.delenv("TICKETS_DB", raising=False)
    monkeypatch.delenv("REVIEW_DB", raising=False)
    return tmp_path


def queue_size(tmp_path):
    from src.api import SupportPipeline
    return len(H.SupportHub(SupportPipeline.from_env()).tickets.list())


def test_load_brings_tickets_in_once(stores, capsys):
    assert H.main(["load", str(DATA / "validation_tickets.json"), "--count", "5"]) == 0
    assert "Added 5 ticket(s)" in capsys.readouterr().out
    assert queue_size(stores) == 5
    assert H.main(["load", str(DATA / "validation_tickets.json"), "--count", "5"]) == 0       # same 5 again
    assert "Added 0 ticket(s) to the queue; 5 were already there" in capsys.readouterr().out
    assert H.main(["load", str(DATA / "validation_tickets.json"), "--skip", "5", "--count", "3"]) == 0
    assert queue_size(stores) == 8


def test_load_reads_json_lines_and_refuses_a_missing_file(stores, tmp_path):
    rows = json.loads((DATA / "validation_tickets.json").read_text())[:2]
    f = tmp_path / "two.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in rows))
    assert H.main(["load", str(f)]) == 0 and queue_size(stores) == 2
    assert H.main(["load", str(tmp_path / "nope.json")]) == 1


def test_reset_needs_confirmation_and_empties_the_queue(stores, monkeypatch, capsys):
    H.main(["load", str(DATA / "validation_tickets.json"), "--count", "3"])
    monkeypatch.setattr("builtins.input", lambda prompt="": "no")
    assert H.main(["reset"]) == 1 and queue_size(stores) == 3
    monkeypatch.setattr("builtins.input", lambda prompt="": "reset")
    assert H.main(["reset"]) == 0
    assert not (stores / "tickets.db").exists() and not (stores / "decisions.db").exists()
    assert queue_size(stores) == 0


def test_each_sample_batch_is_new(stores):
    from src.api import SupportPipeline
    hub = H.SupportHub(SupportPipeline.from_env())
    first = [r for r in hub.simulate(count=4, seed=1) if not r.get("duplicate_delivery")]
    again = [r for r in hub.simulate(count=4, seed=1) if not r.get("duplicate_delivery")]
    other = [r for r in hub.simulate(count=4, seed=2) if not r.get("duplicate_delivery")]
    assert first and not again and other
