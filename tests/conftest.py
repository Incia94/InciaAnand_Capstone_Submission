"""Shared test settings.

REPLY_MODE: the service holds every reply for a person to approve by default (src/review.py).
Most tests check what the pipeline decides and sends, so they run with REPLY_MODE=auto; tests
marked review_mode run with the default, review. The review queue always goes to a temporary file.
"""
import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "review_mode: run with REPLY_MODE=review (replies held for approval)")


@pytest.fixture(autouse=True)
def _reply_mode(monkeypatch, tmp_path_factory, request):
    if request.node.get_closest_marker("review_mode") is None:
        monkeypatch.setenv("REPLY_MODE", "auto")
    else:
        monkeypatch.delenv("REPLY_MODE", raising=False)
    if "REVIEW_DB" not in __import__("os").environ:
        monkeypatch.setenv("REVIEW_DB", str(tmp_path_factory.getbasetemp() / "reviews.db"))
