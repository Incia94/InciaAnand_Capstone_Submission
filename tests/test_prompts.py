"""Tests for the prompt library: the .txt files in prompts/build and prompts/evaluation, the
register in prompts/README.md, and load_prompt() which the src files use to read them."""
import re
from pathlib import Path

import pytest

from evaluation.judge import JUDGE_PROMPT
from src import generate as G

ROOT = Path(__file__).resolve().parents[1] / "prompts"
PINNED = {  # every prompt the code loads, as pinned in the file that loads it
    ("build", *G.SYSTEM_PROMPT), ("build", *G.USER_PROMPT), ("build", *G.REPAIR_PROMPT),
    (JUDGE_PROMPT[2], JUDGE_PROMPT[0], JUDGE_PROMPT[1]),
}
FILES = {(f.parent.name, f.name) for kind in ("build", "evaluation") for f in (ROOT / kind).glob("*.txt")}


def test_library_has_both_folders():
    assert (ROOT / "build").is_dir() and (ROOT / "evaluation").is_dir()


def test_every_pinned_prompt_has_a_file():
    for kind, name, number in PINNED:
        assert (kind, f"{name}.v{number}.txt") in FILES, (kind, name, number)


def test_every_prompt_file_is_loaded_by_code():
    loaded = {(kind, f"{name}.v{number}.txt") for kind, name, number in PINNED}
    assert FILES <= loaded, f"files no code loads: {sorted(FILES - loaded)}"


def test_file_names_follow_the_versioning_convention():
    for _, name in FILES:
        assert re.fullmatch(r"[a-z0-9_]+\.v\d+\.txt", name), name


def test_every_prompt_file_has_a_row_in_the_register():
    register = (ROOT / "README.md").read_text()
    for kind, name in FILES:
        assert f"{kind}/{name}" in register, f"{kind}/{name} missing from prompts/README.md"


def test_load_prompt_reads_the_file_and_versions_it():
    p = G.load_prompt(*G.SYSTEM_PROMPT)
    assert p.text == (ROOT / "build" / "generate_answer.v1.txt").read_text().strip()
    assert re.fullmatch(r"generate_answer\.v1@[0-9a-f]{8}", p.version)
    assert G.load_prompt(*JUDGE_PROMPT).text.startswith("You are reviewing a reply")


def test_a_new_version_file_changes_nothing_until_pinned(tmp_path):
    (tmp_path / "build").mkdir()
    for f in (ROOT / "build").glob("*.txt"):
        (tmp_path / "build" / f.name).write_text(f.read_text())
    (tmp_path / "build" / "generate_answer.v2.txt").write_text("An experimental prompt.")
    assert G.load_prompt(*G.SYSTEM_PROMPT, root=tmp_path).number == 1


def test_environment_override_selects_another_version(tmp_path, monkeypatch):
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "generate_answer.v2.txt").write_text("Version two.")
    monkeypatch.setenv("PROMPT_GENERATE_ANSWER", "v2")
    p = G.load_prompt(*G.SYSTEM_PROMPT, root=tmp_path)
    assert p.number == 2 and p.text == "Version two."


def test_missing_or_empty_file_raises_clearly(tmp_path):
    with pytest.raises(G.PromptError, match="not found"):
        G.load_prompt("generate_answer", root=tmp_path)
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "generate_answer.v1.txt").write_text("   ")
    with pytest.raises(G.PromptError, match="empty"):
        G.load_prompt("generate_answer", root=tmp_path)


def test_render_fills_placeholders_and_rejects_missing_ones():
    user = G.load_prompt(*G.USER_PROMPT)
    assert user.placeholders == {"passages", "ticket"}
    with pytest.raises(G.PromptError, match="missing values"):
        user.render(passages="x")
    out = user.render(passages="P", ticket="Customer says $ticket and ${passages}")
    assert "Customer says $ticket and ${passages}" in out              # customer text is never re-parsed


def test_json_braces_in_prompts_need_no_escaping():
    system = G.load_prompt(*G.SYSTEM_PROMPT)
    assert '{"status":' in system.text and system.placeholders == set() and system.render() == system.text
