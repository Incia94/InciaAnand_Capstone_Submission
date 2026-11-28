"""python -m src.staff: creating, listing, resetting and removing staff accounts at the terminal.

Every answer is typed at a prompt, as a person would; passwords go only through the hidden prompt."""
import pytest

from src import staff as S
from src.staff import StaffDirectory
from tests import fakes

PASSWORD = fakes.password()
NEW = fakes.password()


@pytest.fixture
def terminal(tmp_path, monkeypatch):
    """Answer prompts: `inputs` for visible questions, `secret` for the hidden password prompts."""
    monkeypatch.setenv("STAFF_FILE", str(tmp_path / "staff.json"))
    answers = {"inputs": [], "secret": []}
    monkeypatch.setattr("builtins.input", lambda prompt="": answers["inputs"].pop(0))
    monkeypatch.setattr(S.getpass, "getpass", lambda prompt="": answers["secret"].pop(0))

    def run(argv, inputs=(), secret=()):
        answers["inputs"][:] = list(inputs)
        answers["secret"][:] = list(secret)
        return S.main(argv)
    return run


def test_add_asks_for_every_detail_then_the_password(terminal, capsys):
    assert terminal(["add"], inputs=["priya", "Priya Shah", "Billing", "agent"], secret=[PASSWORD, PASSWORD]) == 0
    assert "They sign in with the username priya" in capsys.readouterr().out
    u = StaffDirectory().check("priya", PASSWORD)
    assert u == {"email": "priya@cloudserve.example", "name": "Priya Shah", "team": "Billing", "role": "agent"}


def test_add_with_details_on_the_command_line_asks_only_for_the_password(terminal):
    argv = ["add", "dev", "--name", "Dev Mehta", "--team", "Deployments", "--role", "supervisor"]
    assert terminal(argv, secret=[PASSWORD, PASSWORD]) == 0
    assert StaffDirectory().check("dev", PASSWORD)["role"] == "supervisor"


def test_add_asks_again_for_an_invalid_role(terminal, capsys):
    assert terminal(["add", "lee", "--name", "Lee", "--team", ""], inputs=["manager", "agent"],
                    secret=[PASSWORD, PASSWORD]) == 0
    assert "Please type one of: supervisor, agent" in capsys.readouterr().out
    assert StaffDirectory().check("lee", PASSWORD)["role"] == "agent"


def test_add_refuses_bad_usernames_duplicates_and_bad_passwords(terminal, capsys):
    assert terminal(["add", "Bad Name!"]) == 1
    assert terminal(["add", "priya", "--name", "P", "--team", "T", "--role", "agent"], secret=[PASSWORD, NEW]) == 1
    assert terminal(["add", "priya", "--name", "P", "--team", "T", "--role", "agent"], secret=["short", "short"]) == 1
    assert StaffDirectory().list() == []                                   # nothing saved by any of these
    terminal(["add", "priya", "--name", "P", "--team", "T", "--role", "agent"], secret=[PASSWORD, PASSWORD])
    assert terminal(["add", "priya", "--name", "Q", "--team", "T", "--role", "agent"]) == 1      # no overwrite
    assert "reset-password priya" in capsys.readouterr().err
    assert StaffDirectory().get("priya")["name"] == "P"


def test_reset_password(terminal):
    terminal(["add", "priya", "--name", "P", "--team", "T", "--role", "agent"], secret=[PASSWORD, PASSWORD])
    assert terminal(["reset-password", "priya"], secret=[NEW, NEW]) == 0
    s = StaffDirectory()
    assert s.check("priya", NEW) and s.check("priya", PASSWORD) is None
    assert s.get("priya")["name"] == "P"                                  # details kept
    assert terminal(["reset-password", "nobody"]) == 1


def test_remove_needs_the_username_typed_to_confirm(terminal):
    terminal(["add", "priya", "--name", "P", "--team", "T", "--role", "agent"], secret=[PASSWORD, PASSWORD])
    assert terminal(["remove", "priya"], inputs=["no"]) == 1 and StaffDirectory().get("priya")
    assert terminal(["remove", "priya"], inputs=["priya"]) == 0 and StaffDirectory().get("priya") is None
    assert terminal(["remove", "priya"]) == 1


def test_list_shows_usernames_never_passwords(terminal, capsys):
    assert terminal(["list"]) == 0 and "No accounts yet" in capsys.readouterr().out
    terminal(["add", "priya", "--name", "Priya Shah", "--team", "Billing", "--role", "agent"],
             secret=[PASSWORD, PASSWORD])
    capsys.readouterr()
    terminal(["list"])
    out = capsys.readouterr().out
    assert "priya" in out and "Priya Shah · Billing" in out and PASSWORD not in out
