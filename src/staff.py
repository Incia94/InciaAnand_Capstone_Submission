"""Staff accounts and sign-in for the support workspace: supervisors and agents.

Accounts live in STAFF_FILE (default storage/staff.json, which git ignores). Only a salted
PBKDF2-SHA256 hash of each password is kept. The workspace (streamlit_app.py) checks the password
with `check` and keeps the signed-in person in that browser session.

Staff sign in with their username (the part of the email before @, or their first name; "admin" is
the supervisor) or their full email.

Passwords are only ever typed in by a person, at the prompt of the commands below. None is read
from code, from .env or from any other setting, and none is stored except as a hash. The demo
accounts (DEMO_STAFF: one supervisor, three agents) are created only when someone runs
`python -m src.staff demo` and types the shared password; the workspace never creates accounts.

Roles
  supervisor  sees every ticket from every channel, assigns them to agents, sees the reports and the
              decision log
  agent       sees the tickets assigned to them, checks Cirrus's suggested reply, edits it and sends it

    python -m src.staff add                     # asks for username, name, team, role, then the password
    python -m src.staff add priya --name "Priya Shah" --team Billing --role agent   # same, fewer questions
    python -m src.staff reset-password priya    # asks for the new password
    python -m src.staff remove priya            # asks you to type the username to confirm
    python -m src.staff list                    # usernames, names, teams, roles; never passwords
    python -m src.staff demo                    # create (or reset) the four demo accounts, one shared password
"""
from __future__ import annotations

import argparse
import getpass
import hashlib
import hmac
import json
import os
import re
import secrets
import sys
import threading
from pathlib import Path
from typing import Optional, Sequence

ROLES = ("supervisor", "agent")
ITERATIONS = 200_000
DEMO_STAFF = [
    ("marcus@cloudserve.example", "Marcus Adeyemi", "Support Operations", "supervisor"),
    ("asha@cloudserve.example", "Asha", "API Support", "agent"),
    ("sofia@cloudserve.example", "Sofia", "Billing", "agent"),
    ("ravi@cloudserve.example", "Ravi", "Deployments", "agent"),
]


def _hash(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, ITERATIONS).hex()


class StaffDirectory:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path or os.getenv("STAFF_FILE", "./storage/staff.json"))
        self._lock = threading.Lock()

    # ---------------- accounts
    def _load(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _save(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass

    def add(self, email: str, name: str, team: str, role: str, password: str) -> dict:
        email = email.strip().lower()
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}")
        if "@" not in email or not name.strip():
            raise ValueError("an email address and a name are required")
        if len(password) < 8:
            raise ValueError("the password must be at least 8 characters")
        salt = secrets.token_bytes(16)
        with self._lock:
            data = self._load()
            data[email] = {"email": email, "name": name.strip(), "team": team.strip(), "role": role,
                           "salt": salt.hex(), "hash": _hash(password, salt)}
            self._save(data)
        return self.get(email)

    def set_password(self, email: str, password: str) -> None:
        if len(password) < 8:
            raise ValueError("the password must be at least 8 characters")
        salt = secrets.token_bytes(16)
        with self._lock:
            data = self._load()
            if email not in data:
                raise ValueError(f"no account {email}")
            data[email].update(salt=salt.hex(), hash=_hash(password, salt))
            self._save(data)

    def remove(self, email: str) -> None:
        with self._lock:
            data = self._load()
            if data.pop(email, None) is None:
                raise ValueError(f"no account {email}")
            self._save(data)

    def resolve(self, login: str) -> Optional[str]:
        """The account email for a sign-in name: the email itself, the part before @, the first name,
        or "admin" for the supervisor. None unless exactly one account matches."""
        login = (login or "").strip().lower()
        data = self._load()
        if login in data:
            return login
        for match in (lambda e, u: e.split("@")[0] == login,
                      lambda e, u: u["name"].split()[0].lower() == login,
                      lambda e, u: login == "admin" and u["role"] == "supervisor"):
            found = [e for e, u in data.items() if match(e, u)]
            if len(found) == 1:
                return found[0]
        return None

    def get(self, email: str) -> Optional[dict]:
        u = self._load().get(self.resolve(email) or "")
        return {k: u[k] for k in ("email", "name", "team", "role")} if u else None

    def list(self, role: Optional[str] = None) -> list[dict]:
        users = [self.get(e) for e in sorted(self._load())]
        return [u for u in users if u and (role is None or u["role"] == role)]

    def check(self, email: str, password: str) -> Optional[dict]:
        u = self._load().get(self.resolve(email) or "")
        if not u:
            _hash(password or "", b"0" * 16)               # same time whether or not the account exists
            return None
        ok = hmac.compare_digest(_hash(password or "", bytes.fromhex(u["salt"])), u["hash"])
        return self.get(u["email"]) if ok else None


DOMAIN = "cloudserve.example"
_USERNAME = re.compile(r"^[a-z][a-z0-9._-]{1,31}$")


def email_for(username: str) -> str:
    """A username becomes <username>@cloudserve.example; a full email address is kept as it is."""
    username = (username or "").strip().lower()
    return username if "@" in username else f"{username}@{DOMAIN}"


def ask_password(label: str) -> Optional[str]:
    """Ask a person for a password twice, without showing it. Nothing else can supply it."""
    first = getpass.getpass(f"{label} (at least 8 characters): ")
    if getpass.getpass("Type it again: ") != first:
        print("The two passwords did not match; nothing was saved.", file=sys.stderr)
        return None
    return first


def ask(label: str, given: Optional[str] = None, choices: Sequence[str] = (), required: bool = True) -> str:
    """A value from the command line if given, otherwise typed at a prompt (asked again until valid)."""
    if given is not None and (not choices or given in choices):
        return given.strip()
    hint = f" ({' / '.join(choices)})" if choices else ""
    while True:
        value = input(f"{label}{hint}: ").strip()
        if choices and value.lower() not in choices:
            print(f"  Please type one of: {', '.join(choices)}.")
            continue
        if required and not value:
            print("  This is needed.")
            continue
        return value.lower() if choices else value


def _add(staff: "StaffDirectory", args) -> int:
    username = ask("Username (what they type to sign in, e.g. priya)", args.username)
    if "@" not in username and not _USERNAME.match(username.lower()):
        print("A username is 2-32 characters: lowercase letters, digits, '.', '_' or '-', starting with a letter.",
              file=sys.stderr)
        return 1
    email = email_for(username)
    if email in staff._load():
        print(f"{email} already exists. To change its password: python -m src.staff reset-password "
              f"{email.split('@')[0]}", file=sys.stderr)
        return 1
    name = ask("Full name", args.name)
    team = ask("Team (e.g. API Support)", args.team, required=False)
    role = ask("Role", args.role, choices=ROLES)
    password = ask_password(f"Password for {email.split('@')[0]}")
    if password is None:
        return 1
    u = staff.add(email, name, team, role, password)
    print(f"Created {u['role']} '{email.split('@')[0]}' ({u['name']}{' · ' + u['team'] if u['team'] else ''}). "
          f"They sign in with the username {email.split('@')[0]} and the password you typed.")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    p = argparse.ArgumentParser(prog="python -m src.staff", description="Staff accounts for the support workspace. "
                                "Every password is typed at the prompt; none can be passed any other way.")
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("add", help="create an account; asks for anything not given, then the password")
    a.add_argument("username", nargs="?", help="what they type to sign in (or a full email address)")
    a.add_argument("--name", help="full name")
    a.add_argument("--team", help="team, e.g. API Support")
    a.add_argument("--role", choices=ROLES)
    r = sub.add_parser("reset-password", help="set a new password for an account (asks for it)")
    r.add_argument("username")
    rm = sub.add_parser("remove", help="delete an account (asks you to confirm)")
    rm.add_argument("username")
    sub.add_parser("demo", help="create (or reset) the demo supervisor and three agents with one shared password")
    sub.add_parser("list", help="the accounts (never the passwords)")
    args = p.parse_args(argv)
    staff = StaffDirectory()
    try:
        if args.command == "list":
            users = staff.list()
            if not users:
                print("No accounts yet. Create one with: python -m src.staff add")
            for u in users:
                print(f"{u['role']:<11} {u['email'].split('@')[0]:<12} {u['name']}"
                      f"{' · ' + u['team'] if u['team'] else ''}")
            return 0
        if args.command == "add":
            return _add(staff, args)
        if args.command in ("reset-password", "remove"):
            email = staff.resolve(args.username)
            if email is None:
                print(f"No account '{args.username}'. See them with: python -m src.staff list", file=sys.stderr)
                return 1
            if args.command == "remove":
                if input(f"Type {email.split('@')[0]} to delete this account: ").strip().lower() \
                        != email.split("@")[0]:
                    print("Not deleted.")
                    return 1
                staff.remove(email)
                print(f"Deleted '{email.split('@')[0]}'.")
                return 0
            password = ask_password(f"New password for {email.split('@')[0]}")
            if password is None:
                return 1
            staff.set_password(email, password)
            print(f"Password changed for '{email.split('@')[0]}'.")
            return 0
        password = ask_password("Shared demo password")
        if password is None:
            return 1
        for email, name, team, role in DEMO_STAFF:
            staff.add(email, name, team, role, password)
            print(f"Saved {role} {email.split('@')[0]:<7} ({name} · {team}).")
        print("Sign in with a username (marcus, asha, sofia or ravi) and the password you typed.")
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
