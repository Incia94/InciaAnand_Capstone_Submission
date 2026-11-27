"""Staff accounts and sign-in for the support workspace: supervisors and agents.

Accounts live in STAFF_FILE (default storage/staff.json, which git ignores). Only a salted
PBKDF2-SHA256 hash of each password is kept. The workspace (streamlit_app.py) checks the password
with `check` and keeps the signed-in person in that browser session.

Staff sign in with their username (the part of the email before @, or their first name; "admin" is
the supervisor) or their full email.

Demo accounts: the four demo accounts below are created on first use with the shared password set
as DEMO_PASSWORD in your .env file (which git ignores). No password is written in the code: without
DEMO_PASSWORD, no demo accounts are made. DEMO_ACCOUNTS=off turns them off; add real accounts with
`python -m src.staff add` before anyone else uses the workspace.

Roles
  supervisor  sees every ticket from every channel, assigns them to agents, sees the reports and the
              decision log
  agent       sees the tickets assigned to them, checks Cirrus's suggested reply, edits it and sends it

    python -m src.staff add --email asha@cloudserve.example --name "Asha" --team "API Support" --role agent
    python -m src.staff demo        # (re)create the demo accounts with DEMO_PASSWORD from .env, or one you type
    python -m src.staff list
"""
from __future__ import annotations

import argparse
import getpass
import hashlib
import hmac
import json
import os
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

    def ensure_demo(self) -> list[str]:
        """Create any missing demo account, with the password from DEMO_PASSWORD (.env). Existing
        accounts are left alone. Nothing happens without DEMO_PASSWORD or with DEMO_ACCOUNTS=off."""
        password = os.getenv("DEMO_PASSWORD", "")
        if not password or os.getenv("DEMO_ACCOUNTS", "on").strip().lower() in ("off", "0", "false", "no"):
            return []
        have = self._load()
        made = []
        for email, name, team, role in DEMO_STAFF:
            if email not in have:
                self.add(email, name, team, role, password)
                made.append(email)
        return made

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


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    p = argparse.ArgumentParser(prog="python -m src.staff", description="Staff accounts for the support workspace.")
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("add", help="add or update an account (asks for the password)")
    a.add_argument("--email", required=True)
    a.add_argument("--name", required=True)
    a.add_argument("--team", default="")
    a.add_argument("--role", choices=ROLES, required=True)
    sub.add_parser("demo", help="(re)create the demo supervisor and three agents with one shared password")
    sub.add_parser("list", help="the accounts (never the passwords)")
    args = p.parse_args(argv)
    staff = StaffDirectory()
    if args.command == "list":
        for u in staff.list():
            print(f"{u['role']:<11} {u['email']:<34} {u['name']} · {u['team']}")
        return 0
    if args.command == "demo":
        password = (os.getenv("STAFF_PASSWORD") or os.getenv("DEMO_PASSWORD")
                    or getpass.getpass("Demo password (at least 8 characters): "))
    else:
        password = os.getenv("STAFF_PASSWORD") or getpass.getpass("Password (at least 8 characters): ")
    try:
        if args.command == "add":
            u = staff.add(args.email, args.name, args.team, args.role, password)
            print(f"Saved {u['role']} {u['email']}.")
        else:
            for email, name, team, role in DEMO_STAFF:
                staff.add(email, name, team, role, password)
                print(f"Saved {role} {email.split('@')[0]:<7} ({name} · {team}).")
            print("Sign in with a username (e.g. marcus, asha) and the demo password.")
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
