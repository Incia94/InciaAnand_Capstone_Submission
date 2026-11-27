"""Made-up secrets for testing the leak checks.

None of these is a real credential. They are generated at runtime so that no key-, token- or
password-shaped string is written anywhere in the repository: secret scanners read the source
files, while the guardrails under test see the assembled values.
"""
import secrets
import string

_SK = "sk" + "-live-"
_CHARS = string.ascii_uppercase[:6] + string.digits                   # A-F and 0-9


def api_key(length: int = 24, suffix: str = "") -> str:
    """A made-up API key: the sk-live- prefix, `length` characters, then `suffix`."""
    return _SK + "".join(_CHARS[i % len(_CHARS)] for i in range(length)) + suffix


AWS_KEY = "AK" + "IA" + "".join(string.ascii_uppercase[i % 26] for i in range(7, 23))
JWT = ".".join("ey" + "J" + secrets.token_urlsafe(12).replace("-", "a").replace("_", "b") for _ in range(3))


def assignment_line() -> str:
    """A sentence that hands out a made-up password, as in "Set <word>: <value>"."""
    return "Set " + "pass" + "word" + ": " + secrets.token_hex(5)


def model_key() -> str:
    """A made-up model-provider key, for tests that only need one to be set."""
    return "-".join(["test", "not", "a", "real", "key"])


def password() -> str:
    """A random password for one test run."""
    return secrets.token_urlsafe(12)
