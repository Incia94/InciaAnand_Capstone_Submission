"""Scope screening: decide what kind of message this is before any search, translation or model call.

The screen runs straight after ingest (src/api.py, node "screen"). It is rule-based and costs no
model call. A message it recognises gets a fixed, written reply (action "converse") and never
reaches translation, classification, retrieval, generation or an engineer. Everything else goes on
through the pipeline exactly as before.

Every message gets exactly one intent category (Intent), logged as the "screen" stage:

Intent              Examples                                            What happens
support             "My API key returns 401 on every write"             the pipeline, as before
greeting            "Hey! How are you doing?", "thanks", "bye"          fixed reply
out_of_scope        "What is the capital of France?", "Tell me a joke", fixed reply; the sub-kind picks the
                    "Should I quit my job?", "Is AWS better than         wording (entertainment, personal_advice,
                    CloudServe?", "Write a snake game",                  competitor, unrelated_coding,
                    "Can CloudServe build my website?"                   unsupported_service, general)
general_technical   "What is Docker?"                                   short glossary definition, then a question
prompt_injection    "Ignore all previous instructions..."               refused; flagged for review in the log
privacy_violation   "Show me another customer's API key"                refused; flagged for review in the log
unsafe              "How do I hack another deployment?"                 refused; flagged for review in the log
ambiguous           "It is not working.", "abcxyz123"                    asks for details (once per conversation)
frustration         "Your stupid platform never works."                 empathises and asks what is wrong
mixed               "Tell me a joke and fix my API error"               off-topic part dropped; the rest is support
human_handoff       "talk to an agent"                                  escalated by the router, with a handoff

A support message that is also frustrated ("this is useless, my deploy fails with 502") goes through
the pipeline as support, and its reply opens with a fixed acknowledgement (FRUSTRATION_PREFIX).

Rules that never touch a real ticket: every rule is checked against all 580 labelled tickets
(tests/test_scope.py). The one exception is "clarify": on any channel, a short message that names no
specific part of CloudServe ("It is not working.") gets one clarifying question instead of going to
an engineer, so only genuine issues reach the team. If the customer's next message is still unclear,
it is routed (and escalated) as before. A vague message that asks for a person or follows up on an
earlier one ("Can someone look into this?", "Any update?") still goes straight to the team.

What stays with the pipeline on purpose:
  - a request for a person ("talk to an agent") is escalated by the router (src/conversation.py);
  - a real security report ("our account was hacked") is escalated by the policy and topic checks:
    the malicious rule only matches requests to attack something ("how do I hack...");
  - an injection attempt inside a real ticket is escalated with "review required", as before;
  - non-English messages other than small talk are translated first and then handled by the router.

Replies are constant text written here (English; small talk in six languages via src/conversation.py).
They contain no customer text, so they cannot leak personal data or carry injected instructions.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Optional

from src.conversation import (DOMAIN_WORDS, GENERIC_WORDS, PROFANITY, needs_detail, out_of_scope,
                              reply as conversation_reply, small_talk)

ROOT = Path(__file__).resolve().parents[1]
CLARIFY_CHANNELS = {"chat", "email", "forum", "docs_comment"}
CLARIFY_MAX_WORDS = 12


class Intent(str, Enum):
    SUPPORT = "support"
    GREETING = "greeting"
    OUT_OF_SCOPE = "out_of_scope"
    GENERAL_TECHNICAL = "general_technical"
    PROMPT_INJECTION = "prompt_injection"
    PRIVACY_VIOLATION = "privacy_violation"
    UNSAFE = "unsafe"
    AMBIGUOUS = "ambiguous"
    FRUSTRATION = "frustration"
    MIXED = "mixed"
    HUMAN_HANDOFF = "human_handoff"


# The default reply for each category that is answered without the pipeline.
SAFE_RESPONSES = {
    "greeting": (
        "Hello! I'm Cirrus, your CloudServe AI "
        "support assistant. How can I help you today?"
    ),
    "out_of_scope": (
        "I'm designed to help with CloudServe "
        "products and services. I can assist with "
        "deployments, APIs, billing, authentication "
        "and account management."
    ),
    "prompt_injection": (
        "I cannot disclose or modify internal "
        "system instructions. How can I help "
        "you with CloudServe?"
    ),
    "privacy_violation": (
        "I cannot provide unauthorised access "
        "to customer information or credentials."
    ),
    "unsafe": (
        "I cannot assist with unauthorised or "
        "harmful activities. I can help with "
        "legitimate security and support concerns."
    ),
    "ambiguous": (
        "Could you provide more details about "
        "the issue you're experiencing?"
    ),
    "frustration": (
        "I'm sorry you're having a frustrating "
        "experience. Please tell me what's not "
        "working so I can help."
    ),
}

# Preliminary security checks, run before anything else.
INJECTION_PATTERNS = [
    r"ignore (all |your )?previous instructions",
    r"reveal (your |the )?system prompt",
    r"forget (all |your )?instructions",
    r"developer mode",
    r"bypass (your |the )?guardrails",
]

# Categories answered with a fixed reply, no ticket. The router's own names for the same replies
# (small_talk, out_of_scope, needs_detail: messages handled after translation) are included.
CONVERSATIONAL = ("greeting", "out_of_scope", "general_technical", "prompt_injection", "privacy_violation",
                  "unsafe", "ambiguous", "frustration", "small_talk", "needs_detail")
FLAGGED = {"prompt_injection", "privacy_violation", "unsafe"}

# Wording for the kinds of out-of-scope message, more specific than the default.
OUT_OF_SCOPE_REPLIES = {
    "entertainment": "I'm better at troubleshooting cloud issues than doing stand-up comedy! Is there anything on "
                     "CloudServe I can help you with?",
    "personal_advice": "That's a big decision, and not one I can advise on. I'm CloudServe's support assistant, so "
                       "I can help with deployments, APIs, billing or your account whenever you need.",
    "competitor": "Different cloud platforms serve different requirements. I can help you understand CloudServe's "
                  "documented capabilities so you can evaluate whether they meet your needs.",
    "unrelated_coding": "I'm focused on CloudServe-related technical assistance. If you need help deploying an "
                        "application or integrating it with our APIs, I'd be happy to assist.",
    "unsupported_service": "CloudServe gives you the platform to deploy, run and scale the applications you build, "
                           "but we don't build websites or applications for customers. When your site is ready, I "
                           "can help you deploy it, connect a domain or set up SSL on CloudServe.",
}
MIXED_PREFIX = "I'll skip the joke, but I can help with the CloudServe part of your message."
FRUSTRATION_PREFIX = "I'm sorry you're having a frustrating experience. Here's what should help."

# Short definitions for "What is X?" questions. Constant text, kept to concepts that come up in
# CloudServe support, each ending with a question that brings the customer back to their issue.
GLOSSARY = {
    ("docker",): (
        "Docker is a platform for packaging applications and their dependencies into portable containers. Are"
        " you looking to deploy a containerised application or troubleshoot a deployment on CloudServe?"),
    ("container", "containers"): (
        "A container packages an application with everything it needs to run, so it behaves the same "
        "everywhere. Are you deploying a container on CloudServe?"),
    ("kubernetes", "k8s"): (
        "Kubernetes is a system for running and scaling containers across many machines. Are you deploying or"
        " scaling a containerised workload on CloudServe?"),
    ("api", "an api", "rest api"): (
        "An API (application programming interface) lets one program request data or actions from another. "
        "Are you working with the CloudServe API, or seeing an error from it?"),
    ("api key", "api keys"): (
        "An API key is a secret token that identifies your application when it calls an API. Do you need help"
        " creating, rotating or fixing a CloudServe API key?"),
    ("401", "401 error", "401 unauthorized", "a 401 error"): (
        "A 401 Unauthorized response means the request was not authenticated, usually because the API key or "
        "token is missing, expired or invalid. Are you seeing 401s from the CloudServe API?"),
    ("403", "403 error", "403 forbidden"): (
        "A 403 Forbidden response means you are authenticated but not allowed to do that action, often "
        "because of the key's scope or your role. Are you seeing 403s on CloudServe?"),
    ("429", "429 error", "rate limit", "rate limiting", "a rate limit"): (
        "A rate limit caps how many requests you can make in a period; going over it returns 429 Too Many "
        "Requests. Are your requests to CloudServe being limited?"),
    ("ci/cd", "ci cd", "cicd", "continuous deployment"): (
        "CI/CD (continuous integration and delivery) builds, tests and deploys your code automatically on "
        "every change. Are you setting up a pipeline that deploys to CloudServe?"),
    ("dns",): (
        "DNS translates domain names into the addresses of servers. Are you connecting a domain to a "
        "CloudServe deployment?"),
    ("ssl", "tls", "https", "an ssl certificate", "ssl certificate"): (
        "SSL/TLS encrypts traffic between browsers and your site; it is what makes HTTPS work. Are you "
        "setting up a certificate for a CloudServe deployment?"),
    ("webhook", "a webhook", "webhooks"): (
        "A webhook is an HTTP request one system sends to another when an event happens. Are you setting up "
        "or debugging CloudServe webhooks?"),
    ("sso", "single sign-on", "single sign on", "saml"): (
        "Single sign-on (SSO) lets your team sign in through your company's identity provider, often using "
        "SAML. Are you setting up SSO for your CloudServe organisation?"),
    ("mfa", "2fa", "two-factor authentication", "multi-factor authentication"): (
        "Multi-factor authentication adds a second step, such as a code from an app, when signing in. Do you "
        "need help with MFA on your CloudServe account?"),
    ("load balancer", "a load balancer", "load balancing"): (
        "A load balancer spreads incoming traffic across several instances of your application. Are you "
        "scaling a deployment on CloudServe?"),
    ("serverless",): (
        "Serverless means running code without managing servers: the platform starts and scales it for you. "
        "Are you deploying functions on CloudServe?"),
    ("rollback", "a rollback"): (
        "A rollback returns an application to an earlier release, usually after a bad deploy. Do you need to "
        "roll back a CloudServe deployment?"),
    ("deployment", "a deployment"): (
        "A deployment is a release of your application running on the platform. Are you having trouble with a"
        " CloudServe deployment?"),
    ("environment variable", "environment variables", "an environment variable", "env var", "env vars"): (
        "Environment variables hold configuration, such as connection strings, outside your code. Are you "
        "setting them for a CloudServe deployment?"),
    ("health check", "a health check", "health checks"): (
        "A health check is a request the platform sends to your application to confirm it is ready to take "
        "traffic. Is a deployment failing its health check on CloudServe?"),
    ("cdn", "a cdn"): (
        "A CDN (content delivery network) serves your static files from locations close to your users. Are "
        "you setting up caching for a CloudServe deployment?"),
    ("oauth",): (
        "OAuth is a standard that lets an application act on a user's behalf without seeing their password. "
        "Are you integrating OAuth with CloudServe?"),
}
_GLOSSARY = {term: text for terms, text in GLOSSARY.items() for term in terms}


def _rx(p: str) -> re.Pattern:
    return re.compile(p, re.I)


_INJECTION_PRELIM = _rx("|".join(f"(?:{p})" for p in INJECTION_PATTERNS))
_INJECTION = _rx(r"\b(?:ignore|disregard|forget|override|bypass)\b.{0,30}\b(?:previous|prior|above|earlier|all|your|"
                 r"the)\b.{0,20}\b(?:instructions?|rules?|prompts?|guidelines?|directions?)|\b(?:reveal|show|print|"
                 r"display|tell me|repeat|output|leak|give me)\b.{0,25}\b(?:system prompt|your prompt|your "
                 r"instructions|hidden instructions|initial instructions|internal instructions|your rules|your "
                 r"configuration|your config)|\bjailbreak|\bdeveloper mode\b|\bdan mode\b|\byou are now\b|\bact as "
                 r"(?:an? )?(?:unrestricted|unfiltered|different)|\bpretend (?:you are|to be)\b")
_MALICIOUS = _rx(r"\b(?:how (?:do|can|could|would|should) (?:i|we|you|one)|help me|teach me|show me how to|tell me "
                 r"how to|explain how to|i want to|i need to|we want to|ways? to|steps to|script to|tool to)\b.{0,40}"
                 r"\b(?:hack|ddos|dos attack|exploit|crack|brute[- ]?force|break into|take down|steal|bypass|"
                 r"phish|spoof|hijack|sniff|inject)|\bhack(?:ing)? into\b|\bddos\b")
_CONFIDENTIAL = _rx(r"\b(?:another|other|someone else'?s?|somebody else'?s?|different|all (?:the )?|every|other "
                    r"people'?s?)\s*(?:cloudserve )?(?:customers?|users?|clients?|compan(?:y|ies)|tenants?|"
                    r"accounts?|organi[sz]ations?|orgs?|people)'?s?\b.{0,25}\b(?:api keys?|keys?|passwords?|"
                    r"credentials?|tokens?|secrets?|data|details|emails?|invoices?|information|info|logs?|"
                    r"deployments?)|\b(?:api keys?|keys?|passwords?|credentials?|tokens?|secrets?|data|details|"
                    r"emails?|invoices?|information|info|logs?)\b.{0,25}\b(?:belonging to|of|for|from|owned by)\b"
                    r".{0,10}\b(?:another|other|all|every|different)\b.{0,15}\b(?:customers?|users?|clients?|"
                    r"compan(?:y|ies)|tenants?|accounts?|organi[sz]ations?|people)")
_ENTERTAIN = _rx(r"\b(?:(?:tell|give|share|say|show) (?:me |us )?(?:a |an |some |another |one )?(?:funny |good |"
                 r"quick |short |silly |random |dad )*(?:jokes?|riddles?|poems?|story|stories|fun facts?|puns?|"
                 r"memes?)|sing (?:me |us )?(?:a )?song|write (?:me |us )?(?:a |an )?(?:poem|song|haiku|story|"
                 r"limerick|rap)|make me laugh|entertain me|play a game|let'?s play|i'?m bored)\b")
_CODING = _rx(r"\b(?:write|code|build|create|make|generate|give me|develop|implement|program|solve)\b.{0,40}\b(?:"
              r"programs?|games?|scripts?|code|functions?|algorithms?|classes|calculators?|sudoku|snake|tic[- ]?tac"
              r"[- ]?toe|chess|leetcode|fibonacci|palindrome|sorting|bubble sort|binary search|todo app|"
              r"to-do app|chatbot)\b|\b(?:leetcode|sudoku|fizzbuzz|fibonacci)\b")
_UNSUPPORTED = _rx(r"\b(?:can|could|will|would|do|does|please)\b.{0,15}\b(?:you|cloudserve|cirrus|your team|you "
                   r"guys)\b.{0,15}\b(?:build|design|create|develop|write|make|code|redesign|program|market|manage|"
                   r"run|do)\b.{0,15}\b(?:my|our|a|an|the)?\s*(?:website|web ?site|site|web app|mobile app|ios app|"
                   r"android app|app|application|logo|store|shop|online store|landing page|marketing|seo|backend|"
                   r"frontend|wordpress|business)\b|\bbuild (?:my|our) (?:website|site|app)\b.{0,10}\bfor (?:me|us)")
_COMPETITOR = _rx(r"\b(?:aws|amazon web services|azure|microsoft azure|gcp|google cloud|heroku|digital ?ocean|"
                  r"vercel|netlify|render\.com|fly\.io|linode|akamai|oracle cloud|ibm cloud|cloudflare|railway|"
                  r"alibaba cloud|ovh|hetzner)\b")
_COMPARE = _rx(r"\b(?:better|worse|best|vs\.?|versus|compare[ds]?|comparison|cheaper|faster|slower|alternatives?|"
               r"instead of|rather than|than|switch to|should i (?:use|choose|pick|go with)|which (?:one|is))\b")
_ADVICE = _rx(r"\b(?:should i (?:quit|leave|resign|marry|break up|divorce|invest|buy|sell|move|date|study|take|"
              r"accept|have)|my (?:boss|girlfriend|boyfriend|wife|husband|partner|parents?|marriage|relationship|"
              r"career|health|diet|life)|career advice|life advice|relationship advice|dating advice|medical advice|"
              r"legal advice|financial advice|investment advice|lose weight|quit my job|break up with|feel "
              r"(?:sad|depressed|lonely|anxious))\b")
_DEFINE = _rx(r"^\s*(?:(?:what|wat)(?:'s| is| are| does)|define|explain|meaning of|what do you mean by|tell me "
              r"(?:about|what)|can you explain|could you explain)\s+(?:an? |the )?(.+?)(?:\s+(?:mean|means|stand "
              r"for|is))?\s*[?.!]*\s*$")
_FRUSTRATION = _rx(r"\b(?:useless|terrible|awful|horrible|worst|hate|ridiculous|garbage|pathetic|sucks?|rubbish|"
                   r"annoying|fed up|sick of|disappointed|frustrat\w*|never works?|doesn'?t ever work|joke of a|"
                   r"waste of (?:time|money)|unacceptable|disgusting|incompetent)\b")
_NOT_RESOLVED = _rx(r"\b(?:still (?:not|isn'?t|doesn'?t|does not|won'?t|can'?t|failing|broken|getting|the same)|"
                    r"didn'?t (?:work|help|fix|solve)|did not (?:work|help|fix|solve)|not (?:fixed|resolved|solved)|"
                    r"same (?:error|problem|issue)|tried (?:that|this|it|those|your)|no luck|doesn'?t help|"
                    r"does not help|no change|nothing changed)\b")
_FOLLOW_UP = _rx(r"\b(?:follow(?:ing)?[- ]?up|previous message|any update|update on|my ticket|case number|call "
                 r"(?:me|us)|someone|somebody|anyone|agent|human|person|manager|representative)\b")
_CONNECTIVE = _rx(r"^\s*(?:,|\.|;|and|also|then|plus|but|after that|and then|&)+\s*")
_EXTRA_GENERIC = set("""platform platforms system systems service services product products cloudserve app apps
application applications site website thing things everything nothing anything stuff it this that enable
enabled enabling disable disabled activate turn switch""".split())
_STOP = set("""a an the and or but if then so to of in on at by for with from about as is are was were be been being
do does did done have has had i me my we our us you your he she it its they them their this that these those
what which who whom whose when where why how can could will would should shall may might must not no yes
please thanks hi hello hey ok okay just also too very really more most some any all each every""".split())


@dataclass(frozen=True)
class Screen:
    intent: Intent                  # the category
    kind: str = ""                  # finer kind, for the log and the reply wording
    reply: Optional[str] = None     # fixed reply, when the category is answered without the pipeline
    language: str = "en"
    cleaned: Optional[str] = None   # MIXED: the message without its off-topic part
    detail: str = ""
    flags: tuple = field(default_factory=tuple)

    @property
    def answered_here(self) -> bool:
        return self.reply is not None


@lru_cache(maxsize=1)
def vocabulary() -> frozenset:
    """Words seen in the tickets and help articles, plus common English: a message with none of them
    and no CloudServe term is treated as unreadable (keyboard mashing, random characters)."""
    words = set(_STOP) | GENERIC_WORDS | _EXTRA_GENERIC
    for name in ("development_tickets.json", "validation_tickets.json", "documentation.json"):
        path = ROOT / "data" / name
        if path.exists():
            words |= set(re.findall(r"[a-z]{2,}", path.read_text(encoding="utf-8").lower()))
    return frozenset(words)


def _specific(text: str) -> list[str]:
    """CloudServe terms in the text that say what the problem is about (not just 'wrong', 'platform')."""
    return [m.group(0).lower() for m in DOMAIN_WORDS.finditer(text or "")
            if m.group(0).lower() not in GENERIC_WORDS | _EXTRA_GENERIC]


def _nonsense(text: str) -> bool:
    tokens = re.findall(r"[^\W_]+", text.lower())
    if not tokens or DOMAIN_WORDS.search(text):
        return False
    vocab = vocabulary()
    return not any(re.sub(r"\d+", "", tok) in vocab for tok in tokens if len(re.sub(r"\d+", "", tok)) >= 2)


def _definition(text: str) -> Optional[str]:
    m = _DEFINE.match(text)
    if not m or len(text.split()) > 9:
        return None
    term = re.sub(r"\s+", " ", m.group(1).lower().strip(" ?.!\"'"))
    term = re.sub(r"^(?:an?|the)\s+", "", term)
    return _GLOSSARY.get(term) or _GLOSSARY.get(f"a {term}")


def _clarified_already(conversation: list) -> bool:
    return any((m or {}).get("text", "").strip() == SAFE_RESPONSES["ambiguous"] for m in conversation or [])


def classify_scope(text: str, channel: str = "chat", conversation: Optional[list] = None,
                   language: str = "en") -> Screen:
    """The message's intent category. SUPPORT (and HUMAN_HANDOFF, MIXED) go on through the pipeline;
    every other category comes with its fixed reply."""
    raw = " ".join((text or "").split())
    support = Screen(Intent.SUPPORT, "support", detail="A support request: routed through the pipeline.")
    if not raw:
        return support
    # 1 preliminary security checks
    if _INJECTION_PRELIM.search(raw) or _INJECTION.search(raw):
        rest = _INJECTION.sub(" ", _INJECTION_PRELIM.sub(" ", raw))
        if not _specific(rest):
            return Screen(Intent.PROMPT_INJECTION, "prompt_injection", SAFE_RESPONSES["prompt_injection"],
                          detail="The message tries to override or reveal the assistant's instructions; refused.",
                          flags=("possible_instruction_injection: refused",))
        return support                                    # a real ticket with injected text: review (ingest flag)
    if _MALICIOUS.search(raw):
        return Screen(Intent.UNSAFE, "malicious_request", SAFE_RESPONSES["unsafe"],
                      detail="The message asks for help attacking or breaking into a system; refused.",
                      flags=("unsafe_request: refused, review",))
    if _CONFIDENTIAL.search(raw):
        return Screen(Intent.PRIVACY_VIOLATION, "confidential_request", SAFE_RESPONSES["privacy_violation"],
                      detail="The message asks for another customer's credentials or data; refused.",
                      flags=("privacy_violation: refused, review",))
    # 2 conversation
    st = small_talk(raw)
    if st is not None:
        if st.kind == "human":
            return Screen(Intent.HUMAN_HANDOFF, "human", detail="The customer asked for a person.")
        reply = SAFE_RESPONSES["greeting"] if st.kind in ("greet", "how") and st.language == "en" \
            else conversation_reply(st.kind, st.language)
        return Screen(Intent.GREETING, st.kind, reply, st.language, detail=f"Small talk ({st.kind}).")
    if language not in ("en", "und"):
        return support                                    # translated first, then routed
    answered = any((m or {}).get("role") in ("cirrus", "assistant", "agent", "bot") for m in conversation or [])
    if answered and _NOT_RESOLVED.search(raw):
        return Screen(Intent.SUPPORT, "follow_up_not_resolved",
                      detail="A follow-up: the customer says the earlier answer did not solve it.",
                      flags=("follow_up_not_resolved",))
    specific = _specific(raw)
    # 3 mixed: an off-topic request alongside a real one
    fun = _ENTERTAIN.search(raw)
    if fun:
        rest = _CONNECTIVE.sub("", (raw[:fun.start()] + " " + raw[fun.end():]).strip().lstrip(",.;& "))
        rest = _CONNECTIVE.sub("", rest)
        if _specific(rest):
            return Screen(Intent.MIXED, "mixed", cleaned=rest[:1].upper() + rest[1:],
                          detail=f"Dropped the off-topic part of the message (\"{fun.group(0)}\").")
        return _out_of_scope("entertainment", "A request for entertainment.")
    # 4 out of scope, by kind
    if _UNSUPPORTED.search(raw):
        return _out_of_scope("unsupported_service", "Asks CloudServe to do work it does not offer.")
    if _COMPETITOR.search(raw) and _COMPARE.search(raw):
        return _out_of_scope("competitor", "Asks to compare CloudServe with a competitor.")
    definition = _definition(raw)
    if definition:
        return Screen(Intent.GENERAL_TECHNICAL, "definition", definition,
                      detail="A general technical question, answered with a short glossary definition.")
    if _CODING.search(raw) and not specific:
        return _out_of_scope("unrelated_coding", "A request to write code unrelated to CloudServe.")
    if _ADVICE.search(raw) and not specific:
        return _out_of_scope("personal_advice", "A request for personal advice.")
    # 5 frustration, ambiguity
    angry = bool(PROFANITY.search(raw) or _FRUSTRATION.search(raw))
    if (angry and not specific) or needs_detail(raw):
        return Screen(Intent.FRUSTRATION, "frustration", SAFE_RESPONSES["frustration"],
                      detail="Frustrated, but says nothing specific about what is wrong; asked for details.")
    if _nonsense(raw):
        return Screen(Intent.AMBIGUOUS, "nonsense", SAFE_RESPONSES["ambiguous"],
                      detail="The message has no recognisable words; asked for details.")
    if (channel in CLARIFY_CHANNELS and not specific and len(raw.split()) <= CLARIFY_MAX_WORDS
            and DOMAIN_WORDS.search(raw) and not _FOLLOW_UP.search(raw) and not _clarified_already(conversation)):
        return Screen(Intent.AMBIGUOUS, "vague", SAFE_RESPONSES["ambiguous"],
                      detail="The message does not say which part of CloudServe is affected; asked "
                             "once before involving an engineer.")
    if out_of_scope(raw, familiarity=1.0):
        return _out_of_scope("general", "The message contains no CloudServe terms.")
    if angry:
        return Screen(Intent.SUPPORT, "support", detail="A support request from a frustrated customer.",
                      flags=("customer_frustrated",))
    return support


def _out_of_scope(kind: str, detail: str) -> Screen:
    return Screen(Intent.OUT_OF_SCOPE, kind, OUT_OF_SCOPE_REPLIES.get(kind, SAFE_RESPONSES["out_of_scope"]),
                  detail=detail)


def screen(text: str, channel: str = "chat", conversation: Optional[list] = None,
           language: str = "en") -> Optional[Screen]:
    """Like classify_scope, but None for a plain support request (nothing to do)."""
    s = classify_scope(text, channel, conversation, language)
    return None if s.intent == Intent.SUPPORT and not s.flags else s
