"""Conversation: messages that are not support tickets, handled before routing.

Two kinds of message reached the router before this module existed and were escalated, because
neither matches a help article:

  small talk     the whole message is a greeting, "how are you", thanks, goodbye, a question
                 about what the assistant is, or a bare acknowledgement ("ok"). It gets a short,
                 fixed reply in the customer's language (English, Spanish, French, German,
                 Italian, Portuguese), and nothing is escalated. A greeting followed by a real
                 question ("hi, my API key returns 401") is NOT small talk: it is routed normally.
  out of scope   the message has nothing to do with CloudServe ("when will my groceries arrive?").
                 It gets a fixed reply saying what the assistant can help with, and how to reach a
                 person if it was about CloudServe after all.

  needs detail   an angry message with no detail ("wtf is wrong with my system"): it swears and names
                 nothing specific about CloudServe beyond generic words such as "wrong", "working" or
                 "problem". It gets a calm reply asking which part of CloudServe is affected and what
                 the error says, and nothing is escalated yet: once the customer says what is wrong, the
                 message is routed normally. An angry message that does say what is wrong ("wtf, my
                 deployment fails with a 502") is routed normally straight away. No ticket in the data
                 contains these words (tests/test_conversation.py checks this), so the labelled
                 "unclear request" tickets ("It is not working. Please help.") still go to a person.

A customer who asks for a person ("agent", "talk to a human") is escalated on purpose.

Out of scope is decided conservatively. The message must contain none of CloudServe's domain words
(DOMAIN_WORDS), and in addition either
  1 be short (at most SHORT_MESSAGE_WORDS words), so the missing domain words are telling, or
  2 be unlike every past ticket (classifier familiarity below OUT_OF_SCOPE_FAMILIARITY).
The short-message rule does not depend on the classifier, so "where are my groceries?" is caught
whatever model is loaded.
All 580 labelled tickets (development and validation) contain at least one domain word, and the
lowest familiarity of a validation ticket is 0.45, so neither rule touches a real ticket in the data
(tests/test_conversation.py checks this). Vague support messages ("it is not working, please help")
contain domain words and are still escalated to a person, as before.

The replies are constant text written here: no model writes them and they contain no customer text,
so they cannot carry personal data or injected instructions. Every one is logged like any other
decision, and the kill switch stops them like any other reply.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

OUT_OF_SCOPE_FAMILIARITY = 0.40
SHORT_MESSAGE_WORDS = 20

PROFANITY = re.compile(
    r"(?<![\w*])(?:wtf|wth|ffs|omfg|f+u+c+k\w*|f[*#@]+c?k\w*|fk|fking|sh[i1*]t\w*|bullshit|crap\w*|damn\w*|"
    r"bloody|stupid|useless|idiot\w*|rubbish|what the (?:hell|heck)|the hell)(?![\w*])", re.I)
# Domain words too general to say what is wrong: an angry message with only these needs more detail.
GENERIC_WORDS = set("""wrong work works working worked workin problem problems issue issues help helps helping
helped urgent urgently fix fixed fixing broken break breaks breaking stuck loading status error errors fail fails
failed failing failure crash crashes crashed crashing down slow slowly support supports""".split())

DOMAIN_WORDS = re.compile(
    r"\b(?:"
    r"cloud\w*|deploy\w*|release\w*|rollback|roll(?:ed|ing)? back|revert\w*|revision\w*|build\w*|"
    r"pipeline\w*|ci\b|cd\b|container\w*|docker|kubernetes|k8s|cluster\w*|instance\w*|server\w*|"
    r"service\w*|app\b|apps\b|application\w*|function\w*|environment\w*|staging|production|prod\b|"
    r"config\w*|variable\w*|secret\w*|api\w*|endpoint\w*|request\w*|sdk|cli\b|webhook\w*|signature\w*|"
    r"pagination|paginat\w*|rate.?limit\w*|throttl\w*|quota\w*|429|401|403|404|500|502|503|error\w*|"
    r"fail\w*|crash\w*|bug\w*|broken|break\w*|outage|down\b|slow\w*|latency|timeout\w*|time.?out|"
    r"performance|load\b|autoscal\w*|scal\w*|memory|cpu|key\b|keys\b|token\w*|credential\w*|"
    r"password\w*|login\w*|log\s?in|logged|sign(?:ed|ing)?\s?in|signin|sso|saml|okta|mfa|2fa|"
    r"authenticat\w*|auth\b|permission\w*|role\w*|access\w*|account\w*|user\w*|team\w*|member\w*|"
    r"org\w*|project\w*|workspace\w*|admin\w*|bill\w*|invoice\w*|charg\w*|payment\w*|pay\b|paid|"
    r"price\w*|pricing|plan\b|plans\b|subscription\w*|upgrade\w*|downgrade\w*|prorat\w*|refund\w*|"
    r"credit\w*|overage\w*|usage|spend\w*|cost\w*|data\b|database\w*|db\b|backup\w*|restore\w*|"
    r"export\w*|import\w*|migrat\w*|region\w*|residency|gdpr|compliance|audit\w*|log\b|logs\b|"
    r"monitor\w*|integrat\w*|github|gitlab|terraform|slack|datadog|support\w*|ticket\w*|documentation|"
    r"docs?\b|article\w*|feature\w*|roadmap|onboard\w*|setup|set\s?up|install\w*|console|dashboard\w*|"
    r"portal|security|breach\w*|hack\w*|compromis\w*|vulnerab\w*|incident\w*|dns|domain\w*|ssl|tls|"
    r"certificat\w*|https?|url\w*|connect\w*|connection\w*|network\w*|storage|bucket\w*|file\w*|"
    r"upload\w*|download\w*|link\w*|email\w*|notification\w*|alert\w*|version\w*|update\w*|upgrade\w*|"
    r"cache\w*|query\w*|queries|traffic|request|cloudserve|work\w*|problem\w*|issue\w*|wrong|loading|"
    r"urgent\w*|help\w*|page\b|pages|paging|result\w*|record\w*|collection\w*|cursor\w*|iterat\w*|"
    r"retention|retain\w*|polic\w*|platform\w*|event\w*|setting\w*|expir\w*|fix\w*|resolv\w*|stuck|"
    r"block\w*|reject\w*|denied|invalid|missing|lost|delet\w*|creat\w*|configur\w*|enabl\w*|disabl\w*|"
    r"limit\w*|status"
    r")", re.I)

# (kind, language, phrase pattern). A message is small talk when nothing but these phrases and
# FILLER words remains. Kinds in priority order: human > identity > how > thanks > bye > greet > ack.
_PHRASES = [
    ("human", "en", r"(?:talk|speak|chat) (?:to|with) (?:a |an |some)?(?:human|person|agent|someone|somebody|"
                    r"representative|real person)|(?:a |an )?(?:human|real person|live agent|agent|"
                    r"representative|customer service)"),
    ("human", "es", r"(?:hablar con )?(?:una persona|un agente|un humano|alguien)"),
    ("human", "fr", r"parler (?:à|a) (?:un humain|une personne|un agent|quelqu'un)"),
    ("human", "de", r"(?:mit )?(?:einem menschen|einem mitarbeiter) sprechen|mit einem menschen"),
    ("identity", "en", r"(?:are you|is this|am i (?:talking|speaking|chatting) (?:to|with)) (?:a |an )?(?:bot|robot|"
                       r"human|person|real person|ai|machine)|who are you|what are you"),
    ("identity", "es", r"(?:eres|es) (?:un )?(?:bot|robot|humano|persona)|qui[eé]n eres"),
    ("how", "en", r"how are (?:you|u|things)(?: doing)?(?: today)?|how(?:'s| is) it going|what'?s up|how do you do"),
    ("how", "es", r"c[oó]mo (?:est[aá]s|est[aá]|andas|va|te va)|qu[eé] tal"),
    ("how", "fr", r"(?:comment )?[çc]a va|comment (?:allez-vous|vas-tu|vas tu|allez vous)"),
    ("how", "de", r"wie geht(?:'s| es)?(?: (?:dir|ihnen))?"),
    ("how", "it", r"come (?:stai|sta|va)"),
    ("how", "pt", r"tudo bem|como (?:vai|est[aá]s?)"),
    ("thanks", "en", r"thanks?(?: you)?(?: (?:so|very) much)?|thx|ty|cheers|much appreciated|appreciate it"),
    ("thanks", "en", r"(?:that|it|this|your answer|the fix) (?:worked|works|helped|fixed it|solved it|did it|"
                     r"did the trick)|(?:it'?s|it is|all|problem|issue) (?:sorted|solved|fixed|resolved|working now)|"
                     r"works now|working now|all good(?: now)?"),
    ("thanks", "es", r"(?:muchas )?gracias"),
    ("thanks", "fr", r"merci(?: beaucoup)?"),
    ("thanks", "de", r"danke(?: (?:schön|schoen|sehr))?|vielen dank"),
    ("thanks", "it", r"grazie(?: mille)?"),
    ("thanks", "pt", r"obrigad[oa]"),
    ("bye", "en", r"(?:good)?bye|see (?:you|ya)(?: later)?|have a (?:nice|good|great) (?:day|one)|that'?s all"),
    ("bye", "es", r"adi[oó]s|hasta (?:luego|pronto)|chao"),
    ("bye", "fr", r"au revoir|[aà] bient[oô]t"),
    ("bye", "de", r"tsch[uü]ss|auf wiedersehen"),
    ("bye", "it", r"arrivederci"),
    ("bye", "pt", r"tchau|at[eé] logo"),
    ("greet", "en", r"hi|hello|hey|hiya|howdy|greetings|good (?:morning|afternoon|evening|day)|yo"),
    ("greet", "es", r"hola|buen(?:os|as) (?:d[ií]as|tardes|noches)|buenas"),
    ("greet", "fr", r"bonjour|salut|bonsoir|coucou"),
    ("greet", "de", r"hallo|guten (?:morgen|tag|abend)|servus|moin"),
    ("greet", "it", r"ciao|buongiorno|buonasera|salve"),
    ("greet", "pt", r"ol[aá]|oi|bom dia|boa (?:tarde|noite)"),
    ("greet", "en", r"namaste"),
    ("greet", "en", r"(?:can you |could you |please )?help(?: me)?(?: please)?|i need (?:some )?help|anyone there"),
    ("ack", "en", r"ok(?:ay)?|sure|got it|great|cool|alright|all right|fine|perfect|yes|no|nope|yep|yeah"),
]
_RX = [(k, lang, re.compile(rf"(?<![\w'])(?:{p})(?![\w'])", re.I)) for k, lang, p in _PHRASES]
_FILLER = set("""now there team cloudserve support bot assistant please again all everyone so much very today and
i im i'm am fine good well you your too to the a an me my we our just there's for ? ! want wanna would like
i'd can could need let quiero necesito puedo je veux ich möchte""".split())
_PRIORITY = ["human", "identity", "how", "thanks", "bye", "greet", "ack"]
_MAX_WORDS = 12


@dataclass(frozen=True)
class SmallTalk:
    kind: str          # human | identity | how | thanks | bye | greet | ack
    language: str      # language of the phrase that matched (en, es, fr, de, it, pt)


def small_talk(text: str) -> Optional[SmallTalk]:
    """The message's kind if it is ONLY small talk, else None (route it as a ticket)."""
    flat = " ".join((text or "").lower().replace("’", "'").split())
    if not flat or len(flat.split()) > _MAX_WORDS:
        return None
    found: list[tuple[str, str]] = []
    rest = flat
    for kind, lang, rx in _RX:
        if rx.search(rest):
            found.append((kind, lang))
            rest = rx.sub(" ", rest)
    words = [w.strip(".,!?¿¡;:-()\"'") for w in rest.split()]
    leftover = [w for w in words if w and w not in _FILLER]
    if not found or leftover:
        return None
    best = min(found, key=lambda kl: _PRIORITY.index(kl[0]))
    lang = next((lang for k, lang in found if lang != "en"), best[1])   # a non-English phrase sets the language
    return SmallTalk(best[0], lang)


def out_of_scope(text: str, familiarity: float) -> bool:
    """True only when the message has no CloudServe domain word AND is either short or unlike every
    past ticket."""
    if DOMAIN_WORDS.search(text or ""):
        return False
    return len((text or "").split()) <= SHORT_MESSAGE_WORDS or familiarity < OUT_OF_SCOPE_FAMILIARITY


def needs_detail(text: str) -> bool:
    """True when the message swears and names nothing specific about CloudServe: ask what is wrong
    instead of escalating a ticket nobody can act on."""
    if not PROFANITY.search(text or ""):
        return False
    specific = [m.group(0).lower() for m in DOMAIN_WORDS.finditer(PROFANITY.sub(" ", text))
                if m.group(0).lower() not in GENERIC_WORDS]
    return not specific


REPLIES = {
    "en": {      # greet/how, needs_detail and out_of_scope are the safe responses of src/scope.py
        "greet": "Hello! I'm Cirrus, your CloudServe AI support assistant. How can I help you today?",
        "how": "Hello! I'm Cirrus, your CloudServe AI support assistant. How can I help you today?",
        "thanks": "You're welcome. If anything else comes up, just write here.",
        "bye": "Thanks for getting in touch. Goodbye!",
        "ack": "Is there anything else I can help you with?",
        "identity": "I'm Cirrus, CloudServe's automated support assistant. I answer from CloudServe's help articles, "
                    "and anything I can't answer goes to a person on the support team. Reply \"agent\" at any time "
                    "to reach a person.",
        "needs_detail": "I'm sorry you're having a frustrating experience. Please tell me what's not working so I "
                        "can help.",
        "out_of_scope": "I'm designed to help with CloudServe products and services. I can assist with deployments, "
                        "APIs, billing, authentication and account management.",
    },
    "es": {
        "greet": "¡Hola! Soy el asistente automático de soporte de CloudServe. Cuéntame en qué necesitas ayuda "
                 "(un despliegue, una clave de API, el inicio de sesión, la facturación o tu cuenta) y buscaré el "
                 "artículo de ayuda adecuado o te pondré en contacto con el equipo.",
        "how": "¡Muy bien, gracias por preguntar! Soy el asistente automático de soporte de CloudServe. ¿En qué "
               "puedo ayudarte hoy?",
        "thanks": "De nada. Si necesitas algo más, escríbeme aquí.",
        "bye": "Gracias por contactar con nosotros. ¡Hasta luego!",
        "ack": "¿Puedo ayudarte con algo más?",
        "identity": "Soy un asistente automático. Respondo con los artículos de ayuda de CloudServe, y lo que no "
                    "puedo responder pasa a una persona del equipo de soporte. Escribe \"agente\" para hablar con "
                    "una persona.",
        "needs_detail": "Siento que algo no funcione; quiero ayudarte a resolverlo. ¿Puedes contarme un poco "
                        "más: qué parte de CloudServe está afectada (un despliegue, la API, el inicio de sesión o "
                        "la facturación), qué intentabas hacer y qué mensaje de error ves?",
        "out_of_scope": "Solo puedo ayudar con preguntas sobre los servicios de CloudServe, como despliegues, API, "
                        "inicio de sesión, facturación, datos o tu cuenta. Tu mensaje no parece ser sobre eso, así "
                        "que no lo he enviado al equipo de soporte. Si es sobre CloudServe, cuéntame un poco más o "
                        "escribe \"agente\" para hablar con una persona.",
    },
    "fr": {
        "greet": "Bonjour ! Je suis l'assistant de support automatique de CloudServe. Dites-moi ce dont vous avez "
                 "besoin (un déploiement, une clé d'API, la connexion, la facturation ou votre compte) et je "
                 "trouverai le bon article d'aide ou je vous mettrai en relation avec l'équipe.",
        "how": "Très bien, merci de demander ! Je suis l'assistant de support automatique de CloudServe. Comment "
               "puis-je vous aider aujourd'hui ?",
        "thanks": "Avec plaisir. Si vous avez besoin d'autre chose, écrivez-moi ici.",
        "bye": "Merci de nous avoir contactés. Au revoir !",
        "ack": "Puis-je vous aider avec autre chose ?",
        "identity": "Je suis un assistant automatique. Je réponds à partir des articles d'aide de CloudServe, et ce "
                    "que je ne peux pas traiter est confié à une personne de l'équipe de support.",
        "needs_detail": "Désolé que quelque chose ne fonctionne pas ; je veux vous aider. Pouvez-vous m'en dire "
                        "un peu plus : quelle partie de CloudServe est concernée (un déploiement, l'API, la "
                        "connexion ou la facturation), ce que vous essayiez de faire et le message d'erreur affiché ?",
        "out_of_scope": "Je ne peux répondre qu'aux questions sur les services de CloudServe : déploiements, API, "
                        "connexion, facturation, données ou votre compte. Votre message ne semble pas en faire "
                        "partie, je ne l'ai donc pas transmis à l'équipe. S'il concerne CloudServe, donnez-moi un "
                        "peu plus de détails.",
    },
    "de": {
        "greet": "Hallo! Ich bin der automatische Support-Assistent von CloudServe. Sagen Sie mir, wobei Sie Hilfe "
                 "brauchen (Deployment, API-Schlüssel, Anmeldung, Abrechnung oder Ihr Konto), und ich finde den "
                 "passenden Hilfeartikel oder leite Sie an das Team weiter.",
        "how": "Danke der Nachfrage, gut! Ich bin der automatische Support-Assistent von CloudServe. Wobei kann "
               "ich Ihnen heute helfen?",
        "thanks": "Gern geschehen. Wenn noch etwas ist, schreiben Sie einfach hier.",
        "bye": "Danke für Ihre Nachricht. Auf Wiedersehen!",
        "ack": "Kann ich Ihnen sonst noch helfen?",
        "identity": "Ich bin ein automatischer Assistent. Ich antworte mit den Hilfeartikeln von CloudServe; was "
                    "ich nicht beantworten kann, übernimmt eine Person aus dem Support-Team.",
        "needs_detail": "Tut mir leid, dass etwas nicht funktioniert; ich helfe gern. Können Sie mir etwas mehr "
                        "sagen: Welcher Teil von CloudServe ist betroffen (Deployment, API, Anmeldung oder "
                        "Abrechnung), was wollten Sie tun, und welche Fehlermeldung sehen Sie?",
        "out_of_scope": "Ich kann nur bei Fragen zu den Diensten von CloudServe helfen, etwa Deployments, APIs, "
                        "Anmeldung, Abrechnung, Daten oder Ihr Konto. Ihre Nachricht scheint nicht dazu zu "
                        "gehören, daher habe ich sie nicht an das Team weitergeleitet. Falls es um CloudServe geht, "
                        "beschreiben Sie es bitte etwas genauer.",
    },
    "it": {
        "greet": "Ciao! Sono l'assistente di supporto automatico di CloudServe. Dimmi di cosa hai bisogno (un "
                 "deployment, una chiave API, l'accesso, la fatturazione o il tuo account) e troverò l'articolo "
                 "giusto o ti metterò in contatto con il team.",
        "how": "Bene, grazie per averlo chiesto! Sono l'assistente di supporto automatico di CloudServe. Come "
               "posso aiutarti oggi?",
        "thanks": "Prego. Se ti serve altro, scrivimi qui.",
        "bye": "Grazie per averci contattato. Arrivederci!",
        "ack": "Posso aiutarti con qualcos'altro?",
        "identity": "Sono un assistente automatico. Rispondo con gli articoli di aiuto di CloudServe; quello a cui "
                    "non posso rispondere passa a una persona del team di supporto.",
        "needs_detail": "Mi dispiace che qualcosa non funzioni; voglio aiutarti. Puoi dirmi qualcosa in più: quale "
                        "parte di CloudServe è coinvolta (un deployment, l'API, l'accesso o la fatturazione), cosa "
                        "stavi cercando di fare e quale messaggio di errore vedi?",
        "out_of_scope": "Posso aiutarti solo con domande sui servizi di CloudServe, come deployment, API, accesso, "
                        "fatturazione, dati o il tuo account. Il tuo messaggio non sembra riguardare questi temi, "
                        "quindi non l'ho inoltrato al team. Se riguarda CloudServe, dammi qualche dettaglio in più.",
    },
    "pt": {
        "greet": "Olá! Sou o assistente de suporte automático da CloudServe. Diga-me em que precisa de ajuda (um "
                 "deploy, uma chave de API, o login, a faturação ou a sua conta) e encontrarei o artigo de ajuda "
                 "certo ou passarei o pedido à equipa.",
        "how": "Muito bem, obrigado por perguntar! Sou o assistente de suporte automático da CloudServe. Em que "
               "posso ajudar hoje?",
        "thanks": "De nada. Se precisar de mais alguma coisa, escreva aqui.",
        "bye": "Obrigado pelo contacto. Até logo!",
        "ack": "Posso ajudar com mais alguma coisa?",
        "identity": "Sou um assistente automático. Respondo com os artigos de ajuda da CloudServe; o que não "
                    "consigo responder passa para uma pessoa da equipa de suporte.",
        "needs_detail": "Lamento que algo não esteja a funcionar; quero ajudar. Pode dizer-me um pouco mais: que "
                        "parte da CloudServe está afetada (um deploy, a API, o login ou a faturação), o que estava "
                        "a tentar fazer e que mensagem de erro aparece?",
        "out_of_scope": "Só posso ajudar com perguntas sobre os serviços da CloudServe, como deploys, APIs, login, "
                        "faturação, dados ou a sua conta. A sua mensagem não parece ser sobre isso, por isso não a "
                        "enviei à equipa. Se for sobre a CloudServe, dê-me um pouco mais de detalhe.",
    },
}


def reply(kind: str, language: str = "en") -> str:
    """The fixed reply for a kind of message, in the customer's language where available."""
    return REPLIES.get(language, REPLIES["en"]).get(kind) or REPLIES["en"][kind]
