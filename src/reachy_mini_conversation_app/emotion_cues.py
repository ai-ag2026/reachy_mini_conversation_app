"""Turn-coupled emotion cues (gap-map Stufe 2, 2026-07-02).

Conservative keyword heuristic mapping a finished AGENT answer (plus the user's utterance) to
one of the app's emotion INTENTS (play_emotion resolves intents to curated library moves).
Deliberately sparse: only clear signals fire, everything else returns None — a robot that
emotes on every sentence reads as twitchy, not alive. The brain-driven variant (AGENT choosing
emotes itself) is the Stufe-3 body-tool surface; this is the local reflex layer.
"""

from __future__ import annotations

import re

# Ordered: first match wins. Patterns are matched case-insensitively against
# "user_text ||| answer_text".
_CUES: tuple[tuple[str, str], ...] = (
    # user-side greetings/goodbyes (the answer usually mirrors them, either side may hit)
    (r"\b(hallo|hi|hey|guten morgen|guten tag|guten abend|willkommen)\b", "greeting"),
    (r"\b(tsch(ü|ue)ss|auf wiedersehen|bis (sp(ä|ae)ter|morgen|bald)|gute nacht)\b", "goodbye"),
    (r"\b(danke|dankesch(ö|oe)n|vielen dank)\b", "grateful"),
    # answer-side affect
    (r"\b(haha|hihi|witz|lustig|zum lachen|k(ö|oe)stlich)\b", "laughing"),
    (r"\b(super|perfekt|klasse|ausgezeichnet|erledigt|geschafft|fertig!)\b", "success"),
    (r"\b(leider|tut mir leid|bedauerlich|schade|misslungen|fehlgeschlagen)\b", "downcast"),
    (r"\b((ü|ue)berrascht|wow|erstaunlich|unglaublich|tats(ä|ae)chlich\?)\b", "amazed"),
    (r"\b(vorsicht|achtung|warnung|riskant|gef(ä|ae)hrlich)\b", "anxious"),
    (r"\b(verwirrend|verstehe nicht|unklar|keine ahnung)\b", "confused"),
    (r"\b(ja[.!]|genau[.!]|richtig[.!]|stimmt[.!])", "yes"),
    (r"\b(nein[.!]|falsch[.!]|leider nein)\b", "no"),
)

_COMPILED = tuple((re.compile(pat, re.IGNORECASE), intent) for pat, intent in _CUES)


def emotion_for_turn(user_text: str, answer_text: str) -> str | None:
    """Return an emotion intent for a completed turn, or None (= no emote, the normal case)."""
    hay = f"{(user_text or '')[:200]} ||| {(answer_text or '')[:400]}"
    for pattern, intent in _COMPILED:
        if pattern.search(hay):
            return intent
    return None
