"""Normalize assistant text for natural speech synthesis."""

from __future__ import annotations
import re


_CODE_FENCE_RE = re.compile(r"```(?:[\w+-]+)?\s*(.*?)```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_MARKDOWN_LINK_RE = re.compile(r"\[([^]]+)]\((?:https?://|mailto:)[^)]+\)")
_URL_RE = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)
_ABSOLUTE_PATH_RE = re.compile(r"(?<!\w)/(?:Applications|Users|private|var|tmp)/\S+")
_REPEAT_COUNTER_RE = re.compile(r"\(\s*[×x]\s*\d+\s*\)", re.IGNORECASE)
_NUMERIC_RANGE_RE = re.compile(r"(?<=\d)\s*–\s*(?=\d)")
_LIST_MARKER_RE = re.compile(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+")
_HEADING_RE = re.compile(r"(?m)^\s*#{1,6}\s*")
_EMPHASIS_RE = re.compile(r"(?<!\w)(\*\*|__|\*|_)(?=\S)(.+?)(?<=\S)\1(?!\w)")
_SPACE_RE = re.compile(r"\s+")


def _trailing_punctuation(value: str) -> str:
    return value[len(value.rstrip(".,;:!?)]")) :]


def normalize_for_speech(text: str) -> str:
    """Convert display-oriented assistant text into plain, pronounceable English."""
    clean = str(text or "").strip()
    if not clean:
        return ""
    clean = _CODE_FENCE_RE.sub(lambda match: match.group(1), clean)
    clean = _INLINE_CODE_RE.sub(lambda match: match.group(1), clean)
    clean = _MARKDOWN_LINK_RE.sub(lambda match: match.group(1), clean)
    clean = _URL_RE.sub(lambda match: "the link" + _trailing_punctuation(match.group()), clean)
    clean = _ABSOLUTE_PATH_RE.sub(
        lambda match: (
            "the application" if re.search(r"\.app(?:/|$)", match.group().rstrip(".,;:!?)]")) else "a file path"
        )
        + _trailing_punctuation(match.group()),
        clean,
    )
    clean = _REPEAT_COUNTER_RE.sub("", clean)
    clean = _HEADING_RE.sub("", clean)
    clean = _LIST_MARKER_RE.sub("", clean)
    clean = _EMPHASIS_RE.sub(lambda match: match.group(2), clean)
    clean = _NUMERIC_RANGE_RE.sub(" to ", clean)
    clean = clean.replace("−", "minus ")
    clean = clean.replace("%", " percent")
    clean = clean.replace("&", " and ")
    clean = re.sub(
        r"(?<=\d)°\s*([FCfc])\b",
        lambda match: " degrees " + ("Celsius" if match.group(1).lower() == "c" else "Fahrenheit"),
        clean,
    )
    clean = re.sub(r"(?<=\d)°", " degrees", clean)
    clean = _SPACE_RE.sub(" ", clean).strip(" ,;:")
    clean = re.sub(r"^-(?!\d)", "", clean).rstrip(" -").strip()
    return clean
