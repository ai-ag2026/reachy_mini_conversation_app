"""Normalize assistant text for natural speech synthesis."""

from __future__ import annotations

import re


_CODE_FENCE_RE = re.compile(r"```(?:[\w+-]+)?\s*(.*?)```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_MARKDOWN_LINK_RE = re.compile(r"\[([^]]+)]\((?:https?://|mailto:)[^)]+\)")
_URL_RE = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)
_ABSOLUTE_APP_PATH_RE = re.compile(r"(?<!\w)/(?:Applications|Users|private|var|tmp)/\S+")
_REPEAT_COUNTER_RE = re.compile(r"\(\s*[×x]\s*\d+\s*\)", re.IGNORECASE)
_NUMERIC_RANGE_RE = re.compile(r"(?<=\d)\s*[–—]\s*(?=\d)")
_LIST_MARKER_RE = re.compile(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+")
_HEADING_RE = re.compile(r"(?m)^\s*#{1,6}\s*")
_EMPHASIS_RE = re.compile(r"(?<!\w)[*_]{1,2}|[*_]{1,2}(?!\w)")
_SPACE_RE = re.compile(r"\s+")


def normalize_for_speech(text: str) -> str:
    """Convert display-oriented assistant text into plain, pronounceable English."""
    clean = str(text or "").strip()
    if not clean:
        return ""
    clean = _CODE_FENCE_RE.sub(lambda match: match.group(1), clean)
    clean = _INLINE_CODE_RE.sub(lambda match: match.group(1), clean)
    clean = _MARKDOWN_LINK_RE.sub(lambda match: match.group(1), clean)
    clean = _URL_RE.sub("the link", clean)
    clean = _ABSOLUTE_APP_PATH_RE.sub(
        lambda match: "the application" + (match.group(0)[-1] if match.group(0)[-1] in ".,;!?" else ""),
        clean,
    )
    clean = _REPEAT_COUNTER_RE.sub("", clean)
    clean = _HEADING_RE.sub("", clean)
    clean = _LIST_MARKER_RE.sub("", clean)
    clean = _EMPHASIS_RE.sub("", clean)
    clean = _NUMERIC_RANGE_RE.sub(" to ", clean)
    clean = clean.replace("%", " percent")
    clean = clean.replace("&", " and ")
    clean = re.sub(r"(?<=\d)°(?=\s*[FCfc]\b)", " degrees ", clean)
    clean = re.sub(r"(?<=\d)°", " degrees", clean)
    clean = _SPACE_RE.sub(" ", clean).strip(" ,;:-")
    return clean
