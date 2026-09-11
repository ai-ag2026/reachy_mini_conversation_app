"""Tests for text prepared for local speech synthesis."""

import pytest

from reachy_mini_conversation_app.speech_text import normalize_for_speech


def test_normalize_for_speech_handles_weather_notation() -> None:
    """Verify normalize for speech handles weather notation."""
    text = "There is a 20–35% chance from 5–10 p.m., around 72°F."
    assert normalize_for_speech(text) == (
        "There is a 20 to 35 percent chance from 5 to 10 p.m., around 72 degrees Fahrenheit."
    )


def test_normalize_for_speech_removes_display_only_syntax() -> None:
    """Verify normalize for speech removes display only syntax."""
    text = "## Result\n- Open [Bear](https://bear.app/) at `/Applications/Bear.app/Contents/MacOS`. (×3)"
    assert normalize_for_speech(text) == "Result Open Bear at the application."


def test_normalize_for_speech_preserves_plain_conversation() -> None:
    """Verify normalize for speech preserves plain conversation."""
    text = "That's a good question. I'd bring a light jacket."
    assert normalize_for_speech(text) == text


@pytest.mark.parametrize(
    "text,expected",
    [
        ("-5°C tonight.", "-5 degrees Celsius tonight."),
        ("-3% on the day.", "-3 percent on the day."),
        ("−5°C tonight.", "minus 5 degrees Celsius tonight."),
        ("- Intro -", "Intro"),
        ("Visit www.example.com.", "Visit the link."),
        ("Visit https://example.com/search?q=weather&units=c.", "Visit the link."),
        ("(https://example.com/path).", "(the link)."),
        ("Open /Users/me/notes.txt.", "Open a file path."),
        ("Open /tmp/report.csv.", "Open a file path."),
        ("Open /private/var/report.csv.", "Open a file path."),
        ("Open /Users/me/Tools/Editor.app/Contents/MacOS.", "Open the application."),
        ("5 * 3 = 15.", "5 * 3 = 15."),
        ("This is **bold** and *emphasized*.", "This is bold and emphasized."),
        ("Call 555—1234.", "Call 555—1234."),
    ],
)
def test_normalization_preserves_meaning(text, expected):
    """Preserve numerical signs, arithmetic, punctuation, and the kind of path."""
    assert normalize_for_speech(text) == expected
