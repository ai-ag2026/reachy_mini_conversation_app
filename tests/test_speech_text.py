"""Tests for text prepared for local speech synthesis."""

from reachy_mini_conversation_app.speech_text import normalize_for_speech


def test_normalize_for_speech_handles_weather_notation() -> None:
    text = "There is a 20–35% chance from 5–10 p.m., around 72°F."
    assert normalize_for_speech(text) == (
        "There is a 20 to 35 percent chance from 5 to 10 p.m., around 72 degrees F."
    )


def test_normalize_for_speech_removes_display_only_syntax() -> None:
    text = "## Result\n- Open [Bear](https://bear.app/) at `/Applications/Bear.app/Contents/MacOS`. (×3)"
    assert normalize_for_speech(text) == "Result Open Bear at the application."


def test_normalize_for_speech_preserves_plain_conversation() -> None:
    text = "That's a good question. I'd bring a light jacket."
    assert normalize_for_speech(text) == text
