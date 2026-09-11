"""Configuration parity for the standalone TTS audition utility."""

import sys
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def audition():
    """Load the utility without running its CLI."""
    spec = importlib.util.spec_from_file_location(
        "tts_voice_audition", Path(__file__).parents[1] / "scripts" / "tts_voice_audition.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dotenv_searches_cwd_and_overrides_environment(audition, monkeypatch, tmp_path):
    """The nearest cwd ancestor .env overrides the shell, as in the app."""
    (tmp_path / ".env").write_text("AGENT_QWEN_TTS_MODEL=dotenv-model\n")
    child = tmp_path / "child"
    child.mkdir()
    monkeypatch.chdir(child)
    monkeypatch.setenv("AGENT_QWEN_TTS_MODEL", "shell-model")
    audition._load_dotenv()
    assert audition.os.environ["AGENT_QWEN_TTS_MODEL"] == "dotenv-model"


@pytest.mark.parametrize("speed,expected", [("0.1", 0.5), ("9", 2.0), ("1.2", 1.2)])
def test_cli_bounds_speed(audition, monkeypatch, tmp_path, speed, expected):
    """The actual synthesis payload uses the same bounds as the app."""
    monkeypatch.setattr(audition, "_load_dotenv", lambda: None)
    payloads = []

    def synthesize(url, payload):
        payloads.append(payload)
        return b"audio", 0.1

    monkeypatch.setattr(audition, "_synthesize", synthesize)
    monkeypatch.setattr(sys, "argv", ["audition", "--speed", speed, "--voice", "voice", "--output", str(tmp_path)])
    assert audition.main() == 0
    assert payloads[0]["speed"] == expected
