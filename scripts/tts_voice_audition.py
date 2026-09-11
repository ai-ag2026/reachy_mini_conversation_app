#!/usr/bin/env python3
"""Generate the same spoken passage with every voice exposed by an OpenAI-compatible TTS server.

Defaults come from the same environment the app uses (AGENT_QWEN_TTS_BASE_URL, AGENT_QWEN_TTS_MODEL,
AGENT_TTS_SPEED, and the bearer token named by AGENT_QWEN_TTS_API_KEY_ENV), read from the process
environment and the nearest .env upward from cwd (.env overrides the shell); the MLX Audio / Kokoro values below are the fallbacks.
"""

from __future__ import annotations
import os
import re
import json
import time
import argparse
import http.client
import urllib.parse
import urllib.request
from pathlib import Path


DEFAULT_BASE_URL = "http://127.0.0.1:5092/v1"
DEFAULT_MODEL = "mlx-community/Kokoro-82M-bf16"
DEFAULT_SPEED = 0.95
DEFAULT_TEXT = (
    "Good evening. Denver has a twenty to thirty-five percent chance of scattered showers "
    "between five and ten p.m. I'd bring a light rain jacket, just in case."
)


def _load_dotenv() -> None:
    """Search upward from cwd and apply .env overrides, matching the app."""
    try:
        from dotenv import find_dotenv, load_dotenv
    except ImportError:
        return
    load_dotenv(find_dotenv(usecwd=True), override=True)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except ValueError:
        return default


def _headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    key = os.getenv(os.getenv("AGENT_QWEN_TTS_API_KEY_ENV", "AGENT_QWEN_TTS_API_KEY"), "").strip()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "voice"


def _get_voices(base_url: str, model: str) -> list[str]:
    url = f"{base_url.rstrip('/')}/audio/voices?{urllib.parse.urlencode({'model': model})}"
    request = urllib.request.Request(url, headers=_headers())
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    entries = payload.get("data", payload) if isinstance(payload, dict) else payload
    voices: list[str] = []
    for entry in entries or []:
        voice = entry.get("id") or entry.get("name") if isinstance(entry, dict) else entry
        if voice:
            voices.append(str(voice))
    return voices


def _synthesize(base_url: str, payload: dict[str, object]) -> tuple[bytes, float]:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/audio/speech",
        data=json.dumps(payload).encode("utf-8"),
        headers=_headers(),
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=180) as response:
        audio = response.read()
    return audio, time.perf_counter() - started


def main() -> int:
    """Synthesize DEFAULT_TEXT (or --text) once per voice and write WAVs plus results.json."""
    _load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default=os.getenv("AGENT_QWEN_TTS_BASE_URL") or DEFAULT_BASE_URL)
    parser.add_argument("--model", default=os.getenv("AGENT_QWEN_TTS_MODEL") or DEFAULT_MODEL)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--speed", type=float, default=_env_float("AGENT_TTS_SPEED", DEFAULT_SPEED))
    parser.add_argument("--voice", action="append", dest="voices", help="Test only this voice; repeatable")
    parser.add_argument("--output", type=Path, default=Path("tts-auditions"))
    args = parser.parse_args()
    args.speed = min(2.0, max(0.5, args.speed))

    voices = args.voices or _get_voices(args.base_url, args.model)
    if not voices:
        parser.error("The server returned no voices; pass --voice explicitly")
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for voice in voices:
        try:
            audio, elapsed = _synthesize(
                args.base_url,
                {
                    "model": args.model,
                    "voice": voice,
                    "input": args.text,
                    "speed": args.speed,
                    "response_format": "wav",
                },
            )
            target = args.output / f"{_safe_name(voice)}.wav"
            target.write_bytes(audio)
            result = {"voice": voice, "seconds": round(elapsed, 3), "bytes": len(audio), "file": str(target)}
            print(f"{voice}: {elapsed:.2f}s -> {target}", flush=True)
        except (OSError, TimeoutError, http.client.HTTPException) as exc:
            result = {"voice": voice, "error": f"{type(exc).__name__}: {exc}"}
            print(f"{voice}: ERROR {exc}", flush=True)
        results.append(result)
        (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
