#!/usr/bin/env python3
"""Generate the same spoken passage with every voice exposed by an MLX Audio server."""

from __future__ import annotations

import argparse
import http.client
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path


DEFAULT_TEXT = (
    "Good evening. Denver has a twenty to thirty-five percent chance of scattered showers "
    "between five and ten p.m. I'd bring a light rain jacket, just in case."
)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "voice"


def _get_voices(base_url: str, model: str) -> list[str]:
    url = f"{base_url.rstrip('/')}/audio/voices?{urllib.parse.urlencode({'model': model})}"
    with urllib.request.urlopen(url, timeout=30) as response:
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
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=180) as response:
        audio = response.read()
    return audio, time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:5092/v1")
    parser.add_argument("--model", default="mlx-community/Kokoro-82M-bf16")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--speed", type=float, default=0.95)
    parser.add_argument("--voice", action="append", dest="voices", help="Test only this voice; repeatable")
    parser.add_argument("--output", type=Path, default=Path("tts-auditions"))
    args = parser.parse_args()

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
