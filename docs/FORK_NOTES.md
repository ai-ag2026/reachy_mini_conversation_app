# Fork notes — generic Local Agent backend

This is a fork of [`pollen-robotics/reachy_mini_conversation_app`](https://github.com/pollen-robotics/reachy_mini_conversation_app)
that adds a **self-hostable `local` backend**: the app is the robot's body (mic/VAD/STT,
TTS/speaker, camera/vision, movement, aliveness) while the *brain* is any OpenAI-compatible
agent you host yourself.

## What this fork adds over upstream

- A `local` value for `BACKEND_PROVIDER`, registered alongside `openai` / `gemini` / `huggingface`.
- A voice handler (`agent_voice_handler.py`) with barge-in and a semantic interrupt gate.
- OpenAI-compatible client seams (`agent_clients.py`): chat brain, TTS, optional low-latency lead-in.
- A one-shot vision tool (`tools/agent_vision.py`) and safe, bounded head-movement tools
  (`tools/agent_safe_movement.py`, `agent_movement_policy.py`) — antennas and body yaw are always
  preserved; deltas are clamped.
- An "aliveness" layer: idle motion, companion reactions, emotion cues, and speech-reactive sway
  (`liveliness.py`, `companion.py`, `emotion_cues.py`, `body_surface.py`).
- Neutral `local-agent` / `local-agent-work` profiles; bring your own via the external-profiles mechanism.

## Configuration

Everything is driven by `AGENT_*` environment variables (see `.env.example`), all defaulting to
`127.0.0.1` placeholders. No real endpoints, hostnames, credentials, or persona ship in this repo.
Point the variables at your own STT / TTS / agent services to run it.

## Safety defaults

The movement policy clamps head deltas and preserves antennas + body yaw. Vision is one-shot and does
not persist raw frames. Live robot actions run through the same policy checks regardless of backend.
