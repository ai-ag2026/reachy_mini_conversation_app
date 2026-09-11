---
title: Reachy Mini Conversation App
emoji: 🎤
colorFrom: red
colorTo: blue
sdk: static
pinned: false
short_description: Talk with Reachy Mini!
suggested_storage: large
tags:
 - reachy_mini
 - reachy_mini_python_app
---

# Reachy Mini conversation app

Conversational app for the Reachy Mini robot combining realtime voice backends, vision pipelines, and choreographed motion libraries.

![Reachy Mini Dance](docs/assets/reachy_mini_dance.gif)

## Table of contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the app](#running-the-app)
- [LLM tools](#llm-tools-exposed-to-the-assistant)
- [Advanced features](#advanced-features)
- [Contributing](#contributing)
- [License](#license)

## Overview
- Real-time audio conversation loop with `fastrtc` for low-latency streaming. Supported backends:
  - **Hugging Face** - default, using the built-in Hugging Face server or your own local endpoint.
  - **OpenAI Realtime** (`gpt-realtime-2`) - requires `OPENAI_API_KEY`.
  - **Gemini Live** (`gemini-3.1-flash-live-preview`) - requires `GEMINI_API_KEY`.
  - **Local Agent** (`local`) - route the brain to your own self-hosted, OpenAI-compatible agent with local STT/TTS. Fully self-hostable, no cloud key required. See [Local Agent backend](#local-agent-backend).
- Vision processing uses the selected realtime backend by default (when the camera tool is used), with optional on-device local vision using SmolVLM2 (CPU/GPU/MPS) via `--local-vision`.
- Layered motion system queues primary moves (dances, emotions, goto poses, breathing) while blending speech-reactive wobble and head-tracking.
- Async tool dispatch integrates robot motion, camera capture, and optional head-tracking capabilities through a Gradio web UI with live transcripts.

## Architecture

The app follows a layered architecture connecting the user, AI services, and robot hardware:

<p align="center">
  <img src="docs/assets/conversation_app_arch.svg" alt="Architecture Diagram" width="600"/>
</p>

## Installation

> [!IMPORTANT]
> Before using this app, you need to install [Reachy Mini's SDK](https://github.com/pollen-robotics/reachy_mini/).<br>
> Windows support is currently experimental and has not been extensively tested. Use with caution.

<details open>
<summary><b>Using uv (recommended)</b></summary>

Set up the project quickly using [uv](https://docs.astral.sh/uv/):

```bash
# macOS (Homebrew)
uv venv --python /opt/homebrew/bin/python3.12 .venv

# Linux / Windows (Python in PATH)
uv venv --python python3.12 .venv

source .venv/bin/activate
uv sync
```

> **Note:** To reproduce the exact dependency set from this repo's `uv.lock`, run `uv sync --frozen`. This ensures `uv` installs directly from the lockfile without re-resolving or updating any versions.

**Install optional features:**
```bash
uv sync --extra local_vision         # Local PyTorch/Transformers vision
uv sync --extra yolo_vision          # YOLO face-detection backend for head tracking
uv sync --extra mediapipe_vision     # MediaPipe-based head-tracking
uv sync --extra all_vision           # All vision features
```

Combine extras or include dev dependencies:
```bash
uv sync --extra all_vision --group dev
```

</details>

<details>
<summary><b>Using pip</b></summary>

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

**Install optional features:**
```bash
pip install -e .[local_vision]          # Local vision stack
pip install -e .[yolo_vision]           # YOLO face-detection backend for head tracking
pip install -e .[mediapipe_vision]      # MediaPipe-based vision
pip install -e .[remote_tools]          # Hugging Face Space tools over MCP
pip install -e .[all_vision]            # All vision features
pip install -e .[dev]                   # Development tools
```

Some wheels (like PyTorch) are large and require compatible CUDA or CPU builds—make sure your platform matches the binaries pulled in by each extra.

</details>

### Optional dependency groups

| Extra | Purpose | Notes |
|-------|---------|-------|
| `local_vision` | Run the local VLM (SmolVLM2) through PyTorch/Transformers | GPU recommended. Ensure compatible PyTorch builds for your platform. |
| `yolo_vision` | YOLOv11n face detection via `ultralytics` and `supervision` | Used as the `yolo` head-tracking backend. Runs on CPU (default). GPU improves performance. |
| `mediapipe_vision` | Lightweight landmark tracking with MediaPipe | Works on CPU. Enables `--head-tracker mediapipe`. |
| `all_vision` | Convenience alias installing every vision extra | Install when you want the flexibility to experiment with every provider. |
| `dev` | Developer tooling (`pytest`, `ruff`, `mypy`) | Development-only dependencies. Use `--group dev` with uv or `[dev]` with pip. |

**Note:** `dev` is a dependency group (not an optional dependency). With uv, use `--group dev`. With pip, use `[dev]`.

## Configuration

The default setup uses the Hugging Face backend and does not require an API key.

Copy `.env.example` to `.env` when you want to switch backends, provide API keys, or point Hugging Face at your own local endpoint.

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | Required for OpenAI Realtime mode. |
| `GEMINI_API_KEY` | Required for Gemini mode. Also accepts `GOOGLE_API_KEY`. Get one at [aistudio.google.com](https://aistudio.google.com/apikey). |
| `BACKEND_PROVIDER` | Realtime backend to use: `huggingface` (default), `openai`, `gemini`, or `local`. |
| `MODEL_NAME` | Optional model override for OpenAI Realtime or Gemini Live. Defaults to `gpt-realtime-2` for OpenAI and `gemini-3.1-flash-live-preview` for Gemini. Hugging Face uses the server's model selection. |
| `REALTIME_TRANSCRIPTION_LANGUAGE` | Optional input transcription language for realtime backends. Defaults to `en`; set to a backend-supported code such as `zh` for Chinese. |
| `HF_REALTIME_CONNECTION_MODE` | Hugging Face connection selector: `deployed` uses the built-in Hugging Face server; `local` uses `HF_REALTIME_WS_URL`. Defaults to `deployed`. |
| `HF_REALTIME_WS_URL` | Direct websocket endpoint for your own Hugging Face backend. Accepts either a base URL like `ws://127.0.0.1:8765/v1` or the full websocket URL `ws://127.0.0.1:8765/v1/realtime`. Used when `HF_REALTIME_CONNECTION_MODE=local`. |
| `HF_HOME` | Cache directory for local Hugging Face downloads (only used with `--local-vision` flag, defaults to `./cache`). |
| `HF_TOKEN` | Optional token for Hugging Face access (for gated/private assets). |
| `AGENT_BASE_URL` | Local Agent brain: base URL of your OpenAI-compatible agent (`{url}/chat/completions`). Defaults to `http://127.0.0.1:8642/v1`. |
| `AGENT_MODEL` | Model name sent to your agent. Defaults to `local-agent`. |
| `AGENT_STT_BASE_URL` | Local speech-to-text endpoint (OpenAI-compatible). Defaults to `http://127.0.0.1:5092/v1`. |
| `AGENT_STT_LANGUAGE` | Language code sent to local STT. Defaults to `en`; set another code, or `auto` to let the STT model detect the language. |
| `AGENT_QWEN_TTS_BASE_URL` / `AGENT_QWEN_TTS_VOICE` | Local text-to-speech endpoint and voice name. Defaults to `http://127.0.0.1:7034/v1` and `default`. |
| `AGENT_TTS_SPEED` / `AGENT_OUTPUT_GAIN` | Synthesis pace (sent to the TTS server, clamped to `0.5`–`2.0`) and post-synthesis loudness. Defaults to `1.0` and `0.9`; `0.95` speed often sounds more natural. |
| `AGENT_PIPELINE_MONITOR` / `AGENT_PIPELINE_MONITOR_PORT` | Loopback-only live STT/LLM/TTS/tool monitor and its port. Defaults to `1` (on) and `8766`. |
| `AGENT_PIPELINE_MONITOR_LOG_CONTENT` | Defaults to `0`: monitor log lines record stage, text length and metadata only. Set `1` to also log the transcript/assistant text. |
| `AGENT_PLATFORM_API_KEY` / `AGENT_PLATFORM_API_KEY_FILE` | Shared secret sent in the platform `hello` frame (the key, or an absolute path to a file containing it; the nonempty variable wins). File contents are stripped and must match the gateway's `REACHY_WS_API_KEY_FILE` (or `REACHY_WS_API_KEY`). A 1008 authentication rejection stops reconnects until the app is restarted. |
| `REACHY_MINI_HOST` | Reachy daemon host/IP for network connection (same as `--robot-host`). Unset = SDK auto-detection. |
| `AGENT_IDLE_BREATHING` / `AGENT_IDLE_ACTIONS` | Idle breathing and random idle movement. Both default to `1` (on); set `0` for quiet microphone operation. |
| `AGENT_SPEECH_WOBBLE` / `AGENT_SPEECH_SWAY` | Speech-driven head wobble and antenna sway. Both default to `1` (on); set `0` to disable. |
| `AGENT_TURN_EMOTES` / `AGENT_ORIENT_TO_SPEAKER` | End-of-turn emotions and microphone-triggered head turns. Both default to `1` (on); set `0` to disable. |
| `AGENT_THINKING_CUE` / `AGENT_THINKING_CUE_MAX_DEG` | Subtle antenna-only cue while waiting for the backend, and its amplitude. Defaults to `1` (on) and `2.0` degrees. |
| `AGENT_ANTENNA_REST_DEG` | Outward antenna rest bias that avoids servo hunting at zero. Defaults to `10` (clamped to `0`–`20`). |
| `AGENT_DAEMON_BASE_URL` | Reachy daemon HTTP API for playback/movement/status. Defaults to `http://127.0.0.1:8000`. |
| `LOCAL_VISION_MODEL` | Hugging Face model path for local vision processing (only used with `--local-vision` flag, defaults to `HuggingFaceTB/SmolVLM2-2.2B-Instruct`). |

### Local Agent backend

The `local` backend turns this app into the **body** for a brain you host yourself. Instead of a cloud realtime model, the loop is:

```
mic → VAD → local STT → your OpenAI-compatible agent (the brain) → local TTS → speaker
                                    ↕
              full body tool surface: camera/vision, head movement, dances,
              emotions, antennas, breathing, speech-reactive wobble, idle aliveness
```

Everything runs against endpoints **you** configure — no cloud key, no vendor lock-in. The full body/functionality is retained: voice (barge-in + semantic gate), one-shot vision, safe bounded head movement, and the aliveness layer (idle motion, companion reactions, emotion cues, speech sway) are all active and driven by your agent's replies and tool calls.

With the local backend running, open `http://127.0.0.1:8766` to watch the live pipeline. The page shows
final STT text, sentence-level LLM output, the exact text submitted to TTS, body tool calls/results, and
stage timing. Text is shown verbatim (it is not redacted), which is why the server only binds to loopback
and keeps a small in-memory history; raw audio is not captured. Log lines carry stage, length and
metadata only unless `AGENT_PIPELINE_MONITOR_LOG_CONTENT=1`. Set `AGENT_PIPELINE_MONITOR=0` to disable it.

To audition every voice exposed by the configured TTS model with identical text and collect per-voice
synthesis timings:

```bash
.venv/bin/python scripts/tts_voice_audition.py --output /private/tmp/reachy-tts-auditions
```

The utility reads `AGENT_QWEN_TTS_BASE_URL`, `AGENT_QWEN_TTS_MODEL` and `AGENT_TTS_SPEED` (from the
environment or `.env`; falling back to a local MLX Audio Kokoro server) and writes one WAV per voice plus
`results.json`. Use `--voice NAME` to benchmark a shortlist.

Minimal `.env` to get started (see `.env.example` for the full list):

```env
BACKEND_PROVIDER=local
AGENT_BASE_URL=http://your-agent-host:8642/v1     # any server exposing /chat/completions
AGENT_STT_BASE_URL=http://your-stt-host:5092/v1   # OpenAI-compatible transcription
AGENT_QWEN_TTS_BASE_URL=http://your-tts-host:7034/v1
AGENT_QWEN_TTS_VOICE=default
REACHY_MINI_CUSTOM_PROFILE=local-agent
```

**Persona.** This fork ships a neutral `local-agent` profile (full mode) and a `local-agent-work` profile (safety-restricted toolset). To give your robot its own character, drop a profile folder into `external_content/external_profiles/` and select it with `REACHY_MINI_CUSTOM_PROFILE`, or point `REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY` at your own directory — no code changes needed.

### Hugging Face Connection Modes

Use the built-in Hugging Face server through the app-managed Space proxy. This is the default for a new install; set it explicitly only when you want to switch back from a saved local endpoint:

```env
BACKEND_PROVIDER=huggingface
HF_REALTIME_CONNECTION_MODE=deployed
```

Run your own realtime voice backend using [speech-to-speech](https://github.com/huggingface/speech-to-speech) on the same machine as the conversation app:

```env
BACKEND_PROVIDER=huggingface
HF_REALTIME_CONNECTION_MODE=local
HF_REALTIME_WS_URL=ws://127.0.0.1:8765/v1/realtime
```

Run your own Hugging Face backend on your laptop and connect to it from Reachy Mini Wireless over the same Wi-Fi network:

```env
BACKEND_PROVIDER=huggingface
HF_REALTIME_CONNECTION_MODE=local
HF_REALTIME_WS_URL=ws://<your-laptop-lan-ip>:8765/v1/realtime
```

For that LAN setup, make sure the backend listens on an address reachable from the robot, not only on `127.0.0.1`.

If the backend stays bound to loopback on your laptop, you can forward it into the robot over SSH instead:

```bash
ssh -N -R 8765:127.0.0.1:8765 <robot-user>@<robot-host>
```

Then set this on the robot:

```env
BACKEND_PROVIDER=huggingface
HF_REALTIME_CONNECTION_MODE=local
HF_REALTIME_WS_URL=ws://127.0.0.1:8765/v1/realtime
```

When using the headless settings UI, selecting `Hugging Face` lets you choose either the built-in server or a local `host:port` target. The UI writes `HF_REALTIME_CONNECTION_MODE` for you, and the local path writes `HF_REALTIME_WS_URL` with a default of `localhost:8765`.

## Running the app

Activate your virtual environment, then launch:

```bash
reachy-mini-conversation-app
```

> [!TIP]
> Make sure the Reachy Mini daemon is running before launching the app. If you see a `TimeoutError`, it means the daemon isn't started. See [Reachy Mini's SDK](https://github.com/pollen-robotics/reachy_mini/) for setup instructions.

The app runs in console mode by default. Add `--gradio` to launch a web UI at http://127.0.0.1:7860/ (required for simulation mode). Vision and head-tracking options are described in the CLI table below.

### CLI options

| Option | Default | Description |
|--------|---------|-------------|
| `--head-tracker {yolo,mediapipe}` | `None` | Select a head-tracking backend when a camera is available. `yolo` uses a local YOLO face detector, `mediapipe` comes from the `reachy_mini_toolbox` package. Requires the matching optional extra. |
| `--no-camera` | `False` | Run without camera capture or head tracking. |
| `--local-vision` | `False` | Use the local vision model (SmolVLM2) for camera-tool requests instead of the selected realtime backend. Requires `local_vision` extra to be installed. |
| `--gradio` | `False` | Launch the Gradio web UI. Without this flag, runs in console mode. Required when running in simulation mode. |
| `--robot-name` | `None` | Optional. Connect to a specific robot by name when running multiple daemons on the same subnet. See [Multiple robots on the same subnet](#advanced-features). |
| `--debug` | `False` | Enable verbose logging for troubleshooting. |

### Examples

```bash
# Run with MediaPipe head tracking
reachy-mini-conversation-app --head-tracker mediapipe

# Run with the YOLO face-detection backend for head tracking
reachy-mini-conversation-app --head-tracker yolo

# Run with local vision processing (requires local_vision extra)
reachy-mini-conversation-app --local-vision

# Audio-only conversation (no camera)
reachy-mini-conversation-app --no-camera

# Launch with Gradio web interface
reachy-mini-conversation-app --gradio
```

> [!WARNING]
> `--local-vision` is not supported when running the conversation app directly on Reachy Mini Wireless / the Raspberry Pi. For local vision, keep the daemon running on the robot and start the conversation app from your laptop or workstation instead.

## LLM tools exposed to the assistant

| Tool | Action | Dependencies |
|------|--------|--------------|
| `move_head` | Queue a head pose change (left/right/up/down/front). | Core install only. |
| `camera` | Capture the latest camera frame and analyze it with the selected realtime backend or the local vision model. | Requires camera worker. Uses local vision when `--local-vision` is enabled. |
| `head_tracking` | Enable or disable head-tracking offsets (not identity recognition - only detects and tracks head position). | Camera worker with configured head tracker (`--head-tracker`). |
| `dance` | Queue a dance from `reachy_mini_dances_library`. | Core install only. |
| `stop_dance` | Clear queued dances. | Core install only. |
| `play_emotion` | Play a recorded emotion clip via Hugging Face datasets. | Core install only. Uses the default open emotions dataset: [`pollen-robotics/reachy-mini-emotions-library`](https://huggingface.co/datasets/pollen-robotics/reachy-mini-emotions-library). |
| `stop_emotion` | Clear queued emotions. | Core install only. |
| `remember` | Save one short, stable fact about the user for future sessions. | Core install only. Stored in the app instance data directory. |
| `forget` | Remove a saved memory fact by matching a short query. | Core install only. |
| `idle_do_nothing` | Explicitly remain idle during an idle turn. Not intended for normal conversation turns. | Core install only. |

> [!NOTE]
> `remember`/`forget` facts are stored in `memory.v1.json` inside the app's instance data directory (`~/.local/share/reachy_mini_conversation_app/` by default, or the instance path used by the desktop launcher). `forget` only removes facts matched by query. To reset all remembered facts, delete this file.

## Advanced features

Built-in motion content is published as open Hugging Face datasets:
- Emotions: [`pollen-robotics/reachy-mini-emotions-library`](https://huggingface.co/datasets/pollen-robotics/reachy-mini-emotions-library)
- Dances: [`pollen-robotics/reachy-mini-dances-library`](https://huggingface.co/datasets/pollen-robotics/reachy-mini-dances-library)

<details>
<summary><b>Custom profiles</b></summary>

Create custom profiles with dedicated instructions and enabled tools.

For normal usage, select a profile from the UI and save it for startup. That selection is persisted in `startup_settings.json`.

If no startup settings have been saved yet, you can still seed startup from the environment with `REACHY_MINI_CUSTOM_PROFILE=<name>` to load `profiles/<name>/`. If neither is set, the `default` profile is used.

Each profile should include `instructions.txt` (prompt text). `tools.txt` (list of allowed tools) is recommended. If missing for a non-default profile, the app falls back to `profiles/default/tools.txt`. Profiles can optionally contain custom tool implementations.

**Custom instructions:**

Write plain-text prompts in `instructions.txt`. To reuse shared prompt pieces, add lines like:
```
[passion_for_lobster_jokes]
[identities/witty_identity]
```
Each placeholder pulls the matching file under `src/reachy_mini_conversation_app/prompts/` (nested paths allowed). See `profiles/example/` for a reference layout.

**Enabling tools:**

List enabled tools in `tools.txt`, one per line. Prefix with `#` to comment out:
```
play_emotion
# move_head

# My custom tool defined locally
sweep_look
```
Tools are resolved first from Python files in the profile folder (custom tools), then from the core library `src/reachy_mini_conversation_app/tools/` (like `dance`, `head_tracking`).
Installed public Hugging Face Space tools can also be enabled here after you add them with `tool-spaces`.

**Custom tools:**

On top of built-in tools found in the core library, you can implement custom tools specific to your profile by adding Python files in the profile folder.
Custom tools must subclass `reachy_mini_conversation_app.tools.core_tools.Tool` (see `profiles/example/sweep_look.py`).

**Edit personalities from the UI:**

When running with `--gradio`, open the "Personality" accordion:
- Select among available profiles (folders under `profiles/`) or the built‑in default.
- Click "Apply" to update the current session instructions live.
- Create a new personality by entering a name and instructions text. It stores files under `profiles/<name>/` and copies `tools.txt` from the `default` profile.

Note: The "Personality" panel updates the conversation instructions. Tool sets are loaded at startup from `tools.txt` and are not hot‑reloaded.

</details>

<details>
<summary><b>Locked profile mode</b></summary>

To create a locked variant of the app that cannot switch profiles, edit `src/reachy_mini_conversation_app/config.py` and set the `LOCKED_PROFILE` constant to the desired profile name:
```python
LOCKED_PROFILE: str | None = "mars_rover"  # Lock to this profile
```
When `LOCKED_PROFILE` is set, the app always uses that profile, ignoring saved startup settings, `REACHY_MINI_CUSTOM_PROFILE`, and the Gradio UI. The UI shows "(locked)" and disables all profile editing controls.
This is useful for creating dedicated clones of the app with a fixed personality. Clone scripts can simply edit this constant to lock the variant.

</details>

<details>
<summary><b>External profiles and tools</b></summary>

You can extend the app with profiles/tools stored outside the repository defaults.

- Core profiles are under `profiles/`.
- Core tools are under `src/reachy_mini_conversation_app/tools/`.

**Recommended layout:**

```text
external_content/
├── external_profiles/
│   └── my_profile/
│       ├── instructions.txt
│       ├── tools.txt        # optional (see fallback behavior below)
│       └── voice.txt        # optional
├── external_tools/
│   └── my_custom_tool.py
└── installed_tool_spaces.json
```

**Environment variables:**

Set these values in your `.env` when you want env-driven external profile/tool selection:

```env
# Optional fallback/manual profile selector:
REACHY_MINI_CUSTOM_PROFILE=my_profile
REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY=./external_content/external_profiles
REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY=./external_content/external_tools
# Optional convenience mode:
# AUTOLOAD_EXTERNAL_TOOLS=1
```

**Loading behavior:**

- **Default/strict mode**: `tools.txt` defines enabled tools explicitly. Every name in `tools.txt` must resolve to either a built-in tool (`src/reachy_mini_conversation_app/tools/`) or an external tool module in `REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY`.
- **Convenience mode** (`AUTOLOAD_EXTERNAL_TOOLS=1`): all valid `*.py` tool files in `REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY` are auto-added.
- **External profile fallback**: if the selected external profile has no `tools.txt`, the app falls back to built-in `profiles/default/tools.txt`.
- **Duplicate safety**: every loaded tool class must expose a unique `Tool.name`. The app now fails fast if two tool implementations claim the same tool name.

This supports both:
1. Local external tools used with built-in/default profile.
2. Local external profiles used with built-in default tools.

</details>

<details>
<summary><b>Public Hugging Face Space tools</b></summary>

You can install public MCP-compatible Hugging Face Spaces as remote tool sources for this app.

```bash
# install + enable in active profile
reachy-mini-conversation-app tool-spaces add <owner/space-name>

# enable in a specific profile
reachy-mini-conversation-app tool-spaces add <owner/space-name> --profile NAME

# install without enabling
reachy-mini-conversation-app tool-spaces add <owner/space-name> --install-only

# list installed spaces
reachy-mini-conversation-app tool-spaces list

# remove an installed space
reachy-mini-conversation-app tool-spaces remove owner/space-name
```

The app validates the public Space slug through the Hugging Face Hub, probes the standard public MCP endpoint, discovers tools, enables them in the active profile's `tools.txt`, and writes the installed Space to:

- `installed_tool_spaces.json` in the managed app instance directory
- `external_content/installed_tool_spaces.json` in terminal mode

Recommended tags for discoverability on Hugging Face:

- `reachy-mini-tool`
- `mcp`

These tags are advisory only. Installation still relies on successful MCP validation, not on tag presence.

</details>

<details>
<summary><b>Multiple robots on the same subnet</b></summary>

If you run multiple Reachy Mini daemons on the same network, use:

```bash
reachy-mini-conversation-app --robot-name <name>
```

`<name>` must match the daemon's `--robot-name` value so the app connects to the correct robot.

</details>

## Contributing

We welcome bug fixes, features, profiles, and documentation improvements. Please review our
[contribution guide](CONTRIBUTING.md) for branch conventions, quality checks, and PR workflow.

Quick start:
- Fork and clone the repo
- Follow the [installation steps](#installation) (include the `dev` dependency group)
- Run contributor checks listed in [CONTRIBUTING.md](CONTRIBUTING.md)

## License

Apache 2.0

### Local-agent runtime requirements

This branch requires **Python 3.12** and **reachy-mini >=1.10,<1.11**. Install with
`uv sync --extra voice` (add `--extra gemini` for the Gemini backend). The voice runtime is
locked to a public Git commit so fresh checkouts and CI do not need a sibling repository.
For local runtime development, run `uv pip install -e ../reachy-hermes-agent` after syncing,
then launch with `uv run --no-sync`; normal syncing restores the pinned runtime. Do not add
its `[robot]` extra: this app owns the Reachy SDK requirement.

The quiet settings in `.env.example` are commented suggestions, not active defaults.
`AGENT_QUICKTAKE_ENABLED`, `AGENT_TOOL_STATUS_ENABLED`, and `AGENT_LEAD_IN_ENABLED` default
to `1`; `AGENT_GAP_FILL_MAX` defaults to `2`. Set all four to `0` for answer-only speech.
The motion switches listed above also default to `1`.

Deploy this client together with the hermes-reachy authentication adapter update. Configure
`AGENT_PLATFORM_API_KEY_FILE` with an absolute path; its stripped contents must equal the
secret in the gateway's `REACHY_WS_API_KEY_FILE`. After a policy close (1008), fix the
credentials and restart the app to reconnect.
