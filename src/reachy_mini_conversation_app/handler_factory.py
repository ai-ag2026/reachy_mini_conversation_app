from __future__ import annotations
import os
from typing import cast

from reachy_mini_conversation_app import config as config_module
from reachy_mini_conversation_app.config import HF_BACKEND, LOCAL_BACKEND, GEMINI_BACKEND, is_gemini_model
from reachy_mini_conversation_app.agent_clients import FastLeadInClient, HermesVoiceClient, QwenVoiceTtsClient
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.agent_voice_handler import AudioTtsClient, TextAgentClient, AgentVoiceHandler
from reachy_mini_conversation_app.conversation_handler import ConversationHandler


def _build_agent_text_client() -> TextAgentClient:
    """Select the AGENT text transport. ``AGENT_TRANSPORT=platform`` routes through
    the gateway's reachy platform adapter over WebSocket (unlocks async/proactive
    delivery); the default ``http`` keeps the stateless /v1/chat/completions path.
    """
    transport = os.getenv("AGENT_TRANSPORT", "http").strip().lower()
    if transport == "platform":
        from reachy_mini_conversation_app.reachy_platform_client import ReachyPlatformClient

        return cast(TextAgentClient, ReachyPlatformClient())
    return cast(TextAgentClient, HermesVoiceClient())


def build_conversation_handler(
    deps: ToolDependencies,
    *,
    backend_provider: str | None = None,
    gradio_mode: bool = False,
    instance_path: str | None = None,
    startup_voice: str | None = None,
    agent_client: TextAgentClient | None = None,
    tts_client: AudioTtsClient | None = None,
    lead_in_client: object | None = None,
    allow_live_agent_clients: bool = False,
) -> ConversationHandler:
    """Build the app conversation handler for the selected backend."""
    provider = (backend_provider or config_module.config.BACKEND_PROVIDER).strip().lower()
    if provider == LOCAL_BACKEND:
        if agent_client is None or tts_client is None:
            if not allow_live_agent_clients:
                raise RuntimeError("AGENT backend requires injected AGENT and TTS clients in this adapter slice.")
            agent_client = _build_agent_text_client()
            tts_client = QwenVoiceTtsClient()
            if lead_in_client is None:
                lead_in_client = FastLeadInClient()  # adaptive fast 9B latency mask
        return AgentVoiceHandler(deps, agent_client=agent_client, tts_client=tts_client, lead_in_client=lead_in_client)
    if provider == GEMINI_BACKEND or (backend_provider is None and is_gemini_model()):
        from reachy_mini_conversation_app.gemini_live import GeminiLiveHandler

        return GeminiLiveHandler(
            deps,
            gradio_mode=gradio_mode,
            instance_path=instance_path,
            startup_voice=startup_voice,
        )
    if provider == HF_BACKEND:
        from reachy_mini_conversation_app.huggingface_realtime import HuggingFaceRealtimeHandler

        return HuggingFaceRealtimeHandler(
            deps,
            gradio_mode=gradio_mode,
            instance_path=instance_path,
            startup_voice=startup_voice,
        )

    from reachy_mini_conversation_app.openai_realtime import OpenaiRealtimeHandler

    return cast(
        ConversationHandler,
        OpenaiRealtimeHandler(
            deps,
            gradio_mode=gradio_mode,
            instance_path=instance_path,
            startup_voice=startup_voice,
        ),
    )
