"""Entrypoint for the Reachy Mini conversation app."""

import os
import sys
import time
import signal
import asyncio
import argparse
import threading
from typing import Any, Dict, List, Optional
from pathlib import Path

import httpx
import gradio as gr
from fastapi import FastAPI
from fastrtc import Stream
from gradio.utils import get_space

from reachy_mini import ReachyMini, ReachyMiniApp
from reachy_mini.io.protocol import SetVolumeCmd
from reachy_mini_conversation_app.utils import (
    CameraVisionInitializationError,
    parse_args,
    setup_logger,
    initialize_camera_and_vision,
    log_connection_troubleshooting,
)


def update_chatbot(chatbot: List[Dict[str, Any]], response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Update the chatbot with AdditionalOutputs."""
    chatbot.append(response)
    return chatbot


def main() -> None:
    """Entrypoint for the Reachy Mini conversation app."""
    args, _ = parse_args()
    if args.command == "tool-spaces":
        from reachy_mini_conversation_app.tool_spaces import handle_tool_spaces_command

        logger = setup_logger(args.debug)
        try:
            raise SystemExit(handle_tool_spaces_command(args))
        except Exception as exc:
            logger.error("tool-spaces command failed: %s", exc)
            raise SystemExit(1) from exc
    run(args)


def run(
    args: argparse.Namespace,
    robot: ReachyMini = None,
    app_stop_event: Optional[threading.Event] = None,
    settings_app: Optional[FastAPI] = None,
    instance_path: Optional[str] = None,
) -> None:
    """Run the Reachy Mini conversation app."""
    # Putting these dependencies here makes the dashboard faster to load when the conversation app is installed
    from reachy_mini_conversation_app.moves import MovementManager
    from reachy_mini_conversation_app.config import (
        HF_BACKEND,
        LOCAL_BACKEND,
        GEMINI_BACKEND,
        OPENAI_BACKEND,
        HF_LOCAL_CONNECTION_MODE,
        config,
        is_gemini_model,
        get_backend_label,
        get_hf_connection_selection,
        refresh_runtime_config_from_env,
    )
    from reachy_mini_conversation_app.startup_settings import (
        StartupSettings,
        load_startup_settings_into_runtime,
    )

    logger = setup_logger(args.debug)
    logger.info("Starting Reachy Mini Conversation App")
    if app_stop_event is None:
        app_stop_event = threading.Event()
    previous_signal_handlers = {}
    managed_robot = None
    managed_movement_manager = None
    saved_speaker_volume: int | None = None

    def _cleanup_runtime_after_signal() -> None:
        if managed_movement_manager is not None:
            try:
                managed_movement_manager.stop(skip_neutral=True)  # goto_sleep follows — no neutral detour
            except Exception as exc:
                logger.debug("Error stopping movement manager after signal: %s", exc)
        if managed_robot is not None:
            # Sleep pose BEFORE disable_motors: without it torque cuts out in the upright
            # NEUTRAL pose and the head drops mechanically. The goto_sleep in run()'s finally
            # can't help — by then the client is already disconnected (audit 2026-07-02).
            try:
                managed_robot.goto_sleep()
            except Exception as exc:
                logger.debug("Error going to sleep pose after signal: %s", exc)
            if saved_speaker_volume is not None:
                try:
                    managed_robot.client.send_command(SetVolumeCmd(volume=saved_speaker_volume))
                    logger.info("Restored robot speaker volume to %s after signal", saved_speaker_volume)
                except Exception as exc:
                    logger.debug("Error restoring speaker volume after signal: %s", exc)
            try:
                managed_robot.disable_motors()
            except Exception as exc:
                logger.debug("Error disabling motors after signal: %s", exc)
            try:
                managed_robot.disable_wobbling()
            except Exception as exc:
                logger.debug("Error disabling wobbling after signal: %s", exc)
            try:
                managed_robot.media.close()
            except Exception as exc:
                logger.debug("Error closing media after signal: %s", exc)
            try:
                managed_robot.client.disconnect()
            except Exception as exc:
                logger.debug("Error disconnecting client after signal: %s", exc)

    _shutdown_signal_seen = threading.Event()

    def _handle_shutdown_signal(signum: int, _frame: object) -> None:
        # Reentrancy guard: the cleanup blocks for seconds (movement stop + sleep pose); a second
        # SIGTERM re-entering mid-cleanup would abort the first pass half-way (audit 2026-07-02).
        if _shutdown_signal_seen.is_set():
            logger.info("Received signal %s during shutdown — already cleaning up", signum)
            return
        _shutdown_signal_seen.set()
        logger.info("Received signal %s, shutting down gracefully", signum)
        app_stop_event.set()
        _cleanup_runtime_after_signal()
        raise SystemExit(128 + signum)

    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous_signal_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, _handle_shutdown_signal)

    startup_settings = StartupSettings()

    if instance_path is not None:
        try:
            from dotenv import load_dotenv

            env_path = Path(instance_path) / ".env"
            if env_path.exists():
                load_dotenv(dotenv_path=str(env_path), override=True)
                refresh_runtime_config_from_env()
                logger.info("Loaded instance configuration from %s", env_path)
        except Exception as e:
            logger.warning("Failed to load instance configuration: %s", e)

        try:
            startup_settings = load_startup_settings_into_runtime(instance_path)
        except Exception as e:
            logger.warning("Failed to load startup settings: %s", e)

    if config.BACKEND_PROVIDER == HF_BACKEND:
        logger.info(
            "Configured backend provider: %s (%s), connection mode: %s",
            config.BACKEND_PROVIDER,
            get_backend_label(config.BACKEND_PROVIDER),
            get_hf_connection_selection().mode,
        )
    else:
        logger.info(
            "Configured backend provider: %s (%s), model: %s",
            config.BACKEND_PROVIDER,
            get_backend_label(config.BACKEND_PROVIDER),
            config.MODEL_NAME,
        )

    from reachy_mini_conversation_app.console import LocalStream
    from reachy_mini_conversation_app.tools.core_tools import ToolDependencies, initialize_tools
    from reachy_mini_conversation_app.conversation_handler import ConversationHandler

    try:
        initialize_tools(instance_path=instance_path)
    except Exception as e:
        logger.error("Failed to initialize tools: %s", e)
        sys.exit(1)

    if args.no_camera and args.head_tracker is not None:
        logger.warning("Head tracking disabled: --no-camera flag is set. Remove --no-camera to enable head tracking.")

    if robot is None:
        try:
            robot_kwargs = {}
            if args.robot_name is not None:
                robot_kwargs["robot_name"] = args.robot_name

            logger.info("Initializing ReachyMini (SDK will auto-detect appropriate backend)")
            robot = ReachyMini(**robot_kwargs)

        except TimeoutError as e:
            logger.error(f"Connection timeout: Failed to connect to Reachy Mini daemon. Details: {e}")
            log_connection_troubleshooting(logger, args.robot_name)
            sys.exit(1)

        except ConnectionError as e:
            logger.error(f"Connection failed: Unable to establish connection to Reachy Mini. Details: {e}")
            log_connection_troubleshooting(logger, args.robot_name)
            sys.exit(1)

        except Exception as e:
            logger.error(f"Unexpected error during robot initialization: {type(e).__name__}: {e}")
            logger.error("Please check your configuration and try again.")
            sys.exit(1)

    managed_robot = robot

    try:
        camera_worker, vision_processor = initialize_camera_and_vision(args, robot)
    except CameraVisionInitializationError as e:
        logger.error("Failed to initialize camera/vision: %s", e)
        sys.exit(1)

    movement_manager = MovementManager(
        current_robot=robot,
        camera_worker=camera_worker,
    )
    managed_movement_manager = movement_manager

    deps = ToolDependencies(
        reachy_mini=robot,
        movement_manager=movement_manager,
        instance_path=instance_path,
        camera_worker=camera_worker,
        vision_processor=vision_processor,
    )
    current_file_path = os.path.dirname(os.path.abspath(__file__))
    logger.debug(f"Current file absolute path: {current_file_path}")
    chatbot = gr.Chatbot(
        type="messages",
        resizable=True,
        avatar_images=(
            os.path.join(current_file_path, "images", "user_avatar.png"),
            os.path.join(current_file_path, "images", "reachymini_avatar.png"),
        ),
    )
    logger.debug(f"Chatbot avatar images: {chatbot.avatar_images}")

    def build_handler(startup_voice: Optional[str] = None) -> ConversationHandler:
        """Build a realtime handler for the current runtime backend config."""
        if config.BACKEND_PROVIDER == LOCAL_BACKEND:
            logger.info(
                "Using %s via AgentVoiceHandler",
                get_backend_label(config.BACKEND_PROVIDER),
            )
            from reachy_mini_conversation_app.handler_factory import build_conversation_handler

            return build_conversation_handler(
                deps,
                gradio_mode=args.gradio,
                instance_path=instance_path,
                startup_voice=startup_voice,
                allow_live_agent_clients=True,
            )
        if is_gemini_model():
            from reachy_mini_conversation_app.gemini_live import GeminiLiveHandler

            logger.info(
                "Using %s via GeminiLiveHandler",
                get_backend_label(config.BACKEND_PROVIDER),
            )
            return GeminiLiveHandler(
                deps,
                gradio_mode=args.gradio,
                instance_path=instance_path,
                startup_voice=startup_voice,
            )
        if config.BACKEND_PROVIDER == HF_BACKEND:
            from reachy_mini_conversation_app.huggingface_realtime import HuggingFaceRealtimeHandler

            hf_connection_selection = get_hf_connection_selection()
            transport_label = (
                "Hugging Face direct websocket"
                if hf_connection_selection.mode == HF_LOCAL_CONNECTION_MODE and hf_connection_selection.has_target
                else "Hugging Face session proxy"
            )
            logger.info(
                "Using %s via Hugging Face realtime handler (%s)",
                get_backend_label(config.BACKEND_PROVIDER),
                transport_label,
            )
            return HuggingFaceRealtimeHandler(
                deps,
                gradio_mode=args.gradio,
                instance_path=instance_path,
                startup_voice=startup_voice,
            )

        from reachy_mini_conversation_app.openai_realtime import OpenaiRealtimeHandler

        logger.info(
            "Using %s via OpenAI realtime handler (OpenAI Realtime API)",
            get_backend_label(config.BACKEND_PROVIDER),
        )
        return OpenaiRealtimeHandler(
            deps,
            gradio_mode=args.gradio,
            instance_path=instance_path,
            startup_voice=startup_voice,
        )

    handler = build_handler(startup_settings.voice)

    stream_manager: gr.Blocks | LocalStream | None = None

    if args.gradio:
        from reachy_mini_conversation_app.gradio_personality import PersonalityUI

        personality_ui = PersonalityUI()
        personality_ui.create_components()
        additional_inputs: list[Any] = [chatbot, *personality_ui.additional_inputs_ordered()]

        if config.BACKEND_PROVIDER in {OPENAI_BACKEND, GEMINI_BACKEND}:
            uses_gemini_backend = is_gemini_model()
            api_key_textbox = gr.Textbox(
                label="GEMINI_API_KEY" if uses_gemini_backend else "OPENAI API Key",
                type="password",
                value=(os.getenv("GEMINI_API_KEY") if uses_gemini_backend else os.getenv("OPENAI_API_KEY"))
                if not get_space()
                else "",
            )
            additional_inputs.insert(1, api_key_textbox)

        stream = Stream(
            handler=handler,
            mode="send-receive",
            modality="audio",
            additional_inputs=additional_inputs,
            additional_outputs=[chatbot],
            additional_outputs_handler=update_chatbot,
            ui_args={"title": "Talk with Reachy Mini"},
        )
        stream_manager = stream.ui
        if not settings_app:
            app = FastAPI()
        else:
            app = settings_app

        personality_ui.wire_events(handler, stream_manager)

        app = gr.mount_gradio_app(app, stream.ui, path="/")
    else:
        # In headless mode, wire settings_app + instance_path to console LocalStream
        stream_manager = LocalStream(
            handler,
            robot,
            settings_app=settings_app,
            instance_path=instance_path,
            handler_factory=build_handler,
            startup_voice=startup_settings.voice,
        )

    # Reachy's daemon runs with --no-wake-up-on-start (decision 2026-06-24: Reachy should be able to
    # sleep by default), so THIS app owns readiness. Enable motors BEFORE the MovementManager loop
    # issues any set_target — the SDK pins all targets to the present pose on enable, so set_target/
    # goto after enable is required to actually move (AGENT P0.2). Then play the standard wake-up
    # gesture while no control loop is running yet. Safe/idempotent if the daemon already woke it.
    # If this fails, DON'T start the control loop blind: the first tick would set_target(NEUTRAL)
    # from the sleep pose — an unbraked head snap on torqueless/half-woken motors (review
    # 2026-07-02 round 2, P2). Retry once (transient heartbeat misses), then abort cleanly.
    try:
        robot.enable_motors()
        robot.wake_up()
    except Exception as exc:
        logger.warning(f"Robot wake/enable on startup failed: {exc} — retrying once")
        time.sleep(2.0)
        try:
            robot.enable_motors()
            robot.wake_up()
        except Exception:
            logger.error("Robot wake/enable failed twice — aborting startup (no blind control loop)")
            try:
                robot.disable_motors()
            except Exception:
                pass
            raise

    # Each async service → its own thread/loop
    movement_manager.start()
    # Audio-reactive head motion is driven by the daemon's wobbler, which
    # taps the media pipeline at push_audio_sample. In headless mode the
    # console stream pushes assistant audio through that pipeline directly.
    # In Gradio mode audio plays in the browser; the handler additionally
    # taps the same call to keep the wobbler fed (see
    # BaseRealtimeHandler._tap_audio_for_daemon_wobbler) — mute the robot
    # speaker to avoid double playback.
    # Non-fatal enhancement: a transient daemon heartbeat miss here used to crash the whole app
    # AFTER the manager started — motors enabled, no goto_sleep (review 2026-07-02 round 2, P2).
    try:
        robot.enable_wobbling()
    except Exception as exc:
        logger.warning(f"enable_wobbling failed (continuing without speech wobble): {exc}")
    if args.gradio:
        # LocalStream.launch() starts the playback pipeline in headless mode.
        # In Gradio mode nothing else does, and push_audio_sample is a no-op
        # until the pipeline is in PLAYING state — so the wobbler stays idle.
        try:
            robot.media.start_playing()
        except Exception as exc:
            logger.warning(f"Failed to start media playback for Gradio mode: {exc}")
        # Audio plays in the browser; silence the robot speaker so we don't
        # hear it twice. GET via REST to read the current level (no side
        # effects), SET via the WS protocol so the daemon does not fire its
        # "test sound" played by POST /api/volume/set on each call.
        try:
            volume_url = f"http://{robot.client.host}:{robot.client.port}/api/volume/current"
            saved_speaker_volume = int(httpx.get(volume_url, timeout=2).json()["volume"])
            robot.client.send_command(SetVolumeCmd(volume=0))
            logger.info(f"Muted robot speaker for Gradio mode (saved volume: {saved_speaker_volume})")
        except Exception as exc:
            logger.warning(f"Could not mute robot speaker: {exc}")
            saved_speaker_volume = None
    if camera_worker:
        try:
            camera_worker.start()
        except Exception as exc:
            logger.warning(f"camera worker start failed (continuing without vision/tracking): {exc}")

    def poll_stop_event() -> None:
        """Poll the stop event to allow graceful shutdown."""
        if app_stop_event is not None:
            app_stop_event.wait()

        logger.info("App stop event detected, shutting down...")
        try:
            stream_manager.close()
        except Exception as e:
            logger.error(f"Error while closing stream manager: {e}")

    if app_stop_event:
        threading.Thread(target=poll_stop_event, daemon=True).start()

    try:
        stream_manager.launch()
    except KeyboardInterrupt:
        logger.info("Keyboard interruption in main thread... closing server.")
    finally:
        movement_manager.stop(skip_neutral=True)  # goto_sleep follows — no neutral detour
        try:
            robot.goto_sleep()  # leave Reachy asleep on exit (daemon keeps --no-wake-up-on-start)
        except Exception as e:
            logger.debug(f"Error putting robot to sleep during shutdown: {e}")
        try:
            robot.disable_wobbling()
        except Exception as e:
            logger.debug(f"Error disabling wobbling during shutdown: {e}")
        if saved_speaker_volume is not None:
            try:
                robot.client.send_command(SetVolumeCmd(volume=saved_speaker_volume))
                logger.info(f"Restored robot speaker volume to {saved_speaker_volume}")
            except Exception as e:
                logger.debug(f"Error restoring speaker volume during shutdown: {e}")
        if camera_worker:
            camera_worker.stop()

        # Ensure media is explicitly closed before disconnecting
        try:
            robot.media.close()
        except Exception as e:
            logger.debug(f"Error closing media during shutdown: {e}")

        # prevent connection to keep alive some threads
        robot.client.disconnect()
        time.sleep(1)
        for sig, previous_handler in previous_signal_handlers.items():
            signal.signal(sig, previous_handler)
        logger.info("Shutdown complete.")


class ReachyMiniConversationApp(ReachyMiniApp):  # type: ignore[misc]
    """Reachy Mini Apps entry point for the conversation app."""

    custom_app_url = "http://0.0.0.0:7860/"
    dont_start_webserver = False

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event) -> None:
        """Run the Reachy Mini conversation app."""
        asyncio.set_event_loop(asyncio.new_event_loop())

        args, _ = parse_args()

        instance_path = self._get_instance_path().parent
        run(
            args,
            robot=reachy_mini,
            app_stop_event=stop_event,
            settings_app=self.settings_app,
            instance_path=instance_path,
        )


if __name__ == "__main__":
    app = ReachyMiniConversationApp()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()
