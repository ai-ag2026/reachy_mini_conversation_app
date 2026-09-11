"""Pytest configuration for path setup."""

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[1].resolve()
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


# Make tests reproducible by ignoring machine-specific profile/tool env config.
# Without this, importing config during test collection can pick up a developer's
# local .env and fail before tests run.
os.environ["REACHY_MINI_SKIP_DOTENV"] = "1"
os.environ.pop("REACHY_MINI_CUSTOM_PROFILE", None)
os.environ.pop("REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY", None)
os.environ.pop("REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY", None)
# Handlers would otherwise start the loopback pipeline-monitor HTTP server on port 8766 as a side
# effect; tests that cover the monitor construct PipelineMonitor directly on an ephemeral port.
os.environ.setdefault("AGENT_PIPELINE_MONITOR", "0")
