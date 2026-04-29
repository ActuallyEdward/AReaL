from __future__ import annotations

import sys
from pathlib import Path


MINI_SWE_AGENT_SRC = (
    Path(__file__).resolve().parents[4] / "mini-swe-agent" / "src"
)


def ensure_mini_swe_agent_on_path() -> None:
    src = str(MINI_SWE_AGENT_SRC)
    if src not in sys.path:
        sys.path.insert(0, src)
