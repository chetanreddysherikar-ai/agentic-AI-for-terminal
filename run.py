"""Entry point: `python run.py` / `crs.cmd`."""
from __future__ import annotations

import warnings

# Some Gemini 3 models use fixed sampling defaults and warn that temperature
# is ignored. It is noise for the user, so drop it.
warnings.filterwarnings("ignore", message=r".*fixed sampling defaults.*")

from agent.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
