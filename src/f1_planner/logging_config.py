"""Per-run structured logging with correlation IDs."""

import logging
import uuid
from pathlib import Path

_LOG_DIR = Path("logs")


def setup_logging(run_id: str | None = None) -> tuple[logging.Logger, str]:
    """Configure logging for one crew run.

    Returns the root logger for the package and the run ID (generated if not
    supplied).  Every log line is tagged with the run ID so that log files
    from different runs can be correlated after the fact.
    """
    run_id = run_id or uuid.uuid4().hex[:8]
    _LOG_DIR.mkdir(exist_ok=True)

    fmt = f"%(asctime)s [{run_id}] %(levelname)s %(name)s: %(message)s"
    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")

    console = logging.StreamHandler()
    console.setFormatter(formatter)

    logfile = logging.FileHandler(
        _LOG_DIR / f"run_{run_id}.log", encoding="utf-8"
    )
    logfile.setFormatter(formatter)

    root = logging.getLogger("f1_planner")
    root.setLevel(logging.INFO)
    root.addHandler(console)
    root.addHandler(logfile)

    return root, run_id
