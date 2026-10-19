import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_FILE = Path(os.environ.get("DRH_LOG_FILE", "logs/deep_research_hub.log"))
_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
# Cloud Run sets K_SERVICE; it only collects stdout/stderr, not log files.
_ON_CLOUD_RUN = bool(os.environ.get("K_SERVICE"))


def setup_logging() -> None:
    """Send app and ADK logs to stdout on Cloud Run, else a rotating file.

    Safe to call more than once.
    """
    root = logging.getLogger()
    if _ON_CLOUD_RUN:
        if not any(getattr(h, "_drh_stdout", False) for h in root.handlers):
            handler = logging.StreamHandler(sys.stdout)
            handler._drh_stdout = True  # type: ignore[attr-defined]
            handler.setFormatter(logging.Formatter(_FORMAT))
            root.addHandler(handler)
    else:
        log_path = LOG_FILE.resolve()
        for h in root.handlers:
            if getattr(h, "baseFilename", None) == str(log_path):
                return
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            log_path, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
        )
        handler.setFormatter(logging.Formatter(_FORMAT))
        root.addHandler(handler)
    if root.level == logging.NOTSET or root.level > logging.INFO:
        root.setLevel(logging.INFO)
