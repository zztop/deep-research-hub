import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_FILE = Path(os.environ.get("DRH_LOG_FILE", "logs/deep_research_hub.log"))
_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def setup_logging() -> None:
    """Send app and ADK logs to a rotating file. Safe to call more than once."""
    root = logging.getLogger()
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
