import json
import logging
from pathlib import Path

# configure a module-level logger that writes JSON lines to a file and to console
logs_dir = Path(__file__).resolve().parents[1] / "logs"
logs_dir.mkdir(parents=True, exist_ok=True)
log_file = logs_dir / "events.jsonl"

logger = logging.getLogger("customer_churn")
logger.setLevel(logging.INFO)

if not logger.handlers:
    # File handler (writes each log message as a JSON line)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)

    # Console handler for convenience
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)


def log_event(event, *args, **kwargs):
    # build JSON object and log the string; the FileHandler stores it in events.jsonl
    payload = {"event": event}
    if args:
        # attach positional data under `data` (single value or list)
        payload["data"] = args[0] if len(args) == 1 else list(args)
    if kwargs:
        payload.update(kwargs)

    # use default=str to safely convert non-serializable objects (e.g., pandas Series)
    logger.info(json.dumps(payload, default=str))