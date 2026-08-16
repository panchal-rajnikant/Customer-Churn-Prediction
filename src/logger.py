import json
import logging

logger = logging.getLogger(__name__)

def log_event(event, **kwargs):
    logger.info(
        json.dumps({
            "event": event,
            **kwargs
        })
    )