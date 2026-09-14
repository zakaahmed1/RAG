import json
import logging
import os
import sys

from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any


# ---------------------------------------------------------
# Request context
# ---------------------------------------------------------

_request_id: ContextVar[str] = ContextVar(
    "request_id",
    default="-",
)


def set_request_id(
    request_id: str,
):
    """
    Associate a request ID with the current execution context.
    """

    return _request_id.set(
        request_id
    )


def reset_request_id(
    token,
):
    """
    Restore the previous request ID context.
    """

    _request_id.reset(
        token
    )


def get_request_id() -> str:
    """
    Return the request ID associated with the current context.
    """

    return _request_id.get()


# ---------------------------------------------------------
# JSON formatter
# ---------------------------------------------------------

class JSONFormatter(
    logging.Formatter
):
    """
    Convert application log records into structured JSON.
    """

    def format(
        self,
        record: logging.LogRecord,
    ) -> str:

        payload: dict[str, Any] = {
            "timestamp": (
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
            "level": record.levelname,
            "logger": record.name,
            "event": getattr(
                record,
                "event",
                record.getMessage(),
            ),
            "request_id": (
                get_request_id()
            ),
        }

        structured_fields = getattr(
            record,
            "structured_fields",
            {},
        )

        payload.update(
            structured_fields
        )

        if record.exc_info:

            payload["exception"] = (
                self.formatException(
                    record.exc_info
                )
            )

        return json.dumps(
            payload,
            default=str,
        )


# ---------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------

def configure_logging():
    """
    Configure structured logging for the RAG application.
    """

    log_level_name = os.getenv(
        "LOG_LEVEL",
        "INFO",
    ).upper()

    log_level = getattr(
        logging,
        log_level_name,
        logging.INFO,
    )

    rag_logger = logging.getLogger(
        "rag"
    )

    rag_logger.handlers.clear()

    handler = logging.StreamHandler(
        sys.stdout
    )

    handler.setFormatter(
        JSONFormatter()
    )

    rag_logger.addHandler(
        handler
    )

    rag_logger.setLevel(
        log_level
    )

    # Prevent logs from also propagating to the root
    # logger and being emitted twice.
    rag_logger.propagate = False


def log_event(
    logger: logging.Logger,
    event: str,
    *,
    level: int = logging.INFO,
    exc_info: bool = False,
    **fields,
):
    """
    Emit one structured application event.
    """

    logger.log(
        level,
        event,
        extra={
            "event": event,
            "structured_fields": fields,
        },
        exc_info=exc_info,
    )