import json
import logging

from app.observability.logging import (
    JSONFormatter,
    reset_request_id,
    set_request_id,
)


def test_json_logging_contains_request_context():

    token = set_request_id(
        "test-request-id"
    )

    try:

        record = logging.LogRecord(
            name="rag.test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="test_event",
            args=(),
            exc_info=None,
        )

        record.event = (
            "test_event"
        )

        record.structured_fields = {
            "duration_ms": 12.5,
            "retrieved_count": 4,
        }

        formatter = (
            JSONFormatter()
        )

        output = formatter.format(
            record
        )

        payload = json.loads(
            output
        )

        assert (
            payload["event"]
            == "test_event"
        )

        assert (
            payload["request_id"]
            == "test-request-id"
        )

        assert (
            payload["duration_ms"]
            == 12.5
        )

        assert (
            payload["retrieved_count"]
            == 4
        )

        assert (
            payload["level"]
            == "INFO"
        )

    finally:

        reset_request_id(
            token
        )