import os
import secrets

from typing import Annotated

from fastapi import (
    Header,
    HTTPException,
)


def configured_api_key() -> str:
    """
    Return the currently configured API key.

    Read dynamically so secrets are not copied into
    application state unnecessarily.
    """

    return os.getenv(
        "RAG_API_KEY",
        "",
    ).strip()


def require_api_key(
    x_api_key: Annotated[
        str | None,
        Header(
            alias="X-API-Key"
        ),
    ] = None,
):
    """
    Require X-API-Key when an API key is configured.

    Development remains convenient when no key is set.
    """

    expected_key = (
        configured_api_key()
    )

    if not expected_key:
        return

    if (
        x_api_key is None
        or not secrets.compare_digest(
            x_api_key,
            expected_key,
        )
    ):
        raise HTTPException(
            status_code=401,
            detail=(
                "Invalid or missing API key."
            ),
        )


def validate_security_configuration():
    """
    Fail fast when production security configuration
    is incomplete.
    """

    app_env = os.getenv(
        "APP_ENV",
        "development",
    ).lower()

    if (
        app_env == "production"
        and not configured_api_key()
    ):
        raise RuntimeError(
            "RAG_API_KEY must be configured "
            "when APP_ENV=production."
        )