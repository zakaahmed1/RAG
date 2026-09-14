import logging

from contextlib import asynccontextmanager
from time import perf_counter
from uuid import uuid4

from fastapi import (
    FastAPI,
    HTTPException,
    Request,
)

from app.api.schemas import (
    HealthResponse,
    QueryRequest,
    QueryResponse,
    StatusResponse,
)

from app.api.service import (
    RAGService,
)

from app.observability.logging import (
    configure_logging,
    log_event,
    reset_request_id,
    set_request_id,
)


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

configure_logging()

logger = logging.getLogger(
    "rag.api"
)


# ---------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------

@asynccontextmanager
async def lifespan(
    app: FastAPI,
):
    """
    Initialise heavyweight RAG resources once when
    the API starts.
    """

    startup_start = perf_counter()

    log_event(
        logger,
        "application_starting",
    )

    rag_service = RAGService()

    rag_service.start()

    app.state.rag_service = (
        rag_service
    )

    startup_ms = (
        perf_counter()
        - startup_start
    ) * 1000

    log_event(
        logger,
        "application_ready",
        startup_ms=round(
            startup_ms,
            2,
        ),
    )

    yield

    log_event(
        logger,
        "application_stopping",
    )


# ---------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------

app = FastAPI(
    title="RAG Knowledge Assistant API",
    description=(
        "API for querying a grounded "
        "Retrieval-Augmented Generation system."
    ),
    version="0.5.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------
# Request observability middleware
# ---------------------------------------------------------

@app.middleware(
    "http"
)
async def request_observability(
    request: Request,
    call_next,
):
    """
    Attach a correlation ID and record HTTP latency.
    """

    request_id = str(
        uuid4()
    )

    request_token = (
        set_request_id(
            request_id
        )
    )

    request_start = (
        perf_counter()
    )

    try:

        response = await call_next(
            request
        )

        duration_ms = (
            perf_counter()
            - request_start
        ) * 1000

        response.headers[
            "X-Request-ID"
        ] = request_id

        log_event(
            logger,
            "http_request_completed",
            method=request.method,
            path=request.url.path,
            status_code=(
                response.status_code
            ),
            duration_ms=round(
                duration_ms,
                2,
            ),
        )

        return response

    except Exception:

        duration_ms = (
            perf_counter()
            - request_start
        ) * 1000

        log_event(
            logger,
            "http_request_failed",
            level=logging.ERROR,
            exc_info=True,
            method=request.method,
            path=request.url.path,
            duration_ms=round(
                duration_ms,
                2,
            ),
        )

        raise

    finally:

        reset_request_id(
            request_token
        )


# ---------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
)
def health():
    """
    Basic API liveness check.
    """

    return {
        "status": "ok"
    }


# ---------------------------------------------------------
# Status endpoint
# ---------------------------------------------------------

@app.get(
    "/status",
    response_model=StatusResponse,
)
def status(
    request: Request,
):
    """
    Return service readiness and active RAG settings.
    """

    rag_service = (
        request.app.state.rag_service
    )

    return rag_service.status()


# ---------------------------------------------------------
# Query endpoint
# ---------------------------------------------------------

@app.post(
    "/query",
    response_model=QueryResponse,
)
def query(
    payload: QueryRequest,
    request: Request,
):
    """
    Answer a question using the indexed knowledge base.
    """

    rag_service = (
        request.app.state.rag_service
    )

    question = (
        payload.question.strip()
    )

    if not question:

        raise HTTPException(
            status_code=422,
            detail=(
                "Question must contain "
                "non-whitespace characters."
            ),
        )

    try:

        result = rag_service.query(
            question
        )

        return result

    except ValueError as exc:

        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except RuntimeError as exc:

        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Query processing failed."
            ),
        ) from exc