from contextlib import asynccontextmanager

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


# ---------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialise heavyweight RAG resources once when
    the API starts.
    """

    rag_service = RAGService()

    rag_service.start()

    app.state.rag_service = (
        rag_service
    )

    yield


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

    question = payload.question.strip()

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