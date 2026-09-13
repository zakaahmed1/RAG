from typing import Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """
    Request body for the RAG query endpoint.
    """

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description=(
            "Question to answer using the indexed "
            "document knowledge base."
        ),
    )


class SourceResponse(BaseModel):
    """
    One retrieved source chunk.
    """

    file: str
    page: Optional[int] = None
    chunk_id: Optional[int] = None
    similarity: float


class QueryResponse(BaseModel):
    """
    Structured RAG response.
    """

    question: str
    answer: str
    retrieval_rejected: bool
    retrieved_count: int
    sources: list[SourceResponse]


class HealthResponse(BaseModel):
    """
    Basic API health response.
    """

    status: str


class StatusResponse(BaseModel):
    """
    Runtime RAG configuration and readiness information.
    """

    ready: bool

    retrieval_mode: str
    top_k: int
    fetch_k: int
    min_similarity: float

    embedding_model: str
    generator_model: str