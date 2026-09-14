import unicodedata

from typing import Optional

from pydantic import (
    BaseModel,
    Field,
    field_validator,
)

from app.config import (
    MAX_QUERY_LENGTH,
)


class QueryRequest(BaseModel):
    """
    Request body for the RAG query endpoint.
    """

    question: str = Field(
        ...,
        min_length=1,
        max_length=MAX_QUERY_LENGTH,
        description=(
            "Question to answer using the indexed "
            "document knowledge base."
        ),
    )

    @field_validator(
        "question",
        mode="before",
    )
    @classmethod
    def normalize_question(
        cls,
        value,
    ):
        """
        Normalise user input and reject unsupported
        control characters.
        """

        if not isinstance(
            value,
            str,
        ):
            return value

        value = unicodedata.normalize(
            "NFC",
            value,
        ).strip()

        for character in value:

            category = (
                unicodedata.category(
                    character
                )
            )

            if (
                category == "Cc"
                and character
                not in {
                    "\n",
                    "\r",
                    "\t",
                }
            ):
                raise ValueError(
                    "Question contains unsupported "
                    "control characters."
                )

        return value


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