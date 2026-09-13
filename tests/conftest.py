from langchain_core.documents import Document
import pytest

from app.retrieval.search import RetrievedChunk


@pytest.fixture
def handbook_chunk():
    document = Document(
        page_content=(
            "Full-time employees receive "
            "25 days of paid annual leave."
        ),
        metadata={
            "file_name": "EmployeeHandbook.pdf",
            "source_path": "EmployeeHandbook.pdf",
            "page_number": 4,
            "chunk_id": 8,
        },
    )

    return RetrievedChunk(
        document=document,
        distance=0.4,
        similarity=0.8,
    )


@pytest.fixture
def security_chunk():
    document = Document(
        page_content=(
            "Lost or stolen company devices "
            "must be reported within 2 hours."
        ),
        metadata={
            "file_name": "ITSecurityPolicy.txt",
            "source_path": "ITSecurityPolicy.txt",
            "page_number": None,
            "chunk_id": 36,
        },
    )

    return RetrievedChunk(
        document=document,
        distance=0.6,
        similarity=0.7,
    )