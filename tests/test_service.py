from app.api.service import RAGService
from app.config import ABSTENTION_TEXT
import pytest


def test_service_rejects_when_no_chunks(
    monkeypatch,
):

    service = RAGService()

    service.vector_store = object()
    service.generator = object()
    service.ready = True

    monkeypatch.setattr(
        "app.api.service.retrieve_documents",
        lambda *args, **kwargs: [],
    )

    result = service.query(
        "What is the CEO's salary?"
    )

    assert result["retrieval_rejected"] is True
    assert result["retrieved_count"] == 0
    assert result["sources"] == []
    assert result["answer"] == ABSTENTION_TEXT


def test_service_generates_answer(
    monkeypatch,
    handbook_chunk,
):

    service = RAGService()

    service.vector_store = object()
    service.generator = object()
    service.ready = True

    monkeypatch.setattr(
        "app.api.service.retrieve_documents",
        lambda *args, **kwargs: [
            handbook_chunk
        ],
    )

    monkeypatch.setattr(
        "app.api.service.generate_answer",
        lambda **kwargs: "25 days.",
    )

    result = service.query(
        "How much annual leave do employees receive?"
    )

    assert result["retrieval_rejected"] is False
    assert result["answer"] == "25 days."
    assert result["retrieved_count"] == 1

    assert (
        result["sources"][0]["file"]
        == "EmployeeHandbook.pdf"
    )

    assert (
        result["sources"][0]["page"]
        == 4
    )


def test_service_requires_ready_state():

    service = RAGService()

    with pytest.raises(
        RuntimeError,
        match="not ready",
    ):
        service.query(
            "Test question"
        )