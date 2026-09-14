from fastapi.testclient import TestClient

from uuid import UUID

import app.api.app as api_module


class FakeRAGService:

    def __init__(self):
        self.ready = False

    def start(self):
        self.ready = True

    def status(self):
        return {
            "ready": True,
            "retrieval_mode": "similarity",
            "top_k": 4,
            "fetch_k": 12,
            "min_similarity": 0.425,
            "embedding_model": (
                "sentence-transformers/"
                "all-MiniLM-L6-v2"
            ),
            "generator_model": (
                "google/flan-t5-base"
            ),
        }

    def query(self, question):
        return {
            "question": question,
            "answer": "25 days.",
            "retrieval_rejected": False,
            "retrieved_count": 1,
            "sources": [
                {
                    "file":
                        "EmployeeHandbook.pdf",
                    "page": 4,
                    "chunk_id": 8,
                    "similarity": 0.72,
                }
            ],
        }


def create_client(monkeypatch):

    monkeypatch.setattr(
        api_module,
        "RAGService",
        FakeRAGService,
    )

    return TestClient(
        api_module.app
    )


def test_health_endpoint(
    monkeypatch,
):

    with create_client(
        monkeypatch
    ) as client:

        response = client.get(
            "/health"
        )

        assert response.status_code == 200

        assert response.json() == {
            "status": "ok"
        }


def test_status_endpoint(
    monkeypatch,
):

    with create_client(
        monkeypatch
    ) as client:

        response = client.get(
            "/status"
        )

        assert response.status_code == 200

        data = response.json()

        assert data["ready"] is True
        assert (
            data["retrieval_mode"]
            == "similarity"
        )
        assert data["top_k"] == 4
        assert data["fetch_k"] == 12
        assert (
            data["min_similarity"]
            == 0.425
        )


def test_query_endpoint(
    monkeypatch,
):

    with create_client(
        monkeypatch
    ) as client:

        response = client.post(
            "/query",
            json={
                "question":
                    "How much annual leave?"
            },
        )

        assert response.status_code == 200

        data = response.json()

        assert data["answer"] == "25 days."

        assert (
            data["retrieval_rejected"]
            is False
        )

        assert (
            data["sources"][0]["page"]
            == 4
        )


def test_blank_question_returns_422(
    monkeypatch,
):

    with create_client(
        monkeypatch
    ) as client:

        response = client.post(
            "/query",
            json={
                "question": "   "
            },
        )

        assert response.status_code == 422


def test_missing_question_returns_422(
    monkeypatch,
):

    with create_client(
        monkeypatch
    ) as client:

        response = client.post(
            "/query",
            json={},
        )

        assert response.status_code == 422


def test_request_id_header(
    monkeypatch,
):

    with create_client(
        monkeypatch
    ) as client:

        response = client.get(
            "/health"
        )

        assert response.status_code == 200

        request_id = (
            response.headers[
                "X-Request-ID"
            ]
        )

        # Raises ValueError if it is not
        # a valid UUID.
        UUID(
            request_id
        )