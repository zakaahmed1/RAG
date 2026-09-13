import pytest

from langchain_core.documents import Document

from app.retrieval.search import (
    distance_to_similarity,
    retrieve_documents,
)


class FakeVectorStore:

    def similarity_search_with_score(
        self,
        query,
        k,
    ):
        documents = [
            Document(
                page_content="Annual leave is 25 days.",
                metadata={
                    "file_name": "EmployeeHandbook.pdf",
                    "source_path": "EmployeeHandbook.pdf",
                    "page_number": 4,
                    "chunk_id": 1,
                },
            ),
            Document(
                page_content="Irrelevant content.",
                metadata={
                    "file_name": "Other.txt",
                    "source_path": "Other.txt",
                    "page_number": None,
                    "chunk_id": 2,
                },
            ),
        ]

        return [
            (documents[0], 0.4),
            (documents[1], 1.6),
        ]


def test_distance_to_similarity():

    similarity = distance_to_similarity(
        0.4
    )

    assert similarity == pytest.approx(
        0.8
    )


def test_similarity_threshold_filters_results():

    vector_store = FakeVectorStore()

    results = retrieve_documents(
        vector_store,
        "How much annual leave?",
        mode="similarity",
        top_k=4,
        fetch_k=12,
        min_similarity=0.425,
    )

    assert len(results) == 1

    assert (
        results[0]
        .document
        .metadata["file_name"]
        == "EmployeeHandbook.pdf"
    )


def test_empty_query_rejected():

    vector_store = FakeVectorStore()

    with pytest.raises(
        ValueError,
        match="Query cannot be empty",
    ):

        retrieve_documents(
            vector_store,
            "   ",
        )