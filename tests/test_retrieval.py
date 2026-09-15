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


class FakeMMRVectorStore:

    def __init__(self):
        self.documents = [
            Document(
                page_content=f"Document {name}",
                metadata={
                    "file_name": f"{name}.txt",
                    "source_path": f"{name}.txt",
                    "page_number": None,
                    "chunk_id": index,
                },
            )
            for index, name in enumerate(
                ("a", "below-threshold", "c", "d"),
                start=1,
            )
        ]
        self.mmr_arguments = None

    def similarity_search_with_score(self, query, k):
        distances = (0.2, 1.4, 0.4, 0.6)
        return list(
            zip(
                self.documents[:k],
                distances[:k],
            )
        )

    def max_marginal_relevance_search(
        self,
        query,
        k,
        fetch_k,
        lambda_mult,
    ):
        self.mmr_arguments = {
            "k": k,
            "fetch_k": fetch_k,
            "lambda_mult": lambda_mult,
        }
        return self.documents[:k]


def test_mmr_backfills_after_threshold_filtering():
    vector_store = FakeMMRVectorStore()

    results = retrieve_documents(
        vector_store,
        "policy question",
        mode="mmr",
        top_k=3,
        fetch_k=4,
        min_similarity=0.425,
        mmr_lambda=0.7,
    )

    assert [
        result.document.metadata["file_name"]
        for result in results
    ] == ["a.txt", "c.txt", "d.txt"]
    assert vector_store.mmr_arguments == {
        "k": 4,
        "fetch_k": 4,
        "lambda_mult": 0.7,
    }
