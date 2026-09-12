from dataclasses import dataclass

from langchain_core.documents import Document

from app.config import (
    FETCH_K,
    MIN_SIMILARITY,
    MMR_LAMBDA_MULT,
    RETRIEVAL_MODE,
    TOP_K,
)


@dataclass(frozen=True)
class RetrievedChunk:
    """
    Represents a retrieved document chunk together with
    its vector-search scoring information.
    """

    document: Document
    distance: float
    similarity: float


def distance_to_similarity(distance: float) -> float:
    """
    Convert squared L2 distance between normalised vectors
    into cosine similarity.

    For unit-normalised vectors:

        squared_l2 = 2 - 2 * cosine_similarity

    therefore:

        cosine_similarity = 1 - squared_l2 / 2
    """

    similarity = 1.0 - (float(distance) / 2.0)

    return max(
        -1.0,
        min(1.0, similarity),
    )


def document_key(document: Document):
    """
    Create a stable key for identifying duplicate chunks.
    """

    metadata = document.metadata

    return (
        metadata.get("source_path"),
        metadata.get("page_number"),
        metadata.get("chunk_id"),
        document.page_content.strip(),
    )


def get_similarity_candidates(
    vector_store,
    query: str,
):
    """
    Retrieve candidate chunks using semantic similarity,
    calculate cosine similarity and remove duplicates.
    """

    raw_results = vector_store.similarity_search_with_score(
        query,
        k=FETCH_K,
    )

    results = []
    seen = set()

    for document, distance in raw_results:

        similarity = distance_to_similarity(
            distance
        )

        if similarity < MIN_SIMILARITY:
            continue

        key = document_key(document)

        if key in seen:
            continue

        seen.add(key)

        results.append(
            RetrievedChunk(
                document=document,
                distance=float(distance),
                similarity=similarity,
            )
        )

    return results


def retrieve_with_similarity(
    vector_store,
    query: str,
):
    """
    Return the highest-similarity chunks.
    """

    candidates = get_similarity_candidates(
        vector_store,
        query,
    )

    return candidates[:TOP_K]


def retrieve_with_mmr(
    vector_store,
    query: str,
):
    """
    Use Maximum Marginal Relevance to select a mixture
    of relevant and non-redundant chunks.

    Similarity scores are taken from the original
    similarity-search candidate set.
    """

    candidates = get_similarity_candidates(
        vector_store,
        query,
    )

    candidate_map = {
        document_key(chunk.document): chunk
        for chunk in candidates
    }

    mmr_documents = (
        vector_store.max_marginal_relevance_search(
            query,
            k=TOP_K,
            fetch_k=FETCH_K,
            lambda_mult=MMR_LAMBDA_MULT,
        )
    )

    results = []
    seen = set()

    for document in mmr_documents:

        key = document_key(document)

        if key in seen:
            continue

        seen.add(key)

        candidate = candidate_map.get(key)

        if candidate is None:
            continue

        results.append(candidate)

    return results


def retrieve_documents(
    vector_store,
    query: str,
):
    """
    Main retrieval entry point.
    """

    if not query.strip():
        raise ValueError(
            "Query cannot be empty."
        )

    if RETRIEVAL_MODE == "similarity":
        return retrieve_with_similarity(
            vector_store,
            query,
        )

    if RETRIEVAL_MODE == "mmr":
        return retrieve_with_mmr(
            vector_store,
            query,
        )

    raise ValueError(
        f"Unsupported retrieval mode: {RETRIEVAL_MODE}"
    )


def format_source_reference(
    chunk: RetrievedChunk,
) -> str:
    """
    Create a human-readable source citation.
    """

    metadata = chunk.document.metadata

    file_name = metadata.get(
        "file_name",
        "Unknown source",
    )

    page_number = metadata.get(
        "page_number"
    )

    if page_number is not None:
        source = (
            f"{file_name}, page {page_number}"
        )
    else:
        source = file_name

    return source