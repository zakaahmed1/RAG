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
    A retrieved document chunk together with
    its vector-search scoring information.
    """

    document: Document
    distance: float
    similarity: float


def distance_to_similarity(distance: float) -> float:
    """
    Convert squared L2 distance between normalised vectors
    into cosine similarity.
    """

    similarity = 1.0 - (float(distance) / 2.0)

    return max(
        -1.0,
        min(1.0, similarity),
    )


def document_key(document: Document):
    """
    Stable identifier for duplicate detection.
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
    fetch_k: int = FETCH_K,
    min_similarity: float = MIN_SIMILARITY,
):
    """
    Retrieve semantic-search candidates, calculate cosine
    similarity, apply thresholding and remove duplicates.
    """

    raw_results = vector_store.similarity_search_with_score(
        query,
        k=fetch_k,
    )

    results = []
    seen = set()

    for document, distance in raw_results:

        similarity = distance_to_similarity(
            distance
        )

        if similarity < min_similarity:
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
    top_k: int = TOP_K,
    fetch_k: int = FETCH_K,
    min_similarity: float = MIN_SIMILARITY,
):
    """
    Return the highest-similarity chunks.
    """

    candidates = get_similarity_candidates(
        vector_store=vector_store,
        query=query,
        fetch_k=max(fetch_k, top_k),
        min_similarity=min_similarity,
    )

    return candidates[:top_k]


def retrieve_with_mmr(
    vector_store,
    query: str,
    top_k: int = TOP_K,
    fetch_k: int = FETCH_K,
    min_similarity: float = MIN_SIMILARITY,
    mmr_lambda: float = MMR_LAMBDA_MULT,
):
    """
    Use Maximum Marginal Relevance to balance relevance
    and diversity.
    """

    fetch_k = max(fetch_k, top_k)

    candidates = get_similarity_candidates(
        vector_store=vector_store,
        query=query,
        fetch_k=fetch_k,
        min_similarity=min_similarity,
    )

    candidate_map = {
        document_key(chunk.document): chunk
        for chunk in candidates
    }

    mmr_documents = vector_store.max_marginal_relevance_search(
        query,
        k=top_k,
        fetch_k=fetch_k,
        lambda_mult=mmr_lambda,
    )

    results = []
    seen = set()

    for document in mmr_documents:

        key = document_key(document)

        if key in seen:
            continue

        candidate = candidate_map.get(key)

        # The MMR result may have been below our threshold.
        if candidate is None:
            continue

        seen.add(key)
        results.append(candidate)

    return results


def retrieve_documents(
    vector_store,
    query: str,
    *,
    mode: str = RETRIEVAL_MODE,
    top_k: int = TOP_K,
    fetch_k: int = FETCH_K,
    min_similarity: float = MIN_SIMILARITY,
    mmr_lambda: float = MMR_LAMBDA_MULT,
):
    """
    Main configurable retrieval entry point.
    """

    if not query.strip():
        raise ValueError("Query cannot be empty.")

    if mode == "similarity":
        return retrieve_with_similarity(
            vector_store=vector_store,
            query=query,
            top_k=top_k,
            fetch_k=fetch_k,
            min_similarity=min_similarity,
        )

    if mode == "mmr":
        return retrieve_with_mmr(
            vector_store=vector_store,
            query=query,
            top_k=top_k,
            fetch_k=fetch_k,
            min_similarity=min_similarity,
            mmr_lambda=mmr_lambda,
        )

    raise ValueError(
        f"Unsupported retrieval mode: {mode}"
    )


def format_source_reference(
    chunk: RetrievedChunk,
) -> str:

    metadata = chunk.document.metadata

    file_name = metadata.get(
        "file_name",
        "Unknown source",
    )

    page_number = metadata.get("page_number")

    if page_number is not None:
        return f"{file_name}, page {page_number}"

    return file_name