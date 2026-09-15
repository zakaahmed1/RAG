from app.ingestion.loader import load_and_split_documents
from app.retrieval.vector_store import (
    create_vector_store,
    save_vector_store,
)


def run_ingestion():
    """
    Build and persist the vector index from source documents.
    """

    print("Starting document ingestion...")

    chunks = load_and_split_documents()

    print(
        f"Created {len(chunks)} document chunks."
    )

    print(
        "Creating FAISS vector store..."
    )

    vector_store = create_vector_store(
        chunks
    )

    print(
        "Saving FAISS vector store..."
    )

    save_vector_store(
        vector_store,
        chunk_count=len(chunks),
    )

    print(
        "Ingestion complete."
    )


if __name__ == "__main__":
    run_ingestion()
