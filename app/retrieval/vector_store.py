from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import (
    EMBEDDING_MODEL_NAME,
    TOP_K,
    VECTOR_STORE_DIR,
)


def create_embedding_model():
    """
    Create the Hugging Face embedding model.
    """

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        encode_kwargs={
            "normalize_embeddings": True
        }
    )

    return embeddings


def create_vector_store(chunks):
    """
    Create a FAISS vector store from document chunks.
    """

    if not chunks:
        raise ValueError(
            "Cannot create vector store: no document chunks supplied."
        )

    embeddings = create_embedding_model()

    vector_store = FAISS.from_documents(
        chunks,
        embeddings,
    )

    return vector_store


def save_vector_store(vector_store):
    """
    Persist the FAISS vector store to disk.
    """

    VECTOR_STORE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    vector_store.save_local(
        str(VECTOR_STORE_DIR)
    )


def load_vector_store():
    """
    Load an existing FAISS vector store from disk.
    """

    index_file = VECTOR_STORE_DIR / "index.faiss"
    metadata_file = VECTOR_STORE_DIR / "index.pkl"

    if not index_file.exists() or not metadata_file.exists():
        raise FileNotFoundError(
            "FAISS vector store not found. "
            "Run 'python -m app.ingestion.ingest' first."
        )

    embeddings = create_embedding_model()

    vector_store = FAISS.load_local(
        str(VECTOR_STORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    return vector_store