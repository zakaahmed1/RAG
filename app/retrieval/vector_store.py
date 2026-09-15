import hashlib
import json
from datetime import datetime, timezone

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCUMENTS_DIR,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_MODEL_REVISION,
    FETCH_K,
    INDEX_MANIFEST_PATH,
    INDEX_SCHEMA_VERSION,
    MIN_SIMILARITY,
    MMR_LAMBDA_MULT,
    RETRIEVAL_MODE,
    SUPPORTED_EXTENSIONS,
    TOP_K,
    VECTOR_STORE_DIR,
)


class IncompatibleIndexError(RuntimeError):
    """Raised when the persisted index does not match runtime config."""


def source_documents_fingerprint(documents_dir=DOCUMENTS_DIR):
    """Return a deterministic SHA-256 fingerprint of source paths/content."""
    digest = hashlib.sha256()
    file_paths = sorted(
        path for path in documents_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not file_paths:
        raise ValueError(f"No supported documents found in {documents_dir}")
    for file_path in file_paths:
        relative_path = file_path.relative_to(documents_dir).as_posix()
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        with file_path.open("rb") as source_file:
            for block in iter(lambda: source_file.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def build_index_manifest(chunk_count=None):
    return {
        "schema_version": INDEX_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "embedding": {
            "model": EMBEDDING_MODEL_NAME,
            "revision": EMBEDDING_MODEL_REVISION,
            "normalize_embeddings": True,
        },
        "chunking": {"chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP},
        "index": {
            "type": "FAISS",
            "distance_strategy": "squared_l2_on_normalized_vectors",
            "chunk_count": chunk_count,
        },
        "retrieval": {
            "mode": RETRIEVAL_MODE,
            "top_k": TOP_K,
            "fetch_k": FETCH_K,
            "min_similarity": MIN_SIMILARITY,
            "mmr_lambda_mult": MMR_LAMBDA_MULT,
        },
        "source_documents": {
            "algorithm": "sha256",
            "fingerprint": source_documents_fingerprint(),
        },
    }


def save_index_manifest(chunk_count=None):
    manifest = build_index_manifest(chunk_count=chunk_count)
    temporary_path = INDEX_MANIFEST_PATH.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_path.replace(INDEX_MANIFEST_PATH)
    return manifest


def validate_index_manifest():
    if not INDEX_MANIFEST_PATH.exists():
        raise IncompatibleIndexError(
            "FAISS index manifest not found. Rebuild the index with "
            "'python -m app.ingestion.ingest'."
        )
    try:
        manifest = json.loads(INDEX_MANIFEST_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise IncompatibleIndexError(
            "FAISS index manifest is unreadable. Rebuild the index."
        ) from error
    expected = build_index_manifest()
    comparisons = {
        "schema_version": (manifest.get("schema_version"), expected["schema_version"]),
        "embedding.model": (manifest.get("embedding", {}).get("model"), expected["embedding"]["model"]),
        "embedding.revision": (manifest.get("embedding", {}).get("revision"), expected["embedding"]["revision"]),
        "embedding.normalize_embeddings": (manifest.get("embedding", {}).get("normalize_embeddings"), expected["embedding"]["normalize_embeddings"]),
        "chunking.chunk_size": (manifest.get("chunking", {}).get("chunk_size"), expected["chunking"]["chunk_size"]),
        "chunking.chunk_overlap": (manifest.get("chunking", {}).get("chunk_overlap"), expected["chunking"]["chunk_overlap"]),
        "source_documents.fingerprint": (manifest.get("source_documents", {}).get("fingerprint"), expected["source_documents"]["fingerprint"]),
    }
    mismatches = [name for name, (actual, wanted) in comparisons.items() if actual != wanted]
    if mismatches:
        raise IncompatibleIndexError(
            "FAISS index is stale or incompatible (mismatch: "
            + ", ".join(mismatches)
            + "). Rebuild it with 'python -m app.ingestion.ingest'."
        )
    return manifest


def create_embedding_model():
    """
    Create the Hugging Face embedding model.
    """

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"revision": EMBEDDING_MODEL_REVISION},
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


def save_vector_store(vector_store, chunk_count=None):
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
    save_index_manifest(chunk_count=chunk_count)


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

    validate_index_manifest()

    embeddings = create_embedding_model()

    vector_store = FAISS.load_local(
        str(VECTOR_STORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    return vector_store
