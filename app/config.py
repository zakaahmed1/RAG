import os

from pathlib import Path


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"

STORAGE_DIR = PROJECT_ROOT / "storage"
VECTOR_STORE_DIR = STORAGE_DIR / "faiss_index"


# ---------------------------------------------------------
# Supported document types
# ---------------------------------------------------------

SUPPORTED_EXTENSIONS = {
    ".txt",
    ".md",
    ".pdf",
    ".docx",
}


# ---------------------------------------------------------
# Document chunking
# ---------------------------------------------------------

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


# ---------------------------------------------------------
# Embeddings
# ---------------------------------------------------------

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------------------------------------
# Retrieval
# ---------------------------------------------------------

RETRIEVAL_MODE = "similarity"

TOP_K = 4
FETCH_K = 12

MMR_LAMBDA_MULT = 0.7

MIN_SIMILARITY = 0.425


# ---------------------------------------------------------
# Generation
# ---------------------------------------------------------

GENERATOR_MODEL_NAME = "google/flan-t5-base"

MAX_NEW_TOKENS = 150

# ---------------------------------------------------------
# Grounding / abstention
# ---------------------------------------------------------

ABSTENTION_TEXT = (
    "I could not find sufficient information "
    "in the supplied documents."
)

# ---------------------------------------------------------
# Observability
# ---------------------------------------------------------

LOG_LEVEL = os.getenv(
    "LOG_LEVEL",
    "INFO",
)

# ---------------------------------------------------------
# Security
# ---------------------------------------------------------

APP_ENV = os.getenv(
    "APP_ENV",
    "development",
).lower()

MAX_QUERY_LENGTH = 2000

RAG_API_KEY = os.getenv(
    "RAG_API_KEY",
    "",
).strip()


def _csv_env(
    name: str,
    default: str = "",
) -> list[str]:
    return [
        value.strip()
        for value in os.getenv(
            name,
            default,
        ).split(",")
        if value.strip()
    ]


TRUSTED_HOSTS = _csv_env(
    "TRUSTED_HOSTS",
    (
        "localhost,"
        "127.0.0.1,"
        "api,"
        "testserver"
    ),
)

ALLOWED_ORIGINS = _csv_env(
    "ALLOWED_ORIGINS"
)

_default_docs = (
    "false"
    if APP_ENV == "production"
    else "true"
)

ENABLE_DOCS = (
    os.getenv(
        "ENABLE_DOCS",
        _default_docs,
    ).lower()
    in {
        "1",
        "true",
        "yes",
    }
)