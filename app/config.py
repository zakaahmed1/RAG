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