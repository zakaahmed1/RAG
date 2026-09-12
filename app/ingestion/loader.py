from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import (
    DOCUMENT_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


def load_document():
    """
    Load the source document from disk.
    """

    if not DOCUMENT_PATH.exists():
        raise FileNotFoundError(
            f"Document not found: {DOCUMENT_PATH}"
        )

    loader = TextLoader(
        str(DOCUMENT_PATH),
        encoding="utf-8",
    )

    documents = loader.load()

    return documents


def split_documents(documents):
    """
    Split loaded documents into smaller chunks suitable
    for embedding and retrieval.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = splitter.split_documents(documents)

    return chunks


def load_and_split_documents():
    """
    Convenience function that loads the source document
    and returns the resulting chunks.
    """

    documents = load_document()

    chunks = split_documents(documents)

    return chunks