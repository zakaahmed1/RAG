from pathlib import Path

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCUMENTS_DIR,
    SUPPORTED_EXTENSIONS,
)


def get_document_loader(file_path: Path):
    """
    Return the appropriate LangChain loader for a file.
    """

    extension = file_path.suffix.lower()

    if extension in {".txt", ".md"}:
        return TextLoader(
            str(file_path),
            encoding="utf-8",
        )

    if extension == ".pdf":
        return PyPDFLoader(
            str(file_path)
        )

    if extension == ".docx":
        return Docx2txtLoader(
            str(file_path)
        )

    return None


def load_documents():
    """
    Load all supported documents from the documents directory.
    """

    if not DOCUMENTS_DIR.exists():
        raise FileNotFoundError(
            f"Documents directory not found: {DOCUMENTS_DIR}"
        )

    file_paths = sorted(
        path
        for path in DOCUMENTS_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not file_paths:
        raise ValueError(
            f"No supported documents found in {DOCUMENTS_DIR}"
        )

    documents = []

    for file_path in file_paths:

        loader = get_document_loader(file_path)

        if loader is None:
            continue

        print(f"Loading: {file_path.name}")

        loaded_documents = loader.load()

        for document in loaded_documents:

            document.metadata["file_name"] = file_path.name
            document.metadata["file_type"] = file_path.suffix.lower()
            document.metadata["source_path"] = str(
                file_path.relative_to(DOCUMENTS_DIR)
            )

            # PyPDFLoader provides zero-indexed page metadata.
            if "page" in document.metadata:
                document.metadata["page_number"] = (
                    document.metadata["page"] + 1
                )

        documents.extend(
            loaded_documents
        )

    return documents


def split_documents(documents):
    """
    Split documents into chunks suitable for embedding.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = splitter.split_documents(
        documents
    )

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index

    return chunks


def load_and_split_documents():
    """
    Load all supported documents and split them into chunks.
    """

    documents = load_documents()

    chunks = split_documents(
        documents
    )

    return chunks