from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import (
    EMBEDDING_MODEL_NAME,
    TOP_K,
)


def create_embedding_model():
    """
    Create the Hugging Face embedding model used
    to convert document chunks into vectors.
    """

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME
    )

    return embeddings


def create_vector_store(chunks):
    """
    Create an in-memory FAISS vector store from
    the supplied document chunks.
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


def create_retriever(vector_store):
    """
    Create a retriever from the FAISS vector store.
    """

    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": TOP_K
        }
    )

    return retriever