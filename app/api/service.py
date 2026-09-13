from threading import Lock

from app.config import (
    ABSTENTION_TEXT,
    EMBEDDING_MODEL_NAME,
    FETCH_K,
    GENERATOR_MODEL_NAME,
    MIN_SIMILARITY,
    MMR_LAMBDA_MULT,
    RETRIEVAL_MODE,
    TOP_K,
)

from app.generation.generator import (
    create_generator,
    generate_answer,
)

from app.retrieval.search import (
    retrieve_documents,
)

from app.retrieval.vector_store import (
    load_vector_store,
)


class RAGService:
    """
    Long-lived RAG service used by the FastAPI application.

    The FAISS index and generation model are loaded once
    at application startup and reused for subsequent
    requests.
    """

    def __init__(self):

        self.vector_store = None
        self.generator = None

        self.ready = False

        # Serialise access to the current local generator.
        # This can be revisited when production concurrency
        # requirements are introduced.
        self._generation_lock = Lock()


    def start(self):
        """
        Load the persisted vector store and generator.
        """

        print(
            "Loading persistent FAISS vector store..."
        )

        self.vector_store = load_vector_store()

        print(
            "Loading language model..."
        )

        self.generator = create_generator()

        self.ready = True

        print(
            "RAG service ready."
        )


    def query(
        self,
        question: str,
    ):
        """
        Retrieve supporting evidence and generate
        a grounded answer.
        """

        if not self.ready:
            raise RuntimeError(
                "RAG service is not ready."
            )

        question = question.strip()

        if not question:
            raise ValueError(
                "Question cannot be empty."
            )

        retrieved_chunks = retrieve_documents(
            self.vector_store,
            question,
            mode=RETRIEVAL_MODE,
            top_k=TOP_K,
            fetch_k=FETCH_K,
            min_similarity=MIN_SIMILARITY,
            mmr_lambda=MMR_LAMBDA_MULT,
        )

        retrieval_rejected = (
            len(retrieved_chunks) == 0
        )

        if retrieval_rejected:

            answer = ABSTENTION_TEXT

        else:

            with self._generation_lock:

                answer = generate_answer(
                    generator=self.generator,
                    question=question,
                    retrieved_chunks=(
                        retrieved_chunks
                    ),
                )

        sources = []

        for chunk in retrieved_chunks:

            metadata = (
                chunk.document.metadata
            )

            sources.append(
                {
                    "file": metadata.get(
                        "file_name",
                        "Unknown source",
                    ),
                    "page": metadata.get(
                        "page_number"
                    ),
                    "chunk_id": metadata.get(
                        "chunk_id"
                    ),
                    "similarity": round(
                        chunk.similarity,
                        4,
                    ),
                }
            )

        return {
            "question": question,
            "answer": answer,
            "retrieval_rejected":
                retrieval_rejected,
            "retrieved_count":
                len(retrieved_chunks),
            "sources": sources,
        }


    def status(self):
        """
        Return the active RAG configuration.
        """

        return {
            "ready": self.ready,
            "retrieval_mode":
                RETRIEVAL_MODE,
            "top_k":
                TOP_K,
            "fetch_k":
                FETCH_K,
            "min_similarity":
                MIN_SIMILARITY,
            "embedding_model":
                EMBEDDING_MODEL_NAME,
            "generator_model":
                GENERATOR_MODEL_NAME,
        }