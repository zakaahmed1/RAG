import logging

from threading import Lock
from time import perf_counter

from app.config import (
    ABSTENTION_TEXT,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_MODEL_REVISION,
    FETCH_K,
    GENERATOR_MODEL_NAME,
    GENERATOR_MODEL_REVISION,
    MIN_SIMILARITY,
    MMR_LAMBDA_MULT,
    RETRIEVAL_MODE,
    TOP_K,
)

from app.generation.generator import (
    create_generator,
    generate_answer,
)

from app.observability.logging import (
    log_event,
)

from app.retrieval.search import (
    retrieve_documents,
)

from app.retrieval.vector_store import (
    load_vector_store,
)


logger = logging.getLogger(
    "rag.service"
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
        self._generation_lock = Lock()


    def start(self):
        """
        Load the persisted vector store and generator.
        """

        service_start = (
            perf_counter()
        )

        # -------------------------------------------------
        # Vector store
        # -------------------------------------------------

        vector_start = (
            perf_counter()
        )

        self.vector_store = (
            load_vector_store()
        )

        vector_ms = (
            perf_counter()
            - vector_start
        ) * 1000

        log_event(
            logger,
            "vector_store_loaded",
            duration_ms=round(
                vector_ms,
                2,
            ),
            retrieval_mode=(
                RETRIEVAL_MODE
            ),
        )

        # -------------------------------------------------
        # Generator
        # -------------------------------------------------

        generator_start = (
            perf_counter()
        )

        self.generator = (
            create_generator()
        )

        generator_ms = (
            perf_counter()
            - generator_start
        ) * 1000

        log_event(
            logger,
            "generator_loaded",
            duration_ms=round(
                generator_ms,
                2,
            ),
            model=(
                GENERATOR_MODEL_NAME
            ),
        )

        self.ready = True

        total_ms = (
            perf_counter()
            - service_start
        ) * 1000

        log_event(
            logger,
            "rag_service_ready",
            duration_ms=round(
                total_ms,
                2,
            ),
        )


    def query(
        self,
        question: str,
    ):
        """
        Retrieve supporting evidence and generate
        a grounded answer.
        """

        total_start = (
            perf_counter()
        )

        if not self.ready:

            raise RuntimeError(
                "RAG service is not ready."
            )

        question = (
            question.strip()
        )

        if not question:

            raise ValueError(
                "Question cannot be empty."
            )

        # -------------------------------------------------
        # Retrieval
        # -------------------------------------------------

        retrieval_start = (
            perf_counter()
        )

        try:

            retrieved_chunks = (
                retrieve_documents(
                    self.vector_store,
                    question,
                    mode=RETRIEVAL_MODE,
                    top_k=TOP_K,
                    fetch_k=FETCH_K,
                    min_similarity=(
                        MIN_SIMILARITY
                    ),
                    mmr_lambda=(
                        MMR_LAMBDA_MULT
                    ),
                )
            )

        except Exception:

            retrieval_ms = (
                perf_counter()
                - retrieval_start
            ) * 1000

            log_event(
                logger,
                "retrieval_failed",
                level=logging.ERROR,
                exc_info=True,
                question_length=len(
                    question
                ),
                retrieval_ms=round(
                    retrieval_ms,
                    2,
                ),
            )

            raise

        retrieval_ms = (
            perf_counter()
            - retrieval_start
        ) * 1000

        retrieval_rejected = (
            len(
                retrieved_chunks
            ) == 0
        )

        top_similarity = None

        if retrieved_chunks:

            top_similarity = max(
                chunk.similarity
                for chunk
                in retrieved_chunks
            )

        log_event(
            logger,
            "retrieval_completed",
            question_length=len(
                question
            ),
            retrieved_count=len(
                retrieved_chunks
            ),
            retrieval_rejected=(
                retrieval_rejected
            ),
            top_similarity=(
                round(
                    top_similarity,
                    4,
                )
                if top_similarity
                is not None
                else None
            ),
            retrieval_ms=round(
                retrieval_ms,
                2,
            ),
        )

        # -------------------------------------------------
        # Generation
        # -------------------------------------------------

        generation_ms = 0.0

        if retrieval_rejected:

            answer = (
                ABSTENTION_TEXT
            )

        else:

            generation_start = (
                perf_counter()
            )

            try:

                with self._generation_lock:

                    answer = (
                        generate_answer(
                            generator=(
                                self.generator
                            ),
                            question=question,
                            retrieved_chunks=(
                                retrieved_chunks
                            ),
                        )
                    )

            except Exception:

                generation_ms = (
                    perf_counter()
                    - generation_start
                ) * 1000

                log_event(
                    logger,
                    "generation_failed",
                    level=logging.ERROR,
                    exc_info=True,
                    generation_ms=round(
                        generation_ms,
                        2,
                    ),
                )

                raise

            generation_ms = (
                perf_counter()
                - generation_start
            ) * 1000

            log_event(
                logger,
                "generation_completed",
                generation_ms=round(
                    generation_ms,
                    2,
                ),
                answer_length=len(
                    answer
                ),
                abstained=(
                    answer
                    == ABSTENTION_TEXT
                ),
            )

        # -------------------------------------------------
        # Sources
        # -------------------------------------------------

        sources = []

        for chunk in (
            retrieved_chunks
        ):

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

        total_ms = (
            perf_counter()
            - total_start
        ) * 1000

        log_event(
            logger,
            "rag_query_completed",
            question_length=len(
                question
            ),
            retrieved_count=len(
                retrieved_chunks
            ),
            retrieval_rejected=(
                retrieval_rejected
            ),
            top_similarity=(
                round(
                    top_similarity,
                    4,
                )
                if top_similarity
                is not None
                else None
            ),
            retrieval_ms=round(
                retrieval_ms,
                2,
            ),
            generation_ms=round(
                generation_ms,
                2,
            ),
            total_ms=round(
                total_ms,
                2,
            ),
        )

        return {
            "question": question,
            "answer": answer,
            "retrieval_rejected":
                retrieval_rejected,
            "retrieved_count":
                len(
                    retrieved_chunks
                ),
            "sources":
                sources,
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
            "embedding_revision":
                EMBEDDING_MODEL_REVISION,
            "generator_model":
                GENERATOR_MODEL_NAME,
            "generator_revision":
                GENERATOR_MODEL_REVISION,
        }
