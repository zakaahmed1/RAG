from app.generation.generator import (
    create_generator,
    generate_answer,
)

from app.retrieval.search import (
    format_source_reference,
    retrieve_documents,
)

from app.retrieval.vector_store import (
    load_vector_store,
)


def initialise_rag():
    """
    Initialise the RAG pipeline using the persisted
    FAISS vector store.
    """

    print(
        "Loading persistent FAISS vector store..."
    )

    vector_store = load_vector_store()

    print(
        "Loading language model..."
    )

    generator = create_generator()

    print(
        "RAG assistant ready."
    )

    return vector_store, generator


def run():
    """
    Run the command-line RAG assistant.
    """

    vector_store, generator = initialise_rag()

    while True:

        question = input(
            "\nAsk a question "
            "(or type 'exit' to quit): "
        ).strip()

        if question.lower() == "exit":
            print("Goodbye.")
            break

        if not question:
            print(
                "Please enter a question."
            )
            continue

        retrieved_chunks = retrieve_documents(
            vector_store,
            question,
        )

        if not retrieved_chunks:

            print(
                "\n--- Answer ---"
            )

            print(
                "I could not find sufficient "
                "information in the supplied documents."
            )

            continue

        answer = generate_answer(
            generator=generator,
            question=question,
            retrieved_chunks=retrieved_chunks,
        )

        print(
            "\n--- Answer ---"
        )

        print(answer)

        print(
            "\n--- Sources ---"
        )

        for index, chunk in enumerate(
            retrieved_chunks,
            start=1,
        ):

            source = format_source_reference(
                chunk
            )

            print(
                f"[{index}] "
                f"{source} "
                f"(similarity: "
                f"{chunk.similarity:.3f})"
            )