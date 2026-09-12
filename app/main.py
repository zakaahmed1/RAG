from app.generation.generator import (
    create_generator,
    generate_answer,
)
from app.retrieval.vector_store import (
    create_retriever,
    load_vector_store,
)


def initialise_rag():
    """
    Initialise the RAG pipeline using an existing
    persistent FAISS vector store.
    """

    print(
        "Loading persistent FAISS vector store..."
    )

    vector_store = load_vector_store()

    print(
        "Creating retriever..."
    )

    retriever = create_retriever(
        vector_store
    )

    print(
        "Loading language model..."
    )

    generator = create_generator()

    print(
        "RAG assistant ready."
    )

    return retriever, generator


def run():
    """
    Run the command-line RAG assistant.
    """

    retriever, generator = initialise_rag()

    while True:

        question = input(
            "\nAsk a question (or type 'exit' to quit): "
        ).strip()

        if question.lower() == "exit":
            print("Goodbye.")
            break

        if not question:
            print(
                "Please enter a question."
            )
            continue

        retrieved_documents = retriever.invoke(
            question
        )

        answer = generate_answer(
            generator=generator,
            question=question,
            retrieved_documents=retrieved_documents,
        )

        print("\n--- Answer ---")
        print(answer)