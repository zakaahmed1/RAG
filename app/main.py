from app.ingestion.loader import load_and_split_documents
from app.retrieval.vector_store import (
    create_vector_store,
    create_retriever,
)
from app.generation.generator import (
    create_generator,
    generate_answer,
)


def initialise_rag():
    """
    Initialise the complete RAG pipeline.
    """

    print("Loading and chunking documents...")

    chunks = load_and_split_documents()

    print(f"Created {len(chunks)} document chunks.")


    print("Creating vector store...")

    vector_store = create_vector_store(chunks)


    print("Creating retriever...")

    retriever = create_retriever(vector_store)


    print("Loading language model...")

    generator = create_generator()


    print("RAG assistant ready.")

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
            print("Please enter a question.")
            continue


        retrieved_documents = retriever.invoke(question)


        answer = generate_answer(
            generator=generator,
            question=question,
            retrieved_documents=retrieved_documents,
        )


        print("\n--- Answer ---")
        print(answer)