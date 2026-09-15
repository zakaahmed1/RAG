from app.api.service import (
    RAGService,
)


def initialise_rag():
    """
    Initialise the shared RAG service used by the API.
    """

    print(
        "Loading RAG service..."
    )

    service = RAGService()
    service.start()

    print(
        "RAG assistant ready."
    )

    return service


def format_cli_source(source):
    """Format service source metadata for terminal output."""

    reference = source["file"]
    page = source.get("page")

    if page is not None:
        reference = f"{reference}, page {page}"

    return (
        f"{reference} "
        f"(similarity: {source['similarity']:.3f})"
    )


def run():
    """
    Run the command-line RAG assistant.
    """

    service = initialise_rag()

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

        result = service.query(
            question,
        )

        print(
            "\n--- Answer ---"
        )

        print(result["answer"])

        if not result["sources"]:
            continue

        print(
            "\n--- Sources ---"
        )

        for index, source in enumerate(
            result["sources"],
            start=1,
        ):
            print(
                f"[{index}] "
                f"{format_cli_source(source)}"
            )
