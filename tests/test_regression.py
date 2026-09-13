import pytest

from app.retrieval.search import (
    retrieve_documents,
)

from app.retrieval.vector_store import (
    load_vector_store,
)


@pytest.mark.integration
def test_aa003_retrieves_device_policy():

    vector_store = load_vector_store()

    question = (
        "If a stolen company laptop has already "
        "been remotely wiped, can the employee "
        "wait until the next day to report it?"
    )

    chunks = retrieve_documents(
        vector_store,
        question,
    )

    evidence = {
        (
            chunk.document.metadata.get(
                "file_name"
            ),
            chunk.document.metadata.get(
                "page_number"
            ),
        )
        for chunk in chunks
    }

    assert (
        "EmployeeHandbook.pdf",
        6,
    ) in evidence

    assert (
        "ITSecurityPolicy.txt",
        None,
    ) in evidence