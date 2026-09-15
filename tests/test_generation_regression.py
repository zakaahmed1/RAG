import pytest

from app.evaluation.metrics import (
    contains_normalized_phrase,
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


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known pre-Phase-12 generation regression: "
        "AA003 can lose its question to prompt truncation "
        "and produce the wrong conclusion."
    ),
)
def test_aa003_generated_answer_rejects_delayed_reporting():
    """The wiped laptop must still be reported within two hours."""

    vector_store = load_vector_store()
    generator = create_generator()
    question = (
        "If a stolen company laptop has already "
        "been remotely wiped, can the employee "
        "wait until the next day to report it?"
    )

    chunks = retrieve_documents(
        vector_store,
        question,
    )
    diagnostics = {}
    answer = generate_answer(
        generator=generator,
        question=question,
        retrieved_chunks=chunks,
        diagnostics=diagnostics,
    )

    reports_two_hour_deadline = (
        contains_normalized_phrase(
            answer,
            "within 2 hours",
        )
        or contains_normalized_phrase(
            answer,
            "within two hours",
        )
    )
    rejects_waiting = any(
        contains_normalized_phrase(
            answer,
            phrase,
        )
        for phrase in (
            "no",
            "cannot wait",
            "can't wait",
            "must not wait",
        )
    )

    assert reports_two_hour_deadline, answer
    assert rejects_waiting, answer
    assert diagnostics["question_fully_retained"] is True
