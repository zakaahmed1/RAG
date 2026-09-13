from app.evaluation.metrics import (
    chunk_matches_evidence,
    evidence_recall,
    hit_at_k,
    is_abstention,
    keyword_coverage,
    normalize_text,
    reciprocal_rank,
)


def test_normalize_text_converts_number_words():
    assert normalize_text(
        "Within one hour."
    ) == "within 1 hour"


def test_keyword_coverage_handles_written_numbers():
    score = keyword_coverage(
        "The incident must be reported within one hour.",
        ["1"],
    )

    assert score == 1.0


def test_keyword_coverage_multiple_keywords():
    score = keyword_coverage(
        "Multi-factor authentication is mandatory.",
        ["multi", "factor"],
    )

    assert score == 1.0


def test_chunk_matches_expected_evidence(
    handbook_chunk,
):
    evidence = {
        "source": "EmployeeHandbook.pdf",
        "page": 4,
    }

    assert chunk_matches_evidence(
        handbook_chunk,
        evidence,
    )


def test_evidence_recall_all_evidence_found(
    handbook_chunk,
    security_chunk,
):
    evidence = [
        {
            "source": "EmployeeHandbook.pdf",
            "page": 4,
        },
        {
            "source": "ITSecurityPolicy.txt",
            "page": None,
        },
    ]

    recall = evidence_recall(
        [
            handbook_chunk,
            security_chunk,
        ],
        evidence,
    )

    assert recall == 1.0


def test_evidence_recall_partial(
    handbook_chunk,
):
    evidence = [
        {
            "source": "EmployeeHandbook.pdf",
            "page": 4,
        },
        {
            "source": "ITSecurityPolicy.txt",
            "page": None,
        },
    ]

    recall = evidence_recall(
        [handbook_chunk],
        evidence,
    )

    assert recall == 0.5


def test_hit_at_k_any(
    handbook_chunk,
):
    evidence = [
        {
            "source": "EmployeeHandbook.pdf",
            "page": 4,
        },
        {
            "source": "ITSecurityPolicy.txt",
            "page": None,
        },
    ]

    assert hit_at_k(
        [handbook_chunk],
        evidence,
        "any",
    ) == 1


def test_hit_at_k_all_requires_every_source(
    handbook_chunk,
):
    evidence = [
        {
            "source": "EmployeeHandbook.pdf",
            "page": 4,
        },
        {
            "source": "ITSecurityPolicy.txt",
            "page": None,
        },
    ]

    assert hit_at_k(
        [handbook_chunk],
        evidence,
        "all",
    ) == 0


def test_reciprocal_rank(
    handbook_chunk,
    security_chunk,
):
    expected = [
        {
            "source": "EmployeeHandbook.pdf",
            "page": 4,
        }
    ]

    rank = reciprocal_rank(
        [
            security_chunk,
            handbook_chunk,
        ],
        expected,
    )

    assert rank == 0.5


def test_abstention_detection():
    answer = (
        "I could not find sufficient information "
        "in the supplied documents."
    )

    assert is_abstention(answer)