import re

from app.config import ABSTENTION_TEXT

def normalize_text(text: str) -> str:
    """
    Normalise text for deterministic answer evaluation.
    """

    text = text.lower()

    number_words = {
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
        "eleven": "11",
        "twelve": "12",
        "thirteen": "13",
        "fourteen": "14",
        "fifteen": "15",
        "sixteen": "16",
        "seventeen": "17",
        "eighteen": "18",
        "nineteen": "19",
        "twenty": "20",
    }

    for word, number in number_words.items():
        text = re.sub(
            rf"\b{word}\b",
            number,
            text,
        )

    text = re.sub(
        r"[^\w\s]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def chunk_matches_evidence(
    chunk,
    evidence,
):
    """
    Check whether a retrieved chunk matches one
    labelled evidence item.
    """

    metadata = chunk.document.metadata

    actual_source = metadata.get("file_name")
    actual_page = metadata.get("page_number")

    expected_source = evidence.get("source")
    expected_page = evidence.get("page")

    if actual_source != expected_source:
        return False

    if expected_page is not None:
        return actual_page == expected_page

    return True


def evidence_recall(
    retrieved_chunks,
    expected_evidence,
):
    """
    Fraction of labelled evidence items retrieved.
    """

    if not expected_evidence:
        return None

    matched = 0

    for evidence in expected_evidence:

        if any(
            chunk_matches_evidence(
                chunk,
                evidence,
            )
            for chunk in retrieved_chunks
        ):
            matched += 1

    return matched / len(expected_evidence)


def hit_at_k(
    retrieved_chunks,
    expected_evidence,
    evidence_requirement,
):
    """
    Determine whether retrieval satisfies the
    labelled evidence requirement.

    any = at least one expected evidence item retrieved
    all = every expected evidence item retrieved
    """

    recall = evidence_recall(
        retrieved_chunks,
        expected_evidence,
    )

    if recall is None:
        return None

    if evidence_requirement == "any":
        return int(recall > 0)

    if evidence_requirement == "all":
        return int(recall == 1.0)

    raise ValueError(
        "Supported questions must use "
        "'any' or 'all' evidence requirements."
    )


def relevant_rank(
    retrieved_chunks,
    expected_evidence,
):
    """
    Rank of the first retrieved chunk matching
    any labelled evidence item.
    """

    for rank, chunk in enumerate(
        retrieved_chunks,
        start=1,
    ):

        if any(
            chunk_matches_evidence(
                chunk,
                evidence,
            )
            for evidence in expected_evidence
        ):
            return rank

    return None


def reciprocal_rank(
    retrieved_chunks,
    expected_evidence,
):
    """
    Reciprocal rank of the first relevant chunk.
    """

    rank = relevant_rank(
        retrieved_chunks,
        expected_evidence,
    )

    if rank is None:
        return 0.0

    return 1.0 / rank


def keyword_coverage(
    answer,
    expected_keywords,
):
    """
    Proportion of expected answer keywords appearing
    in the generated answer.
    """

    if not expected_keywords:
        return None

    normalized_answer = normalize_text(
        answer
    )

    hits = 0

    for keyword in expected_keywords:

        normalized_keyword = normalize_text(
            str(keyword)
        )

        if normalized_keyword in normalized_answer:
            hits += 1

    return hits / len(expected_keywords)


def is_abstention(answer: str) -> bool:
    """
    Detect the standard abstention response.
    """

    return (
        normalize_text(ABSTENTION_TEXT)
        in normalize_text(answer)
    )