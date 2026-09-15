import re

from app.config import ABSTENTION_TEXT


def normalize_text(text: str) -> str:
    """Normalise text for deterministic evaluation."""
    text = str(text).lower()
    number_words = {
        "zero": "0", "one": "1", "two": "2", "three": "3",
        "four": "4", "five": "5", "six": "6", "seven": "7",
        "eight": "8", "nine": "9", "ten": "10", "eleven": "11",
        "twelve": "12", "thirteen": "13", "fourteen": "14",
        "fifteen": "15", "sixteen": "16", "seventeen": "17",
        "eighteen": "18", "nineteen": "19", "twenty": "20",
    }
    for word, number in number_words.items():
        text = re.sub(rf"\b{word}\b", number, text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def contains_normalized_phrase(text, phrase):
    """Match complete normalized tokens/phrases, never substrings."""
    normalized_text = normalize_text(text)
    normalized_phrase = normalize_text(phrase)
    if not normalized_phrase:
        return False
    return f" {normalized_phrase} " in f" {normalized_text} "


def chunk_matches_evidence(chunk, evidence):
    """Match source/page and the labelled supporting text."""
    metadata = chunk.document.metadata
    if metadata.get("file_name") != evidence.get("source"):
        return False
    expected_page = evidence.get("page")
    if expected_page is not None and metadata.get("page_number") != expected_page:
        return False
    supporting_text = evidence.get("supporting_text")
    if not supporting_text:
        raise ValueError("Every evidence alternative requires supporting_text.")
    return contains_normalized_phrase(chunk.document.page_content, supporting_text)


def group_matches_evidence(retrieved_chunks, evidence_group):
    """Return true when any acceptable alternative satisfies a group."""
    alternatives = evidence_group.get("alternatives", [])
    if not alternatives:
        raise ValueError("Every required evidence group needs alternatives.")
    return any(
        chunk_matches_evidence(chunk, alternative)
        for alternative in alternatives
        for chunk in retrieved_chunks
    )


def evidence_recall(retrieved_chunks, required_evidence):
    """Fraction of independently required evidence groups retrieved."""
    if not required_evidence:
        return None
    matched = sum(
        group_matches_evidence(retrieved_chunks, group)
        for group in required_evidence
    )
    return matched / len(required_evidence)


def hit_at_k(retrieved_chunks, required_evidence):
    """Return one only when every required evidence group is satisfied."""
    recall = evidence_recall(retrieved_chunks, required_evidence)
    return None if recall is None else int(recall == 1.0)


def relevant_rank(retrieved_chunks, required_evidence):
    """Rank of the first chunk satisfying any labelled alternative."""
    alternatives = [
        alternative
        for group in required_evidence
        for alternative in group.get("alternatives", [])
    ]
    for rank, chunk in enumerate(retrieved_chunks, start=1):
        if any(chunk_matches_evidence(chunk, item) for item in alternatives):
            return rank
    return None


def reciprocal_rank(retrieved_chunks, required_evidence):
    """Reciprocal rank of the first content-valid evidence chunk."""
    rank = relevant_rank(retrieved_chunks, required_evidence)
    return 0.0 if rank is None else 1.0 / rank


def keyword_coverage(answer, expected_keywords):
    """Fraction of expected token/phrase labels present in the answer."""
    if not expected_keywords:
        return None
    hits = sum(
        contains_normalized_phrase(answer, keyword)
        for keyword in expected_keywords
    )
    return hits / len(expected_keywords)


def is_abstention(answer: str) -> bool:
    """Detect the standard abstention response."""
    return contains_normalized_phrase(answer, ABSTENTION_TEXT)
