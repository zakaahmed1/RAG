import pytest
from app.evaluation.metrics import (
    chunk_matches_evidence, contains_normalized_phrase, evidence_recall,
    hit_at_k, is_abstention, keyword_coverage, normalize_text, reciprocal_rank,
)
def evidence(source,page,supporting_text):
    return {"source":source,"page":page,"supporting_text":supporting_text}
def group(*alternatives):
    return {"alternatives":list(alternatives)}
def test_normalize_text_converts_number_words():
    assert normalize_text("Within one hour.")=="within 1 hour"
def test_phrase_matching_uses_token_boundaries():
    assert contains_normalized_phrase("Report within 1 hour","1")
    assert not contains_normalized_phrase("Report within 10 hours","1")
    assert not contains_normalized_phrase("The cap is 150","15")
def test_keyword_coverage_handles_written_numbers():
    assert keyword_coverage("The incident must be reported within one hour.",["1"])==1.0
def test_keyword_coverage_matches_complete_phrase():
    assert keyword_coverage("Multi-factor authentication is mandatory.",["multi factor","authentication"])==1.0
def test_chunk_requires_source_page_and_supporting_text(handbook_chunk):
    assert chunk_matches_evidence(handbook_chunk,evidence("EmployeeHandbook.pdf",4,"25 days of paid annual leave"))
def test_same_page_without_supporting_fact_does_not_match(handbook_chunk):
    assert not chunk_matches_evidence(handbook_chunk,evidence("EmployeeHandbook.pdf",4,"unused leave must be taken by 31 March"))
def test_evidence_requires_supporting_text(handbook_chunk):
    with pytest.raises(ValueError,match="supporting_text"):
        chunk_matches_evidence(handbook_chunk,{"source":"EmployeeHandbook.pdf","page":4})
def test_alternative_sources_satisfy_one_required_group(security_chunk):
    required=[group(evidence("EmployeeHandbook.pdf",6,"reported within 2 hours"),evidence("ITSecurityPolicy.txt",None,"reported within 2 hours"))]
    assert evidence_recall([security_chunk],required)==1.0
    assert hit_at_k([security_chunk],required)==1
def test_independent_required_groups_report_partial_recall(handbook_chunk,security_chunk):
    required=[group(evidence("EmployeeHandbook.pdf",4,"25 days of paid annual leave")),group(evidence("ITSecurityPolicy.txt",None,"reported within 2 hours"))]
    assert evidence_recall([handbook_chunk],required)==0.5
    assert hit_at_k([handbook_chunk],required)==0
    assert hit_at_k([handbook_chunk,security_chunk],required)==1
def test_reciprocal_rank_uses_content_valid_match(handbook_chunk,security_chunk):
    required=[group(evidence("EmployeeHandbook.pdf",4,"25 days of paid annual leave"))]
    assert reciprocal_rank([security_chunk,handbook_chunk],required)==0.5
def test_abstention_detection():
    assert is_abstention("I could not find sufficient information in the supplied documents.")
