import pytest
from app.evaluation.evaluate import build_run_provenance, validate_dataset
def test_provenance_records_reproducibility_inputs():
    p=build_run_provenance(retrieval_only=True,split="test")
    assert len(p["dataset_sha256"])==64
    assert p["embedding"]["revision"] and p["generation"]["revision"]
    assert p["chunking"]["chunk_size"]==500 and p["retrieval"]["top_k"]==4
    assert p["evaluation"]=={"retrieval_only":True,"split":"test"}
def test_dataset_validation_rejects_legacy_evidence_schema():
    with pytest.raises(ValueError,match="required_evidence"):
        validate_dataset([{"id":"Q1","supported":True,"expected_evidence":[{"source":"policy.txt","page":None}]}])
def test_dataset_validation_rejects_missing_supporting_text():
    data=[{"id":"Q1","supported":True,"required_evidence":[{"alternatives":[{"source":"policy.txt","page":None}]}]}]
    with pytest.raises(ValueError,match="supporting_text"):
        validate_dataset(data)
