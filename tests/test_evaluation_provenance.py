import pytest
from app.evaluation.evaluate import (
    build_prompt_usage_summary,
    build_run_provenance,
    validate_dataset,
)
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


def test_prompt_usage_summary_aggregates_generation_diagnostics():
    rows = [
        {
            "prompt_token_usage": {
                "tokens_before_truncation": 600,
                "prompt_truncated": True,
                "context_fully_retained": False,
                "question_fully_retained": False,
            }
        },
        {
            "prompt_token_usage": {
                "tokens_before_truncation": 400,
                "prompt_truncated": False,
                "context_fully_retained": True,
                "question_fully_retained": True,
            }
        },
        {"prompt_token_usage": None},
    ]

    summary = build_prompt_usage_summary(rows)

    assert summary == {
        "generated_answers": 2,
        "truncation_rate": 0.5,
        "context_fully_retained_rate": 0.5,
        "question_fully_retained_rate": 0.5,
        "maximum_tokens_before_truncation": 600,
    }
