import json
import pytest
from app.retrieval import vector_store


def test_source_fingerprint_changes_with_document_content(tmp_path):
    document = tmp_path / "policy.txt"
    document.write_text("first version", encoding="utf-8")
    first = vector_store.source_documents_fingerprint(tmp_path)
    document.write_text("second version", encoding="utf-8")
    second = vector_store.source_documents_fingerprint(tmp_path)
    assert first != second


def test_validate_manifest_rejects_embedding_revision_mismatch(tmp_path, monkeypatch):
    manifest_path = tmp_path / "manifest.json"
    monkeypatch.setattr(vector_store, "INDEX_MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(vector_store, "source_documents_fingerprint", lambda: "documents-hash")
    manifest = vector_store.build_index_manifest(chunk_count=3)
    manifest["embedding"]["revision"] = "different-revision"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(vector_store.IncompatibleIndexError, match="embedding.revision"):
        vector_store.validate_index_manifest()


def test_validate_manifest_accepts_matching_index(tmp_path, monkeypatch):
    manifest_path = tmp_path / "manifest.json"
    monkeypatch.setattr(vector_store, "INDEX_MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(vector_store, "source_documents_fingerprint", lambda: "documents-hash")
    manifest = vector_store.build_index_manifest(chunk_count=3)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    loaded = vector_store.validate_index_manifest()
    assert loaded["index"]["chunk_count"] == 3
