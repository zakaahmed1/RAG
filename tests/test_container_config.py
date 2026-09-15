import re
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_dockerfile_pins_base_image_and_uses_non_root_user():
    dockerfile = (
        PROJECT_ROOT
        / "Dockerfile"
    ).read_text(encoding="utf-8")

    assert re.search(
        r"python:3\.13-slim@sha256:[0-9a-f]{64}",
        dockerfile,
    )
    assert "USER appuser:appuser" in dockerfile
    assert "HF_HOME=/home/appuser/.cache/huggingface" in dockerfile


def test_compose_enforces_runtime_hardening():
    compose = yaml.safe_load(
        (
            PROJECT_ROOT
            / "docker-compose.yaml"
        ).read_text(encoding="utf-8")
    )

    for service_name in ("api", "ui", "ingest"):
        service = compose["services"][service_name]

        assert service["user"] == "10001:10001"
        assert service["read_only"] is True
        assert service["cap_drop"] == ["ALL"]
        assert (
            "no-new-privileges:true"
            in service["security_opt"]
        )
        assert service["pids_limit"] > 0
        assert service["tmpfs"]

    api_volumes = compose["services"]["api"]["volumes"]
    assert (
        "huggingface_cache:/home/appuser/.cache/huggingface"
        in api_volumes
    )
    assert all(
        "/root/" not in volume
        for volume in api_volumes
    )

    ingest = compose["services"]["ingest"]
    assert ingest["profiles"] == ["tools"]
    assert (
        "./storage/faiss_index:/app/storage/faiss_index"
        in ingest["volumes"]
    )
