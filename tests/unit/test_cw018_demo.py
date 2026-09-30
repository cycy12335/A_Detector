"""CW-018 container/demo contract tests; no Docker daemon required."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from ai_detector.service.config import ServiceConfig
from ai_detector.service.errors import BundleLoadError
from ai_detector.service.runtime import load_runtime
from scripts.run_demo import preflight


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEMO_CONFIG = PROJECT_ROOT / "configs" / "demo_service.yaml"
COMPOSE_FILE = PROJECT_ROOT / "compose.demo.yaml"
DOCKERIGNORE_FILE = PROJECT_ROOT / ".dockerignore"


def test_demo_config_is_read_only_and_uses_frozen_inputs() -> None:
    config = ServiceConfig.from_yaml(DEMO_CONFIG)

    assert config.storage_path is None
    assert config.bundle_sha256 == "f02bb50e61c5f70c85448430c8bb653bd6e347f400220771500ee84a70135a1c"
    assert config.policy_sha256 == "7ed07f0b784715c01b444035bb34ea5166d1b376ff07e10428561d898d6824f0"
    assert config.host == "127.0.0.1"
    assert config.expected_source_features == 199


def test_demo_preflight_loads_the_accepted_bundle_and_policy() -> None:
    config, runtime, policy = preflight(DEMO_CONFIG)

    assert config.name == "ai-detector-risk-demo"
    assert runtime.bundle_sha256 == config.bundle_sha256
    assert len(runtime.source_features) == 199
    assert policy.policy_version == "cw006-risk-policy-v1"


def test_tampered_or_missing_bundle_fails_before_deserialization(tmp_path: Path) -> None:
    config = ServiceConfig.from_yaml(DEMO_CONFIG)
    missing = replace(config, bundle_path=tmp_path / "missing.joblib")
    with pytest.raises(BundleLoadError) as missing_error:
        load_runtime(missing)
    assert missing_error.value.code == "bundle_missing"

    tampered_path = tmp_path / "tampered.joblib"
    tampered_path.write_bytes(config.bundle_path.read_bytes() + b"tamper")
    tampered = replace(config, bundle_path=tampered_path)
    with pytest.raises(BundleLoadError) as tampered_error:
        load_runtime(tampered)
    assert tampered_error.value.code == "bundle_hash_mismatch"


def test_compose_is_local_non_root_read_only_and_has_healthcheck() -> None:
    compose = yaml.safe_load(COMPOSE_FILE.read_text(encoding="utf-8"))
    service = compose["services"]["risk-demo"]

    assert service["ports"] == ["127.0.0.1:8000:8000"]
    assert service["read_only"] is True
    assert service["user"] == "10001:10001"
    assert service["cap_drop"] == ["ALL"]
    assert service["security_opt"] == ["no-new-privileges:true"]
    assert service["healthcheck"]["test"][0:2] == ["CMD", "python"]
    serialized = json.dumps(compose, ensure_ascii=False).lower()
    assert "deepseek" not in serialized
    assert "cw007_decisions.sqlite3" not in serialized


def test_docker_context_excludes_models_data_and_credentials() -> None:
    dockerignore = DOCKERIGNORE_FILE.read_text(encoding="utf-8")

    for entry in ("artifacts", "data", ".handoff", "*.env", "*.key", "*.pem", "secrets.*"):
        assert entry in dockerignore
