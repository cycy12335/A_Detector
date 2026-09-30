"""CW-017 请求边界、离线编排和安全降级测试。"""

from __future__ import annotations

from typing import Any, Mapping

import pytest

from ai_detector.investigation import (
    ControlledGenerator,
    GenerationConfig,
    OfflineFakeTransport,
    ProviderTimeout,
    build_context,
    build_request,
    generate_candidate,
    mutate_candidate,
    synthetic_decision,
)
from ai_detector.retrieval import build_index


@pytest.fixture(scope="module")
def index():
    return build_index()


@pytest.fixture(scope="module")
def snapshot(index):
    chunk = next(item for item in index.chunks if item.authority == "policy")
    return synthetic_decision("complete_low", policy_sha256=chunk.source_sha256, policy_version=chunk.source_version)


@pytest.fixture(scope="module")
def context(index, snapshot):
    return build_context(snapshot, index)


def _transport(snapshot, context, mutation: str | None = None) -> OfflineFakeTransport:
    candidate = generate_candidate(snapshot, context)
    if mutation:
        candidate = mutate_candidate(candidate, mutation)
    return OfflineFakeTransport(candidate)


def test_request_is_bounded_and_excludes_raw_fields(index, snapshot, context) -> None:
    request = build_request(snapshot, context, GenerationConfig(), index)
    assert request.payload["schema_version"] == "cw017_generation_request_v1"
    assert request.payload["tools"] == []
    assert set(request.payload["input"]) == {"decision", "evidence", "output_schema_version", "disposition"}
    assert "features" not in request.payload["input"]["decision"]
    assert all("source_path" not in item for item in request.payload["input"]["evidence"])
    assert request.size_chars <= 12_000


def test_offline_verified_candidate_must_pass_cw016(index, snapshot, context) -> None:
    transport = _transport(snapshot, context)
    outcome = ControlledGenerator(GenerationConfig(), transport).generate(snapshot, context, index)
    assert outcome.status == "verified"
    assert outcome.validation_status == "verified"
    assert outcome.error_codes == ()
    assert outcome.draft is not None
    assert transport.calls == 1


def test_unevaluable_candidate_is_not_verified(index, context) -> None:
    chunk = next(item for item in index.chunks if item.authority == "policy")
    snapshot = synthetic_decision("unevaluable", policy_sha256=chunk.source_sha256, policy_version=chunk.source_version)
    transport = _transport(snapshot, context=build_context(snapshot, index))
    outcome = ControlledGenerator(GenerationConfig(), transport).generate(snapshot, build_context(snapshot, index), index)
    assert outcome.status == "insufficient_evidence"
    assert outcome.validation_status == "insufficient_evidence"


@pytest.mark.parametrize(
    ("mutation", "code"),
    [("extra_summary", "draft_schema_invalid"), ("flagged_conflict", "fact_mismatch"), ("citation_unknown", "citation_set_invalid")],
)
def test_untrusted_candidate_is_rejected(index, snapshot, context, mutation, code) -> None:
    outcome = ControlledGenerator(GenerationConfig(), _transport(snapshot, context, mutation)).generate(snapshot, context, index)
    assert outcome.status == "rejected"
    assert outcome.validation_status == "rejected"
    assert outcome.error_codes == (code,)
    assert outcome.draft is None


def test_timeout_retries_only_within_call_limit(index, snapshot, context) -> None:
    transport = _transport(snapshot, context)
    transport.failures = [ProviderTimeout()]
    config = GenerationConfig(max_retries=1, max_calls=2)
    outcome = ControlledGenerator(config, transport).generate(snapshot, context, index)
    assert outcome.status == "verified"
    assert outcome.attempts == 2
    assert outcome.calls == 2


def test_repeated_timeout_degrades_without_candidate(index, snapshot, context) -> None:
    transport = _transport(snapshot, context)
    transport.failures = [ProviderTimeout(), ProviderTimeout()]
    outcome = ControlledGenerator(GenerationConfig(max_retries=1, max_calls=2), transport).generate(snapshot, context, index)
    assert outcome.status == "degraded"
    assert outcome.error_codes == ("provider_timeout",)
    assert outcome.draft is None


def test_real_mode_requires_credential_without_calling_transport(index, snapshot, context) -> None:
    transport = _transport(snapshot, context)
    config = GenerationConfig(mode="real", provider_name="selected-provider", model_name="selected-model", credential_env="CW017_TEST_KEY", max_cost_usd=1.0)
    outcome = ControlledGenerator(config, transport).generate(snapshot, context, index)
    assert outcome.status == "degraded"
    assert outcome.error_codes == ("credentials_missing",)
    assert transport.calls == 0


def test_cost_limit_blocks_before_or_after_transport(index, snapshot, context) -> None:
    transport = _transport(snapshot, context)
    transport.cost_usd = 0.01
    config = GenerationConfig(max_cost_usd=0.0)
    outcome = ControlledGenerator(config, transport).generate(snapshot, context, index)
    assert outcome.status == "degraded"
    assert outcome.error_codes == ("cost_limit_exceeded",)


class _MalformedTransport:
    def complete(self, request: Mapping[str, Any], *, timeout_seconds: float) -> str:
        return '{"schema_version":"wrong","candidate":{}}'


def test_malformed_response_degrades_without_exposing_payload(index, snapshot, context) -> None:
    outcome = ControlledGenerator(GenerationConfig(), _MalformedTransport()).generate(snapshot, context, index)
    assert outcome.status == "degraded"
    assert outcome.error_codes == ("response_schema_invalid",)
    assert "candidate" not in outcome.audit_metadata()
