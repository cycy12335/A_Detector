"""CW-016 的有限草稿 schema、确定性适配器和独立事实/引用校验。

候选对象始终被视为不可信数据。生成器只提供一个离线测试适配器，不能调用工具或外部模型；
校验器不信任生成器，重新从严格 ``DecisionRecord`` 派生字段和 CW-015 索引核对每个观察。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping

from ai_detector.retrieval import EvidenceChunk, RetrievalIndex, RetrievalError, build_index
from ai_detector.service.schemas import DecisionRecord
from ai_detector.storage.config import RECORD_SCHEMA_VERSION
from ai_detector.storage.records import StoredDecision, StorageContractError, encode_rule_ids

SCHEMA_VERSION = "cw016_draft_v1"
CONTEXT_SCHEMA_VERSION = "cw016_context_v1"
MAX_OBSERVATIONS = 4
MAX_CITATIONS_PER_OBSERVATION = 3
MAX_OUTPUT_CHARS = 8_000
ALLOWED_OBSERVATION_TYPES = (
    "model_threshold",
    "rule_evaluation",
    "research_signal",
    "known_limitations",
)
ALLOWED_STATUSES = ("verified", "insufficient_evidence")
RESEARCH_ONLY = "research_only"
POLICY_CITATIONS: dict[str, tuple[str, ...]] = {
    "model_threshold": ("cw015-policy-comparison",),
    "rule_evaluation": ("cw015-policy-rules", "cw015-policy-comparison"),
    "research_signal": ("cw015-policy-fusion", "cw015-policy-comparison"),
    "known_limitations": ("cw015-policy-limitations", "cw015-policy-fusion"),
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class InvestigationError(ValueError):
    """输入、上下文或草稿违反 CW-016 契约。"""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}" if detail else code)


@dataclass(frozen=True)
class DecisionSnapshot:
    """仅由 ``StoredDecision`` 已校验的派生字段组成的输入快照。"""

    record_schema_version: str
    decision_id: str
    model_version: str
    score: float
    score_semantics: str
    threshold: float
    threshold_policy: str
    flagged: bool
    policy_version: str
    policy_sha256: str
    policy_semantics: str
    rule_score: float
    matched_rule_ids: tuple[str, ...]
    unevaluable_rule_ids: tuple[str, ...]
    rule_evaluation_complete: bool
    signal_points: int | None
    risk_level: str


@dataclass(frozen=True)
class InvestigationContext:
    """与一个决策快照绑定的、只含政策证据的有界上下文。"""

    schema_version: str
    decision_id: str
    policy_version: str
    policy_sha256: str
    citation_ids: tuple[str, ...]


@dataclass(frozen=True)
class DraftValidation:
    """候选校验的安全结果；报告只使用 code，不回显候选载荷。"""

    status: str
    error_codes: tuple[str, ...]
    draft: dict[str, Any] | None

    @property
    def accepted(self) -> bool:
        return self.status in ALLOWED_STATUSES


def _exact_keys(value: Any, expected: set[str], code: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise InvestigationError(code)


def _text(value: Any, code: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvestigationError(code)
    return value


def _finite(value: Any, code: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise InvestigationError(code)
    return float(value)


def _bool(value: Any, code: str) -> bool:
    if not isinstance(value, bool):
        raise InvestigationError(code)
    return value


def _string_list(value: Any, code: str, *, max_items: int = 8) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) > max_items:
        raise InvestigationError(code)
    result = tuple(_text(item, code) for item in value)
    if len(set(result)) != len(result):
        raise InvestigationError(code)
    return result


def _stored_row(record: DecisionRecord) -> dict[str, Any]:
    """把 Pydantic 的严格派生字段交给既有跨字段校验器再校验一次。"""
    data = record.model_dump()
    data["matched_rule_ids"] = encode_rule_ids(data["matched_rule_ids"])
    data["unevaluable_rule_ids"] = encode_rule_ids(data["unevaluable_rule_ids"])
    data["flagged"] = int(data["flagged"])
    data["rule_evaluation_complete"] = int(data["rule_evaluation_complete"])
    return data


def _as_decision_record(value: DecisionRecord | Mapping[str, Any]) -> DecisionRecord:
    if isinstance(value, DecisionRecord):
        record = value
    elif isinstance(value, Mapping):
        try:
            record = DecisionRecord.model_validate(dict(value))
        except Exception as error:  # Pydantic errors may include input values; do not expose them.
            raise InvestigationError("decision_schema_invalid") from error
    else:
        raise InvestigationError("decision_schema_invalid")
    try:
        StoredDecision.from_row(_stored_row(record))
    except (StorageContractError, TypeError, ValueError) as error:
        raise InvestigationError("decision_contract_invalid") from error
    return record


def snapshot_from_decision(value: DecisionRecord | Mapping[str, Any]) -> DecisionSnapshot:
    """严格接受既有 ``DecisionRecord``，只复制允许进入调查的派生字段。"""
    record = _as_decision_record(value)
    data = record.model_dump()
    if data["record_schema_version"] != RECORD_SCHEMA_VERSION:
        raise InvestigationError("decision_schema_invalid")
    if not _SHA256.fullmatch(data["policy_sha256"]):
        raise InvestigationError("decision_contract_invalid")
    return DecisionSnapshot(
        record_schema_version=data["record_schema_version"],
        decision_id=data["decision_id"],
        model_version=data["model_version"],
        score=float(data["score"]),
        score_semantics=data["score_semantics"],
        threshold=float(data["threshold"]),
        threshold_policy=data["threshold_policy"],
        flagged=data["flagged"],
        policy_version=data["policy_version"],
        policy_sha256=data["policy_sha256"],
        policy_semantics=data["policy_semantics"],
        rule_score=float(data["rule_score"]),
        matched_rule_ids=tuple(data["matched_rule_ids"]),
        unevaluable_rule_ids=tuple(data["unevaluable_rule_ids"]),
        rule_evaluation_complete=data["rule_evaluation_complete"],
        signal_points=data["signal_points"],
        risk_level=data["risk_level"],
    )


def _policy_chunks(index: RetrievalIndex, snapshot: DecisionSnapshot) -> tuple[EvidenceChunk, ...]:
    chunks = tuple(chunk for chunk in index.chunks if chunk.authority == "policy")
    if not chunks or snapshot.policy_version not in index.policy_versions:
        raise InvestigationError("policy_version_mismatch")
    if any(chunk.source_version != snapshot.policy_version or chunk.source_sha256 != snapshot.policy_sha256 for chunk in chunks):
        raise InvestigationError("policy_digest_mismatch")
    return chunks


def build_context(value: DecisionSnapshot | DecisionRecord | Mapping[str, Any], index: RetrievalIndex | None = None) -> InvestigationContext:
    snapshot = value if isinstance(value, DecisionSnapshot) else snapshot_from_decision(value)
    evidence_index = index or build_index()
    chunks = _policy_chunks(evidence_index, snapshot)
    return InvestigationContext(
        schema_version=CONTEXT_SCHEMA_VERSION,
        decision_id=snapshot.decision_id,
        policy_version=snapshot.policy_version,
        policy_sha256=snapshot.policy_sha256,
        citation_ids=tuple(sorted(chunk.chunk_id for chunk in chunks)),
    )


def generate_candidate(snapshot: DecisionSnapshot, context: InvestigationContext) -> dict[str, Any]:
    """生成一个确定性的、仍需独立校验的不可信候选。"""
    if snapshot.decision_id != context.decision_id:
        raise InvestigationError("context_decision_mismatch")
    return {
        "schema_version": SCHEMA_VERSION,
        "decision_id": snapshot.decision_id,
        "policy_version": snapshot.policy_version,
        "policy_sha256": snapshot.policy_sha256,
        "status": "verified" if snapshot.rule_evaluation_complete else "insufficient_evidence",
        "disposition": RESEARCH_ONLY,
        "observations": [
            {
                "observation_id": "obs-model-threshold",
                "observation_type": "model_threshold",
                "decision_fields": ["score", "threshold", "flagged"],
                "citation_ids": list(POLICY_CITATIONS["model_threshold"]),
                "value": {"flagged": snapshot.flagged},
            },
            {
                "observation_id": "obs-rule-evaluation",
                "observation_type": "rule_evaluation",
                "decision_fields": ["matched_rule_ids", "unevaluable_rule_ids", "rule_evaluation_complete"],
                "citation_ids": list(POLICY_CITATIONS["rule_evaluation"]),
                "value": {
                    "matched_rule_ids": list(snapshot.matched_rule_ids),
                    "unevaluable_rule_ids": list(snapshot.unevaluable_rule_ids),
                    "complete": snapshot.rule_evaluation_complete,
                },
            },
            {
                "observation_id": "obs-research-signal",
                "observation_type": "research_signal",
                "decision_fields": ["flagged", "rule_score", "signal_points", "risk_level"],
                "citation_ids": list(POLICY_CITATIONS["research_signal"]),
                "value": {
                    "flagged": snapshot.flagged,
                    "rule_score": snapshot.rule_score,
                    "signal_points": snapshot.signal_points,
                    "risk_level": snapshot.risk_level,
                },
            },
            {
                "observation_id": "obs-known-limitations",
                "observation_type": "known_limitations",
                "decision_fields": ["score_semantics", "policy_semantics", "risk_level"],
                "citation_ids": list(POLICY_CITATIONS["known_limitations"]),
                "value": {"codes": ["uncalibrated_score", "research_signal_only", "anonymous_c13"]},
            },
        ],
    }


def _validate_citations(raw_ids: Any, expected: tuple[str, ...], context: InvestigationContext, index: RetrievalIndex) -> tuple[str, ...]:
    ids = _string_list(raw_ids, "citation_schema_invalid", max_items=MAX_CITATIONS_PER_OBSERVATION)
    if ids != expected:
        raise InvestigationError("citation_set_invalid")
    if any(citation_id not in context.citation_ids for citation_id in ids):
        raise InvestigationError("citation_not_in_context")
    for citation_id in ids:
        try:
            chunk = index.citation(citation_id)
        except RetrievalError as error:
            raise InvestigationError("citation_not_found") from error
        if chunk.authority != "policy" or chunk.source_version != context.policy_version or chunk.source_sha256 != context.policy_sha256:
            raise InvestigationError("citation_version_mismatch")
    return ids


def _validate_observation(raw: Any, snapshot: DecisionSnapshot, context: InvestigationContext, index: RetrievalIndex) -> dict[str, Any]:
    _exact_keys(raw, {"observation_id", "observation_type", "decision_fields", "citation_ids", "value"}, "observation_schema_invalid")
    observation_id = _text(raw["observation_id"], "observation_schema_invalid")
    observation_type = _text(raw["observation_type"], "observation_schema_invalid")
    if observation_type not in ALLOWED_OBSERVATION_TYPES:
        raise InvestigationError("observation_type_invalid")
    decision_fields = _string_list(raw["decision_fields"], "decision_fields_invalid", max_items=8)
    citation_ids = _validate_citations(raw["citation_ids"], POLICY_CITATIONS[observation_type], context, index)
    value = raw["value"]
    if observation_type == "model_threshold":
        _exact_keys(value, {"flagged"}, "observation_value_invalid")
        if _bool(value["flagged"], "observation_value_invalid") != snapshot.flagged or set(decision_fields) != {"score", "threshold", "flagged"}:
            raise InvestigationError("fact_mismatch")
    elif observation_type == "rule_evaluation":
        _exact_keys(value, {"matched_rule_ids", "unevaluable_rule_ids", "complete"}, "observation_value_invalid")
        matched = _string_list(value["matched_rule_ids"], "observation_value_invalid")
        unevaluable = _string_list(value["unevaluable_rule_ids"], "observation_value_invalid")
        if (matched, unevaluable, _bool(value["complete"], "observation_value_invalid")) != (snapshot.matched_rule_ids, snapshot.unevaluable_rule_ids, snapshot.rule_evaluation_complete):
            raise InvestigationError("fact_mismatch")
        if set(decision_fields) != {"matched_rule_ids", "unevaluable_rule_ids", "rule_evaluation_complete"}:
            raise InvestigationError("decision_fields_invalid")
    elif observation_type == "research_signal":
        _exact_keys(value, {"flagged", "rule_score", "signal_points", "risk_level"}, "observation_value_invalid")
        if value["flagged"] is not snapshot.flagged or _finite(value["rule_score"], "observation_value_invalid") != snapshot.rule_score or value["signal_points"] != snapshot.signal_points or value["risk_level"] != snapshot.risk_level:
            raise InvestigationError("fact_mismatch")
        if set(decision_fields) != {"flagged", "rule_score", "signal_points", "risk_level"}:
            raise InvestigationError("decision_fields_invalid")
    else:
        _exact_keys(value, {"codes"}, "observation_value_invalid")
        codes = _string_list(value["codes"], "observation_value_invalid", max_items=4)
        if set(codes) != {"uncalibrated_score", "research_signal_only", "anonymous_c13"}:
            raise InvestigationError("limitation_codes_invalid")
        if set(decision_fields) != {"score_semantics", "policy_semantics", "risk_level"}:
            raise InvestigationError("decision_fields_invalid")
    return {
        "observation_id": observation_id,
        "observation_type": observation_type,
        "decision_fields": list(decision_fields),
        "citation_ids": list(citation_ids),
        "value": value,
    }


def validate_candidate(candidate: Any, snapshot: DecisionSnapshot, context: InvestigationContext, index: RetrievalIndex | None = None) -> DraftValidation:
    """独立校验候选；任何错误只返回稳定分类码，不返回候选载荷。"""
    evidence_index = index or build_index()
    try:
        _exact_keys(candidate, {"schema_version", "decision_id", "policy_version", "policy_sha256", "status", "disposition", "observations"}, "draft_schema_invalid")
        if candidate["schema_version"] != SCHEMA_VERSION:
            raise InvestigationError("draft_version_mismatch")
        if candidate["decision_id"] != snapshot.decision_id or candidate["decision_id"] != context.decision_id:
            raise InvestigationError("context_decision_mismatch")
        if candidate["policy_version"] != snapshot.policy_version or candidate["policy_version"] != context.policy_version:
            raise InvestigationError("policy_version_mismatch")
        if candidate["policy_sha256"] != snapshot.policy_sha256 or candidate["policy_sha256"] != context.policy_sha256 or not _SHA256.fullmatch(candidate["policy_sha256"]):
            raise InvestigationError("policy_digest_mismatch")
        status = candidate["status"]
        if status not in ALLOWED_STATUSES or candidate["disposition"] != RESEARCH_ONLY:
            raise InvestigationError("draft_status_invalid")
        observations = candidate["observations"]
        if not isinstance(observations, list) or len(observations) != MAX_OBSERVATIONS:
            raise InvestigationError("observation_count_invalid")
        validated = [_validate_observation(item, snapshot, context, evidence_index) for item in observations]
        types = [item["observation_type"] for item in validated]
        if set(types) != set(ALLOWED_OBSERVATION_TYPES) or len(types) != len(set(types)):
            raise InvestigationError("observation_set_invalid")
        expected_status = "verified" if snapshot.rule_evaluation_complete else "insufficient_evidence"
        if status != expected_status:
            raise InvestigationError("evidence_status_invalid")
        public = {
            "schema_version": SCHEMA_VERSION,
            "decision_id": snapshot.decision_id,
            "policy_version": context.policy_version,
            "policy_sha256": context.policy_sha256,
            "status": status,
            "disposition": RESEARCH_ONLY,
            "observations": validated,
        }
        if len(str(public)) > MAX_OUTPUT_CHARS:
            raise InvestigationError("draft_size_exceeded")
        return DraftValidation(status, (), public)
    except InvestigationError as error:
        return DraftValidation("rejected", (error.code,), None)
    except (TypeError, AttributeError, KeyError, ValueError):
        return DraftValidation("rejected", ("draft_schema_invalid",), None)
