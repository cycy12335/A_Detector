"""CW-015 的来源校验、证据构建和确定性词法检索。

这里的 ``score`` 只是可解释的词项重合排序分数，不是模型分数、概率或风险判断。
索引只从受审查来源的精确行范围构建；运行阶段不会读取交易数据、数据库或模型。
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from ai_detector.config import PROJECT_ROOT, resolve_path

MANIFEST_PATH = "configs/retrieval_sources.json"
MANIFEST_SCHEMA_VERSION = "cw015_sources_v1"
QUERY_MAX_LENGTH = 512
MAX_TOP_K = 10
MIN_OUTPUT_CHARS = 256
MAX_OUTPUT_CHARS = 16_000
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TOKEN = re.compile(r"[A-Za-z0-9_]+|[\u3400-\u4dbf\u4e00-\u9fff]+")
_FORBIDDEN_SOURCE_PARTS = {
    "data",
    "artifacts",
    "database",
    "databases",
    "models",
    "raw",
    "processed",
}


class SourceManifestError(ValueError):
    """来源清单或来源内容不能安全进入知识库。"""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


class RetrievalError(ValueError):
    """检索请求或索引契约不合法。"""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    path: str
    sha256: str
    version: str
    authority: str
    fragments: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class EvidenceChunk:
    """可由来源清单复核的一个证据片段。"""

    chunk_id: str
    text: str
    source_id: str
    source_path: str
    source_sha256: str
    source_version: str
    authority: str
    line_start: int
    line_end: int
    locator: str
    tags: tuple[str, ...]

    def citation(self) -> dict[str, Any]:
        return {
            "citation_id": self.chunk_id,
            "source_id": self.source_id,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "source_version": self.source_version,
            "locator": self.locator,
            "authority": self.authority,
            "tags": list(self.tags),
        }


@dataclass(frozen=True)
class RetrievalHit:
    chunk: EvidenceChunk
    score: float
    matched_terms: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "citation": self.chunk.citation(),
            "text": self.chunk.text,
            "retrieval_score": self.score,
            "matched_terms": list(self.matched_terms),
        }


def _assert_relative_path(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SourceManifestError("source_path_unsafe", f"{label} 必须是非空字符串")
    text = value.replace("\\", "/")
    windows = PureWindowsPath(value)
    posix = PurePosixPath(text)
    if (
        "\\" in value
        or windows.is_absolute()
        or bool(windows.drive)
        or posix.is_absolute()
        or text.startswith("/")
        or text.endswith("/")
    ):
        raise SourceManifestError("source_path_unsafe", f"{label} 必须是项目相对路径")
    if any(part in {"", ".", ".."} for part in text.split("/")):
        raise SourceManifestError("source_path_unsafe", f"{label} 含非法路径组件")
    parts = tuple(part.casefold() for part in posix.parts)
    if _FORBIDDEN_SOURCE_PARTS.intersection(parts):
        raise SourceManifestError("source_not_allowed", f"{label} 不在检索白名单")
    return text


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _exact_keys(value: Any, expected: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        raise SourceManifestError("manifest_schema_invalid", f"{label} 字段集合不匹配")


def _parse_manifest(raw: Any) -> tuple[SourceSpec, ...]:
    _exact_keys(raw, {"schema_version", "knowledge_base_version", "sources"}, "manifest")
    if raw["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise SourceManifestError("manifest_version_mismatch", "来源清单 schema 版本不匹配")
    if not isinstance(raw["knowledge_base_version"], str) or not raw["knowledge_base_version"].strip():
        raise SourceManifestError("manifest_schema_invalid", "知识库版本不能为空")
    sources = raw["sources"]
    if not isinstance(sources, list) or not sources:
        raise SourceManifestError("manifest_schema_invalid", "sources 必须是非空列表")
    parsed: list[SourceSpec] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(sources):
        label = f"sources[{index}]"
        _exact_keys(item, {"source_id", "path", "sha256", "version", "authority", "fragments"}, label)
        source_id = item["source_id"]
        if not isinstance(source_id, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{2,63}", source_id):
            raise SourceManifestError("manifest_schema_invalid", f"{label}.source_id 不合法")
        if source_id in seen_ids:
            raise SourceManifestError("manifest_schema_invalid", "source_id 不得重复")
        seen_ids.add(source_id)
        path = _assert_relative_path(item["path"], label=f"{label}.path")
        sha256 = item["sha256"]
        if not isinstance(sha256, str) or not _SHA256.fullmatch(sha256):
            raise SourceManifestError("manifest_schema_invalid", f"{label}.sha256 不合法")
        version = item["version"]
        authority = item["authority"]
        if not isinstance(version, str) or not version.strip() or authority not in {"policy", "explanation"}:
            raise SourceManifestError("manifest_schema_invalid", f"{label}.version/authority 不合法")
        fragments = item["fragments"]
        if not isinstance(fragments, list) or not fragments:
            raise SourceManifestError("manifest_schema_invalid", f"{label}.fragments 必须非空")
        parsed_fragments: list[dict[str, Any]] = []
        seen_fragment_ids: set[str] = set()
        for fragment_index, fragment in enumerate(fragments):
            fragment_label = f"{label}.fragments[{fragment_index}]"
            _exact_keys(fragment, {"chunk_id", "line_start", "line_end", "locator", "tags"}, fragment_label)
            chunk_id = fragment["chunk_id"]
            if not isinstance(chunk_id, str) or not re.fullmatch(r"cw015-[a-z0-9_-]{3,80}", chunk_id):
                raise SourceManifestError("manifest_schema_invalid", f"{fragment_label}.chunk_id 不合法")
            if chunk_id in seen_fragment_ids:
                raise SourceManifestError("manifest_schema_invalid", "chunk_id 不得重复")
            seen_fragment_ids.add(chunk_id)
            start, end = fragment["line_start"], fragment["line_end"]
            if isinstance(start, bool) or not isinstance(start, int) or isinstance(end, bool) or not isinstance(end, int):
                raise SourceManifestError("manifest_schema_invalid", f"{fragment_label} 行号必须是整数")
            if start < 1 or end < start:
                raise SourceManifestError("manifest_schema_invalid", f"{fragment_label} 行范围不合法")
            if not isinstance(fragment["locator"], str) or not fragment["locator"].strip():
                raise SourceManifestError("manifest_schema_invalid", f"{fragment_label}.locator 不能为空")
            tags = fragment["tags"]
            if not isinstance(tags, list) or not tags or any(not isinstance(tag, str) or not tag for tag in tags):
                raise SourceManifestError("manifest_schema_invalid", f"{fragment_label}.tags 不合法")
            parsed_fragments.append(
                {"chunk_id": chunk_id, "line_start": start, "line_end": end, "locator": fragment["locator"], "tags": tuple(tags)}
            )
        parsed.append(SourceSpec(source_id, path, sha256, version, authority, tuple(parsed_fragments)))
    return tuple(parsed)


def load_source_manifest(path: str | Path = MANIFEST_PATH) -> tuple[SourceSpec, ...]:
    """读取来源清单；只接受项目相对路径，并在解析前验证清单本身的 JSON 形状。"""
    relative = _assert_relative_path(str(path), label="manifest.path")
    manifest_path = resolve_path(relative)
    if not manifest_path.is_file():
        raise SourceManifestError("manifest_missing", "来源清单不存在")
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SourceManifestError("manifest_parse_failed", f"来源清单无法读取：{type(error).__name__}") from error
    return _parse_manifest(raw)


def _read_verified_source(spec: SourceSpec) -> tuple[Path, list[str]]:
    source_path = resolve_path(spec.path)
    project_root = PROJECT_ROOT.resolve()
    resolved = source_path.resolve()
    if not resolved.is_relative_to(project_root):
        raise SourceManifestError("source_path_unsafe", f"来源 {spec.source_id} 解析后越出项目根")
    if not source_path.is_file():
        raise SourceManifestError("source_missing", f"来源 {spec.source_id} 不存在")
    try:
        actual = _sha256(source_path)
    except OSError as error:
        raise SourceManifestError("source_unreadable", f"来源 {spec.source_id} 无法读取") from error
    if actual != spec.sha256:
        raise SourceManifestError("source_hash_mismatch", f"来源 {spec.source_id} 摘要不符")
    try:
        lines = source_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise SourceManifestError("source_unreadable", f"来源 {spec.source_id} 不是可读 UTF-8 文本") from error
    return source_path, lines


def build_index(manifest_path: str | Path = MANIFEST_PATH) -> RetrievalIndex:
    """校验全部来源并从清单行范围构建索引。"""
    specs = load_source_manifest(manifest_path)
    chunks: list[EvidenceChunk] = []
    for spec in specs:
        _source_path, lines = _read_verified_source(spec)
        for fragment in spec.fragments:
            start, end = fragment["line_start"], fragment["line_end"]
            if end > len(lines):
                raise SourceManifestError("locator_invalid", f"片段 {fragment['chunk_id']} 超出来源行数")
            text = "\n".join(lines[start - 1 : end]).strip()
            if not text:
                raise SourceManifestError("locator_invalid", f"片段 {fragment['chunk_id']} 为空")
            chunks.append(
                EvidenceChunk(
                    chunk_id=fragment["chunk_id"],
                    text=text,
                    source_id=spec.source_id,
                    source_path=spec.path,
                    source_sha256=spec.sha256,
                    source_version=spec.version,
                    authority=spec.authority,
                    line_start=start,
                    line_end=end,
                    locator=fragment["locator"],
                    tags=tuple(fragment["tags"]),
                )
            )
    return RetrievalIndex(tuple(chunks))


def _tokens(text: str) -> tuple[str, ...]:
    tokens: list[str] = []
    for item in _TOKEN.findall(text.casefold()):
        if re.fullmatch(r"[\u3400-\u4dbf\u4e00-\u9fff]+", item):
            if len(item) == 1:
                tokens.append(item)
            else:
                tokens.extend(item[index : index + 2] for index in range(len(item) - 1))
        else:
            tokens.append(item)
    return tuple(dict.fromkeys(tokens))


@dataclass(frozen=True)
class RetrievalIndex:
    chunks: tuple[EvidenceChunk, ...]

    def __post_init__(self) -> None:
        ids = [chunk.chunk_id for chunk in self.chunks]
        if not ids or len(set(ids)) != len(ids):
            raise RetrievalError("index_invalid", "索引必须包含唯一证据 ID")
        if len({chunk.source_version for chunk in self.chunks if chunk.authority == "policy"}) > 1:
            raise RetrievalError("policy_version_mismatch", "索引不得混合多个政策版本")

    @property
    def policy_versions(self) -> tuple[str, ...]:
        return tuple(sorted({chunk.source_version for chunk in self.chunks if chunk.authority == "policy"}))

    def citation(self, citation_id: str) -> EvidenceChunk:
        for chunk in self.chunks:
            if chunk.chunk_id == citation_id:
                return chunk
        raise RetrievalError("citation_not_found", "引用 ID 不存在")

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        max_output_chars: int = 4_000,
        required_policy_version: str | None = None,
    ) -> tuple[RetrievalHit, ...]:
        if not isinstance(query, str) or not query.strip() or len(query) > QUERY_MAX_LENGTH:
            raise RetrievalError("query_invalid", "查询必须是非空且不超过 512 字符的文本")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= MAX_TOP_K:
            raise RetrievalError("top_k_invalid", "top_k 必须位于 1..10")
        if isinstance(max_output_chars, bool) or not isinstance(max_output_chars, int) or not MIN_OUTPUT_CHARS <= max_output_chars <= MAX_OUTPUT_CHARS:
            raise RetrievalError("output_limit_invalid", "max_output_chars 超出允许范围")
        if required_policy_version is not None and required_policy_version not in self.policy_versions:
            raise RetrievalError("policy_version_mismatch", "请求的政策版本不在索引中")
        query_tokens = _tokens(query)
        if not query_tokens:
            return ()
        query_set = set(query_tokens)
        scored: list[tuple[float, EvidenceChunk, tuple[str, ...]]] = []
        normalized_query = " ".join(query_tokens)
        for chunk in self.chunks:
            chunk_tokens = set(_tokens(chunk.text))
            matched = tuple(token for token in query_tokens if token in chunk_tokens)
            if not matched:
                continue
            score = len(matched) / len(query_set)
            normalized_text = " ".join(_tokens(chunk.text))
            if normalized_query and normalized_query in normalized_text:
                score += 0.25
            scored.append((score, chunk, matched))
        scored.sort(key=lambda item: (-item[0], item[1].chunk_id))
        selected: list[RetrievalHit] = []
        used_chars = 0
        for score, chunk, matched in scored[:top_k]:
            citation_overhead = len(json.dumps(chunk.citation(), ensure_ascii=False))
            item_chars = len(chunk.text) + citation_overhead
            if item_chars > max_output_chars:
                continue
            if selected and used_chars + item_chars > max_output_chars:
                break
            selected.append(RetrievalHit(chunk, round(score, 8), matched))
            used_chars += item_chars
        return tuple(selected)

    def validate_hit(self, hit: RetrievalHit) -> None:
        """确认返回对象仍是该索引的完整引用，不接受伪造 citation。"""
        stored = self.citation(hit.chunk.chunk_id)
        if hit.chunk != stored:
            raise RetrievalError("citation_invalid", "引用内容与索引不一致")

    def as_dict(self) -> dict[str, Any]:
        return {"schema_version": MANIFEST_SCHEMA_VERSION, "chunks": [chunk.citation() | {"text": chunk.text} for chunk in self.chunks]}


def retrieve_from_manifest(query: str, **kwargs: Any) -> tuple[RetrievalHit, ...]:
    """一次性构建并检索，适合 CLI；服务化调用应复用 ``RetrievalIndex``。"""
    return build_index().retrieve(query, **kwargs)

