"""CW-015 可追溯、完全离线的规则证据检索。"""

from .core import (
    EvidenceChunk,
    RetrievalError,
    RetrievalIndex,
    SourceManifestError,
    build_index,
    load_source_manifest,
    retrieve_from_manifest,
)

__all__ = [
    "EvidenceChunk",
    "RetrievalError",
    "RetrievalIndex",
    "SourceManifestError",
    "build_index",
    "load_source_manifest",
    "retrieve_from_manifest",
]
