"""CW-016 结构化调查草稿与证据一致性校验。"""

from .draft import (
    DecisionSnapshot,
    DraftValidation,
    InvestigationContext,
    InvestigationError,
    build_context,
    generate_candidate,
    snapshot_from_decision,
    validate_candidate,
)
from .synthetic import mutate_candidate, synthetic_decision

__all__ = [
    "DraftValidation",
    "DecisionSnapshot",
    "InvestigationContext",
    "InvestigationError",
    "build_context",
    "generate_candidate",
    "snapshot_from_decision",
    "validate_candidate",
    "mutate_candidate",
    "synthetic_decision",
]
