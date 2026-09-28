"""模型层：基线分类器与阈值选择。"""

from ai_detector.models.baseline import (
    DUMMY_MODEL_KEY,
    LOGISTIC_MODEL_KEY,
    ModelSpecError,
    build_estimator,
    build_model_specs,
)
from ai_detector.models.threshold import (
    THRESHOLD_RULE_DESCRIPTION,
    ThresholdChoice,
    select_threshold,
)

__all__ = [
    "DUMMY_MODEL_KEY",
    "LOGISTIC_MODEL_KEY",
    "THRESHOLD_RULE_DESCRIPTION",
    "ModelSpecError",
    "ThresholdChoice",
    "build_estimator",
    "build_model_specs",
    "select_threshold",
]
