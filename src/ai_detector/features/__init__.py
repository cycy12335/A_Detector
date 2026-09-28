"""特征层：时间切分与仅在训练集拟合的预处理管线。"""

from ai_detector.features.pipeline import (
    build_model_pipeline,
    build_preprocessor,
    split_features_target,
)
from ai_detector.features.split import (
    SplitError,
    TimeSplit,
    assert_split_is_valid,
    time_based_split,
)

__all__ = [
    "SplitError",
    "TimeSplit",
    "assert_split_is_valid",
    "build_model_pipeline",
    "build_preprocessor",
    "split_features_target",
    "time_based_split",
]
