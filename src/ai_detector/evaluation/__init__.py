"""评估层：二分类指标计算与机器可读报告输出。"""

from ai_detector.evaluation.metrics import (
    MetricComputationError,
    check_probabilities,
    coerce_binary_labels,
    compute_binary_metrics,
)
from ai_detector.evaluation.report import (
    environment_versions,
    jsonable,
    write_json_report,
)

__all__ = [
    "MetricComputationError",
    "check_probabilities",
    "coerce_binary_labels",
    "compute_binary_metrics",
    "environment_versions",
    "jsonable",
    "write_json_report",
]
