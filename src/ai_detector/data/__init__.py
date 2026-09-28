"""数据层：交易数据的模式定义与确定性模拟生成。"""

from ai_detector.data.schema import (
    ALL_COLUMNS,
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    ID_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    TIMESTAMP_COLUMN,
    SchemaError,
    validate_transactions,
)

__all__ = [
    "ALL_COLUMNS",
    "CATEGORICAL_FEATURES",
    "FEATURE_COLUMNS",
    "ID_COLUMNS",
    "NUMERIC_FEATURES",
    "TARGET_COLUMN",
    "TIMESTAMP_COLUMN",
    "SchemaError",
    "validate_transactions",
]
