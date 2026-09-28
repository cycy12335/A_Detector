"""CW-009 质量门禁的稳定错误码与异常。

## 细节只允许是"安全措辞"

:class:`QualityGateError` 的 ``detail`` 会被写进 JSON 报告与控制台。因此它**只**允许包含：
相对路径、字段名、case ID、数量、枚举值与异常类别名。绝不写入特征取值、完整特征映射、
请求体、数据库行、UUID、幂等摘要、绝对路径、bundle 内容或 pytest 输出。

这条纪律由 :mod:`ai_detector.quality.report` 的脱敏断言兜底：报告在落盘前会被再检查一次，
发现绝对路径、UUID、计划外的 64 位摘要、feature 名或类别取值就**失败关闭**（退出 6）。
"""

from __future__ import annotations


class QualityGateErrorCode:
    """门禁失败原因分类。取值稳定，允许写入报告。"""

    # --- 配置与路径 ---
    CONFIG_UNREADABLE = "config_unreadable"
    CONFIG_SCHEMA_VIOLATION = "config_schema_violation"
    CONFIG_PATH_UNSAFE = "config_path_unsafe"
    CONFIG_PATH_ALIAS = "config_path_alias"

    # --- 预检 ---
    INPUT_MISSING = "input_missing"
    INPUT_HASH_MISMATCH = "input_hash_mismatch"
    INPUT_CONTRACT_VIOLATION = "input_contract_violation"
    CORPUS_SCHEMA_VIOLATION = "corpus_schema_violation"
    CORPUS_CONTRACT_VIOLATION = "corpus_contract_violation"
    RUNTIME_LOAD_FAILED = "runtime_load_failed"
    AGREEMENT_VIOLATION = "agreement_violation"

    # --- 回归 ---
    REGRESSION_MISMATCH = "regression_mismatch"
    REGRESSION_NONDETERMINISTIC = "regression_nondeterministic"
    REGRESSION_BASELINE_MISMATCH = "regression_baseline_mismatch"

    # --- pytest 阶段 ---
    PYTEST_FAILED = "pytest_failed"
    PYTEST_TIMEOUT = "pytest_timeout"
    PYTEST_SPAWN_FAILED = "pytest_spawn_failed"

    # --- 报告 ---
    REPORT_WRITE_FAILED = "report_write_failed"
    REPORT_UNSAFE = "report_unsafe"

    # --- 兜底 ---
    INTERNAL_ERROR = "internal_error"


class QualityGateError(RuntimeError):
    """门禁以稳定分类码失败。``detail`` 必须是安全措辞（见模块文档）。"""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = str(code)
        self.detail = str(detail)
        super().__init__(f"{self.code}: {self.detail}" if self.detail else self.code)
