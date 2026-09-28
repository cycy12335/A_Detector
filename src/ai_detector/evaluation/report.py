"""机器可读 JSON 报告的写出与环境版本采集。

报告中的每个数值都由代码计算，不存在手工填写；报告路径由配置决定，默认写入
``artifacts/reports/``（已在 ``.gitignore`` 中排除）。
"""

from __future__ import annotations

import importlib
import json
import platform
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

#: 报告需要记录的运行时依赖：(报告字段名, 导入名)。
#: 必须与 ``pyproject.toml`` 的 ``[project].dependencies`` 保持一致，否则报告无法
#: 完整复现实验环境。PyYAML 用于读取 ``configs/*.yaml``，属实际运行时依赖，因此同样记录。
RECORDED_DEPENDENCIES: tuple[tuple[str, str], ...] = (
    ("joblib", "joblib"),
    ("scikit-learn", "sklearn"),
    ("PyYAML", "yaml"),
)


def _module_version(module_name: str) -> str:
    """返回模块版本号；模块缺失或未声明 ``__version__`` 时返回明确标记而不是崩溃。"""
    try:
        module = importlib.import_module(module_name)
    except ImportError:  # pragma: no cover - 依赖缺失属异常环境
        return "unavailable"
    version = getattr(module, "__version__", None)
    return str(version) if version else "unknown"


def environment_versions() -> dict[str, str]:
    """采集 Python 与关键依赖版本，便于复现实验环境。"""
    versions = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }
    for field_name, module_name in RECORDED_DEPENDENCIES:
        versions[field_name] = _module_version(module_name)
    return versions


def jsonable(value: Any) -> Any:
    """把 numpy / pandas / dataclass 值递归转换为可被 ``json`` 序列化的对象。"""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def write_json_report(report: dict[str, Any], path: str | Path) -> Path:
    """写出 UTF-8 JSON 报告，必要时创建父目录。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(jsonable(report), handle, ensure_ascii=False, indent=2, sort_keys=False)
        handle.write("\n")
    return output_path
