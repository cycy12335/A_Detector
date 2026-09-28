"""CW-005 只读风险评分服务。

**导入本包没有任何副作用**：不读取配置文件、不加载 bundle、不启动服务器。
这些动作分别发生在显式调用 :func:`ai_detector.service.config.ServiceConfig.from_yaml`
与 :func:`ai_detector.service.app.create_app` 时，模型则在应用的 lifespan 启动阶段
加载一次。

对外真正的入口只有两个：

- :func:`create_app`：构造 FastAPI 应用（可注入 fake runtime 供测试使用）；
- :class:`ServiceConfig`：解析并校验服务配置（含路径安全约束）。
"""

from __future__ import annotations

from ai_detector.service.app import create_app
from ai_detector.service.config import ServiceConfig, ServiceConfigError
from ai_detector.service.errors import ApiError, BundleLoadError
from ai_detector.service.runtime import BundleRuntime, ScoringRuntime, load_runtime
from ai_detector.service.schemas import SCORE_SEMANTICS, THRESHOLD_POLICY

__all__ = [
    "ApiError",
    "BundleLoadError",
    "BundleRuntime",
    "SCORE_SEMANTICS",
    "ServiceConfig",
    "ServiceConfigError",
    "ScoringRuntime",
    "THRESHOLD_POLICY",
    "create_app",
    "load_runtime",
]
