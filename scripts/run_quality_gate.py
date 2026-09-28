"""CW-009 一键质量门禁入口。

用法（项目根目录下执行）::

    python scripts/run_quality_gate.py --config configs/quality_gate.yaml

它依次执行：

1. **preflight**：以二进制方式核对 bundle、policy、service config、corpus 四个输入的
   SHA-256，任一不符就**不加载模型、不跑 pytest**，以退出码 2 失败关闭；
2. **regression**：用生产顺序（规则求值 → 模型评分 → 融合）对 12 个纯合成 case 求实际
   结果，与语料冻结的期望逐值比较（score 用配置的绝对容差，其余字段精确相等），并在同一
   进程里再跑一遍做确定性核对；
3. **pytest**：用当前解释器 ``sys.executable -m pytest`` 运行**完整**测试套件。

退出码（冻结）：0=全部通过；2=配置/预检/语料契约失败；3=回归或确定性失败；
4=pytest 非零（含无法启动）；5=pytest 超时；6=报告写入失败。

报告只写 ``artifacts/reports/cw009_quality_gate/report.json``（继续被 Git 忽略），
且不含任何 feature 取值、请求体、数据库内容、UUID、绝对路径或 pytest 输出。

诊断用 ``--regression-only`` 仍然执行完整的 preflight + regression + report，只是不运行
pytest；它是唯一的一个降级开关，**没有**跳过哈希、放宽容差或更新 oracle 的参数。
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ai_detector.quality.gate import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
