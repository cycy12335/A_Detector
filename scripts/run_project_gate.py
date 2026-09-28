"""CW-012 本地总门禁入口。

用法（项目根目录下执行）::

    python scripts/run_project_gate.py

它按固定顺序运行两个阶段：

1. **static**：用冻结规则集（``E9,F63,F7,F82,F811,F401``）检查
   ``src/ai_detector``、``scripts``、``performance``、``project_gate``、``tests``
   与仓库根 ``conftest.py``；
2. **quality_gate**：原样执行 CW-009 的默认命令
   ``python scripts/run_quality_gate.py --config configs/quality_gate.yaml``，
   即完整的 preflight → regression → pytest 链路（全量 pytest 只跑这一次）。

退出码（稳定）：0=全部通过；2=静态违规；3=静态工具失败；4=静态超时；
5=CW-009 门禁失败；6=报告写入失败。

报告写 ``artifacts/reports/cw012_project_gate/report.json``（继续被 Git 忽略），
只含阶段、命令形状、退出码、耗时、违规计数、CW-009 状态/计数与工具版本；
不含完整 stdout/stderr、测试节点 ID、请求/特征值、密钥、绝对路径、
bundle 内容或数据库行。
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_gate.gate import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
