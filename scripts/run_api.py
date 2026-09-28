"""启动只读评分服务的本地 Uvicorn 进程。

用法（项目根目录下执行）::

    python scripts/run_api.py --config configs/service.yaml

设计要点：

- 默认只监听 ``127.0.0.1:8000``（host/port 来自配置，且配置层只接受回环地址）。
  本服务没有认证、限流与 TLS，**不得直接暴露到公网**。
- 配置里的相对路径永远相对项目根目录解析，因此从其它工作目录调用时给绝对配置路径即可。
- 不做任何训练、不访问 IEEE-CIS 原始 CSV、不重新评估 test。

停止：在终端按 Ctrl+C。本脚本不会自行 fork 或把服务留在后台。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 让脚本在未执行 `pip install -e .` 时也能从源码树直接运行。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ai_detector.service.app import create_app  # noqa: E402
from ai_detector.service.config import DEFAULT_CONFIG_PATH, ServiceConfig  # noqa: E402
from ai_detector.service.config import ServiceConfigError  # noqa: E402
from ai_detector.service.logging import configure_logging  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="启动只读风险评分服务（默认仅监听 127.0.0.1）。"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(DEFAULT_CONFIG_PATH),
        help=f"服务配置路径（相对项目根目录），默认 {DEFAULT_CONFIG_PATH}",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["critical", "error", "warning", "info", "debug"],
        help="uvicorn 日志级别，默认 info",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        config = ServiceConfig.from_yaml(args.config)
    except ServiceConfigError as error:
        print(f"服务配置加载失败：{error}", file=sys.stderr)
        return 2

    configure_logging()
    app = create_app(config)

    import uvicorn

    print(
        f"启动只读评分服务 {config.name}（{config.api_version}）；"
        f"监听 http://{config.host}:{config.port}（仅本机）。"
    )
    print("提示：模型分数不是概率，flagged 不是生产处置；本服务无认证/限流/TLS，不得暴露公网。")
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level=args.log_level,
        # **必须显式关闭 Uvicorn 的默认 access log**：它会原样记录请求行，
        # 包括未知 path 与 query string（例如 /SECRET?token=...），
        # 从而绕过本服务 completion log 的模板路径 / <unmatched> 脱敏。
        # 关闭后每个请求只剩我们自己的白名单结构化日志。
        access_log=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
