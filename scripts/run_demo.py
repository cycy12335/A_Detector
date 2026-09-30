"""Start the CW-018 read-only Compose demo after fail-closed input validation.

The model bundle is deliberately supplied by a read-only host bind mount. This entrypoint
loads the bundle and policy before starting Uvicorn, so a missing, tampered, or incompatible
input exits with a stable safe error instead of starting a misleading demo.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ai_detector.service.app import create_app  # noqa: E402
from ai_detector.service.config import DEFAULT_CONFIG_PATH, ServiceConfig, ServiceConfigError  # noqa: E402
from ai_detector.service.runtime import load_policy_engine, load_runtime  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="启动 CW-018 只读 Docker 演示。")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(DEFAULT_CONFIG_PATH),
        help="演示配置路径，默认 configs/demo_service.yaml。",
    )
    return parser


def preflight(config_path: Path):
    """Parse and fully validate the model/policy before opening a listening socket."""
    config = ServiceConfig.from_yaml(config_path)
    runtime = load_runtime(config)
    policy = load_policy_engine(config)
    return config, runtime, policy


def _safe_error_code(error: BaseException) -> str:
    code = getattr(error, "code", None)
    if isinstance(code, str) and code:
        return code
    if isinstance(error, ServiceConfigError):
        return "config_invalid"
    return type(error).__name__.lower()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config, runtime, policy = preflight(args.config)
    except Exception as error:  # noqa: BLE001 - safe stable code only
        print(f"demo_preflight_failed:{_safe_error_code(error)}", file=sys.stderr)
        return 2

    import uvicorn

    app = create_app(
        config,
        runtime_loader=lambda _config: runtime,
        policy_loader=lambda _config: policy,
    )
    # The config remains loopback-only. Inside the isolated container we bind the app to
    # all container interfaces; Compose publishes only 127.0.0.1 on the host.
    uvicorn.run(app, host="0.0.0.0", port=config.port, access_log=False, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
