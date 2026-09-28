"""受控本机 Locust 运行与安全报告；不接受远程目标。"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from urllib.parse import urlparse

import httpx
import yaml

from performance.analysis import MeasurementError, analyze_events

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "artifacts/reports/cw011_performance"
RUNTIME_DIR = ROOT / "artifacts/runtime"
DEFAULT_DB = RUNTIME_DIR / "cw007_decisions.sqlite3"
SERVICE_CONFIG = ROOT / "configs/service.yaml"


@dataclass(frozen=True)
class Profile:
    name: str
    users: int
    spawn_rate: int
    warmup_s: int
    measure_s: int
    minimum_per_endpoint: int


PROFILES = (
    Profile("low", 1, 1, 5, 25, 30),
    Profile("concurrent", 4, 2, 8, 30, 40),
)
SMOKE_PROFILE = Profile("smoke", 1, 1, 1, 5, 1)


class RunError(RuntimeError):
    """安全、稳定的测量失败分类。"""

    def __init__(self, code: str, profile_result: dict | None = None):
        self.code = code
        self.profile_result = profile_result
        super().__init__(code)


def validate_target(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or parsed.port is None:
        raise RunError("non_loopback_target")
    if parsed.username or parsed.password or parsed.path not in ("", "/") or parsed.query or parsed.fragment:
        raise RunError("non_loopback_target")


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def stop_process(process: subprocess.Popen[bytes] | None) -> None:
    """成功、失败和超时路径统一回收；不向控制台转发子进程输出。"""
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=8)


def _spawn(command: list[str], *, env: dict[str, str] | None = None) -> subprocess.Popen[bytes]:
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    return subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=flags,
    )


def _wait_ready(target: str, process: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + 25
    with httpx.Client(timeout=1.0, trust_env=False) as client:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RunError("service_start_failed")
            try:
                response = client.get(target + "/health/ready")
                if response.status_code == 200:
                    return
                if response.status_code == 503:
                    raise RunError("service_not_ready")
            except httpx.TransportError:
                pass
            time.sleep(0.2)
    raise RunError("service_start_timeout")


def _make_config(port: int, database_name: str, config_path: Path) -> None:
    data = yaml.safe_load(SERVICE_CONFIG.read_text(encoding="utf-8"))
    data["service"]["host"] = "127.0.0.1"
    data["service"]["port"] = port
    data["storage"]["path"] = f"artifacts/runtime/{database_name}"
    config_path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _versions() -> dict[str, str]:
    result = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("locust", "fastapi", "uvicorn", "lightgbm"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = "missing"
    return result


#: 新鲜度判定对文件系统时间戳精度的容差（秒）。只用于抵消时间戳取整，
#: 不足以让上一轮的样本蒙混过关——陈旧样本至少是分钟级的。
_MTIME_TOLERANCE_S = 1.0


def _round_events_path(events_path: Path) -> Path:
    """本轮**独占**的事件文件路径。

    每一轮都用新名字，因此上一轮的样本在物理上不可能出现在这个路径上：
    "本轮统计只来自本轮样本"不是靠比较时间戳保证的，而是靠路径从一开始就不同。
    """
    return events_path.with_name(f".{events_path.name}.{uuid.uuid4().hex}.round")


def _fresh_events(path: Path, *, round_started: float) -> None:
    """本轮事件的新鲜度校验：必须存在、非空，且修改时间不早于本轮开始。

    这是隔离之外的纵深防御：即便有人把文件放回本轮路径（例如从备份还原），
    陈旧的时间戳也会让它被拒绝。

    Raises:
        RunError: ``no_new_events``（缺失或空）或 ``stale_events``（时间戳早于本轮）。
    """
    if not path.is_file():
        raise RunError("no_new_events")
    stat = path.stat()
    if stat.st_size == 0:
        raise RunError("no_new_events")
    if stat.st_mtime < round_started - _MTIME_TOLERANCE_S:
        raise RunError("stale_events")


def _discard_round_events(path: Path) -> None:
    """尽力删除本轮临时事件文件，**绝不**掩盖原始失败分类。

    进程已在调用前回收，正常情况下文件句柄已经释放；这里对 Windows 上偶发的短暂
    占用做有限重试。若仍失败则放弃——把一次 ``locust_timeout`` 变成
    ``PermissionError`` 比留下一个临时文件更糟。
    """
    for attempt in range(5):
        try:
            path.unlink(missing_ok=True)
            return
        except OSError:
            if attempt == 4:
                return
            time.sleep(0.05)


def _execute_locust(profile: Profile, target: str, round_path: Path) -> tuple[int, float]:
    """启动无界面 Locust、等待结束并回收进程，返回 ``(退出码, 墙钟耗时)``。"""
    env = os.environ.copy()
    env.update({
        "CW011_EVENTS_PATH": str(round_path),
        "CW011_WARMUP_SECONDS": str(profile.warmup_s),
        "CW011_MEASURE_SECONDS": str(profile.measure_s),
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
    })
    command = [
        sys.executable, "-X", "utf8", "-m", "locust", "-f", "performance/locustfile.py",
        "--headless", "--host", target,
        "--users", str(profile.users), "--spawn-rate", str(profile.spawn_rate),
        "--run-time", f"{profile.warmup_s + profile.measure_s}s",
        "--only-summary", "--exit-code-on-error", "3",
    ]
    process = None
    started = time.monotonic()
    try:
        process = _spawn(command, env=env)
        try:
            exit_code = process.wait(timeout=profile.warmup_s + profile.measure_s + 30)
        except subprocess.TimeoutExpired as error:
            raise RunError("locust_timeout") from error
    except OSError as error:
        raise RunError("locust_spawn_failed") from error
    finally:
        stop_process(process)
    return exit_code, time.monotonic() - started


def _run_profile(profile: Profile, target: str, events_path: Path) -> dict:
    validate_target(target)
    events_path.parent.mkdir(parents=True, exist_ok=True)
    # 本轮隔离：把事件写到独占文件，绝不读写 `events_path` 上可能存在的旧样本。
    round_path = _round_events_path(events_path)
    round_path.unlink(missing_ok=True)
    round_started = time.time()
    published = False
    try:
        exit_code, duration = _execute_locust(profile, target, round_path)
        try:
            _fresh_events(round_path, round_started=round_started)
        except RunError as error:
            # 本轮没有可用的新样本时**绝不回退**到任何既有事件文件：
            # 非零退出按退出码归类，退出 0 则是"没写出样本"这一独立失败类别。
            if exit_code != 0:
                raise RunError("locust_failed") from error
            raise
        try:
            stats = analyze_events(
                round_path, seconds=profile.measure_s, minimum=profile.minimum_per_endpoint
            )
        except MeasurementError as error:
            raise RunError("invalid_measurement") from error
        result = {
            "profile": asdict(profile),
            "locust_exit_code": exit_code,
            "elapsed_wall_s": duration,
            "events_file": f"{profile.name}_events.jsonl",
            "stats": {name: entry.as_dict() for name, entry in stats.items()},
        }
        if exit_code != 0 or any(entry.failures for entry in stats.values()):
            raise RunError("request_failures", result)
        # 只有完整成功的这一轮才发布到正式文件名：`*_events.jsonl` 因此始终
        # 与最近一次成功测量逐字段对应，失败不会破坏上一条可追溯的原始事件。
        os.replace(round_path, events_path)
        published = True
        return result
    finally:
        # **单一**清理点：spawn 失败、等待超时、新鲜度不足、统计失败与请求失败
        # 全都经过这里。清理若只挂在后续统计段，超时路径就会遗留已经写入的
        # 本轮临时事件——那正是首次整改复验的阻断项。
        if not published:
            _discard_round_events(round_path)


def _write_report(report: dict, prefix: str) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_DIR / f"{prefix}.json"
    md_path = REPORT_DIR / f"{prefix}.md"
    temporary = REPORT_DIR / f".{prefix}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, json_path)
    finally:
        temporary.unlink(missing_ok=True)
    lines = [
        f"# CW-011 本机性能报告（{report['status']}）",
        "",
        f"- 时间（UTC）：{report['created_at_utc']}",
        f"- 错误类别：{report['error_code'] or '无'}",
        f"- 命令：`{report['command_shape']}`",
        f"- 环境：{', '.join(f'{name}={value}' for name, value in report['environment'].items())}",
        f"- 服务 API 版本：{report['service_api_version']}；模型版本：{report['model_version']}",
        f"- 模型指纹：`{report['inputs']['bundle_sha256']}`；政策指纹：`{report['inputs']['policy_sha256']}`",
        "- 请求权重：score 4 / risk 3 / decision 3；请求体为固定合成特征，decision 不使用幂等键。",
        "- 方法：Locust 无界面模式；预热样本未计入统计；分位数使用最近秩算法。",
        "- 限制：本机观测值受硬件与运行条件影响，不代表生产 SLO、容量或真实交易负载。",
        "",
        "| 档位 | 用户数 | 爬升率/秒 | 预热秒 | 测量秒 | 每接口最小样本 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for entry in report["profiles"]:
        profile = entry["profile"]
        lines.append(
            f"| {profile['name']} | {profile['users']} | {profile['spawn_rate']} | "
            f"{profile['warmup_s']} | {profile['measure_s']} | {profile['minimum_per_endpoint']} |"
        )
    lines.extend([
        "",
        "| 档位 | 接口 | 请求 | 失败 | 失败率 | RPS | P50 ms | P95 ms | P99 ms |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for profile in report["profiles"]:
        for name, stats in profile["stats"].items():
            lines.append(
                f"| {profile['profile']['name']} | {name} | {stats['requests']} | "
                f"{stats['failures']} | {stats['failure_rate']:.2%} | {stats['rps']:.2f} | {stats['p50_ms']:.2f} | "
                f"{stats['p95_ms']:.2f} | {stats['p99_ms']:.2f} |"
            )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(*, smoke: bool = False) -> int:
    """运行固定本机档位，始终写有限报告并清理临时资源。"""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    label = "smoke_report" if smoke else "report"
    token = uuid.uuid4().hex
    db_name = f"cw011_perf_{token}.sqlite3"
    database = RUNTIME_DIR / db_name
    config = REPORT_DIR / f".service_{token}.yaml"
    baseline_db_hash = _digest(DEFAULT_DB)
    service_data = yaml.safe_load(SERVICE_CONFIG.read_text(encoding="utf-8"))["service"]
    report = {
        "work_package": "CW-011",
        "schema_version": "cw011_performance_report_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "failed",
        "error_code": None,
        "mode": "smoke" if smoke else "formal",
        "command_shape": "python scripts/run_performance.py [--smoke]",
        "environment": _versions(),
        "service_api_version": service_data["api_version"],
        "model_version": service_data["model_version"],
        "inputs": {
            "service_sha256": _digest(SERVICE_CONFIG),
            "policy_sha256": _digest(ROOT / "configs/risk_policy.yaml"),
            "bundle_sha256": _digest(ROOT / "artifacts/models/cw004/lightgbm_bundle.joblib"),
            "corpus_sha256": _digest(ROOT / "tests/fixtures/cw009/model_regression_cases.json"),
        },
        "request_weights": {"score": 4, "risk": 3, "decision": 3},
        "profiles": [],
        "limitations": "本机合成流量；不是生产 SLO、容量承诺或真实交易表现。",
    }
    service = None
    try:
        port = _port()
        target = f"http://127.0.0.1:{port}"
        validate_target(target)
        _make_config(port, db_name, config)
        service = _spawn([sys.executable, "scripts/run_api.py", "--config", str(config)])
        _wait_ready(target, service)
        for profile in (SMOKE_PROFILE,) if smoke else PROFILES:
            event_path = REPORT_DIR / f"{profile.name}_events.jsonl"
            outcome = _run_profile(profile, target, event_path)
            report["profiles"].append(outcome)
        report["status"] = "passed"
    except RunError as error:
        report["error_code"] = error.code
        if error.profile_result is not None:
            report["profiles"].append(error.profile_result)
    except (OSError, ValueError, yaml.YAMLError) as error:
        report["error_code"] = "runner_failed"
    except Exception:
        report["error_code"] = "runner_internal_error"
    finally:
        stop_process(service)
        config.unlink(missing_ok=True)
        for suffix in ("", "-wal", "-shm"):
            database.with_name(database.name + suffix).unlink(missing_ok=True)
        if _digest(DEFAULT_DB) != baseline_db_hash:
            report["status"] = "failed"
            report["error_code"] = "default_database_changed"
        _write_report(report, label)
    return 0 if report["status"] == "passed" else 2
