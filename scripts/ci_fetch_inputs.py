"""CW-014 CI 输入供给：把干净检出缺少的被忽略输入按**冻结摘要**校验后放到位。

用法（项目根目录下执行）::

    python scripts/ci_fetch_inputs.py

它读取两处配置：

- ``configs/ci_inputs.yaml``：三个输入的路径 / 摘要 / 供给方式；
- ``configs/quality_gate.yaml``：**bundle 的授权来源**。清单里的 bundle 摘要必须与它
  逐项一致，否则以退出码 3 失败关闭——清单是转述，不是第二授权来源。

每个输入的处理：

============================  ====================================================
``supply``                    行为
============================  ====================================================
``release_asset``             文件必须已由工作流下载到目标路径；校验摘要。
``committed_fixture``         校验夹具摘要，然后**原子**放到目标路径（已一致则跳过）。
``not_required``              不需要该文件；若它意外存在，仍校验摘要，不符则失败关闭。
============================  ====================================================

退出码（稳定）：0=全部就绪；2=缺失或摘要不符（**失败关闭**）；3=清单与门禁配置不一致；
6=写入失败。任何情况下都不会把字节不符的文件放到目标路径。

**安全**：只读写项目相对路径；不打印文件内容；报告只含路径、大小、摘要与状态。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_gate.bundle_delivery import stream_digest  # noqa: E402

CI_INPUTS_REL = "configs/ci_inputs.yaml"
QUALITY_GATE_REL = "configs/quality_gate.yaml"
REPORT_REL = "artifacts/reports/cw014_ci/inputs_report.json"

EXIT_OK = 0
EXIT_MISSING_OR_MISMATCH = 2
EXIT_MANIFEST = 3
EXIT_WRITE = 6

SUPPLY_RELEASE = "release_asset"
SUPPLY_FIXTURE = "committed_fixture"
SUPPLY_NOT_REQUIRED = "not_required"
KNOWN_SUPPLY = frozenset({SUPPLY_RELEASE, SUPPLY_FIXTURE, SUPPLY_NOT_REQUIRED})


class InputError(RuntimeError):
    """带稳定退出码的输入供给失败。"""

    def __init__(self, code: str, detail: str, *, exit_code: int) -> None:
        self.code = str(code)
        self.detail = str(detail)
        self.exit_code = int(exit_code)
        super().__init__(f"{self.code}: {self.detail}")


@dataclass(frozen=True)
class Spec:
    """一个被忽略输入的期望。"""

    label: str
    path: str
    sha256: str
    size: int
    supply: str
    fixture: str | None


def load_specs(root: Path = PROJECT_ROOT) -> list[Spec]:
    """读清单，并交叉核对 bundle 与门禁配置一致。"""
    try:
        raw = yaml.safe_load((root / CI_INPUTS_REL).read_text(encoding="utf-8"))["ci_inputs"]
        gate = yaml.safe_load((root / QUALITY_GATE_REL).read_text(encoding="utf-8"))["quality_gate"]
        entries = raw["inputs"]
    except (OSError, UnicodeError, yaml.YAMLError, KeyError, TypeError) as error:
        raise InputError(
            "config_unreadable", f"无法读取清单或门禁配置：{type(error).__name__}",
            exit_code=EXIT_MANIFEST,
        ) from error

    specs: list[Spec] = []
    for label, entry in entries.items():
        try:
            spec = Spec(
                label=str(label),
                path=str(entry["path"]),
                sha256=str(entry["sha256"]),
                size=int(entry["size"]),
                supply=str(entry["supply"]),
                fixture=entry.get("fixture"),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise InputError(
                "entry_invalid", f"{label} 的清单条目形状非法", exit_code=EXIT_MANIFEST
            ) from error
        if spec.supply not in KNOWN_SUPPLY:
            raise InputError(
                "supply_unknown", f"{label} 的 supply={spec.supply} 未知", exit_code=EXIT_MANIFEST
            )
        if len(spec.sha256) != 64:
            raise InputError(
                "digest_invalid", f"{label} 的 sha256 形状非法", exit_code=EXIT_MANIFEST
            )
        specs.append(spec)

    declared = {spec.label: spec for spec in specs}
    bundle = declared.get("bundle")
    if bundle is None:
        raise InputError("bundle_absent", "清单必须声明 bundle", exit_code=EXIT_MANIFEST)
    authorized = gate["inputs"]["bundle"]
    for field, expected, actual in (
        ("path", authorized["path"], bundle.path),
        ("sha256", authorized["sha256"], bundle.sha256),
    ):
        if expected != actual:
            raise InputError(
                "bundle_not_authoritative",
                f"清单里的 bundle.{field} 与门禁配置不一致",
                exit_code=EXIT_MANIFEST,
            )
    return specs


def _verify_existing(path: Path, spec: Spec) -> dict:
    """校验一个已存在的文件；返回安全的状态字典。"""
    if not path.is_file():
        raise InputError("input_missing", f"{spec.path} 不存在", exit_code=EXIT_MISSING_OR_MISMATCH)
    digest, size = stream_digest(path)
    if digest != spec.sha256 or size != spec.size:
        raise InputError(
            "input_digest_mismatch", f"{spec.path} 的字节与冻结摘要不符",
            exit_code=EXIT_MISSING_OR_MISMATCH,
        )
    return {"status": "verified", "size": size, "sha256": digest}


def _place_fixture(root: Path, path: Path, fixture: str, spec: Spec) -> dict:
    """把已审查夹具原子放到目标路径（已一致则跳过，不重写）。"""
    source = root / fixture
    result = _verify_existing(source, spec)
    if path.is_file():
        digest, _ = stream_digest(path)
        if digest == spec.sha256:
            return {**result, "status": "already_present"}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, path)
    except OSError as error:
        raise InputError(
            "write_failed", f"写入 {spec.path} 失败：{type(error).__name__}", exit_code=EXIT_WRITE
        ) from error
    finally:
        temporary.unlink(missing_ok=True)
    written, size = stream_digest(path)
    if written != spec.sha256 or size != spec.size:  # pragma: no cover - 防御性
        raise InputError(
            "write_verify_failed", f"{spec.path} 写入后复核不符", exit_code=EXIT_WRITE
        )
    return {**result, "status": "placed"}


def provide_inputs(root: Path = PROJECT_ROOT) -> dict:
    """按清单供给全部输入；任一失败即抛出（失败关闭）。"""
    specs = load_specs(root)
    report: dict = {"work_package": "CW-014", "status": "ok", "inputs": {}}
    for spec in specs:
        path = root / spec.path
        if spec.supply == SUPPLY_FIXTURE:
            assert spec.fixture is not None
            outcome = _place_fixture(root, path, spec.fixture, spec)
        elif spec.supply == SUPPLY_RELEASE:
            outcome = _verify_existing(path, spec)
        else:  # not_required：不需要，但存在就必须是对的
            if path.is_file():
                outcome = _verify_existing(path, spec)
            else:
                outcome = {"status": "not_required"}
        report["inputs"][spec.label] = {
            "path": spec.path,
            "supply": spec.supply,
            **outcome,
        }
    return report


def _write_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ci_fetch_inputs.py",
        description="按冻结摘要校验并放置干净检出缺少的 CI 输入。",
    )
    parser.add_argument("--report", default=REPORT_REL, help="报告的项目相对路径")
    args = parser.parse_args(argv)
    try:
        report = provide_inputs()
    except InputError as error:
        report = {
            "work_package": "CW-014",
            "status": "failed",
            "error_code": error.code,
        }
        print(f"inputs      failed  error={error.code}")
        try:
            _write_report(report, Path(args.report))
        except OSError:
            return EXIT_WRITE
        return error.exit_code
    for label, entry in report["inputs"].items():
        print(f"  {label:22s} {entry['supply']:18s} {entry['status']}")
    print(f"inputs      ok  ({len(report['inputs'])} 项)")
    try:
        _write_report(report, Path(args.report))
    except OSError:
        print("报告写入失败（report_write_failed）")
        return EXIT_WRITE
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
