"""CW-012 CI 输入清单：干净检出运行总门禁到底需要什么、从哪来、怎么验、缺了会怎样。

清单是**数据**，不是散落在文档里的散文：``docs/ci_readiness.md`` 由它渲染，
``tests/integration/test_cw012_ci_inputs.py`` 断言它与 CW-009 门禁配置逐项一致，
因此新增一个必填输入却忘了登记会被测试抓住。

## 关键事实

当前目录**不是 Git 仓库**（没有 ``.git``），``.gitignore`` 又把 ``artifacts/**/*``
整体排除，而 CW-009 默认门禁**必须**有已验收的
``artifacts/models/cw004/lightgbm_bundle.joblib``。所以：**从干净检出直接运行总门禁
一定会在 preflight 阶段失败关闭**——这不是缺陷，而是本包要如实记录并验证的缺口。
"""

from __future__ import annotations

from dataclasses import dataclass

#: 输入分类（稳定字符串）。
COMMITTED = "committed"
IGNORED = "ignored"

#: 缺失时的行为（稳定字符串）。
FAIL_CLOSED = "fail_closed"


@dataclass(frozen=True)
class CiInput:
    """一个 CI 运行所需的输入。

    Args:
        label: 与 CW-009 门禁配置 ``inputs`` 段对齐的名字（非门禁输入则是描述性名字）。
        path: 项目相对路径。
        kind: :data:`COMMITTED`（随代码走）或 :data:`IGNORED`（被 ``.gitignore`` 排除）。
        trust: 可信来源——谁有权产生它、如何避免"来源不明的替代品"。
        verification: 校验方式。
        missing_behaviour: 缺失时总门禁的行为。
        needed_by: 哪个阶段需要它。
    """

    label: str
    path: str
    kind: str
    trust: str
    verification: str
    missing_behaviour: str
    needed_by: str

    @property
    def committed(self) -> bool:
        return self.kind == COMMITTED


#: CW-009 门禁配置里冻结的四个输入（label 与 ``inputs`` 段的键一致）。
GATE_INPUTS: tuple[CiInput, ...] = (
    CiInput(
        label="service_config",
        path="configs/service.yaml",
        kind=COMMITTED,
        trust="随代码提交、经评审；由项目负责人维护。",
        verification="配置内冻结 SHA-256 `4dd211557214cee85abf1f427f72bfcec3510f6a4d01a928becb141fffb1c865`；CW-009 预检按字节比对。",
        missing_behaviour=FAIL_CLOSED,
        needed_by="quality_gate（preflight）",
    ),
    CiInput(
        label="bundle",
        path="artifacts/models/cw004/lightgbm_bundle.joblib",
        kind=IGNORED,
        trust=(
            "**当前没有可信供给方式**。只能由本仓库 CW-004 训练入口在本机产出；"
            "不得下载、不得用第三方或来源不明的模型替代。"
        ),
        verification="配置内冻结 SHA-256 `f02bb50e61c5f70c85448430c8bb653bd6e347f400220771500ee84a70135a1c`（55,619 B）；预检按字节比对。",
        missing_behaviour=FAIL_CLOSED,
        needed_by="quality_gate（preflight → regression → pytest）",
    ),
    CiInput(
        label="policy",
        path="configs/risk_policy.yaml",
        kind=COMMITTED,
        trust="随代码提交、经评审；由项目负责人维护。",
        verification="配置内冻结 SHA-256 `7ed07f0b784715c01b444035bb34ea5166d1b376ff07e10428561d898d6824f0`；预检按字节比对。",
        missing_behaviour=FAIL_CLOSED,
        needed_by="quality_gate（preflight）",
    ),
    CiInput(
        label="corpus",
        path="tests/fixtures/cw009/model_regression_cases.json",
        kind=COMMITTED,
        trust="随代码提交的固定回归语料（12 个纯合成 case），是 golden master。",
        verification="配置内冻结 SHA-256 `d58166b70223af0adb443b9acff0690e6e3c571110dc11b16c2395a8d6fdc4b0`；预检按字节比对。",
        missing_behaviour=FAIL_CLOSED,
        needed_by="quality_gate（preflight → regression）",
    ),
)

#: 门禁之外、由本包引入的输入与运行前提。
SUPPORTING_INPUTS: tuple[CiInput, ...] = (
    CiInput(
        label="source_code",
        path="src/ai_detector",
        kind=COMMITTED,
        trust="随代码提交。",
        verification="静态检查（冻结规则集）+ 全量 pytest。",
        missing_behaviour=FAIL_CLOSED,
        needed_by="static、quality_gate",
    ),
    CiInput(
        label="gate_sources",
        path="project_gate",
        kind=COMMITTED,
        trust="随代码提交（本包新增）。",
        verification="静态检查 + CW-012 专项测试。",
        missing_behaviour=FAIL_CLOSED,
        needed_by="static、本入口自身",
    ),
    CiInput(
        label="project_config",
        path="pyproject.toml",
        kind=COMMITTED,
        trust="随代码提交；冻结静态规则集与依赖分组。",
        verification="静态规则集与代码常量一致性测试；ruff 版本用 == 锁定。",
        missing_behaviour=FAIL_CLOSED,
        needed_by="static（依赖与规则）、导入路径",
    ),
    CiInput(
        label="quality_extra",
        path="（pip extra：static、service、tree、test）",
        kind=COMMITTED,
        trust="公开 PyPI 的二进制 wheel；ruff 用 == 精确锁版本。",
        verification="`pip check`；`python -m ruff --version`；缺失时静态阶段失败关闭。",
        missing_behaviour=FAIL_CLOSED,
        needed_by="static、quality_gate",
    ),
)

#: 完整清单。
CI_INPUTS: tuple[CiInput, ...] = GATE_INPUTS + SUPPORTING_INPUTS


def by_label(label: str) -> CiInput:
    for entry in CI_INPUTS:
        if entry.label == label:
            return entry
    raise KeyError(label)


def ignored_inputs() -> tuple[CiInput, ...]:
    """被 ``.gitignore`` 排除、因此**不会**自动出现在干净检出里的输入。"""
    return tuple(entry for entry in CI_INPUTS if not entry.committed)
