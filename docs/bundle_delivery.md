# CW-013 离线 bundle 交付与隔离复现

本文件记录**本地、离线、可验证**的模型产物搬运路径，以及把门禁搬到别的机器时
还缺什么。它刻意不做两件事：

- **不发布模型**：交付包只写在被 Git 忽略的目录里，不上传、不进版本控制；
- **不声称 hosted CI 已运行**：没有远端、没有流水线文件，隔离副本也只是
  "已提交源码的副本"。

---

## 1. 信任锚只有一个

导出的**来源**与导入的**验收标准**都取自 CW-009 门禁配置
`configs/quality_gate.yaml` 的 `quality_gate.inputs.bundle`（路径 + SHA-256）。
交付包里的 `manifest.json` 只是自描述，**不能**充当授权来源：

- 导出前先把真实 bundle 的流式 SHA-256 与配置里的冻结值比对，不符就**不导出**；
- 导入时把包里的实际字节摘要与**本地**配置的冻结值比对——即便有人把载荷改坏、
  同时把清单改成与新字节自洽，导入仍然失败（有专门的区分性测试守住这一点）。

因此"换一个来源不明的模型"既导不出、也导不进。

## 2. 三条命令

```powershell
# 1) 从本机已验收 bundle 导出交付包（校验冻结摘要后才写）
python scripts/export_bundle.py

# 2) 把交付包导入到门禁配置要求的目标路径（原子写入；已有同摘要目标则幂等成功）
python scripts/import_bundle.py

# 3) 隔离复现（不携带 artifacts/** 的源码副本）
python scripts/verify_isolated_bundle_gate.py          # 快：解析路径 + 缺 bundle 必须失败
python scripts/verify_isolated_bundle_gate.py --full   # 再加：导入后跑原样总门禁
```

交付包默认落在 `artifacts/delivery/cw013_lightgbm_bundle/`（载荷 + 清单），
报告落在 `artifacts/reports/cw013_delivery/`，两者都被 `.gitignore` 的 `artifacts/**` 排除。

**目录边界（硬约束，越界即失败关闭）**

| 选项 | 允许范围 | 越界时 |
| --- | --- | --- |
| `--package-dir` | 项目内被忽略的 `artifacts/delivery/` 子树 | `package_dir_not_ignored`，退出码 4；**任何写入之前**就拒绝 |
| `--stage` | 项目内被忽略的 `artifacts/tmp_tests/` 之下的专用子目录 | `stage_not_ignored`，退出码 2；**任何递归删除之前**就拒绝 |

`--package-dir` 之所以要限制：交付包里是模型二进制，**绝不能**写进会被提交的目录
（`configs/`、`src/` …）。`--stage` 之所以要限制：脚本会对 `<stage>/checkout`
执行递归删除，未校验的 `--stage`（尤其绝对路径）可能指向项目外。脚本另外只删除
**本轮新建的** `run-<uuid>/` 目录，并在结束时清理它。

退出码（导出/导入共用，稳定）：`0` 成功（新建，或已是**完整合法**且与冻结配置一致的包）；
`2` 信任摘要不符；`3` 交付包结构或清单不合格；`4` 路径不安全（含越出交付区）；
`5` 目标已存在且字节不同（拒绝覆盖）；`6` 写入或报告失败。上表中的
`--stage` 属于隔离验证命令，其退出码为 `0`（符合预期）或 `2`（预期不符/路径被拒绝）。

**`already_present` 的含义**：导出时若目录里已有内容，它必须先是**完整、合法、与冻结
配置一致**的包，导出才会返回 `already_present`。只有载荷在、清单缺失（或清单损坏、
有多余条目）都会失败关闭——否则"导出成功"会产出一个根本导入不了的包。
`exported` 与 `already_present` 都保证该包随后可被 `import_bundle` 成功导入。

`verify_isolated_bundle_gate.py`：`0` 全部符合预期；`2` 预期不符
（解析回指、缺失未失败、门禁未通过等）。

## 3. 搬到一个新环境的标准步骤

1. **制作**（有已验收 bundle 的机器）：`python scripts/export_bundle.py`；
   确认输出 `status=exported`、载荷大小 55,619 B、摘要 `f02bb50e…35a1c`。
2. **保管**：把 `artifacts/delivery/cw013_lightgbm_bundle/` 整个目录作为一个整体搬运
   （载荷 + 清单必须成对）。它不含密钥，但**仍是模型产物**：不要提交、不要上传到
   未经批准的位置。
3. **导入**（目标机器）：把该目录放到目标项目根的同一相对路径下，运行
   `python scripts/import_bundle.py`；它会把字节与**目标机器的**门禁配置冻结摘要比对
   后再落盘。
4. **验证**：`python scripts/run_project_gate.py`；静态 0 违规、回归 12/12、
   全量 pytest 通过即为可用。

## 4. 隔离复现是怎么做到"不回指真实仓库"的

CW-012 发现本项目以 **editable** 方式安装在 `.venv`：
`__editable__.ai_detector-0.1.0.pth` 注册的是 **MetaPathFinder**，优先级高于
`sys.path`，所以**只设 `PYTHONPATH` 不够**——`import ai_detector` 永远解析到真实仓库。

`scripts/verify_isolated_bundle_gate.py` 的做法是新建一个独立 venv，只放两个
**普通路径** `.pth`：

| `.pth` | 指向 | 作用 |
| --- | --- | --- |
| `_cw013_copy_src.pth` | `<副本>/src` | 项目代码从副本解析（`PROJECT_ROOT` 因此等于副本根） |
| `_cw013_host_deps.pth` | 宿主 venv 的 `site-packages` | 第三方依赖可导入 |
| （另复制 `ruff.exe`） | 宿主 venv 的 `Scripts/` | `python -m ruff` 的包装器按 `sys.executable` 旁找二进制 |

关键在于 `site` 只为**当前解释器的 site 目录**处理 `.pth`，被当作普通路径加进来的
目录不在此列——因此宿主的 editable finder **不会注册**，不可能回指真实仓库。
脚本会打印 `ai_detector.__file__` 与 `PROJECT_ROOT` 的实际解析路径并断言都在副本内。

**限制（如实记录）**：隔离的是**代码解析**，不是**依赖**——第三方包复用宿主 venv 里
已安装的同一版本，没有重新下载或锁文件校验。要连依赖一起隔离，需要另一个包管理决定。

## 5. 隔离验证实测结果

| 阶段 | 结果 |
| --- | --- |
| 模块解析 | `ai_detector` → `<副本>/src/...`；`PROJECT_ROOT` = 副本根；不在真实 `src` 下 |
| 缺 bundle 反例 | 总门禁 **exit 5**；静态如实 passed；质量门禁 failed；`bundle` 输入 `false`；`input_missing` |
| 导入 | **exit 0 / status=imported**，目标写入后按冻结摘要复核 |
| 导入后原样总门禁 | **exit 0**：静态 0 违规；回归 12/12、偏差 0.0、确定性 true；全量 pytest **1854 passed / 22 skipped / 0 failed** |
| 真实工作目录 | 冻结文件与默认库逐字节未变 |

副本里的 22 个 skip **全部**来自 `tests/integration/test_cw013_isolated_copy.py`：
该模块在隔离副本内部**结构性跳过**（`python -m venv` 无法从 venv 内部再建 venv，
嵌套验证既无意义也不可行）。真实仓库里的默认运行不受影响——专项 66 项全部真实运行
（44 交付 + 22 隔离），全量 pytest 的 skip 仍是 3 个（与 CW-012 基线相同）。

## 6. 全量套件还需要两个被忽略的产物（CW-013 的新发现）

CW-012 的输入审计说"bundle 是唯一被忽略的**门禁**输入"——这对 **preflight** 成立。
隔离验证发现**全量 pytest** 还需要两个被 `.gitignore` 排除的本地产物：

| 产物 | 谁需要它 | 缺失时 |
| --- | --- | --- |
| `artifacts/models/cw003/rule_baseline_state.json` | `tests/unit/test_cw006_policy_config.py` 核对政策阈值与 CW-003 状态一致 | 该测试失败（`FileNotFoundError`） |
| `artifacts/runtime/cw007_decisions.sqlite3`（默认库） | `tests/integration/test_cw009_quality_gate.py` 断言它逐字节未变 | 该测试失败 |

在真实工作目录里它们**存在**，所以本地跑得通；干净检出里没有，于是首次隔离运行
实测有 3 项失败（另一个是副本漏了已提交的 `.gitignore`，已修正）。
本次隔离验证**直接复制**这两个产物来跑完整套件，并把它们的**可信供给方式列为未解决**——
离线交付包只承载 bundle，不承载它们。

## 7. hosted CI 仍缺什么（未解决）

| 项目 | 状态 |
| --- | --- |
| 离线导出/导入 + 冻结摘要校验 | ✅ 本地完成 |
| 隔离源码副本复现（缺 bundle 失败 → 导入后总门禁通过） | ✅ 本地完成 |
| 代码托管/执行平台 | ❌ 未确定（无 `.git`、无组织/仓库决定、无 runner 决定） |
| 交付包的分发与保管 | ❌ 未确定（不许上传模型；需要制品存储、访问控制与保留策略） |
| 另外两个被忽略产物的供给 | ❌ 未确定（见第 6 节） |
| 依赖隔离/锁文件 | ❌ 未确定（当前复用宿主 venv 的同版本依赖） |
| 在本平台真实运行过一次的流水线 | ❌ 尚不存在 |

因此 **P3 的 CI 项仍未完成**。本包交付的是"本地可信搬运 + 隔离复现"，
不是一条跑过的流水线。
