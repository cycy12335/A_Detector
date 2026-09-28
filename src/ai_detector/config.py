"""配置加载与路径解析。

所有相对路径都相对项目根目录解析，因此命令从项目根目录执行时不依赖当前
工作目录，也不会写入任何个人绝对路径。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

#: ``src/ai_detector/config.py`` -> 上溯三级即项目根目录。
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ConfigError(ValueError):
    """配置文件缺失或字段不合法时抛出。"""


def resolve_path(path: str | Path) -> Path:
    """把相对路径解析为相对项目根目录的绝对路径。"""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate)


@dataclass(frozen=True)
class SplitConfig:
    """时间切分比例。"""

    train_ratio: float = 0.6
    validation_ratio: float = 0.2

    def __post_init__(self) -> None:
        if not 0.0 < self.train_ratio < 1.0:
            raise ConfigError(f"train_ratio 必须位于 (0, 1)，实际为 {self.train_ratio}")
        if not 0.0 < self.validation_ratio < 1.0:
            raise ConfigError(
                f"validation_ratio 必须位于 (0, 1)，实际为 {self.validation_ratio}"
            )
        if self.train_ratio + self.validation_ratio >= 1.0:
            raise ConfigError("train_ratio 与 validation_ratio 之和必须小于 1，测试集取剩余部分")


@dataclass(frozen=True)
class BaselineConfig:
    """``configs/baseline.yaml`` 的类型化表示。"""

    seed: int
    input_path: Path
    report_path: Path
    model_dir: Path
    split: SplitConfig
    threshold_grid: tuple[float, ...]
    models: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> BaselineConfig:
        """从已解析的 YAML 映射构造配置，缺字段时立即报错。"""
        required = ("seed", "input_path", "report_path", "model_dir", "split", "threshold_grid", "models")
        missing = [key for key in required if key not in raw]
        if missing:
            raise ConfigError("配置缺少字段：" + ", ".join(missing))

        split_raw = raw["split"]
        models = raw["models"]
        if not isinstance(models, dict) or len(models) < 2:
            raise ConfigError("配置的 models 至少需要包含两个模型（Dummy 与 LogisticRegression）")

        grid = tuple(float(value) for value in raw["threshold_grid"])
        if not grid or any(not 0.0 < value < 1.0 for value in grid):
            raise ConfigError("threshold_grid 必须是非空的 (0, 1) 区间阈值列表")

        return cls(
            seed=int(raw["seed"]),
            input_path=resolve_path(raw["input_path"]),
            report_path=resolve_path(raw["report_path"]),
            model_dir=resolve_path(raw["model_dir"]),
            split=SplitConfig(
                train_ratio=float(split_raw["train_ratio"]),
                validation_ratio=float(split_raw["validation_ratio"]),
            ),
            threshold_grid=grid,
            models=models,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> BaselineConfig:
        """从 YAML 文件加载配置。"""
        config_path = resolve_path(path)
        if not config_path.is_file():
            raise ConfigError(f"配置文件不存在：{config_path}")
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        if not isinstance(raw, dict):
            raise ConfigError(f"配置文件顶层必须是映射：{config_path}")
        return cls.from_mapping(raw)
