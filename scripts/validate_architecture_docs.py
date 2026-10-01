"""Validate the auditable CW-019 architecture document without network access."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "architecture.md"
README = ROOT / "README.md"
MANIFEST = ROOT / "configs" / "retrieval_sources.json"

REQUIRED_TEXT = (
    "## 2. 可追溯系统图",
    "## 3. 数据流一：离线数据到可交付模型",
    "## 4. 数据流二：请求时评分、政策、调查候选",
    "## 5. 数据流三：Docker 演示与托管门禁",
    "## 8. 关键取舍",
    "scripts/run_api.py",
    "scripts/run_deepseek_generation.py",
    "compose.demo.yaml",
    "GitHub Actions",
    "DEEPSEEK_API_KEY",
    "[返回项目总览 README](../README.md)",
    "validation 同时用于 LightGBM 早停和 F1 阈值冻结",
    "offline_generate[CW-016 generate_candidate]",
    "offline_validator[CW-016 validate_candidate]",
    "199",
    "12/12",
    "f02bb50e61c5f70c85448430c8bb653bd6e347f400220771500ee84a70135a1c",
    "7ed07f0b784715c01b444035bb34ea5166d1b376ff07e10428561d898d6824f0",
)


def _mermaid_blocks(text: str) -> list[str]:
    fence = chr(96) * 3
    return re.findall(fence + r"mermaid\r?\n(.*?)\r?\n" + fence, text, flags=re.DOTALL)


def validate() -> dict[str, object]:
    errors: list[str] = []
    if not DOC.is_file():
        return {"ok": False, "errors": ["architecture_doc_missing"], "diagram_count": 0}
    text = DOC.read_text(encoding="utf-8")
    for marker in REQUIRED_TEXT:
        if marker not in text:
            errors.append(f"missing:{marker}")
    blocks = _mermaid_blocks(text)
    if len(blocks) < 3:
        errors.append("expected_at_least_three_mermaid_blocks")
    for index, block in enumerate(blocks, start=1):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines or lines[0] not in {"flowchart LR", "flowchart TD"}:
            errors.append(f"diagram_{index}_not_flowchart")
        if not any("-->" in line or "-.->" in line for line in lines):
            errors.append(f"diagram_{index}_has_no_directional_edge")
    order_checks = (
        (
            "## 3. 数据流一：离线数据到可交付模型",
            ("fit --> train_model", "train_model --> validation_scores", "validation_scores --> threshold"),
        ),
        (
            "## 4. 数据流二：请求时评分、政策、调查候选",
            ("context --> candidate", "candidate -->|离线| offline_generate", "offline_generate --> offline_validator"),
        ),
        (
            "## 4. 数据流二：请求时评分、政策、调查候选",
            (
                "deepseek[CW-017 DeepSeek Responses API]",
                "external_candidate[不可信外部候选]",
                "external_candidate --> validator",
            ),
        ),
    )
    for index, (section, sequence) in enumerate(order_checks, start=1):
        section_start = text.find(section)
        positions = [text.find(marker, section_start) for marker in sequence]
        if section_start < 0 or any(position < 0 for position in positions) or positions != sorted(positions):
            errors.append(f"diagram_order_{index}_invalid")
    if re.search(r"sk-[A-Za-z0-9]{12,}", text):
        errors.append("possible_api_key_in_document")
    if "CW-003 train-only 特征与阈值" in text or "验证集只用于冻结阈值" in text:
        errors.append("stale_validation_boundary_wording")
    readme_text = README.read_text(encoding="utf-8")
    if "docs/architecture.md" not in readme_text:
        errors.append("readme_architecture_link_missing")
    lines = readme_text.splitlines()
    approved = "\n".join(lines[297:302]) + "\n" if len(lines) >= 302 else ""
    approved_sha = hashlib.sha256(approved.encode("utf-8")).hexdigest()
    if approved_sha != "137a65b431f482e3f2ee048b438226fa9acdb83c6bb3a8750740b92744a31361":
        errors.append("approved_readme_fragment_changed")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    readme_entry = next(item for item in manifest["sources"] if item["source_id"] == "readme-data-limits")
    readme_sha = hashlib.sha256(README.read_bytes()).hexdigest()
    if readme_entry["sha256"] != readme_sha:
        errors.append("retrieval_manifest_readme_sha_mismatch")
    return {
        "ok": not errors,
        "errors": errors,
        "diagram_count": len(blocks),
        "approved_readme_fragment_sha256": approved_sha,
        "readme_sha256": readme_sha,
    }


def main() -> int:
    result = validate()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
